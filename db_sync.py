"""
Read-only synchronisation of the dashboard's CSV data files from the project's PostgreSQL database.

The Streamlit dashboard (Dashboard.py) always plots from the three CSV files in this folder. This module keeps those
files current by querying the database for records newer than the last row of each CSV and appending them, replacing
the old workflow in which the Scrape*.py scripts scraped the loggers directly and pushed the CSVs to GitHub.

Guarantees
----------
* READ-ONLY: the database session is opened with ``default_transaction_read_only=on`` *and* ``readonly=True``, so the
  server itself rejects any write. The only statements issued are SELECTs (and catalogue lookups).
* NO CREDENTIAL LEAKS: connection parameters come from the local ``.env`` file (or ``.streamlit/secrets.toml``) and are
  never printed, logged, or included in error messages. Database driver errors can contain the host name, so they are
  replaced with generic messages; only the exception *class* is logged.
* VALUES PASS THROUGH UNCHANGED: no clipping, zeroing, or rounding. Logger sentinel values (e.g. -99.9) and real zeros
  are written exactly as stored. Only SQL NULL becomes an empty CSV field (which pandas reads back as NaN). Cleaning
  is left to the dashboard's "Remove Suspicious Values" option.
* CSV FORMAT IS PRESERVED: each file keeps its own header, separator, decimal mark, encoding (UTF-8) and line ending.

Column mapping
--------------
The legacy scrapers inserted rows with ``cursor.copy_from`` and no column list, i.e. *by position*. The database
tables therefore hold the CSV's columns in the same order (minus the constant Latitude/Longitude columns, plus the
weather station's battery voltage ``Batt_V`` at position 3). This module relies on that same positional contract,
and verifies it before every write (column count, ``datetime`` first column, numeric types, and an overlap check that
the database row at the CSV's last timestamp matches the CSV's last row).

Command line (never prints configuration values)
------------------------------------------------
    python db_sync.py --describe          # table columns, types, row counts, time span
    python db_sync.py --dry-run           # fetch + validate, write nothing
    python db_sync.py --rebuild [--force] # one-time full overwrite of the CSVs from the database
    python db_sync.py                     # incremental append (what the dashboard runs)
Settings can also come from environment variables (e.g. GitHub Actions secrets, see
.github/workflows/sync-data.yml); variables already set in the environment take precedence over .env.
Exit status is 1 if any dataset could not be synced or rebuilt.
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

log = logging.getLogger(__name__)

# Fixed position of the profiler platform. The database tables do not store it; the CSVs carry it in every row.
PROFILER_LAT = 62.474464
PROFILER_LON = 6.461324

# Keys read from .env / secrets. No defaults: if a key is missing, syncing is skipped and the CSVs are used as-is.
CONNECTION_KEYS = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD", "schema_name")

# Name of the timestamp column in every table (the only column name the legacy scrapers referenced explicitly).
DB_TIME_COLUMN = "datetime"

# Postgres types accepted for measurement columns. Anything else means the positional mapping is wrong.
NUMERIC_TYPES = {"smallint", "integer", "bigint", "real", "double precision", "numeric"}
TIME_TYPES = {"timestamp without time zone", "timestamp with time zone"}

# Serialises CSV writes. Streamlit runs every browser session as a thread in one process, and this module is imported
# once per process, so a module-level lock prevents two sessions from appending the same rows simultaneously.
_WRITE_LOCK = threading.Lock()

# Tolerance for the overlap check (CSV values are written with at most 3-4 significant decimals).
_OVERLAP_TOLERANCE = 1e-3


# ----------------------------------------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SyncSpec:
    """Describes how one database table maps onto one CSV file."""
    key: str                    # short identifier used in results and the CLI
    label: str                  # human-readable name shown in the dashboard sidebar
    csv_path: str               # CSV file name, relative to the project folder
    table_key: str              # .env key holding the database table name
    sep: str                    # CSV field separator
    decimal: str                # CSV decimal mark
    n_db_columns: int           # expected number of columns in the database table
    db_drop_positions: tuple = ()               # 0-based DB column positions not stored in the CSV
    constants: dict = field(default_factory=dict)  # CSV columns absent from the DB, filled with constant values
    int_columns: tuple = ()     # CSV columns written as integers (nullable)
    depth_column: str | None = None  # rows with no depth are unusable for profiles (Dashboard casts depth to int)


SPECS = {
    "hourly": SyncSpec(
        key="hourly", label="Profiler hourly surface data", csv_path="Profiler_modem_SondeHourly.csv",
        table_key="DB_TABLE_HOURLY", sep=",", decimal=".", n_db_columns=13,
        constants={"Latitude": PROFILER_LAT, "Longitude": PROFILER_LON}, int_columns=("RECORD",)),
    "profiles": SyncSpec(
        key="profiles", label="Profiler vertical profiles", csv_path="Profiler_modem_PFL_Step.csv",
        table_key="DB_TABLE_PROFILES", sep=",", decimal=".", n_db_columns=16,
        constants={"lat": PROFILER_LAT, "lon": PROFILER_LON}, int_columns=("RECORD", "PFL_Counter", "CntRS232"),
        depth_column="sensorParms(9)"),
    "weather": SyncSpec(
        key="weather", label="Weather station", csv_path="All_time.csv",
        table_key="DB_TABLE_WEATHER", sep=";", decimal=",", n_db_columns=21,
        db_drop_positions=(2,)),  # position 2 = battery voltage, which the CSV has never stored
}


class SyncConfigError(Exception):
    """Raised when configuration is missing. The message names missing *keys* only, never values."""


class SyncValidationError(Exception):
    """Raised when fetched data fails a consistency check; nothing is written when this is raised."""


def _project_dir() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def load_db_config() -> dict:
    """
    Return the connection settings and table names as a dict.

    Looks in the project's ``.env`` file first (without overriding variables already set in the environment), then
    in Streamlit secrets under ``[postgres]`` (for deployments where no .env file exists).
    Raises SyncConfigError listing only the *names* of missing keys.
    """
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_project_dir(), ".env"), override=False)

    wanted = CONNECTION_KEYS + tuple(spec.table_key for spec in SPECS.values())
    cfg = {k: os.getenv(k) for k in wanted}

    missing = [k for k, v in cfg.items() if not v]
    if missing:
        try:  # Optional fallback to .streamlit/secrets.toml; absent secrets raise, which is fine.
            import streamlit as st
            secrets = st.secrets.get("postgres", {})
            for k in missing:
                if secrets.get(k):
                    cfg[k] = str(secrets[k])
        except Exception:
            pass
        missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise SyncConfigError("Database sync not configured; missing setting(s): " + ", ".join(missing))
    return cfg


def _connect(cfg: dict):
    """Open a read-only connection. Read-only is enforced twice: by the session option and by set_session."""
    import psycopg2
    conn = psycopg2.connect(
        dbname=cfg["DB_NAME"], user=cfg["DB_USER"], password=cfg["DB_PASSWORD"],
        host=cfg["DB_HOST"], port=cfg["DB_PORT"],
        connect_timeout=5,  # keep the dashboard responsive if the server is unreachable
        options="-c default_transaction_read_only=on",
    )
    conn.set_session(readonly=True, autocommit=True)
    return conn


def _sanitize(exc: Exception) -> str:
    """
    Turn an exception into a message safe to show users. Driver errors may contain host/user names, so their text is
    discarded; only the exception class is kept (and logged).
    """
    log.warning("Database sync error: %s", type(exc).__name__)
    if isinstance(exc, (SyncConfigError, SyncValidationError)):
        return str(exc)  # these messages are written by this module and contain no configuration values
    try:
        import psycopg2
        if isinstance(exc, psycopg2.OperationalError):
            return "Could not reach the database (network or login problem)."
        if isinstance(exc, psycopg2.Error):
            return f"Database query failed ({type(exc).__name__})."
    except ImportError:
        return "Database driver (psycopg2) is not installed."
    return f"Sync failed ({type(exc).__name__})."


# ----------------------------------------------------------------------------------------------------------------------
# CSV helpers
# ----------------------------------------------------------------------------------------------------------------------

def read_header(path: str) -> tuple[str, str]:
    """Return (raw header line, line terminator) of a CSV, so rewritten files keep the exact header and line endings."""
    with open(path, "rb") as f:
        first = f.readline()
    terminator = "\r\n" if first.endswith(b"\r\n") else "\n"
    header_line = first.decode("utf-8").rstrip("\r\n")
    return header_line, terminator


def _header_columns(spec: SyncSpec, header_line: str) -> list[str]:
    return header_line.split(spec.sep)


def read_csv_data(spec: SyncSpec, path: str) -> pd.DataFrame:
    """
    Read a dashboard CSV with the same separator/decimal conventions as Dashboard.py. 'NAN' (the logger's own
    missing-value marker, present in older rows) is treated as missing alongside empty fields.
    """
    df = pd.read_csv(path, sep=spec.sep, decimal=spec.decimal, na_values=["NAN"], encoding="utf-8")
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], format="ISO8601")
    return df


def _last_csv_row(spec: SyncSpec, path: str) -> pd.Series | None:
    """The row with the latest timestamp (max, not last line, in case rows are out of order); None if empty."""
    df = read_csv_data(spec, path)
    if df.empty:
        return None
    return df.loc[df[df.columns[0]].idxmax()]


def _format_for_csv(spec: SyncSpec, df: pd.DataFrame, terminator: str, header: bool) -> str:
    """Serialise rows in the CSV's own dialect. NULL -> empty field; timestamps -> ISO 8601 without timezone."""
    out = df.copy()
    time_col = out.columns[0]
    out[time_col] = out[time_col].dt.strftime("%Y-%m-%dT%H:%M:%S")
    buf = io.StringIO()
    out.to_csv(buf, sep=spec.sep, decimal=spec.decimal, index=False, header=header, na_rep="",
               lineterminator=terminator)
    return buf.getvalue()


# ----------------------------------------------------------------------------------------------------------------------
# Database access (SELECT only)
# ----------------------------------------------------------------------------------------------------------------------

def describe_table(conn, schema: str, table: str) -> list[tuple[str, str]]:
    """Return [(column_name, data_type), ...] in ordinal order, from the information_schema catalogue."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = %s AND table_name = %s ORDER BY ordinal_position",
            (schema, table))
        return cur.fetchall()


def _check_schema(spec: SyncSpec, columns: list[tuple[str, str]]) -> None:
    """
    Verify the positional contract before trusting it: right number of columns, timestamp first, numbers elsewhere.
    A failure here means the table layout changed, so nothing is written.
    """
    if len(columns) != spec.n_db_columns:
        raise SyncValidationError(
            f"{spec.label}: database table has {len(columns)} columns, expected {spec.n_db_columns}; not syncing.")
    name, dtype = columns[0]
    if name != DB_TIME_COLUMN or dtype not in TIME_TYPES:
        raise SyncValidationError(f"{spec.label}: first database column is not a '{DB_TIME_COLUMN}' timestamp.")
    bad = [n for n, t in columns[1:] if t not in NUMERIC_TYPES]
    if bad:
        raise SyncValidationError(f"{spec.label}: non-numeric database column(s) {bad}; not syncing.")


def fetch_rows(conn, schema: str, table: str, columns: list[str], after: datetime | None) -> pd.DataFrame:
    """
    SELECT the given columns, optionally only rows with datetime >= ``after`` (inclusive, so the overlap row can be
    compared). Identifiers are quoted by psycopg2.sql and the timestamp is a bound parameter: no string-built SQL.
    """
    from psycopg2 import sql
    query = sql.SQL("SELECT {cols} FROM {schema}.{table}").format(
        cols=sql.SQL(", ").join(sql.Identifier(c) for c in columns),
        schema=sql.Identifier(schema), table=sql.Identifier(table))
    params = None
    if after is not None:
        query += sql.SQL(" WHERE {t} >= %s").format(t=sql.Identifier(DB_TIME_COLUMN))
        params = (after.to_pydatetime() if isinstance(after, pd.Timestamp) else after,)
    query += sql.SQL(" ORDER BY {t}").format(t=sql.Identifier(DB_TIME_COLUMN))
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
    return pd.DataFrame(rows, columns=columns)


# ----------------------------------------------------------------------------------------------------------------------
# Transformation + validation (pure functions: unit-tested without a database)
# ----------------------------------------------------------------------------------------------------------------------

def transform(spec: SyncSpec, db_df: pd.DataFrame, csv_columns: list[str]) -> pd.DataFrame:
    """
    Map database rows onto the CSV's columns by position.

    * drops DB-only columns (``db_drop_positions``) and appends the constant columns (lat/lon)
    * converts timestamps to naive datetimes (wall-clock values as originally inserted)
    * converts measurements to numbers WITHOUT altering them; SQL NULL stays missing
    * drops rows with no timestamp (cannot be placed in a time series) and, for profiles, no depth
    """
    keep = [c for i, c in enumerate(db_df.columns) if i not in spec.db_drop_positions]
    data_columns = [c for c in csv_columns if c not in spec.constants]
    if len(keep) != len(data_columns):
        raise SyncValidationError(
            f"{spec.label}: {len(keep)} database columns cannot map onto {len(data_columns)} CSV columns.")
    out = db_df[keep].copy()
    out.columns = data_columns

    time_col = data_columns[0]
    ts = pd.to_datetime(out[time_col])
    if getattr(ts.dt, "tz", None) is not None:
        # timestamptz is returned in the session time zone; dropping the zone keeps the wall-clock value inserted.
        ts = ts.dt.tz_localize(None)
    out[time_col] = ts

    for col in data_columns[1:]:
        # pd.to_numeric handles Decimal (Postgres numeric) and None; it never changes a valid number's value.
        values = pd.to_numeric(out[col], errors="raise")
        if col in spec.int_columns:
            values = values.astype("Int64")  # nullable: a NULL stays empty instead of turning ints into floats
        else:
            values = values.astype("float64")
        out[col] = values

    for col, value in spec.constants.items():
        out[col] = value
    out = out[csv_columns]

    dropped = int(out[time_col].isna().sum())
    out = out[out[time_col].notna()]
    if spec.depth_column:
        no_depth = out[spec.depth_column].isna()
        dropped += int(no_depth.sum())
        out = out[~no_depth]
    if dropped:
        log.warning("%s: dropped %d row(s) with no timestamp/depth", spec.label, dropped)
    return out.reset_index(drop=True)


def _rows_match(spec: SyncSpec, db_row: pd.Series, csv_row: pd.Series) -> bool:
    """True if a database row equals the CSV row (NaN == NaN, floats within tolerance)."""
    for col in db_row.index[1:]:
        if col in spec.constants:
            continue
        a, b = db_row[col], csv_row[col]
        a_missing, b_missing = pd.isna(a), pd.isna(b)
        if a_missing or b_missing:
            if a_missing != b_missing:
                return False
            continue
        if not math.isclose(float(a), float(b), rel_tol=0, abs_tol=_OVERLAP_TOLERANCE):
            return False
    return True


def validate_new_rows(spec: SyncSpec, new: pd.DataFrame, csv_columns: list[str], after: datetime | None) -> None:
    """Final checks before any write; raises SyncValidationError (and nothing is written) on failure."""
    if list(new.columns) != csv_columns:
        raise SyncValidationError(f"{spec.label}: transformed columns do not match the CSV header.")
    time_col = csv_columns[0]
    if new[time_col].duplicated().any():
        raise SyncValidationError(f"{spec.label}: database returned duplicate timestamps; not syncing.")
    if not new[time_col].is_monotonic_increasing:
        raise SyncValidationError(f"{spec.label}: database rows are not in time order; not syncing.")
    if after is not None and (new[time_col] <= after).any():
        raise SyncValidationError(f"{spec.label}: rows not newer than the CSV's last record; not syncing.")


# ----------------------------------------------------------------------------------------------------------------------
# Sync operations
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class SyncResult:
    key: str
    label: str
    status: str            # "updated", "up to date", "skipped", "error", "dry run", "rebuilt"
    rows_added: int = 0
    last_timestamp: str = ""
    message: str = ""


def _prepare(spec: SyncSpec, conn, cfg: dict, base_dir: str):
    """Shared set-up: locate the CSV, read its header, check the table layout. Returns everything a sync needs."""
    path = os.path.join(base_dir, spec.csv_path)
    if not os.path.exists(path):
        raise SyncValidationError(f"{spec.label}: CSV file {spec.csv_path} not found.")
    header_line, terminator = read_header(path)
    csv_columns = _header_columns(spec, header_line)
    schema, table = cfg["schema_name"], cfg[spec.table_key]
    db_columns = describe_table(conn, schema, table)
    if not db_columns:
        raise SyncValidationError(f"{spec.label}: database table not found.")
    _check_schema(spec, db_columns)
    return path, header_line, terminator, csv_columns, schema, table, [c for c, _ in db_columns]


def sync_csv(spec: SyncSpec, conn, cfg: dict, base_dir: str | None = None, dry_run: bool = False) -> SyncResult:
    """
    Append database rows newer than the CSV's latest record. Idempotent: a second call adds nothing.
    Steps: check layout -> fetch rows from the CSV's last timestamp (inclusive) -> overlap check -> transform ->
    validate -> single append under the write lock.
    """
    base_dir = base_dir or _project_dir()
    path, _, terminator, csv_columns, schema, table, db_cols = _prepare(spec, conn, cfg, base_dir)

    with _WRITE_LOCK:  # the read-last-row / append pair must be atomic with respect to other sessions
        last_row = _last_csv_row(spec, path)
        after = last_row.iloc[0] if last_row is not None else None
        fetched = transform(spec, fetch_rows(conn, schema, table, db_cols, after), csv_columns)

        time_col = csv_columns[0]
        if after is not None:
            overlap = fetched[fetched[time_col] == after]
            # Overlap guard: if the DB holds the CSV's last record, it must match. A mismatch means the positional
            # mapping or time zone is wrong, and appending would silently corrupt the series.
            if not overlap.empty and not _rows_match(spec, overlap.iloc[0], last_row):
                raise SyncValidationError(
                    f"{spec.label}: database row at {after:%Y-%m-%d %H:%M} differs from the CSV; not syncing.")
            fetched = fetched[fetched[time_col] > after].reset_index(drop=True)

        validate_new_rows(spec, fetched, csv_columns, after)
        last = fetched[time_col].iloc[-1] if not fetched.empty else after
        last_str = f"{last:%Y-%m-%d %H:%M}" if last is not None else ""
        if fetched.empty:
            return SyncResult(spec.key, spec.label, "up to date", 0, last_str)
        if dry_run:
            return SyncResult(spec.key, spec.label, "dry run", len(fetched), last_str,
                              f"would append {len(fetched)} row(s) from {fetched[time_col].iloc[0]:%Y-%m-%d %H:%M}")

        text = _format_for_csv(spec, fetched, terminator, header=False)
        with open(path, "rb+") as f:
            # Make sure the existing last line is terminated, so the first new row does not merge into it.
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    text = terminator + text
            f.seek(0, os.SEEK_END)
            f.write(text.encode("utf-8"))  # one write call: a validation failure above means nothing is written
    log.info("%s: appended %d row(s)", spec.label, len(fetched))
    return SyncResult(spec.key, spec.label, "updated", len(fetched), last_str)


def _summary(spec: SyncSpec, df: pd.DataFrame) -> str:
    t = df[df.columns[0]]
    nulls = {c: int(n) for c, n in df.isna().sum().items() if n}
    return (f"{len(df)} rows, {t.min():%Y-%m-%d %H:%M} .. {t.max():%Y-%m-%d %H:%M}, "
            f"missing values per column: {nulls or 'none'}")


def rebuild_csv(spec: SyncSpec, conn, cfg: dict, base_dir: str | None = None, force: bool = False) -> SyncResult:
    """
    One-time full overwrite of a CSV with the database contents (keeping the CSV's exact header line).

    Data-loss guard: refuses (unless force=True) if the database starts later than the CSV, or holds fewer rows than
    the CSV over the period both cover, because history that exists only in the CSV would be thrown away.
    The new file is written to <csv>.tmp, re-read and checked, then swapped in with os.replace (atomic on the same
    volume), so a failure at any point leaves the original file untouched.
    """
    base_dir = base_dir or _project_dir()
    path, header_line, terminator, csv_columns, schema, table, db_cols = _prepare(spec, conn, cfg, base_dir)
    with _WRITE_LOCK:
        old = read_csv_data(spec, path)
        new = transform(spec, fetch_rows(conn, schema, table, db_cols, None), csv_columns)
        validate_new_rows(spec, new, csv_columns, None)
        if new.empty:
            raise SyncValidationError(f"{spec.label}: database table is empty; refusing to overwrite the CSV.")

        time_col = csv_columns[0]
        old_t, new_t = old[old.columns[0]], new[time_col]
        report = f"before: {_summary(spec, old)}\n    after:  {_summary(spec, new)}"
        if not old.empty and not force:
            in_window = new_t.between(old_t.min(), old_t.max()).sum()
            if new_t.min() > old_t.min():
                raise SyncValidationError(
                    f"{spec.label}: database starts {new_t.min():%Y-%m-%d}, after the CSV ({old_t.min():%Y-%m-%d}); "
                    f"rebuilding would drop history. Use --force to override.\n    {report}")
            if in_window < len(old):
                raise SyncValidationError(
                    f"{spec.label}: database has {in_window} rows in the CSV's period vs {len(old)} in the CSV; "
                    f"rebuilding would drop rows. Use --force to override.\n    {report}")

        tmp = path + ".tmp"
        try:
            body = _format_for_csv(spec, new, terminator, header=False)
            with open(tmp, "wb") as f:
                f.write((header_line + terminator + body).encode("utf-8"))
            check = read_csv_data(spec, tmp)  # the rewritten file must parse back to the same shape
            if len(check) != len(new) or list(check.columns) != csv_columns:
                raise SyncValidationError(f"{spec.label}: rewritten file failed the read-back check.")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return SyncResult(spec.key, spec.label, "rebuilt", len(new), f"{new_t.max():%Y-%m-%d %H:%M}", report)


def sync_all(base_dir: str | None = None, dry_run: bool = False) -> list[SyncResult]:
    """
    Incrementally sync every CSV over a single read-only connection. Never raises: every failure is reported as a
    SyncResult with a sanitized message, so the dashboard can always fall back to the existing CSVs.
    """
    try:
        cfg = load_db_config()
        conn = _connect(cfg)
    except Exception as exc:  # missing config, missing driver, unreachable server
        msg = _sanitize(exc)
        return [SyncResult(s.key, s.label, "skipped", message=msg) for s in SPECS.values()]
    results = []
    try:
        for spec in SPECS.values():
            try:
                results.append(sync_csv(spec, conn, cfg, base_dir, dry_run))
            except Exception as exc:  # one bad table must not block the others
                results.append(SyncResult(spec.key, spec.label, "error", message=_sanitize(exc)))
    finally:
        conn.close()
    return results


# ----------------------------------------------------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------------------------------------------------

def _cli() -> int:
    parser = argparse.ArgumentParser(description="Read-only sync of the dashboard CSVs from PostgreSQL.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--describe", action="store_true", help="show table layouts, row counts and time spans")
    group.add_argument("--dry-run", action="store_true", help="fetch and validate new rows, write nothing")
    group.add_argument("--rebuild", action="store_true", help="overwrite each CSV with the full database table")
    parser.add_argument("--force", action="store_true", help="with --rebuild: override the data-loss guard")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if not (args.describe or args.rebuild):
        results = sync_all(dry_run=args.dry_run)
        for r in results:
            print(f"[{r.status}] {r.label}: {r.rows_added} row(s); last record {r.last_timestamp or '-'}"
                  f"{'; ' + r.message if r.message else ''}")
        # Non-zero exit if any dataset could not be synced, so schedulers (e.g. GitHub Actions) report the failure.
        # (The dashboard calls sync_all() directly and is unaffected.)
        return 1 if any(r.status in ("error", "skipped") for r in results) else 0

    try:
        cfg = load_db_config()
        conn = _connect(cfg)
    except Exception as exc:
        print("ERROR:", _sanitize(exc))
        return 1
    status = 0
    try:
        for spec in SPECS.values():
            try:
                if args.describe:
                    cols = describe_table(conn, cfg["schema_name"], cfg[spec.table_key])
                    print(f"\n{spec.label} ({spec.table_key}): {len(cols)} columns (expected {spec.n_db_columns})")
                    for name, dtype in cols:
                        print(f"    {name:<28} {dtype}")
                    if cols:
                        df = transform(spec, fetch_rows(conn, cfg["schema_name"], cfg[spec.table_key],
                                                        [c for c, _ in cols], None),
                                       _header_columns(spec, read_header(
                                           os.path.join(_project_dir(), spec.csv_path))[0]))
                        print("    database:", _summary(spec, df) if not df.empty else "empty")
                        print("    csv:     ", _summary(spec, read_csv_data(
                            spec, os.path.join(_project_dir(), spec.csv_path))))
                else:
                    r = rebuild_csv(spec, conn, cfg, force=args.force)
                    print(f"[rebuilt] {spec.label}: {r.rows_added} rows\n    {r.message}")
            except Exception as exc:
                status = 1
                print(f"[error] {spec.label}: {_sanitize(exc)}")
    finally:
        conn.close()
    return status


if __name__ == "__main__":
    raise SystemExit(_cli())
