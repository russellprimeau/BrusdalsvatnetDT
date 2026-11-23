"""
Small Bokeh helpers that give every water-quality / weather parameter its own subplot.

Parameters with different units (e.g. pressure in mBar next to precipitation in mm/hr) are unreadable on one shared
y-axis, so the Historic pages of Dashboard.py draw one figure per parameter and stack them with linked axes. Kept in
a separate module, free of Dashboard.py's heavy imports, so the layout logic can be unit-tested.

Bokeh is pinned to 2.4.3 because Streamlit's st.bokeh_chart requires it; only APIs available in 2.4.3 are used.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from bokeh.layouts import gridplot
from bokeh.models import DatetimeTickFormatter, Range1d
from bokeh.plotting import figure

# Height of each subplot in pixels; shorter panels when many are stacked so the page stays navigable.
SUBPLOT_HEIGHT = 220
SUBPLOT_HEIGHT_COMPACT = 170
COMPACT_THRESHOLD = 6

# Same tick format the single-axis plots used, so dates read identically to the previous version.
DATETIME_FORMATTER = dict(days="%Y/%m/%d", hours="%Y/%m/%d %H:%M")


def units_from_label(label: str) -> str:
    """
    Return the units in a trailing parenthesis, e.g. 'Temperature (Celsius)' -> 'Celsius'.
    Labels without units (e.g. 'pH') are returned unchanged, so the axis is never left blank.
    """
    match = re.search(r"\(([^()]*)\)\s*$", label)
    return match.group(1) if match else label


def subplot_height(n_subplots: int) -> int:
    """Height per subplot, compact when more than COMPACT_THRESHOLD subplots are stacked."""
    return SUBPLOT_HEIGHT_COMPACT if n_subplots > COMPACT_THRESHOLD else SUBPLOT_HEIGHT


def shared_time_range(begin: datetime, end: datetime) -> Range1d:
    """
    The x-range all time-series subplots share (sharing one Range1d object links their pan/zoom).
    Padding rules are unchanged from the original single-axis plots: wide ranges get 3 days either side,
    short ranges get a little extra after the end date so the last day is fully visible.
    """
    if end - begin > timedelta(days=62):
        return Range1d(begin - timedelta(days=3), end + timedelta(days=3))
    return Range1d(begin, end + timedelta(days=1, hours=3))


def time_series_figure(title: str, x_range: Range1d, height: int, y_label: str | None = None):
    """A figure for one parameter, on the shared time axis, stretching to the container width."""
    p = figure(title=title, x_range=x_range, x_axis_type="datetime", height=height, sizing_mode="stretch_width")
    p.xaxis.formatter = DatetimeTickFormatter(**DATETIME_FORMATTER)
    p.yaxis.axis_label = y_label if y_label is not None else units_from_label(title)
    p.title.text_font_size = "12pt"
    return p


def fit_y_range(p, series: list, x_range: Range1d, pad_fraction: float = 0.05) -> bool:
    """
    Fit a subplot's y-axis to the data inside the selected time window.

    Bokeh's automatic y-range spans *all* data in the source, so a value from another year (e.g. a -99.9 error
    code shown with the "Raw" filter) would squash the visible lines. ``series`` is a list of (times, values) pairs,
    e.g. one per depth contour. Returns False, leaving Bokeh's automatic range, if the window holds no finite data.
    """
    lo, hi = np.inf, -np.inf
    start, end = pd.Timestamp(x_range.start), pd.Timestamp(x_range.end)
    for times, values in series:
        t = pd.to_datetime(pd.Series(np.asarray(times)))
        v = pd.to_numeric(pd.Series(np.asarray(values)), errors="coerce")
        window = v[(t >= start).to_numpy() & (t <= end).to_numpy()].to_numpy(dtype=float)
        window = window[np.isfinite(window)]
        if window.size:
            lo, hi = min(lo, window.min()), max(hi, window.max())
    if not np.isfinite(lo):
        return False
    pad = (hi - lo) * pad_fraction or max(abs(hi) * pad_fraction, 1e-3)  # flat series still get a visible band
    p.y_range = Range1d(lo - pad, hi + pad)
    return True


def stack_figures(figs: list, ncols: int = 1, x_label: str | None = "Time"):
    """
    Arrange subplots in a grid with a single merged toolbar (one set of pan/zoom/save buttons for all panels).
    With a shared x-axis (x_label given), the label is only drawn on the bottom row to save vertical space; tick
    labels stay on every panel. With x_label=None each figure keeps its own x-axis label (e.g. per-parameter units).
    """
    if not figs:
        raise ValueError("stack_figures needs at least one figure")
    ncols = max(1, min(ncols, len(figs)))
    if x_label is not None:
        n_last_row = len(figs) % ncols or ncols
        for i, p in enumerate(figs):
            p.xaxis.axis_label = x_label if i >= len(figs) - n_last_row else None
    return gridplot(figs, ncols=ncols, merge_tools=True, toolbar_location="right", sizing_mode="stretch_width")
