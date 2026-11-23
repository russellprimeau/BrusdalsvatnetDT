# BrusdalsvatnetDT
Collects and visualizes data from several systems which monitor water quality in the Brusdalsvatnet drinking water reservoir in Ålesund, Norway.

This repository includes Python scripts and data files for completing several tasks related to creating and maintaining a digital twin of the Brusdalsvatnet drinking water reservoir, for the purpose of monitoring water quality.

The core function of this architecture is to aggregate data from a variety of sensing platforms and use that data to update a hydrodynamic water quality model, the 'digital twin.' The framework includes applications for visualizing the collected data as well as the modeled water quality. The modelling is performed offline in a Windows environment using the open source version of the Delft3D FM hydrodynamic simulation software developed by Deltares. Example model output files are included in the repository.

The data collected by the sensing platforms is written to .csv files, which the dashboard plots. Each time the dashboard starts (and at most every 15 minutes while it runs, or on demand with the sidebar's "Check for new data" button) it queries the project's PostgreSQL database for records newer than the last row of each .csv file and appends them. The database is accessed **read-only**. If it cannot be reached, or no credentials are configured, the dashboard shows a notice in the sidebar and plots the .csv files as they are.

The data aquisition systems which monitor the Brusdalsvatnet reservoir include a stationary profiling platform, with winched and stationary instruments; a mobile USV with a winch instrument with similar instruments to the profiler; a second mobile USV with sample collection capabilities; and a weather station managed by a commercial partner. At present, the profiler and weather station are the only data acquisition systems capable of providing near real-time data upload to the web, and the only systems for which a substantial library of historic data exists. The profiler station is inactive during the colder months to avoid ice damage.  Various hardware and software limitations prevent real-time datasharing from the USVs, though work is ongoing to establish these capabilities.

The USV systems have the capability to collect data and samples from anywhere on the surface of Brusdalsvatnet, and at any location down to the maximum depth allowed by the winch. This allows for precise targeting of data collection, to refine knowledge of lake conditions at locations and times where it is most consequential. One goal of the digital twin project is to add automatic functions based on "informative path planning" algorithms which choose the time and location for data collection to maximmize the model accuracy. The path planning capabilities in the present application are limited. Currently the app app lacks direct connectivity with the USVs, and is limited to offline path planning. A sampling mission can be planned in the app, but it must be exported as a .csv file and uploaded to one of the USVs manually via a USB or similar connection.

<img width="554" alt="Simplified DT Dataflow" src="https://github.com/user-attachments/assets/e47f117e-2cd8-4e0c-a395-283183dd538e" />

## Files in the Repository

### Code

**Dashboard.py** drives the online Streamlit app which contains all UI functions, such as data viewing and path planning. To run locally, it must be called from the terminal as "(directory)> streamlit run Dashboard.py". On the "Historic" pages each selected parameter is drawn in its own panel; the panels share the time axis (or the depth axis, for vertical profiles), so panning or zooming one moves them all.

**db_sync.py** keeps the data .csv files current from the PostgreSQL database. It is called automatically by Dashboard.py, and can also be run from the terminal:
- `python db_sync.py --describe` shows the database tables' layout, row counts and time spans.
- `python db_sync.py --dry-run` checks for new records without writing anything.
- `python db_sync.py` appends new records (what the dashboard does).
- `python db_sync.py --rebuild` overwrites each .csv file with the full database table. It refuses if the database holds less history than the .csv file, unless `--force` is added.

The sync never alters values: logger error codes (e.g. -99.9) and zeros are kept as stored, and missing values (SQL NULL) become empty fields. Note that the legacy scrapers stored missing readings as 0 in the database, so older records may contain zeros that were originally missing. The dashboard's "Remove Suspicious Values" option filters implausible values at display time. Appended rows change the .csv files in your working copy; they are not committed or pushed automatically.

**Configuration (.env).** Database connection settings are read from a local `.env` file, which is excluded from git and must never be committed. Copy `.env.example` to `.env` and fill in the values. For a Streamlit Community Cloud deployment, the same keys can instead be placed under a `[postgres]` section in the app's secrets (template: `.streamlit/secrets.toml.example`; the real `.streamlit/secrets.toml` is excluded from git).

**Automatic updates (GitHub Actions).** The workflow `.github/workflows/sync-data.yml` runs `db_sync.py` once a week (Mondays, 03:17 UTC) and commits any new records to the .csv files in this repository; the online Streamlit app redeploys from each new commit. It needs these repository secrets (Settings > Secrets and variables > Actions): `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`, `DB_SCHEMA` (the value of `schema_name`), `DB_TABLE_HOURLY`, `DB_TABLE_PROFILES`, `DB_TABLE_WEATHER`. It can also be started by hand from the Actions tab, in one of four modes: `sync`, `dry-run` (report only), `describe` (table layouts and time spans) or `rebuild` (overwrite the .csv files from the database, see above). The output of each run is shown on its summary page. Workflow logs of a public repository are public; they contain no connection settings.

**plot_utils.py** contains the helpers that arrange one subplot per parameter.

**tests/** contains automated tests for the sync, the plotting helpers and the scrapers' data handling (run with `python -m pytest tests`; pytest is not needed to run the app).

**OfflineUtility.py** drives an offline version of the Streamlit app with extended features for pre- and post-processing of Delft3D models and sensor data. This is especially useful for avoiding file size constraints inherent to online Streamlit apps.

**Legacy scrapers.** The Scrape*.py scripts and their batch files are kept as examples of how the data was originally collected; the dashboard no longer depends on them. Their data-logger addresses, logins and database table names are read from `.env` (see `.env.example`). Missing readings are stored as NULL (not 0), and logger error codes are kept as reported.

**ScrapeHourly.py** scrapes all available data from an online table of readings from the profiler platform's stationary near-surface instruments, which log new values hourly, and writes it to a .csv file. All records newer than the latest record already in the .csv are written.

**ScrapeHourlyBatch.bat** A Windows batch file which the Windows Task Scheduler utility can be configured to call in order to regularly run ScrapeHourly.py.

**ScrapeStep.py** scrapes all available data from an online table of recent readings from the profiler platform's depth profiling instruments, which log new values twice a day, and writes it to a .csv file. All records newer than the latest record already in the .csv are written.

**ScrapeStepBatch.bat** A Windows batch file which the Windows Task Scheduler utility can be configured to call in order to run ScrapeStep.py.

**ScrapeWeather.py** scrapes all available data from an online table of readings from the weather station, and writes it to a .csv file. Different instruments on the weather station have different sample rates. To save memory, only hourly measurements newer than the latest record already in the .csv are written. Note that the weather station's online log is password-protected; the address and login are read from `.env` and are not part of this repository. The script will not execute correctly unless valid settings are provided.

**ScrapeWeatherBatch.bat** A Windows batch file which the Windows Task Scheduler utility can be configured to call in order to run ScrapeWeather.py.

**requirements.txt** a list of dependencies (Python packages, with compatibility/versioning specifications as needed) which is used by Streamlit for launching the Dashboard app.

### Data files

**Profiler_modem_PFL_Step.csv** raw data from the vertical profiler on board the profiling platform, collected from June 2020 to September 2023. Twice a day, the profiler lowers an instrument which records various water quality parameters at intervals as well as some metadata. This data is visualized on the "Historic > Vertical Profiles" feature of the dashboard, both as time series of the values at each depth, and from each individual profiling (vs. depth).

**Profiler_modem_SondeHourly.csv** raw data logged hourly by a stationary instrument near the surface on board the profiling platform from June 2020 to September 2023. This data is visualized on the "Historic > Hourly Surface Data" feature of the dashboard.

**All_time.csv** raw data from the weather station.

**mission.csv** an example of an offline mission plan for the Otter USV (tab-separated: latitude, longitude, type, velocity, waypoint radius, timeout; no header row). The path planner in the app keeps each visitor's mission in their own browser session and offers it for download in this same format; it no longer writes to this file.

**Sample_Priority.csv** and **Sampling_Priority.csv** are related to the USV path planning function of the app.

**.mis** files are log files generated by the Maritime Robotics Otter USV, used to plot missions.

**.nc** and **.tim** are Delft3D input and output files, for running and visualizing the hydrodynamic model.
