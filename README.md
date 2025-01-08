# BrusdalsvatnetDT
Collects and visualizes data from several systems which monitor Brusdalsvatnet water quality.

This repository includes Python scripts and data files for completing several tasks related to creating and maintaining a digital twin of the Brusdalsvatnet drinking water reservoir, for the purpose of monitoring water quality. However, not all components of the digital twin have been created or connected. Work is ongoing on various hardware and software components.

The core function of this architecture is to aggregate data from a variety of sensing platforms and use that data to update a hydrodynamic water quality model, the 'digital twin.' The framework includes applications for visualizing the collected data as well as the modeled water quality. The modelling is performed offline in a Windows environment using the open source version of the Delft3D FM hydrodynamic simulation software developed by Deltares. Example model output files are included in the repository.

The data collected by the sensing platforms is written to .csv files within the GitHub repository.

The data aquisition systems which monitor the Brusdalsvatnet reservoir include a stationary profiling platform, with winched and stationary instruments; a mobile USV with a winch instrument with similar instruments to the profiler; a second mobile USV with sample collection capabilities; and a weather station managed by a commercial partner. At present, the profiler and weather station are the only data acquisition systems capable of providing near real-time data upload to the web, and the only systems for which a substantial library of historic data exists. The profiler station is inactive during the colder months to avoid ice damage.  Various hardware and software limitations prevent real-time datasharing from the USVs, though work is ongoing to establish these capabilities.

The USV systems have the capability to collect data and samples from anywhere on the surface of Brusdalsvatnet, and at any location down to the maximum depth allowed by the winch. This allows for precise targeting of data collection, to refine knowledge of lake conditions at locations and times where it is most consequential. One goal of the digital twin project is to add automatic functions based on "informative path planning" algorithms which choose the time and location for data collection to maximmize the model accuracy. The path planning capabilities in the present application are limited. Currently the app app lacks direct connectivity with the USVs, and is limited to offline path planning. A sampling mission can be planned in the app, but it must be exported as a .csv file and uploaded to one of the USVs manually via a USB or similar connection.


<img width="554" alt="Simplified DT Dataflow" src="https://github.com/user-attachments/assets/e47f117e-2cd8-4e0c-a395-283183dd538e" />


Files in the repository include:

"Dashboard.py": drives the online Streamlit app which contains all UI functions, such as data viewing and path planning. To run locally, it must be called from the terminal as "(directory)> streamlit run Dashboard.py".

"OfflineUtility.py": drives an offline version of the Streamlit app with extended features for pre- and post-processing of Delft3D models and sensor data. This is especially useful for avoiding file size constraints inherent to online Streamlit apps.

"ScrapeHourly.py": scrapes all available data from an online table of readings from the profiler platform's stationary near-surface instruments, which log new values hourly, and writes it to a .csv file. All records newer than the latest record already in the .csv are written.

"ScrapeHourlyBatch.bat": A Windows batch file which the Windows Task Scheduler utility can be configured to call in order to regularly run ScrapeHourly.py.

"ScrapeStep.py": scrapes all available data from an online table of recent readings from the profiler platform's depth profiling instruments, which log new values twice a day, and writes it to a .csv file. All records newer than the latest record already in the .csv are written.

"ScrapeStepBatch.bat": A Windows batch file which the Windows Task Scheduler utility can be configured to call in order to run ScrapeStep.py.

"ScrapeWeather.py": scrapes all available data from an online table of readings from the weather station, and writes it to a .csv file. Different instruments on the weather station have different sample rates. To save memory, only hourly measurements newer than the latest record already in the .csv are written. Note that the weather station's online log is password-protected and the authentication information has been removed from this public file. The script will not execute correctly unless valid authentication is added.

"ScrapeWeatherBatch.bat": A Windows batch file which the Windows Task Scheduler utility can be configured to call in order to run ScrapeWeather.py.

"requirements.txt": a list of dependencies (Python packages, with compatibility/versioning specifications as needed) which is used by Streamlit for launching the Dashboard app.

Data files:

"Profiler_modem_PFL_Step.csv": raw data from the vertical profiler on board the profiling platform, collected from June 2020 to September 2023. Twice a day, the profiler lowers an instrument which records various water quality parameters at intervals as well as some metadata. This data is visualized on the "Historic > Vertical Profiles" feature of the dashboard, both as time series of the values at each depth, and from each individual profiling (vs. depth).

"Profiler_modem_SondeHourly.csv": raw data logged hourly by a stationary instrument near the surface on board the profiling platform from June 2020 to September 2023. This data is visualized on the "Historic > Hourly Surface Data" feature of the dashboard.

"All_time.csv": raw data from the weather station.

"mission.csv": a placeholder for a file containing an offline plan for the Otter USV. An external data structure (this file) is temporarily being used as a workaround because of the difficulty of extracting data from the static Folium interface inside Streamlit, until a better solution is developed using caching. An interface exists in the app for downloading useable mission files from the data temporarily written to this file.

"Sample_Priority.csv" and "Sampling_Priority.csv" are related to the USV path planning function of the app.

".mis" files are log files generated by the Maritime Robotics Otter USV, used to plot missions.

".nc" and ".tim" are Delft3D input and output files, for running and visualizing the hydrodynamic model.
