# BrusdalsvatnetDT
Collects and visualizes data from several systems which monitor Brusdalsvatnet water quality.

This repository includes Python scripts and data files for completing several tasks related to creating and maintaining a digital twin of the Brusdalsvatnet drinking water reservoir, for the purpose of monitoring water quality. However, not all components of the digital twin have been created or connected. Work is ongoing on various hardware and software components. This repository contains only the functional parts of the overall framework.

The core function of this architecture is to aggregate data from a variety of sensing platforms and use that data to update a hydrologic model of the water quality, the 'digital twin.' The framework includes applications for visualizing the collected data as well as the modeled water quality. 

The data collected by the sensing platforms is stored in a database. At present, a non-public mySQL database on a local harddrive is used as a stand-in for a long-term solution using more reliable disk space. The long-term database solution may use mySQL, InfluxDB, PostgreSQL or another approach, depending on input from various collaborators. 

Scripts with features for reading and writing data to the mySQL database are included only as examples of how these functions will be performed in future versions. Because the stand-in database is not reliably online, the data analysis and visualization functions in this repository use references to static data files saved in the repository.

The data aquisition systems which monitor the Brusdalsvatnet reservoir include a stationary profiling platform, with winched and static instruments; a mobile USV with a winch instrument with similar instruments to the profiling platform; a second mobile USV with sample collection capabilities; and a weather station managed by a commercial partner. At present, the profiling platform is both the only data acquisition system capable of providing near real-time data upload to the web, and the only system for which a substantial library of historic data exists. Data is sparse because the instrument is removed during the colder months to avoid ice damage.  Various hardware and software limitations prevent real-time datasharing from the USV or weather station, though work is ongoing to establish these capabilities.

The USV systems have the capability to collect data and samples from anywhere on Brusdalsvatnet, offering a significant improvemnt in monitoring capabilities. In order to maximize the value of the data which these platforms collect, the digital twin application will include capabilities for offline and online path planning, including automatic functions based on "informative path planning" algorithms. This adaptive sampling should to improve the fidelity of the hydrologic model of the lake, which in turn will allow improved management of Ålesund drinking water quality. The path planning capabilities in the present application are limited, since the hardware is not yet in place to allow the digital twin application to transmit navigation commands to either USV. For the time being, the only working capability which exists is the ability to export a set of sampling coordinates, which can be uploaded to one of the USVs as a form of offline path path planning.


Files in the repository include:

"Dashboard.py": launches a Streamlit app which contains all UI functions, such as data viewing and path planning. Must be called from the terminal as "(directory)> streamlit run Dashboard.py".

"requirements.txt": a list of dependencies (Python packages, with compatibility/versioning specifications as needed) which is used by Streamlit for launching the Dashboard app.

"Scraper.py": called by the Windows Task Scheduler to collect the last hourly reading from the profiler platform's IP address and add it as a new record in the appropriate table of the database. Note the the password for the existing database has been obscured for publication.

"BrusdalsvatnetViewer.py": as a predecessor of the Streamlit app, this script implements several functions for locally pre-processing, analyzing, and viewing the data from the two numerical datasets provided by the profiler platform. Data is viewed using an offline UI implemented with matplotlib and tkinter.

"SondePlotter.py": contains a function for plotting the time series data in a UI. Called by BrusdalsvatnetViewer.py.

Data files:

"XXm_grid_ps.geojson": each file contains GIS data representing the output of a numerical hydrologic modeling software such as Delft3D or GEMSS. Each file stores divides the lake into a 2-dimensional 100m x 100m grid at a particular depth based on public bathymetry data, which is essentially a coarse mesh of finite elements. Each grid square contains data representing the modeled water quality parameter values within that grid. At present each element contains a single node representing the average value within a 100m x 100m x 10m rectangular prism, but this could be extended depending on the chose hydrologic modeling program to include additional nodes (vertices, etc.). The geojson files are used to display this data in the dashboard interface in the "Current Hydrologic Model" window.

"Profiler_modem_PFL_Step.csv": contains raw data from the vertical profiler on board the profiling platform, collected from June 2020 to September 2023. Twice a day, the profiler lowers an instrument which records various water quality parameters at intervals as well as some metadata. This data is visualized on the "Historic > Vertical Profiles" feature of the dashboard, both as time series of the values at each depth, and from each individual profiling (vs. depth).

"Profiler_modem_SondeHourly.csv": contains raw data logged hourly by a stationary instrument near the surface on board the profiling platform from June 2020 to September 2023. This data is visualized on the "Historic > Hourly Surface Data" feature of the dashboard.

"mission.csv": a placeholder for a file containing an offline plan for the Otter USV. An external data structure (this file) is temporarily being used as a workaround because of the difficulty of extracting data from the static Folium interface inside Streamlit, until a better solution is developed using caching. An interface exists in the app for downloading useable mission files from the data temporarily written to this file.
