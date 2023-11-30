# BrusdalsvatnetDT
Collects and visualizes data from several systems for monitoring Brusdalsvatnet water quality.

This repository includes Python scripts and data files for completing several tasks related to creating and maintaining a digital twin of the Brusdalsvatnet drinking water reservoir, for the purpose of monitoring water quality. However, not all components of the digital twin have been created or connected. Work is ongoing on various hardware and software components. This repository contains only the functional parts of the overall framework.

The core function of the digital twin is to aggregate data from a variety of sensing platforms and use that data to update a hydrologic model of the water quality. The framework includes applications for visualizing the collected data as well as the modeled water quality. 

The data collected by the sensing platforms is stored in a database. At present, a non-public mySQL database on a local harddrive is used as a stand-in for a long-term solution using more reliable disk space, which may use an alternative free, open-source database architecture such as InfluxDB or PostgreSQL. 

Scripts for reading and writing data to the mySQL database are included only as examples of how these functions will be performed. Because the stand-in database is not reliably online, the data analysis and visualization functions in this repository use references to static data files referenced

The data aquisition systems which monitor the Brusdalsvatnet reservoir include a stationary profiling platform, with winched and static instruments; a mobile USV with a winch instrument with similar instruments to the profiling platform; a second mobile USV with sample collection capabilities; and a weather station managed by a commercial partner. At present, the profiling platform is both the only data acquisition system capable of providing near real-time data upload to the web, and the only system for which a substantial library of historic data exists. Various hardware and software limitations prevent real-time datasharing from the USV or weather station, though work is ongoing to establish these capabilities.

The USV systems have the capability to collect data and samples from anywhere on Brusdalsvatnet, offering a significant improvemnt in monitoring capabilities. In order to maximize the value of the data which these platforms collect, the digital twin application will include capabilities for offline and online path planning, including automatic functions based on "informative path planning" algorithms. This adaptive sampling should to improve the fidelity of the hydrologic model of the lake, which in turn will allow improved management of Ålesund drinking water quality. The path planning capabilities in the present application are limited, since the hardware is not yet in place to allow the digital twin application to transmit navigation commands to either USV. For the time being, the only working capability which exists is the ability to export a set of sampling coordinates, which can be uploaded to one of the USVs as a form of offline path path planning.

Files in the repository include:

"Dashboard.py": launches a Streamlit app which contains all UI functions, such as data viewing and path planning. Must be called from the terminal as "(directory)> streamlit run Dashboard.py".
"Scraper.py": called by the Windows Task Scheduler to collect the last hourly reading from the profiler platform's IP address and add it as a new record in the appropriate table of the database. Note the the password for the existing database has been obscured for publication.
