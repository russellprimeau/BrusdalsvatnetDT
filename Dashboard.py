# Framework for a selection-driven, hierarchical app for viewing water quality data and controlling
# data acquisition systems using Streamlit as an interface.

# Launch by opening the terminal to the script's location and entering "streamlit run Dashboard.py".

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
import tempfile2 as tempfile
from streamlit_folium import st_folium, folium_static
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, DataRange1d, HoverTool, Range1d
from bokeh.palettes import Viridis256, Category20_20
from bokeh.layouts import column
from datetime import date, time, datetime, timedelta
from dateutil.relativedelta import relativedelta
import xarray as xr
import matplotlib.pyplot as plt
import contextily as ctx
import dfm_tools as dfmt
import plotly.express as px


def main():
    st.set_page_config("Brusdalsvatnet WQ Dashboard", layout="wide")
    st.sidebar.title("Choose Mode")
    selected_page = st.sidebar.radio("", ["Historic", "Hydrodynamic Model", "Interactive (Path Planning)"])

    if selected_page == "Historic":
        historic()
    elif selected_page == "Hydrodynamic Model":
        current()
    elif selected_page == "Interactive (Path Planning)":
        interactive()


def historic():
    st.header("Brusdalsvatnet Water Quality Dashboard")
    st.title("Historic Instrument Data")
    st.markdown("### Choose a source and format to view data from previous sampling missions")

    # Radio button for selecting the data source
    source = st.radio(
        "Select a data collection platform to display its past measurements",
        options=["Profiler Station", "Weather Station", "USV (Maritime Robotics Otter)", "USV (OceanAlpha SL40)"],
        horizontal=True)

    if source == "Profiler Station":
        # Radio button for selecting the dataset
        profiler_data = st.radio(
            "Select a dataset to display",
            options=["Hourly Surface Data", "Vertical Profiles"],  # "Current Cache"],  # Link times out
            horizontal=True)

        # Display the selected plot based on user choice
        if profiler_data == "Hourly Surface Data":
            hourly()
        else:
            vertical()
    elif source == "Weather Station":
        weather()
    else:
        usv_plot()
    st.write(
        "Find a bug? Or have an idea for how to improve the app? "
        "Please log suggestions [here](https://github.com/russellprimeau/BrusdalsvatnetDT/issues).")


# Function to the upload new profiler data from CSV
def upload_weather_csv():
    csv_file2 = "All_time.csv"  # Replace with the actual file path
    df = pd.read_csv(csv_file2, sep=';', decimal=',', parse_dates=['Time'], date_format='%Y-%m-%dT%H:%M:%S', header=0)

    # Add units to column names
    if df is not None:
        if df.columns[0] == "Time":
            column_names = {"Time": "Timestamp",
                            "1818_time: AA[mBar]": "Instantaneous atmospheric pressure (mBar)",
                            "1818_time: DD Retning[°]": "Wind direction 10minRollingAvg (°)",
                            "1818_time: DX_l[°]": "Hourly average wind direction (°)",
                            "1818_time: FF Hastighet[m/s]": "Average wind speed (m/s)",
                            "1818_time: FG_l[m/s]": "Maximum sustained wind speed, 3-second span (m/s)",
                            "1818_time: FG_tid_l[N/A]": "Time of maximum 3s Gust",
                            "1818_time: FX Kast[m/s]": "Maximum sustained wind speed, 10-minute span (m/s)",
                            "1818_time: FX_tid_l[N/A]": "Time of maximum 10 minute gust",
                            "1818_time: PO Trykk stasjonshøyde[mBar]": "Hourly average atmospheric pressure at station (mBar)",
                            "1818_time: PP[mBar]": "Maximum pressure differential, 3-hour span (mBar)",
                            "1818_time: PR Trykk redusert til havnivå[mBar]": "Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)",
                            "1818_time: QLI Langbølget[W/m2]": "Longwave (IR) radiation (W/m2)",
                            "1818_time: QNH[mBar]": "Instantaneous sea-level atmospheric pressure (mBar)",
                            "1818_time: QSI Kortbølget[W/m2]": "Shortwave (solar) radiation (W/m2)",
                            "1818_time: RR_1[mm]": "Hourly precipitation (mm/hr)",
                            "1818_time: TA Middel[°C]": "Instantaneous temperature (°C)",
                            "1818_time: TA_a_Max[°C]": "Hourly maximum temperature (°C)",
                            "1818_time: TA_a_Min[°C]": "Hourly minimum temperature (°C)",
                            "1818_time: UU Luftfuktighet[%RH]": "Average humidity (% relative humidity)"
            }
            df = df.rename(columns=column_names)  # Assign column names for profiler data

        # st.write("Uploaded DataFrame:")
        # st.dataframe(df)

        # Data cleaning
        for parameter in df.columns:
            df[parameter] = df[parameter].apply(lambda x: np.nan if x == 'NAN' else x)
        df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce', downcast='float')

        # Convert the time column to a datetime object
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(lambda x: x.to_pydatetime())
        df = df.sort_values(by="Timestamp")
        df = df.drop(columns=["Instantaneous atmospheric pressure (mBar)", "Wind direction 10minRollingAvg (°)",
                              "Time of maximum 3s Gust", "Time of maximum 10 minute gust",
                              "Instantaneous atmospheric pressure compensated for temperature, humidity and station elevation (mBar)",
                              "Instantaneous sea-level atmospheric pressure (mBar)","Instantaneous temperature (°C)"])
    return df


def weather():
    st.title("Brusdalen Weather Station")
    st.markdown("##### 62.484778°N 6.479667°E, 69 MASL")

    p = figure()

    df = upload_weather_csv()

    # Check if there are at least two columns in the DataFrame
    if len(df.columns) < 2:
        st.warning("DataFrame must have at least two columns for X and Y variables.")
        return None

    # Multi-select to select multiple Y variables, including "Select All"
    mc1, mc2 = st.columns(2, gap="small")
    with mc1:
        selected_variables = st.multiselect(
            "Select weather parameters to plot",
            ["Select All"] + list(df.columns[1:]), default=["Select All"]
        )

    clean_setting = st.radio(
        "Choose how to filter the dataset",
        options=["Remove Suspicious Values", "Raw"],
        horizontal=True
    )

    if clean_setting == "Remove Suspicious Values":

        # Set negative shortwave values to 0 (this is very common and appears to represent a calibration issue)
        df['Shortwave (solar) radiation (W/m2)'] = (
            np.where(df['Shortwave (solar) radiation (W/m2)'] < 0, 0, df['Shortwave (solar) radiation (W/m2)']))

        # Define conditions for each parameter which indicate errors in the data
        error_conditions = {
            "Timestamp": (df['Timestamp'] < pd.to_datetime('2000-01-01')) | (
                    df['Timestamp'] > pd.to_datetime('2099-12-31')),
            'Hourly average wind direction (°)': (df['Hourly average wind direction (°)'] < 0) | (df['Hourly average wind direction (°)'] > 360),
            "Average wind speed (m/s)": (df["Average wind speed (m/s)"] < 0) | (
                        df["Average wind speed (m/s)"] > 100),
            'Maximum sustained wind speed, 3-second span (m/s)': (df['Maximum sustained wind speed, 3-second span (m/s)'] < 0) |
                                                      (df['Maximum sustained wind speed, 3-second span (m/s)'] > 100),
            'Maximum sustained wind speed, 10-minute span (m/s)': (
                    df['Maximum sustained wind speed, 10-minute span (m/s)'] < 0) |
                                                      (df['Maximum sustained wind speed, 10-minute span (m/s)'] > 100),
            'Hourly average atmospheric pressure at station (mBar)': (df['Hourly average atmospheric pressure at station (mBar)'] < 860) | (df['Hourly average atmospheric pressure at station (mBar)'] > 1080),
            'Maximum pressure differential, 3-hour span (mBar)': (df['Maximum pressure differential, 3-hour span (mBar)'] < 0) | (df['Maximum pressure differential, 3-hour span (mBar)'] > 50),
            'Longwave (IR) radiation (W/m2)': (df['Longwave (IR) radiation (W/m2)'] < 0) | (
                    df['Longwave (IR) radiation (W/m2)'] > 750),
            'Shortwave (solar) radiation (W/m2)': (df['Shortwave (solar) radiation (W/m2)'] < 0) | (
                    df['Shortwave (solar) radiation (W/m2)'] > 900),
            'Hourly precipitation (mm/hr)': (df['Hourly precipitation (mm/hr)'] < 0) | (
                    df['Hourly precipitation (mm/hr)'] > 50),
            'Hourly maximum temperature (°C)': (df['Hourly maximum temperature (°C)'] < -40) | (df['Hourly maximum temperature (°C)'] > 40),
            'Hourly minimum temperature (°C)': (df['Hourly minimum temperature (°C)'] < -40) | (df['Hourly minimum temperature (°C)'] > 40),
            'Average humidity (% relative humidity)': (df['Average humidity (% relative humidity)'] < 0) | (df['Average humidity (% relative humidity)'] > 100)
        }

        # Replace values meeting the error conditions with np.nan using boolean indexing
        for col, condition in error_conditions.items():
            df.loc[condition, col] = np.nan

        # Define start and end timestamps for a range to drop (if any periods are obviously unreliable
        # start_removal = pd.to_datetime('2022-03-24 00:00')
        # end_removal = pd.to_datetime('2022-04-22 00:00')
        # Create boolean mask for rows to keep (outside the time range)
        # mask = (df['Timestamp'] < start_removal) | (df['Timestamp'] > end_removal)
        # # Drop rows not satisfying the mask (within the time range)
        # df = df[mask]

        st.write("Some suspicious values have been removed from the dataset, but errors may remain.")
    else:
        st.write("All logged values are displayed, which includes known errors such as uncalibrated measurements.")

    def DefineRange(daterange):
        if daterange == "Last Month":
            st.session_state.ksd = date.today() - timedelta(days=31)
            st.session_state.knd = date.today()
        elif daterange == "Last Year":
            st.session_state.ksd = datetime.now() - relativedelta(years=1)
            st.session_state.knd = datetime.now()
        elif daterange == "Maximum Extent":
            st.session_state.ksd = first_date
            st.session_state.knd = last_date

    first_date = df.iloc[0, 0]
    last_date = df.iloc[-1, 0]

    set_begin_date = date.today() - timedelta(days=31)
    set_last_date = date.today()
    daterange = ""

    dc1, dc2, dc3, dc4 = st.columns(4, gap="small")
    dc1.date_input("Begin plot range:", value=set_begin_date, key="ksd")
    dc2.date_input("End plot range:", value=set_last_date, key="knd")

    b1, b2, b3, b4, b5, b6 = st.columns(6, gap="small")
    b1.button("Maximum Extent", on_click=DefineRange, args=("Maximum Extent",))
    b2.button("Last Year", on_click=DefineRange, args=("Last Year",))
    b3.button("Last Month", on_click=DefineRange, args=("Last Month",))

    set_begin_date = datetime.combine(st.session_state.ksd, time())
    set_last_date = datetime.combine(st.session_state.knd, time())


    # Check if "Select All" is chosen
    if "Select All" in selected_variables:
        selected_variables = list(df.columns[1:])

    # Create a ColumnDataSource
    source = ColumnDataSource(df)
    time_difference = timedelta(hours=2)

    def update_w_hourly(selected_variables):
        p.title.text = f'Weather Parameters vs. Time'
        for variable, color in zip(selected_variables, Category20_20):
            # Convert 'Date' to a pandas Series to use shift operation
            date_series = pd.Series(source.data['Timestamp'])

            # Add a new column 'Gap' indicating when a gap is detected within each 'Depth' group
            source.data['Gap'] = (date_series - date_series.shift(1)) > time_difference

            # Replace the 'Value' with NaN when a gap is detected
            source.data[variable] = np.where(source.data['Gap'], np.nan, source.data[variable])

            line_render = p.line(
                x="Timestamp", y=variable, line_width=2, color=color, source=source, legend_label=variable
            )
            p.add_tools(HoverTool(renderers=[line_render], tooltips=[("Time", "@Timestamp{%Y-%m-%d %H:%M}"),
                                                                     (variable, f'@{{{variable}}}')],
                                  formatters={"@Timestamp": "datetime", }, mode="vline"))
            p.renderers.append(line_render)

    # Call the update_plot function with the selected variables for the first plot
    if not selected_variables:
        st.write("Please select at least one parameter to plot.")
    else:
        update_w_hourly(selected_variables)
        # Set plot properties
        p.title.text_font_size = "16pt"
        p.xaxis.axis_label = "Time"
        # p.xlim = (set_begin_date, set_last_date)
        plotrange = set_last_date - set_begin_date
        if plotrange > timedelta(days=62):
            p.x_range = Range1d(set_begin_date - timedelta(days=3), set_last_date + timedelta(days=3))
        else:
            p.x_range = Range1d(set_begin_date, set_last_date + timedelta(days=1, hours=3))
        p.yaxis.axis_label = "Parameter Value(s)"
        p.legend.title = "Weather Parameters"
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"  # Hide lines on legend click
        # p.add_layout(p.legend[0], 'below')  # Option to move the legend out of the plotspace
        # Set the x-axis formatter to display dates in the desired format
        p.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%Y/%m/%d %H:%M")
        # show(p)
        st.bokeh_chart(p, use_container_width=True)
        st.write("Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
                 "Click legend entries to toggle series on/off.")

# Function to the upload new profiler data from CSV
def upload_hourly_csv_page():
    csv_file2 = "Profiler_modem_SondeHourly.csv"  # Replace with the actual file path
    df = pd.read_csv(csv_file2, parse_dates=['TIMESTAMP'])

    # Add units to column names
    if df is not None:
        if df.columns[0] == "Timestamp":
            column_names = {
                "id": "id",
                "Timestamp": "Timestamp",
                "Record_Number": "Record Number",
                "Temperature": "Temperature (Celsius)",
                "Conductivity": "Conductivity (microSiemens/centimeter)",
                "Specific_Conductivity": "Specific Conductivity (microSiemens/centimeter)",
                "Salinity": "Salinity (parts per thousand, ppt)",
                "pH": "pH",
                "DO": "Dissolved Oxygen (% saturation)",
                "Turbidity_NTU": "Turbidity (NTU)",
                "Turbidity_FNU": "Turbidity (FNU)",
                "Position": "Depth (m)",
                "fDOM_RFU": "fDOM (RFU)",
                "fDOM_QSU": "fDOM (parts per billion QSU)",
                "lat": "Latitude",
                "lon": "Longitude",
            }
            df = df.rename(columns=column_names)  # Assign column names for profiler data
        elif df.columns[0] == "TIMESTAMP":
            column_names = {
                "TIMESTAMP": "Timestamp",
                "RECORD": "Record Number",
                "sensorParms(1)": "Temperature (Celsius)",
                "sensorParms(2)": "Conductivity (microSiemens/centimeter)",
                "sensorParms(3)": "Specific Conductivity (microSiemens/centimeter)",
                "sensorParms(4)": "Salinity (parts per thousand, ppt)",
                "sensorParms(5)": "pH",
                "sensorParms(6)": "Dissolved Oxygen (% saturation)",
                "sensorParms(7)": "Turbidity (NTU)",
                "sensorParms(8)": "Turbidity (FNU)",
                "sensorParms(9)": "Vertical Position (m)",
                "sensorParms(10)": "fDOM (RFU)",
                "sensorParms(11)": "fDOM (parts per billion QSU)",
                "lat": "Latitude",
                "lon": "Longitude",
            }
            df = df.rename(columns=column_names)  # Assign column names for profiler data

        # st.write("Uploaded DataFrame:")
        # st.dataframe(df)

        # Convert text as read-in to appropriate pandas formats
        for parameter in df.columns:
            df[parameter] = df[parameter].apply(lambda x: np.nan if x == 'NAN' else x)
        df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce', downcast='float')

        # Convert the time column to a datetime object (if not already)
        # df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(lambda x: x.to_pydatetime())
        df = df.sort_values(by="Timestamp")
        df = df.drop(columns=['Record Number', 'Vertical Position (m)'])
    return df


# Create a Bokeh figure
def hourly():
    st.title("Hourly Surface Data From the Profiler Station")
    p = figure(title="Time Series Data at 2.9m Depth")

    df = upload_hourly_csv_page()

    # Check if there are at least two columns in the DataFrame
    if len(df.columns) < 2:
        st.warning("DataFrame must have at least two columns for X and Y variables.")
        return None

    # Multi-select to select multiple Y variables, including "Select All"
    mc1, mc2 = st.columns(2, gap="small")
    with mc1:
        selected_variables = st.multiselect(
            "Select water quality parameters to plot",
            ["Select All"] + list(df.columns[1:11]), default=["Select All"]
        )

    clean_setting = st.radio(
        "Choose how to filter the dataset",
        options=["Remove Suspicious Values", "Raw"],
        horizontal=True
    )

    if clean_setting == "Remove Suspicious Values":
        # Define conditions for each parameter which indicate errors in the data
        error_conditions = {
            "Timestamp": (df['Timestamp'] < pd.to_datetime('2000-01-01')) | (
                    df['Timestamp'] > pd.to_datetime('2099-12-31')),
            "Temperature (Celsius)": (df['Temperature (Celsius)'] < 1) | (df['Temperature (Celsius)'] > 25),
            "Conductivity (microSiemens/centimeter)": (df['Conductivity (microSiemens/centimeter)'] < 0) |
                                                      (df['Conductivity (microSiemens/centimeter)'] > 45),
            "Specific Conductivity (microSiemens/centimeter)": (
                    df['Specific Conductivity (microSiemens/centimeter)'] < 1),
            "Salinity (parts per thousand, ppt)": (df['Salinity (parts per thousand, ppt)'] < 0),
            "pH": (df['pH'] < 2) | (df['pH'] > 12),
            "Dissolved Oxygen (% saturation)": (df['Dissolved Oxygen (% saturation)'] < 10) | (
                    df['Dissolved Oxygen (% saturation)'] > 120),
            "Turbidity (NTU)": (df['Turbidity (NTU)'] < 0),
            "Turbidity (FNU)": (df['Turbidity (FNU)'] < 0),
            "fDOM (RFU)": (df['fDOM (RFU)'] < 0) | (df['fDOM (RFU)'] > 100),
            "fDOM (parts per billion QSU)": (df['fDOM (parts per billion QSU)'] < 0) | (df['fDOM (parts per billion QSU)'] > 300),
            "Latitude": (df['Latitude'] < -90) | (df['Latitude'] > 90),
            "Longitude": (df['Longitude'] < -180) | (df['Longitude'] > 180)
        }

        # Replace values meeting the error conditions with np.nan using boolean indexing
        for col, condition in error_conditions.items():
            df.loc[condition, col] = np.nan

        # Define start and end timestamps for the range to drop
        start_removal = pd.to_datetime('2022-03-24 00:00')
        end_removal = pd.to_datetime('2022-04-22 00:00')

        # Create boolean mask for rows to keep (outside the time range)
        mask = (df['Timestamp'] < start_removal) | (df['Timestamp'] > end_removal)

        # Drop rows not satisfying the mask (within the time range)
        df = df[mask]

        st.write("Some suspicious values have been removed from the dataset, but errors may remain.")
    else:
        st.write("All logged values are displayed, which includes known errors such as uncalibrated measurements.")

    def DefineRange(daterange):
        if daterange == "Last Month":
            st.session_state.ksd = date.today() - timedelta(days=31)
            st.session_state.knd = date.today()
        elif daterange == "Last Year":
            st.session_state.ksd = datetime.now() - relativedelta(years=1)
            st.session_state.knd = datetime.now()
        elif daterange == "Maximum Extent":
            st.session_state.ksd = first_date
            st.session_state.knd = last_date

    first_date = df.iloc[0, 0]
    last_date = df.iloc[-1, 0]

    set_begin_date = date.today() - timedelta(days=31)
    set_last_date = date.today()
    daterange = ""

    dc1, dc2, dc3, dc4 = st.columns(4, gap="small")
    dc1.date_input("Begin plot range:", value=set_begin_date, key="ksd")
    dc2.date_input("End plot range:", value=set_last_date, key="knd")

    b1, b2, b3, b4, b5, b6 = st.columns(6, gap="small")
    b1.button("Maximum Extent", on_click=DefineRange, args=("Maximum Extent",))
    b2.button("Last Year", on_click=DefineRange, args=("Last Year",))
    b3.button("Last Month", on_click=DefineRange, args=("Last Month",))

    set_begin_date = datetime.combine(st.session_state.ksd, time())
    set_last_date = datetime.combine(st.session_state.knd, time())


    # Check if "Select All" is chosen
    if "Select All" in selected_variables:
        selected_variables = list(df.columns[1:11])

    # Create a ColumnDataSource
    source = ColumnDataSource(df)
    time_difference = timedelta(hours=2)

    def update_hourly(selected_variables):
        p.title.text = f'Water Quality Parameters vs. Time'

        for variable, color in zip(selected_variables, Category20_20):
            # Convert 'Date' to a pandas Series to use shift operation
            date_series = pd.Series(source.data['Timestamp'])

            # Add a new column 'Gap' indicating when a gap is detected within each 'Depth' group
            source.data['Gap'] = (date_series - date_series.shift(1)) > time_difference

            # Replace the 'Value' with NaN when a gap is detected
            source.data[variable] = np.where(source.data['Gap'], np.nan, source.data[variable])

            line_render = p.line(
                x="Timestamp", y=variable, line_width=2, color=color, source=source, legend_label=variable
            )
            p.add_tools(HoverTool(renderers=[line_render], tooltips=[("Time", "@Timestamp{%Y-%m-%d %H:%M}"),
                                                                     (variable, f'@{{{variable}}}')],
                                  formatters={"@Timestamp": "datetime", }, mode="vline"))
            p.renderers.append(line_render)

    # Call the update_plot function with the selected variables for the first plot
    if not selected_variables:
        st.write("Please select at least one parameter to plot.")
    else:
        update_hourly(selected_variables)
        # Set plot properties
        p.title.text_font_size = "16pt"
        p.xaxis.axis_label = "Time"
        # p.xlim = (set_begin_date, set_last_date)
        plotrange = set_last_date - set_begin_date
        if plotrange > timedelta(days=62):
            p.x_range = Range1d(set_begin_date - timedelta(days=3), set_last_date + timedelta(days=3))
        else:
            p.x_range = Range1d(set_begin_date, set_last_date + timedelta(days=1, hours=3))
        p.yaxis.axis_label = "Parameter Value(s)"
        p.legend.title = "Water Quality Parameters"
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"  # Hide lines on legend click
        # p.add_layout(p.legend[0], 'below')  # Option to move the legend out of the plotspace
        # Set the x-axis formatter to display dates in the desired format
        p.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%Y/%m/%d %H:%M")
        # show(p)
        st.bokeh_chart(p, use_container_width=True)
        st.write("Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
                 "Click legend entries to toggle series on/off.")


def vertical():
    ###################################################################################################################
    # Import and pre-process data

    # Read data from a CSV file into a Pandas DataFrame, skipping metadata rows
    csv_file2 = "Profiler_modem_PFL_Step.csv"  # Replace with the actual file path
    df = pd.read_csv(csv_file2)

    # Assign column names for profiler data
    column_names = {
        "TIMESTAMP": "Timestamp",
        "RECORD": "Record Number",
        "PFL_Counter": "Day",
        "CntRS232": "CntRS232",
        "RS232Dpt": "Vertical Position1 (m)",
        "sensorParms(1)": "Temperature (Celsius)",
        "sensorParms(2)": "Conductivity (microSiemens/centimeter)",
        "sensorParms(3)": "Specific Conductivity (microSiemens/centimeter)",
        "sensorParms(4)": "Salinity (parts per thousand, ppt)",
        "sensorParms(5)": "pH",
        "sensorParms(6)": "Dissolved Oxygen (% saturation)",
        "sensorParms(7)": "Turbidity (NTU)",
        "sensorParms(8)": "Turbidity (FNU)",
        "sensorParms(9)": "Vertical Position (m)",
        "sensorParms(10)": "fDOM (RFU)",
        "sensorParms(11)": "fDOM (parts per billion QSU)",
        "lat": "Latitude",
        "lon": "Longitude",
    }
    df = df.rename(columns=column_names)

    # Convert the time column to a datetime object
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(lambda x: x.to_pydatetime())

    # Drop extraneous variables
    df = df.drop(columns=['Record Number', 'Day', 'CntRS232', 'Vertical Position1 (m)'])

    # Create new columns for date and time
    df['Date'] = df['Timestamp'].dt.date
    df['Time'] = df['Timestamp'].dt.time

    # Add rounded depth for plotting
    df['Depth'] = df['Vertical Position (m)'].round().astype(int)

    # Reorder columns so that water parameters are to the right of 'metadata' (time, location)
    column_to_move = 'Longitude'
    df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]
    column_to_move = 'Latitude'
    df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]
    column_to_move = 'Depth'
    df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]
    column_to_move = 'Time'
    df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]
    column_to_move = 'Date'
    df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]
    column_to_move = 'Timestamp'
    df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]

    # Data cleaning
    for parameter in df.columns:
        df[parameter] = df[parameter].apply(lambda x: np.nan if x == 'NAN' else x)
    df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce', downcast='float')

    # Convert 'Date' to datetime objects, so it can be used to sort Vertical Profiles
    df['Date'] = pd.to_datetime(df['Date'])

    ###################################################################################################################
    # Plot 1: time series sorted by depths in Bokeh

    # Create Bokeh figure for the first plot
    p1 = figure(x_axis_label='Date', title='Water Quality Parameters vs. Date by Depth')

    st.title("Vertical Profile Data")
    st.markdown("### Time Series Data By Depth Contour")

    # # Option to include an image of the profiler alongside the title (commented because of image rights)
    # col1, col2 = st.columns([1, 1])
    #
    # # Add content to the first column
    # col1.title("Vertical Profiler Data")
    #
    # # Add content to the second column
    # image_url = "https://marvel-b1-cdn.bc0a.com/f00000000170758/www.ysi.com/image%20library/commerce%20products/
    # product%20listing%20pages/monitoring%20buoys%20and%20platforms/water-monitoring-data-buoys-environmental-
    # vehicle-platforms.jpg"
    # col2.image(image_url, use_column_width=True)

    # Add a multiselect box for dependent variables in the first plot
    variables_to_plot_p1 = ["Temperature (Celsius)", "Conductivity (microSiemens/centimeter)",
                            "Specific Conductivity (microSiemens/centimeter)", "Salinity (parts per thousand, ppt)",
                            "pH",
                            "Dissolved Oxygen (% saturation)", "Turbidity (NTU)", "Turbidity (FNU)", "fDOM (RFU)",
                            "fDOM (parts per billion QSU)"]
    mc1, mc2 = st.columns(2, gap="small")
    with mc1:
        selected_variables_p1 = st.multiselect('Select Water Quality Parameters', variables_to_plot_p1, default=["Temperature (Celsius)"])

    # User input for depth selection
    dc1, dc2 = st.columns(2, gap="small")
    with dc1:
        depth_options = st.multiselect(
            "Select depths at which to plot parameters (in meters)",
            options=["1m Intervals", "2m Intervals", "5m Intervals", "10m Intervals", "20m Intervals"] + list(
                df['Depth'].unique()),
            default=["10m Intervals"]  # Default is 0m, 10m, 20m...
        )

    clean_setting = st.radio(
        "Choose how to filter the dataset",
        options=["Remove Suspicious Values", "Raw"],
        horizontal=True
    )

    if clean_setting == "Remove Suspicious Values":
        # Define conditions for each parameter which indicate errors in the data
        error_conditions = {
            "Timestamp": (df['Timestamp'] < pd.to_datetime('2000-01-01')) | (
                    df['Timestamp'] > pd.to_datetime('2099-12-31')),
            "Temperature (Celsius)": (df['Temperature (Celsius)'] < 1) | (df['Temperature (Celsius)'] > 25),
            "Conductivity (microSiemens/centimeter)": (df['Conductivity (microSiemens/centimeter)'] < 0) |
                                                      (df['Conductivity (microSiemens/centimeter)'] > 45),
            "Specific Conductivity (microSiemens/centimeter)": (
                    df['Specific Conductivity (microSiemens/centimeter)'] < 1),
            "Salinity (parts per thousand, ppt)": (df['Salinity (parts per thousand, ppt)'] < 0),
            "pH": (df['pH'] < 1) | (df['pH'] > 13),
            "Dissolved Oxygen (% saturation)": (df['Dissolved Oxygen (% saturation)'] < 10) | (
                    df['Dissolved Oxygen (% saturation)'] > 120),
            "Turbidity (NTU)": (df['Turbidity (NTU)'] < 0),
            "Turbidity (FNU)": (df['Turbidity (FNU)'] < 0),
            "fDOM (RFU)": (df['fDOM (RFU)'] < 0) | (df['fDOM (RFU)'] > 100),
            "fDOM (parts per billion QSU)": (df['fDOM (parts per billion QSU)'] < 0) | (df['fDOM (parts per billion QSU)'] > 300),
            "Latitude": (df['Latitude'] < -90) | (df['Latitude'] > 90),
            "Longitude": (df['Longitude'] < -180) | (df['Longitude'] > 180)
        }

        # Replace values meeting the error conditions with np.nan using boolean indexing
        for col, condition in error_conditions.items():
            df.loc[condition, col] = np.nan

        # Define start and end timestamps for the range to drop
        # start_removal = pd.to_datetime('2022-03-24 00:00')
        # end_removal = pd.to_datetime('2022-04-22 00:00')
        #
        # # Create boolean mask for rows to keep (outside the time range)
        # mask = (df['Timestamp'] < start_removal) | (df['Timestamp'] > end_removal)

        # Drop rows not satisfying the mask (within the time range)
        # df = df[mask]

        st.write("Some suspicious values have been removed from the dataset, but errors may remain.")
    else:
        st.write("All logged values are displayed, which includes known errors such as uncalibrated measurements.")

    def DefineRange(daterange):
        if daterange == "Last Month":
            st.session_state.ksd2 = date.today() - timedelta(days=31)
            st.session_state.knd2 = date.today()
        elif daterange == "Last Year":
            st.session_state.ksd2 = datetime.now() - relativedelta(years=1)
            st.session_state.knd2 = datetime.now()
        elif daterange == "Maximum Extent":
            st.session_state.ksd2 = first_date
            st.session_state.knd2 = last_date

    first_date = df.iloc[0, 0]
    last_date = df.iloc[-1, 0]

    set_begin_date = date.today() - timedelta(days=31)
    set_last_date = date.today()
    daterange = ""

    dc1, dc2, dc3, dc4 = st.columns(4, gap="small")
    dc1.date_input("Begin plot range:", value=set_begin_date, key="ksd2")
    dc2.date_input("End plot range:", value=set_last_date, key="knd2")

    b1, b2, b3, b4, b5, b6 = st.columns(6, gap="small")
    b1.button("Maximum Extent", on_click=DefineRange, args=("Maximum Extent",))
    b2.button("Last Year", on_click=DefineRange, args=("Last Year",))
    b3.button("Last Month", on_click=DefineRange, args=("Last Month",))

    set_begin_date = datetime.combine(st.session_state.ksd2, time())
    set_last_date = datetime.combine(st.session_state.knd2, time())

    # Handle special options
    selected_depths = []
    if "1m Intervals" in depth_options:
        selected_depths = list(df['Depth'].unique())
    elif "2m Intervals" in depth_options:
        selected_depths.extend(sorted([1] + list(range(0, 80, 2))))
    elif "5m Intervals" in depth_options:
        selected_depths.extend(sorted([1] + list(range(0, 80, 5))))
    elif "10m Intervals" in depth_options:
        selected_depths.extend(sorted([1] + list(range(0, 80, 10))))
    elif "20m Intervals" in depth_options:
        selected_depths.extend(sorted([1] + list(range(0, 80, 20))))
    else:
        selected_depths = depth_options

    # Filter DataFrame based on selected depths
    filtered_df = df[df['Depth'].isin(selected_depths)]

    max_gap_days = 1.5

    if not selected_variables_p1 or not selected_depths:
        st.write("Please select at least one parameter and depth contour to plot.")
    else:
        # Group the data by 'Depth' and create separate ColumnDataSources for each group
        grouped_data = filtered_df.groupby('Depth')

        num_colors = (len(selected_depths)) * (len(selected_variables_p1))
        viridis_colors = Viridis256
        step = len(viridis_colors) // num_colors
        viridis_subset = viridis_colors[::step][:num_colors]

        # Callback function for variable selection in the first plot
        def update_plot_p1(selected_variables_p1):
            # Group the data by 'Depth' and create separate ColumnDataSources for each group
            # grouped_data = df.groupby('Depth')

            p1.title.text = f'{", ".join(selected_variables_p1)} vs. Date for Different Depths'
            p1.renderers = []  # Remove existing renderers

            for i, (depth, group) in enumerate(grouped_data):
                depth_source = ColumnDataSource(group)

                for j, var in enumerate(selected_variables_p1):
                    # Convert 'Date' to a pandas Series to use shift operation
                    date_series = pd.Series(depth_source.data['Date'])

                    # Add a new column 'Gap' indicating when a gap is detected within each 'Depth' group
                    depth_source.data['Gap'] = (date_series - date_series.shift(1)).dt.days > max_gap_days

                    # Replace the 'Value' with NaN when a gap is detected
                    depth_source.data[var] = np.where(depth_source.data['Gap'], np.nan, depth_source.data[var])
                    renderer = p1.line(x='Timestamp', y=var, source=depth_source, line_width=2,
                                       line_color=viridis_subset[num_colors - (1 + i + j * len(selected_variables_p1))],
                                       legend_label=f'{depth}m: {var}')
                    p1.add_tools(HoverTool(renderers=[renderer],
                                           tooltips=[("Time", "@Timestamp{%Y-%m-%d %H:%M}"), ("Depth", f'{depth}'),
                                                     (var, f'@{{{var}}}')], formatters={"@Timestamp": "datetime", },
                                           mode="vline"))
                    p1.renderers.append(renderer)

        # Call the update_plot function with the selected variables for the first plot
        update_plot_p1(selected_variables_p1)

        # Show legend for the first plot
        p1.legend.title = 'Depth'
        p1.legend.location = "top_left"
        p1.legend.label_text_font_size = '10px'
        p1.legend.click_policy = "hide"  # Hide lines on legend click
        p1.yaxis.axis_label = "Variable Value(s)"
        p1.xaxis.axis_label = "Time"
        plotrange = set_last_date - set_begin_date
        if plotrange > timedelta(days=62):
            p1.x_range = Range1d(set_begin_date - timedelta(days=3), set_last_date + timedelta(days=3))
        else:
            p1.x_range = Range1d(set_begin_date, set_last_date + timedelta(days=1, hours=3))
        p1.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%y/%m/%d %H:%M")

        # Display the Bokeh chart for the first plot using Streamlit
        st.bokeh_chart(p1, use_container_width=True)
        st.write("Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
                 "Click legend entries to toggle series on/off.")

    ###################################################################################################################
    # Plot 2: Instantaneous Vertical Profile
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Instantaneous Profile")

    # Add a multiselect box for parameters in the second plot
    mc1, mc2 = st.columns(2, gap="small")
    with mc1:
        selected_variables_p2 = st.multiselect('Select water quality parameters', variables_to_plot_p1,
                                               default=["Temperature (Celsius)"])

        # Add a multiselect box for date for the second plot
        selected_dates_p2 = st.multiselect('Select dates for vertical profiles by typing or scrolling',
                                           df['Date'].dt.strftime('%Y-%m-%d').unique(),
                                           default=df['Date'].iloc[-1:].dt.strftime('%Y-%m-%d'))

        profile_times = ['00:00 AM (Night)', '12:00 PM (Day)']

        # Add a multiselect box for choosing between plotting the AM or PM profiling
        nightman_dayman = st.radio("Select between night or day profile (profiles are usually collected twice per day)",
                                   profile_times, horizontal=True)  # Assuming the first column is x

    # Create Bokeh figure for the second plot only if a date is selected
    if not selected_dates_p2 or not selected_variables_p2:
        st.write("Please select at least one parameter and profile date to plot.")
    else:
        # Create Bokeh figure for the second plot
        p2 = figure(x_axis_label=f'{", ".join(selected_variables_p2)}', y_axis_label='Depth',
                    title=f'Vertical Profile for {", ".join(selected_variables_p2)} on {", ".join(selected_dates_p2)} '
                          f'at {", ".join(nightman_dayman)}')

        # Callback function for variable selection and date range in the second plot
        def update_plot_p2(selected_variables_p2, selected_dates_p2):
            p2.title.text = f'Vertical Profile for {", ".join(selected_variables_p2)} on {", ".join(selected_dates_p2)}'

            p2.renderers = []  # Remove existing renderers

            for j, date_val in enumerate(selected_dates_p2):
                # Filter data based on selected date for the second plot
                filtered_data_p2 = df[df['Date'] == pd.to_datetime(date_val)]
                if nightman_dayman == '00:00 AM (Night)':
                    filtered_data_p2 = filtered_data_p2[filtered_data_p2['Time'] < pd.to_datetime('12:00:00').time()]
                else:
                    filtered_data_p2 = filtered_data_p2[filtered_data_p2['Time'] >= pd.to_datetime('12:00:00').time()]

                # Sort the data by 'Depth'
                filtered_data_p2 = filtered_data_p2.sort_values(by='Depth')

                # Create Bokeh ColumnDataSource for the second plot
                source_plot2 = ColumnDataSource(filtered_data_p2)

                for i, var in enumerate(selected_variables_p2):
                    line_renderer = p2.line(x=var, y='Depth', source=source_plot2, line_width=1,
                                            line_color=Category20_20[i + j * len(selected_dates_p2)],
                                            legend_label=f'{var} : {date_val}')
                    p2.add_tools(
                        HoverTool(renderers=[line_renderer], tooltips=[("Depth", '@Depth'), (var, f'@{{{var}}}')],
                                  mode="vline"))
                    p2.renderers.append(line_renderer)

            # Reverse the direction of the Y-axis
            p2.y_range = DataRange1d(start=source_plot2.data['Depth'].max() + 3, end=0)

        # Call the update_plot function with the selected variables and date for the second plot
        update_plot_p2(selected_variables_p2, selected_dates_p2)

        # Show legend for the second plot
        p2.legend.title = 'Parameters'
        p2.legend.location = "top_left"
        p2.legend.label_text_font_size = '10px'
        p2.legend.click_policy = "hide"  # Hide lines on legend click

        # Display the Bokeh chart for the second plot using Streamlit
        st.bokeh_chart(p2, use_container_width=True)
        st.write("Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
                 "Click legend entries to toggle series on/off.")


def usv_plot():
    def display_mis(filename):
        track = pd.read_csv(filename)
        column_names = ['Time, ms', 'Mode', 'Status', 'Lat', 'Lon', 'Speed, m/s', 'Heading 1', 'Heading 2', 'var3',
                        'var4', 'var5', 'var6', 'var7', 'Battery', 'var9', 'var10', 'var11', 'Mission', 'Waypoint']
        track.columns = column_names
        track['Time, ms'] = pd.to_datetime(track['Time, ms'], unit='ms')
        track['Time'] = track['Time, ms'].dt.floor('s')

        exclude_cols = ['Time, ms', 'Time', 'Lat', 'Lon']
        included_cols = [col for col in track.columns if col not in exclude_cols]
        hover_labels = st.multiselect(label="Choose log parameters to display",
                                      options=["Select All"] + included_cols)

        if "Select All" in hover_labels:
            hover_labels = included_cols

        center_lat = (track['Lat'].max() + track['Lat'].min())/2
        center_lon = (track['Lon'].max() + track['Lon'].min())/2


        fig = px.line_mapbox(track, lat="Lat", lon="Lon", hover_data=['Time'] + hover_labels)

        fig.update_layout(mapbox_style="open-street-map", mapbox_zoom=12, mapbox_center_lat=center_lat,
                          mapbox_center_lon=center_lon, margin={"r": 0, "t": 0, "l": 0, "b": 0}, width=1600,
                          height=800)

        st.plotly_chart(fig, use_container_width=True)

    # Get all files in the current directory
    all_files = os.listdir()

    filtered_files = [f for f in all_files if f.endswith('.mis')] + ["Upload your own"]
    hc1, hc2 = st.columns(2, gap="small")
    with hc1:
        selected_file = st.selectbox(label="Select which mission log to display", options=filtered_files)

    if selected_file == "Upload your own":
        hc1, hc2 = st.columns(2, gap="small")
        with hc1:
            uploaded = st.file_uploader(label='Upload a mission log file')
        # Create a temp filepath to use to access the uploaded file
        if uploaded is not None:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(uploaded.read())
                file_path = temp_file.name
            display_mis(file_path)
    else:
        display_mis(selected_file)


def display_map(o_file):
    crs = 'EPSG:4326'
    raster_res = 50
    umag_clim = None
    scale = 1.5

    # Open and merge mapfile with xugrid(xarray) and print netcdf structure
    uds_map = dfmt.open_partitioned_dataset(o_file)

    datavars = list(uds_map.data_vars)  # List of all data variables
    print("uds_map", uds_map)
    # But don't use that. Create a list of only parameters which have a mesh2d_nFaces coordinate suitable for plots:
    includes_coordinate = "mesh2d_nFaces"
    excludes_coordinates = ["mesh2d_nEdges", "mesh2d_nNodes", "mesh2d_nMax_face_nodes"]
    mesh2d_nFaces_list = []
    for name, var in uds_map.data_vars.items():
        if (includes_coordinate in var.dims and all(coord not in var.dims for coord in excludes_coordinates)):
            mesh2d_nFaces_list.append(name)
    # print("mesh2d_nFaces_list", mesh2d_nFaces_list)

    # Dictionary of more descriptive names for known Delft3D output variables
    # (but there are more which are not described in documentation)
    parameter_names = {
        "Projected coordinate system": "wgs84",
        "z-coordinate of mesh nodes (m)": "mesh2d_node_z",
        "X-coordinate bounds of mesh faces (m)": "mesh2d_face_x_bnd",
        "Y-coordinate bounds of mesh faces (m)": "mesh2d_face_y_bnd",
        "Edge type (relation between edge and flow geometry)": "mesh2d_edge_type",
        "Cell area (m^2)": "mesh2d_flowelem_ba",
        "Flow element center bedlevel (m)": "mesh2d_flowelem_bl",
        "Latest computational timestep size in each output interval (s)": "timestep",
        "Water level (m)": "mesh2d_s1",
        "Water depth (m)": "mesh2d_waterdepth",
        "Velocity at velocity point, n-component (m/s)": "mesh2d_u1",
        "Flow element center velocity vector, x-component (m/s)": "mesh2d_ucx",
        "Flow element center velocity vector, y-component (m/s)": "mesh2d_ucy",
        "Flow element center velocity magnitude (m/s)": "mesh2d_ucmag",
        "Discharge through flow link at current time (m^3/s)": "mesh2d_q1",
        "Salinity in flow element (.001)": "mesh2d_sa1",
        "Temperature in flow element (C)": "mesh2d_tem1",
        "Flow element center wind velocity vector, x-component (m/s)": "mesh2d_windx",
        "Flow element center wind velocity vector, y-component (m/s)": "mesh2d_windy",
        "Edge wind velocity, x-component (m/s)": "mesh2d_windxu",
        "Edge wind velocity, y-component (m/s)": "mesh2d_windyu",
    }

    # Create a list of more descriptive parameter names for display in the UI
    mesh2d_nFaces_list = [key for key, value in parameter_names.items() if value in mesh2d_nFaces_list]
    # mesh2d_nFaces_list = [parameter_names.get(value, value) for value in mesh2d_nFaces_list]

    # Extract and reformat a list of all time steps, and a dictionary for converting the calendar time to an index
    times = uds_map.coords["time"]
    formatted_times = times.dt.strftime("%Y-%m-%d %H:%M:%S")
    formatted_times = formatted_times.values.tolist()
    times_dict = {value: i for i, value in enumerate(formatted_times)}
    # print('formatted_times', type(formatted_times), formatted_times)
    # print('times[0]', type(times[0]), times[0])

    # Check if the uploaded map file includes layers
    if uds_map.dims['mesh2d_nLayers'] is not None:
        num_layers = uds_map.dims['mesh2d_nLayers']
    else:
        num_layers = 0

    xmin = 6.385  #
    xmax = 6.57
    ymin = 62.46
    ymax = 62.49


    # Plot water level on map
    fig_surface, ax = plt.subplots(figsize=(20,3))

    cc1, cc2 = st.columns(2, gap="small")
    with cc1:
        parameter_key = st.selectbox("Select parameter to display", mesh2d_nFaces_list)
        parameter = parameter_names.get(parameter_key)
        selected_time_key = st.selectbox("Select the time to display", list(times_dict.keys()))
        selected_time_index = times_dict.get(selected_time_key)

        if num_layers > 1:
            max_depth = uds_map['mesh2d_waterdepth'].max().to_numpy()[()]
            # print("max_depth data: type, size, shape:", type(max_depth), max_depth.size, max_depth.shape, max_depth)
            layer_depths = np.round(np.linspace(0, max_depth - max_depth/num_layers, num_layers))
            layer_depths = layer_depths.tolist()
            layer_list = list(reversed(range(0, num_layers)))
            depth_selected = st.selectbox("Select depth layer to display (m)", layer_depths)  # Create the dropdown menu
            layer = layer_list[layer_depths.index(depth_selected)]
            # print("in depth ", max_depth, " from ", layer_depths, " selected ", depth_selected, " indicating layer ", layer+1)
            pc = uds_map[parameter].isel(time=selected_time_index, mesh2d_nLayers=layer,
                                     missing_dims='ignore').ugrid.plot(cmap='jet', add_colorbar=False)
        else:
            pc = uds_map[parameter].isel(time=selected_time_index, missing_dims='ignore').ugrid.plot(cmap='jet', add_colorbar=False)

    if crs is None:
        ax.set_aspect('equal')
    else:
        ctx.add_basemap(ax=ax, source=ctx.providers.OpenTopoMap, crs=crs, attribution=False)
    # fig_surface.suptitle(parameter_key)
    colorbar = plt.colorbar(pc, orientation="vertical", fraction=0.01, pad=0.001)
    # Set colorbar label
    colorbar.set_label(parameter_key)

    # Add a slider for selecting where to take a cross-section of the simulated water body
    latlon = st.radio("Choose the orientation of the cross section", options=("Longitude", "Latitude"), horizontal=True)
    if latlon == "Longitude":
        cross_section = st.slider("Select the longitude of the cross section for depth view", min_value=xmin,
                              max_value=xmax, value=(xmin + xmax) / 2, step=.001, format="%.3f")
        ax.axvline(cross_section, color='red')
        line_array = np.array([[cross_section, ymin],
                               [cross_section, ymax]])
    else:
        cross_section = st.slider("Select the longitude of the cross section for depth view", min_value=ymin,
                                  max_value=ymax, value=(ymin + ymax) / 2, step=.001, format="%.3f")
        ax.axhline(cross_section, color='red')
        line_array = np.array([[xmin, cross_section],
                               [xmax, cross_section]])

    ax.set_aspect('equal')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("")
    st.markdown(f"### {parameter_key} at depth of {depth_selected} m at {selected_time_key}")
    st.pyplot(fig_surface)


    # Plot a cross-section of the selected parameter at the line defined using the 'cross_section' slider
    if line_array is not None:
        uds_crs = dfmt.polyline_mapslice(uds_map.isel(time=selected_time_index), line_array)
        fig_cross, ax = plt.subplots(figsize=(20,3))
        cs = uds_crs[parameter].ugrid.plot(cmap='jet', add_colorbar=False)
        # Calculate the range for x and y data
        # x_range = max(data_x) - min(data_x)
        # y_range = max(data_y) - min(data_y)

        # # Calculate the 1% extra for the limits
        # x_limit = x_range * 0.01
        # y_limit = y_range * 0.01
        #
        # # Set the limits for x and y axis on the 'ax' object
        # ax.set_xlim(min(data_x) - x_limit, max(data_x) + x_limit)
        # ax.set_ylim(min(data_y) - y_limit, max(data_y) + y_limit)

        # Plot your data on the 'ax' object
        # ax.plot(data_x, data_y)
        ax.set_xlabel("Position, m")
        ax.set_ylabel("Depth, m")
        ax.set_title("")
        # fig_surface.suptitle(parameter_key)
        colorbar = plt.colorbar(cs, orientation="vertical", fraction=0.1, pad=0.001)
        # Set colorbar label
        colorbar.set_label(parameter_key)

        st.markdown(f"### Cross-section of {parameter_key} at {latlon} = {cross_section}, {selected_time_key}")
        st.pyplot(fig_cross)


def display_his(o_file):
    file_nc_his = o_file
    sel_slice_x, sel_slice_y = slice(50000, 55000), slice(None, 424000)
    crs = 'EPSG:4326'
    raster_res = 50
    umag_clim = None
    scale = 1.5
    line_array = np.array([[53181.96942503, 424270.83361629],
                           [55160.15232593, 416913.77136685]])

    # Open hisfile with xarray and print netcdf structure
    if file_nc_his is not None:
        ds_his = xr.open_mfdataset(file_nc_his, preprocess=dfmt.preprocess_hisnc)
        print('ds_his', ds_his)
        print('ds_his contains along station_id:\n', ds_his.coords['stations'].values)
    else:
        st.write("No time series data is available in this directory.")

    num_layers = ds_his.dims['laydim']
    max_depth = -ds_his.coords['zcoordinate_c'].min().to_numpy()[()]
    # print("max_depth data: type, size, shape:", type(max_depth), max_depth.size, max_depth.shape, max_depth, num_layers)
    layer_depths = np.round(np.linspace(0, max_depth - max_depth / num_layers, num_layers))
    layer_depths = layer_depths.tolist()
    layer_list = list(reversed(range(0, num_layers)))

    # Example of how to extract data fields from map.nc file:
    includes_coordinate = "stations"
    excludes_coordinates = ["station_geom_node_count",  "station_id", "station_geom", "station_geom_node_coordx",
                            "station_geom_node_coordy", 'bedlevel']
    station_var_list = []
    for name, var in ds_his.data_vars.items():
        if (includes_coordinate in var.dims and all(coord not in var.dims for coord in excludes_coordinates)):
            station_var_list.append(name)
    # print("station_var_list", station_var_list)

    includes_coordinate = "source_sink"
    excludes_coordinates = ["station_geom_node_count"]
    source_sink_var_list = []
    for name, var in ds_his.data_vars.items():
        if (includes_coordinate in var.dims and all(coord not in var.dims for coord in excludes_coordinates)):
            source_sink_var_list.append(name)
    # print("source_sink_var_list", source_sink_var_list)

    hc1, hc2 = st.columns(2, gap="small")
    with hc1:
        hisoptions = ['Time series', 'Instantaneous (vs. depth)']
        plottype = st.radio("Choose which type of data to display:", options=hisoptions, horizontal=True)
        locations = st.multiselect("Select stations at which to plot", ds_his.coords['stations'].values, default=ds_his.coords['stations'].values[0])
        feature = st.selectbox("Select a variable to plot", station_var_list)


    if plottype == hisoptions[0]:
        hc1, hc2 = st.columns(2, gap="small")
        with hc1:
            axis_options = ['Single Depth', 'All Depths']
            axis_count = st.radio("Select whether to display values from a single depth or all depths simultaneously", options=axis_options)
        if axis_count == axis_options[0]:
            hc1, hc2 = st.columns(2, gap="small")
            with hc1:
                depth_selected = st.selectbox("Select depth at which to plot", layer_depths)
            layers = layer_list[layer_depths.index(depth_selected)]

            data_fromhis_xr = ds_his[feature].sel(stations=locations, laydim=layers)
            fig, ax = plt.subplots(figsize=(20,3))
            data_fromhis_xr.plot.line('-', ax=ax, x='time')
            ax.legend(data_fromhis_xr.stations.to_series(), fontsize=9)  # optional, to reduce legend font size
            # data_fromhis_xr_dailymean = data_fromhis_xr.resample(time='D').mean(
            #     dim='time')  # add daily mean values in the back #TODO: raises "TypeError: __init__() got an unexpected keyword argument 'base'" since py39 environment
            # data_fromhis_xr_dailymean.plot.line('-', ax=ax, x='time', add_legend=False, zorder=0, linewidth=.8,
            #                                     color='grey')
            ax.set_xlabel('Time')
            ax.set_ylabel(feature)
            fig.tight_layout()
            st.pyplot(fig)
        else:
            # plot his data: temperature zt at one station
            if file_nc_his is not None:
                for i, station in enumerate(locations):
                    ds_his_sel = ds_his.isel(stations=i).isel(time=slice(0, 50))
                    fig_z, ax = plt.subplots(1, 1, figsize=(20,3))
                    pc = dfmt.plot_ztdata(ds_his_sel, varname=feature, ax=ax,
                                          cmap='jet')  # temperature pcolormesh
                    # CS = dfmt.plot_ztdata(ds_his_sel, varname=feature, ax=ax, only_contour=True, levels=9,
                    #                       colors='k', linewidths=0.8, linestyles='solid')  # temperature contour
                    # ax.clabel(CS, fontsize=10)
                    colorbar = plt.colorbar(pc, orientation="vertical", fraction=0.1, pad=0.001)
                    colorbar.set_label(feature)
                    ax.set_xlabel(feature)
                    ax.set_ylabel("Depth")
                    st.markdown(f"### {feature} vs. time at {locations[i]}")
                    st.pyplot(fig_z)

    else:
        hc1, hc2 = st.columns(2, gap="small")
        with hc1:
            realtimes = list(ds_his.coords['time'].values)
            plottime = st.selectbox("Select times at which to plot instantaneous values vs. depth",
                                  ds_his.coords['time'].values)

        time_list = [i for i in range(ds_his.dims['time'])]
        # print('time list', time_list, type(time_list))
        # print('real times', realtimes, type(realtimes))

        timeindex = time_list[realtimes.index(plottime)]
        data_fromhis_xr = ds_his[feature].sel(stations=locations).isel(time=timeindex)
        fig_instant_profile, ax = plt.subplots(figsize=(20,6))
        data_fromhis_xr.T.plot.line('-', ax=ax, y='zcoordinate_c')
        # ax.legend(data_fromhis_xr.stations.to_series(), fontsize=9)  # optional, to reduce legend font size
        # ax.set_aspect('equal')
        ax.set_xlabel(feature)
        ax.set_ylabel("Depth")
        ax.set_title("")
        st.markdown(f"### {feature} at {plottime}")
        fig_instant_profile.tight_layout()
        st.pyplot(fig_instant_profile)


def current():
    """
    Display and explore Delft3D output files
    """
    st.header("Brusdalsvatnet Water Quality Dashboard")
    st.title("Hydrodynamic Model of Current Conditions")

    # Get all files in the current directory
    all_files = os.listdir()

    output_options = ["Spatial distributions (map file)", "Fixed locations (history file)"]
    d3d_output = st.radio("Select which type of model outputs to display",
                          options=output_options, horizontal=True)

    # Filter files based on extension
    if d3d_output == output_options[1]:
        filtered_files = [f for f in all_files if f.endswith('his.nc')] + ["Upload your own"]
        hc1, hc2 = st.columns(2, gap="small")
        with hc1:
            selected_file = st.selectbox(label="Select which model output to display", options=filtered_files)
        if selected_file == "Upload your own":
            hc1, hc2 = st.columns(2, gap="small")
            with hc1:
                uploaded = st.file_uploader(label='Upload your own Delft3D history output file (his.nc), maximum size 200MB', type='nc')
            # Create a temp filepath to use to access the uploaded file
            if uploaded is not None:
                if uploaded.name.endswith('his.nc'):
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded.read())
                        file_path = temp_file.name
                    display_his(file_path)
                else:
                    st.markdown("### File uploaded is not a valid history file")
        else:
            display_his(selected_file)

    else:
        filtered_files = [f for f in all_files if f.endswith('map.nc')] + ["Upload your own"]
        # if uploaded is not None:
        #     filtered_files.append(uploaded.name)
        hc1, hc2 = st.columns(2, gap="small")
        with hc1:
            selected_file = st.selectbox(label="Select which model output to display", options=filtered_files)
        if selected_file == "Upload your own":
            hc1, hc2 = st.columns(2, gap="small")
            with hc1:
                uploaded = st.file_uploader(label='Upload your own Delft3D NetCDF map output file (map.nc), maximum size 200MB', type='nc')
            # Create a temp filepath to use to access the uploaded file
            if uploaded is not None:
                if uploaded.name.endswith('map.nc'):
                    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                        temp_file.write(uploaded.read())
                        file_path = temp_file.name
                    display_map(file_path)
                else:
                    st.markdown("### File uploaded is not a valid map file")
        else:
            display_map(selected_file)


def interactive():
    st.header("Brusdalsvatnet Water Quality Dashboard")
    st.title("Path Planning ")
    st.write("Select locations for sampling missions by clicking on the map")
    mission_columns = ["Latitude", "Longitude", "Type", "Velocity (m/s)", "Waypoint radius (m)", "Timeout (s)"]

    # CSV file path
    csv_file_path = "mission.csv"

    # Read an existing CSV file into a DataFrame
    try:
        df_coord = pd.read_csv(csv_file_path, sep='\t', encoding='utf-8', header=None, names=mission_columns)
    except FileNotFoundError:
        # If the file doesn't exist, create an empty DataFrame
        df_coord = pd.DataFrame()

    offline_plan(csv_file_path, df_coord)

    # Button to overwrite "mission.csv"
    if st.button("Begin new mission (clear previous waypoints)"):
        st.write("You might need to click more than once and wait a moment...")
        empty_df = pd.DataFrame()
        empty_df.to_csv(csv_file_path, sep='\t', index=False)

    st.write(
        "Find a bug? Or have an idea for how to improve the app? "
        "Please log suggestions [here](https://github.com/russellprimeau/BrusdalsvatnetDT/issues).")


def offline_plan(csv_file_path, df_coord):
    def get_pos(lat, lng):
        return lat, lng

    m = folium.Map(location=[62.476994, 6.469730], zoom_start=13)
    m.add_child(folium.LatLngPopup())
    map1 = st_folium(m, use_container_width=True)

    data = None
    if map1.get("last_clicked"):
        data = get_pos(map1["last_clicked"]["lat"], map1["last_clicked"]["lng"])

        # Update the map with a marker for the clicked point
        folium.Marker(data, popup=str(data)).add_to(m)

    if data is not None:
        st.write("Sampling mission: ")  # Writes to the app
        # Append the new values to the DataFrame as a new row
        new_row = pd.DataFrame({"Latitude": [data[0]], "Longitude": [data[1]], "Type": 1, "Velocity": 4.12,
                                "Waypoint": 6.0, "Timeout": 60.00})
        new_row.columns = df_coord.columns
        df_coord = pd.concat([df_coord, new_row], ignore_index=True)
        st.dataframe(df_coord)
        st.write("Hover the cursor over the table for the option to export as a .CSV file.")  # Writes to the app

        # Add markers for each coordinate in the DataFrame
        for index, row in df_coord.iterrows():
            folium.Marker([row['Latitude'], row['Longitude']], popup=index).add_to(m)

        # Write the updated DataFrame back to the CSV file
        df_coord.to_csv(csv_file_path, sep='\t', index=False, header=False)


if __name__ == "__main__":
    main()
