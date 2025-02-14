# Framework for a selection-driven, hierarchical app for viewing water quality data and controlling
# data acquisition systems using Streamlit as an interface.

# Launch by opening the terminal to the script's location and entering "streamlit run Dashboard.py".

import os
import numpy as np
import pandas as pd
import streamlit as st
import folium
import tempfile2 as tempfile
from streamlit_folium import st_folium, folium_static
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, DataRange1d, HoverTool, Range1d
from bokeh.palettes import Viridis256, Category20_20
# from bokeh.layouts import column
from datetime import date, time, datetime, timedelta
from dateutil.relativedelta import relativedelta
import xarray as xr
import matplotlib.pyplot as plt
import contextily as ctx
import dfm_tools as dfmt
import plotly.express as px
import math
import bisect


def main():
    st.set_page_config("Brusdalsvatnet WQ Dashboard", layout="wide")
    st.sidebar.title("Choose Mode")
    selected_page = st.sidebar.radio("", ["Historic", "Hydrodynamic Model", "Interactive (Path Planning)"])
    # Get all files in the current directory
    all_files = os.listdir()

    if selected_page == "Historic":
        historic()
    elif selected_page == "Hydrodynamic Model":
        st.title("Hydrodynamic Model")
        current(all_files, directory_path=None)
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
                            "1818_time: PO Trykk stasjonshøyde[mBar]":
                                "Hourly average atmospheric pressure at station (mBar)",
                            "1818_time: PP[mBar]": "Maximum pressure differential, 3-hour span (mBar)",
                            "1818_time: PR Trykk redusert til havnivå[mBar]":
                                "Instantaneous atmospheric pressure compensated for temperature, humidity and station "
                                "elevation (mBar)",
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
                              "Instantaneous atmospheric pressure compensated for temperature, "
                              "humidity and station elevation (mBar)",
                              "Instantaneous sea-level atmospheric pressure (mBar)", "Instantaneous temperature (°C)"])
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
            'Hourly average wind direction (°)': (df['Hourly average wind direction (°)'] < 0) | (
                    df['Hourly average wind direction (°)'] > 360),
            "Average wind speed (m/s)": (df["Average wind speed (m/s)"] < 0) | (
                    df["Average wind speed (m/s)"] > 100),
            'Maximum sustained wind speed, 3-second span (m/s)': (df[
                                                                      'Maximum sustained wind speed, 3-second span (m/s)'] < 0) |
                                                                 (df[
                                                                      'Maximum sustained wind speed, 3-second span (m/s)'] > 100),
            'Maximum sustained wind speed, 10-minute span (m/s)': (
                                                                          df[
                                                                              'Maximum sustained wind speed, 10-minute span (m/s)'] < 0) |
                                                                  (df[
                                                                       'Maximum sustained wind speed, 10-minute span (m/s)'] > 100),
            'Hourly average atmospheric pressure at station (mBar)': (df[
                                                                          'Hourly average atmospheric pressure at station (mBar)'] < 860) | (
                                                                             df[
                                                                                 'Hourly average atmospheric pressure at station (mBar)'] > 1080),
            'Maximum pressure differential, 3-hour span (mBar)': (df[
                                                                      'Maximum pressure differential, 3-hour span (mBar)'] < 0) | (
                                                                         df[
                                                                             'Maximum pressure differential, 3-hour span (mBar)'] > 50),
            'Longwave (IR) radiation (W/m2)': (df['Longwave (IR) radiation (W/m2)'] < 0) | (
                    df['Longwave (IR) radiation (W/m2)'] > 750),
            'Shortwave (solar) radiation (W/m2)': (df['Shortwave (solar) radiation (W/m2)'] < 0) | (
                    df['Shortwave (solar) radiation (W/m2)'] > 900),
            'Hourly precipitation (mm/hr)': (df['Hourly precipitation (mm/hr)'] < 0) | (
                    df['Hourly precipitation (mm/hr)'] > 50),
            'Hourly maximum temperature (°C)': (df['Hourly maximum temperature (°C)'] < -40) | (
                    df['Hourly maximum temperature (°C)'] > 40),
            'Hourly minimum temperature (°C)': (df['Hourly minimum temperature (°C)'] < -40) | (
                    df['Hourly minimum temperature (°C)'] > 40),
            'Average humidity (% relative humidity)': (df['Average humidity (% relative humidity)'] < 0) | (
                    df['Average humidity (% relative humidity)'] > 100)
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
        p.add_layout(p.legend[0], 'right')
        # p.legend.location = "top_left"
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
            "Salinity (parts per thousand, ppt)": (df['Salinity (parts per thousand, ppt)'] < 0) |
                                                      (df['Salinity (parts per thousand, ppt)'] > .03),
            "pH": (df['pH'] < 2) | (df['pH'] > 12),
            "Dissolved Oxygen (% saturation)": (df['Dissolved Oxygen (% saturation)'] < 10) | (
                    df['Dissolved Oxygen (% saturation)'] > 120),
            "Turbidity (NTU)": (df['Turbidity (NTU)'] < 0),
            "Turbidity (FNU)": (df['Turbidity (FNU)'] < 0),
            "fDOM (RFU)": (df['fDOM (RFU)'] < 0) | (df['fDOM (RFU)'] > 100),
            "fDOM (parts per billion QSU)": (df['fDOM (parts per billion QSU)'] < 0) | (
                    df['fDOM (parts per billion QSU)'] > 300),
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

    def update_hourly(selected_variables, source):
        p.title.text = f'Water Quality Parameters vs. Time'

        # Add Nan values so line chart will include gaps where no data is available
        # Convert the 'Timestamp' column to datetime if it's not already
        source.data['Timestamp'] = pd.to_datetime(source.data['Timestamp'])

        # Create a DataFrame from the ColumnDataSource data
        df_gap = pd.DataFrame(source.data)

        # Define the threshold for the gap (e.g., 2 hours)
        threshold = pd.Timedelta(hours=2)

        # List to store new rows
        new_rows = []

        # Iterate over the rows and check for gaps in the 'Timestamp' column
        for i in range(1, len(df_gap)):
            if df_gap['Timestamp'][i] - df_gap['Timestamp'][i - 1] > threshold:
                # Calculate the average 'Timestamp'
                avg_timestamp = df_gap['Timestamp'][i - 1] + (df_gap['Timestamp'][i] - df_gap['Timestamp'][i - 1]) / 2

                # Create a new row with 'NaN' values in all columns except 'Timestamp'
                new_row = {col: np.nan for col in df.columns}
                new_row['Timestamp'] = avg_timestamp

                # Append the new row to the list of new rows
                new_rows.append(new_row)

        # Convert the list of new rows to a DataFrame and append it to the original DataFrame
        new_rows_df = pd.DataFrame(new_rows)
        df_gap = pd.concat([df_gap, new_rows_df], ignore_index=True)

        # Sort the DataFrame by the 'Timestamp' column to maintain order
        df_gap = df_gap.sort_values(by='Timestamp').reset_index(drop=True)

        # Convert the updated DataFrame back to a ColumnDataSource
        source = ColumnDataSource(df_gap)

        for variable, color in zip(selected_variables, Category20_20):
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
        update_hourly(selected_variables, source)
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
        p.add_layout(p.legend[0], 'right')
        # p.legend.location = "top_left"
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
    depth_csv = "Profiler_modem_PFL_Step.csv"  # Replace with the actual file path
    df = pd.read_csv(depth_csv)

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
        selected_variables_p1 = st.multiselect('Select Water Quality Parameters', variables_to_plot_p1,
                                               default=["Temperature (Celsius)"])

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
            "Salinity (parts per thousand, ppt)": (df['Salinity (parts per thousand, ppt)'] < 0) |
                                                      (df['Salinity (parts per thousand, ppt)'] > .03),

            "pH": (df['pH'] < 1) | (df['pH'] > 13),
            "Dissolved Oxygen (% saturation)": (df['Dissolved Oxygen (% saturation)'] < 10) | (
                    df['Dissolved Oxygen (% saturation)'] > 120),
            "Turbidity (NTU)": (df['Turbidity (NTU)'] < 0),
            "Turbidity (FNU)": (df['Turbidity (FNU)'] < 0),
            "fDOM (RFU)": (df['fDOM (RFU)'] < 0) | (df['fDOM (RFU)'] > 100),
            "fDOM (parts per billion QSU)": (df['fDOM (parts per billion QSU)'] < 0) | (
                    df['fDOM (parts per billion QSU)'] > 300),
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

    if not selected_variables_p1 or not selected_depths:
        st.write("Please select at least one parameter and depth contour to plot.")
    else:
        # Group the data by 'Depth' and create separate ColumnDataSources for each group
        grouped_data = filtered_df.groupby('Depth')

        num_colors = (len(selected_depths))
        viridis_colors = Viridis256
        step = len(viridis_colors) // num_colors
        viridis_subset = viridis_colors[::step][:num_colors]
        viridis_subset = viridis_subset[::-1]
        line_styles = ['solid', 'dashed', 'dotdash', 'dotted']

        # Callback function for variable selection in the first plot
        def update_plot_p1(selected_variables_p1):
            # Group the data by 'Depth' and create separate ColumnDataSources for each group
            # grouped_data = df.groupby('Depth')

            p1.title.text = f'{", ".join(selected_variables_p1)} vs. Date for Different Depths'
            p1.renderers = []  # Remove existing renderers

            for i, (depth, group) in enumerate(grouped_data):
                depth_source = ColumnDataSource(group)

                # Convert the 'Timestamp' column to datetime if it's not already
                depth_source.data['Timestamp'] = pd.to_datetime(depth_source.data['Timestamp'])

                # Create a DataFrame from the ColumnDataSource data
                df = pd.DataFrame(depth_source.data)

                # Define the threshold for showing a gap between consecutive measurments
                threshold = pd.Timedelta(hours=15)

                # List to store new rows
                new_rows = []

                # Iterate over the rows and check for gaps in the 'Timestamp' column
                for k in range(1, len(df)):
                    if df['Timestamp'][k] - df['Timestamp'][k - 1] > threshold:
                        # Calculate the average 'Timestamp'
                        avg_timestamp = df['Timestamp'][k - 1] + (df['Timestamp'][k] - df['Timestamp'][k - 1]) / 2

                        # Create a new row with 'NaN' values in all columns except 'Timestamp'
                        new_row = {col: np.nan for col in df.columns}
                        new_row['Timestamp'] = avg_timestamp

                        # Append the new row to the list of new rows
                        new_rows.append(new_row)

                # Convert the list of new rows to a DataFrame and append it to the original DataFrame
                new_rows_df = pd.DataFrame(new_rows)
                df = pd.concat([df, new_rows_df], ignore_index=True)

                # Sort the DataFrame by the 'Timestamp' column to maintain order
                df = df.sort_values(by='Timestamp').reset_index(drop=True)

                # Convert the updated DataFrame back to a ColumnDataSource
                depth_source = ColumnDataSource(df)

                for j, var in enumerate(selected_variables_p1):
                    renderer = p1.line(x='Timestamp', y=var, source=depth_source, line_width=2,
                                       line_color=viridis_subset[i],
                                       legend_label=f'{depth}m: {var}', line_dash=line_styles[j])
                    p1.add_tools(HoverTool(renderers=[renderer],
                                           tooltips=[("Time", "@Timestamp{%Y-%m-%d %H:%M}"), ("Depth", f'{depth}'),
                                                     (var, f'@{{{var}}}')], formatters={"@Timestamp": "datetime", },
                                           mode="vline"))
                    p1.renderers.append(renderer)

        # Call the update_plot function with the selected variables for the first plot
        update_plot_p1(selected_variables_p1)

        # Show legend for the first plot
        p1.legend.title = 'Depth'
        # p1.legend.location = "top_left"
        p1.add_layout(p1.legend[0], 'right')
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
        h1, h2 = st.columns(2, gap='small')
        with h1:
            hover_labels = st.multiselect(label="Choose log parameters to display",
                                          options=["Select All"] + included_cols)

        if "Select All" in hover_labels:
            hover_labels = included_cols

        center_lat = (track['Lat'].max() + track['Lat'].min()) / 2
        center_lon = (track['Lon'].max() + track['Lon'].min()) / 2

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


def rename_ds(ds):
    # Dictionary of more descriptive names for known Delft3D output variables, with duplicate keys
    # for alternate capitalization and element vs. point conventions
    # (but there are likely more which are not described well in documentation)
    parameter_names = {
        "timestep": "Timestep (s)",
        "wgs84": "Projected coordinate system",
        "mesh2d_node_z": "z-coordinate of mesh nodes (m)",
        "mesh2d_face_x_bnd": "X-coordinate bounds of mesh faces (m)",
        "mesh2d_face_y_bnd": "Y-coordinate bounds of mesh faces (m)",
        "mesh2d_edge_type": "Edge type (relation between edge and flow geometry)",
        "mesh2d_flowelem_ba": "Cell area (m^2)",
        "mesh2d_flowelem_bl": "Bed level (m below mean surface elevation)",
        "mesh2d_s1": "Water level (m above mean surface elevation)",
        "mesh2d_waterdepth": "Water depth (m above bed level)",
        "mesh2d_u1": "Velocity at velocity point (m/s)",
        "mesh2d_ucx": "Velocity vector, x-component (m/s)",
        "mesh2d_ucy": "Velocity vector, y-component (m/s)",
        "mesh2d_ucz": "Velocity vector, z-component (m/s)",
        "mesh2d_ucmag": "Velocity magnitude (m/s)",
        "mesh2d_ucxa": "Velocity vector, depth-averaged x-component (m/s)",
        "mesh2d_ucya": "Velocity vector, depth-averaged y-component (m/s)",
        "mesh2d_ucmaga": "Velocity magnitude, depth-averaged (m/s)",
        "mesh2d_ww1": "Upward velocity component",
        "mesh2d_q1": "Discharge through flow link (m^3/s)",
        "mesh2d_sa1": "Salinity (ppt)",
        "mesh2d_tem1": "Temperature (C)",
        "mesh2d_windx": "Wind velocity vector, x-component (m/s)",
        "mesh2d_windy": "Wind velocity vector, y-component (m/s)",
        "mesh2d_windxu": "Edge wind velocity, x-component (m/s)",
        "mesh2d_windyu": "Edge wind velocity, y-component (m/s)",
        "mesh2d_station_id": "Station ID",
        "mesh2d_station_name": "Station Name",
        "mesh2d_station_x_coordinate": "Station x-coordinate (non-snapped)",
        "mesh2d_station_y_coordinate": "Station y-coordinate (non-snapped)",
        "mesh2d_zcoordinate_c": "Vertical coordinate, layer center",
        "mesh2d_zcoordinate_w": "Vertical coordinate, layer interface",
        "mesh2d_zcoordinate_wu": "Vertical coordinate, cell edge and layer interface",
        "mesh2d_waterlevel": "Water level (m above mean surface elevation)",
        "mesh2d_bedlevel": "Bed level (m below mean surface elevation)",
        "mesh2d_tausx": "x-components of mean bottom shear stress vector (Pa)",
        "mesh2d_tausy": "y-components of mean bottom shear stress vector (Pa)",
        "mesh2d_x_velocity": "x-components of layer velocity vector (m/s)",
        "mesh2d_y_velocity": "y-components of layer velocity vector (m/s)",
        "mesh2d_z_velocity": "z-components of depth-averaged velocity vector (m/s)",
        "mesh2d_depth-averaged_x_velocity": "x-components of depth-averaged velocity vector (m/s)",
        "mesh2d_depth-averaged_y_velocity": "y-components of depth-averaged velocity vector (m/s)",
        "mesh2d_tke": "Turbulent kinetic energy (m^2/s^2)",
        "mesh2d_vicww": "Turbulent vertical eddy viscosity (m^2/s)",
        "mesh2d_eps": "Turbulent energy dissipation (m^2/s^3)",
        "mesh2d_tau": "Turbulent time scale (1/s)",
        "mesh2d_rich": "Richardson number (%)",
        "mesh2d_salinity": "Salinity (ppt)",
        "mesh2d_velocity_magnitude": "Velocity magnitude (m/s)",
        "mesh2d_discharge_magnitude": "Average discharge (m^3/s)",
        "mesh2d_R": "Roller energy (J/m^2)",
        "mesh2d_hwav": "Significant wave height (m)",
        "mesh2d_twav": "Wave period (s)",
        "mesh2d_phiwav": "Wave length from direction (deg from N)",
        "mesh2d_rlabda": "Wave length (m)",
        "mesh2d_uorb": "Orbital velocity (m/s)",
        "mesh2d_vstokes": "y-component of Stokes drift (m/s)",
        "mesh2d_wtau": "Mean bed shear stress (Pa).",
        "mesh2d_temperature": "Temperature (◦C)",
        "mesh2d_wind": "Wind speed (m/s)",
        "mesh2d_Tair": "Air temperature (◦C)",
        "mesh2d_rhum": "Relative humidity (%)",
        "mesh2d_clou": "Cloudiness (%)",
        "mesh2d_Qsun": "Solar influx (W/m^2)",
        "mesh2d_Qeva": "Evaporative heat flux (W/m^2)",
        "mesh2d_Qcon": "Sensible heat flux (W/m^2)",
        "mesh2d_Qlong": "Long wave back radiation (W/m^2)",
        "mesh2d_Qfreva": "Free convection evaporative heat flux (W/m^2)",
        "mesh2d_Qfrcon": "Free convection sensible heat flux (W/m^2)",
        "mesh2d_Qtot": "Total heat flux (W/m^2)",
        "mesh2d_density": "Density (kg/m^2)",
        "mesh2d_seddif": "Sediment vertical diffusion (m^2/s)",
        "mesh2d_sed": "Sediment concentration (kg/m^3)",
        "mesh2d_ws": "Sediment settling velocity (m/s)",
        "mesh2d_taub": "Bed shear stress for morphology (Pa)",
        "mesh2d_sbcx": "x-component of current-related bedload transport (kg/s/m)",
        "mesh2d_sbcy": "y-component of current-related bedload transport (kg/s/m)",
        "mesh2d_sbwx": "x-component of wave-related bedload transport (kg/s/m)",
        "mesh2d_wbxy": "y-component of wave-related bedload transport (kg/s/m)",
        "mesh2d_sswx": "x-component of wave-related suspended transport (kg/s/m)",
        "mesh2d_sswy": "y-component of wave-related suspended transport (kg/s/m)",
        "mesh2d_sscx": "x-component of current-related suspended transport (kg/s/m)",
        "mesh2d_sscy": "y-component of current-related suspended transport (kg/s/m)",
        "mesh2d_sourse": "Source term suspended sediment transport (kg/m^3/s)",
        "mesh2d_sinkse": "Sink term suspended sediment transport (kg/m^3/s)",
        "mesh2d_bodsed": "Available sediment mass in bed (kg/m^2)",
        "mesh2d_dpsed": "Sediment thickness in bed (m)",
        "mesh2d_msed": "Available sediment mass in bed layer (kg/m^2)",
        "mesh2d_thlyr": "Thickness of bed layer (m)",
        "mesh2d_poros": "Porosity of bed layer (%)",
        "mesh2d_lyrfrac": "Volume fraction in bed layer (m)",
        "mesh2d_frac": "(Underlayer) IUn",
        "mesh2d_mudfrac": "Mud fraction in top layer (%)",
        "mesh2d_sandfrac": "Sand fraction in top layer (%)",
        "mesh2d_fixfac": "Reduction factor due to limited sediment thickness (%)",
        "mesh2d_hidexp": "Hiding and exposure factor (%)",
        "mesh2d_mfluff": "Sediment mass in fluff layer (%)",
        "mesh2d_sediment_concentration": "Sediment concentration (kg/m^3)",
        "mesh2d_patm": "Atmospheric pressure (N/m^2)",
        "mesh2d_rain": "Precipitation rate (mm/day)",
        "mesh2d_inflitration_cap": "Infiltration capacity (mm/hr)",
        "mesh2d_inflitration_actual": "Infiltration (mm/hr)",
        "mesh2d_Node_z": "z-coordinate of mesh nodes (m)",
        "mesh2d_Face_x_bnd": "X-coordinate bounds of mesh faces (m)",
        "mesh2d_Face_y_bnd": "Y-coordinate bounds of mesh faces (m)",
        "mesh2d_Edge_type": "Edge type (relation between edge and flow geometry)",
        "mesh2d_Flowelem_ba": "Cell area (m^2)",
        "mesh2d_Flowelem_bl": "Bed level (m below mean surface elevation)",
        "mesh2d_S1": "Water level (m above mean surface elevation)",
        "mesh2d_Waterdepth": "Water depth (m above bed level)",
        "mesh2d_U1": "Velocity at velocity point (m/s)",
        "mesh2d_Ucx": "Velocity vector, x-component (m/s)",
        "mesh2d_Ucy": "Velocity vector, y-component (m/s)",
        "mesh2d_Ucmag": "Velocity magnitude (m/s)",
        "mesh2d_Q1": "Discharge through flow link (m^3/s)",
        "mesh2d_Sa1": "Salinity (ppt)",
        "mesh2d_Tem1": "Temperature (C)",
        "mesh2d_Windx": "Wind velocity vector, x-component (m/s)",
        "mesh2d_Windy": "Wind velocity vector, y-component (m/s)",
        "mesh2d_Windxu": "Edge wind velocity, x-component (m/s)",
        "mesh2d_Windyu": "Edge wind velocity, y-component (m/s)",
        "mesh2d_Station_id": "Station ID",
        "mesh2d_Station_name": "Station Name",
        "mesh2d_Station_x_coordinate": "Station x-coordinate (non-snapped)",
        "mesh2d_Station_y_coordinate": "Station y-coordinate (non-snapped)",
        "mesh2d_Zcoordinate_c": "Vertical coordinate, layer center",
        "mesh2d_Zcoordinate_w": "Vertical coordinate, layer interface",
        "mesh2d_Zcoordinate_wu": "Vertical coordinate, cell edge and layer interface",
        "mesh2d_Waterlevel": "Water level (m above mean surface elevation)",
        "mesh2d_Bedlevel": "Bed level (m below mean surface elevation)",
        "mesh2d_Tausx": "x-components of mean bottom shear stress vector (Pa)",
        "mesh2d_Tausy": "y-components of mean bottom shear stress vector (Pa)",
        "mesh2d_X_velocity": "x-components of layer velocity vector (m/s)",
        "mesh2d_Y_velocity": "y-components of layer velocity vector (m/s)",
        "mesh2d_Z_velocity": "z-components of depth-averaged velocity vector (m/s)",
        "mesh2d_Depth-averaged_x_velocity": "x-components of depth-averaged velocity vector (m/s)",
        "mesh2d_Depth-averaged_y_velocity": "y-components of depth-averaged velocity vector (m/s)",
        "mesh2d_Tke": "Turbulent kinetic energy (m^2/s^2)",
        "mesh2d_Vicww": "Turbulent vertical eddy viscosity (m^2/s)",
        "mesh2d_Eps": "Turbulent energy dissipation (m^2/s^3)",
        "mesh2d_Tau": "Turbulent time scale (1/s)",
        "mesh2d_Rich": "Richardson number (%)",
        "mesh2d_Salinity": "Salinity (ppt)",
        "mesh2d_Velocity_magnitude": "Velocity magnitude (m/s)",
        "mesh2d_Discharge_magnitude": "Average discharge (m^3/s)",
        "mesh2d_Hwav": "Significant wave height (m)",
        "mesh2d_Twav": "Wave period (s)",
        "mesh2d_Phiwav": "Wave length from direction (deg from N)",
        "mesh2d_Rlabda": "Wave length (m)",
        "mesh2d_Uorb": "Orbital velocity (m/s)",
        "mesh2d_Vstokes": "y-component of Stokes drift (m/s)",
        "mesh2d_Wtau": "Mean bed shear stress (Pa).",
        "mesh2d_Temperature": "Temperature (◦C)",
        "mesh2d_Wind": "Wind speed (m/s)",
        "mesh2d_Rhum": "Relative humidity (%)",
        "mesh2d_Clou": "Cloudiness (%)",
        "mesh2d_Density": "Density (kg/m^2)",
        "mesh2d_Seddif": "Sediment vertical diffusion (m^2/s)",
        "mesh2d_Sed": "Sediment concentration (kg/m^3)",
        "mesh2d_Ws": "Sediment settling velocity (m/s)",
        "mesh2d_Taub": "Bed shear stress for morphology (Pa)",
        "mesh2d_Sbcx": "x-component of current-related bedload transport (kg/s/m)",
        "mesh2d_Sbcy": "y-component of current-related bedload transport (kg/s/m)",
        "mesh2d_Sbwx": "x-component of wave-related bedload transport (kg/s/m)",
        "mesh2d_Wbxy": "y-component of wave-related bedload transport (kg/s/m)",
        "mesh2d_Sswx": "x-component of wave-related suspended transport (kg/s/m)",
        "mesh2d_Sswy": "y-component of wave-related suspended transport (kg/s/m)",
        "mesh2d_Sscx": "x-component of current-related suspended transport (kg/s/m)",
        "mesh2d_Sscy": "y-component of current-related suspended transport (kg/s/m)",
        "mesh2d_Sourse": "Source term suspended sediment transport (kg/m^3/s)",
        "mesh2d_Sinkse": "Sink term suspended sediment transport (kg/m^3/s)",
        "mesh2d_Bodsed": "Available sediment mass in bed (kg/m^2)",
        "mesh2d_Dpsed": "Sediment thickness in bed (m)",
        "mesh2d_Msed": "Available sediment mass in bed layer (kg/m^2)",
        "mesh2d_Thlyr": "Thickness of bed layer (m)",
        "mesh2d_Poros": "Porosity of bed layer (%)",
        "mesh2d_Lyrfrac": "Volume fraction in bed layer (m)",
        "mesh2d_Frac": "(Underlayer) IUn",
        "mesh2d_Mudfrac": "Mud fraction in top layer (%)",
        "mesh2d_Sandfrac": "Sand fraction in top layer (%)",
        "mesh2d_Fixfac": "Reduction factor due to limited sediment thickness (%)",
        "mesh2d_Hidexp": "Hiding and exposure factor (%)",
        "mesh2d_Mfluff": "Sediment mass in fluff layer (%)",
        "mesh2d_Sediment_concentration": "Sediment concentration (kg/m^3)",
        "mesh2d_Patm": "Atmospheric pressure (N/m^2)",
        "mesh2d_Rain": "Precipitation rate (mm/day)",
        "mesh2d_Inflitration_cap": "Infiltration capacity (mm/hr)",
        "mesh2d_Inflitration_actual": "Infiltration (mm/hr)",
        "mesh2d_r": "Roller energy (J/m^2)",
        "mesh2d_tair": "Air temperature (◦C)",
        "mesh2d_qsun": "Solar influx (W/m^2)",
        "mesh2d_qeva": "Evaporative heat flux (W/m^2)",
        "mesh2d_qcon": "Sensible heat flux (W/m^2)",
        "mesh2d_qlong": "Long wave back radiation (W/m^2)",
        "mesh2d_qfreva": "Free convection evaporative heat flux (W/m^2)",
        "mesh2d_qfrcon": "Free convection sensible heat flux (W/m^2)",
        "mesh2d_qtot": "Total heat flux (W/m^2)",
        "mesh2d_source_sink_prescribed_discharge": "Prescribed discharge (m^3/s)",
        "mesh2d_source_sink_cumulative_volume": "Cumulative volume (m^3)",
        "mesh2d_source_sink_current_discharge": "Current discharge (m^3/s)",
        "mesh2d_source_sink_discharge_average": "Average discharge (m^3/s)",
        "mesh2d_source_sink_prescribed_salinity_increment": "Prescribed salinity (ppt)",
        "mesh2d_source_sink_prescribed_temperature_increment": "Prescribed temperature (◦C)",
        "node_z": "z-coordinate of mesh nodes (m)",
        "face_x_bnd": "X-coordinate bounds of mesh faces (m)",
        "face_y_bnd": "Y-coordinate bounds of mesh faces (m)",
        "edge_type": "Edge type (relation between edge and flow geometry)",
        "flowelem_ba": "Cell area (m^2)",
        "flowelem_bl": "Bed level (m below mean surface elevation)",
        "s1": "Water level (m above mean surface elevation)",
        "waterdepth": "Water depth (m above bed level)",
        "u1": "Velocity at velocity point (m/s)",
        "ucx": "Velocity vector, x-component (m/s)",
        "ucy": "Velocity vector, y-component (m/s)",
        "ucmag": "Velocity magnitude (m/s)",
        "ucxa": "Velocity vector, depth-averaged x-component (m/s)",
        "ucya": "Velocity vector, depth-averaged y-component (m/s)",
        "ucmaga": "Velocity magnitude, depth-averaged (m/s)",
        "q1": "Discharge through flow link (m^3/s)",
        "sa1": "Salinity (ppt)",
        "tem1": "Temperature (C)",
        "windx": "Wind velocity vector, x-component (m/s)",
        "windy": "Wind velocity vector, y-component (m/s)",
        "windxu": "Edge wind velocity, x-component (m/s)",
        "windyu": "Edge wind velocity, y-component (m/s)",
        "station_id": "Station ID",
        "station_name": "Station Name",
        "station_x_coordinate": "Station x-coordinate (non-snapped)",
        "station_y_coordinate": "Station y-coordinate (non-snapped)",
        "zcoordinate_c": "Vertical coordinate, layer center",
        "zcoordinate_w": "Vertical coordinate, layer interface",
        "zcoordinate_wu": "Vertical coordinate, cell edge and layer interface",
        "waterlevel": "Water level (m above mean surface elevation)",
        "bedlevel": "Bed level (m below mean surface elevation)",
        "tausx": "x-components of mean bottom shear stress vector (Pa)",
        "tausy": "y-components of mean bottom shear stress vector (Pa)",
        "x_velocity": "x-components of layer velocity vector (m/s)",
        "y_velocity": "y-components of layer velocity vector (m/s)",
        "z_velocity": "z-components of depth-averaged velocity vector (m/s)",
        "depth-averaged_x_velocity": "x-components of depth-averaged velocity vector (m/s)",
        "depth-averaged_y_velocity": "y-components of depth-averaged velocity vector (m/s)",
        "tke": "Turbulent kinetic energy (m^2/s^2)",
        "vicww": "Turbulent vertical eddy viscosity (m^2/s)",
        "eps": "Turbulent energy dissipation (m^2/s^3)",
        "tau": "Turbulent time scale (1/s)",
        "rich": "Richardson number (%)",
        "salinity": "Salinity (ppt)",
        "velocity_magnitude": "Velocity magnitude (m/s)",
        "discharge_magnitude": "Average discharge (m^3/s)",
        "R": "Roller energy (J/m^2)",
        "hwav": "Significant wave height (m)",
        "twav": "Wave period (s)",
        "phiwav": "Wave length from direction (deg from N)",
        "rlabda": "Wave length (m)",
        "uorb": "Orbital velocity (m/s)",
        "vstokes": "y-component of Stokes drift (m/s)",
        "wtau": "Mean bed shear stress (Pa).",
        "temperature": "Temperature (◦C)",
        "wind": "Wind speed (m/s)",
        "Tair": "Air temperature (◦C)",
        "rhum": "Relative humidity (%)",
        "clou": "Cloudiness (%)",
        "Qsun": "Solar influx (W/m^2)",
        "Qeva": "Evaporative heat flux (W/m^2)",
        "Qcon": "Sensible heat flux (W/m^2)",
        "Qlong": "Long wave back radiation (W/m^2)",
        "Qfreva": "Free convection evaporative heat flux (W/m^2)",
        "Qfrcon": "Free convection sensible heat flux (W/m^2)",
        "Qtot": "Total heat flux (W/m^2)",
        "density": "Density (kg/m^2)",
        "seddif": "Sediment vertical diffusion (m^2/s)",
        "sed": "Sediment concentration (kg/m^3)",
        "ws": "Sediment settling velocity (m/s)",
        "taub": "Bed shear stress for morphology (Pa)",
        "sbcx": "x-component of current-related bedload transport (kg/s/m)",
        "sbcy": "y-component of current-related bedload transport (kg/s/m)",
        "sbwx": "x-component of wave-related bedload transport (kg/s/m)",
        "wbxy": "y-component of wave-related bedload transport (kg/s/m)",
        "sswx": "x-component of wave-related suspended transport (kg/s/m)",
        "sswy": "y-component of wave-related suspended transport (kg/s/m)",
        "sscx": "x-component of current-related suspended transport (kg/s/m)",
        "sscy": "y-component of current-related suspended transport (kg/s/m)",
        "sourse": "Source term suspended sediment transport (kg/m^3/s)",
        "sinkse": "Sink term suspended sediment transport (kg/m^3/s)",
        "bodsed": "Available sediment mass in bed (kg/m^2)",
        "dpsed": "Sediment thickness in bed (m)",
        "msed": "Available sediment mass in bed layer (kg/m^2)",
        "thlyr": "Thickness of bed layer (m)",
        "poros": "Porosity of bed layer (%)",
        "lyrfrac": "Volume fraction in bed layer (m)",
        "frac": "(Underlayer) IUn",
        "mudfrac": "Mud fraction in top layer (%)",
        "sandfrac": "Sand fraction in top layer (%)",
        "fixfac": "Reduction factor due to limited sediment thickness (%)",
        "hidexp": "Hiding and exposure factor (%)",
        "mfluff": "Sediment mass in fluff layer (%)",
        "sediment_concentration": "Sediment concentration (kg/m^3)",
        "patm": "Atmospheric pressure (N/m^2)",
        "rain": "Precipitation rate (mm/day)",
        "inflitration_cap": "Infiltration capacity (mm/hr)",
        "inflitration_actual": "Infiltration (mm/hr)",
        "Timestep": "Latest computational timestep size in each output interval (s)",
        "Wgs84": "Projected coordinate system",
        "Node_z": "z-coordinate of mesh nodes (m)",
        "Face_x_bnd": "X-coordinate bounds of mesh faces (m)",
        "Face_y_bnd": "Y-coordinate bounds of mesh faces (m)",
        "Edge_type": "Edge type (relation between edge and flow geometry)",
        "Flowelem_ba": "Cell area (m^2)",
        "Flowelem_bl": "Bed level (m below mean surface elevation)",
        "S1": "Water level (m above mean surface elevation)",
        "Waterdepth": "Water depth (m above bed level)",
        "U1": "Velocity at velocity point (m/s)",
        "Ucx": "Velocity vector, x-component (m/s)",
        "Ucy": "Velocity vector, y-component (m/s)",
        "Ucmag": "Velocity magnitude (m/s)",
        "Q1": "Discharge through flow link (m^3/s)",
        "Sa1": "Salinity (ppt)",
        "Tem1": "Temperature (C)",
        "Windx": "Wind velocity vector, x-component (m/s)",
        "Windy": "Wind velocity vector, y-component (m/s)",
        "Windxu": "Edge wind velocity, x-component (m/s)",
        "Windyu": "Edge wind velocity, y-component (m/s)",
        "Station_id": "Station ID",
        "Station_name": "Station Name",
        "Station_x_coordinate": "Station x-coordinate (non-snapped)",
        "Station_y_coordinate": "Station y-coordinate (non-snapped)",
        "Zcoordinate_c": "Vertical coordinate, layer center",
        "Zcoordinate_w": "Vertical coordinate, layer interface",
        "Zcoordinate_wu": "Vertical coordinate, cell edge and layer interface",
        "Waterlevel": "Water level (m above mean surface elevation)",
        "Bedlevel": "Bed level (m below mean surface elevation)",
        "Tausx": "x-components of mean bottom shear stress vector (Pa)",
        "Tausy": "y-components of mean bottom shear stress vector (Pa)",
        "X_velocity": "x-components of layer velocity vector (m/s)",
        "Y_velocity": "y-components of layer velocity vector (m/s)",
        "Z_velocity": "z-components of depth-averaged velocity vector (m/s)",
        "Depth-averaged_x_velocity": "x-components of depth-averaged velocity vector (m/s)",
        "Depth-averaged_y_velocity": "y-components of depth-averaged velocity vector (m/s)",
        "Tke": "Turbulent kinetic energy (m^2/s^2)",
        "Vicww": "Turbulent vertical eddy viscosity (m^2/s)",
        "Eps": "Turbulent energy dissipation (m^2/s^3)",
        "Tau": "Turbulent time scale (1/s)",
        "Rich": "Richardson number (%)",
        "Salinity": "Salinity (ppt)",
        "Velocity_magnitude": "Velocity magnitude (m/s)",
        "Discharge_magnitude": "Average discharge (m^3/s)",
        "Hwav": "Significant wave height (m)",
        "Twav": "Wave period (s)",
        "Phiwav": "Wave length from direction (deg from N)",
        "Rlabda": "Wave length (m)",
        "Uorb": "Orbital velocity (m/s)",
        "Vstokes": "y-component of Stokes drift (m/s)",
        "Wtau": "Mean bed shear stress (Pa).",
        "Temperature": "Temperature (◦C)",
        "Wind": "Wind speed (m/s)",
        "Rhum": "Relative humidity (%)",
        "Clou": "Cloudiness (%)",
        "Density": "Density (kg/m^2)",
        "Seddif": "Sediment vertical diffusion (m^2/s)",
        "Sed": "Sediment concentration (kg/m^3)",
        "Ws": "Sediment settling velocity (m/s)",
        "Taub": "Bed shear stress for morphology (Pa)",
        "Sbcx": "x-component of current-related bedload transport (kg/s/m)",
        "Sbcy": "y-component of current-related bedload transport (kg/s/m)",
        "Sbwx": "x-component of wave-related bedload transport (kg/s/m)",
        "Wbxy": "y-component of wave-related bedload transport (kg/s/m)",
        "Sswx": "x-component of wave-related suspended transport (kg/s/m)",
        "Sswy": "y-component of wave-related suspended transport (kg/s/m)",
        "Sscx": "x-component of current-related suspended transport (kg/s/m)",
        "Sscy": "y-component of current-related suspended transport (kg/s/m)",
        "Sourse": "Source term suspended sediment transport (kg/m^3/s)",
        "Sinkse": "Sink term suspended sediment transport (kg/m^3/s)",
        "Bodsed": "Available sediment mass in bed (kg/m^2)",
        "Dpsed": "Sediment thickness in bed (m)",
        "Msed": "Available sediment mass in bed layer (kg/m^2)",
        "Thlyr": "Thickness of bed layer (m)",
        "Poros": "Porosity of bed layer (%)",
        "Lyrfrac": "Volume fraction in bed layer (m)",
        "Frac": "(Underlayer) IUn",
        "Mudfrac": "Mud fraction in top layer (%)",
        "Sandfrac": "Sand fraction in top layer (%)",
        "Fixfac": "Reduction factor due to limited sediment thickness (%)",
        "Hidexp": "Hiding and exposure factor (%)",
        "Mfluff": "Sediment mass in fluff layer (%)",
        "Sediment_concentration": "Sediment concentration (kg/m^3)",
        "Patm": "Atmospheric pressure (N/m^2)",
        "Rain": "Precipitation rate (mm/day)",
        "Inflitration_cap": "Infiltration capacity (mm/hr)",
        "Inflitration_actual": "Infiltration (mm/hr)",
        "r": "Roller energy (J/m^2)",
        "tair": "Air temperature (◦C)",
        "qsun": "Solar influx (W/m^2)",
        "qeva": "Evaporative heat flux (W/m^2)",
        "qcon": "Sensible heat flux (W/m^2)",
        "qlong": "Long wave back radiation (W/m^2)",
        "qfreva": "Free convection evaporative heat flux (W/m^2)",
        "qfrcon": "Free convection sensible heat flux (W/m^2)",
        "qtot": "Total heat flux (W/m^2)",
        "source_sink_prescribed_discharge": "Prescribed discharge (m^3/s)",
        "source_sink_cumulative_volume": "Cumulative volume (m^3)",
        "source_sink_current_discharge": "Current discharge (m^3/s)",
        "source_sink_discharge_average": "Average discharge (m^3 /s)",
        "source_sink_prescribed_salinity_increment": "Prescribed salinity (ppt)",
        "source_sink_prescribed_temperature_increment": "Prescribed temperature (◦C)"
    }
    filtered_dict = {k: v for k, v in parameter_names.items() if k in ds.data_vars}
    new = ds.rename(name_dict=filtered_dict)
    return new


def display_map(o_file):
    # Open and merge mapfile with xugrid(xarray) and print netcdf structure
    uds_map_o = dfmt.open_partitioned_dataset(o_file)
    uds_map = rename_ds(uds_map_o)
    print("uds_map.attrs", uds_map.attrs)

    ###################################################################################################################
    # 1. Analyze the file contents. Create references for plotting.
    # print(uds_map)
    # Set coordinate reference system for background map
    crs = 'EPSG:4326'

    # Get extents of map from attributes, for setting plot limits
    try:
        xmin_abs = uds_map.attrs['geospatial_lon_min']
    except:
        xmin_abs = 6.387478790667275

    try:
        xmax_abs = uds_map.attrs['geospatial_lon_max']
    except:
        xmax_abs = 6.56781074840919

    try:
        ymin_abs = uds_map.attrs['geospatial_lat_min']
    except:
        ymin_abs = 62.46442857883768

    try:
        ymax_abs = uds_map.attrs['geospatial_lat_max']
    except:
        ymax_abs = 62.48487637586751

    # Create lists of data variables based on their dimensionality
    includes_coordinate = "mesh2d_nFaces"
    excludes_coordinates = ["mesh2d_nEdges", "mesh2d_nNodes", "mesh2d_nMax_face_nodes"]
    mesh2d_nFaces_list = []
    for name, var in uds_map.data_vars.items():
        if (includes_coordinate in var.dims and all(coord not in var.dims for coord in excludes_coordinates)):
            mesh2d_nFaces_list.append(name)

    includes_coordinate = "mesh2d_nEdges"
    excludes_coordinates = ["mesh2d_nFaces", "mesh2d_nNodes", "mesh2d_nMax_face_nodes"]
    mesh2d_nEdges_list = []
    for name, var in uds_map.data_vars.items():
        if (includes_coordinate in var.dims and all(coord not in var.dims for coord in excludes_coordinates)):
            mesh2d_nEdges_list.append(name)

    includes_coordinate = "mesh2d_nNodes"
    excludes_coordinates = ["mesh2d_nFaces", "mesh2d_nEdges", "mesh2d_nMax_face_nodes"]
    mesh2d_nNodes_list = []
    for name, var in uds_map.data_vars.items():
        if (includes_coordinate in var.dims and all(coord not in var.dims for coord in excludes_coordinates)):
            mesh2d_nNodes_list.append(name)

    includes_coordinate = "mesh2d_nMax_face_nodes"
    # excludes_coordinates = ["mesh2d_nFaces", "mesh2d_nEdges", "mesh2d_nNodes"]
    mesh2d_nMax_face_nodes_list = []
    for name, var in uds_map.data_vars.items():
        if (includes_coordinate in var.dims and all(coord not in var.dims for coord in excludes_coordinates)):
            mesh2d_nMax_face_nodes_list.append(name)

    includes_coordinate = "mesh2d_nLayers"
    # excludes_coordinates = ["mesh2d_nFaces", "mesh2d_nEdges", "mesh2d_nNodes"]
    mesh2d_nLayers_list = []
    for name, var in uds_map.data_vars.items():
        if (includes_coordinate in var.dims and all(coord not in var.dims for coord in excludes_coordinates)):
            mesh2d_nLayers_list.append(name)

    includes_coordinate = "mesh2d_nInterfaces"
    # excludes_coordinates = ["mesh2d_nFaces", "mesh2d_nEdges", "mesh2d_nNodes"]
    mesh2d_nInterfaces_list = []
    for name, var in uds_map.data_vars.items():
        if (includes_coordinate in var.dims and all(coord not in var.dims for coord in excludes_coordinates)):
            mesh2d_nInterfaces_list.append(name)

    # Handle the possibility that some dimensions or coordinates are not present in the dataset
    if 'mesh2d_nLayers' in uds_map.dims:
        num_layers = uds_map.sizes['mesh2d_nLayers']
    else:
        num_layers = None
    if 'time' in uds_map.dims:
        num_times = uds_map.sizes['time']
        # Extract and reformat a list of all time steps, and create a dictionary for converting the calendar time to an index
        times = uds_map.coords["time"]
        formatted_times = times.dt.strftime("%Y-%m-%d %H:%M:%S")
        formatted_times = formatted_times.values.tolist()
        times_dict = {value: i for i, value in enumerate(formatted_times)}
    else:
        num_times = None

    # Option to display dimensionality and contents of the file
    on_off = ["Display file attributes", "Hide"]
    print_attrs = st.radio(label='', options=on_off, horizontal=True, index=1)
    if print_attrs == on_off[0]:
        a1, a2, a3 = st.columns(3, gap='small')
        with a1:
            st.markdown("#### Dimensions")
            st.write("Faces (elements):", uds_map.sizes['mesh2d_nFaces'])
            st.write("Edges (links):", uds_map.sizes['mesh2d_nEdges'])
            st.write("Nodes:", uds_map.sizes['mesh2d_nNodes'])
            st.write("Layers:", num_layers)
            st.write("Timesteps:", num_times)
        with a2:
            st.markdown("#### Data variables")
            st.write("Data variables per face", mesh2d_nFaces_list)
            st.write("Data variables per edge", mesh2d_nEdges_list)
            st.write("Data variables per node", mesh2d_nNodes_list)
            st.write("Data variables per mesh2d_nMax_face_nodes", mesh2d_nMax_face_nodes_list)
            st.write("Data variables per mesh2d_nLayers", mesh2d_nLayers_list)
            st.write("Data variables per mesh2d_nInterfaces", mesh2d_nInterfaces_list)
        with a3:
            st.markdown("#### Timing")
            st.write('Start:', uds_map.attrs["time_coverage_start"])
            st.write('End:', uds_map.attrs["time_coverage_end"])
            st.write('Duration:', uds_map.attrs["time_coverage_duration"])
            st.write("Time resolution:", uds_map.attrs['time_coverage_resolution'])
            st.write("Run date:", uds_map.attrs['date_modified'])
    else:
        st.empty()

    # Some useful commands for looking at what is in each dimension/coordinate/data variable:
    # st.write('uds_map.data_vars', uds_map.coords['mesh2d_layer_z'].values)
    # my_variable_values = uds_map.coords['mesh2d_layer_z'].values
    # df = pd.DataFrame({'mesh2d_layer_z': my_variable_values})
    # st.write(df)
    #
    # # Sort the DataFrame by the data variable values
    # sorted_df = df.sort_values(by='mesh2d_layer_z')
    # st.write(sorted_df)
    #
    # csv_filename = 'sorted_values.csv'
    # sorted_df.to_csv(csv_filename, index=False)
    # print("Wrote mesh2d_layer_z values to csv")

    ###################################################################################################################
    # 2. Plot the distribution of map parameters in the x,y plane using selections based on references in #1 above.
    fig_surface, ax = plt.subplots(figsize=(20, 5))

    cc1, cc2 = st.columns(2, gap="small")
    with cc1:
        # Convert the list of faces to more descriptive parameter names for display in the UI selection window
        # mesh2d_nFaces_list = ([value for key, value in parameter_names.items() if key in mesh2d_nFaces_list] +
        #                       [item for item in mesh2d_nFaces_list if item not in parameter_names.keys()])

        parameter = st.selectbox("Select parameter to display", mesh2d_nFaces_list)
        vmin = np.nanmin(uds_map.data_vars[parameter].values)
        vmax = np.nanmax(uds_map.data_vars[parameter].values)

        # Check if the parameter is defined for multiple timesteps and ask for selection if so
        if num_times is not None:
            if 'time' in uds_map[parameter].dims:
                concise = uds_map[parameter].dropna(dim='time', how='all')
                num_times = concise.sizes['time']
                selected_time_key = st.selectbox("Select the time to display", list(times_dict.keys()))
                selected_time_index = times_dict.get(selected_time_key)
            else:
                num_times = None

        # Check if the parameter is defined for multiple layers and ask for selection if so
        if num_layers is not None:
            if 'mesh2d_nLayers' in uds_map[parameter].dims:
                concise = uds_map[parameter].dropna(dim='mesh2d_nLayers', how='all')
                num_layers = concise.sizes['mesh2d_nLayers']
                layer_list = list(reversed(range(0, num_layers)))
                # st.write('uds_map.coords:', uds_map.coords)
                # st.write('uds_map.dims:', uds_map.dims)
                if 'mesh2d_layer_sigma_z' in uds_map.coords:
                    # Superseded logic preserved here as example:
                    # depths = uds_map.coords['mesh2d_layer_sigma_z'].values
                    # layer_depths = depths + abs(depths[1] - depths[0])
                    # layer_depths = layer_depths[::-1].tolist()
                    # st.write('depths from mixed mesh2d_layer_sigma_z', depths)

                    depths = uds_map.coords['mesh2d_layer_sigma_z'].values
                    layer_depths = depths[::-1].tolist()
                elif 'mesh2d_layer_z' in uds_map.coords or 'mesh2d_layer_sigma_z' in uds_map.coords:
                    depths = uds_map.coords['mesh2d_layer_z'].values
                    layer_depths = depths[::-1].tolist()
                elif 'mesh2d_layer_sigma' in uds_map.coords:
                    depths = uds_map.coords['mesh2d_layer_sigma'].values
                    layer_depths = depths[::-1].tolist()
                else:
                    st.write('Contingency scenario: none of mesh2d_layer_sigma_z, mesh2d_layer_sigma_, '
                             'mesh2d_layer_z in uds_map.coords')
                    # Handle sigma or mixed layers by creating an approximation of the depths
                    max_depth = uds_map['z-coordinate of mesh nodes (m)'].max().to_numpy()[()]
                    # print("max_depth data: type, size, shape:", type(max_depth), max_depth.size, max_depth.shape, max_depth)
                    layer_depths = np.round(np.linspace(0, max_depth - max_depth / num_layers, num_layers))
                    layer_depths = layer_depths.tolist()
                    # print("in depth ", max_depth, " from ", layer_depths, " selected ", depth_selected, " indicating layer ", layer+1)
                depth_selected = st.selectbox("Select layer to display, by depth below mean surface elevation (m)",
                                              layer_depths)
                layer = layer_list[layer_depths.index(depth_selected)]
            else:
                st.write('Map file is not 3D (single layer, mesh2d_nLayers is NOT in uds_map[parameter].dims)')
                num_layers = None

    # Set plot limits. Location is intended to preserve option to add slider for zoom functionality later with 'scaler'.
    # aspect = st.slider("Zoom", min_value=-0.5, max_value=1.0, value=0.01)
    scaler = 1.0
    aspect = 0.02
    xavg = (xmax_abs + xmin_abs) / 2
    yavg = (ymax_abs + ymin_abs) / 2
    x_int = (xmax_abs - xmin_abs) / 2
    y_int = (ymax_abs - ymin_abs) / 2
    xmin = xavg - x_int * (1 + scaler * aspect)
    xmax = xavg + x_int * (1 + scaler * aspect)
    ymin = yavg - y_int * (1 + scaler)
    ymax = yavg + y_int * (1 + scaler)

    # Add a slider for selecting where to take a cross-section of the simulated water body
    line_array = None
    if num_layers is not None:
        latlon = st.radio("Choose the orientation of the cross section", options=("Longitude", "Latitude"),
                          horizontal=True)
        if latlon == "Longitude":
            cross_section = st.slider("Select the longitude of the cross section for depth view", min_value=xmin_abs,
                                      max_value=xmax_abs, value=(xmin_abs + xmax_abs) / 2, step=.001, format="%.3f")
            ax.axvline(cross_section, color='red')
            line_array = np.array([[cross_section, ymin_abs],
                                   [cross_section, ymax_abs]])
        else:
            cross_section = st.slider("Select the latitude of the cross section for depth view", min_value=ymin_abs,
                                      max_value=ymax_abs, value=(ymin_abs + ymax_abs) / 2, step=.001, format="%.3f")
            ax.axhline(cross_section, color='red')
            line_array = np.array([[xmin_abs, cross_section],
                                   [xmax_abs, cross_section]])

    # Set up plot depending on dimensionality of parameter: is it defined at multiple timesteps or layers
    if num_layers is None:
        if num_times is None:
            pc = uds_map[parameter].isel(missing_dims='ignore').ugrid.plot(cmap='jet', add_colorbar=False,
                                                                           vmin=vmin, vmax=vmax)
            st.markdown(f"### {parameter}")
        else:
            pc = uds_map[parameter].isel(time=selected_time_index, missing_dims='ignore').ugrid.plot(cmap='jet',
                                                                                                     add_colorbar=False,
                                                                                                     vmin=vmin,
                                                                                                     vmax=vmax)
            st.markdown(f"### {parameter} at {selected_time_key}")
    else:
        if num_times is None:
            pc = uds_map[parameter].isel(mesh2d_nLayers=layer, missing_dims='ignore').ugrid.plot(cmap='jet',
                                                                                                 add_colorbar=False,
                                                                                                 vmin=vmin,
                                                                                                 vmax=vmax)
            st.markdown(f"### {parameter} at depth of {depth_selected} m")
        else:
            pc = uds_map[parameter].isel(time=selected_time_index, mesh2d_nLayers=layer,
                                         missing_dims='ignore').ugrid.plot(cmap='jet', add_colorbar=False,
                                                                           vmin=vmin, vmax=vmax)
            st.markdown(f"### {parameter} at depth of {depth_selected} m at {selected_time_key}")

    if crs is None:
        ax.set_aspect('equal')
    else:
        ctx.add_basemap(ax=ax, source=ctx.providers.OpenTopoMap, crs=crs, attribution=False)
    # fig_surface.suptitle(parameter)  # Add a title within the figure's limits

    # Add a colorbar to show the values of the colors
    fraction = 0.01  # Percentage of the figure's width given to the colorbar (scale/legend)
    colorbar = plt.colorbar(pc, orientation="vertical", fraction=fraction, pad=0.001)
    colorbar.set_label(parameter)  # Set colorbar label

    ax.set_aspect('equal')
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_title("")
    ax.set_position([0, 0, 1, 1])
    plt.tight_layout()
    st.pyplot(fig_surface)

    ###################################################################################################################
    # 3. Plot x- or y- vs. z cross-section of the selections in #2 above
    if line_array is not None:
        # dfmt functions expect the original data variable names, so use the dataset as originally imported
        uds_crs = dfmt.polyline_mapslice(uds_map_o.isel(time=selected_time_index), line_array)
        uds_crs = rename_ds(uds_crs)  # then rename once imported
        # uda = uds_crs[parameter]
        # fig_1, ax_1 = plt.subplots(figsize=(12, 5))
        # st.write(uda.coords)
        # st.write(uda.dims)
        # st.write(uda['mesh2d_layer_sigma_z'].values)
        # uda.ugrid.plot(cmap='jet')
        # st.pyplot(fig_1)

        # Experimental code to plot data from a dataframe
        # # Initialize an empty DataFrame
        # df_rewrite = pd.DataFrame()
        #
        # # Extract data along coordinates x, y, z
        # for x in uds_crs[parameter].values:
        #     for y in ds.coords['y'].values:
        #         for z in ds.coords['z'].values:
        #             row = {'x': x, 'y': y, 'z': z, 'data': ds.sel(x=x, y=y, z=z).data.item()}
        #             df_rewrite = df_rewrite.append(row, ignore_index=True)


        fig_cross, ax = plt.subplots(figsize=(20, 5))
        # st.write('uds_crs[parameter].coords', uds_crs[parameter].coords)
        # st.write('uds_crs[parameter].values', uds_crs[parameter].values)
        #
        # st.write('uds_crs[parameter].coords[mesh2d_layer_sigma_z]', uds_crs[parameter].coords['mesh2d_layer_sigma_z'])
        # st.write('uds_crs[parameter].coords[mesh2d_layer_sigma_z].values',
        #          uds_crs[parameter].coords['mesh2d_layer_sigma_z'].values)
        cross = uds_crs[parameter].ugrid.plot(cmap='jet', add_colorbar=False, vmin=vmin, vmax=vmax)
        x_coords = uds_crs[parameter].coords['mesh2d_face_x'].values
        y_coords = uds_crs[parameter].coords['mesh2d_face_y'].values
        # Calculate the range for x and y data
        x_range = max(x_coords) - min(x_coords)
        # y_range = max(y_coords) - min(y_coords)


        # Calculate the 1% extra for the limits
        x_limit = x_range * 0.01
        # y_limit = y_range * 0.01

        # Set the limits for x and y axis on the 'ax' object
        # st.write(min(x_coords), max(x_coords))
        # ax.set_xlim(min(x_coords), max(x_coords))
        # ax.set_ylim(min(y_coords) - y_limit, max(y_coords) + y_limit)

        # Plot your data on the 'ax' object
        # ax.plot(data_x, data_y)
        if latlon == "Longitude":
            ax.set_xlabel(f"Position, m north of latitude {ymin_abs}")
        else:
            ax.set_xlabel(f"Position, m east of longitude {xmin_abs}")
        ax.set_ylabel("Depth, m")
        ax.set_title("")
        # fig_surface.suptitle(parameter)
        colorbar = plt.colorbar(cross, orientation="vertical", fraction=fraction, pad=0.001)
        # Set colorbar label
        colorbar.set_label(parameter)

        st.markdown(f"### Cross-section of {parameter} at {latlon} = {cross_section}, {selected_time_key}")
        ax.set_position([0, 0, 1, 1])
        plt.tight_layout()
        st.pyplot(fig_cross)


def display_error(ds_his, feature, column_name, errorplot, errorplots, location, offline):
    depths = np.unique(ds_his.coords['zcoordinate_c'].values)
    # interval = abs(depths[1] - depths[0]) / 2
    ds_his["zcoordinate_c"] = ds_his["zcoordinate_c"].round(2)  # Use cell-center depth, not 1.25m 'interval' correction
    data_for_bokeh = ds_his[feature].sel(stations=location)
    his_df = data_for_bokeh.to_dataframe()
    his_df.reset_index()
    his_df = his_df.dropna(subset=[feature])  # Drop rows for layers with no value for selected feature
    his_df = his_df.reset_index()

    if errorplot == errorplots[0] and not offline:
        df = upload_hourly_csv_page()

        # Check if there are at least two columns in the DataFrame
        if len(df.columns) < 2:
            st.warning("DataFrame must have at least two columns for X and Y variables.")
            return None

        # Define conditions for each parameter which indicate errors in the data
        error_conditions = {
            "Timestamp": (df['Timestamp'] < pd.to_datetime('2000-01-01')) | (
                    df['Timestamp'] > pd.to_datetime('2099-12-31')),
            "Temperature (Celsius)": (df['Temperature (Celsius)'] < 1) | (df['Temperature (Celsius)'] > 25),
            "Conductivity (microSiemens/centimeter)": (df['Conductivity (microSiemens/centimeter)'] < 0) |
                                                      (df['Conductivity (microSiemens/centimeter)'] > 45),
            "Specific Conductivity (microSiemens/centimeter)": (
                    df['Specific Conductivity (microSiemens/centimeter)'] < 1),
            "Salinity (parts per thousand, ppt)": (df['Salinity (parts per thousand, ppt)'] < 0) |
                                                  (df['Salinity (parts per thousand, ppt)'] > .03),
            "pH": (df['pH'] < 2) | (df['pH'] > 12),
            "Dissolved Oxygen (% saturation)": (df['Dissolved Oxygen (% saturation)'] < 10) | (
                    df['Dissolved Oxygen (% saturation)'] > 120),
            "Turbidity (NTU)": (df['Turbidity (NTU)'] < 0),
            "Turbidity (FNU)": (df['Turbidity (FNU)'] < 0),
            "fDOM (RFU)": (df['fDOM (RFU)'] < 0) | (df['fDOM (RFU)'] > 100),
            "fDOM (parts per billion QSU)": (df['fDOM (parts per billion QSU)'] < 0) | (
                    df['fDOM (parts per billion QSU)'] > 300),
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

        # Since temperature gradients are large at the surface, interpolate the model value at the sensor's depth
        # Set the target x value where we want to interpolate y

        # Ensure that x and y columns are numeric
        his_df['zcoordinate_c'] = pd.to_numeric(his_df['zcoordinate_c'], errors='coerce')
        his_df[feature] = pd.to_numeric(his_df[feature], errors='coerce')

        # Function to interpolate y for each group at target_x
        def interpolate_at_target_x(group, target_x):
            # Sort the group by x values
            group_sorted = group.sort_values('zcoordinate_c')
            # Interpolate the y value at target_x
            interpolated_y = np.interp(target_x, group_sorted['zcoordinate_c'], group_sorted[feature])
            # Create a new row with the interpolated value
            new_row = {'time': group.name, 'zcoordinate_c': reference, feature: interpolated_y}
            return new_row

        # Apply the interpolation for each group and collect new rows
        reference = -2.95  # Average depth at which hourly data is measured
        new_rows = his_df.groupby('time').apply(lambda group: interpolate_at_target_x(group, reference)).tolist()

        # Create a new DataFrame from the new rows
        his_df = pd.DataFrame(new_rows)

        # Sort the column for efficient comparison
        df_sorted = his_df.sort_values(by='zcoordinate_c')

        # Drop all rows not from the layer most closely matching the sensor's depth
        surface_depth = df_sorted.loc[
            df_sorted.index[bisect.bisect_left(df_sorted['zcoordinate_c'], reference)], 'zcoordinate_c']
        surfacehis_df = his_df[his_df['zcoordinate_c'] == surface_depth]
        surfacehis_df = surfacehis_df.reset_index()
        df = df.set_index('Timestamp')  # Set the index for interpolation

        def interpolate_times(df, target_df, target_column, column_name):
            """
            Interpolates the value of a specified column for every datetime in a target column of a different DataFrame.

            Parameters:
            - df (pd.DataFrame): DataFrame with a datetime index where interpolation is performed.
            - target_df (pd.DataFrame): DataFrame containing the target datetimes.
            - target_column (str): The column in target_df containing the target datetimes.
            - column_name (str): The column in df for which to interpolate the values.

            Returns:
            - result_df (pd.DataFrame): DataFrame with the target datetimes and the interpolated values.
            """
            # Ensure the index of df is a DatetimeIndex
            if not isinstance(df.index, pd.DatetimeIndex):
                raise ValueError("The DataFrame df index must be a DatetimeIndex.")

            # Drop duplicate indices
            df = df[~df.index.duplicated(keep='first')]

            # Ensure df is sorted by the datetime index
            df = df.sort_index()

            # Ensure target_column exists in target_df
            if target_column not in target_df.columns:
                raise ValueError(f"The target column {target_column} does not exist in the target DataFrame.")

            # Convert target_column to Timestamps if not already
            target_df[target_column] = pd.to_datetime(target_df[target_column])

            # Ensure the target datetimes are within the range of df index
            if (target_df[target_column] < df.index.min()).any() or (target_df[target_column] > df.index.max()).any():
                raise ValueError("One or more target datetimes are outside the range of the DataFrame index.")

            # Reindex df to include the target datetimes and interpolate
            all_datetimes = df.index.union(target_df[target_column])
            df_interpolated = df.reindex(all_datetimes).interpolate(method='time')

            # Extract the interpolated values for the target datetimes
            interpolated_values = df_interpolated.loc[target_df[target_column], column_name].values

            # Create the result DataFrame
            result_df = target_df.copy()
            result_df[f'Reference {column_name}'] = interpolated_values
            result_df[feature].rename(f'Model {feature}')
            result_df = result_df.rename(columns={feature: f'Model {feature}'})
            return result_df

        result = interpolate_times(df, surfacehis_df, 'time', column_name)

        # Compute some vector statistics comparing the model and reference values
        error_signals = [f'Model {feature}', f'Reference {column_name}', 'Difference', 'Absolute Error',
                         'Squared Error', 'Percent Error']
        result[error_signals[2]] = result[error_signals[1]] - result[error_signals[0]]
        result[error_signals[3]] = abs(result[error_signals[2]])
        result[error_signals[4]] = result[error_signals[2]] ** 2
        result[error_signals[5]] = 100 * result[error_signals[2]] / result[error_signals[1]]

        # Compute summary (scalar) statistics
        MM = result[error_signals[0]].mean()
        RM = result[error_signals[1]].mean()
        MSTDEV = result[error_signals[0]].std()
        RSTDEV = result[error_signals[1]].std()
        correlation = result[f'Model {feature}'].corr(result[f'Reference {column_name}'])
        # covariance = result[f'Model {feature}'].cov(result[f'Reference {column_name}'])
        SSE = result[error_signals[2]].sum()
        MAE = result[error_signals[3]].sum() / len(result[error_signals[3]])
        MSE = result[error_signals[4]].sum() / len(result[error_signals[4]])
        RMSE = math.sqrt(MSE)
        MPE = result[error_signals[5]].sum() / len(result[error_signals[5]])
        statvalues1 = {'Statistic': ['Mean', 'Standard Deviation'],
                       'Model': [MM, MSTDEV],
                       'Reference Data': [RM, RSTDEV]}

        statvalues2 = {'Statistic': ['Correlation',
                                     "Sum of Squares Error", "Mean Absolute Error", "Mean Squared Error",
                                     "Root Mean Squared Error", 'Mean Percent Error'],
                       'Comparison': [correlation, SSE, MAE, MSE, RMSE, MPE]}

        stats_df1 = pd.DataFrame(statvalues1)
        stats_df1 = stats_df1.reset_index(drop=True)

        stats_df2 = pd.DataFrame(statvalues2)
        stats_df2 = stats_df2.reset_index(drop=True)

        c1, c2 = st.columns(2, gap='small')
        with c1:
            error_stats = st.multiselect("Choose which error statistics to plot", error_signals,
                                         default=error_signals[2])

        def update_p_err(p_err, df, error_type, palette):
            p_err.renderers = []  # Remove existing renderers
            err_source = ColumnDataSource(df)
            for i, (error) in enumerate(error_type):
                renderer = p_err.line(x='time', y=error_type[i], source=err_source, line_width=2,
                                      line_color=palette[i],
                                      legend_label=error)
                p_err.add_tools(HoverTool(renderers=[renderer],
                                          tooltips=[("Time", "@time{%Y-%m-%d %H:%M}"),
                                                    (error_type[i], f'@{{{error}}}')],
                                          formatters={"@time": "datetime", },
                                          mode="vline"))
                p_err.renderers.append(renderer)

        # Call the update_plot function with the selected variables for the first plot
        p_err = figure(title=f'Error in modeled {feature} at 2.95m depth at Profiler Station')
        if len(error_stats) < 1 or error_stats is None:
            st.warning("Please select at least one value to plot.")
        else:
            num_colors = len(error_stats)
            viridis_colors = Viridis256
            step = len(viridis_colors) // num_colors
            viridis_subset = viridis_colors[::step][:num_colors]
            update_p_err(p_err, result, error_stats, viridis_subset)
            p_err.add_layout(p_err.legend[0], 'right')
            p_err.legend.title = 'Series'
            p_err.legend.location = "top_left"
            p_err.legend.label_text_font_size = '10px'
            p_err.legend.click_policy = "hide"  # Hide lines on legend click
            p_err.yaxis.axis_label = "Values"
            p_err.xaxis.axis_label = "Time"
            p_err.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%y/%m/%d %H:%M")
            st.bokeh_chart(p_err, use_container_width=True)  # Display the Bokeh chart using Streamlit
            st.write("Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
                     "Click legend entries to toggle series on/off.")

        st.write(f"Time-averaged summary statistics for error in modeled {feature} at surface")
        c1, c2, c3, c4 = st.columns(4, gap='small')
        if not offline:
            with c1:
                st.markdown("##### Series Statistics")
                st.dataframe(stats_df1, hide_index=True)
            with c2:
                st.markdown("##### Comparative Statistics")
                st.dataframe(stats_df2, hide_index=True)
    elif errorplot == errorplots[1]:
        # Compute and plot error at all depths

        # Import and pre-process vertical profile data
        depth_csv = "Profiler_modem_PFL_Step.csv"  # Replace with the actual file path
        df = pd.read_csv(depth_csv)

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

        # Define conditions for each parameter which indicate errors in the data
        error_conditions = {
            "Timestamp": (df['Timestamp'] < pd.to_datetime('2000-01-01')) | (
                    df['Timestamp'] > pd.to_datetime('2099-12-31')),
            "Temperature (Celsius)": (df['Temperature (Celsius)'] < 1) | (df['Temperature (Celsius)'] > 25),
            "Conductivity (microSiemens/centimeter)": (df['Conductivity (microSiemens/centimeter)'] < 0) |
                                                      (df['Conductivity (microSiemens/centimeter)'] > 45),
            "Specific Conductivity (microSiemens/centimeter)": (
                    df['Specific Conductivity (microSiemens/centimeter)'] < 1),
            "Salinity (parts per thousand, ppt)": (df['Salinity (parts per thousand, ppt)'] < 0) |
                                                  (df['Salinity (parts per thousand, ppt)'] > .03),
            "pH": (df['pH'] < 1) | (df['pH'] > 13),
            "Dissolved Oxygen (% saturation)": (df['Dissolved Oxygen (% saturation)'] < 10) | (
                    df['Dissolved Oxygen (% saturation)'] > 120),
            "Turbidity (NTU)": (df['Turbidity (NTU)'] < 0),
            "Turbidity (FNU)": (df['Turbidity (FNU)'] < 0),
            "fDOM (RFU)": (df['fDOM (RFU)'] < 0) | (df['fDOM (RFU)'] > 100),
            "fDOM (parts per billion QSU)": (df['fDOM (parts per billion QSU)'] < 0) | (
                    df['fDOM (parts per billion QSU)'] > 300),
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

        # Define a function to floor datetime to the nearest 12 hours (for grouping profiling missions)
        def floor_to_nearest_12_hours(dt):
            # Calculate the number of hours since the beginning of the datetime
            hours = dt.hour
            # Calculate the floored hour by subtracting the remainder of hours divided by 12
            floored_hour = hours - (hours % 12)
            # Replace the hour and minute in the original datetime with the floored hour
            return dt.replace(hour=floored_hour, minute=0, second=0, microsecond=0)

        # Apply the function to the 'datetime' column
        df['profile_time'] = df['Timestamp'].apply(floor_to_nearest_12_hours)

        # Drop all rows from model data which do not have a matching datetime in the reference (sensor) dataset
        mask1 = his_df['time'].isin(df['profile_time'])
        filtered_his_df = his_df[mask1]  # Filter df1 using the mask

        # Drop all rows from sensor data which do not have a matching datetime in the model dataset (for size reduction)
        mask2 = df['profile_time'].isin(filtered_his_df['time'])
        filtered_df = df[mask2]  # Filter df1 using the mask

        # Ensure that x and y columns are numeric, convert depth to model convention (negative values)
        filtered_df['Depth'] = -1 * pd.to_numeric(filtered_df['Depth'], errors='coerce')
        filtered_df[column_name] = pd.to_numeric(filtered_df[column_name], errors='coerce')
        filtered_his_df[feature] = pd.to_numeric(filtered_his_df[feature], errors='coerce')

        # Set indices to ensure proper interpolation
        filtered_df = filtered_df.set_index('profile_time')
        filtered_his_df = filtered_his_df.set_index('time')

        # Drop all rows from model data which are below the lowest sensor data from that date (would be extrapolation)
        min_reference_values = filtered_df.groupby(filtered_df.index)['Depth'].min()

        def filter_rows(group, min_reference_values):
            time = group.name
            min_value = min_reference_values.loc[time]
            return group[group['zcoordinate_c'] >= min_value]

        # Apply the filtering to each group in df1
        refiltered_his_df = filtered_his_df.groupby(filtered_his_df.index).apply(
            lambda group: filter_rows(group, min_reference_values))

        # Drop the extra index level added by groupby + apply
        filtered_his_df = refiltered_his_df.reset_index(level=0, drop=True)

        # Function to interpolate y-values
        def interpolate_depth(group, df1):
            time = group.name
            df1_group = df1.loc[time].sort_values('Depth')
            return group['zcoordinate_c'].apply(lambda x: np.interp(x, df1_group['Depth'], df1_group[column_name]))

        # Apply the interpolation
        result = filtered_his_df.copy()
        result[f'Reference {column_name}'] = filtered_his_df.groupby('time').apply(lambda group:
                                                                                   interpolate_depth(group,
                                                                                                     filtered_df)).reset_index(
            level=0,
            drop=True)
        result = result.rename(columns={feature: f'Model {feature}'})
        result['zcoordinate_c'] = result['zcoordinate_c']  # Use cell-center depth, not +1.25m 'interval' correction

        # Compute some vector statistics comparing the model and reference values
        error_signals = [f'Model {feature}', f'Reference {column_name}', 'Difference', 'Absolute Error',
                         'Squared Error', 'Percent Error']
        result[error_signals[2]] = result[error_signals[1]] - result[error_signals[0]]
        result[error_signals[3]] = abs(result[error_signals[2]])
        result[error_signals[4]] = result[error_signals[2]] ** 2
        result[error_signals[5]] = 100 * result[error_signals[2]] / result[error_signals[1]]

        dpt_av_opts = ["Individual layers", 'Depth-averaged']
        if not offline:
            c1, c2 = st.columns(2, gap='small')
            with c1:
                dpt_av = st.radio("Choose how to compare model output to the reference sensor data", dpt_av_opts,
                                  horizontal=True)
                error_stats = st.multiselect("Choose which error statistics to plot", error_signals,
                                             default=error_signals[2])
        else:
            dpt_av = dpt_av_opts[1]

        if dpt_av == dpt_av_opts[0]:
            ###########################################################################################################
            # Plot time series of feature sorted by depths in Bokeh

            # Create Bokeh figure for the first plot
            p1 = figure(x_axis_label='Date', title=f'Error statistics as depth contours vs. time for {feature}')

            if not errorplot:
                st.write("Please select at least one parameter and depth contour to plot.")
            else:
                # Group the data by 'Depth' and create separate ColumnDataSources for each group
                grouped_data = result.groupby('zcoordinate_c')

                def update_err_contour(selected_variables_p1, grouped_data):

                    num_colors = len(grouped_data)
                    viridis_colors = Viridis256
                    step = len(viridis_colors) // num_colors
                    viridis_subset = viridis_colors[::step][:num_colors]

                    p1.title.text = f'{selected_variables_p1} vs. Date for Different Depths'
                    p1.renderers = []  # Remove existing renderers
                    line_styles = ['solid', 'dashed', 'dotdash', 'dotted']

                    for i, (var) in enumerate(selected_variables_p1):
                        for j, (depth, group) in enumerate(grouped_data):
                            depth_source = ColumnDataSource(group)
                            renderer = p1.line(x='time', y=var, source=depth_source, line_width=2,
                                               line_color=viridis_subset[j],
                                               legend_label=f'{depth}m {var}', line_dash=line_styles[i])
                            p1.add_tools(HoverTool(renderers=[renderer],
                                                   tooltips=[("Time", "@time{%Y-%m-%d %H:%M}"), ("Depth", f'{depth}'),
                                                             (var, f'@{{{var}}}')], formatters={"@time": "datetime", },
                                                   mode="vline"))
                            p1.renderers.append(renderer)

                # Call the update_plot function with the selected variables for the first plot
                update_err_contour(error_stats, grouped_data)

                # Show legend for the first plot
                p1.legend.title = 'Depth'
                # p1.legend.location = "top_left"
                p1.add_layout(p1.legend[0], 'right')
                p1.legend.label_text_font_size = '10px'
                p1.legend.click_policy = "hide"  # Hide lines on legend click
                p1.yaxis.axis_label = "Values"
                p1.xaxis.axis_label = "Time"
                p1.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%y/%m/%d %H:%M")

                # Display the Bokeh chart for the first plot using Streamlit
                st.bokeh_chart(p1, use_container_width=True)
                st.write("Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
                         "Click legend entries to toggle series on/off.")

                ######################################################################################################
                # Display scalar statistics
                st.write(f"Time-averaged summary statistics for error in modeled {feature}")

                # Compute summary (scalar) statistics
                MM = result[error_signals[0]].mean()
                RM = result[error_signals[1]].mean()
                MSTDEV = result[error_signals[0]].std()
                RSTDEV = result[error_signals[1]].std()
                correlation = result[f'Model {feature}'].corr(result[f'Reference {column_name}'])
                SSE = result[error_signals[2]].sum()
                MAE = result[error_signals[3]].sum() / len(result[error_signals[3]])
                MSE = result[error_signals[4]].sum() / len(result[error_signals[4]])
                RMSE = math.sqrt(MSE)
                MPE = result[error_signals[5]].sum() / len(result[error_signals[5]])
                statvalues1 = {'Statistic': ['Mean', 'Standard Deviation'],
                               'Model': [MM, MSTDEV],
                               'Reference Data': [RM, RSTDEV]}

                statvalues2 = {'Statistic': ['Correlation',
                                             "Sum of Squares Error", "Mean Absolute Error", "Mean Squared Error",
                                             "Root Mean Squared Error", 'Mean Percent Error'],
                               'Comparison': [correlation, SSE, MAE, MSE, RMSE, MPE]}

                stats_df1 = pd.DataFrame(statvalues1)
                stats_df1 = stats_df1.reset_index(drop=True)

                stats_df2 = pd.DataFrame(statvalues2)
                stats_df2 = stats_df2.reset_index(drop=True)

                c1, c2, c3, c4 = st.columns(4, gap='small')
                if not offline:
                    with c1:
                        st.markdown("##### Series Statistics")
                        st.dataframe(stats_df1, hide_index=True)
                    with c2:
                        st.markdown("##### Comparative Statistics")
                        st.dataframe(stats_df2, hide_index=True)
                return stats_df1, stats_df2

        elif dpt_av == dpt_av_opts[1]:
            ##########################################################################################################
            # Plot depth-averaged error statistics

            # Perform depth-averaging for all timesteps
            result = result.reset_index()
            columns_to_include = result.columns[6:]
            df_filtered = result[['time'] + list(columns_to_include)]
            depth_av_df = df_filtered.groupby('time').mean().reset_index()

            if not offline:

                # Create Bokeh figure for the first plot
                p1 = figure(x_axis_label='Date', title=f'Error statistics as depth contours vs. time for {feature}')

                if not errorplot:
                    st.write("Please select at least one parameter and depth contour to plot.")
                else:

                    def update_err_contour(selected_variables_p1, result):

                        num_colors = len(selected_variables_p1)
                        viridis_colors = Viridis256
                        step = len(viridis_colors) // num_colors
                        viridis_subset = viridis_colors[::step][:num_colors]

                        p1.title.text = f'Depth-Averaged {selected_variables_p1} vs. Time'
                        p1.renderers = []  # Remove existing renderers
                        line_styles = ['solid', 'dashed', 'dotdash', 'dotted']

                        for i, (var) in enumerate(selected_variables_p1):
                            depth_source = ColumnDataSource(result)
                            renderer = p1.line(x='time', y=var, source=depth_source, line_width=2,
                                               line_color=viridis_subset[i],
                                               legend_label=f'{var}')
                            p1.add_tools(HoverTool(renderers=[renderer],
                                                   tooltips=[("Time", "@time{%Y-%m-%d %H:%M}"),
                                                             (var, f'@{{{var}}}')], formatters={"@time": "datetime", },
                                                   mode="vline"))
                            p1.renderers.append(renderer)

                    # Call the update_plot function with the selected variables for the first plot
                    update_err_contour(error_stats, depth_av_df)

                    # Show legend for the first plot
                    p1.legend.title = 'Depth'
                    # p1.legend.location = "top_left"
                    p1.add_layout(p1.legend[0], 'right')
                    p1.legend.label_text_font_size = '10px'
                    p1.legend.click_policy = "hide"  # Hide lines on legend click
                    p1.yaxis.axis_label = feature
                    p1.xaxis.axis_label = "Time"
                    p1.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%y/%m/%d %H:%M")

                    # Display the Bokeh chart for the first plot using Streamlit
                    st.bokeh_chart(p1, use_container_width=True)
                    st.write(
                        "Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
                        "Click legend entries to toggle series on/off.")

            ######################################################################################################
            # Display scalar statistics

            # Compute summary (scalar) statistics
            MM = result[error_signals[0]].mean()
            RM = result[error_signals[1]].mean()
            MSTDEV = result[error_signals[0]].std()
            RSTDEV = result[error_signals[1]].std()
            correlation = result[f'Model {feature}'].corr(result[f'Reference {column_name}'])
            # covariance = result[f'Model {feature}'].cov(result[f'Reference {column_name}'])
            SSE = result[error_signals[2]].sum()
            MAE = result[error_signals[3]].sum() / len(result[error_signals[3]])
            MSE = result[error_signals[4]].sum() / len(result[error_signals[4]])
            RMSE = math.sqrt(MSE)
            MPE = result[error_signals[5]].sum() / len(result[error_signals[5]])

            statvalues1 = {'Statistic': ['Mean', 'Standard Deviation'],
                           'Model': [MM, MSTDEV],
                           'Reference Data': [RM, RSTDEV]}

            statvalues2 = {'Statistic': ['Correlation',
                                         "Sum of Squares Error", "Mean Absolute Error", "Mean Squared Error",
                                         "Root Mean Squared Error", 'Mean Percent Error'],
                           'Comparison': [correlation, SSE, MAE, MSE, RMSE, MPE]}

            stats_df1 = pd.DataFrame(statvalues1)
            stats_df1 = stats_df1.reset_index(drop=True)

            stats_df2 = pd.DataFrame(statvalues2)
            stats_df2 = stats_df2.reset_index(drop=True)

            c1, c2, c3, c4 = st.columns(4, gap='small')
            if not offline:
                with c1:
                    st.markdown("##### Series Statistics")
                    st.dataframe(stats_df1, hide_index=True)
                with c2:
                    st.markdown("##### Comparative Statistics")
                    st.dataframe(stats_df2, hide_index=True)
            return stats_df1, stats_df2


def display_his(o_file):
    # Open hisfile with xarray and print netcdf structure
    ds_his_o = xr.open_mfdataset(o_file, preprocess=dfmt.preprocess_hisnc)
    ds_his = rename_ds(ds_his_o)

    ###################################################################################################################
    # 1. Analyze the file contents. Create references for plotting.
    # print('ds_his', ds_his)
    # print('All data_vars', ds_his.data_vars)
    # print('station coordx, coordy', ds_his['station_geom_node_coordx'].values, ds_his['station_geom_node_coordy'].values)
    # print('source_sink coordx, coordy', ds_his['source_sink_geom_node_coordx'].values,
    #       ds_his['source_sink_geom_node_coordy'].values)

    # Handle the possibility that some dimensions or coordinates are not present in the dataset
    if 'stations' in ds_his.dims:
        num_stations = ds_his.sizes['stations']
    else:
        num_stations = None
    if 'time' in ds_his.dims:
        num_times = ds_his.sizes['time']
        # Extract and reformat a list of all time steps, and create a dictionary for converting the calendar time to an index
        times = ds_his.coords["time"]
        formatted_times = times.dt.strftime("%Y-%m-%d %H:%M:%S")
        formatted_times = formatted_times.values.tolist()
        times_dict = {value: i for i, value in enumerate(formatted_times)}
    else:
        num_times = None
    if 'laydim' in ds_his.dims:
        num_layers = ds_his.sizes['laydim']
    else:
        num_layers = None
    if 'cross_section' in ds_his.dims:
        num_source_sink = ds_his.sizes['cross_section']
    else:
        num_cross_section = None
    if 'source_sink' in ds_his.dims:
        num_source_sink = ds_his.sizes['source_sink']
    else:
        num_source_sink = None

    # Create lists of data variables based on their dimensionality (must exist at location and times)
    includes_coordinate = ["stations", "time"]
    excludes_coordinates = ["station_geom_nNodes", "source_sink_geom_nNodes", "source_sink_pts",
                            "cross_section_geom_nNodes"]
    stations_list = []
    for name, var in ds_his.data_vars.items():
        if (all(coord in var.dims for coord in includes_coordinate) and
                all(coord not in var.dims for coord in excludes_coordinates)):
            stations_list.append(name)

    includes_coordinate = ["source_sink", "time"]
    excludes_coordinates = ["station_geom_nNodes", "source_sink_geom_nNodes", "source_sink_pts",
                            "cross_section_geom_nNodes"]
    source_sink_list = []
    for name, var in ds_his.data_vars.items():
        if (all(coord in var.dims for coord in includes_coordinate) and
                all(coord not in var.dims for coord in excludes_coordinates)):
            source_sink_list.append(name)

    includes_coordinate = ["cross_section", "time"]
    excludes_coordinates = ["station_geom_nNodes", "source_sink_geom_nNodes", "source_sink_pts",
                            "cross_section_geom_nNodes"]
    cross_section_list = []
    for name, var in ds_his.data_vars.items():
        if (all(coord in var.dims for coord in includes_coordinate) and
                all(coord not in var.dims for coord in excludes_coordinates)):
            cross_section_list.append(name)

    # Option to display dimensionality and contents of the file
    on_off = ["Display file attributes", "Hide"]
    print_attrs = st.radio(label='', options=on_off, horizontal=True, index=1)
    if print_attrs == on_off[0]:
        a1, a2, a3 = st.columns(3, gap='small')
        with a1:
            st.markdown("#### Dimensions")
            st.write("Observation Points:", num_stations)
            st.write("Sources/Sinks:", num_source_sink)
            st.write("Time steps:", num_times)
        with a2:
            st.markdown("#### Data variables")
            st.write("Data variables per observation point", stations_list)
            st.write("Data variables per source/sink", source_sink_list)
            st.write("Data variables per cross_section", cross_section_list)
        with a3:
            st.markdown("#### Timing")
            st.write('Start:', ds_his.attrs["time_coverage_start"])
            st.write('End:', ds_his.attrs["time_coverage_end"])
            st.write('Duration:', ds_his.attrs["time_coverage_duration"])
            st.write("Time resolution:", ds_his.attrs['time_coverage_resolution'])
            st.write("Run date:", ds_his.attrs['date_modified'])
    else:
        st.empty()

    hc1, hc2 = st.columns(2, gap="small")
    with hc1:
        hisoptions = ['Time series', 'Instantaneous (vs. depth)', 'Comparison against reference data (sensor)']
        plottype = st.radio("Choose which type of data to display:", options=hisoptions, horizontal=True)

    if plottype == hisoptions[0]:
        pointoptions = ["Observation Points", "Observation Cross-Sections", "Sources/Sinks"]
        hc1, hc2 = st.columns(2, gap="small")
        with hc1:
            pointtype = st.radio("Choose which type of location to plot:", options=pointoptions, horizontal=True)

            if pointtype == pointoptions[0]:
                locations = st.multiselect("Select observation points to plot",
                                           ds_his.coords['stations'].values,
                                           default=ds_his.coords['stations'].values[0])
                feature = st.selectbox("Select a variable to plot", stations_list)
            elif pointtype == pointoptions[2]:
                locations = st.multiselect("Select sources/sinks to plot", ds_his.coords['source_sink'].values,
                                           default=ds_his.coords['source_sink'].values[0])
                feature = st.selectbox("Select a variable to plot", source_sink_list)
                num_layers = None
            else:
                locations = st.multiselect("Select observation cross-section to plot",
                                           ds_his.coords['cross_section'].values,
                                           default=ds_his.coords['cross_section'].values[0])
                feature = st.selectbox("Select a variable to plot", cross_section_list)
    elif plottype == hisoptions[1]:
        hc1, hc2 = st.columns(2, gap="small")
        with hc1:
            locations = st.multiselect("Select observation points to plot",
                                       ds_his.coords['stations'].values,
                                       default=ds_his.coords['stations'].values[0])
            feature = st.selectbox("Select a variable to plot", stations_list)
    elif plottype == hisoptions[2]:
        # Dictionary for converting parameter names as {model name : profiler name}
        compatibility = {"Salinity (ppt)": "Salinity (parts per thousand, ppt)",
                         "Temperature (◦C)": "Temperature (Celsius)"}
        errorplots = ["Hourly (depth = 2.95 m)", "Depth profiles (12-hour sample rate)"]
        c1, c2 = st.columns(2, gap='small')
        with c1:
            feature = st.selectbox("Select a variable to compare", compatibility.keys())
            column_name = compatibility.get(feature)  # 'feature' name in reference dataset
            errorplot = st.radio("Select a sensor dataset for comparison", errorplots, horizontal=True)
            location = st.selectbox("Select observation points to plot against the profiler data",
                                       ds_his.coords['stations'].values)
        display_error(ds_his=ds_his, feature=feature, column_name=column_name, errorplot=errorplot,
                      errorplots=errorplots, location=location, offline=False)

    ###################################################################################################################
    # 2. Plot time series data for the selected point(s), at one or several depths
    if plottype == hisoptions[0]:
        layer_depths_his = []
        if num_layers is not None:
            if 'laydim' in ds_his[feature].dims:
                concise = ds_his[feature].dropna(dim='laydim', how='all')
                num_layers = concise.sizes['laydim']
                layer_list = list(reversed(range(0, num_layers)))
                if 'zcoordinate_c' in ds_his.coords:
                    ds_his["zcoordinate_c"] = ds_his["zcoordinate_c"].round(2)
                    depths = np.unique(ds_his.coords['zcoordinate_c'].values)
                    interval = abs(depths[1] - depths[0]) / 2
                    layer_depths_his = depths  # Use actual cell-center values, without 1.25 'interval' offset
                    layer_depths_his = list(layer_depths_his[::-1])
                    layer_depths_his = [x for x in layer_depths_his if not math.isnan(x)]
                else:
                    max_depth = -ds_his.coords['zcoordinate_c'].min().to_numpy()[()]
                    layer_depths_his = np.round(np.linspace(0, max_depth - max_depth / num_layers, num_layers))
                    layer_depths_his = layer_depths_his.tolist()
                hc1, hc2 = st.columns(2, gap="small")
                with hc1:
                    depth_selected = st.multiselect("Select depth at which to plot", ["All"] + layer_depths_his,
                                                    default="All")
                    if "All" in depth_selected:
                        layers = layer_depths_his
                    else:
                        layers = depth_selected
            else:
                num_layers = 1
                layers = [0]
        else:
            num_layers = 1
            layers = [0]

        if pointtype == pointoptions[0]:
            data_for_bokeh = ds_his[feature].sel(stations=locations)
        elif pointtype == pointoptions[1]:
            data_for_bokeh = ds_his[feature].sel(cross_section=locations)
        elif pointtype == pointoptions[2]:
            data_for_bokeh = ds_his[feature].sel(source_sink=locations)
        his_df = data_for_bokeh.to_dataframe()

        def update_p_his(p_his, df, selected_variables_p_his, grouptype, groupvar, layers):

            # Group the data by depth and create separate ColumnDataSources for each group
            df.reset_index()
            if groupvar == 'zcoordinate_c':
                df[groupvar] = df[groupvar]  # Use cell-center depth without 1.25m 'interval' correction for cell top
                df_filtered = df[df[groupvar].isin(layers)]
                grouped_data = df_filtered.groupby(groupvar)
            else:
                grouped_data = df.groupby(groupvar)

            if grouped_data.ngroups != 0:
                num_colors = grouped_data.ngroups
                viridis_colors = Viridis256
                step = len(viridis_colors) // num_colors
                viridis_subset = viridis_colors[::step][:num_colors]

                p_his.title.text = f'{selected_variables_p_his} vs. Time at {locations}'
                p_his.renderers = []  # Remove existing renderers

                for i, (groupname, group) in enumerate(grouped_data):
                    if groupvar == 'zcoordinate_c':
                        ingroup = group.groupby('stations')
                        for j, (obspoint, obs) in enumerate(ingroup):
                            groupname_source = ColumnDataSource(obs)
                            renderer = p_his.line(x='time', y=selected_variables_p_his, source=groupname_source,
                                                  line_width=2,
                                                  line_color=viridis_subset[i],
                                                  legend_label=f'{obspoint}: {groupname}')
                            p_his.add_tools(HoverTool(renderers=[renderer],
                                                      tooltips=[("Time", "@time{%Y-%m-%d %H:%M}"),
                                                                (grouptype, f'{groupname}'),
                                                                ('Location', f'{obspoint}'),
                                                                (selected_variables_p_his,
                                                                 f'@{{{selected_variables_p_his}}}')],
                                                      formatters={"@time": "datetime", },
                                                      mode="vline"))
                            p_his.renderers.append(renderer)
                    else:
                        groupname_source = ColumnDataSource(group)
                        renderer = p_his.line(x='time', y=selected_variables_p_his, source=groupname_source,
                                              line_width=2,
                                              line_color=viridis_subset[i],
                                              legend_label=f'{groupname}')
                        p_his.add_tools(HoverTool(renderers=[renderer],
                                                  tooltips=[("Time", "@time{%Y-%m-%d %H:%M}"),
                                                            (grouptype, f'{groupname}'),
                                                            (selected_variables_p_his,
                                                             f'@{{{selected_variables_p_his}}}')],
                                                  formatters={"@time": "datetime", },
                                                  mode="vline"))
                        p_his.renderers.append(renderer)

        # Call the update_plot function with the selected variables for the first plot
        p_his = figure()
        df_reset = his_df.reset_index()
        if 'zcoordinate_c' in df_reset.columns:
            groupvar = 'zcoordinate_c'
            grouptype = 'Depth'
        if 'mesh2d_bldepth' in df_reset.columns:
            groupvar = 'mesh2d_bldepth'
            grouptype = 'Sigma Layer'
        elif 'stations' in df_reset.columns:
            groupvar = 'stations'
            grouptype = 'Observation Point'
        elif 'source_sink' in df_reset.columns:
            groupvar = 'source_sink'
            grouptype = 'Source/Sink'
        elif 'cross_section' in df_reset.columns:
            groupvar = 'cross_section'
            grouptype = 'Observation Cross Section'
        update_p_his(p_his, df_reset, feature, grouptype=grouptype, groupvar=groupvar, layers=layers)

        # Show legend for the first plot
        p_his.add_layout(p_his.legend[0], 'right')
        p_his.legend.title = grouptype
        p_his.legend.location = "top_left"
        p_his.legend.label_text_font_size = '10px'
        p_his.legend.click_policy = "hide"  # Hide lines on legend click
        p_his.yaxis.axis_label = f"{feature}"
        p_his.xaxis.axis_label = "Time"
        # plotrange = set_last_date - set_begin_date
        # if plotrange > timedelta(days=62):
        #     p_his.x_range = Range1d(set_begin_date - timedelta(days=3), set_last_date + timedelta(days=3))
        # else:
        #     p_his.x_range = Range1d(set_begin_date, set_last_date + timedelta(days=1, hours=3))
        p_his.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%y/%m/%d %H:%M")

        # Display the Bokeh chart for the first plot using Streamlit
        st.bokeh_chart(p_his, use_container_width=True)

    ###################################################################################################################
    # 3. Plot parameters vs. depth at selected point(s) at one time
    elif plottype == hisoptions[1]:
        if 'laydim' in ds_his[feature].dims:
            data_for_bokeh = ds_his[feature].sel(stations=locations)
            his_df = data_for_bokeh.to_dataframe()
            df_reset = his_df.reset_index()
            print(df_reset['time'].dtypes)
            hc1, hc2 = st.columns(2, gap="small")
            with hc1:
                plottime = st.multiselect("Select times at which to plot instantaneous values vs. depth",
                                          np.unique(df_reset['time']), default=df_reset['time'].iloc[:1])

            p2 = figure(x_axis_label=f'{feature}', y_axis_label='Depth',
                        title=f'Vertical Profile for {feature} at {plottime}')

            def update_instant(feature, plottime):

                p2.title.text = f'Vertical Profile for {feature}'
                p2.renderers = []  # Remove existing renderers
                # Filter DataFrame based on selected depths
                filtered_df_p1 = df_reset[df_reset['stations'].isin(locations)]

                for j, date_val in enumerate(plottime):
                    # Filter data based on selected date for the second plot
                    filtered_data_p2 = filtered_df_p1[filtered_df_p1['time'] == date_val]
                    date_string = np.datetime_as_string(date_val, unit='m')

                    # Sort the data by 'Depth'
                    filtered_data_p2 = filtered_data_p2.sort_values(by='zcoordinate_c')

                    station_groups = filtered_data_p2.groupby('stations')

                    for i, (var, group) in enumerate(station_groups):
                        # Create Bokeh ColumnDataSource for the second plot

                        source_plot2 = ColumnDataSource(group)

                        line_renderer = p2.line(x=feature, y='zcoordinate_c', source=source_plot2, line_width=1,
                                                line_color=Category20_20[j + i * len(plottime)],
                                                legend_label=f'{var} at {date_string}')
                        p2.add_tools(HoverTool(renderers=[line_renderer], tooltips=[("Depth", '@zcoordinate_c'),
                                                                                    ('Time', date_string),
                                                                                    (feature, f'@{{{feature}}}')],
                                               mode="hline"))
                        p2.renderers.append(line_renderer)

            # Call the update_plot function with the selected variables and date for the second plot
            update_instant(feature, plottime)

            # Show legend for the second plot
            p2.legend.title = 'Parameters'
            p2.add_layout(p2.legend[0], 'right')
            p2.legend.location = "top_left"
            p2.legend.label_text_font_size = '10px'
            p2.legend.click_policy = "hide"  # Hide lines on legend click

            # Display the Bokeh chart for the second plot using Streamlit
            st.bokeh_chart(p2, use_container_width=True)
            st.write("Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
                     "Click legend entries to toggle series on/off.")
        else:
            st.warning(f"Selected variable does not vary with depth.")


def current(all_files, directory_path):
    """
    Display and explore Delft3D output files
    """

    output_options = ["Spatial distributions (map file)", "Fixed locations (history file)"]
    d3d_output = st.radio("Select which type of model outputs to display",
                          options=output_options, horizontal=True)

    # Filter files based on extension
    if d3d_output == output_options[1]:
        filtered_files = [f for f in all_files if f.endswith('his.nc')] + ["Upload your own"]
        hc1, hc2 = st.columns(2, gap="small")
        with hc1:
            selected_file = st.selectbox(label="Select which output to display", options=filtered_files)
            if directory_path is not None and selected_file != "Upload your own":
                selected_file = os.path.join(directory_path, selected_file)
        if selected_file == "Upload your own":
            hc1, hc2 = st.columns(2, gap="small")
            with hc1:
                uploaded = st.file_uploader(
                    label='Upload your own Delft3D history output file (his.nc), maximum size 200MB', type='nc')
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
    elif d3d_output == output_options[0]:
        filtered_files = [f for f in all_files if f.endswith('map.nc')] + ["Upload your own"]
        # if uploaded is not None:
        #     filtered_files.append(uploaded.name)
        hc1, hc2 = st.columns(2, gap="small")
        with hc1:
            selected_file = st.selectbox(label="Select which model output to display", options=filtered_files)
            if directory_path is not None and selected_file != "Upload your own":
                selected_file = os.path.join(directory_path, selected_file)
        if selected_file == "Upload your own":
            hc1, hc2 = st.columns(2, gap="small")
            with hc1:
                uploaded = st.file_uploader(
                    label='Upload your own Delft3D NetCDF map output file (map.nc), maximum size 200MB', type='nc')
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
