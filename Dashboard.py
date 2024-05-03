# Framework for a selection-driven, hierarchical app for viewing water quality data and controlling
# data acquisition systems using Streamlit as an interface.

# Launch by opening the terminal to the script's location and entering "streamlit run Dashboard.py".

import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium, folium_static
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, DataRange1d, HoverTool, Range1d
from bokeh.palettes import Viridis256, Category20_20, Spectral11
from bokeh.layouts import column
from datetime import date, time, datetime, timedelta
from dateutil.relativedelta import relativedelta
import xarray as xr
import matplotlib.pyplot as plt
import contextily as ctx
import dfm_tools as dfmt


def main():
    st.set_page_config("Brusdalsvatnet WQ Dashboard", layout="wide")
    st.sidebar.title("Choose Mode")
    selected_page = st.sidebar.radio("", ["Historic", "Current Hydrodynamic Model", "Interactive (Path Planning)"])

    if selected_page == "Historic":
        historic()
    elif selected_page == "Current Hydrodynamic Model":
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
        elif profiler_data == "Current Cache":
            st.markdown("#### During the monitoring season the profiler station normally publicly broadcasts recent "
                        "measurements")
            # IP of the station
            website_url = "https://89.9.10.123/"
            # Use st.components.iframe to embed the website
            st.components.v1.iframe(website_url, height=600)
        else:
            vertical()
    elif source == "Weather Station":
        weather()
        # st.markdown("#### Access to weather station data currently requires a commercial license from Volue")
        # # URL of the Volue commercial platform which currently stores this data
        # website_url = "https://sensordata.no/vdv.php/historical/714"
        #
        # # Use st.components.iframe to embed the website
        # st.components.v1.iframe(website_url, height=600)
    else:
        st.write("Sorry, no data available in the dashboard from the USVs at this time")
    st.write(
        "Find a bug? Or have an idea for how to improve the app? "
        "Please log suggestions [here](https://github.com/russellprimeau/BrusdalsvatnetDT/issues).")


# Function to the upload new profiler data from CSV
@st.cache_data
def upload_weather_csv_page():
    csv_file2 = "Profiler_modem_SondeHourly.csv"  # Replace with the actual file path
    df = pd.read_csv(csv_file2, skiprows=[0, 2, 3])

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
                "fDOM_QSU": "fDOM (ppb QSU)",
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
                "sensorParms(11)": "fDOM (ppb QSU)",
                "lat": "Latitude",
                "lon": "Longitude",
            }
            df = df.rename(columns=column_names)  # Assign column names for profiler data

        # st.write("Uploaded DataFrame:")
        # st.dataframe(df)

        # Data cleaning
        for parameter in df.columns:
            df[parameter] = df[parameter].apply(lambda x: np.nan if x == 'NAN' else x)
        df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce', downcast='float')

        # Convert the time column to a datetime object
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(lambda x: x.to_pydatetime())
        df = df.sort_values(by="Timestamp")
        df = df.drop(columns=['Record Number'])

    return df


def weather():
    st.title("Weather Station Data")
    st.markdown("#### Sorry, the weather page is being updated at the moment.")
    # pw = figure(title="Time Series Data at 2.9m Depth")
    #
    # file_paths = ['All_10min.csv', 'All_time.csv', 'All_Prec_int_hr.csv', 'All_min.csv']
    # dfs = []  # List of uploaded dataframes
    # variables = []  # List of column names
    # for file_path in file_paths:
    #     df = pd.read_csv(file_path, sep=';', decimal=',')
    #     df['Time'] = pd.to_datetime(df['Time'])
    #     # Check for error values
    #     columns_to_check = df.columns[1:]
    #     variables + list(df.columns[1:])
    #     for col in columns_to_check:
    #         df.replace({col: -99.9}, pd.NA, inplace=True)
    #         # df[col].replace(999.9, pd.NA, inplace=True)
    #     dfs.append(df)
    #
    # # Multi-select to select multiple Y variables, including "Select All"
    # selected_wvariables = st.multiselect(
    #     "Select water quality parameters to plot",
    #     ["Select All"] + variables,
    # )
    #
    # # Check if "Select All" is chosen
    # if "Select All" in selected_wvariables:
    #     selected_wvariables = variables
    #
    # # Create a ColumnDataSource
    # source = ColumnDataSource(df)
    # time_difference = timedelta(hours=12)
    #
    # def update_weather(selected_wvariables):
    #     p.title.text = f'Weather Parameters vs. Time'
    #
    #     for variable, color in zip(selected_wvariables, Spectral11):
    #         # Convert 'Date' to a pandas Series to use shift operation
    #         date_series = pd.Series(source.data['Timestamp'])
    #
    #         # Add a new column 'Gap' indicating when a gap is detected within each 'Depth' group
    #         source.data['Gap'] = (date_series - date_series.shift(1)) > time_difference
    #
    #         # Replace the 'Value' with NaN when a gap is detected
    #         source.data[variable] = np.where(source.data['Gap'], np.nan, source.data[variable])
    #
    #         line_render = p.line(
    #             x="Timestamp", y=variable, line_width=2, color=color, source=source, legend_label=variable
    #         )
    #         p.renderers.append(line_render)
    #
    # # Call the update_plot function with the selected variables for the first plot
    # if not selected_wvariables:
    #     st.write("Please select at least one parameter to plot.")
    # else:
    #     update_weather(selected_wvariables)
    #     # Set plot properties
    #     pw.title.text_font_size = "16pt"
    #     pw.xaxis.axis_label = "Time"
    #     pw.yaxis.axis_label = "Variable Value(s)"
    #     pw.legend.title = "Weather Parameters"
    #     pw.legend.click_policy = "hide"  # Hide lines on legend click
    #     # Set the x-axis formatter to display dates in the desired format
    #     pw.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%Y/%m/%d %H:%M")
    #     st.bokeh_chart(pw, use_container_width=True)
    #     st.write("Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
    #              "Click legend entries to toggle series on/off.")


# Function to the upload new profiler data from CSV
@st.cache_data
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
                "fDOM_QSU": "fDOM (ppb QSU)",
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
                "sensorParms(11)": "fDOM (ppb QSU)",
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
            ["Select All"] + list(df.columns[1:11]),
        )

    clean_setting = st.radio(
        "Choose how to filter the dataset",
        options=["Raw", "Remove Suspicious Values"],
        horizontal=True
    )

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

    set_begin_date = first_date
    set_last_date = last_date
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

    if clean_setting == "Remove Suspicious Values":
        # Define conditions for each parameter which indicate errors in the data
        error_conditions = {
            "Timestamp": (df['Timestamp'] < pd.to_datetime('2000-01-01')) | (
                    df['Timestamp'] > pd.to_datetime('2099-12-31')),
            "Temperature (Celsius)": (df['Temperature (Celsius)'] < -5) | (df['Temperature (Celsius)'] > 25),
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
            "fDOM (ppb QSU)": (df['fDOM (ppb QSU)'] < 0) | (df['fDOM (ppb QSU)'] > 300),
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

    # Check if "Select All" is chosen
    if "Select All" in selected_variables:
        selected_variables = list(df.columns[1:11])

    # Create a ColumnDataSource
    source = ColumnDataSource(df)
    time_difference = timedelta(hours=2)

    def update_hourly(selected_variables):
        p.title.text = f'Water Quality Parameters vs. Time'

        for variable, color in zip(selected_variables, Spectral11):
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
            p.x_range = Range1d(set_begin_date, set_last_date)
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
        "sensorParms(11)": "fDOM (ppb QSU)",
        "lat": "Latitude",
        "lon": "Longitude",
    }
    df = df.rename(columns=column_names)

    # Convert the time column to a datetime object
    df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(lambda x: x.to_pydatetime())

    print('columns', df.columns)

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

    st.title("Vertical Profiler Data")
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
                            "fDOM (ppb QSU)"]
    mc1, mc2 = st.columns(2, gap="small")
    with mc1:
        selected_variables_p1 = st.multiselect('Select Water Quality Parameters', variables_to_plot_p1, default=[])

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
        options=["Raw", "Remove Suspicious Values"],
        horizontal=True
    )

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

    set_begin_date = first_date
    set_last_date = last_date
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

    if clean_setting == "Remove Suspicious Values":
        # Define conditions for each parameter which indicate errors in the data
        error_conditions = {
            "Timestamp": (df['Timestamp'] < pd.to_datetime('2000-01-01')) | (
                    df['Timestamp'] > pd.to_datetime('2099-12-31')),
            "Temperature (Celsius)": (df['Temperature (Celsius)'] < -5) | (df['Temperature (Celsius)'] > 25),
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
            "fDOM (ppb QSU)": (df['fDOM (ppb QSU)'] < 0) | (df['fDOM (ppb QSU)'] > 300),
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
        p1.legend.label_text_font_size = '10px'
        p1.legend.click_policy = "hide"  # Hide lines on legend click
        p1.yaxis.axis_label = "Variable Value(s)"
        p1.xaxis.axis_label = "Time"
        plotrange = set_last_date - set_begin_date
        if plotrange > timedelta(days=62):
            p1.x_range = Range1d(set_begin_date - timedelta(days=3), set_last_date + timedelta(days=3))
        else:
            p1.x_range = Range1d(set_begin_date, set_last_date)
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
        selected_variables_p2 = st.multiselect('Select Water Quality Parameters)', variables_to_plot_p1, default=[])

        # Add a multiselect box for date for the second plot
        selected_dates_p2 = st.multiselect('Select Date for Vertical Profile (search by typing YYYY-MM-DD)',
                                           df['Date'].dt.strftime('%Y-%m-%d').unique())

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
                if nightman_dayman == 'Night/AM/00:00':
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
        p2.legend.label_text_font_size = '10px'
        p2.legend.click_policy = "hide"  # Hide lines on legend click

        # Display the Bokeh chart for the second plot using Streamlit
        st.bokeh_chart(p2, use_container_width=True)
        st.write("Use the buttons on the right to interact with the chart: pan, zoom, full screen, save, etc. "
                 "Click legend entries to toggle series on/off.")


def current():
    # Function to create Folium map with customizable style
    st.header("Brusdalsvatnet Water Quality Dashboard")
    st.title("Hydrodynamic Model of Current Conditions")



    st.write("Select depth layer to display:")
    options_list = range(0, 20)
    layer = st.selectbox("Select a number:", options_list)  # Create the dropdown menu

    file_nc_his = None
    file_nc_map = r"C:\Users\Russell\Documents\Deltares\FM Projects\Automatic.dsproj_data\ForWAQ\dflowfm\output\ForWAQ_map.nc"
    rename_mapvars = {}
    sel_slice_x, sel_slice_y = slice(50000, 55000), slice(None, 424000)
    layer = 34
    crs = 'EPSG:4326'
    raster_res = 50
    umag_clim = None
    scale = 1.5
    line_array = np.array([[53181.96942503, 424270.83361629],
                           [55160.15232593, 416913.77136685]])

    # Open hisfile with xarray and print netcdf structure
    if file_nc_his is not None:
        ds_his = xr.open_mfdataset(file_nc_his, preprocess=dfmt.preprocess_hisnc)

    # Plot his data: waterlevel at stations
    if file_nc_his is not None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ds_his.waterlevel.plot.line(ax=ax, x='time')
        ax.legend(ds_his.stations.to_series(), loc=1, fontsize=8)  # Optional, to change legend location

    # Open and merge mapfile with xugrid(xarray) and print netcdf structure
    uds_map = dfmt.open_partitioned_dataset(file_nc_map)
    uds_map = uds_map.rename(rename_mapvars)
    print('uds_map!', uds_map)
    print('uds_map[mesh2d_tem1].isel(time=-1)', uds_map['mesh2d_tem1'].isel(time=-1))





    # Plot water level on map
    fig1, ax = plt.subplots(figsize=(10, 4))
    pc = uds_map['mesh2d_tem1'].isel(time=-1, mesh2d_nLayers=layer, nmesh2d_layer=layer,
                                     missing_dims='ignore').ugrid.plot(cmap='jet')
    if crs is None:
        ax.set_aspect('equal')
    else:
        ctx.add_basemap(ax=ax, source=ctx.providers.Esri.WorldImagery, crs=crs, attribution=False)
    fig1.suptitle("Temperature")
    st.pyplot(fig1)

    layer = 18
    # Plot water level on map
    fig2, ax = plt.subplots(figsize=(10, 4))
    pc = uds_map['mesh2d_sa1'].isel(time=-1, mesh2d_nLayers=layer, nmesh2d_layer=layer,
                                     missing_dims='ignore').ugrid.plot(cmap='jet')
    if crs is None:
        ax.set_aspect('equal')
    else:
        ctx.add_basemap(ax=ax, source=ctx.providers.Esri.WorldImagery, crs=crs, attribution=False)
    fig2.suptitle("Salinity")

    st.pyplot(fig2)
    #
    pc2 = uds_map['mesh2d_tem1'].isel(time=3) + 5
    if crs is None:
        ax.set_aspect('equal')
    else:
        ctx.add_basemap(ax=ax, source=ctx.providers.Esri.WorldImagery, crs=crs, attribution=False)

    # Plot eastward velocities on map, on layer
    fig3, ax = plt.subplots(figsize=(10, 4))
    fig3.suptitle("Surface velocity")
    pc = uds_map['mesh2d_ucx'].isel(time=3, mesh2d_nLayers=layer, nmesh2d_layer=layer,
                                    missing_dims='ignore').ugrid.plot(cmap='jet')
    if crs is None:
        ax.set_aspect('equal')
    else:
        ctx.add_basemap(ax=ax, source=ctx.providers.Esri.WorldImagery, crs=crs, attribution=False)
    st.pyplot(fig3)
    #
    # # Plot eastward velocities on map, on depth from waterlevel/z0/bedlevel
    # uds_map_atdepths = dfmt.get_Dataset_atdepths(data_xr=uds_map.isel(time=3), depths=-5, reference='waterlevel')
    # figy, ax = plt.subplots(figsize=(10, 4))
    # pc = uds_map_atdepths['mesh2d_ucx'].ugrid.plot(cmap='jet')
    # if crs is None:
    #     ax.set_aspect('equal')
    # else:
    #     ctx.add_basemap(ax=ax, source=ctx.providers.Esri.WorldImagery, crs=crs, attribution=False)
    #
    # # velocity magnitude and quiver
    # uds_quiv = uds_map.isel(time=-1, mesh2d_nLayers=-2, nmesh2d_layer=-2, missing_dims='ignore')
    # varn_ucx, varn_ucy = 'mesh2d_ucx', 'mesh2d_ucy'
    # magn_attrs = {'long_name': 'velocity magnitude', 'units': 'm/s'}
    # uds_quiv['magn'] = np.sqrt(uds_quiv[varn_ucx] ** 2 + uds_quiv[varn_ucy] ** 2).assign_attrs(magn_attrs)
    # raster_quiv = dfmt.rasterize_ugrid(uds_quiv[[varn_ucx, varn_ucy]], resolution=raster_res)
    # st.pyplot(figy)
    #
    # # Plot
    # figx, ax = plt.subplots(figsize=(10, 4))
    # pc = uds_quiv['magn'].ugrid.plot(cmap='jet')
    # raster_quiv.plot.quiver(x='mesh2d_face_x', y='mesh2d_face_y', u=varn_ucx, v=varn_ucy, color='w', scale=scale,
    #                         add_guide=False)
    # pc.set_clim(umag_clim)
    # figx.tight_layout()
    # if crs is None:
    #     ax.set_aspect('equal')
    # else:
    #     ctx.add_basemap(ax=ax, source=ctx.providers.Esri.WorldImagery, crs=crs, attribution=False)
    # st.pyplot(figx)



    # def create_map(selected_atrib, map_center, zoom_level=13):
    #     # Create a Folium map
    #     m = folium.Map(location=map_center, zoom_start=zoom_level)
    #
    #     # Add GeoJSON layers to the map with customizable style
    #     for file_key, color in selected_atrib:
    #         geojson_path = file_key_to_path[file_key]
    #         gdf = gpd.read_file(geojson_path)
    #         style_function = lambda x: {'fillColor': color, 'color': color}
    #         folium.GeoJson(gdf, style_function=style_function).add_to(m)
    #
    #     return m
    #
    # # Hardcoded GeoJSON file paths, colors, and map center
    # geojson_paths_and_colors = {
    #     "Surface": (r"0m_grid_ps.geojson", "blue"),
    #     "10m": (r"10m_grid_ps.geojson", "green"),
    #     "20m": (r"20m_grid_ps.geojson", "red"),
    #     "30m": (r"30m_grid_ps.geojson", "black"),
    #     "40m": (r"40m_grid_ps.geojson", "yellow"),
    #     "50m": (r"50m_grid_ps.geojson", "green"),
    #     "60m": (r"60m_grid_ps.geojson", "green"),
    #     "70m": (r"70m_grid_ps.geojson", "green"),
    #     "80m": (r"80m_grid_ps.geojson", "green"),
    #     "90m": (r"90m_grid_ps.geojson", "green"),
    #     "100m": (r"100m_grid_ps.geojson", "green"),
    # }
    #
    # hardcoded_map_center = [62.476994, 6.469730]
    #
    # # Convert the dictionary to a list for the multiselect widget
    # file_key_to_path = {key: path for key, (path, _) in geojson_paths_and_colors.items()}
    # multiselect_options = list(geojson_paths_and_colors.keys())
    #
    # # Display map with selectable GeoJSON overlays
    # st.subheader("Displays water quality parameters on 100m x 100m grid for a given depth")
    #
    # # Multiselect widget to choose GeoJSON files
    # mc1, mc2 = st.columns(2, gap="small")
    # with mc1:
    #     selected_files_and_colors = st.multiselect("Select depth at which to display parameters", multiselect_options)
    #
    #     chloropleth_options = ["Temperature (Celsius)", "Conductivity (microSiemens/centimeter)",
    #                            "Specific Conductivity (microSiemens/centimeter)", "Salinity (parts per thousand, ppt)",
    #                            "pH",
    #                            "Dissolved Oxygen (% saturation)", "Turbidity (NTU)", "Turbidity (FNU)", "fDOM (RFU)",
    #                            "fDOM (ppb QSU)"]
    #
    #     # Multiselect widget to choose GeoJSON files
    #     selected_param = st.multiselect("Select parameter by which to color-code", chloropleth_options)
    #
    # # Display map if files are selected
    # if selected_files_and_colors:
    #     # Create Folium map with selected GeoJSON files and colors
    #     folium_map = create_map(
    #         [(file_key, geojson_paths_and_colors[file_key][1]) for file_key in selected_files_and_colors],
    #         map_center=hardcoded_map_center, zoom_level=13)
    #
    #     # Display Folium map using folium_static
    #     folium_static(folium_map, width=1300)
    #     st.write("Find a bug? Or have an idea for how to improve the app? "
    #              "Please log suggestions [here](https://github.com/russellprimeau/BrusdalsvatnetDT/issues).")
    # else:
    #     st.info("Please select at least one depth and model parameter to display.")


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
