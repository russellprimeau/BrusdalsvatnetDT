# Framework for a selection-driven, hierarchical app for viewing water quality data and controlling
# data acquisition systems using Streamlit as an interface.

# Launch by opening the terminal to the script's location and entering "streamlit run Dashboard.py".

import numpy as np
import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium, folium_static
import geopandas as gpd
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, DataRange1d, Legend
from bokeh.palettes import Viridis256, Category20_20, Spectral11
from bokeh.layouts import column
from datetime import date, datetime, timedelta


def main():
    st.set_page_config("Brusdalsvatnet WQ Dashboard", layout="wide")
    st.sidebar.title("Choose Mode")
    selected_page = st.sidebar.radio("", ["Historic", "Current Hydrologic Model", "Interactive (Path Planning)"])

    if selected_page == "Historic":
        historic()
    elif selected_page == "Current Hydrologic Model":
        current()
    elif selected_page == "Interactive (Path Planning)":
        interactive()


def historic():
    st.header("Brusdalsvatnet Water Quality Dashboard")
    st.title("Historic Sampling Data")
    st.markdown("### Choose a source and format to view past sampling data")

    # Radio button for ing the data source
    source = st.radio(
        " a data collection platform to display its past measurements",
        options=["Profiler Station", "USV (Maritime Robotics Otter)", "USV (OceanAlpha SL40)", "Weather Station"],
        horizontal=True)

    if source == "Profiler Station":
        # Radio button for ing the dataset
        profiler_data = st.radio(
            "Select a dataset to display",
            options=["Hourly Surface Data", "Vertical Profiles", "Current Cache"],
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
        st.markdown("#### Access to weather station data currently requires a commercial license from Volue")
        # URL of the Volue commercial platform which currently stores this data
        website_url = "https://sensordata.no/vdv.php/historical/714"

        # Use st.components.iframe to embed the website
        st.components.v1.iframe(website_url, height=600)
    else:
        st.write("Sorry, no data available in the dashboard from the USVs at this time")


# Function to show the upload CSV page
@st.cache_data
def upload_hourly_csv_page():
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
                "fDOM_QSU": "fDOM (QSU)",
                "lat": "latitude",
                "lon": "longitude",
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
                "sensorParms(11)": "fDOM (QSU)",
                "lat": "latitude",
                "lon": "longitude",
            }
            df = df.rename(columns=column_names)  # Assign column names for profiler data

        # st.write("Uploaded DataFrame:")
        # st.dataframe(df)

        # Data cleaning
        for column in df.columns:
            df[column] = df[column].apply(lambda x: np.nan if x == 'NAN' else x)
        df.iloc[:, 4:] = df.iloc[:, 4:].apply(pd.to_numeric, errors='coerce', downcast='float')

        # Convert the time column to a datetime object
        df['Timestamp'] = pd.to_datetime(df['Timestamp']).apply(lambda x: x.to_pydatetime())
        df = df.sort_values(by="Timestamp")
        df = df.drop(columns=['Record Number'])

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
    selected_variables = st.multiselect(
        "Select water quality parameters to plot",
        ["Select All"] + list(df.columns[1:12]),
    )

    # Check if "Select All" is chosen
    if "Select All" in selected_variables:
        selected_variables = list(df.columns[1:12])

    # Create a ColumnDataSource
    source = ColumnDataSource(df)
    time_difference = timedelta(hours=12)


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
            p.renderers.append(line_render)

    # Call the update_plot function with the selected variables for the first plot
    if not selected_variables:
        st.write("Please select at least one parameter to plot.")
    else:
        update_hourly(selected_variables)
        # Set plot properties
        p.title.text_font_size = "16pt"
        p.xaxis.axis_label = "Time"
        p.yaxis.axis_label = "Variable Value(s)"
        p.legend.title = "Water Quality Parameters"
        # Set the x-axis formatter to display dates in the desired format
        p.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%y/%m/%d %H:%M")
        st.bokeh_chart(p, use_container_width=True)
        st.markdown("##### Use the pan, zoom, save and reset buttons on the right to interact with the chart.")
        st.write("Find a bug? Or have an idea for how to improve the app? Please log suggestions [here].(https://github.com/russellprimeau/BrusdalsvatnetDT/issues)")


def vertical():
    ###################################################################################################################
    # Import and pre-process data

    # Read data from a CSV file into a Pandas DataFrame, skipping metadata rows
    csv_file2 = "Profiler_modem_PFL_Step.csv"  # Replace with the actual file path
    df = pd.read_csv(csv_file2, skiprows=[0, 2, 3])

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
        "sensorParms(11)": "fDOM (QSU)",
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
    for column in df.columns:
        df[column] = df[column].apply(lambda x: np.nan if x == 'NAN' else x)
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
                            "fDOM (QSU)"]
    selected_variables_p1 = st.multiselect('Select Water Quality Parameters',
                                           variables_to_plot_p1, default=[])

    # User input for depth selection
    depth_options = st.multiselect(
        "Select depths at which to plot parameters (in meters)",
        options=["1m Intervals", "2m Intervals", "5m Intervals", "10m Intervals", "20m Intervals"] + list(
            df['Depth'].unique()),
        default=["10m Intervals"]  # Default is 0m, 10m, 20m...
    )

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
                    p1.renderers.append(renderer)

        # Call the update_plot function with the selected variables for the first plot
        update_plot_p1(selected_variables_p1)

        # Show legend for the first plot
        p1.legend.title = 'Depth'
        p1.legend.label_text_font_size = '10px'
        p1.yaxis.axis_label = "Variable Value(s)"
        p1.xaxis.axis_label = "Time"
        p1.xaxis.formatter = DatetimeTickFormatter(days="%Y/%m/%d", hours="%y/%m/%d %H:%M")

        # Display the Bokeh chart for the first plot using Streamlit
        st.bokeh_chart(p1, use_container_width=True)
        st.markdown("##### Use the pan, zoom, save and reset buttons on the right to interact with the chart.")

    ###################################################################################################################
    # Plot 2: Instantaneous Vertical Profile
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### Instantaneous Profile")

    # Add a multiselect box for parameters in the second plot
    selected_variables_p2 = st.multiselect('Select Water Quality Parameters)', variables_to_plot_p1, default=[])

    # Add a multiselect box for date for the second plot
    selected_dates_p2 = st.multiselect('Select Date for Vertical Profile (search by typing YYYY-MM-DD)',
                                       df['Date'].dt.strftime('%Y-%m-%d').unique())

    profile_times = ['00:00 AM (Night)', '12:00 PM (Day)']

    # Add a multiselect box for choosing between plotting the AM or PM profiling
    nightman_dayman = st.radio("Select between night or day profile (profiles are usually collected twice per day)",
                               profile_times)  # Assuming the first column is x

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
                    p2.renderers.append(line_renderer)

            # Reverse the direction of the Y-axis
            p2.y_range = DataRange1d(start=source_plot2.data['Depth'].max() + 3, end=0)

        # Call the update_plot function with the selected variables and date for the second plot
        update_plot_p2(selected_variables_p2, selected_dates_p2)

        # Show legend for the second plot
        p2.legend.title = 'Parameters'
        p2.legend.label_text_font_size = '10px'

        # Display the Bokeh chart for the second plot using Streamlit
        st.bokeh_chart(p2, use_container_width=True)
        st.markdown("##### Use the pan, zoom, save and reset buttons on the right to interact with the chart.")
        st.write("Find a bug? Or have an idea for how to improve the app? Please log suggestions [here].(https://github.com/russellprimeau/BrusdalsvatnetDT/issues)")

def current():
    # Function to create Folium map with customizable style
    st.header("Brusdalsvatnet Water Quality Dashboard")
    st.title("Hydrological Model of Current Conditions")

    def create_map(selected_files_and_colors, map_center, zoom_level=13):
        # Create a Folium map
        m = folium.Map(location=map_center, zoom_start=zoom_level)

        # Add GeoJSON layers to the map with customizable style
        for file_key, color in selected_files_and_colors:
            geojson_path = file_key_to_path[file_key]
            gdf = gpd.read_file(geojson_path)
            style_function = lambda x: {'fillColor': color, 'color': color}
            folium.GeoJson(gdf, style_function=style_function).add_to(m)

        return m

    # Hardcoded GeoJSON file paths, colors, and map center
    geojson_paths_and_colors = {
        "Surface": (r"0m_grid_ps.geojson", "blue"),
        "10m": (r"10m_grid_ps.geojson", "green"),
        "20m": (r"20m_grid_ps.geojson", "red"),
        "30m": (r"30m_grid_ps.geojson", "black"),
        "40m": (r"40m_grid_ps.geojson", "yellow"),
        "50m": (r"50m_grid_ps.geojson", "green"),
        "60m": (r"60m_grid_ps.geojson", "green"),
        "70m": (r"70m_grid_ps.geojson", "green"),
        "80m": (r"80m_grid_ps.geojson", "green"),
        "90m": (r"90m_grid_ps.geojson", "green"),
        "100m": (r"100m_grid_ps.geojson", "green"),
    }

    hardcoded_map_center = [62.476994, 6.469730]

    # Convert the dictionary to a list for the multiselect widget
    file_key_to_path = {key: path for key, (path, _) in geojson_paths_and_colors.items()}
    multiselect_options = list(geojson_paths_and_colors.keys())

    # Display map with selectable GeoJSON overlays
    st.subheader("Displays water quality parameters on 100m x 100m grid for a given depth")

    # Multiselect widget to choose GeoJSON files
    selected_files_and_colors = st.multiselect("Select depth at which to display parameters", multiselect_options)

    chloropleth_options = ["Temperature (Celsius)", "Conductivity (microSiemens/centimeter)",
                            "Specific Conductivity (microSiemens/centimeter)", "Salinity (parts per thousand, ppt)",
                            "pH",
                            "Dissolved Oxygen (% saturation)", "Turbidity (NTU)", "Turbidity (FNU)", "fDOM (RFU)",
                            "fDOM (QSU)"]

    # Multiselect widget to choose GeoJSON files
    selected_param = st.multiselect("Select parameter by which to color-code", chloropleth_options)

    # Display map if files are selected
    if selected_files_and_colors:
        # Create Folium map with selected GeoJSON files and colors
        folium_map = create_map(
            [(file_key, geojson_paths_and_colors[file_key][1]) for file_key in selected_files_and_colors],
            map_center=hardcoded_map_center, zoom_level=13)

        # Display Folium map using folium_static
        folium_static(folium_map, width=1300)
        st.write("Find a bug? Or have an idea for how to improve the app? Please log suggestions [here].(https://github.com/russellprimeau/BrusdalsvatnetDT/issues)")
    else:
        st.info("Please select at least one depth and model parameter to display.")


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

    st.write("Find a bug? Or have an idea for how to improve the app? Please log suggestions [here].(https://github.com/russellprimeau/BrusdalsvatnetDT/issues)")

def offline_plan(csv_file_path, df_coord):
    def get_pos(lat, lng):
        return lat, lng

    m = folium.Map(location=[62.476994, 6.469730], zoom_start=13)
    m.add_child(folium.LatLngPopup())
    map = st_folium(m, use_container_width=True)

    data = None
    if map.get("last_clicked"):
        data = get_pos(map["last_clicked"]["lat"], map["last_clicked"]["lng"])

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
