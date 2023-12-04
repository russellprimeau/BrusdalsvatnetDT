# Read hourly and depth data from CSV files. Script used as a testbed for parsing, 
# cleaning and re-formatting data of each type. Used for DTT project to create correlate matrix.

import pandas as pd
import numpy as np
from SondePlotter import hour_time_series_plot
from SondePlotter import hour_correlate_matrix
from StepPlotter import step_time_series_plot

csv_file1 = "Profiler_modem_SondeHourly.csv"
title1 = "Hourly Average of Water Quality Parameters\nat 2.9m depth"
title2 = "Correlation Matrix for Water Quality Parameter2, 2020 - 2023"
df = pd.read_csv(csv_file1, skiprows=[0, 2, 3])

# Column names from metadata
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
    "lat": "Latitude",
    "lon": "Longitude",
}
df = df.rename(columns=column_names)  # Assign column names for profiler data
df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert the time column to a datetime object

# Data cleaning
for column in df.columns:
    df[column] = df[column].apply(lambda x: np.nan if x == 'NAN' else x)
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce', downcast='float')

# Reorder columns so that water parameters are to the right of 'metadata' (time, location)
column_to_move = 'Longitude'
df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]
column_to_move = 'Latitude'
df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]
column_to_move = 'Timestamp'
df = df[[column_to_move] + [col for col in df.columns if col != column_to_move]]

# Write the DataFrame to a CSV file
# output_csv = "formatted_output.csv"  # Replace with the desired output file path
# df.to_csv(output_csv, index=False)


# Display plots
hour_time_series_plot(df, title1)  # Static plot (no animation)
# hour_correlate_matrix(df, title2)  # Correlate matrix

# Step data
######################################################################################################################
# # Read the CSV file into a Pandas DataFrame, skipping metadata rows
# csv_file2 = "Profiler_modem_PFL_Step.csv"  # Replace with the actual file path
# title3 = "Vertical Water Quality Profile"
# df_step = pd.read_csv(csv_file2, skiprows=[0, 2, 3])
#
# # Assign column names for profiler data
# column_names = {
#     "TIMESTAMP": "Timestamp",
#     "RECORD": "Record Number",
#     "PFL_Counter": "Day",
#     "CntRS232": "CntRS232",
#     "RS232Dpt": "Vertical Position1 (m)",
#     "sensorParms(1)": "Temperature (Celsius)",
#     "sensorParms(2)": "Conductivity (microSiemens/centimeter)",
#     "sensorParms(3)": "Specific Conductivity (microSiemens/centimeter)",
#     "sensorParms(4)": "Salinity (parts per thousand, ppt)",
#     "sensorParms(5)": "pH",
#     "sensorParms(6)": "Dissolved Oxygen (% saturation)",
#     "sensorParms(7)": "Turbidity (NTU)",
#     "sensorParms(8)": "Turbidity (FNU)",
#     "sensorParms(9)": "Vertical Position (m)",
#     "sensorParms(10)": "fDOM (RFU)",
#     "sensorParms(11)": "fDOM (QSU)",
#     "lat": "Latitude",
#     "lon": "Longitude",
# }
# df_step = df_step.rename(columns=column_names)
#
# # Convert the time column to a datetime object
# df_step["Timestamp"] = pd.to_datetime(df_step["Timestamp"])
# # Drop extraneous variables
# df_step = df_step.drop(columns=['Record Number', 'Day', 'CntRS232', 'Vertical Position1 (m)'])
#
# # Create new columns for date and time
# df_step['Date'] = df_step['Timestamp'].dt.date
# # print('reduced date', df_step['Date'])
# df_step['Time'] = df_step['Timestamp'].dt.time
# # print('reduced time', df_step['Time'])
#
# # Drop the original 'DateTime' column
# df_step.drop('Timestamp', axis=1, inplace=True)
# # Add rounded depth for plotting
# df_step['Rounded_Depth'] = df_step['Vertical Position (m)'].round().astype(int)
#
# # Reorder columns so that water parameters are to the right of 'metadata' (time, location)
# column_to_move = 'Longitude'
# df_step = df_step[[column_to_move] + [col for col in df_step.columns if col != column_to_move]]
# column_to_move = 'Latitude'
# df_step = df_step[[column_to_move] + [col for col in df_step.columns if col != column_to_move]]
# column_to_move = 'Rounded_Depth'
# df_step = df_step[[column_to_move] + [col for col in df_step.columns if col != column_to_move]]
# column_to_move = 'Time'
# df_step = df_step[[column_to_move] + [col for col in df_step.columns if col != column_to_move]]
# column_to_move = 'Date'
# df_step = df_step[[column_to_move] + [col for col in df_step.columns if col != column_to_move]]
#
# # Data cleaning
# for column in df_step.columns:
#     df_step[column] = df_step[column].apply(lambda x: np.nan if x == 'NAN' else x)
# df_step.iloc[:, 3:] = df_step.iloc[:, 3:].apply(pd.to_numeric, errors='coerce', downcast='float')
#
# # Write the DataFrame to a CSV file
# output_csv = "step_formatted_output.csv"  # Replace with the desired output file path
# df_step.to_csv(output_csv, index=False)
#
# step_time_series_plot(df_step, title3)  # Static plot
