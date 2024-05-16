import csv
from datetime import datetime
import pandas as pd


def convert_to_minutes_relative(df_convo, datetime_col, ref_time):
    """
  Converts datetime values in a DataFrame column to minutes relative to a reference time.

  Args:
      df: The pandas DataFrame containing the datetime column.
      datetime_col: The name of the column containing datetime values.
      reference_time: A pandas.Timestamp object representing the reference time.

  Returns:
      A new DataFrame with the same columns as the input DataFrame, with the specified
      column containing the datetime values converted to minutes relative to the reference time.
  """

    # Ensure reference time is a pandas.Timestamp
    if not isinstance(ref_time, pd.Timestamp):
        ref_time = pd.Timestamp(ref_time)

    # Convert datetime column to timedelta (difference from reference time)
    df_convo['time_diff'] = df_convo[datetime_col] - ref_time

    # Convert timedelta to minutes (assuming microseconds are not relevant)
    df_convo['time_diff_minutes'] = df_convo['time_diff'] / pd.Timedelta(minutes=1)

    # Drop the original datetime column if desired
    df_convo = df_convo.drop(datetime_col, axis=1)
    df_convo = df_convo.drop('time_diff', axis=1)
    return df_convo


def filter_csv_by_date_range(start_date, end_date, reference_time):
    """
  Reads a CSV file, filters data by date range and specified columns,
  performs optional data manipulation, and writes the result to a new CSV file.

  Args:
    start_date: String representing the start date in YYYY-MM-DD format.
    end_date: String representing the end date in YYYY-MM-DD format.
    reference_time: time relative to which output file time is expressed (in minutes)
  """
    input_file = "All_time.csv"
    wind_output = "windxy.tim"
    met_output = "FlowFM_meteo.tim"
    wind_columns = ["Average wind speed (m/s)", "Hourly average wind direction (°)"]
    met_columns = ["Average humidity (% relative humidity)", "Avg. Temp (°C)", "Cloud cover",
                   "Shortwave (solar) radiation (W/m2)"]
    
    # Read data into a pandas DataFrame
    df = pd.read_csv(input_file, header=0, sep=';', decimal=',')

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
                          "Instantaneous sea-level atmospheric pressure (mBar)", "Instantaneous temperature (°C)"])

    # Filter data by date range
    filtered_df = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]

    # Calculate "average" temperature:
    filtered_df["Avg. Temp (°C)"] = (df['Hourly maximum temperature (°C)'] + df['Hourly minimum temperature (°C)']) / 2

    # Add (missing but irrelevant) cloud cover data:
    filtered_df["Cloud cover"] = 0

    # Get rid of negative radiation values
    filtered_df.loc[filtered_df['Shortwave (solar) radiation (W/m2)'] < 0, 'Shortwave (solar) radiation (W/m2)'] = 0

    # Convert the time column to minutes relative to the reference date
    filtered_df = convert_to_minutes_relative(filtered_df, "Timestamp", reference_time)

    col_to_wind = ["time_diff_minutes", *wind_columns]
    wind_df = filtered_df.assign(**{col: filtered_df[col].apply(lambda x: f'{x:.7e}') for col in col_to_wind})

    # Select specified columns
    wind_df = wind_df[col_to_wind]

    # Write filtered data to a new CSV file
    wind_df.to_csv(wind_output, index=False, header=False, sep=' ')

    # Repeat with met data
    col_to_met = ["time_diff_minutes", *met_columns]
    met_df = filtered_df.assign(**{col: filtered_df[col].apply(lambda x: f'{x:.7e}') for col in col_to_met})
    # Select specified columns
    met_df = met_df[col_to_met]
    # Write filtered data to a new CSV file
    met_df.to_csv(met_output, index=False, header=False, sep=' ')


def main():  
    start_date = datetime.strptime('2024-05-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end_date = datetime.strptime('2024-05-10 00:00:00', '%Y-%m-%d %H:%M:%S')
    reference_time = pd.Timestamp('2024-05-01 00:00:00')
    filter_csv_by_date_range(start_date, end_date, reference_time)

if __name__ == "__main__":
    main()
