import requests
from bs4 import BeautifulSoup
import pandas as pd
import subprocess
import os
import logging
from datetime import datetime


def run_command(command):
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(f"Command succeeded: {' '.join(command)}")
        logging.info(f"Command succeeded: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {' '.join(command)}. Error message: {e.stderr}")
        logging.error(f"Command failed: {' '.join(command)}. Error message: {e.stderr}")


def push():
    # Add all changes to the staging area
    run_command(['git', 'add', '.'])

    # Get list of modified files using git diff
    modified_files = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True,
                                    text=True).stdout.strip()

    # Construct commit message (optional)
    if modified_files:
        commit_message = f"Changes to: {modified_files}"  # Replace with your preferred format
    else:
        commit_message = "No changes detected"  # Optional for clarity

    # Commit the changes
    run_command(['git', 'commit', '-m', commit_message])

    # Push the changes to the remote repository
    run_command(['git', 'push', 'origin', 'main'])


def scrape_and_clean():
    # URL of the website
    url = "http://89.9.10.123/?command=TableDisplay&table=SondeHourly&records=24"

    # Send a GET request to the website
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the table on the page (you may need to inspect the HTML to find the correct tag and class)
        table = soup.find('table')

        # Extract data from the table and create a Pandas DataFrame
        data = []
        for row in table.find_all('tr'):
            cols = row.find_all(['td', 'th'])
            cols = [col.text.strip() for col in cols]
            data.append(cols)

        # Create a DataFrame from the scraped data with transposition
        df = pd.DataFrame(data[0:], columns=data[0])

        # Set the first row as column headers
        df.columns = df.iloc[0]

        # Drop the first row (which is now the header row)
        df = df[1:]

        latitude = 62.474464
        longitude = 6.461324

        # Add latitude and longitude columns
        df.insert(13, 'Latitude', latitude)
        df.insert(14, 'Longitude', longitude)

        # Column names from metadata
        column_names = {
            "TimeStamp": "Timestamp",
            "Record": "Record_Number",
            "sensorParms(1)": "Temperature",
            "sensorParms(2)": "Conductivity",
            "sensorParms(3)": "Specific_Conductivity",
            "sensorParms(4)": "Salinity",
            "sensorParms(5)": "pH",
            "sensorParms(6)": "DO",
            "sensorParms(7)": "Turbidity_NTU",
            "sensorParms(8)": "Turbidity_FNU",
            "sensorParms(9)": "Position",
            "sensorParms(10)": "fDOM_RFU",
            "sensorParms(11)": "fDOM_QSU",
        }
        df = df.rename(columns=column_names)  # Assign column names for profiler data
        df['Record_Number'] = df['Record_Number'].astype(int)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert the time column to a datetime object
        iso_format = '%Y-%m-%dT%H:%M:%S'
        df['Timestamp'] = df['Timestamp'].dt.strftime(iso_format)  # Convert datetime objects to ISO 8601 format strings

        # Data cleaning
        for column in df.columns[0:2]:
            df[column] = df[column].apply(lambda x: 0 if x == 'NAN' else x)
        for column in df.columns[2:13]:  # Apply rounding to columns 2+
            df[column] = df[column].apply(lambda x: 0 if x == 'NAN' else round(float(x), 3))
    else:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
        logging.error(f"Unsuccessful attempt to connect at {current_time}. Request returned: {response.status_code}")
        df = pd.DataFrame()
    return df


def get_last_line(csv_file):
    """
    Reads the entire CSV file and returns the last line as a dataframe.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: The last line of the CSV file as a dataframe.
    """
    # Read the CSV file into a dataframe
    df = pd.read_csv(csv_file)

    # Return the last row (using negative indexing) as a single-row dataframe
    return df.iloc[-1:]


def write(df, destination):
    """
    Append the new data to the end of the CSV
    """
    try:
        ref = get_last_line(destination)
        filtered_df = df[df['Timestamp'] > ref.iloc[0, 0]]
        filtered_df.to_csv(destination, mode='a', index=False, header=False)
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
        logging.info(f"Successfully appended records to {destination} at {current_time}")
    except:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
        logging.error(f"Failed to appended records to {destination} at {current_time}")


if __name__ == '__main__':
    # Change directory to project location
    os.chdir(r"C:\Users\russelbp\GitHub\BrusdalsvatnetDT")

    data_file = "Profiler_modem_SondeHourly.csv"

    # Create log for debugging automation
    log_file = "Scheduled_ScrapeHourly.log"  # Define the log file path (optional, change filename if needed)
    logging.basicConfig(filename=log_file, level=logging.INFO)  # Configure logging

    # Pull changes from the remote repository
    run_command(['git', 'pull', 'origin', 'main'])

    new_lines = scrape_and_clean()
    write(new_lines, data_file)

    push()
