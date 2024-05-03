# Pulls data from Profiler IP and writes to local mySQL database

import requests
from bs4 import BeautifulSoup
import pandas as pd
import subprocess
import os
import csv
from datetime import datetime
import logging


def scrape_and_clean():
    # URL of the website
    url = "http://89.9.10.123/?command=TableDisplay&table=PFL_Step&records=24"

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

        # print(df)

        # current_record_tag = soup.find('b', string=lambda t: t and 'Current Record:' in t)
        # record_date_tag = soup.find('b', string=lambda t: t and 'Record Date:' in t)
        latitude = 62.474464
        longitude = 6.461324

        # try:
        #     # Check if the tags are found before accessing their next siblings
        #     if current_record_tag:
        #         current_record = current_record_tag.next_sibling.strip()
        #     else:
        #         current_record = "Not Found"
        #
        #     if record_date_tag:
        #         record_date = record_date_tag.next_sibling.strip()
        #     else:
        #         record_date = "Not Found"

        # Add Record date and Current record as the first two columns
        # df.insert(0, 'Record date', record_date)
        # df.insert(1, 'Current record', current_record)


        # except AttributeError as e:
        #     print(f"Error extracting data from the website: {e}")


        # Column names from metadata
        column_names = {
            "TimeStamp": "Timestamp",
            "Record": "Record",
            "PFL_Counter": "PFL_Counter",
            "_CntRS232": "_CntRS232",
            "_RS232Dpt": "_RS232Dpt",
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
        df['Record'] = df['Record'].astype(int)
        df['PFL_Counter'] = df['PFL_Counter'].astype(int)
        df['_CntRS232'] = df['_CntRS232'].astype(int)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert the time column to a datetime object
        iso_format = '%Y-%m-%dT%H:%M:%S'
        df['Timestamp'] = df['Timestamp'].dt.strftime(iso_format)  # Convert datetime objects to ISO 8601 format strings

        # Data cleaning
        for column in df.columns[0:4]:
            df[column] = df[column].apply(lambda x: 0 if x == 'NAN' else x)
        for column in df.columns[4:]:  # Apply rounding to columns 2+
            df[column] = df[column].apply(lambda x: 0 if x == 'NAN' else round(float(x), 3))

        df.insert(len(df.columns), 'Latitude', latitude)
        df.insert(len(df.columns), 'Longitude', longitude)
    else:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
        logging.info(f"Unsuccessful attempt to connect at {current_time}. Request returned: {response.status_code}")
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
        logging.info(f"Failed to appended records to {destination} at {current_time}")

def push_to_remote(project_dir, filename, branch_name="main"):
    command = ["git", "push", "origin", branch_name]
    # Change directory to project location
    os.chdir(project_dir)

    # # Add specific file for commit
    # subprocess.run(["git", "add", filename], check=True)

    # Add all modified files for commit
    subprocess.run(["git", "add", filename], check=True)  # Raise error if fails

    # Get list of modified files using git diff
    modified_files = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True,
                                    text=True).stdout.strip()

    # Construct commit message (optional)
    if modified_files:
        commit_message = f"Changes to: {modified_files}"  # Replace with your preferred format
    else:
        commit_message = "No changes detected"  # Optional for clarity

    # Commit all selected files
    subprocess.run(["git", "commit", "-m", commit_message], check=True)  # Optional

    # Push commits to remote repository (by default, main)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()
    logging.info(f"Push output: {output.decode()}, with error code {error.decode()}")
    # print("Output: ", output.decode())
    # print("Error: ", error.decode())


if __name__ == "__main__":
    # Set output file
    data_file = "Profiler_modem_PFL_Step.csv"

    # Create log for debugging automation
    log_file = "Scheduled_ScrapeStep.log"  # Define the log file path (optional, change filename if needed)
    logging.basicConfig(filename=log_file, level=logging.INFO)  # Configure logging

    # Execute functions:
    new_lines = scrape_and_clean()
    write(new_lines, data_file)
    push_to_remote(r"C:\Users\russelbp\GitHub\BrusdalsvatnetDT", data_file)
