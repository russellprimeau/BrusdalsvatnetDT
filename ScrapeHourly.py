# Pulls data from Profiler IP and writes to local mySQL database

import requests
from bs4 import BeautifulSoup
import pandas as pd
import pymysql
from datetime import datetime


def scrape_and_save_data():
    # URL of the website
    url = "http://89.9.10.123/?command=NewestRecord&table=SondeHourly"

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
        df = pd.DataFrame(data[0:], columns=data[0]).transpose()

        # Set the first row as column headers
        df.columns = df.iloc[0]

        # Drop the first row (which is now the header row)
        df = df[1:]

        current_record_tag = soup.find('b', string=lambda t: t and 'Current Record:' in t)
        record_date_tag = soup.find('b', string=lambda t: t and 'Record Date:' in t)

        try:
            # Check if the tags are found before accessing their next siblings
            if current_record_tag:
                current_record = current_record_tag.next_sibling.strip()
            else:
                current_record = "Not Found"

            if record_date_tag:
                record_date = record_date_tag.next_sibling.strip()
            else:
                record_date = "Not Found"

            # Add Record date and Current record as the first two columns
            df.insert(0, 'Record date', record_date)
            df.insert(1, 'Current record', current_record)

        except AttributeError as e:
            print(f"Error extracting data from the website: {e}")

        # Column names from metadata
        column_names = {
            "Record date": "Timestamp",
            "Current record": "Record_Number",
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
        df["Timestamp"] = df["Timestamp"].dt.strftime(
            '%Y-%m-%d %H:%M:%S')  # Reformat Timestamp to MySQL datetime format

        # Data cleaning
        for column in df.columns[0:2]:
            df[column] = df[column].apply(lambda x: 0 if x == 'NAN' else x)
        for column in df.columns[2:]:  # Apply rounding to columns 2+
            df[column] = df[column].apply(lambda x: 0 if x == 'NAN' else round(float(x), 3))


if __name__ == "__main__":
    scrape_and_save_data()
