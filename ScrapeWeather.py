# Logs in to the Brusdalen weather station (1818) and scrapes data using Selenium and Firefox,
# then writes data to CSV and pushes updates to GitHub. Can be called from a batch script.
# Since the script is not executed by Streamlit, requirements are not included in the project "requirements.txt" file.
# In addition to python environment requirements, additional (free, open-source) software and configurations is
# required on the machine where it is executed:
# 1. Firefox browser, in a version which accepts basic authentication credentials through the URL (125 works)
# 2. GeckoDriver, added to the system's environment variable so selenium can call it
#
# Update the following lines before running:
# os.chdir(r"C:\Users\Russell\Documents\GitHub\Thesis-Related\BrusdalsvatnetDT")  # Path to Git project directory
# my_username = ''  # Login credentials for data logger's http site
# my_password = ''  # Login credentials for data logger's http site


import time
import pandas as pd
import subprocess
import os
import logging
import psycopg2
from io import StringIO
import datetime
from datetime import datetime
from selenium import webdriver  # Probably not sufficient for login features; try selenium-wire instead
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException, StaleElementReferenceException
from dotenv import load_dotenv


load_dotenv()  # take environment variables

# Database connection parameters
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
schema_name = os.getenv("schema_name")


def copy_to_database(df, table_name):
    # Connect to PostgreSQL
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD, host=DB_HOST, port=DB_PORT
            )
        cur = conn.cursor()
    except Exception as e:
        logging.error(f"Error connecting to {DB_NAME}: {e}")

    # Ensure schema is set
    cur.execute(f"SET search_path TO {schema_name};")

    # Query for the last record in the database
    cur.execute("SELECT * FROM brusdalsvatnet.brusdalen_weather_station_hourly ORDER BY datetime DESC LIMIT 1;")

    # Fetch row
    rows = cur.fetchone()

    # Get column names
    col_names = [desc[0] for desc in cur.description]
    # Convert to Pandas DataFrame
    df_ref = pd.DataFrame([rows], columns=col_names)

    # Convert datetime column to datetime objects
    df_ref['datetime'] = pd.to_datetime(df_ref['datetime'])

    # Get the time of the latest record in the database
    latest_timestamp = df_ref['datetime'].max()

    # Drop rows from df with timestamp equal or less than the latest timestamp in the database
    df_filtered = df[df['Time'] > latest_timestamp]

    # Convert DataFrame to CSV format in memory
    output = StringIO()
    df_filtered.to_csv(output, sep="\t", index=False, header=False, na_rep="NULL")  # Use tab as delimiter
    output.seek(0)  # Move cursor to the start

    # Use copy_from to insert data
    cur.copy_from(output, table_name, sep="\t", null="NULL")

    # Commit and close
    try:
        conn.commit()
        cur.close()
        conn.close()
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
        logging.info(f"Successfully wrote new lines to {table_name} at {current_time}")
    except Exception as e:
        logging.error(f"Error in commit to PostgreSQL database table {table_name}: {e}")


def run_command(command):
    try:
        subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        # print(f"Command succeeded: {' '.join(command)}")
        logging.info(f"Command succeeded: {' '.join(command)}")
    except subprocess.CalledProcessError as e:
        # print(f"Command failed: {' '.join(command)}. Error message: {e.stderr}")
        logging.error(f"Command failed: {' '.join(command)}. Error message: {e.stderr}")


def push():
    # Add all changes to the staging area
    run_command(['git', 'add', '.'])

    # Get list of modified files using git diff
    modified_files = subprocess.run(["git", "diff", "--cached", "--name-only"], capture_output=True,
                                    text=True).stdout.strip()

    logging.info(f"modified files: {modified_files}")

    # Construct commit message (optional)
    if modified_files:
        commit_message = f"Changes to: {modified_files}"
    else:
        commit_message = "No changes detected"  # Optional for clarity

    # Commit the changes
    try:
        run_command(['git', 'commit', '-m', commit_message])
    except Exception as e:
        logging.error(f"Error in committing to GitHub: {e}")

    # Push the changes to the remote repository
    try:
        run_command(['git', 'push', 'origin', 'main'])
    except Exception as e:
        logging.error(f"Error in push to GitHub: {e}")


def get_last_line(csv_file):
    """
  Reads the entire CSV file and returns the last line as a dataframe.

  Args:
      csv_file (str): Path to the CSV file.

  Returns:
      pandas.DataFrame: The last line of the CSV file as a dataframe.
  """
    # Read the CSV file into a dataframe
    last_df = pd.read_csv(csv_file, sep=";", decimal=",", parse_dates=['Time'], date_format='%Y-%m-%dT%H:%M:%S', header=0)

    # Return the last row (using negative indexing) as a single-row dataframe
    return last_df.iloc[-1:]


def write(df, destination):
    """
  Append the new data to the end of the CSV
  """
    try:
        # Filter out times already in the CSV
        ref = get_last_line(destination)
        filtered_df = df[df['Time'] > ref.iloc[0, 0]].copy()

        # Convert the 'Time' column to ISO 8601 format strings (a long method to avoid some strange Python errors)
        iso_format = '%Y-%m-%dT%H:%M:%S'
        filtered_df['Time_str'] = filtered_df['Time'].dt.strftime(iso_format)
        filtered_df.drop('Time', axis=1, inplace=True)
        cols = ['Time_str'] + [col for col in filtered_df.columns if col != 'Time_str']
        filtered_df = filtered_df[cols]

        try:
            filtered_df.to_csv(destination, mode='a', index=False, header=False, sep=";", decimal=",")
        except Exception as e:
            logging.error(f"Couldn't write to file, with error {e}")
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
        logging.info(f"Successfully appended records to {destination} at {current_time}")
    except Exception as e:
        now = datetime.now()
        current_time = now.strftime("%Y-%m-%d %H:%M:%S")  # Format: YYYY-MM-DD HH:MM:SS
        logging.error(f"Failed to appended records to {destination} at {current_time} with error {e}")


def scrape_and_clean():

    def click(driver, element_xpath):
        try:
            element = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, element_xpath)))
            element.click()
        except Exception as e:
            logging.error(f"Couldn't find button: {element_xpath}, with error {e}")

    def scrape_table_with_xpath(driver, head_row_xpath, body_xpath, columns):
        try:
            head_data = []
            time.sleep(10)  # This hack seems to be much more effective that WebDriverWait to ensure table is loaded
            # 'while' loop is to prevent script from reading the default table, by checking for the correct # of columns

            while len(head_data) != columns:
                try:
                    head_row = WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, head_row_xpath)))
                    head_cells = head_row.find_elements(By.TAG_NAME, "th")
                    head_data = [cell.text for cell in head_cells]
                    # print('head_data', len(head_data), head_data)
                    if len(head_data) == columns:
                        break
                except StaleElementReferenceException as e:
                    # If a stale element is encountered, continue the loop to retry
                    logging.error(f"Stale element in header: {e}")
                    continue


            counter = 0
            restart = 0
            table_data = []
            while True:
                try:
                    # On each new iteration in the "while" condition, reload the table to replace stale elements
                    body = WebDriverWait(driver, 60).until(EC.presence_of_element_located((By.XPATH, body_xpath)))
                    rows = body.find_elements(By.TAG_NAME, "tr")  # Find the rows within the table body

                    # rows = wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, 'tr')))
                    # print(f'table returns: {table}, table body {tbody}')
                    # print(f'{len(rows)} rows')

                    # Loop through the rows in the body (should be 25 for "Time" table)
                    for index, _ in enumerate(rows):
                        # in_rows = table.find_elements(By.TAG_NAME, 'tr')  # Find rows in the table body
                        # in_rows = wait.until(EC.presence_of_all_elements_located((By.TAG_NAME, 'tr')))

                        row = rows[index+restart]
                        # print(f'In attempt {counter} at index {index}, combined {index + restart}, '
                        #       f'with {len(table_data)} rows written to table_data')
                        cells = row.find_elements(By.TAG_NAME, 'td')
                        row_data = [cell.text for cell in cells]
                        if row_data:
                            table_data.append(row_data)
                        # print(f"table_data row {index + restart}: {row_data}")
                        if (index+restart) == len(rows)-1:
                            break
                    break  # Exit the loop if all rows are processed successfully
                except StaleElementReferenceException:
                    # If a stale element is encountered, continue the loop to retry, starting from the previous row
                    restart = len(table_data)
                    counter += 1
                    continue
            # print('head_data in:', len(head_data), '\n', head_data)
            # print('table_data in:', len(table_data), '\n', table_data)

            return head_data, table_data
        except (TimeoutException, NoSuchElementException) as e:
            logging.error(f"Attempt to scrape table ended with Error: {e}")
            return None

    # Pass Authentication in URL:
    # Method is deprecated in most browsers, but still works in Firefox as of 125.03, although it may
    # require additional manual inputs the first time.
    # Credentials:
    my_username = '*'
    my_password = '*'
    full_url = 'http://89.9.0.217/tables.html'
    target_url = '89.9.0.217/tables.html'
    simple_url = f'http://{my_username}:{my_password}@{target_url}'

    # Characteristics of the table to be scraped, including full XPATHs
    time_xpath_out =        '/html/body/div[1]/div[2]/ul/li[10]/a'  # XPATH of the button for opening the dataset
    table_view_xpath_out =  '/html/body/div[1]/div[3]/ul/li[2]/a'  # XPATH of button for table (not record) view
    head_row_xpath_out =    '/html/body/div[1]/div[3]/div[2]/div[2]/div[2]/table/thead/tr'
    body_xpath_out =        '/html/body/div[1]/div[3]/div[2]/div[2]/div[2]/table/tbody'
    columns_out: int = 25

    options = FirefoxOptions()
    options.add_argument("--headless")
    driver_out = webdriver.Firefox(options=options)

    try:
        driver_out.get(simple_url)
        logging.info(f"Successfully opened {full_url} using GeckoDriver")

        click(driver_out, time_xpath_out)
        click(driver_out, table_view_xpath_out)
        head_data_out, table_data_out = scrape_table_with_xpath(driver_out, head_row_xpath_out, body_xpath_out,
                                                                columns_out)

        if table_data_out:  # Check if data was retrieved successfully
            scraped_df = pd.DataFrame(table_data_out)
            scraped_df.columns = head_data_out
            scraped_df = scraped_df.drop(['Timestamp', 'St_info', 'Par_60_min'], axis=1)

            # Convert timestamps to ISO 8601 time format
            original_format = '%Y%m%d%H%M%S'

            # Convert time column to datetime objects (assuming it's named 'date_time') and sort accordingly
            scraped_df['Tid_str'] = pd.to_datetime(scraped_df['Tid_str'], format=original_format)
            scraped_df.sort_values(by='Tid_str', inplace=True)  # Sort by time

            # Rename columns to match the sensordata.no output
            column_mapping = {"Tid_str": "Time", "AA": "1818_time: AA[mBar]", "Batt_V": "1818_time: Batt_V[V]",
                              "DD_l": "1818_time: DD Retning[°]", "DX_l": "1818_time: DX_l[°]",
                              "FF_l": "1818_time: FF Hastighet[m/s]", "FG_l": "1818_time: FG_l[m/s]",
                              "FG_tid_l": "1818_time: FG_tid_l[N/A]", "FX_l": "1818_time: FX Kast[m/s]",
                              "FX_tid_l": "1818_time: FX_tid_l[N/A]", "PO": "1818_time: PO Trykk stasjonshøyde[mBar]",
                              "PP": "1818_time: PP[mBar]", "PR": "1818_time: PR Trykk redusert til havnivå[mBar]",
                              "QLI_Avg": "1818_time: QLI Langbølget[W/m2]", "QNH": "1818_time: QNH[mBar]",
                              "QSI_Avg": "1818_time: QSI Kortbølget[W/m2]", "RR_1": "1818_time: RR_1[mm]",
                              "TA_a": "1818_time: TA Middel[°C]", "TA_a_Max": "1818_time: TA_a_Max[°C]",
                              "TA_a_Min": "1818_time: TA_a_Min[°C]", "UU": "1818_time: UU Luftfuktighet[%RH]"}
            scraped_df.rename(columns=column_mapping, inplace=True)

            # Reorder columns to match the sensordata.no output
            column_order = ["Time", "1818_time: AA[mBar]", "1818_time: Batt_V[V]", "1818_time: DD Retning[°]",
                     "1818_time: DX_l[°]", "1818_time: FF Hastighet[m/s]", "1818_time: FG_l[m/s]",
                     "1818_time: FG_tid_l[N/A]", "1818_time: FX Kast[m/s]", "1818_time: FX_tid_l[N/A]",
                     "1818_time: PO Trykk stasjonshøyde[mBar]", "1818_time: PP[mBar]",
                     "1818_time: PR Trykk redusert til havnivå[mBar]", "1818_time: QLI Langbølget[W/m2]",
                     "1818_time: QNH[mBar]", "1818_time: QSI Kortbølget[W/m2]", "1818_time: RR_1[mm]",
                     "1818_time: TA Middel[°C]", "1818_time: TA_a_Max[°C]", "1818_time: TA_a_Min[°C]",
                     "1818_time: UU Luftfuktighet[%RH]"]
            scraped_df = scraped_df[column_order]
            scraped_df.reset_index(drop=True, inplace=True)

            # Convert data from strings to numeric data types
            cols_to_convert = ["1818_time: AA[mBar]", "1818_time: Batt_V[V]", "1818_time: DD Retning[°]",
                     "1818_time: DX_l[°]", "1818_time: FF Hastighet[m/s]", "1818_time: FG_l[m/s]",
                     "1818_time: FG_tid_l[N/A]", "1818_time: FX Kast[m/s]", "1818_time: FX_tid_l[N/A]",
                     "1818_time: PO Trykk stasjonshøyde[mBar]", "1818_time: PP[mBar]",
                     "1818_time: PR Trykk redusert til havnivå[mBar]", "1818_time: QLI Langbølget[W/m2]",
                     "1818_time: QNH[mBar]", "1818_time: QSI Kortbølget[W/m2]", "1818_time: RR_1[mm]",
                     "1818_time: TA Middel[°C]", "1818_time: TA_a_Max[°C]", "1818_time: TA_a_Min[°C]",
                     "1818_time: UU Luftfuktighet[%RH]"]

            # Remove ',' used as thousands separators
            def remove_comma(col):
                """Removes commas from a Pandas Series."""
                return col.str.replace(',', '')
            for col in cols_to_convert:
                scraped_df[col] = remove_comma(scraped_df[col])

            # Convert data from strings to datetime and numeric types for manipulation
            scraped_df[cols_to_convert] = scraped_df[cols_to_convert].apply(pd.to_numeric)
            scraped_df['Time'] = scraped_df['Time'].astype('datetime64[ns]')
        else:
            logging.error("Scraping failed.")
            scraped_df = pd.DataFrame()
    finally:
        # Close the browser window
        driver_out.quit()
    return scraped_df


if __name__ == '__main__':
    # Change directory to project location
    os.chdir(r"C:\Users\russelbp\GitHub\BrusdalsvatnetDT")  # Remote desktop
    # os.chdir(r"C:\Users\Russell\Documents\GitHub\Thesis-Related\BrusdalsvatnetDT")  # Local

    data_file = "All_Time.csv"

    # Create log for debugging automation
    log_file = "Scheduled_ScrapeWeather.log"  # Define the log file path (optional, change filename if needed)
    logging.basicConfig(filename=log_file, level=logging.INFO)  # Configure logging

    # Pull changes from the remote repository
    run_command(['git', 'pull', 'origin', 'main'])

    # Scrape data from online, reformat, and write to CSV file
    new_lines = scrape_and_clean()
    # logging.info('\n' + new_lines.to_string())  # write df to log file
    to_csv = new_lines.drop(columns=["1818_time: Batt_V[V]"], axis=1)
    write(to_csv, data_file)

    table_name = 'brusdalen_weather_station_hourly'
    try:
        # Code that may raise an exception
        copy_to_database(new_lines, table_name)
    except Exception as e:
        logging.error(f"Error in push to PostgreSQL database: {e}")
    # Push changes to remote origin
    push()