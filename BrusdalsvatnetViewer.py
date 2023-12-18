import pandas as pd
import numpy as np
from SondePlotter import create_time_series_plot
from SondePlotter import create_correlate_matrix
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, firwin


csv_file1 = "Profiler_modem_SondeHourly.csv"
title1 = "Hourly Water Quality Parameters\nat 2.9m depth"
title2 = "Correlation Matrix for Water Quality Parameter2, 2020 - 2023"
title3 = "Hourly Water Quality Parameters\nat 2.9m depth, With Interpolation"

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
}
df = df.rename(columns=column_names)  # Assign column names for profiler data
df["Timestamp"] = pd.to_datetime(df["Timestamp"])  # Convert the time column to a datetime object

# Data cleaning: prevent non-numeric values from interfering with functions
for column in df.columns:
    df[column] = df[column].apply(lambda x: np.nan if x == 'NAN' else x)
df.iloc[:, 1:] = df.iloc[:, 1:].apply(pd.to_numeric, errors='coerce', downcast='float')

# # Display plots
create_time_series_plot(df, title1)  # Static plot
create_correlate_matrix(df, title2)  # Correlate matrix

#######################################################################################################################
# Extract a long segment of continuous data to apply time series analysis

# Step 1: Extract data within a time range where the series is mostly continuous
start_date = '2020-06-20'
end_date = '2020-07-23'
df.reset_index(inplace=True)

df_filtered = df[(df['Timestamp'] >= start_date) & (df['Timestamp'] <= end_date)]
print('len', len(df_filtered))

# Step 2: Check for gaps
# Sort the DataFrame based on the timestamp
df_sorted = df_filtered.sort_values(by='Timestamp')

# Calculate time differences between consecutive timestamps
time_diff = df_sorted['Timestamp'].diff()

# Check if there are any gaps in the hourly data
threshold = pd.Timedelta(hours=1)
gaps = time_diff[time_diff > threshold]

# Print the gaps
if gaps.empty:
    print("No gaps in the time series data.")
else:
    print("Gaps found in the time series data:")
    print(gaps)

#######################################################################################################################
# In fact, there is a gap in overnight data virtually every day. To enable the use of frequency analysis methods,
# which require a constant sampling rate, interpolate through these overnight gaps.

# Step 1: Find the minimum timestamp in the existing DataFrame
min_timestamp = df_sorted['Timestamp'].min()

# Step 2: Create a DataFrame with a continuous time index starting from the minimum timestamp
time_index = pd.date_range(start=min_timestamp, end=end_date, freq='1H')  # Adjust frequency as needed

# Create an empty DataFrame with the continuous time index
df_continuous_time = pd.DataFrame(index=time_index)

# Step 3: Merge your original DataFrame with the continuous time index DataFrame
df_merged = pd.merge(df_continuous_time, df, left_index=True, right_on='Timestamp', how='left')
df_merged = df_merged.set_index('Timestamp')

# Step 4: Interpolate missing values for each column
df_merged = df_merged.interpolate(method='linear')
df_backup = df_merged.copy()

# Plot cleaned data
# df_merged.reset_index(inplace=True)
# create_time_series_plot(df_merged, title3)

# For the sake of limiting the scope, the analysis will be limited to Temperature data, since this column has
# greater variance and fewer gaps than other parameters, and trends are easier to understand intuitively
var = "Temperature (Celsius)"

#######################################################################################################################
# Apply FFT and plot to analyze frequency-domain trends. Expect trends are seasonal and diurnal variation.


# Function to perform FFT and extract frequency and amplitude information
def compute_fft(data, sampling_rate):
    n = len(data)
    fft_result = np.fft.fft(data)
    frequencies = np.fft.fftfreq(n, d=1/sampling_rate)
    amplitudes = np.abs(fft_result) / n  # Normalizing by the number of points
    return frequencies[:n//2], amplitudes[:n//2]


# Function to identify the top N frequencies based on amplitude
def identify_top_frequencies(frequencies, amplitudes, duration, top_n):
    sorted_indices = np.argsort(amplitudes)[::-1]  # Sort in descending order
    top_indices = sorted_indices[:top_n]
    top_frequencies = frequencies[top_indices] / duration  # Convert to Hertz
    top_amplitudes = amplitudes[top_indices]
    for freq, amp in zip(top_frequencies, top_amplitudes):
        print(f'Frequency: {freq} Hz, Amplitude: {amp}')
    return frequencies[top_indices], amplitudes[top_indices]


# Function to reconstruct the time series from the top frequencies
def reconstruct_time_series(frequencies, amplitudes, sampling_rate, duration):
    t = np.arange(0, duration, 1/sampling_rate)
    reconstructed_signal = np.zeros_like(t)
    components = []
    for freq, amp in zip(frequencies, amplitudes):
        component = amp * np.sin(2 * np.pi * freq * t)
        components.append(component)
        reconstructed_signal += component
    return t, reconstructed_signal, components


# Choose the variable to analyze (the signal)

signal = df_merged[var].values
fs = 1  # Sampling frequency (samples per hour)

# Normalize the signal
normalized_signal = (signal - np.mean(signal)) / np.std(signal)

# Create a time vector for plotting the reconstructed signal as a check
duration = len(signal)  # Total duration of the time series
t = np.arange(0, duration, 1/fs)

# Compute FFT and extract frequency and amplitude information
frequencies, amplitudes = compute_fft(normalized_signal, fs)

# Identify the top 10 frequencies based on amplitude
top_frequencies, top_amplitudes = identify_top_frequencies(frequencies, amplitudes, duration, top_n=10)

# Reconstruct the time series using the top frequencies
t_reconstructed, signal_reconstructed, components = reconstruct_time_series(top_frequencies, top_amplitudes, fs, duration)

# Plot the original, reconstructed, and individual components of the time series
plt.figure(figsize=(12, 8))
plt.plot(t, normalized_signal * np.std(signal) + np.mean(signal), label='Original Signal', alpha=0.8)

# Plot individual components
for idx, component in enumerate(components):
    plt.plot(t_reconstructed, component * np.std(signal) + np.mean(signal), label=f'Component {idx+1}', linestyle='--', alpha=0.8)

# Plot the sum of components (reconstructed signal)
plt.plot(t_reconstructed, signal_reconstructed * np.std(signal) + np.mean(signal), label='Reconstructed Signal', linestyle='-', linewidth=2)
plt.title('Original, Reconstructed, and Individual Components of the Time Series')
plt.xlabel('Time (hours since start of series)')
plt.ylabel(var)
plt.legend()

#######################################################################################################################
# Apply a low-pass filter from the scipy package to remove anomalous data that is likely instrument error.


# Function to apply a Finite Impulse Response (FIR) low-pass filter:
def fir_lowpass_filter(data, cutoff_freq, sampling_freq, filterparam=10):
    """
    Apply a Finite Impulse Response (FIR) low-pass filter
    :param data: input signal to filter
    :param cutoff_freq: Specify a maximum frequency to pass through the filter;
            components above this frequency will be excluded from the output.
    :param sampling_freq: frequency at which the input data was sampled
    :param filterparam: the number of coefficients or 'taps' in the filter. Each tap corresponds to a coefficient
            that multiplies a specific delayed input sample. The term "tap" comes from the idea of tapping into the
            delay line at different points.number of 'taps' to include in the filter. Increasing the number of taps
            generally allows for a more selective or precise filter response, but increases computational complexity.
    :return:
    """
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    taps = firwin(filterparam, normal_cutoff, window='hamming')
    y = lfilter(taps, 1.0, data)
    return y


# Since the FIR Filter adds a notable phase shift which complicates the analysis, also try using an
# alternative filter method, in this case a Butterworth Filter from the same scipy signal processing library:
def butter_lowpass_filter(data, cutoff_freq, sampling_freq, filterparam=4):
    """
    Apply a Butterworth low-pass filter to an input signal
    :param data: input signal to filter
    :param cutoff_freq: Specify a maximum frequency to pass through the filter;
            components above this frequency will be excluded from the output.
    :param sampling_freq: frequency at which the input data was sampled
    :param filterparam: Filter order. Higher orders provide steeper roll-off but may introduce phase distortion.
    :return:
    """
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(filterparam, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y


# Since the filter functions are similar, this function abstracts the tasks of providing the
# filter functions with arguments and plotting the outputs
def filter_tune(title, df, var, cutoffs, filter, filterparam, sample_freq=fs):
    """
    Apply a filter function to a signal using a range of cutoff frequencies

    :param df: dataframe with a Timestamp column and at least one column containing a continuous time-domain signal
    :param cutoffs: an array of cutoff frequencies
    :param filter: a function which filters a signal argument and returns the filtered series
    :return: Applies a filter to a signal, then plots the original signal and iterations of the filtered signal
    """
    df_original = df.copy()
    num_columns = len(df.columns)+1
    # Apply the filter with a range of cutoff frequencies
    for idx, cutoff in enumerate(cutoffs):
        df['Filter cutoff frequency=' + str(cutoff)] = filter(df[var].values, cutoff, sample_freq, filterparam)

    # Plot the original and filtered signals
    plt.figure(figsize=(12, 6))
    df.reset_index(inplace=True)
    plt.plot(df['Timestamp'], df[var], label='Original')
    for idx, column in enumerate(df.columns[num_columns:]):
        plt.plot(df['Timestamp'], df[column], label=column, linestyle='--')

    plt.title('Signal Before and After ' + title)
    plt.xlabel('Timestamp')
    plt.ylabel('Amplitude')
    plt.legend()
    return df


# Create a range of cutoff frequencies on a log scale
fs = 1  # Sampling frequency, samples per hour
cutoffs = np.logspace(-6, -.3012, num=10)  # maximum cutoff frequency is half the sampling frequency: fs/2
# print('cutoffs', cutoffs)

df_merged_FIR3 = filter_tune('FIR Low-Pass Filter, 3 taps', df_merged, var, cutoffs, fir_lowpass_filter, filterparam=3, sample_freq=fs)
df_merged = df_backup.copy()
df_merged_FIR10 = filter_tune('FIR Low-Pass Filter, 10 taps', df_merged, var, cutoffs, fir_lowpass_filter, filterparam=10, sample_freq=fs)
df_merged = df_backup.copy()
df_merged_FIR60 = filter_tune('FIR Low-Pass Filter, 60 taps', df_merged, var, cutoffs, fir_lowpass_filter, filterparam=60, sample_freq=fs)
df_merged = df_backup.copy()


# Conclusion: Increasing the number of 'taps' or order of the FIR filter increases the phase shift linearly, which is expected based on the derivation.
# For the same input signal, the phase shift is approximately constant regardless of the cutoff frequency
# However, lower-order (fewer tap) versions of the filter are less effective at removing undesired component frequencies, and closely resemble to unfiltered input.

df_merged_butter = filter_tune('Butterworth Low-Pass Filter', df_merged, var, cutoffs, butter_lowpass_filter, filterparam=4, sample_freq=fs)
print(df_merged_butter.columns)




# By contrast, the phase shift associated with the Butterworth filter is dependent on the cutoff frequency, and is very pronounced at lower frequencies.


# Run FFT on the filtered signals and plot the spectrums alongside the original spectrum to visualize
# their effectiveness:

# Normalize the signals:

# Normalize the signal
signal_FIR3 = df_merged_FIR3.iloc[:, -8].values
signal_FIR60 = df_merged_FIR60.iloc[:, -8].values
signal_butter = df_merged_butter.iloc[:, -8].values

# normalized_FIR3 = (signal_FIR3 - np.mean(signal_FIR3)) / np.std(signal_FIR3)
# normalized_FIR60 = (signal_FIR60 - np.mean(signal_FIR60)) / np.std(signal_FIR60)
# normalized_butter = (signal_butter - np.mean(signal_butter)) / np.std(signal_butter)

normalized_FIR3 = (signal_FIR3 - np.mean(signal)) / np.std(signal)
normalized_FIR60 = (signal_FIR60 - np.mean(signal)) / np.std(signal)
normalized_butter = (signal_butter - np.mean(signal)) / np.std(signal)

duration_FIR3 = len(normalized_FIR3)  # Total duration of the time series
duration_FIR60 = len(normalized_FIR60)  # Total duration of the time series
duration_butter = len(normalized_butter)  # Total duration of the time series



# frequencies, amplitudes = compute_fft(normalized_signal, fs)

frequencies_FIR3, amplitudes_FIR3 = compute_fft(normalized_FIR3, fs)
frequencies_FIR60, amplitudes_FIR60 = compute_fft(normalized_FIR60, fs)
frequencies_butter, amplitudes_butter = compute_fft(normalized_butter, fs)


# Plot amplitudes across the full spectrum for the pre- and post-filter signals
plt.figure(figsize=(10, 5))
plt.plot(frequencies / duration, amplitudes, label='Original Signal', linestyle='-', linewidth=2)
plt.plot(frequencies_FIR3 / duration_FIR3, amplitudes_FIR3, label='FIR, 3 taps', linestyle=':', linewidth=4)
plt.plot(frequencies_FIR60 / duration_FIR60, amplitudes_FIR60, label='FIR, 60 taps', linestyle='-.', linewidth=2)
plt.plot(frequencies_butter / duration_butter, amplitudes_butter, label='Butterworth', linestyle='--', linewidth=2)
plt.title('Component frequencies and amplitudes from FFT')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid()
plt.legend()

plt.show()
