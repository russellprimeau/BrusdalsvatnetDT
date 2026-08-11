# Water Quality & Weather Data QA/QC Processing Summary

## Overview
Three data streams are preprocessed before storage/use. Minimal threshold-based QA/QC is applied to water quality sensors; weather data undergoes extensive validation.

---

## 1. Profiler_modem_SondeHourly.csv
**Source:** Profiler modem (YSI sonde), scraped hourly
**Location:** ScrapeHourly.py (lines 171–174)

### Sensor Channels & Processing
| Channel | Raw → Processed | QA/QC Rule |
|---------|-----------------|-----------|
| Temperature | float | Round to 3 decimals |
| Conductivity | float | Round to 3 decimals |
| Specific Conductivity | float | Round to 3 decimals |
| Salinity | float | Round to 3 decimals |
| pH | float | Round to 3 decimals |
| DO (mg/L or %) | float | Round to 3 decimals |
| Turbidity NTU | float | Round to 3 decimals |
| Turbidity FNU | float | Round to 3 decimals |
| Position/Depth | float | Round to 3 decimals |
| fDOM RFU | float | Round to 3 decimals |
| fDOM QSU | float | Round to 3 decimals |
| All columns | "NAN" string | Replace with 0 |

### Pseudocode
```
for each record:
  for col in [Temperature, Conductivity, ..., fDOM_QSU]:
    if col == "NAN":
      col ← 0
    else:
      col ← round(float(col), 3)
```

---

## 2. Profiler_modem_PFL_Step.csv
**Source:** Profiler modem (step/profile mode), scraped at varying intervals
**Location:** ScrapeStep.py (lines 179–182)

### Sensor Channels & Processing
| Channel | Raw → Processed | QA/QC Rule |
|---------|-----------------|-----------|
| Temperature | float | Round to 3 decimals |
| Conductivity | float | Round to 3 decimals |
| Specific Conductivity | float | Round to 3 decimals |
| Salinity | float | Round to 3 decimals |
| pH | float | Round to 3 decimals |
| DO | float | Round to 3 decimals |
| Turbidity NTU | float | Round to 3 decimals |
| Turbidity FNU | float | Round to 3 decimals |
| Position/Depth | float | Round to 3 decimals |
| fDOM RFU | float | Round to 3 decimals |
| fDOM QSU | float | Round to 3 decimals |
| All columns | "NAN" string | Replace with 0 |

### Pseudocode
```
for each record:
  for col in [Timestamp, Record, PFL_Counter, _CntRS232]:
    if col == "NAN":
      col ← 0
  for col in [_RS232Dpt, Temperature, ..., fDOM_QSU]:
    if col == "NAN":
      col ← 0
    else:
      col ← round(float(col), 3)
```

---

## 3. All_time.csv (Advanced Threshold-Based QA/QC)
**Source:** Brusdalen weather station (1818), scraped hourly
**Location:** OfflineUtility.py, filter_met() (lines 329–369)

### Weather Parameter Validation & Processing
| Parameter | Valid Range | Out-of-Range Action | Rounding |
|-----------|-------------|-------------------|----------|
| Timestamp | 2000-01-01 to 2099-12-31 | → np.nan | — |
| Atmospheric Pressure (station) | 860–1080 mBar | → np.nan | — |
| Wind Direction (hourly avg) | 0–360° | → np.nan | — |
| Wind Speed (avg) | 0–100 m/s | → np.nan | — |
| Wind Speed (3s gust max) | 0–100 m/s | → np.nan | — |
| Wind Speed (10min gust max) | 0–100 m/s | → np.nan | — |
| Pressure Differential (3hr span) | 0–50 mBar | → np.nan | — |
| Longwave (IR) Radiation | 0–750 W/m² | → np.nan | — |
| Shortwave (Solar) Radiation | 0–900 W/m² | → np.nan; also: if <0 then 0 | — |
| Precipitation | 0–50 mm/hr | → np.nan | — |
| Temperature (hourly max/min) | –40 to +40 °C | → np.nan | — |
| Humidity (relative) | 0–100 %RH | → np.nan | — |
| String "NAN" | — | → np.nan | — |

### Pseudocode
```
error_conditions = {
  "Timestamp": (df['Timestamp'] < 2000-01-01) OR (df['Timestamp'] > 2099-12-31),
  "Atmospheric Pressure": (val < 860) OR (val > 1080),
  "Wind Direction": (val < 0) OR (val > 360),
  "Wind Speed": (val < 0) OR (val > 100),
  "Pressure Differential": (val < 0) OR (val > 50),
  "IR Radiation": (val < 0) OR (val > 750),
  "Solar Radiation": (val < 0) OR (val > 900),
  "Precipitation": (val < 0) OR (val > 50),
  "Temperature": (val < -40) OR (val > 40),
  "Humidity": (val < 0) OR (val > 100)
}

for col, condition in error_conditions.items():
  df.loc[condition, col] = np.nan

# Special case: set negative solar radiation to 0 (not flagged as error)
df['Solar Radiation'] = where(df['Solar Radiation'] < 0, 0, df['Solar Radiation'])
```

---

## 4. Advanced Profiler QA/QC (Applied in Delft3D Forcing Generation)
**Location:** OfflineUtility.py, gen_forcing() (lines 448–471)
**Applied to:** Profiler_modem_SondeHourly.csv data before using as model input

### Advanced Water Quality Thresholds
| Channel | Valid Range | Out-of-Range Action |
|---------|-------------|-------------------|
| Temperature | 1–25 °C | → np.nan |
| Conductivity | 0–45 µS/cm | → np.nan |
| Specific Conductivity | >1 µS/cm | → np.nan |
| Salinity | >0 ppt | → np.nan |
| pH | 2–12 | → np.nan |
| DO (% saturation) | 10–120 % | → np.nan |
| Turbidity NTU | >0 | → np.nan |
| Turbidity FNU | >0 | → np.nan |
| fDOM RFU | 0–100 | → np.nan |
| fDOM QSU | 0–300 | → np.nan |
| Latitude | –90 to +90 | → np.nan |
| Longitude | –180 to +180 | → np.nan |

### Pseudocode
```
error_conditions = {
  "Temperature": (val < 1) OR (val > 25),
  "Conductivity": (val < 0) OR (val > 45),
  "Specific Conductivity": (val < 1),
  "Salinity": (val < 0),
  "pH": (val < 2) OR (val > 12),
  "DO": (val < 10) OR (val > 120),
  "fDOM (RFU)": (val < 0) OR (val > 100),
  "fDOM (QSU)": (val < 0) OR (val > 300),
  "Latitude": (val < -90) OR (val > 90),
  "Longitude": (val < -180) OR (val > 180)
}

for col, condition in error_conditions.items():
  df.loc[condition, col] = np.nan

# Drop rows where critical columns are null
df = df.dropna(subset=['Temperature'])
```

---

## Data Flow Summary

```
Raw Sensor Data (HTTP scrape)
        ↓
ScrapeHourly.py / ScrapeStep.py
├─ NAN → 0
└─ Round to 3 decimals
        ↓
Store: Profiler_modem_SondeHourly.csv / Profiler_modem_PFL_Step.csv
        ↓
        ├─ (Direct use in database)
        │
        └─ (Optional) OfflineUtility.py gen_forcing()
           └─ Apply advanced thresholds (temperature 1–25°C, etc.)
           └─ Drop rows with np.nan in critical columns
           └─ Output: Delft3D model input

Raw Weather Data (Selenium scrape)
        ↓
ScrapeWeather.py
├─ Column rename & reorder
├─ Convert strings to numeric
└─ Parse datetime
        ↓
Store: All_time.csv
        ↓
        └─ (Optional) OfflineUtility.py filter_met()
           └─ Apply threshold validation (860–1080 mBar pressure, etc.)
           └─ Flag out-of-range as np.nan
           └─ Output: Delft3D model input (wind, heat, rainfall .tim files)
```

---

## Summary: Frequencies & Alteration Impact

| Data Stream | Scrape Frequency | Primary QA/QC | Typical NaN Replacement Rate |
|-------------|------------------|---------------|------------------------------|
| SondeHourly | ~1 per hour | Round to 3 decimals | <1% (mostly from "NAN" strings) |
| PFL_Step | ~2–3 per day | Round to 3 decimals | <1% |
| Weather | ~1 per hour | Threshold validation (12 parameters) | 2–5% (depends on sensor calibration) |
| Profiler (model input) | On-demand | Advanced thresholds (10 parameters) | 5–15% (stricter acceptance window) |

**Key Takeaway:** Raw sensor values are minimally altered in routine storage. Stricter QA/QC is applied *only when preparing data for hydrodynamic model input* (Delft3D), at which point 5–15% of profiler records may be flagged as invalid based on physical plausibility.
