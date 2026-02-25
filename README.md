# Physics Toolbox - Driving Data Simulator

A Python-based simulator that generates realistic driving kinematics data from different driver profiles, mimicking data collected from smartphone sensors (accelerometer and gyroscope).

## Overview

This project simulates the Physics Toolbox application, which gathers kinematics data from device sensors. It generates synthetic driving data for various driver types, complete with:

- **G-force measurements** (total and per axis)
- **Linear accelerometer data** (x, y, z axes)
- **Gyroscope data** (rotation rates)

Data is stored as **Parquet files** under a date-partitioned layout and assessed through a rule-based safety scoring system.

## Driver Profiles

The simulator includes 5 distinct driver types, defined in `config.yaml`:

| Profile | Characteristics |
|---------|----------------|
| `normal` | Average driver with moderate acceleration and smooth turns |
| `aggressive` | Fast acceleration, hard braking, sharp turns |
| `cautious` | Gentle acceleration, early braking, slow turns |
| `erratic` | Unpredictable with sudden changes in behavior |
| `smooth` | Very gradual changes, minimal sudden movements |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Driving Data

Rides are stored under `data/<date>/<driver_id>/<ride_id>.parquet`. A driver registry at `data/drivers.json` tracks known drivers and is seeded automatically on first run (one driver per profile).

```bash
# List registered drivers
python generate_driving_data.py --list-drivers

# Generate a ride for a specific driver
python generate_driving_data.py --driver-id <id> --ride-type aggressive

# Generate one ride for every registered driver
python generate_driving_data.py --all --ride-type normal

# Override duration, sampling rate, or date
python generate_driving_data.py --all --ride-type smooth \
    --duration 15 --sampling-rate 100 --date 2026-02-25
```

All simulation defaults (duration, sampling rate) and profile parameters are controlled by `config.yaml`.

### 2. Assess Driving Safety

Pass the path to any ride Parquet file:

```bash
python assess_driving_safety.py data/2026-02-25/<driver_id>/<ride_id>.parquet

# Write the JSON report to a custom location
python assess_driving_safety.py data/.../ride.parquet -o my_report.json

# Use a different reports directory
python assess_driving_safety.py data/.../ride.parquet --reports-dir reports/2026-02
```

The assessment system provides:

- **Overall Safety Score** (0–100) with risk level classification
- **Event Detection**: Hard/moderate acceleration, braking, and turning events
- **Smoothness Analysis**: Driving consistency metrics
- **Intensity Metrics**: Maximum and average force measurements
- **Risk Breakdown**: Weighted scoring across multiple factors
- **Personalized Recommendations**: Specific suggestions for improvement
- **JSON Reports**: Saved to `reports/` (created automatically if absent)

### 3. Visualize Data

Open the Jupyter notebook for quick visual inspection of a single ride:

```bash
jupyter notebook driving_data_analysis.ipynb
```

Set `RIDE_FILE` in the first cell to point at any ride Parquet file. Running all cells will:

- Display time-series plots of all sensor channels with threshold overlays
- Display a G-G diagram (lateral vs. longitudinal acceleration)
- Save both figures as PNG files to the `images/` directory (created automatically)

## Safety Assessment System

The rule-based system evaluates driving safety using five weighted factors:

### Assessment Factors

| Factor | Weight | What it measures |
|--------|--------|-----------------|
| Hard Events | 30% | Hard acceleration (>2.5 m/s²), hard braking (<-2.5 m/s²), sharp turns (>2.0 m/s²) |
| Moderate Events | 15% | Moderate acceleration/braking (1.5–2.5 m/s²), moderate turns (1.2–2.0 m/s²) |
| G-Force | 20% | High g-force events (>1.3g) and peak g-force |
| Smoothness | 20% | Acceleration variance and standard deviation |
| Overall Intensity | 15% | Mean absolute acceleration throughout the drive |

Thresholds and weights are fully configurable in `config.yaml`.

### Risk Level Classification

| Safety Score | Risk Level | Description |
|--------------|------------|-------------|
| 80–100 | **SAFE** | Excellent driving behavior with minimal risk |
| 65–79 | **LOW RISK** | Good driving with minor concerns |
| 50–64 | **MODERATE RISK** | Acceptable but with notable risky behaviors |
| 35–49 | **HIGH RISK** | Dangerous driving patterns detected |
| 0–34 | **VERY HIGH RISK** | Extremely dangerous driving behavior |

### Example Output

```
OVERALL SAFETY ASSESSMENT
  Safety Score: 44.2/100
  Risk Level: HIGH RISK
  Description: Dangerous driving patterns detected
  [████████░░░░░░░░░░░░] 44.2%

DETECTED EVENTS
  Hard Acceleration Events: 243 (24.30/min)
  Hard Braking Events: 7 (0.70/min)
  Sharp Turn Events: 0 (0.00/min)

RECOMMENDATIONS
  1. Reduce hard acceleration: 24.3 events per minute detected
  2. Overall driving intensity is high. Practice more gradual speed changes.
  3. CRITICAL: Your driving pattern shows dangerous behavior.
     Consider taking a defensive driving course.
```

## Configuration

All parameters live in `config.yaml`:

```yaml
simulation:
  duration_minutes: 10
  sampling_rate: 50       # Hz

output:
  data_dir: data
  images_dir: images
  reports_dir: reports

assessment:
  thresholds: { ... }
  weights: { ... }

profiles:
  normal: { ... }
  aggressive: { ... }
  # ...
```

## Data Format

Each Parquet file stores one ride. Ride metadata (driver ID, ride type, date, sampling rate) is embedded as Parquet key-value schema metadata.

| Column | Description | Unit |
|--------|-------------|------|
| `timestamp` | UTC time of sample | datetime |
| `time_seconds` | Elapsed time since ride start | s |
| `accel_x` | Forward/backward acceleration | m/s² |
| `accel_y` | Left/right acceleration | m/s² |
| `accel_z` | Up/down acceleration | m/s² |
| `gforce_x` | Forward/backward g-force | g |
| `gforce_y` | Left/right g-force | g |
| `gforce_z` | Up/down g-force | g |
| `gforce_total` | Total g-force magnitude | g |
| `gyro_x` | Roll rate | rad/s |
| `gyro_y` | Pitch rate | rad/s |
| `gyro_z` | Yaw rate | rad/s |

## Project Structure

```
physics-toolbox/
├── generate_driving_data.py    # Data generation & driver registry
├── assess_driving_safety.py    # Rule-based safety assessment
├── driving_data_analysis.ipynb # Single-ride visualizer notebook
├── config.yaml                 # All parameters and thresholds
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── data/                       # Generated rides (git-ignored)
│   ├── drivers.json            #   Driver registry
│   └── <date>/
│       └── <driver_id>/
│           └── <ride_id>.parquet
├── images/                     # Saved notebook plots (git-ignored)
│   ├── <ride_id>_timeseries.png
│   └── <ride_id>_gg_diagram.png
└── reports/                    # JSON assessment reports (git-ignored)
    └── <ride_id>.json
```

## License

MIT License
