# Physics Toolbox - Driving Data Simulator

A Python-based simulator that generates realistic driving kinematics data from different driver profiles, mimicking data collected from smartphone sensors (accelerometer and gyroscope).

## Overview

This project simulates the Physics Toolbox application, which gathers kinematics data from device sensors. It generates synthetic driving data for various driver types, complete with:
- **G-force measurements** (total and per axis)
- **Linear accelerometer data** (x, y, z axes)
- **Gyroscope data** (rotation rates)

## Driver Profiles

The simulator includes 5 distinct driver types:

1. **Normal** - Average driver with moderate acceleration and smooth turns
2. **Aggressive** - Fast acceleration, hard braking, sharp turns
3. **Cautious** - Gentle acceleration, early braking, slow turns
4. **Erratic** - Unpredictable with sudden changes in driving behavior
5. **Smooth** - Very gradual changes, minimal sudden movements

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Driving Data

Run the data generation script:

```bash
python generate_driving_data.py
```

This will create a `data/` directory with:
- Individual CSV files for each driver type
- A combined CSV file with all driver types
- Metadata JSON file

### 2. Assess Driving Safety

Analyze driving safety with the rule-based assessment system:

```bash
# Assess a single driver
python assess_driving_safety.py data/driving_data_aggressive.csv

# Assess all driver types and compare
python assess_driving_safety.py --all

# Save report to specific file
python assess_driving_safety.py data/driving_data_normal.csv -o my_report.json
```

The assessment system provides:
- **Overall Safety Score** (0-100) with risk level classification
- **Event Detection**: Hard acceleration, braking, and turning events
- **Smoothness Analysis**: Driving consistency metrics
- **Intensity Metrics**: Maximum and average force measurements
- **Risk Breakdown**: Weighted scoring across multiple factors
- **Personalized Recommendations**: Specific suggestions for improvement
- **JSON Reports**: Detailed structured data saved for each assessment

### 3. Visualize Data

Open the Jupyter notebook:

```bash
jupyter notebook driving_data_analysis.ipynb
```

The notebook includes:
- Time series plots of all sensor data
- Comparative analysis between driver types
- Statistical summaries
- Frequency analysis
- Event detection
- 3D acceleration space visualization

## Safety Assessment System

The rule-based assessment system evaluates driving safety using five key factors with weighted scoring:

### Assessment Factors

1. **Hard Events (30% weight)**: Dangerous maneuvers
   - Hard acceleration (>2.5 m/s²)
   - Hard braking (<-2.5 m/s²)
   - Sharp turns (>2.0 m/s² lateral)

2. **Moderate Events (15% weight)**: Less severe but notable events
   - Moderate acceleration/braking (1.5-2.5 m/s²)
   - Moderate turns (1.2-2.0 m/s² lateral)

3. **G-Force (20% weight)**: Peak force measurements
   - High g-force events (>1.3g)
   - Maximum g-force levels during drive

4. **Smoothness (20% weight)**: Driving consistency
   - Acceleration variance and standard deviation
   - Gradual vs. abrupt transitions

5. **Overall Intensity (15% weight)**: Average force levels
   - Mean absolute acceleration
   - Overall driving aggressiveness

### Risk Level Classification

| Safety Score | Risk Level | Description |
|--------------|------------|-------------|
| 80-100 | **SAFE** | Excellent driving behavior with minimal risk |
| 65-79 | **LOW RISK** | Good driving with minor concerns |
| 50-64 | **MODERATE RISK** | Acceptable but with notable risky behaviors |
| 35-49 | **HIGH RISK** | Dangerous driving patterns detected |
| 0-34 | **VERY HIGH RISK** | Extremely dangerous driving behavior |

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
  1. ⚠️ Reduce hard acceleration: 24.3 events per minute detected
  2. ⚠️ Overall driving intensity is high. Practice more gradual speed changes.
  3. 🚨 CRITICAL: Your driving pattern shows dangerous behavior.
```

## Data Format

Each CSV file contains the following columns:

| Column | Description | Unit |
|--------|-------------|------|
| `timestamp` | Time of measurement | datetime |
| `time_seconds` | Elapsed time | seconds |
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
| `driver_type` | Driver profile | string |

## Customization

You can customize the simulation parameters in `generate_driving_data.py`:

```python
df = generate_all_driver_types(
    output_dir='data',           # Output directory
    duration_minutes=10,         # Duration of each drive
    sampling_rate=50            # Samples per second (Hz)
)
```

## Safety Assessment System

The rule-based assessment system evaluates driving safety using five key factors:

1. **Hard Events (30% weight)**: Dangerous maneuvers like hard acceleration/braking
2. **Moderate Events (15% weight)**: Less severe but notable events
3. **G-Force (20% weight)**: Peak force measurements during driving
4. **Smoothness (20% weight)**: Consistency and gradual transitions
5. **Overall Intensity (15% weight)**: Average force levels throughout drive

### Risk Levels

- **SAFE** (80-100): Excellent driving behavior with minimal risk
- **LOW RISK** (65-79): Good driving with minor concerns
- **MODERATE RISK** (50-64): Acceptable but with notable risky behaviors
- **HIGH RISK** (35-49): Dangerous driving patterns detected
- **VERY HIGH RISK** (0-34): Extremely dangerous driving behavior

## Project Structure

```
physics-toolbox/
├── generate_driving_data.py    # Data generation script
├── assess_driving_safety.py    # Safety assessment system
├── driving_data_analysis.ipynb # Visualization notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
└── data/                       # Generated data (created on first run)
    ├── driving_data_normal.csv
    ├── driving_data_aggressive.csv
    ├── driving_data_cautious.csv
    ├── driving_data_erratic.csv
    ├── driving_data_smooth.csv
    ├── driving_data_all.csv
    ├── metadata.json
    └── safety_report_*.json    # Assessment reports
```

## License

MIT License
