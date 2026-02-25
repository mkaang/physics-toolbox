"""
Physics Toolbox - Driving Data Simulator
Generates realistic driving kinematics data for different driver profiles
"""

import json
import uuid
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta
from pathlib import Path
import yaml


def load_config(config_path='config.yaml'):
    """Load simulation configuration and driver profiles from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class DrivingSimulator:
    """Simulates driving kinematics data with different driver profiles"""

    def __init__(self, ride_type='normal', duration_minutes=10, sampling_rate=50, profiles=None):
        """
        Initialize the driving simulator

        Parameters:
        -----------
        ride_type : str
            Riding style profile; must match a key in the profiles dict
        duration_minutes : float
            Duration of the drive in minutes
        sampling_rate : int
            Samples per second (Hz)
        profiles : dict
            Profile parameters loaded from config; falls back to defaults if None
        """
        self.ride_type = ride_type
        self.duration_minutes = duration_minutes
        self.sampling_rate = sampling_rate
        self.num_samples = int(duration_minutes * 60 * sampling_rate)

        if profiles is None:
            profiles = load_config()['profiles']

        if ride_type not in profiles:
            raise ValueError(f"Unknown ride type '{ride_type}'. Available: {list(profiles.keys())}")

        self.profile = profiles[ride_type]

    def generate_base_acceleration(self):
        """Generate base forward/backward acceleration patterns"""
        accel = np.random.normal(
            self.profile['accel_mean'],
            self.profile['accel_std'] * self.profile['smoothness'],
            self.num_samples
        )

        num_events = int(self.num_samples * self.profile['event_frequency'])
        event_indices = np.random.choice(self.num_samples, num_events, replace=False)

        for idx in event_indices:
            if np.random.random() > 0.5:
                event_strength = np.random.uniform(0.3, self.profile['accel_max'])
                event_duration = int(self.sampling_rate * np.random.uniform(1, 3))
            else:
                event_strength = -np.random.uniform(0.3, self.profile['brake_intensity'])
                event_duration = int(self.sampling_rate * np.random.uniform(0.5, 2))

            start_idx = max(0, idx - event_duration // 2)
            end_idx = min(self.num_samples, idx + event_duration // 2)
            event_window = np.linspace(-1, 1, end_idx - start_idx)
            envelope = np.exp(-event_window**2 / (self.profile['smoothness'] * 0.5))
            accel[start_idx:end_idx] += event_strength * envelope

        window_size = int(self.sampling_rate * 0.1)
        accel = np.convolve(accel, np.ones(window_size) / window_size, mode='same')

        return accel

    def generate_lateral_acceleration(self):
        """Generate lateral (turning) acceleration"""
        accel_y = np.random.normal(0, 0.05, self.num_samples)

        num_turns = int(self.duration_minutes * 60 / 15)
        turn_indices = np.random.choice(self.num_samples, num_turns, replace=False)

        for idx in turn_indices:
            turn_direction = np.random.choice([-1, 1])
            turn_strength = turn_direction * np.random.uniform(0.2, self.profile['turn_intensity'])
            turn_duration = int(self.sampling_rate * np.random.uniform(1, 3))

            start_idx = max(0, idx - turn_duration // 2)
            end_idx = min(self.num_samples, idx + turn_duration // 2)
            turn_window = np.linspace(0, np.pi, end_idx - start_idx)
            envelope = np.sin(turn_window)
            accel_y[start_idx:end_idx] += turn_strength * envelope

        window_size = int(self.sampling_rate * 0.15)
        accel_y = np.convolve(accel_y, np.ones(window_size) / window_size, mode='same')

        return accel_y

    def generate_vertical_acceleration(self):
        """Generate vertical acceleration (road bumps, gravity)"""
        accel_z = np.ones(self.num_samples) * 1.0

        vibration = np.random.normal(0, 0.02, self.num_samples)
        accel_z += vibration

        num_bumps = int(self.duration_minutes * 2)
        bump_indices = np.random.choice(self.num_samples, num_bumps, replace=False)

        for idx in bump_indices:
            bump_strength = np.random.uniform(-0.15, 0.15)
            bump_duration = int(self.sampling_rate * 0.3)

            start_idx = max(0, idx - bump_duration // 2)
            end_idx = min(self.num_samples, idx + bump_duration // 2)
            bump_window = np.linspace(0, 2 * np.pi, end_idx - start_idx)
            envelope = np.sin(bump_window)
            accel_z[start_idx:end_idx] += bump_strength * envelope

        return accel_z

    def acceleration_to_gforce(self, accel_x, accel_y, accel_z):
        """Convert acceleration to g-force"""
        g = 9.81
        gforce_x = accel_x / g
        gforce_y = accel_y / g
        gforce_z = accel_z
        gforce_total = np.sqrt(gforce_x**2 + gforce_y**2 + gforce_z**2)
        return gforce_x, gforce_y, gforce_z, gforce_total

    def generate_gyroscope(self, accel_y):
        """Generate gyroscope data (rotation rates) based on lateral acceleration"""
        gyro_z = np.gradient(accel_y) * 2.0
        gyro_z += np.random.normal(0, 0.02, self.num_samples)
        gyro_y = np.random.normal(0, 0.03, self.num_samples)
        gyro_x = -accel_y * 0.5 + np.random.normal(0, 0.02, self.num_samples)
        return gyro_x, gyro_y, gyro_z

    def generate_data(self):
        """Generate complete driving dataset"""
        print(f"Generating {self.ride_type!r} ride data for {self.duration_minutes} minutes...")

        accel_x = self.generate_base_acceleration()
        accel_y = self.generate_lateral_acceleration()
        accel_z = self.generate_vertical_acceleration()

        gforce_x, gforce_y, gforce_z, gforce_total = self.acceleration_to_gforce(
            accel_x, accel_y, accel_z
        )

        gyro_x, gyro_y, gyro_z = self.generate_gyroscope(accel_y)

        start_time = datetime.now()
        timestamps = [start_time + timedelta(seconds=i / self.sampling_rate)
                      for i in range(self.num_samples)]

        df = pd.DataFrame({
            'timestamp': timestamps,
            'time_seconds': np.linspace(0, self.duration_minutes * 60, self.num_samples),
            'accel_x': accel_x,
            'accel_y': accel_y,
            'accel_z': accel_z,
            'gforce_x': gforce_x,
            'gforce_y': gforce_y,
            'gforce_z': gforce_z,
            'gforce_total': gforce_total,
            'gyro_x': gyro_x,
            'gyro_y': gyro_y,
            'gyro_z': gyro_z,
        })

        print(f"Generated {len(df)} samples")
        return df


# ---------------------------------------------------------------------------
# Driver registry
# ---------------------------------------------------------------------------

def _short_id(seed=None, length=8):
    """Return a short hex ID. Deterministic if seed (str) is given, random otherwise."""
    if seed is not None:
        return uuid.uuid5(uuid.NAMESPACE_DNS, seed).hex[:length]
    return uuid.uuid4().hex[:length]


def _registry_path(data_dir: str) -> Path:
    return Path(data_dir) / 'drivers.json'


def load_registry(config: dict) -> dict:
    """
    Load the driver registry from disk.

    If it does not exist, seed it with 5 drivers (one per profile) using
    deterministic short IDs, then persist it.

    Registry schema:
        { "<driver_id>": { "created_at": str }, ... }

    Drivers carry no fixed ride type — ride_type is chosen per ride.
    """
    path = _registry_path(config['output']['data_dir'])
    if path.exists():
        with open(path) as f:
            return json.load(f)

    registry = {}
    for profile_name in config['profiles']:
        driver_id = _short_id(seed=profile_name)
        registry[driver_id] = {'created_at': datetime.now().isoformat()}

    path.parent.mkdir(parents=True, exist_ok=True)
    _save_registry(registry, path)
    print(f"Initialized driver registry with {len(registry)} drivers → {path}")
    return registry


def _save_registry(registry: dict, path: Path) -> None:
    with open(path, 'w') as f:
        json.dump(registry, f, indent=2)


def get_or_create_driver(driver_id: str, registry: dict, config: dict) -> dict:
    """
    Return the registry entry for driver_id, creating it if unknown.
    New drivers get a bare entry with just a creation timestamp.
    """
    if driver_id in registry:
        return registry[driver_id]

    entry = {'created_at': datetime.now().isoformat()}
    registry[driver_id] = entry
    _save_registry(registry, _registry_path(config['output']['data_dir']))
    print(f"Registered new driver: {driver_id}")
    return entry


def list_drivers(registry: dict) -> None:
    print(f"\n{'DRIVER ID':<12}  CREATED AT")
    print("-" * 35)
    for driver_id, entry in registry.items():
        print(f"{driver_id:<12}  {entry['created_at']}")
    print()


# ---------------------------------------------------------------------------
# Parquet I/O
# ---------------------------------------------------------------------------

def save_ride(df, ride_type, driver_id, ride_id, date_str, data_dir,
              duration_minutes, sampling_rate):
    """
    Save one ride as a parquet file at:
        <data_dir>/<date>/<driver_id>/<ride_id>.parquet

    ride_type, driver_id, ride_id, date and simulation settings are embedded
    as parquet key-value metadata.
    """
    ride_dir = Path(data_dir) / date_str / driver_id
    ride_dir.mkdir(parents=True, exist_ok=True)
    out_file = ride_dir / f"{ride_id}.parquet"

    table = pa.Table.from_pandas(df, preserve_index=False)
    file_meta = {
        'ride_type': ride_type,
        'driver_id': driver_id,
        'ride_id': ride_id,
        'date': date_str,
        'duration_minutes': str(duration_minutes),
        'sampling_rate': str(sampling_rate),
    }
    merged = {**(table.schema.metadata or {}),
              **{k.encode(): v.encode() for k, v in file_meta.items()}}
    pq.write_table(table.replace_schema_metadata(merged), out_file)
    print(f"Saved: {out_file}")
    return out_file


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_rides(targets, config, duration_minutes, sampling_rate, date_str):
    """
    Simulate and persist one ride per (driver_id, ride_type) pair.

    Parameters
    ----------
    targets : list[tuple[str, str]]
        (driver_id, ride_type) pairs to simulate.
    config : dict
        Loaded YAML config.
    duration_minutes : float
    sampling_rate : int
    date_str : str

    Returns
    -------
    list[dict]
        One dict per ride: ride_type, driver_id, ride_id, date, file, df.
    """
    profiles = config['profiles']
    data_dir = config['output']['data_dir']

    Path(config['output']['images_dir']).mkdir(parents=True, exist_ok=True)
    Path(config['output']['reports_dir']).mkdir(parents=True, exist_ok=True)

    rides = []
    for driver_id, ride_type in targets:
        ride_id = _short_id()
        simulator = DrivingSimulator(
            ride_type=ride_type,
            duration_minutes=duration_minutes,
            sampling_rate=sampling_rate,
            profiles=profiles,
        )
        df = simulator.generate_data()
        out_file = save_ride(
            df, ride_type, driver_id, ride_id,
            date_str, data_dir, duration_minutes, sampling_rate,
        )
        rides.append({
            'ride_type': ride_type,
            'driver_id': driver_id,
            'ride_id': ride_id,
            'date': date_str,
            'file': out_file,
            'df': df,
        })

    print(f"\nGeneration complete — {len(rides)} ride(s) stored under '{data_dir}/'")
    return rides


def print_summary(rides):
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    for ride in rides:
        df = ride['df']
        print(f"\nride_type: {ride['ride_type']}  |  driver: {ride['driver_id']}  |  ride: {ride['ride_id']}")
        print(f"  Max g-force:    {df['gforce_total'].max():.3f} g")
        print(f"  Mean |accel_x|: {df['accel_x'].abs().mean():.3f} m/s²")
        print(f"  Max |accel_x|:  {df['accel_x'].abs().max():.3f} m/s²")
        print(f"  Mean |gyro_z|:  {df['gyro_z'].abs().mean():.3f} rad/s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    config = load_config()
    sim_cfg = config['simulation']
    available_profiles = list(config['profiles'].keys())

    parser = argparse.ArgumentParser(
        description='Generate simulated driving kinematics data and store as parquet.\n\n'
                    'Layout: <data_dir>/<date>/<driver_id>/<ride_id>.parquet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--driver-id',
        metavar='ID',
        help='ID of the driver to simulate a ride for. '
             'If unknown, a new driver is registered automatically.',
    )
    mode.add_argument(
        '--all',
        action='store_true',
        help='Generate one ride for every driver in the registry (requires --ride-type).',
    )
    mode.add_argument(
        '--list-drivers',
        action='store_true',
        help='Print the driver registry and exit.',
    )

    parser.add_argument(
        '--ride-type',
        metavar='PROFILE',
        choices=available_profiles,
        help=f'Riding style profile for this ride. '
             f'Available: {", ".join(available_profiles)}.',
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        metavar='PATH',
        help='Path to YAML config file (default: config.yaml).',
    )
    parser.add_argument(
        '--duration',
        type=float,
        default=None,
        metavar='MINUTES',
        help=f'Drive duration in minutes (default: {sim_cfg["duration_minutes"]} from config).',
    )
    parser.add_argument(
        '--sampling-rate',
        type=int,
        default=None,
        metavar='HZ',
        help=f'Sampling rate in Hz (default: {sim_cfg["sampling_rate"]} from config).',
    )
    parser.add_argument(
        '--date',
        default=None,
        metavar='YYYY-MM-DD',
        help='Partition date (default: today).',
    )

    args = parser.parse_args()

    duration_minutes = args.duration or sim_cfg['duration_minutes']
    sampling_rate = args.sampling_rate or sim_cfg['sampling_rate']
    date_str = args.date or datetime.now().strftime('%Y-%m-%d')

    registry = load_registry(config)

    if args.list_drivers:
        list_drivers(registry)
        return

    if not args.ride_type and not args.list_drivers:
        parser.error("--ride-type is required when generating rides.")

    if args.all:
        targets = [(did, args.ride_type) for did in registry]
    elif args.driver_id:
        get_or_create_driver(args.driver_id, registry, config)
        targets = [(args.driver_id, args.ride_type)]
    else:
        parser.print_help()
        return

    rides = generate_rides(targets, config, duration_minutes, sampling_rate, date_str)
    print_summary(rides)


if __name__ == '__main__':
    main()
