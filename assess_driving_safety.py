"""
Physics Toolbox - Drive Safety Assessment System
Rule-based analysis of driving data to assess safety and risk levels
"""

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import json
import yaml
from datetime import datetime
from pathlib import Path
import argparse


def load_config(config_path='config.yaml'):
    """Load assessment configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class DriveSafetyAssessor:
    """Rule-based system for assessing driving safety from sensor data"""

    def __init__(self, ride_file, config_path='config.yaml'):
        """
        Initialize assessor with a ride parquet file.

        Parameters:
        -----------
        ride_file : str or Path
            Path to a ride parquet file, e.g.
            'data/2026-02-24/<driver_id>/<ride_id>.parquet'
        config_path : str or Path
            Path to the YAML config file (default: config.yaml)
        """
        cfg = load_config(config_path)
        self.THRESHOLDS = cfg['assessment']['thresholds']
        self.WEIGHTS = cfg['assessment']['weights']

        self.ride_file = Path(ride_file)
        self.df = None
        self.assessment = {}
        self.ride_type = None
        self.driver_id = None
        self.ride_id = None
        self.date = None
        self.duration_minutes = None

    def load_data(self):
        """Load driving data and metadata from the ride parquet file"""
        print(f"Loading data from: {self.ride_file}")

        pf = pq.read_table(self.ride_file)
        raw_meta = pf.schema.metadata or {}
        meta = {k.decode(): v.decode() for k, v in raw_meta.items()}

        self.ride_type = meta.get('ride_type', 'unknown')
        self.driver_id = meta.get('driver_id', 'unknown')
        self.ride_id = meta.get('ride_id', self.ride_file.stem)
        self.date = meta.get('date', 'unknown')

        self.df = pf.to_pandas()
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        self.duration_minutes = self.df['time_seconds'].max() / 60

        print(f"  Ride type: {self.ride_type}")
        print(f"  Driver ID:   {self.driver_id}")
        print(f"  Ride ID:     {self.ride_id}")
        print(f"  Date:        {self.date}")
        print(f"  Duration: {self.duration_minutes:.2f} minutes")
        print(f"  Samples: {len(self.df)}")

    def detect_events(self):
        """Detect dangerous and risky driving events"""
        events = {
            'hard_acceleration': 0,
            'hard_braking': 0,
            'moderate_acceleration': 0,
            'moderate_braking': 0,
            'sharp_turns': 0,
            'moderate_turns': 0,
            'high_gforce_events': 0,
            'moderate_gforce_events': 0
        }

        events['hard_acceleration'] = (self.df['accel_x'] > self.THRESHOLDS['hard_accel']).sum()
        events['hard_braking'] = (self.df['accel_x'] < self.THRESHOLDS['hard_brake']).sum()

        moderate_accel_mask = (self.df['accel_x'] > self.THRESHOLDS['moderate_accel']) & \
                              (self.df['accel_x'] <= self.THRESHOLDS['hard_accel'])
        events['moderate_acceleration'] = moderate_accel_mask.sum()

        moderate_brake_mask = (self.df['accel_x'] < self.THRESHOLDS['moderate_brake']) & \
                              (self.df['accel_x'] >= self.THRESHOLDS['hard_brake'])
        events['moderate_braking'] = moderate_brake_mask.sum()

        events['sharp_turns'] = (self.df['accel_y'].abs() > self.THRESHOLDS['sharp_turn']).sum()

        moderate_turn_mask = (self.df['accel_y'].abs() > self.THRESHOLDS['moderate_turn']) & \
                             (self.df['accel_y'].abs() <= self.THRESHOLDS['sharp_turn'])
        events['moderate_turns'] = moderate_turn_mask.sum()

        events['high_gforce_events'] = (self.df['gforce_total'] > self.THRESHOLDS['high_gforce']).sum()

        moderate_gforce_mask = (self.df['gforce_total'] > self.THRESHOLDS['moderate_gforce']) & \
                               (self.df['gforce_total'] <= self.THRESHOLDS['high_gforce'])
        events['moderate_gforce_events'] = moderate_gforce_mask.sum()

        events_per_min = {
            'hard_acceleration_per_min': events['hard_acceleration'] / self.duration_minutes,
            'hard_braking_per_min': events['hard_braking'] / self.duration_minutes,
            'sharp_turns_per_min': events['sharp_turns'] / self.duration_minutes,
            'total_hard_events_per_min': (events['hard_acceleration'] +
                                          events['hard_braking'] +
                                          events['sharp_turns']) / self.duration_minutes
        }

        events.update(events_per_min)
        return events

    def analyze_smoothness(self):
        """Analyze driving smoothness based on acceleration variance"""
        smoothness_metrics = {
            'accel_x_std': self.df['accel_x'].std(),
            'accel_y_std': self.df['accel_y'].std(),
            'accel_x_variance': self.df['accel_x'].var(),
            'accel_y_variance': self.df['accel_y'].var(),
            'gyro_z_std': self.df['gyro_z'].std(),
        }

        accel_smoothness = max(0, 100 - (smoothness_metrics['accel_x_std'] * 50))
        lateral_smoothness = max(0, 100 - (smoothness_metrics['accel_y_std'] * 50))

        smoothness_metrics['acceleration_smoothness_score'] = accel_smoothness
        smoothness_metrics['lateral_smoothness_score'] = lateral_smoothness
        smoothness_metrics['overall_smoothness_score'] = (accel_smoothness + lateral_smoothness) / 2

        return smoothness_metrics

    def analyze_intensity(self):
        """Analyze overall driving intensity"""
        intensity_metrics = {
            'max_forward_accel': self.df['accel_x'].max(),
            'max_backward_accel': abs(self.df['accel_x'].min()),
            'max_lateral_accel': self.df['accel_y'].abs().max(),
            'max_gforce': self.df['gforce_total'].max(),
            'mean_abs_accel_x': self.df['accel_x'].abs().mean(),
            'mean_abs_accel_y': self.df['accel_y'].abs().mean(),
            'mean_gforce': self.df['gforce_total'].mean(),
            'max_yaw_rate': self.df['gyro_z'].abs().max(),
        }

        intensity_score = min(100, (
            intensity_metrics['mean_abs_accel_x'] * 20 +
            intensity_metrics['mean_abs_accel_y'] * 15 +
            (intensity_metrics['max_gforce'] - 1.0) * 50
        ))

        intensity_metrics['intensity_score'] = intensity_score
        return intensity_metrics

    def calculate_risk_scores(self, events, smoothness, intensity):
        """Calculate risk scores for different aspects of driving"""
        hard_events_total = events['hard_acceleration'] + events['hard_braking'] + events['sharp_turns']
        hard_events_score = min(100, hard_events_total / self.duration_minutes * 5)

        moderate_events_total = events['moderate_acceleration'] + events['moderate_braking'] + events['moderate_turns']
        moderate_events_score = min(100, moderate_events_total / self.duration_minutes * 2)

        gforce_score = min(100, (
            events['high_gforce_events'] / len(self.df) * 200 +
            (intensity['max_gforce'] - 1.0) * 100
        ))

        smoothness_risk_score = 100 - smoothness['overall_smoothness_score']
        intensity_risk_score = intensity['intensity_score']

        return {
            'hard_events_score': hard_events_score,
            'moderate_events_score': moderate_events_score,
            'gforce_score': gforce_score,
            'smoothness_risk_score': smoothness_risk_score,
            'intensity_risk_score': intensity_risk_score
        }

    def calculate_overall_safety_score(self, risk_scores):
        """Calculate overall safety score using weighted average"""
        overall_risk = (
            risk_scores['hard_events_score'] * self.WEIGHTS['hard_events'] +
            risk_scores['moderate_events_score'] * self.WEIGHTS['moderate_events'] +
            risk_scores['gforce_score'] * self.WEIGHTS['gforce'] +
            risk_scores['smoothness_risk_score'] * self.WEIGHTS['smoothness'] +
            risk_scores['intensity_risk_score'] * self.WEIGHTS['overall_intensity']
        )
        safety_score = 100 - overall_risk
        return safety_score, overall_risk

    def determine_risk_level(self, safety_score):
        """Determine risk level and classification"""
        if safety_score >= 80:
            return 'SAFE', 'Excellent driving behavior with minimal risk'
        elif safety_score >= 65:
            return 'LOW RISK', 'Good driving with minor concerns'
        elif safety_score >= 50:
            return 'MODERATE RISK', 'Acceptable driving but with notable risky behaviors'
        elif safety_score >= 35:
            return 'HIGH RISK', 'Dangerous driving patterns detected'
        else:
            return 'VERY HIGH RISK', 'Extremely dangerous driving behavior'

    def generate_recommendations(self, events, smoothness, intensity, risk_level):
        """Generate specific recommendations based on analysis"""
        recommendations = []

        if events['hard_acceleration_per_min'] > 2:
            recommendations.append(
                f"Reduce hard acceleration: {events['hard_acceleration_per_min']:.1f} events per minute detected"
            )

        if events['hard_braking_per_min'] > 2:
            recommendations.append(
                f"Reduce hard braking: {events['hard_braking_per_min']:.1f} events per minute detected. "
                "Maintain more following distance."
            )

        if events['sharp_turns_per_min'] > 1.5:
            recommendations.append(
                f"Reduce sharp turning: {events['sharp_turns_per_min']:.1f} sharp turns per minute. "
                "Slow down before turns."
            )

        if smoothness['overall_smoothness_score'] < 50:
            recommendations.append(
                "Improve smoothness: Avoid sudden acceleration/deceleration changes"
            )

        if intensity['max_gforce'] > 1.25:
            recommendations.append(
                f"High g-forces detected ({intensity['max_gforce']:.2f}g). "
                "Reduce aggressive maneuvers."
            )

        if intensity['mean_abs_accel_x'] > 0.5:
            recommendations.append(
                "Overall driving intensity is high. Practice more gradual speed changes."
            )

        if risk_level in ['HIGH RISK', 'VERY HIGH RISK']:
            recommendations.append(
                "CRITICAL: Your driving pattern shows dangerous behavior. "
                "Consider taking a defensive driving course."
            )

        if not recommendations:
            recommendations.append("Excellent! Continue maintaining safe driving habits.")

        return recommendations

    def assess(self):
        """Run complete safety assessment"""
        print("\n" + "=" * 70)
        print("DRIVE SAFETY ASSESSMENT")
        print("=" * 70)

        self.load_data()

        print("\nAnalyzing driving patterns...")
        events = self.detect_events()
        smoothness = self.analyze_smoothness()
        intensity = self.analyze_intensity()

        risk_scores = self.calculate_risk_scores(events, smoothness, intensity)
        safety_score, overall_risk = self.calculate_overall_safety_score(risk_scores)
        risk_level, risk_description = self.determine_risk_level(safety_score)
        recommendations = self.generate_recommendations(events, smoothness, intensity, risk_level)

        self.assessment = {
            'metadata': {
                'ride_type': self.ride_type,
                'driver_id': self.driver_id,
                'ride_id': self.ride_id,
                'date': self.date,
                'duration_minutes': self.duration_minutes,
                'samples': len(self.df),
                'assessment_time': datetime.now().isoformat()
            },
            'events': events,
            'smoothness': smoothness,
            'intensity': intensity,
            'risk_scores': risk_scores,
            'safety_score': safety_score,
            'overall_risk': overall_risk,
            'risk_level': risk_level,
            'risk_description': risk_description,
            'recommendations': recommendations
        }

        return self.assessment

    def print_report(self):
        """Print detailed assessment report"""
        a = self.assessment

        print("\n" + "=" * 70)
        print("DRIVE SAFETY REPORT")
        print("=" * 70)

        print(f"\nRide Type: {a['metadata']['ride_type'].upper()}")
        print(f"Driver ID:   {a['metadata']['driver_id']}")
        print(f"Ride ID:     {a['metadata']['ride_id']}")
        print(f"Date:        {a['metadata']['date']}")
        print(f"Duration:    {a['metadata']['duration_minutes']:.2f} minutes")
        print(f"Assessed at: {a['metadata']['assessment_time']}")

        print("\n" + "-" * 70)
        print("OVERALL SAFETY ASSESSMENT")
        print("-" * 70)
        print(f"\n  Safety Score: {a['safety_score']:.1f}/100")
        print(f"  Risk Level: {a['risk_level']}")
        print(f"  Description: {a['risk_description']}")

        filled = int(a['safety_score'] / 5)
        empty = 20 - filled
        print(f"\n  [{'█' * filled}{'░' * empty}] {a['safety_score']:.1f}%")

        print("\n" + "-" * 70)
        print("DETECTED EVENTS")
        print("-" * 70)
        print(f"\n  Hard Acceleration Events: {a['events']['hard_acceleration']} "
              f"({a['events']['hard_acceleration_per_min']:.2f}/min)")
        print(f"  Hard Braking Events: {a['events']['hard_braking']} "
              f"({a['events']['hard_braking_per_min']:.2f}/min)")
        print(f"  Sharp Turn Events: {a['events']['sharp_turns']} "
              f"({a['events']['sharp_turns_per_min']:.2f}/min)")
        print(f"\n  Moderate Acceleration: {a['events']['moderate_acceleration']}")
        print(f"  Moderate Braking: {a['events']['moderate_braking']}")
        print(f"  Moderate Turns: {a['events']['moderate_turns']}")

        print("\n" + "-" * 70)
        print("INTENSITY METRICS")
        print("-" * 70)
        print(f"\n  Max G-Force: {a['intensity']['max_gforce']:.3f}g")
        print(f"  Max Forward Acceleration: {a['intensity']['max_forward_accel']:.2f} m/s²")
        print(f"  Max Braking: {a['intensity']['max_backward_accel']:.2f} m/s²")
        print(f"  Max Lateral Acceleration: {a['intensity']['max_lateral_accel']:.2f} m/s²")
        print(f"  Mean Absolute Accel (Forward): {a['intensity']['mean_abs_accel_x']:.3f} m/s²")
        print(f"  Mean Absolute Accel (Lateral): {a['intensity']['mean_abs_accel_y']:.3f} m/s²")

        print("\n" + "-" * 70)
        print("SMOOTHNESS ANALYSIS")
        print("-" * 70)
        print(f"\n  Overall Smoothness Score: {a['smoothness']['overall_smoothness_score']:.1f}/100")
        print(f"  Acceleration Smoothness: {a['smoothness']['acceleration_smoothness_score']:.1f}/100")
        print(f"  Lateral Smoothness: {a['smoothness']['lateral_smoothness_score']:.1f}/100")
        print(f"  Forward/Backward Std Dev: {a['smoothness']['accel_x_std']:.3f} m/s²")
        print(f"  Lateral Std Dev: {a['smoothness']['accel_y_std']:.3f} m/s²")

        print("\n" + "-" * 70)
        print("RISK SCORE BREAKDOWN")
        print("-" * 70)
        print(f"\n  Hard Events Risk: {a['risk_scores']['hard_events_score']:.1f}/100 "
              f"(weight: {self.WEIGHTS['hard_events'] * 100:.0f}%)")
        print(f"  Moderate Events Risk: {a['risk_scores']['moderate_events_score']:.1f}/100 "
              f"(weight: {self.WEIGHTS['moderate_events'] * 100:.0f}%)")
        print(f"  G-Force Risk: {a['risk_scores']['gforce_score']:.1f}/100 "
              f"(weight: {self.WEIGHTS['gforce'] * 100:.0f}%)")
        print(f"  Smoothness Risk: {a['risk_scores']['smoothness_risk_score']:.1f}/100 "
              f"(weight: {self.WEIGHTS['smoothness'] * 100:.0f}%)")
        print(f"  Intensity Risk: {a['risk_scores']['intensity_risk_score']:.1f}/100 "
              f"(weight: {self.WEIGHTS['overall_intensity'] * 100:.0f}%)")

        print("\n" + "-" * 70)
        print("RECOMMENDATIONS")
        print("-" * 70)
        print()
        for i, rec in enumerate(a['recommendations'], 1):
            print(f"  {i}. {rec}")

        print("\n" + "=" * 70)
        print("END OF REPORT")
        print("=" * 70 + "\n")

    def save_report(self, reports_dir='reports', output_file=None):
        """Save assessment report as a JSON file under reports_dir"""
        Path(reports_dir).mkdir(parents=True, exist_ok=True)

        if output_file is None:
            output_file = Path(reports_dir) / f"{self.ride_id}.json"

        def convert_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(item) for item in obj]
            return obj

        assessment_serializable = convert_types(self.assessment)

        with open(output_file, 'w') as f:
            json.dump(assessment_serializable, f, indent=2)

        print(f"Report saved to: {output_file}")
        return str(output_file)


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description='Assess driving safety from Physics Toolbox sensor data'
    )
    parser.add_argument(
        'ride_file',
        help="Path to a ride parquet file, e.g. 'data/2026-02-24/<driver_id>/<ride_id>.parquet'"
    )
    parser.add_argument(
        '--reports-dir',
        default='reports',
        help='Directory to write JSON reports (default: reports)'
    )
    parser.add_argument(
        '--output',
        '-o',
        help='Custom output file path for a single JSON report'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        metavar='PATH',
        help='Path to YAML config file (default: config.yaml).',
    )

    args = parser.parse_args()

    assessor = DriveSafetyAssessor(args.ride_file, config_path=args.config)
    assessor.assess()
    assessor.print_report()
    assessor.save_report(reports_dir=args.reports_dir, output_file=args.output)


if __name__ == '__main__':
    main()
