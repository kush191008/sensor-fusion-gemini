"""
Sensor Simulator: Generates realistic multi-sensor telemetry with intentional
drift, hardware failure, dynamic noise, and known ground truth for validation.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

def generate_sensor_stream(
    n_samples: int = 1000,
    seed: int = 42,
    drift_rate: float = 0.035,
    failure_step: Optional[int] = None
) -> pd.DataFrame:
    """
    Simulates a 4-sensor industrial telemetry array measuring an environmental process:
    - ground_truth: The underlying true environmental physical state (for benchmark evaluation).
    - sensor_temp (Drifting): Gradual linear & thermal aging drift.
    - sensor_baseline (Healthy): Reference sensor with standard white noise.
    - sensor_humidity (Failure): Operates normally until failure_step, then locks to saturation value.
    - sensor_aux (Intermittent Noise): Environmental turbulence with heteroskedastic variance.
    """
    if failure_step is None or failure_step >= n_samples:
        failure_step = int(n_samples * 0.7)

    np.random.seed(seed)
    t = np.arange(n_samples)

    # 1. Ground Truth Physical Process (e.g. ambient environmental temperature dynamics)
    # Includes slow oscillation (ambient trend) + higher frequency variation
    ground_truth = 25.0 + 3.5 * np.sin(t / 90.0) + 1.2 * np.cos(t / 25.0)

    # 2. Sensor 1: Temperature Sensor with systematic calibration drift
    noise_1 = np.random.normal(0, 0.35, n_samples)
    drift = drift_rate * t
    sensor_temp = ground_truth + drift + noise_1

    # 3. Sensor 2: High-reliability Reference Sensor (Healthy baseline)
    noise_2 = np.random.normal(0, 0.45, n_samples)
    sensor_baseline = ground_truth + noise_2

    # 4. Sensor 3: Transducer with sudden catastrophic lockup / failure at failure_step
    noise_3 = np.random.normal(0, 0.50, n_samples)
    sensor_humidity = ground_truth.copy() + noise_3
    # Fail at failure_step: locks out at saturation reading (99.0)
    failed_len = n_samples - failure_step
    if failed_len > 0:
        sensor_humidity[failure_step:] = 99.0 + np.random.normal(0, 0.05, failed_len)

    # 5. Sensor 4: Auxiliary sensor subjected to intermittent environmental bursts
    base_noise_4 = np.random.normal(0, 0.40, n_samples)
    # Intermittent noise burst between t=300 and t=450
    burst_mask = (t >= 300) & (t <= 450)
    base_noise_4[burst_mask] += np.random.normal(0, 2.2, np.sum(burst_mask))
    sensor_aux = ground_truth + base_noise_4

    df = pd.DataFrame({
        'timestamp': t,
        'ground_truth': np.round(ground_truth, 4),
        'sensor_temp': np.round(sensor_temp, 4),
        'sensor_baseline': np.round(sensor_baseline, 4),
        'sensor_humidity': np.round(sensor_humidity, 4),
        'sensor_aux': np.round(sensor_aux, 4)
    })

    return df

def save_sample_data(filepath: str = None) -> str:
    """Generates and persists the sample sensor dataset."""
    if filepath is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        data_dir = os.path.join(base_dir, 'data')
        os.makedirs(data_dir, exist_ok=True)
        filepath = os.path.join(data_dir, 'sample_sensor_data.csv')

    df = generate_sensor_stream()
    df.to_csv(filepath, index=False)
    print(f"[OK] Generated {len(df)} samples saved to: {filepath}")
    return filepath

if __name__ == "__main__":
    save_sample_data()
