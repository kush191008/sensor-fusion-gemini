"""
Sensor Simulator: Generates realistic multi-sensor telemetry across multiple industrial domains
with intentional drift, hardware failure, dynamic noise, known ground truth, physical spatial coordinates,
and cyber-physical adversarial attack injection (Chaos Monkey).
"""

import os
from enum import Enum
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd


class MissionScenario(str, Enum):
    INDUSTRIAL_TURBINE = "Industrial Power Plant Gas Turbine"
    AEROSPACE_DRONE = "Autonomous Drone Flight Controller"
    SMART_AGRICULTURE = "Smart Agriculture Climate Station"


class CyberAttackType(str, Enum):
    NONE = "Nominal (No Attack)"
    EMI_SURGE = "⚡ High-Voltage EMI Lightning Surge"
    SPOOFING = "🕵️ Man-in-the-Middle Sensor Spoofing"
    CRYO_FREEZE = "🧊 Cryogenic Transducer Freeze"


SCENARIO_CONFIGS = {
    MissionScenario.INDUSTRIAL_TURBINE: {
        "title": "Industrial Power Plant Gas Turbine",
        "description": "Monitors turbine combustion temperature, casing vibration, and lubrication oil pressure.",
        "target_unit": "°C",
        "base_val": 650.0,
        "osc_amplitude": 25.0,
        "labels": {
            "sensor_temp": "Exhaust Gas Temp Probe (Thermal Degradation Drift)",
            "sensor_baseline": "Redundant Core Thermocouple (Healthy Baseline)",
            "sensor_humidity": "Lube Oil Pressure Transducer (ADC Saturation Lockup)",
            "sensor_aux": "Bearing Casing Accelerometer (Transient Vibration Turbulence)"
        },
        "short_labels": {
            "sensor_temp": "EGT Probe 1",
            "sensor_baseline": "Core Temp 2",
            "sensor_humidity": "Lube Pressure",
            "sensor_aux": "Vibration Sensor"
        },
        "coordinates": {
            "sensor_temp": {"x": 82, "y": 50, "zone": "Exhaust Manifold Duct", "temp_c": 645},
            "sensor_baseline": {"x": 50, "y": 50, "zone": "Combustion Chamber Core", "temp_c": 652},
            "sensor_humidity": {"x": 22, "y": 30, "zone": "Lubrication Sump Return", "temp_c": 85},
            "sensor_aux": {"x": 35, "y": 72, "zone": "HP Bearing Casing", "temp_c": 110}
        }
    },
    MissionScenario.AEROSPACE_DRONE: {
        "title": "Autonomous Drone Flight Controller",
        "description": "Monitors UAV flight dynamics, altitude estimation, and inertial measurement telemetry.",
        "target_unit": "meters",
        "base_val": 120.0,
        "osc_amplitude": 15.0,
        "labels": {
            "sensor_temp": "Barometric Altimeter (Diurnal Thermal Bias Drift)",
            "sensor_baseline": "LiDAR Rangefinder (Precision Altitude Baseline)",
            "sensor_humidity": "Differential Airspeed Pitot (Icing Rail Lockup)",
            "sensor_aux": "IMU Vertical Accelerometer (Wind Shear Turbulences)"
        },
        "short_labels": {
            "sensor_temp": "Baro Altimeter",
            "sensor_baseline": "LiDAR Ground",
            "sensor_humidity": "Pitot Airspeed",
            "sensor_aux": "IMU Z-Accel"
        },
        "coordinates": {
            "sensor_temp": {"x": 48, "y": 52, "zone": "Avionics Core Bay", "temp_c": 38},
            "sensor_baseline": {"x": 50, "y": 18, "zone": "Ventral Gimbal Pod", "temp_c": 22},
            "sensor_humidity": {"x": 88, "y": 50, "zone": "Nose Pitot Probe", "temp_c": -5},
            "sensor_aux": {"x": 50, "y": 80, "zone": "Center Airframe Bulkhead", "temp_c": 32}
        }
    },
    MissionScenario.SMART_AGRICULTURE: {
        "title": "Smart Agriculture Climate Station",
        "description": "Monitors micro-climate soil potential, atmospheric temperature, and humidity sensors.",
        "target_unit": "°C",
        "base_val": 25.0,
        "osc_amplitude": 3.5,
        "labels": {
            "sensor_temp": "Ambient Thermocouple (Aging Calibration Drift)",
            "sensor_baseline": "Platinum RTD Reference (Healthy Baseline)",
            "sensor_humidity": "Capacitive Humidity Sensor (Electrolyte Saturation Lockup)",
            "sensor_aux": "Canopy Pyranometer (Cloud Intermittency Noise)"
        },
        "short_labels": {
            "sensor_temp": "Thermocouple",
            "sensor_baseline": "RTD Reference",
            "sensor_humidity": "Humidity Sensor",
            "sensor_aux": "Pyranometer"
        },
        "coordinates": {
            "sensor_temp": {"x": 50, "y": 72, "zone": "Upper Canopy Mast", "temp_c": 26},
            "sensor_baseline": {"x": 28, "y": 72, "zone": "Radiation Shield Enclosure", "temp_c": 25},
            "sensor_humidity": {"x": 50, "y": 22, "zone": "Subsurface Root Matrix", "temp_c": 19},
            "sensor_aux": {"x": 76, "y": 88, "zone": "Solar Radiation Arm", "temp_c": 31}
        }
    }
}


def generate_sensor_stream(
    n_samples: int = 1000,
    seed: int = 42,
    drift_rate: float = 0.035,
    failure_step: Optional[int] = None,
    scenario: MissionScenario = MissionScenario.SMART_AGRICULTURE,
    attack: CyberAttackType = CyberAttackType.NONE
) -> pd.DataFrame:
    """
    Simulates a 4-channel telemetry array measuring a physical process across domain scenarios
    with optional cyber-physical attack injection.
    """
    if failure_step is None or failure_step >= n_samples:
        failure_step = int(n_samples * 0.7)

    np.random.seed(seed)
    t = np.arange(n_samples)

    cfg = SCENARIO_CONFIGS.get(scenario, SCENARIO_CONFIGS[MissionScenario.SMART_AGRICULTURE])
    base_val = cfg["base_val"]
    amp = cfg["osc_amplitude"]

    # Ground Truth Physical Process
    ground_truth = base_val + amp * np.sin(t / 90.0) + (amp * 0.35) * np.cos(t / 25.0)

    # Sensor 1 (Drift): Systematic calibration degradation
    noise_scale_1 = amp * 0.1
    noise_1 = np.random.normal(0, noise_scale_1, n_samples)
    drift = (drift_rate * amp * 0.5) * t
    sensor_temp = ground_truth + drift + noise_1

    # Sensor 2 (Baseline): High-accuracy reference sensor
    noise_scale_2 = amp * 0.12
    noise_2 = np.random.normal(0, noise_scale_2, n_samples)
    sensor_baseline = ground_truth + noise_2

    # Sensor 3 (Failure): Catastrophic lockup at saturation value
    noise_scale_3 = amp * 0.14
    noise_3 = np.random.normal(0, noise_scale_3, n_samples)
    sensor_humidity = ground_truth.copy() + noise_3
    failed_len = n_samples - failure_step
    if failed_len > 0:
        if scenario == MissionScenario.SMART_AGRICULTURE:
            saturation_val = 99.0
        elif scenario == MissionScenario.INDUSTRIAL_TURBINE:
            saturation_val = base_val + amp * 5.0
        else:  # AEROSPACE_DRONE
            saturation_val = base_val + amp * 5.0
        sensor_humidity[failure_step:] = saturation_val + np.random.normal(0, 0.05, failed_len)

    # Sensor 4 (Auxiliary): Intermittent non-stationary turbulence
    base_noise_4 = np.random.normal(0, amp * 0.11, n_samples)
    burst_start = int(n_samples * 0.3)
    burst_end = int(n_samples * 0.45)
    burst_mask = (t >= burst_start) & (t <= burst_end)
    base_noise_4[burst_mask] += np.random.normal(0, amp * 0.6, np.sum(burst_mask))
    sensor_aux = ground_truth + base_noise_4

    # Cyber-Physical Attack Injection (Chaos Monkey)
    if attack == CyberAttackType.EMI_SURGE:
        # High-voltage surge at t=500 to 520 across multiple channels
        surge_idx = np.arange(int(n_samples * 0.5), min(int(n_samples * 0.5) + 20, n_samples))
        sensor_temp[surge_idx] += amp * 4.5
        sensor_aux[surge_idx] -= amp * 3.8
    elif attack == CyberAttackType.SPOOFING:
        # Stealthy quadratic bias spoofing on baseline reference sensor starting at t=400
        spoof_idx = np.arange(int(n_samples * 0.4), n_samples)
        spoof_ramp = 0.0005 * amp * ((spoof_idx - spoof_idx[0]) ** 1.6)
        sensor_baseline[spoof_idx] += spoof_ramp
    elif attack == CyberAttackType.CRYO_FREEZE:
        # Freeze sensor 4 at t=350 to constant rail
        freeze_idx = np.arange(int(n_samples * 0.35), n_samples)
        sensor_aux[freeze_idx] = float(ground_truth[freeze_idx[0]])

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
    """Generates and saves the sample sensor dataset."""
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
