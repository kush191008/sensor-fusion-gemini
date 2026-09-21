"""
Adaptive Kalman Fusion Engine with Bayesian Uncertainty Quantification.
Dynamically reweights sensor measurement covariance based on cognitive AI diagnostics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional


class AdaptiveKalmanFusion:
    """
    2-State Discrete Kalman Filter (Position and Velocity / Rate of Change).
    Adapts observation matrices and measurement noise covariances dynamically
    according to Gemini's sensor health and drift assessments.
    """

    def __init__(
        self,
        dt: float = 1.0,
        process_noise_q: float = 0.08,
        base_measurement_variance: float = 0.35
    ):
        self.dt = dt
        # State: [estimated_process_value, rate_of_change]
        self.x = np.array([25.0, 0.0], dtype=float)

        # State transition matrix F
        self.F = np.array([
            [1.0, self.dt],
            [0.0, 1.0]
        ], dtype=float)

        # Process noise covariance Q (Piecewise white noise acceleration)
        dt2 = self.dt ** 2
        dt3 = self.dt ** 3 / 3.0
        dt2_2 = self.dt ** 2 / 2.0
        self.Q = process_noise_q * np.array([
            [dt3, dt2_2],
            [dt2_2, dt]
        ], dtype=float)

        # State estimation error covariance P
        self.P = np.eye(2, dtype=float) * 2.0

        self.base_variance = base_measurement_variance

    def run_fusion_pipeline(
        self,
        df: pd.DataFrame,
        diagnostics: Dict[str, Dict[str, Any]]
    ) -> pd.DataFrame:
        """
        Executes complete fusion loop over telemetry dataframe.
        Returns dataframe augmented with:
        - fused_estimate
        - uncertainty_sigma
        - uncertainty_upper (fused + 2*sigma)
        - uncertainty_lower (fused - 2*sigma)
        - raw_naive_average (unweighted average for comparison)
        """
        sensor_cols = [c for c in df.columns if c.startswith('sensor_')]
        m = len(sensor_cols)

        # Extract drift and adaptation parameters from Gemini diagnostics
        drift_slopes = []
        noise_scalars = []
        is_active = []

        for col in sensor_cols:
            diag = diagnostics.get(col, {})
            status = diag.get('status', 'HEALTHY')
            slope = diag.get('drift_rate_per_100', 0.0)
            scalar = diag.get('recommended_noise_scalar', 1.0)

            # If sensor suffered catastrophic failure, its pre-trip noise is nominal (1.0),
            # and it will be completely excluded once it trips the rail lockup (>80.0)
            if status == 'FAILED':
                scalar = 1.0

            drift_slopes.append(slope / 100.0)
            noise_scalars.append(scalar)
            is_active.append(True)

        drift_slopes = np.array(drift_slopes)
        noise_scalars = np.array(noise_scalars)

        # Pre-allocate output arrays
        n = len(df)
        fused_series = np.zeros(n)
        uncertainty_sigma = np.zeros(n)
        naive_average = np.zeros(n)

        # Reset filter state to initial reading
        first_readings = df[sensor_cols].iloc[0].values
        self.x = np.array([float(np.median(first_readings)), 0.0])
        self.P = np.eye(2) * 1.5

        for k in range(n):
            raw_readings = df[sensor_cols].iloc[k].values
            naive_average[k] = np.mean(raw_readings)

            # 1. State Prediction
            x_pred = self.F @ self.x
            P_pred = self.F @ self.P @ self.F.T + self.Q

            # 2. Dynamic Correction of Drifting Sensors
            # Correct systematic drift: reading - (slope * k)
            corrected_measurements = raw_readings.copy()
            for i in range(m):
                if drift_slopes[i] != 0.0 and diagnostics.get(sensor_cols[i], {}).get('has_drift', False):
                    corrected_measurements[i] -= (drift_slopes[i] * k)

            # 3. Dynamic Measurement Noise Covariance R_k
            # If sensor saturated or failed at this step, set its noise to near-infinite
            R_diag = []
            valid_indices = []
            for i in range(m):
                reading_val = raw_readings[i]
                sensor_stat = diagnostics.get(sensor_cols[i], {}).get('status', 'HEALTHY')
                
                # Real-time sanity gate: exclude sensor if reading is stuck/saturated (>80) 
                # or if permanently decommissioned
                if reading_val > 80.0 or (sensor_stat == 'FAILED' and reading_val > 50.0):
                    continue

                var = self.base_variance * noise_scalars[i]
                R_diag.append(var)
                valid_indices.append(i)

            if len(valid_indices) > 0:
                # Active sensor observation matrix H
                H = np.zeros((len(valid_indices), 2))
                H[:, 0] = 1.0

                z_active = corrected_measurements[valid_indices]
                R_mat = np.diag(R_diag)

                # Innovation
                y = z_active - (H @ x_pred)
                S = H @ P_pred @ H.T + R_mat

                # Kalman Gain
                K = P_pred @ H.T @ np.linalg.inv(S)

                # State Update
                self.x = x_pred + K @ y

                # Joseph stabilized covariance update
                I_KH = np.eye(2) - K @ H
                self.P = I_KH @ P_pred @ I_KH.T + K @ R_mat @ K.T
            else:
                # No valid sensors available: propagate pure prediction
                self.x = x_pred
                self.P = P_pred

            # Record state and Bayesian 1-sigma uncertainty
            fused_series[k] = self.x[0]
            # Standard deviation of state uncertainty
            sigma = float(np.sqrt(max(self.P[0, 0], 1e-6)))
            uncertainty_sigma[k] = sigma

        out_df = df.copy()
        out_df['naive_average'] = np.round(naive_average, 4)
        out_df['fused_estimate'] = np.round(fused_series, 4)
        out_df['uncertainty_sigma'] = np.round(uncertainty_sigma, 4)
        out_df['uncertainty_upper'] = np.round(fused_series + 2.0 * uncertainty_sigma, 4)
        out_df['uncertainty_lower'] = np.round(fused_series - 2.0 * uncertainty_sigma, 4)

        if 'ground_truth' in out_df.columns:
            out_df['naive_error'] = np.abs(out_df['naive_average'] - out_df['ground_truth'])
            out_df['fused_error'] = np.abs(out_df['fused_estimate'] - out_df['ground_truth'])

        return out_df

    @staticmethod
    def calculate_performance_metrics(fusion_results_df: pd.DataFrame) -> Dict[str, float]:
        """Calculates benchmark comparative metrics (RMSE, MAE, Uncertainty Coverage)."""
        metrics = {}
        if 'ground_truth' not in fusion_results_df.columns:
            return metrics

        gt = fusion_results_df['ground_truth'].values
        naive = fusion_results_df['naive_average'].values
        fused = fusion_results_df['fused_estimate'].values
        upper = fusion_results_df['uncertainty_upper'].values
        lower = fusion_results_df['uncertainty_lower'].values

        metrics['naive_rmse'] = float(np.sqrt(np.mean((naive - gt) ** 2)))
        metrics['fused_rmse'] = float(np.sqrt(np.mean((fused - gt) ** 2)))
        metrics['rmse_improvement_pct'] = float(
            ((metrics['naive_rmse'] - metrics['fused_rmse']) / max(metrics['naive_rmse'], 1e-6)) * 100.0
        )
        metrics['naive_mae'] = float(np.mean(np.abs(naive - gt)))
        metrics['fused_mae'] = float(np.mean(np.abs(fused - gt)))

        # Bayesian Credible Interval Coverage (should be ~95% for 2-sigma)
        in_bounds = (gt >= lower) & (gt <= upper)
        metrics['confidence_interval_coverage_pct'] = float(np.mean(in_bounds) * 100.0)

        return metrics


if __name__ == "__main__":
    from sensor_simulator import generate_sensor_stream
    from gemini_analyzer import GeminiSensorAnalyzer

    df_test = generate_sensor_stream(n_samples=1000)
    analyzer = GeminiSensorAnalyzer()
    diagnostics = analyzer.analyze_stream(df_test)

    fusion = AdaptiveKalmanFusion()
    results = fusion.run_fusion_pipeline(df_test, diagnostics)
    perf = fusion.calculate_performance_metrics(results)

    print("\n--- Fusion Performance Summary ---")
    print(f"Naive Unweighted Average RMSE: {perf['naive_rmse']:.3f}")
    print(f"Gemini-Adaptive Kalman RMSE:   {perf['fused_rmse']:.3f}")
    print(f"Accuracy Improvement:          {perf['rmse_improvement_pct']:.1f}%")
    print(f"Bayesian 95% Coverage:         {perf['confidence_interval_coverage_pct']:.1f}%")
