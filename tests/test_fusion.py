"""
Unit and Integration Tests for Sensor Fusion Pipeline.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from sensor_simulator import generate_sensor_stream
from gemini_analyzer import GeminiSensorAnalyzer
from fusion_engine import AdaptiveKalmanFusion


class TestSensorFusionPipeline(unittest.TestCase):

    def setUp(self):
        self.df = generate_sensor_stream(n_samples=500, drift_rate=0.04, failure_step=350)
        self.analyzer = GeminiSensorAnalyzer()
        self.diagnostics = self.analyzer.analyze_stream(self.df)
        self.fusion = AdaptiveKalmanFusion()

    def test_sensor_stream_generation(self):
        """Verify data generator produces valid, non-empty, shaped dataframe."""
        self.assertEqual(len(self.df), 500)
        expected_cols = {'timestamp', 'ground_truth', 'sensor_temp', 'sensor_baseline', 'sensor_humidity', 'sensor_aux'}
        self.assertTrue(expected_cols.issubset(set(self.df.columns)))
        self.assertFalse(self.df.isna().any().any())

    def test_gemini_analyzer_diagnostics(self):
        """Verify diagnostics contain required fields and correctly identify faults."""
        for col in ['sensor_temp', 'sensor_baseline', 'sensor_humidity', 'sensor_aux']:
            self.assertIn(col, self.diagnostics)
            diag = self.diagnostics[col]
            self.assertIn('status', diag)
            self.assertIn('confidence', diag)
            self.assertIn('recommended_noise_scalar', diag)
            self.assertGreaterEqual(diag['confidence'], 50)

        # Humidity sensor should be detected as FAILED
        self.assertEqual(self.diagnostics['sensor_humidity']['status'], 'FAILED')
        # Temp sensor should have drift detected
        self.assertTrue(self.diagnostics['sensor_temp']['has_drift'])

    def test_adaptive_kalman_performance(self):
        """Verify Kalman filter achieves substantial RMSE reduction and uncertainty bounds."""
        results = self.fusion.run_fusion_pipeline(self.df, self.diagnostics)
        metrics = self.fusion.calculate_performance_metrics(results)

        # Verification that Fused RMSE is significantly lower than naive average
        self.assertLess(metrics['fused_rmse'], metrics['naive_rmse'])
        self.assertGreater(metrics['rmse_improvement_pct'], 40.0)

        # Verification of Bayesian uncertainty coverage (around 95%)
        self.assertGreaterEqual(metrics['confidence_interval_coverage_pct'], 80.0)

        # Uncertainty sigma should be higher after failure at step 350 than before
        pre_failure_sigma = np.mean(results['uncertainty_sigma'].iloc[100:300])
        post_failure_sigma = np.mean(results['uncertainty_sigma'].iloc[380:480])
        self.assertGreater(post_failure_sigma, pre_failure_sigma)


if __name__ == '__main__':
    unittest.main()
