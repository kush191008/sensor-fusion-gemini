"""
Unit and Integration Tests for Sensor Fusion Pipeline.
Validates multi-scenario telemetry generation, Gemini diagnostics, Copilot, and Audit Report generation.
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# Add src to system path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))

from sensor_simulator import generate_sensor_stream, MissionScenario
from gemini_analyzer import GeminiSensorAnalyzer
from fusion_engine import AdaptiveKalmanFusion


class TestSensorFusionPipeline(unittest.TestCase):

    def setUp(self):
        self.df = generate_sensor_stream(n_samples=500, drift_rate=0.04, failure_step=350)
        self.analyzer = GeminiSensorAnalyzer()
        self.diagnostics = self.analyzer.analyze_stream(self.df)
        self.fusion = AdaptiveKalmanFusion()

    def test_multi_scenario_generation(self):
        """Verify telemetry stream generation across all mission scenarios."""
        for scenario in [MissionScenario.INDUSTRIAL_TURBINE, MissionScenario.AEROSPACE_DRONE, MissionScenario.SMART_AGRICULTURE]:
            df_scen = generate_sensor_stream(n_samples=400, scenario=scenario)
            self.assertEqual(len(df_scen), 400)
            self.assertFalse(df_scen.isna().any().any())
            self.assertIn('ground_truth', df_scen.columns)

    def test_gemini_analyzer_diagnostics(self):
        """Verify diagnostics contain required fields and correctly identify faults."""
        for col in ['sensor_temp', 'sensor_baseline', 'sensor_humidity', 'sensor_aux']:
            self.assertIn(col, self.diagnostics)
            diag = self.diagnostics[col]
            self.assertIn('status', diag)
            self.assertIn('confidence', diag)
            self.assertIn('recommended_noise_scalar', diag)
            self.assertGreaterEqual(diag['confidence'], 50)

        # Humidity/Pressure transducer should be detected as FAILED
        self.assertEqual(self.diagnostics['sensor_humidity']['status'], 'FAILED')
        # Temperature/Drifting transducer should have drift detected
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

    def test_telemetry_copilot(self):
        """Verify the Gemini Telemetry Copilot answers technical queries with grounded context."""
        context = {
            'scenario_title': 'Industrial Gas Turbine',
            'n_samples': 500,
            'current_fused_val': 652.4,
            'current_uncertainty': 0.38,
            'diagnostics': self.diagnostics
        }
        res = self.analyzer.chat_with_telemetry("Why did you isolate Sensor 3?", context)
        self.assertIsInstance(res, str)
        self.assertIn("Sensor 3", res)
        self.assertTrue("isolate" in res.lower() or "rail" in res.lower())

    def test_audit_report_generation(self):
        """Verify automated generation of the ISO/IEEE Incident Audit Report."""
        context = {
            'scenario_title': 'Industrial Gas Turbine',
            'n_samples': 500,
            'current_fused_val': 652.4,
            'current_uncertainty': 0.38,
            'diagnostics': self.diagnostics,
            'naive_rmse': 14.5,
            'fused_rmse': 0.31,
            'improvement_pct': 97.8,
            'coverage_pct': 98.5
        }
        report = self.analyzer.generate_incident_audit_report(context)
        self.assertIsInstance(report, str)
        self.assertIn("AUDIT REPORT", report)
        self.assertIn("Transducer Channel Diagnostic Audit", report)


if __name__ == '__main__':
    unittest.main()
