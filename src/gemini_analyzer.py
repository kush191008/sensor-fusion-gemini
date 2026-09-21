"""
Gemini Diagnostic Analyzer: Uses Google Gemini (or Intelligent Cognitive Mock)
to evaluate sensor telemetry patterns, classify sensor health, estimate drift rates,
and compute dynamic Kalman filter adaptation parameters.
"""

import os
import json
import re
import warnings
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Try importing google.genai or google.generativeai
try:
    from google import genai
    from google.genai import types
    GENAI_CLIENT_AVAILABLE = True
except ImportError:
    GENAI_CLIENT_AVAILABLE = False

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        import google.generativeai as legacy_genai
    LEGACY_GENAI_AVAILABLE = True
except ImportError:
    LEGACY_GENAI_AVAILABLE = False


class GeminiSensorAnalyzer:
    """
    Cognitive Sensor Diagnostics Engine.
    Leverages Gemini 2.0 Flash for multi-sensor drift and failure detection.
    Features an autonomous mock simulation mode when offline or without API keys.
    """

    def __init__(self, api_key: Optional[str] = None):
        load_dotenv()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.is_live = False
        self.client = None

        if self.api_key and self.api_key.strip() and self.api_key != "your_gemini_api_key_here":
            try:
                if GENAI_CLIENT_AVAILABLE:
                    self.client = genai.Client(api_key=self.api_key.strip())
                    self.is_live = True
                elif LEGACY_GENAI_AVAILABLE:
                    legacy_genai.configure(api_key=self.api_key.strip())
                    self.legacy_model = legacy_genai.GenerativeModel('gemini-2.0-flash')
                    self.is_live = True
            except Exception as e:
                print(f"[WARN] Failed to initialize live Gemini client: {e}. Defaulting to Cognitive Mock Engine.")
                self.is_live = False
        else:
            print("[INFO] No Gemini API key detected. Running in Intelligent Cognitive Mock mode.")

    def analyze_stream(self, df: pd.DataFrame, window_size: int = 150) -> Dict[str, Dict[str, Any]]:
        """
        Analyzes all sensor channels in the telemetry DataFrame.
        Returns diagnostic reports and Kalman adaptation weights for each channel.
        """
        sensor_columns = [c for c in df.columns if c.startswith('sensor_')]
        diagnostics = {}

        # Compute multi-sensor spatial consensus (median across non-saturated sensors)
        sensor_matrix = df[sensor_columns].values
        # Filter out extreme rail saturation values (> 80.0) from consensus calculation
        masked_matrix = np.where(sensor_matrix > 80.0, np.nan, sensor_matrix)
        consensus = np.nanmedian(masked_matrix, axis=1)

        for sensor in sensor_columns:
            series = df[sensor].values
            diagnostics[sensor] = self.diagnose_sensor(sensor, series, consensus=consensus, window_size=window_size)

        return diagnostics

    def diagnose_sensor(
        self,
        sensor_name: str,
        readings: np.ndarray,
        consensus: Optional[np.ndarray] = None,
        window_size: int = 150
    ) -> Dict[str, Any]:
        """
        Diagnoses a single sensor channel using live Gemini or the Cognitive Heuristic Engine.
        """
        n = len(readings)
        recent_win = readings[-window_size:] if n >= window_size else readings
        early_win = readings[:window_size] if n >= window_size else readings

        mean_full = float(np.mean(readings))
        std_full = float(np.std(readings))
        min_full = float(np.min(readings))
        max_full = float(np.max(readings))

        early_mean = float(np.mean(early_win))
        recent_mean = float(np.mean(recent_win))
        overall_trend = float(recent_mean - early_mean)

        # High-frequency differential noise estimation (filters out slow physical process waves)
        diffs = np.diff(readings)
        estimated_noise_std = float(np.std(diffs) / np.sqrt(2.0)) if len(diffs) > 1 else 0.5

        # Residual analysis against multi-sensor consensus (unsupervised drift isolation)
        time_steps = np.arange(n)
        if consensus is not None and len(consensus) == n:
            residual = readings - consensus
            # Ignore points where sensor is rail-saturated
            valid_mask = readings < 80.0
            if np.sum(valid_mask) > 10:
                slope_per_sample = float(np.polyfit(time_steps[valid_mask], residual[valid_mask], 1)[0])
            else:
                slope_per_sample = 0.0
        else:
            slope_per_sample = float(np.polyfit(time_steps, readings, 1)[0]) if n > 1 else 0.0

        drift_per_100 = round(slope_per_sample * 100, 4)

        # Saturation & stuck-at fault metrics
        is_saturated = bool(np.sum(readings[-50:] >= 90.0) > 40)
        recent_noise = float(np.std(np.diff(recent_win)) / np.sqrt(2.0)) if len(recent_win) > 1 else 0.5

        # Build feature summary
        features = {
            "sensor_name": sensor_name,
            "total_samples": n,
            "mean": round(mean_full, 2),
            "std": round(std_full, 2),
            "min": round(min_full, 2),
            "max": round(max_full, 2),
            "estimated_noise_std": round(estimated_noise_std, 3),
            "recent_noise_std": round(recent_noise, 3),
            "trend_delta": round(overall_trend, 2),
            "drift_slope_per_100": drift_per_100,
            "is_saturated_stuck": is_saturated,
            "sample_tail": [round(float(x), 2) for x in readings[-10:]]
        }

        if self.is_live:
            try:
                return self._query_gemini_live(sensor_name, features)
            except Exception as e:
                print(f"[WARN] Live Gemini call failed for {sensor_name}: {e}. Falling back to Cognitive Mock.")

        return self._cognitive_mock_diagnose(sensor_name, features)

    def _query_gemini_live(self, sensor_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """Calls Google Gemini API with structured JSON output instructions."""
        prompt = f"""
You are a senior Principal IoT Sensor Diagnostics & Sensor Fusion Architect.
Analyze the statistical telemetry summary of this sensor:

Sensor Name: {sensor_name}
Features:
- Total Samples: {features['total_samples']}
- Global Mean: {features['mean']}
- Global Std Dev: {features['std']}
- Value Range: [{features['min']}, {features['max']}]
- Early Window Mean: {features['early_mean']} vs Recent Window Mean: {features['recent_mean']}
- Net Trend Shift: {features['trend_delta']}
- Linear Drift Slope per 100 samples: {features['drift_slope_per_100']}
- Recent Tail Variance: {features['recent_variance']}
- Saturated / Stuck-at rail: {features['is_saturated_stuck']}
- Last 10 Telemetry Readings: {features['sample_tail']}

Diagnose this sensor and respond ONLY with a valid JSON object matching this exact schema:
{{
    "status": "HEALTHY" | "DRIFTING" | "FAILED" | "NOISY",
    "has_drift": true | false,
    "drift_rate_per_100": float,
    "correction_bias": float,
    "recommended_noise_scalar": float,
    "confidence": int,
    "root_cause": "concise technical physical root cause explanation",
    "recommended_action": "Kalman adaptation recommendation"
}}

Rules:
- For FAILED / stuck sensors (e.g. saturated at 99): status must be "FAILED", noise scalar >= 1000.0 (isolate).
- For continuous systematic shift: status must be "DRIFTING", provide accurate drift_rate_per_100 and estimated correction_bias.
- For high burst variance: status must be "NOISY", noise scalar between 3.0 and 8.0.
- For stable baseline: status must be "HEALTHY", noise scalar = 1.0.
- Do NOT wrap in markdown explanation, return JSON only.
"""
        response_text = ""
        if GENAI_CLIENT_AVAILABLE and self.client:
            response = self.client.models.generate_content(
                model='gemini-2.0-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json"
                )
            )
            response_text = response.text
        elif LEGACY_GENAI_AVAILABLE and hasattr(self, 'legacy_model'):
            response = self.legacy_model.generate_content(prompt)
            response_text = response.text

        # Extract and parse JSON
        cleaned_text = response_text.strip()
        if '```' in cleaned_text:
            match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL)
            if match:
                cleaned_text = match.group(1).strip()

        parsed = json.loads(cleaned_text)
        # Ensure standard fields
        parsed['source'] = 'Gemini 2.0 Flash (Live API)'
        return parsed

    def _cognitive_mock_diagnose(self, sensor_name: str, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Autonomous Cognitive Heuristic Engine.
        Simulates Gemini's reasoning output deterministically based on physical features.
        """
        drift_slope = features['drift_slope_per_100']
        is_stuck = features['is_saturated_stuck']
        noise_std = features['estimated_noise_std']
        trend = features['trend_delta']

        if is_stuck or (features['max'] >= 90.0 and features['sample_tail'][-1] >= 90.0):
            return {
                "status": "FAILED",
                "has_drift": False,
                "drift_rate_per_100": 0.0,
                "correction_bias": 0.0,
                "recommended_noise_scalar": 10000.0,
                "confidence": 99,
                "root_cause": "Transducer hardware rail saturation / ADC latch-up lock at 99.0 reading.",
                "recommended_action": "Completely isolate sensor (set Kalman measurement weight to 0.0).",
                "source": "Gemini Cognitive Reasoner (Simulation Mode)"
            }
        elif abs(drift_slope) >= 1.0:
            # Drifting sensor (e.g. slope > 0.01 per sample = 1.0 per 100)
            return {
                "status": "DRIFTING",
                "has_drift": True,
                "drift_rate_per_100": drift_slope,
                "correction_bias": round(drift_slope * features['total_samples'] / 100.0, 2),
                "recommended_noise_scalar": 2.0,
                "confidence": 95,
                "root_cause": "Thermal transducer aging causing continuous positive calibration drift.",
                "recommended_action": f"Subtract dynamic slope correction ({drift_slope:.3f}/100 samples) and adapt Kalman R by 2.0x.",
                "source": "Gemini Cognitive Reasoner (Simulation Mode)"
            }
        elif noise_std > 0.65 or features.get('recent_noise_std', 0.0) > 0.65:
            return {
                "status": "NOISY",
                "has_drift": False,
                "drift_rate_per_100": drift_slope,
                "correction_bias": 0.0,
                "recommended_noise_scalar": 3.5,
                "confidence": 91,
                "root_cause": "Intermittent environmental turbulence causing elevated heteroskedastic measurement noise.",
                "recommended_action": "Inflate measurement covariance R by 3.5x to rely more on predictive state model.",
                "source": "Gemini Cognitive Reasoner (Simulation Mode)"
            }
        else:
            return {
                "status": "HEALTHY",
                "has_drift": False,
                "drift_rate_per_100": drift_slope,
                "correction_bias": 0.0,
                "recommended_noise_scalar": 1.0,
                "confidence": 98,
                "root_cause": "Sensor operating within nominal precision thresholds and stationary noise profile.",
                "recommended_action": "Maintain 100% nominal trust in Kalman measurement update.",
                "source": "Gemini Cognitive Reasoner (Simulation Mode)"
            }


if __name__ == "__main__":
    from sensor_simulator import generate_sensor_stream
    df_sample = generate_sensor_stream(n_samples=800)
    analyzer = GeminiSensorAnalyzer()
    print("\n--- Testing Gemini Diagnostic Analyzer ---")
    results = analyzer.analyze_stream(df_sample)
    for s_name, diag in results.items():
        print(f"\n[{s_name.upper()}] -> Status: {diag['status']} ({diag['confidence']}%)")
        print(f"  Root Cause: {diag['root_cause']}")
        print(f"  Recommended Action: {diag['recommended_action']}")
        print(f"  Noise Scalar (R multiplier): {diag['recommended_noise_scalar']}")
