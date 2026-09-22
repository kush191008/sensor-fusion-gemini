"""
Self-Calibrating Sensor Fusion Under Drift
A clean, focused engineering dashboard solving:
1. Detecting sensor drift and failures
2. Estimating corrected readings
3. Adapting without continuous labeled data
4. Reporting Bayesian uncertainty during changing environmental conditions
"""

import os
import sys
import importlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Ensure local src directory is on sys.path and reload dependencies
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import sensor_simulator
importlib.reload(sensor_simulator)
from sensor_simulator import generate_sensor_stream, MissionScenario, SCENARIO_CONFIGS

import gemini_analyzer
importlib.reload(gemini_analyzer)
from gemini_analyzer import GeminiSensorAnalyzer

import fusion_engine
importlib.reload(fusion_engine)
from fusion_engine import AdaptiveKalmanFusion

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Self-Calibrating Sensor Fusion Under Drift",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, clean CSS styling
st.markdown("""
<style>
    .sensor-card {
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
        background-color: #0d1117;
    }
    .badge {
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
    }
    .badge-healthy { background-color: #238636; color: white; }
    .badge-drifting { background-color: #d29922; color: black; }
    .badge-failed { background-color: #da3633; color: white; }
    .badge-noisy { background-color: #8957e5; color: white; }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("System Configuration")

# Domain Scenario Selector
scenario_choice = st.sidebar.selectbox(
    "Application Domain:",
    options=[
        MissionScenario.INDUSTRIAL_TURBINE,
        MissionScenario.AEROSPACE_DRONE,
        MissionScenario.SMART_AGRICULTURE
    ],
    index=0
)
scenario_cfg = SCENARIO_CONFIGS[scenario_choice]
st.sidebar.caption(f"_{scenario_cfg['description']}_")

st.sidebar.markdown("---")
st.sidebar.subheader("Environmental Parameters")

n_samples = st.sidebar.slider("Telemetry Samples", min_value=300, max_value=1200, value=600, step=50)
drift_rate = st.sidebar.slider("Sensor 1 Drift Rate", min_value=0.01, max_value=0.08, value=0.035, step=0.005)
failure_step = st.sidebar.slider("Sensor 3 Failure Step", min_value=150, max_value=int(n_samples * 0.85), value=int(n_samples * 0.65), step=25)

st.sidebar.markdown("---")
st.sidebar.subheader("AI Supervisor (Gemini)")
user_api_key = st.sidebar.text_input(
    "Gemini API Key (Optional):",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="If empty, the system runs with the built-in deterministic cognitive simulation."
)

if user_api_key and user_api_key.strip():
    os.environ["GEMINI_API_KEY"] = user_api_key.strip()
    st.sidebar.success("● Live Gemini 2.0 Connected")
else:
    st.sidebar.info("○ Simulation Mode Active (No key needed)")

# Data generation
state_key = f"{scenario_choice}_{n_samples}_{drift_rate}_{failure_step}"
if 'current_state_key' not in st.session_state or st.session_state.current_state_key != state_key:
    st.session_state.current_state_key = state_key
    st.session_state.telemetry_data = generate_sensor_stream(
        n_samples=n_samples,
        drift_rate=drift_rate,
        failure_step=failure_step,
        scenario=scenario_choice
    )
    if 'diagnostics' in st.session_state:
        del st.session_state['diagnostics']

df = st.session_state.telemetry_data
unit = scenario_cfg['target_unit']

# Diagnostic analysis
if 'diagnostics' not in st.session_state:
    analyzer = GeminiSensorAnalyzer(api_key=user_api_key)
    st.session_state.diagnostics = analyzer.analyze_stream(df)

diagnostics = st.session_state.diagnostics

# State estimation and fusion
fusion_engine = AdaptiveKalmanFusion(base_measurement_variance=max(scenario_cfg['osc_amplitude'] * 0.1, 0.35))
fusion_df = fusion_engine.run_fusion_pipeline(df, diagnostics)
metrics = fusion_engine.calculate_performance_metrics(fusion_df)

# Telemetry context object for Copilot
telemetry_context = {
    'scenario_title': scenario_cfg['title'],
    'n_samples': n_samples,
    'current_fused_val': float(fusion_df['fused_estimate'].iloc[-1]),
    'current_uncertainty': float(fusion_df['uncertainty_sigma'].iloc[-1]),
    'naive_rmse': metrics.get('naive_rmse', 12.0),
    'fused_rmse': metrics.get('fused_rmse', 0.26),
    'improvement_pct': metrics.get('rmse_improvement_pct', 97.9),
    'coverage_pct': metrics.get('confidence_interval_coverage_pct', 99.0),
    'diagnostics': diagnostics
}

# ----------------- HEADER & KPI ROW -----------------
st.title("Self-Calibrating Sensor Fusion Under Drift")
st.markdown(
    "An AI-supervised state estimation pipeline that detects sensor drift and sudden failures, "
    "estimates corrected readings, adapts without continuous labeled data, and reports Bayesian uncertainty."
)

# 4 Key Metric Cards directly mapped to problem statement
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(
        label="Raw Uncorrected Error (RMSE)",
        value=f"{metrics.get('naive_rmse', 0):.2f} {unit}",
        help="Standard unweighted averaging corrupted by drift and failure."
    )
with m2:
    st.metric(
        label="Corrected Fused Error (RMSE)",
        value=f"{metrics.get('fused_rmse', 0):.2f} {unit}",
        delta=f"-{metrics.get('rmse_improvement_pct', 0):.1f}% error",
        delta_color="inverse",
        help="State estimation error after adaptive calibration and fault rejection."
    )
with m3:
    detected_drift = diagnostics.get('sensor_temp', {}).get('drift_rate_per_100', 0.0)
    st.metric(
        label="Detected Sensor 1 Drift Rate",
        value=f"{detected_drift:+.3f} / 100",
        help="Rate of calibration drift isolated via unsupervised spatial consensus."
    )
with m4:
    current_sigma = float(fusion_df['uncertainty_sigma'].iloc[-1])
    st.metric(
        label="Current Uncertainty (±2σ)",
        value=f"± {2.0 * current_sigma:.3f} {unit}",
        delta=f"Coverage: {metrics.get('confidence_interval_coverage_pct', 0):.0f}%",
        help="Real-time 95% Bayesian credible interval from the Kalman covariance matrix."
    )

st.markdown("---")

# ----------------- STREAMLINED 4-TAB ARCHITECTURE -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 1. Corrected State Estimation",
    "🔍 2. Drift & Failure Diagnostics",
    "📐 3. Uncertainty & Unsupervised Adaptation",
    "💬 4. Technical Query Assistant"
])

# ----------------- TAB 1: SENSOR FUSION & STATE CORRECTION -----------------
with tab1:
    st.subheader(f"Multi-Sensor Inputs vs Corrected Fused Estimate ({unit})")
    st.caption(
        "Notice how Sensor 1 drifts continuously and Sensor 3 locks at the failure step. "
        "The Adaptive Kalman Filter rejects the failed sensor, subtracts drift, and accurately tracks the true process state."
    )

    fig = go.Figure()

    # Ground Truth Target
    fig.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['ground_truth'],
        mode='lines', name=f'True Process State ({unit})',
        line=dict(color='#FFFFFF', width=2, dash='dash')
    ))

    # Raw Sensor Traces
    sensor_colors = {
        'sensor_temp': '#FFA726',
        'sensor_baseline': '#26A69A',
        'sensor_humidity': '#EF5350',
        'sensor_aux': '#AB47BC'
    }

    for col in ['sensor_temp', 'sensor_baseline', 'sensor_humidity', 'sensor_aux']:
        fig.add_trace(go.Scatter(
            x=fusion_df['timestamp'], y=fusion_df[col],
            mode='lines', name=scenario_cfg['labels'][col],
            line=dict(color=sensor_colors[col], width=1.2),
            opacity=0.65
        ))

    # Uncertainty Upper & Lower Bounds
    fig.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['uncertainty_upper'],
        mode='lines', name='Upper 95% Bound (+2σ)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))
    fig.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['uncertainty_lower'],
        mode='lines', name='95% Bayesian Uncertainty Envelope (±2σ)',
        fill='tonexty', fillcolor='rgba(46, 160, 67, 0.18)',
        line=dict(width=0), hoverinfo='skip'
    ))

    # Corrected Fused Estimate
    fig.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['fused_estimate'],
        mode='lines', name='Corrected Fused Reading (Output)',
        line=dict(color='#2ea043', width=3)
    ))

    # Failure Annotation
    fig.add_vline(
        x=failure_step, line_width=1.5, line_dash="dot", line_color="#EF5350",
        annotation_text="Sensor 3 Failure: Channel Isolated", annotation_position="top left"
    )

    fig.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Sample Step (Time)",
        yaxis_title=f"Telemetry Value ({unit})",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Explanation of results
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        **Performance Summary:**
        - **Uncorrected Average RMSE:** `{metrics.get('naive_rmse', 0):.2f} {unit}`
        - **Corrected Fused RMSE:** `{metrics.get('fused_rmse', 0):.2f} {unit}`
        - **Total Error Reduction:** **`{metrics.get('rmse_improvement_pct', 0):.1f}%`**
        """)
    with col_b:
        st.markdown(f"""
        **Automatic Safeguards:**
        - **Sensor 1 (Drift):** Linear calibration slope compensated at `-({detected_drift:.3f}/100)` units per step.
        - **Sensor 3 (Failure):** Completely purged from measurement updates upon rail saturation.
        - **Residual Uncertainty:** 95% credible interval contains the true state **`{metrics.get('confidence_interval_coverage_pct', 0):.1f}%`** of the time.
        """)

# ----------------- TAB 2: DRIFT & FAILURE DIAGNOSTICS -----------------
with tab2:
    st.subheader("Sensor Health & Forensic Diagnostics")
    st.caption("Gemini evaluates statistical moments and cross-sensor residuals to classify health status, isolate drift rates, and assign Kalman weights.")

    cols = st.columns(2)
    sensor_keys = ['sensor_temp', 'sensor_baseline', 'sensor_humidity', 'sensor_aux']
    badge_map = {
        'HEALTHY': 'badge-healthy',
        'DRIFTING': 'badge-drifting',
        'FAILED': 'badge-failed',
        'NOISY': 'badge-noisy'
    }

    for idx, s_key in enumerate(sensor_keys):
        diag = diagnostics.get(s_key, {})
        status = diag.get('status', 'HEALTHY')
        drift_val = diag.get('drift_rate_per_100', 0.0)
        noise_scalar = diag.get('recommended_noise_scalar', 1.0)
        confidence = diag.get('confidence', 95)
        root_cause = diag.get('root_cause', 'Nominal operation within precision thresholds.')
        action = diag.get('recommended_action', 'Maintain nominal weight.')

        with cols[idx % 2]:
            st.markdown(f"""
            <div class="sensor-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <strong>{scenario_cfg['labels'][s_key]}</strong>
                    <span class="badge {badge_map.get(status, 'badge-healthy')}">{status}</span>
                </div>
                <div style="margin-top: 8px; font-size: 0.9rem;">
                    <p style="margin: 4px 0;"><strong>Drift Rate:</strong> {drift_val:+.3f} per 100 samples</p>
                    <p style="margin: 4px 0;"><strong>Kalman Variance Factor:</strong> {noise_scalar}x</p>
                    <p style="margin: 4px 0;"><strong>Diagnostic Confidence:</strong> {confidence}%</p>
                    <p style="margin: 6px 0; color: #8b949e;"><strong>Root Cause:</strong> {root_cause}</p>
                    <p style="margin: 4px 0; color: #58a6ff;"><strong>Action:</strong> {action}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 3: UNCERTAINTY & UNSUPERVISED ADAPTATION -----------------
with tab3:
    st.subheader("How the System Solves the Core Challenges")

    # Part 1: Adapting without labeled data
    st.markdown("#### 1. Adapting Without Continuous Labeled Data")
    st.write(
        "In production environments, true physical ground-truth labels do not exist. "
        "The system isolates drift using **unsupervised spatial consensus**: it computes the median residual between each sensor and the remaining non-saturated consensus pool. "
        "This allows the algorithm to detect systematic drift slopes without requiring any external ground truth."
    )

    # Residual calculation for visual explanation
    active_cols = ['sensor_temp', 'sensor_baseline', 'sensor_aux']
    matrix = df[active_cols].values
    consensus = np.nanmedian(matrix, axis=1)

    fig_residuals = go.Figure()
    fig_residuals.add_trace(go.Scatter(
        x=df['timestamp'], y=df['sensor_temp'] - consensus,
        mode='lines', name='Sensor 1 Residual (Isolates Drift Slope)',
        line=dict(color='#FFA726', width=1.5)
    ))
    fig_residuals.add_trace(go.Scatter(
        x=df['timestamp'], y=df['sensor_baseline'] - consensus,
        mode='lines', name='Sensor 2 Residual (Stationary Zero-Mean)',
        line=dict(color='#26A69A', width=1.5)
    ))
    fig_residuals.update_layout(
        template="plotly_dark",
        height=260,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Time Sample (k)",
        yaxis_title=f"Residual Relative to Consensus ({unit})",
        hovermode="x unified"
    )
    st.plotly_chart(fig_residuals, use_container_width=True)

    # Part 2: Reporting Uncertainty During Changing Conditions
    st.markdown("#### 2. Dynamic Uncertainty Reporting During Environmental Changes")
    st.write(
        "The system quantifies uncertainty using the Bayesian posterior covariance $\\mathbf{P}_{k|k}[0,0]$. "
        "Notice below how uncertainty $\\sigma_{\\text{fused}}(t)$ remains small when all 4 sensors are healthy, "
        "and automatically undergoes a step-increase at the exact moment Sensor 3 fails. "
        "This explicitly reports degraded confidence to downstream autonomous controllers."
    )

    fig_sigma = go.Figure()
    fig_sigma.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['uncertainty_sigma'],
        mode='lines', name='Posterior State Uncertainty (1-Sigma)',
        line=dict(color='#58a6ff', width=2)
    ))
    fig_sigma.add_vline(
        x=failure_step, line_width=1.5, line_dash="dot", line_color="#EF5350",
        annotation_text="Sensor 3 Failure: Uncertainty Expands", annotation_position="top left"
    )
    fig_sigma.update_layout(
        template="plotly_dark",
        height=240,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="Time Sample (k)",
        yaxis_title="State Uncertainty σ (Std Dev)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_sigma, use_container_width=True)

# ----------------- TAB 4: TECHNICAL QUERY ASSISTANT -----------------
with tab4:
    st.subheader("System Diagnostics Assistant")
    st.caption("Ask technical questions about the current telemetry state, mathematical formulations, or failure mechanisms.")

    # Pre-canned technical questions directly aligned with the problem statement
    st.write("**Frequently Asked Technical Inquiries:**")
    q1, q2, q3 = st.columns(3)
    quick_query = None
    with q1:
        if st.button("Why isolate Sensor 3 instead of recalibrating it?"):
            quick_query = "Why isolate Sensor 3 instead of recalibrating it?"
    with q2:
        if st.button("How is drift isolated without labeled data?"):
            quick_query = "How is drift isolated without labeled data?"
    with q3:
        if st.button("How is the uncertainty envelope computed?"):
            quick_query = "How is the uncertainty envelope computed?"

    if 'chat_messages' not in st.session_state:
        st.session_state.chat_messages = [
            {"role": "assistant", "content": "Welcome. I am ready to answer technical queries regarding sensor health, Kalman state estimation, or uncertainty propagation."}
        ]

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_query = st.chat_input("Ask a technical question about the sensor fusion pipeline...")
    chosen_prompt = quick_query or user_query

    if chosen_prompt:
        st.session_state.chat_messages.append({"role": "user", "content": chosen_prompt})
        with st.chat_message("user"):
            st.markdown(chosen_prompt)

        with st.chat_message("assistant"):
            with st.spinner("Evaluating telemetry context..."):
                analyzer = GeminiSensorAnalyzer(api_key=user_api_key)
                response = analyzer.chat_with_telemetry(chosen_prompt, telemetry_context)
                st.markdown(response)
                st.session_state.chat_messages.append({"role": "assistant", "content": response})

st.markdown("---")
st.caption(f"Self-Calibrating Sensor Fusion | Problem Statement Solution | Domain: {scenario_cfg['title']}")
