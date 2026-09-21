"""
Streamlit Command Center: Interactive Dashboard for AI Sensor Fusion & Drift Detection
Powered by Gemini 2.0 Flash, Adaptive Kalman Filtering, and Bayesian Uncertainty Quantification.
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

# Ensure local src directory is on sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sensor_simulator import generate_sensor_stream
from gemini_analyzer import GeminiSensorAnalyzer
from fusion_engine import AdaptiveKalmanFusion

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Sensor Fusion | Gemini Diagnostics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .metric-card {
        background-color: #161b22;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #30363d;
        margin-bottom: 12px;
    }
    .status-badge {
        font-weight: 700;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
    }
    .status-healthy { background-color: #238636; color: white; }
    .status-drifting { background-color: #d29922; color: black; }
    .status-failed { background-color: #da3633; color: white; }
    .status-noisy { background-color: #8957e5; color: white; }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.title("⚙️ Telemetry & Model Controls")

# API Key Manager
st.sidebar.subheader("🔑 Google AI Studio Key")
user_api_key = st.sidebar.text_input(
    "Gemini API Key (optional):",
    type="password",
    value=os.getenv("GEMINI_API_KEY", ""),
    help="Free key from aistudio.google.com/apikey. If left blank, runs in Cognitive Mock Mode."
)

if user_api_key and user_api_key.strip():
    os.environ["GEMINI_API_KEY"] = user_api_key.strip()
    st.sidebar.success("🟢 Live Gemini API Configured")
else:
    st.sidebar.info("💡 Running in Cognitive Simulation Mode (No Key Needed)")

st.sidebar.markdown("---")
st.sidebar.subheader("🎛️ Telemetry Fault Injection")

n_samples = st.sidebar.slider("Telemetry Samples", min_value=300, max_value=1200, value=800, step=50)
drift_rate = st.sidebar.slider("Sensor 1 Drift Rate (Slope)", min_value=0.01, max_value=0.08, value=0.035, step=0.005)
failure_step = st.sidebar.slider("Sensor 3 Hardware Lockup Step", min_value=200, max_value=int(n_samples * 0.9), value=int(n_samples * 0.7), step=50)

if st.sidebar.button("🔄 Regenerate Telemetry Stream"):
    st.session_state.telemetry_data = generate_sensor_stream(
        n_samples=n_samples,
        drift_rate=drift_rate,
        failure_step=failure_step
    )
    if 'diagnostics' in st.session_state:
        del st.session_state['diagnostics']
    st.sidebar.success("Telemetry regenerated!")

# Initialize session state telemetry if not present
if 'telemetry_data' not in st.session_state:
    st.session_state.telemetry_data = generate_sensor_stream(
        n_samples=n_samples,
        drift_rate=drift_rate,
        failure_step=failure_step
    )

df = st.session_state.telemetry_data

# ----------------- HEADER & HERO -----------------
st.title("📡 AI-Powered Sensor Fusion & Drift Diagnostics")
st.markdown("""
**Cognitive Telemetry Monitoring with Gemini 2.0 Flash + Adaptive Kalman State Estimation + Bayesian Uncertainty Bounds**
""")

# Top-level tabs
tab_raw, tab_gemini, tab_fusion, tab_arch = st.tabs([
    "📊 Raw Telemetry Stream",
    "🤖 Gemini AI Diagnostics",
    "⚖️ Adaptive Kalman Fusion & Uncertainty",
    "📐 Architecture & Math"
])

# ----------------- TAB 1: RAW TELEMETRY -----------------
with tab_raw:
    st.subheader("Raw Multi-Sensor Telemetry vs Underlying Ground Truth")
    st.caption("Inspect raw sensor behaviors: Sensor 1 drifts continuously; Sensor 3 suffers catastrophic saturation lockup; Sensor 4 has intermittent burst noise.")

    fig_raw = go.Figure()

    # Ground Truth
    fig_raw.add_trace(go.Scatter(
        x=df['timestamp'], y=df['ground_truth'],
        mode='lines', name='Ground Truth (Process State)',
        line=dict(color='#FFFFFF', width=2.5, dash='dash')
    ))

    # Sensors
    colors = {
        'sensor_temp': '#FFA726',
        'sensor_baseline': '#26A69A',
        'sensor_humidity': '#EF5350',
        'sensor_aux': '#AB47BC'
    }
    labels = {
        'sensor_temp': 'Sensor 1: Temp (Continuous Thermal Drift)',
        'sensor_baseline': 'Sensor 2: Reference (Healthy Baseline)',
        'sensor_humidity': 'Sensor 3: Humidity (Rail Lockup at t={})'.format(failure_step),
        'sensor_aux': 'Sensor 4: Auxiliary (Burst Turbulence)'
    }

    for col in ['sensor_temp', 'sensor_baseline', 'sensor_humidity', 'sensor_aux']:
        fig_raw.add_trace(go.Scatter(
            x=df['timestamp'], y=df[col],
            mode='lines', name=labels[col],
            line=dict(color=colors[col], width=1.5)
        ))

    # Add vertical line for failure
    fig_raw.add_vline(
        x=failure_step, line_width=1.5, line_dash="dot", line_color="#EF5350",
        annotation_text="Sensor 3 Lockup", annotation_position="top left"
    )

    fig_raw.update_layout(
        template="plotly_dark",
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time Sample (k)",
        yaxis_title="Telemetry Measurement",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_raw, use_container_width=True)

    # Telemetry Summary Table
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    with col_t1:
        st.metric("Sensor 1 (Temp)", f"{df['sensor_temp'].iloc[-1]:.2f}", delta=f"+{df['sensor_temp'].iloc[-1] - df['ground_truth'].iloc[-1]:.2f} Drift")
    with col_t2:
        st.metric("Sensor 2 (Baseline)", f"{df['sensor_baseline'].iloc[-1]:.2f}", delta=f"{df['sensor_baseline'].iloc[-1] - df['ground_truth'].iloc[-1]:.2f}")
    with col_t3:
        st.metric("Sensor 3 (Humidity)", f"{df['sensor_humidity'].iloc[-1]:.2f}", delta="SATURATED", delta_color="inverse")
    with col_t4:
        st.metric("Sensor 4 (Aux)", f"{df['sensor_aux'].iloc[-1]:.2f}", delta=f"{df['sensor_aux'].iloc[-1] - df['ground_truth'].iloc[-1]:.2f}")

# ----------------- TAB 2: GEMINI DIAGNOSTICS -----------------
with tab_gemini:
    st.subheader("Cognitive Diagnostics & Root Cause Analysis")
    st.write("Gemini evaluates statistical trend features, autocorrelation, and cross-channel variance to classify sensor health and output adaptive parameters.")

    run_btn = st.button("🚀 Run Gemini Cognitive Diagnostics", type="primary")

    if run_btn or 'diagnostics' not in st.session_state:
        with st.spinner("Analyzing telemetry stream patterns with Gemini..."):
            analyzer = GeminiSensorAnalyzer(api_key=user_api_key)
            diagnostics = analyzer.analyze_stream(df)
            st.session_state.diagnostics = diagnostics

    diagnostics = st.session_state.diagnostics

    # Display Diagnostic Cards
    cols = st.columns(2)
    sensor_map = {
        'sensor_temp': ('Sensor 1 (Temperature)', '🌡️'),
        'sensor_baseline': ('Sensor 2 (Reference Baseline)', '🎯'),
        'sensor_humidity': ('Sensor 3 (Humidity Transducer)', '💧'),
        'sensor_aux': ('Sensor 4 (Auxiliary Vibration/Turbulence)', '⚡')
    }

    badge_classes = {
        'HEALTHY': 'status-healthy',
        'DRIFTING': 'status-drifting',
        'FAILED': 'status-failed',
        'NOISY': 'status-noisy'
    }

    for idx, (s_key, (display_name, icon)) in enumerate(sensor_map.items()):
        diag = diagnostics.get(s_key, {})
        status = diag.get('status', 'HEALTHY')
        conf = diag.get('confidence', 95)
        root_cause = diag.get('root_cause', 'Nominal operation')
        action = diag.get('recommended_action', 'Maintain nominal trust')
        r_scalar = diag.get('recommended_noise_scalar', 1.0)
        source = diag.get('source', 'Gemini Engine')

        with cols[idx % 2]:
            st.markdown(f"""
            <div class="metric-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4>{icon} {display_name}</h4>
                    <span class="status-badge {badge_classes.get(status, 'status-healthy')}">{status}</span>
                </div>
                <p><strong>Confidence:</strong> {conf}% | <strong>Engine:</strong> {source}</p>
                <p><strong>Diagnostic Root Cause:</strong><br><span style="color:#8b949e">{root_cause}</span></p>
                <p><strong>Adaptive Action:</strong><br><span style="color:#58a6ff">{action}</span></p>
                <div style="display:flex; justify-content:space-between; margin-top:8px;">
                    <small>Drift Rate: {diag.get('drift_rate_per_100', 0.0):.3f} / 100 samples</small>
                    <small>Measurement Variance Scalar (R): <strong>{r_scalar}x</strong></small>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ----------------- TAB 3: ADAPTIVE FUSION & UNCERTAINTY -----------------
with tab_fusion:
    st.subheader("Adaptive Kalman Fusion vs Naive Averaging")
    st.caption("See how Gemini's adaptive noise reweighting completely rejects the failed sensor and corrects drift, while Bayesian uncertainty expands appropriately.")

    # Run Fusion
    fusion_engine = AdaptiveKalmanFusion()
    fusion_df = fusion_engine.run_fusion_pipeline(df, st.session_state.diagnostics)
    metrics = fusion_engine.calculate_performance_metrics(fusion_df)

    # Metric Banner
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric("Naive Fusion RMSE", f"{metrics['naive_rmse']:.2f}")
    with m_col2:
        st.metric("Gemini-Adaptive RMSE", f"{metrics['fused_rmse']:.2f}", delta=f"-{metrics['rmse_improvement_pct']:.1f}% Error", delta_color="inverse")
    with m_col3:
        st.metric("Mean Absolute Error (MAE)", f"{metrics['fused_mae']:.2f}", delta=f"vs Naive {metrics['naive_mae']:.2f}")
    with m_col4:
        st.metric("95% Bayesian Coverage", f"{metrics['confidence_interval_coverage_pct']:.1f}%", help="Percentage of true process states within estimated ±2σ bounds")

    # Main Fusion & Uncertainty Plot
    fig_fused = go.Figure()

    # Naive Average
    fig_fused.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['naive_average'],
        mode='lines', name='Naive Unweighted Average (Disrupted by drift/failure)',
        line=dict(color='#da3633', width=1.5, dash='dot')
    ))

    # Ground Truth
    fig_fused.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['ground_truth'],
        mode='lines', name='True Physical State (Target)',
        line=dict(color='#FFFFFF', width=2.5, dash='dash')
    ))

    # Uncertainty Upper Bound
    fig_fused.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['uncertainty_upper'],
        mode='lines', name='Upper 95% Bound (+2σ)',
        line=dict(width=0), showlegend=False, hoverinfo='skip'
    ))

    # Uncertainty Lower Bound with Fill
    fig_fused.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['uncertainty_lower'],
        mode='lines', name='Bayesian 95% Uncertainty Envelope (±2σ)',
        fill='tonexty', fillcolor='rgba(46, 160, 67, 0.20)',
        line=dict(width=0), hoverinfo='skip'
    ))

    # Fused Kalman State Estimate
    fig_fused.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['fused_estimate'],
        mode='lines', name='Gemini-Adaptive Kalman State Estimate',
        line=dict(color='#2ea043', width=3.0)
    ))

    # Mark Failure Point
    fig_fused.add_vline(
        x=failure_step, line_width=1.5, line_dash="dot", line_color="#da3633",
        annotation_text="Sensor 3 Failure: Uncertainty Expands", annotation_position="bottom right"
    )

    fig_fused.update_layout(
        template="plotly_dark",
        height=480,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Time Sample (k)",
        yaxis_title="Estimated Physical State",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig_fused, use_container_width=True)

    # Uncertainty Dynamics Over Time
    st.subheader("📈 Dynamic Bayesian Uncertainty Metric (σ_fused)")
    st.caption("Notice the step increase in posterior state standard deviation at t=700 when the failed sensor is safely excluded from the measurement update.")

    fig_sigma = go.Figure()
    fig_sigma.add_trace(go.Scatter(
        x=fusion_df['timestamp'], y=fusion_df['uncertainty_sigma'],
        mode='lines', name='Posterior State Uncertainty (1-Sigma)',
        line=dict(color='#58a6ff', width=2.0)
    ))
    fig_sigma.add_vline(
        x=failure_step, line_width=1.5, line_dash="dot", line_color="#da3633"
    )
    fig_sigma.update_layout(
        template="plotly_dark",
        height=220,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="Time Sample (k)",
        yaxis_title="Uncertainty σ (Std Dev)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_sigma, use_container_width=True)

# ----------------- TAB 4: ARCHITECTURE & MATH -----------------
with tab_arch:
    st.markdown("""
    ### 📐 System Formulation & Mathematical Framework

    #### 1. Cognitive AI Telemetry Reasoner (Gemini 2.0 Flash)
    Traditional signal processing algorithms struggle to distinguish between **environmental non-stationarity** and **hardware transducer degradation**. Gemini acts as a cognitive supervisory layer:
    - Analyzes statistical moments, polynomial trend slopes, and cross-channel covariance.
    - Diagnoses physical failure modes (thermal aging, sensor rail saturation, ADC latchup).
    - Outputs structured telemetry JSON specifying the dynamic covariance reweighting scalar $\\mathbf{R}_i$.

    #### 2. Discrete State-Space Model
    The true process dynamics are modeled as a 2-state discrete kinematic system:
    $$\\mathbf{x}_k = \\begin{bmatrix} x_k \\\\ \\dot{x}_k \\end{bmatrix}, \\quad \\mathbf{x}_{k+1} = \\mathbf{F}\\mathbf{x}_k + \\mathbf{w}_k$$
    Where $\\mathbf{F} = \\begin{bmatrix} 1 & \\Delta t \\\\ 0 & 1 \\end{bmatrix}$ and $\\mathbf{w}_k \\sim \\mathcal{N}(0, \\mathbf{Q})$ represents continuous white noise process acceleration.

    #### 3. Dynamic Measurement Covariance Matrix $\\mathbf{R}_k$
    Observations from $m$ sensors $\\mathbf{z}_k = [z_1, \\dots, z_m]^T$ map to the state via $\\mathbf{H} = [1, 0]^T$:
    $$\\mathbf{R}_k = \\text{diag}\\left(w_1 \\sigma_1^2, w_2 \\sigma_2^2, \\dots, w_m \\sigma_m^2\\right)$$
    - Healthy Sensor: $w_i = 1.0$
    - Drifting Sensor: Drift bias subtracted $\\hat{z}_i = z_i - (s_i \\cdot k)$, with $w_i = 2.5$
    - Failed Sensor: $w_i \\to \\infty$ (Zero Kalman gain, completely isolated)

    #### 4. Bayesian Uncertainty Quantification
    The posterior estimation error covariance $\\mathbf{P}_{k|k}$ tracks the residual uncertainty of the fused state:
    $$\\sigma_{\\text{fused}, k} = \\sqrt{\\mathbf{P}_{k|k}[0, 0]}$$
    The 95% Bayesian credible interval reported in real-time is:
    $$\\mathcal{CI}_{95\\%} = \\left[ \\hat{x}_{k|k} - 2\\sigma_{\\text{fused}, k}, \\; \\hat{x}_{k|k} + 2\\sigma_{\\text{fused}, k} \\right]$$
    """)

st.markdown("---")
st.caption("🚀 Built for Google Gemini Hackathon | Team: Kush & Co. | Powered by Gemini 2.0 Flash & Adaptive Kalman State Estimation")
