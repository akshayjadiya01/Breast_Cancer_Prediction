# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.datasets import load_breast_cancer
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Page config & styles
# ---------------------------
st.set_page_config(
    page_title="🧬 Breast Cancer Prediction Dashboard",
    page_icon="🩷",
    layout="wide"
)

# Inline CSS for gradient header, cards, animation, soft UI
st.markdown(
    """
    <style>
    /* Page background and fonts */
    :root{
        --soft-pink: #ffeef6;
        --pastel-blue: #e7f5ff;
        --card-bg: linear-gradient(135deg, rgba(255,230,240,0.85), rgba(235,245,255,0.85));
        --accent: #ff6fa3;
        --muted: #6b7280;
    }
    body {
        background: linear-gradient(180deg, #ffffff 0%, #fffafc 100%);
        color: #0f172a;
        font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Centered header */
    .header {
        text-align: center;
        padding-top: 6px;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    .title-gradient {
        font-size: 32px;
        font-weight: 800;
        background: -webkit-linear-gradient(90deg, #ff8ab6, #7cc6ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline-block;
        letter-spacing: -0.5px;
        margin-bottom: 4px;
    }
    .subtitle {
        font-size: 14px;
        color: var(--muted);
        margin-top: 0px;
    }

    /* Gradient card for prediction result */
    .result-card {
        border-radius: 14px;
        padding: 16px;
        box-shadow: 0 6px 18px rgba(15,23,42,0.06);
        background: linear-gradient(135deg, rgba(255,240,245,0.95), rgba(235,245,255,0.95));
        border: 1px solid rgba(255,111,172,0.12);
    }

    /* Animated probability text */
    .pulse {
        display:inline-block;
        animation: pulse 1.6s infinite;
        font-weight:700;
    }
    @keyframes pulse {
        0% { transform: translateY(0px) scale(1); opacity: 1; }
        50% { transform: translateY(-4px) scale(1.02); opacity: 0.95; }
        100% { transform: translateY(0px) scale(1); opacity: 1; }
    }

    /* Soft input labels on sidebar */
    .stSidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(255,250,252,0.6), rgba(245,252,255,0.6));
        border-radius: 10px;
        padding: 12px;
    }

    /* Footer */
    .footer {
        margin-top: 20px;
        padding-top: 8px;
        padding-bottom: 24px;
        color: #6b7280;
        font-size: 13px;
    }
    hr.soft {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, rgba(255,111,172,0.15), rgba(124,198,255,0.15));
        margin-top: 8px;
        margin-bottom: 8px;
    }

    /* Make charts and containers responsive */
    @media (max-width: 600px) {
        .title-gradient { font-size: 24px; }
        .subtitle { font-size: 13px; }
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------
# Header
# ---------------------------
st.markdown(
    """
    <div class="header">
        <div class="title-gradient">🧬 Breast Cancer Prediction Dashboard</div>
        <div class="subtitle">Empowering Early Detection through AI</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# Utility: load model and scaler
# ---------------------------
@st.cache_resource
def load_artifacts(model_path="breast_cancer_model.pkl", scaler_path="scaler.pkl"):
    model = None
    scaler = None
    errs = []
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        errs.append(f"Model file not found: {model_path}")

    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        errs.append(f"Scaler file not found: {scaler_path}")

    return model, scaler, errs

model, scaler, load_errors = load_artifacts()

# ---------------------------
# Feature names & defaults (from sklearn dataset)
# ---------------------------
dataset = load_breast_cancer()
FEATURE_NAMES = list(dataset.feature_names)
DEFAULT_MEANS = dict(zip(FEATURE_NAMES, np.round(dataset.data.mean(axis=0), 3)))

# Order of features to show in sidebar (grouped by mean importance /brevity)
SIDEBAR_FEATURES = FEATURE_NAMES  # show all 30 features as sliders

# Icon emojis (some mapping, repeated emoji if none exact)
EMOJI_MAP = {
    "mean radius": "📏", "mean texture": "🎚️", "mean perimeter": "🧭", "mean area": "🧩", "mean smoothness": "✨",
    "mean compactness": "🔵", "mean concavity": "🔻", "mean concave points": "🔺", "mean symmetry": "⚖️", "mean fractal dimension": "🌿",
    "radius error": "📐", "texture error": "🔬", "perimeter error": "📏", "area error": "🧩", "smoothness error": "✨",
    "compactness error": "🔵", "concavity error": "🔻", "concave points error": "🔺", "symmetry error": "⚖️", "fractal dimension error": "🌿",
    "worst radius": "🏁", "worst texture": "🎛️", "worst perimeter": "🧭", "worst area": "🧩", "worst smoothness": "✨",
    "worst compactness": "🔵", "worst concavity": "🔻", "worst concave points": "🔺", "worst symmetry": "⚖️", "worst fractal dimension": "🌿"
}

# ---------------------------
# Sidebar: stylized inputs
# ---------------------------
with st.sidebar:
    st.markdown("### 🧾 Patient / Tumor Inputs")
    st.markdown("Adjust values (sliders) — defaults are dataset means.")
    st.write("")  # spacing

    # Create sliders for each feature (use compact layout columns for mobile friendliness)
    input_vals = {}
    for feat in SIDEBAR_FEATURES:
        emoji = EMOJI_MAP.get(feat, "🔹")
        default = float(DEFAULT_MEANS.get(feat, 0.0))
        # compute a reasonable slider range from dataset statistics (min, max)
        col_index = FEATURE_NAMES.index(feat)
        arr = dataset.data[:, col_index]
        mn, mx = float(np.min(arr)), float(np.max(arr))
        # widen the slider bounds slightly for flexibility
        span = max(1e-6, mx - mn)
        slider_min = float(max(0, mn - 0.2 * span))
        slider_max = float(mx + 0.2 * span)
        step = float(span / 200) if span > 0 else 0.1

        # nicer label: emoji + short name
        label = f"{emoji} {feat}"
        input_vals[feat] = st.slider(label, min_value=round(slider_min, 5), max_value=round(slider_max, 5),
                                    value=round(default, 5), step=round(step, 6))

    st.markdown("---")
    st.markdown("Made by Akshay Jadiya")
    st.markdown("[GitHub](https://github.com/akshayjadiya01)")

# ---------------------------
# Top tabs
# ---------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Prediction", "📊 Model Insights", "ℹ️ About"])

# ---------------------------
# Prediction Tab
# ---------------------------
# ==========================================================
# 🧬 ADVANCED PREDICTION TAB
# ==========================================================
import numbers

def infer_label_mapping_from_model(model):
    """
    Detect and return label mapping {class_value: 'Malignant'/'Benign'}
    Works for models trained with numeric or string labels ('M'/'B' or 'malignant'/'benign').
    """
    classes = list(model.classes_)
    if len(classes) < 2:
        return {classes[0]: "Unknown"}

    # Numeric (0/1)
    if all(isinstance(c, numbers.Number) for c in classes):
        mean_coef = float(np.mean(model.coef_)) if hasattr(model, "coef_") else 0
        # Default sklearn dataset: 0=malignant, 1=benign
        if mean_coef >= 0:
            return {classes[0]: "Malignant", classes[1]: "Benign"}
        else:
            return {classes[0]: "Benign", classes[1]: "Malignant"}

    # Text labels (M/B or malignant/benign)
    lower = [str(c).strip().lower() for c in classes]
    mapping = {}
    for i, s in enumerate(lower):
        if "malig" in s or s == "m":
            mapping[classes[i]] = "Malignant"
        elif "benig" in s or s == "b":
            mapping[classes[i]] = "Benign"
    if len(mapping) < 2:
        # Fill missing
        other = [c for c in classes if c not in mapping][0]
        mapping[other] = "Benign" if list(mapping.values())[0] == "Malignant" else "Malignant"
    return mapping
# ==========================================================
# 🏠 PREDICTION TAB
# ==========================================================
# ==========================================================
# 🏠 PREDICTION TAB
# ==========================================================
with tab1:
    st.markdown(
        """
        <div style='text-align:center;margin-bottom:25px;'>
            <h2 style='
                background: -webkit-linear-gradient(90deg,#ff6fa3,#7cc6ff);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight:900;
                margin-bottom:5px;'>
                🩺 Smart Prediction
            </h2>
            <p style='color:#6b7280;'>AI-powered analysis to determine whether the tumor is <b>malignant</b> or <b>benign</b>.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1, 1])

    # ======================================================
    # LEFT SIDE → INPUT SUMMARY + PREDICT
    # ======================================================
    with col1:
        with st.expander("🧾 View All Input Values", expanded=False):
            st.dataframe(pd.DataFrame([input_vals]).T.rename(columns={0: "Value"}), height=380)

        predict_clicked = st.button("🚀 Predict Now", use_container_width=True, type="primary")

    # ======================================================
    # RIGHT SIDE → RESULT CARD
    # ======================================================
    with col2:
        result_placeholder = st.empty()

    # ======================================================
    # PERFORM PREDICTION
    # ======================================================
    if load_errors:
        st.error("⚠️ Missing artifacts:\n" + "\n".join(load_errors))
    elif predict_clicked:
        try:
            # Prepare input
            sample = np.array([input_vals[f] for f in FEATURE_NAMES]).reshape(1, -1)
            sample_df = pd.DataFrame(sample, columns=FEATURE_NAMES)
            X_scaled = scaler.transform(sample_df)

            # Predict
            pred_prob = float(model.predict_proba(X_scaled)[0, 1])
            pred_class = model.predict(X_scaled)[0]

            # Infer label mapping
            label_mapping = infer_label_mapping_from_model(model)
            human_label = label_mapping.get(pred_class, "Unknown")

            # Style & colors
            if human_label == "Malignant":
                gradient_bg = "linear-gradient(135deg,#ffe0eb,#ffd7e8)"
                accent_color = "#b91c1c"
                emoji = "⚠️"
            else:
                gradient_bg = "linear-gradient(135deg,#e9fff6,#e3f2ff)"
                accent_color = "#065f46"
                emoji = "✅"

            # Card HTML
            result_html = f"""
            <div style="
                background:{gradient_bg};
                border-radius:20px;
                padding:25px;
                box-shadow:0 6px 20px rgba(0,0,0,0.08);
                border:1px solid rgba(255,255,255,0.4);
                ">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <div style="font-size:15px;color:#475569;">Prediction Result</div>
                        <div style="font-size:30px;font-weight:900;margin-top:8px;color:{accent_color};">
                            {emoji} <span class="pulse">{human_label}</span>
                        </div>
                        <div style="font-size:14px;color:#475569;margin-top:5px;">
                            Confidence: <b>{pred_prob*100:.2f}%</b>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="font-size:13px;color:#6b7280;">Model Probability</div>
                        <div style="font-size:32px;font-weight:900;color:{accent_color};">
                            {pred_prob*100:.0f}%
                        </div>
                    </div>
                </div>
            </div>
            """
            result_placeholder.markdown(result_html, unsafe_allow_html=True)

            # Progress bar
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🎯 Prediction Confidence")
            st.progress(int(pred_prob * 100))
            st.markdown(
                f"<p style='text-align:center;color:#64748b;font-size:13px;margin-top:6px;'>"
                f"The model predicts this tumor is <b>{human_label}</b> with {pred_prob*100:.2f}% confidence."
                f"</p>",
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.markdown(
            """
            <div style="
                border-radius:16px;
                padding:26px;
                background:linear-gradient(135deg,#fafafa,#ffffff);
                border:1px solid rgba(0,0,0,0.05);
                text-align:center;
                color:#6b7280;
                box-shadow:0 4px 10px rgba(0,0,0,0.04);
            ">
                🧠 Awaiting input — adjust sliders on the left and press <b>Predict Now</b> to see results.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------
# Model Insights Tab
# ---------------------------
with tab2:
    st.markdown("### Model Insights: Feature influence & coefficients")

    if load_errors:
        st.warning("Model artifacts not loaded — insights unavailable.")
    else:
        # Try to compute coefficients and feature importance
        try:
            # For logistic regression, coef_ shape (1, n_features) for binary
            coefs = model.coef_.flatten()
            coef_df = pd.DataFrame({
                "feature": FEATURE_NAMES,
                "coefficient": coefs,
                "abs_coef": np.abs(coefs)
            }).sort_values("abs_coef", ascending=False).reset_index(drop=True)

            # Top 10 features by absolute coefficient
            top10 = coef_df.head(10).copy()
            fig = px.bar(top10[::-1], x="abs_coef", y="feature", orientation="h",
                         labels={"abs_coef": "Absolute Coefficient (importance)", "feature": ""},
                         title="Top 10 Influential Features (by absolute coefficient)")
            fig.update_layout(margin=dict(l=0, r=10, t=40, b=10), height=420)

            st.plotly_chart(fig, use_container_width=True)

            # Coefficient direction chart (positive vs negative)
            coef_sorted = coef_df.sort_values("coefficient")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=coef_sorted["coefficient"],
                y=coef_sorted["feature"],
                orientation="h",
                marker=dict(
                    color=coef_sorted["coefficient"],
                    colorscale="RdBu",
                    showscale=False,
                )
            ))
            fig2.update_layout(title="Feature Coefficients (direction & magnitude)", height=600, margin=dict(l=0, r=10, t=40, b=10))
            st.plotly_chart(fig2, use_container_width=True)

            # Coefficients table
            with st.expander("Show coefficients table"):
                st.dataframe(coef_df[["feature", "coefficient"]].assign(coefficient=lambda df: df["coefficient"].round(6)).reset_index(drop=True))

        except Exception as e:
            st.error(f"Failed to compute model insights: {e}")

# ---------------------------
# About Tab
# ---------------------------
with tab3:
    st.markdown("### About this project")
    st.markdown(
        """
        **Project:** Breast Cancer Prediction using Machine Learning  
        **Author:** Akshay Jadiya  
        **GitHub:** [https://github.com/akshayjadiya01](https://github.com/akshayjadiya01)

        This demo loads a pre-trained Logistic Regression model and a scaler to predict whether a tumor is malign or benign based on the classic Wisconsin Breast Cancer dataset (sklearn).  
        It is intended as a portfolio / educational tool to showcase:
        - Model inference (scaling + predict)
        - Model interpretability (coefficients & feature importance)
        - Clean, responsive Streamlit UI
        """
    )
    st.markdown("#### Data source")
    st.markdown("- The Breast Cancer Diagnostic dataset sourced from `Kaggle`, containing detailed measurements of breast cell nuclei used to predict whether a tumor is Malignant or Benign.).")
    st.markdown("#### Notes & disclaimers")
    st.markdown(
        """
        - This app is for demonstration and educational purposes **only**. It is **not** a medical diagnostic tool.
        - Real clinical deployment requires robust validation, regulatory approvals, and clinician oversight.
        """
    )

# ---------------------------
# Footer
# ---------------------------
st.markdown("<hr class='soft'/>", unsafe_allow_html=True)
st.markdown(
    f"""
    <div class="footer">
        <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
            <div><strong>GitHub</strong> — <a href="https://github.com/akshayjadiya01">akshayjadiya01</a></div>
            <div style="text-align:center;">“AI helps us see patterns — humans decide with care.”</div>
            <div style="text-align:right;">Built by Akshay Jadiya</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------
# End of app
# ---------------------------
