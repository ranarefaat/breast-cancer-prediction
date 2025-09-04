import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ------------------------------
# Streamlit Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧬",
    layout="wide"
)

# ------------------------------
# Load Model and Scaler
# ------------------------------
@st.cache_resource
def load_artifacts():
    try:
        with open(r"C:\Users\Rana\Documents\GitHub\breast-cancer-prediction\model\SVC_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(r"C:\Users\Rana\Documents\GitHub\breast-cancer-prediction\model\scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model/scaler: {e}")
        return None, None

model, scaler = load_artifacts()

# ------------------------------
# Feature Names (must match training)
# ------------------------------
feature_names = [
    'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1',
    'compactness1', 'concavity1', 'concave_points1', 'symmetry1',
    'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2',
    'smoothness2', 'compactness2', 'concavity2', 'concave_points2',
    'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3',
    'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3',
    'symmetry3', 'fractal_dimension3'
]

# ------------------------------
# App Title & Description
# ------------------------------
st.title("🧬 Breast Cancer Prediction AI")
st.markdown("""
This AI model predicts whether breast cancer is **benign** (non-cancerous) or **malignant** (cancerous) 
based on 30 medical features extracted from digitized images of fine needle aspirates (FNA) of breast masses.
""")

# ------------------------------
# Input Fields (3 columns)
# ------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Mean Features")
    radius1 = st.number_input("Radius Mean", 0.0, 30.0, 10.0)
    texture1 = st.number_input("Texture Mean", 0.0, 40.0, 15.0)
    perimeter1 = st.number_input("Perimeter Mean", 0.0, 200.0, 50.0)
    area1 = st.number_input("Area Mean", 0.0, 2500.0, 500.0)
    smoothness1 = st.number_input("Smoothness Mean", 0.0, 0.2, 0.1, format="%.3f")
    compactness1 = st.number_input("Compactness Mean", 0.0, 0.5, 0.1, format="%.3f")
    concavity1 = st.number_input("Concavity Mean", 0.0, 0.5, 0.05, format="%.3f")
    concave_points1 = st.number_input("Concave Points Mean", 0.0, 0.2, 0.02, format="%.3f")
    symmetry1 = st.number_input("Symmetry Mean", 0.0, 0.5, 0.1, format="%.3f")
    fractal_dimension1 = st.number_input("Fractal Dimension Mean", 0.0, 0.1, 0.05, format="%.3f")

with col2:
    st.subheader("Standard Error Features")
    radius2 = st.number_input("Radius SE", 0.0, 5.0, 0.5)
    texture2 = st.number_input("Texture SE", 0.0, 5.0, 0.5)
    perimeter2 = st.number_input("Perimeter SE", 0.0, 10.0, 1.0)
    area2 = st.number_input("Area SE", 0.0, 500.0, 50.0)
    smoothness2 = st.number_input("Smoothness SE", 0.0, 0.1, 0.01, format="%.3f")
    compactness2 = st.number_input("Compactness SE", 0.0, 0.1, 0.02, format="%.3f")
    concavity2 = st.number_input("Concavity SE", 0.0, 0.1, 0.02, format="%.3f")
    concave_points2 = st.number_input("Concave Points SE", 0.0, 0.1, 0.01, format="%.3f")
    symmetry2 = st.number_input("Symmetry SE", 0.0, 0.1, 0.02, format="%.3f")
    fractal_dimension2 = st.number_input("Fractal Dimension SE", 0.0, 0.1, 0.005, format="%.3f")

with col3:
    st.subheader("Worst Features")
    radius3 = st.number_input("Radius Worst", 0.0, 40.0, 15.0)
    texture3 = st.number_input("Texture Worst", 0.0, 50.0, 20.0)
    perimeter3 = st.number_input("Perimeter Worst", 0.0, 300.0, 100.0)
    area3 = st.number_input("Area Worst", 0.0, 5000.0, 1000.0)
    smoothness3 = st.number_input("Smoothness Worst", 0.0, 0.3, 0.1, format="%.3f")
    compactness3 = st.number_input("Compactness Worst", 0.0, 1.0, 0.2, format="%.3f")
    concavity3 = st.number_input("Concavity Worst", 0.0, 1.0, 0.2, format="%.3f")
    concave_points3 = st.number_input("Concave Points Worst", 0.0, 0.3, 0.1, format="%.3f")
    symmetry3 = st.number_input("Symmetry Worst", 0.0, 1.0, 0.2, format="%.3f")
    fractal_dimension3 = st.number_input("Fractal Dimension Worst", 0.0, 0.2, 0.08, format="%.3f")

# ------------------------------
# Collect input
# ------------------------------
input_data = {
    'radius1': radius1, 'texture1': texture1, 'perimeter1': perimeter1, 'area1': area1,
    'smoothness1': smoothness1, 'compactness1': compactness1, 'concavity1': concavity1,
    'concave_points1': concave_points1, 'symmetry1': symmetry1, 'fractal_dimension1': fractal_dimension1,
    'radius2': radius2, 'texture2': texture2, 'perimeter2': perimeter2, 'area2': area2,
    'smoothness2': smoothness2, 'compactness2': compactness2, 'concavity2': concavity2,
    'concave_points2': concave_points2, 'symmetry2': symmetry2, 'fractal_dimension2': fractal_dimension2,
    'radius3': radius3, 'texture3': texture3, 'perimeter3': perimeter3, 'area3': area3,
    'smoothness3': smoothness3, 'compactness3': compactness3, 'concavity3': concavity3,
    'concave_points3': concave_points3, 'symmetry3': symmetry3, 'fractal_dimension3': fractal_dimension3
}

# ------------------------------
# Prediction
# ------------------------------
if st.button("🔍 Predict Diagnosis", type="primary"):
    if model is not None and scaler is not None:
        try:
            input_df = pd.DataFrame([input_data])[feature_names]
            input_scaled = scaler.transform(input_df)

            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]

            st.header("📋 Prediction Results")

            if prediction == 1:  # 1 = Benign
                st.success("🎉 **Prediction: BENIGN** (Non-cancerous)")
            else:  # 0 = Malignant
                st.error("⚠️ **Prediction: MALIGNANT** (Cancerous)")

            st.subheader("Confidence Scores:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Malignant Confidence", f"{prediction_proba[0]*100:.2f}%")
            with col2:
                st.metric("Benign Confidence", f"{prediction_proba[1]*100:.2f}%")

            st.progress(float(prediction_proba[0]), text="Malignant Probability")

        except Exception as e:
            st.error(f"Error making prediction: {e}")
    else:
        st.error("Model or scaler not loaded. Please check your pickle files.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This tool is for educational and research purposes only. Not a substitute for professional medical advice.")
