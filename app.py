import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Breast Cancer Predictor",
    page_icon="🧬",
    layout="wide"
)

# Load your trained model (update the path to your actual model file)
@st.cache_resource
def load_model():
    try:
        model = joblib.load(r"C:\Users\Rana\Desktop\cancer prediction\model\model.pkl")  # Change to your model filename
        return model
    except FileNotFoundError:
        st.error("Model file not found. Please make sure 'model.pkl' is in the same directory.")
        return None

model = load_model()

# Feature names (must match exactly with your training data)
feature_names = [
    'radius1', 'texture1', 'perimeter1', 'area1', 'smoothness1',
    'compactness1', 'concavity1', 'concave_points1', 'symmetry1',
    'fractal_dimension1', 'radius2', 'texture2', 'perimeter2', 'area2',
    'smoothness2', 'compactness2', 'concavity2', 'concave_points2',
    'symmetry2', 'fractal_dimension2', 'radius3', 'texture3', 'perimeter3',
    'area3', 'smoothness3', 'compactness3', 'concavity3', 'concave_points3',
    'symmetry3', 'fractal_dimension3'
]

# App title and description
st.title("🧬 Breast Cancer Prediction AI")
st.markdown("""
This AI model predicts whether breast cancer is **benign** (non-cancerous) or **malignant** (cancerous) 
based on 30 medical features extracted from digitized images of fine needle aspirates (FNA) of breast masses.
""")

# Create input sections with appropriate ranges
st.header("📊 Enter Patient Features")

# Create three columns for better organization
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Mean Features")
    radius1 = st.number_input("Radius Mean", min_value=0.0, max_value=30.0, value=10.0, step=0.1, help="Mean of distances from center to points on the perimeter")
    texture1 = st.number_input("Texture Mean", min_value=0.0, max_value=40.0, value=15.0, step=0.1, help="Standard deviation of gray-scale values")
    perimeter1 = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=50.0, step=0.1, help="Mean size of the core tumor")
    area1 = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=500.0, step=1.0, help="Mean area of the core tumor")
    smoothness1 = st.number_input("Smoothness Mean", min_value=0.0, max_value=0.2, value=0.1, step=0.001, format="%.3f", help="Mean of local variation in radius lengths")
    compactness1 = st.number_input("Compactness Mean", min_value=0.0, max_value=0.5, value=0.1, step=0.001, format="%.3f", help="Mean of perimeter² / area - 1.0")
    concavity1 = st.number_input("Concavity Mean", min_value=0.0, max_value=0.5, value=0.05, step=0.001, format="%.3f", help="Mean severity of concave portions of the contour")
    concave_points1 = st.number_input("Concave Points Mean", min_value=0.0, max_value=0.2, value=0.02, step=0.001, format="%.3f", help="Mean number of concave portions of the contour")
    symmetry1 = st.number_input("Symmetry Mean", min_value=0.0, max_value=0.5, value=0.1, step=0.001, format="%.3f", help="Mean symmetry of the tumor")
    fractal_dimension1 = st.number_input("Fractal Dimension Mean", min_value=0.0, max_value=0.1, value=0.05, step=0.001, format="%.3f", help="Mean 'coastline approximation' - 1")

with col2:
    st.subheader("Standard Error Features")
    radius2 = st.number_input("Radius SE", min_value=0.0, max_value=5.0, value=0.5, step=0.01, help="Standard error of distances from center to points on the perimeter")
    texture2 = st.number_input("Texture SE", min_value=0.0, max_value=5.0, value=0.5, step=0.01, help="Standard error of gray-scale values")
    perimeter2 = st.number_input("Perimeter SE", min_value=0.0, max_value=10.0, value=1.0, step=0.1, help="Standard error of tumor size")
    area2 = st.number_input("Area SE", min_value=0.0, max_value=500.0, value=50.0, step=1.0, help="Standard error of tumor area")
    smoothness2 = st.number_input("Smoothness SE", min_value=0.0, max_value=0.1, value=0.01, step=0.001, format="%.3f", help="Standard error of local variation in radius lengths")
    compactness2 = st.number_input("Compactness SE", min_value=0.0, max_value=0.1, value=0.02, step=0.001, format="%.3f", help="Standard error of compactness")
    concavity2 = st.number_input("Concavity SE", min_value=0.0, max_value=0.1, value=0.02, step=0.001, format="%.3f", help="Standard error of concavity")
    concave_points2 = st.number_input("Concave Points SE", min_value=0.0, max_value=0.1, value=0.01, step=0.001, format="%.3f", help="Standard error of concave points")
    symmetry2 = st.number_input("Symmetry SE", min_value=0.0, max_value=0.1, value=0.02, step=0.001, format="%.3f", help="Standard error of symmetry")
    fractal_dimension2 = st.number_input("Fractal Dimension SE", min_value=0.0, max_value=0.1, value=0.005, step=0.001, format="%.3f", help="Standard error of fractal dimension")

with col3:
    st.subheader("Worst Features")
    radius3 = st.number_input("Radius Worst", min_value=0.0, max_value=40.0, value=15.0, step=0.1, help="Largest mean of distances from center to points on the perimeter")
    texture3 = st.number_input("Texture Worst", min_value=0.0, max_value=50.0, value=20.0, step=0.1, help="Largest standard deviation of gray-scale values")
    perimeter3 = st.number_input("Perimeter Worst", min_value=0.0, max_value=300.0, value=100.0, step=1.0, help="Largest tumor size")
    area3 = st.number_input("Area Worst", min_value=0.0, max_value=5000.0, value=1000.0, step=10.0, help="Largest tumor area")
    smoothness3 = st.number_input("Smoothness Worst", min_value=0.0, max_value=0.3, value=0.1, step=0.001, format="%.3f", help="Largest local variation in radius lengths")
    compactness3 = st.number_input("Compactness Worst", min_value=0.0, max_value=1.0, value=0.2, step=0.001, format="%.3f", help="Largest compactness value")
    concavity3 = st.number_input("Concavity Worst", min_value=0.0, max_value=1.0, value=0.2, step=0.001, format="%.3f", help="Largest concavity value")
    concave_points3 = st.number_input("Concave Points Worst", min_value=0.0, max_value=0.3, value=0.1, step=0.001, format="%.3f", help="Largest number of concave points")
    symmetry3 = st.number_input("Symmetry Worst", min_value=0.0, max_value=1.0, value=0.2, step=0.001, format="%.3f", help="Largest symmetry value")
    fractal_dimension3 = st.number_input("Fractal Dimension Worst", min_value=0.0, max_value=0.2, value=0.08, step=0.001, format="%.3f", help="Largest fractal dimension value")

# Create a dictionary with all the input values
input_data = {
    'radius1': radius1,
    'texture1': texture1,
    'perimeter1': perimeter1,
    'area1': area1,
    'smoothness1': smoothness1,
    'compactness1': compactness1,
    'concavity1': concavity1,
    'concave_points1': concave_points1,
    'symmetry1': symmetry1,
    'fractal_dimension1': fractal_dimension1,
    'radius2': radius2,
    'texture2': texture2,
    'perimeter2': perimeter2,
    'area2': area2,
    'smoothness2': smoothness2,
    'compactness2': compactness2,
    'concavity2': concavity2,
    'concave_points2': concave_points2,
    'symmetry2': symmetry2,
    'fractal_dimension2': fractal_dimension2,
    'radius3': radius3,
    'texture3': texture3,
    'perimeter3': perimeter3,
    'area3': area3,
    'smoothness3': smoothness3,
    'compactness3': compactness3,
    'concavity3': concavity3,
    'concave_points3': concave_points3,
    'symmetry3': symmetry3,
    'fractal_dimension3': fractal_dimension3
}

# Prediction button
if st.button("🔍 Predict Diagnosis", type="primary"):
    if model is not None:
        try:
            # Convert input to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure column order matches training data
            input_df = input_df[feature_names]
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            prediction_proba = model.predict_proba(input_df)[0]
            
            # Display results
            st.header("📋 Prediction Results")
            
            if prediction == 0:
                st.success("🎉 **Prediction: BENIGN** (Non-cancerous)")
                st.info("This suggests the tumor is likely not cancerous. However, please consult with a healthcare professional for definitive diagnosis.")
            else:
                st.error("⚠️ **Prediction: MALIGNANT** (Cancerous)")
                st.warning("This suggests the tumor may be cancerous. Please consult immediately with a healthcare professional for further evaluation.")
            
            # Show confidence scores
            st.subheader("Confidence Scores:")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Benign Confidence", f"{prediction_proba[0]*100:.2f}%")
            with col2:
                st.metric("Malignant Confidence", f"{prediction_proba[1]*100:.2f}%")
            
            # Show progress bars
            st.progress(float(prediction_proba[1]), text="Malignant Probability")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    else:
        st.error("Model not loaded. Please check if the model file exists.")

# Add some information about the features
with st.expander("ℹ️ About the Features"):
    st.markdown("""
    These features are computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. 
    They describe characteristics of the cell nuclei present in the image.
    
    - **Mean Features**: Average of each nuclear characteristic
    - **Standard Error (SE) Features**: Standard error of each nuclear characteristic  
    - **Worst Features**: Largest (mean of the three largest values) of each nuclear characteristic
    
    *Source: UCI Machine Learning Repository - Breast Cancer Wisconsin (Diagnostic) Data Set*
    """)

# Footer
st.markdown("---")
st.caption("⚠️ **Disclaimer**: This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis.")