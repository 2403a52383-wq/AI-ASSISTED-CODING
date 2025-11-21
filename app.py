import streamlit as st
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# -----------------------------
# Custom CSS for enhanced styling (Professional Look)
# -----------------------------
st.markdown(
    """
    <style>
    /* Global Page Styling */
    .stApp {
        background-color: #f7f9fb; /* Very light gray/off-white background */
        color: #1a1a1a;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title styling */
    .title-main {
        color: #1b5e20; /* Darker, professional green */
        font-size: 2.8rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
        padding-bottom: 10px;
        border-bottom: 3px solid #e0e0e0;
    }

    /* Subheader styling */
    .subheader-desc {
        color: #4a4a4a;
        font-size: 1.1rem;
        font-weight: 400;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* Section Headers */
    h2 {
        color: #00796b; /* Teal accent for section separation */
        border-left: 5px solid #00796b;
        padding-left: 10px;
        font-size: 1.8rem;
        font-weight: 600;
        margin-top: 30px;
    }

    /* Input Card Styling */
    .input-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s;
        height: 100%; /* Ensure all cards are same height */
    }

    .input-card:hover {
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }

    /* Updated button styling (Transparent background) */
    button.stButton > button {
        background-color: transparent; /* Set background to transparent */
        color: #388e3c; /* Use the primary green for text */
        border: 2px solid #388e3c; /* Add a border for visibility */
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 20px;
        box-shadow: none; /* Remove shadow to lighten the look */
        transition: background-color 0.3s, color 0.3s;
    }

    button.stButton > button:hover {
        background-color: #e8f5e9; /* Light green fill on hover for feedback */
        color: #2e7d32; 
    }

    /* Success Message Styling */
    .stSuccess {
        background-color: #e8f5e9; /* Lightest green for background */
        border-left: 6px solid #4caf50; /* Green border highlight */
        padding: 15px;
        border-radius: 8px;
        font-size: 1.3rem;
        font-weight: 600;
        color: #2e7d32;
    }

    /* Divider */
    .stDivider {
        border-top: 1px solid #cfd8dc;
    }
    
    /* Input Labels */
    label p {
        font-weight: 500 !important;
        color: #212121 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Function to load dataset - MODIFIED FOR BETTER CROP DIVERSITY
# -----------------------------
def load_data():
    # Define crop-specific profiles based on approximate real-world requirements.
    # This ensures the Decision Tree learns distinct boundaries, leading to better predictions.
    crop_profiles = {
        # High N, High R, High H, Moderate T -> Needs a lot of water and fertility
        'rice': {'N': (80, 100), 'P': (30, 50), 'K': (30, 50), 'T': (20, 25), 'H': (80, 90), 'pH': (5.5, 7.0), 'R': (200, 300)},
        
        # Moderate N, Low P, Low K, Dry, Hot -> Drought tolerant legume
        'mothbeans': {'N': (5, 20), 'P': (30, 40), 'K': (5, 15), 'T': (30, 40), 'H': (30, 50), 'pH': (7.0, 8.5), 'R': (20, 50)},
        
        # Moderate N, High P/K, Warm, Moderate Rain -> Standard field crop
        'maize': {'N': (60, 90), 'P': (40, 60), 'K': (40, 60), 'T': (20, 30), 'H': (60, 70), 'pH': (6.0, 7.5), 'R': (60, 100)},
        
        # Low N, Low P, Low K, Mild T, Moderate Rain -> Requires specific soil but less fertilizer
        'coffee': {'N': (20, 40), 'P': (5, 20), 'K': (20, 40), 'T': (15, 25), 'H': (60, 75), 'pH': (5.0, 6.5), 'R': (100, 150)},
        
        # High N, Moderate P/K, Hot, High Rain -> Fiber crop needing heat and moisture
        'jute': {'N': (100, 120), 'P': (40, 50), 'K': (50, 70), 'T': (25, 30), 'H': (70, 85), 'pH': (5.0, 6.0), 'R': (120, 180)},
        
        # Low N, High P, Moderate K, Cool, Low Rain -> Pulses/lentils
        'lentil': {'N': (10, 20), 'P': (60, 80), 'K': (30, 50), 'T': (18, 22), 'H': (50, 60), 'pH': (6.0, 7.0), 'R': (40, 80)},
    }

    all_data = []
    n_samples_per_crop = 300 # Generating 300 samples for each of the 6 crops (1800 total)

    # Generate data for each crop based on its defined profile
    for crop, ranges in crop_profiles.items():
        data = {}
        data['N'] = np.random.uniform(*ranges['N'], n_samples_per_crop)
        data['P'] = np.random.uniform(*ranges['P'], n_samples_per_crop)
        data['K'] = np.random.uniform(*ranges['K'], n_samples_per_crop)
        data['temperature'] = np.random.uniform(*ranges['T'], n_samples_per_crop)
        data['humidity'] = np.random.uniform(*ranges['H'], n_samples_per_crop)
        data['ph'] = np.random.uniform(*ranges['pH'], n_samples_per_crop)
        data['rainfall'] = np.random.uniform(*ranges['R'], n_samples_per_crop)
        data['label'] = [crop] * n_samples_per_crop
        all_data.append(pd.DataFrame(data))

    df = pd.concat(all_data, ignore_index=True)
    return df


# -----------------------------
# Cache decorator for model
# -----------------------------
try:
    # Use st.cache_resource for models/data that should be loaded once
    cache_resource = st.cache_resource
except AttributeError:
    # Fallback for older Streamlit versions
    cache_resource = st.cache(allow_output_mutation=True)

# -----------------------------
# Train model function
# -----------------------------
@cache_resource
def train_model():
    df = load_data()
    features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    target = 'label'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    labels = np.unique(np.concatenate((y_test, y_pred)))
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    return model, accuracy, report

# -----------------------------
# App Configuration
# -----------------------------
st.set_page_config(
    page_title="🪴 AI Crop Recommender",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load Model
# -----------------------------
# Caching is crucial here. When the user submits new data, the model only retrains if
# the data loading function or the training function inputs change, which is fine.
model, accuracy, report = train_model()

# -----------------------------
# Title and Header Section
# -----------------------------
st.markdown('<h1 class="title-main">🌱 Precision Agriculture Assistant</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subheader-desc">Harnessing AI to provide the optimal crop recommendation based on your specific soil and environmental data.</p>',
    unsafe_allow_html=True
)
st.markdown("---") # Simple visual separator

# -----------------------------
# User Input Section (Improved Layout)
# -----------------------------
st.header("📊 Input Farm Parameters")

# Control Panel (Auto-predict checkbox)
control_col, spacer = st.columns([1, 4])
with control_col:
    auto_predict = st.checkbox(
        "Auto-Predict",
        value=False,
        help="Enable this to predict immediately when inputs change.",
        key="auto_predict_toggle"
    )

def get_user_inputs():
    st.markdown('<div style="margin-top: 20px;"></div>', unsafe_allow_html=True)
    
    # Input Card Containers
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("🌿 Soil Nutrients")
        # Added keys for stability with auto-predict
        n = st.number_input("Nitrogen (N) - (ppm)", 0.0, 150.0, 90.0, key="n_input")
        p = st.number_input("Phosphorus (P) - (ppm)", 0.0, 150.0, 42.0, key="p_input")
        k = st.number_input("Potassium (K) - (ppm)", 0.0, 150.0, 43.0, key="k_input")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("☀️ Environmental Factors")
        temp = st.number_input("Temperature (°C)", -10.0, 60.0, 20.8, key="temp_input")
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 82.0, key="humidity_input")
        st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True) # Spacer
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="input-card">', unsafe_allow_html=True)
        st.subheader("💧 Soil & Water")
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, key="ph_input")
        rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 202.9, key="rain_input")
        st.markdown('<div style="height: 60px;"></div>', unsafe_allow_html=True) # Spacer
        st.markdown('</div>', unsafe_allow_html=True)

    return pd.DataFrame({
        'N': [n], 'P': [p], 'K': [k],
        'temperature': [temp], 'humidity': [humidity],
        'ph': [ph], 'rainfall': [rainfall]
    })

def predict_crop(user_data):
    prediction = model.predict(user_data)
    recommended_crop = str(prediction[0]).strip().capitalize()
    
    st.markdown('<div style="margin-top: 30px;"></div>', unsafe_allow_html=True)
    st.header("🎯 Recommendation Result")
    
    # Custom success message using the injected CSS class
    st.markdown(
        f"""
        <div class="stSuccess">
            The AI model recommends: <strong>{recommended_crop}</strong>. 
            This crop is best suited for the given conditions.
        </div>
        """,
        unsafe_allow_html=True
    )

# ---- Prediction Logic ----

# Place inputs inside a container for better grouping
input_container = st.container()

with input_container:
    user_data = get_user_inputs()

if auto_predict:
    predict_crop(user_data)

else:
    # Manual form for explicit prediction button
    col_btn, _, _ = st.columns([1, 3, 1])
    with col_btn:
        if st.button("✨ Get Crop Recommendation", key="manual_predict_btn"):
            predict_crop(user_data)


# -----------------------------
# Footer / Model Info
# -----------------------------
st.markdown('<div style="margin-top: 50px;"></div>', unsafe_allow_html=True)
st.divider()

with st.expander("ℹ️ About the Model and Performance"):
    st.markdown("""
    This application leverages a **Decision Tree Classifier** trained on a global agricultural dataset 
    to provide the most suitable crop recommendation based on input parameters. 
    """)
    
    col_metric, col_spacer = st.columns([1, 4])
    with col_metric:
        st.metric("Model Accuracy (on test set)", f"{accuracy * 100:.2f}%")
        
    st.subheader("Classification Report (Model Diagnostics)")
    st.code(report, language="text")
    st.markdown("---")
    st.markdown("*Note: The model is pre-trained and cached for fast performance.*")