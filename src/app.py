import os
import pickle
import streamlit as st
import pandas as pd

# ------------------------------
# Custom CSS Styling
# ------------------------------
st.markdown("""
    <style>
        body {
            font-family: Consolas, monospace;
        }
        .header-container {
            background-color: #f0f2f6;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
        }
        .header-container h1 {
            color: #1A73E8;
            margin-bottom: 10px;
        }
        .header-container p {
            color: #555;
            font-size: 18px;
        }
        .subheader {
            text-align: center;
            margin-bottom: 20px;
            color: #D93025;
        }
        .center-button {
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 30px;
        }
        .result-box {
            background-color: #f0f2f6;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-top: 30px;
            border: 2px solid #1A73E8;
        }
        .result-title {
            color: #000;
            font-size: 26px;
            font-weight: bold;
            margin: 0;
        }
        .result-text {
            color: #000;
            font-size: 16px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ------------------------------
# Header Section without GIF inside header-container
# ------------------------------
st.markdown("""
    <div class="header-container">
        <h1>📱 Mobile Price Prediction</h1>
        <p>
            Provide your mobile's specifications below for an accurate prediction of its price category. 
            Our advanced Random Forest model analyzes features like battery, camera, memory, and connectivity.
        </p>
    </div>
""", unsafe_allow_html=True)

# Place the GIF outside of the header-container div
st.markdown("""
    <div style="text-align: center; margin-bottom: 30px;">
        <img src="https://i.gifer.com/5hoN.gif" alt="Mobile Prediction GIF" height="250px" width="400px">
    </div>
""", unsafe_allow_html=True)

st.markdown('<h2 class="subheader">Enter Mobile Specifications</h2>', unsafe_allow_html=True)

# ------------------------------
# Load the Model (Cached)
# ------------------------------
@st.cache_resource
def load_model():
    # Navigate one level up from 'src' folder to 'model' folder
    model_path = os.path.join(os.path.dirname(__file__), '..', 'model', 'mobile_price_model.pkl')
    if os.path.exists(model_path):
        with open(model_path, 'rb') as file:
            return pickle.load(file)
    else:
        st.error("Model file not found in 'model/' directory. Please check the path.")
        return None

model = load_model()

# ------------------------------
# Input Widgets for Mobile Features (Single Column Layout)
# ------------------------------
battery_power = st.number_input('Battery Power (mAh)', min_value=500, max_value=2000, value=1000, step=50)
clock_speed   = st.number_input('Clock Speed (GHz)', min_value=0.5, max_value=3.0, value=1.5, step=0.1)
fc            = st.slider('Front Camera (MP)', min_value=0, max_value=20, value=5)
int_memory    = st.number_input('Internal Memory (GB)', min_value=2, max_value=128, value=16, step=2)
mobile_wt     = st.number_input('Mobile Weight (grams)', min_value=80, max_value=250, value=150, step=5)
n_cores       = st.slider('Number of Cores', min_value=1, max_value=8, value=4)
px_height     = st.slider('Screen Resolution Height (px)', min_value=200, max_value=3000, value=800)

blue          = st.selectbox('Bluetooth', options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
dual_sim      = st.selectbox('Dual SIM', options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
four_g        = st.selectbox('4G Support', options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
three_g       = st.selectbox('3G Support', options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
touch_screen  = st.selectbox('Touch Screen', options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
wifi          = st.selectbox('WiFi', options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
ram           = st.slider('RAM (MB)', min_value=256, max_value=4096, value=2048, step=256)

m_dep         = st.number_input('Mobile Depth (cm)', min_value=0.0, max_value=1.0, value=0.5, step=0.01, format="%.2f")
pc            = st.slider('Primary Camera (MP)', min_value=0, max_value=20, value=12)
px_width      = st.slider('Screen Resolution Width (px)', min_value=100, max_value=2000, value=600)
sc_h          = st.slider('Screen Height (cm)', min_value=5, max_value=20, value=10)
sc_w          = st.slider('Screen Width (cm)', min_value=2, max_value=10, value=5)
talk_time     = st.number_input('Talk Time (hours)', min_value=1, max_value=50, value=10)

# ------------------------------
# Prediction Button (Centered)
# ------------------------------
with st.container():
    st.markdown('<div class="center-button">', unsafe_allow_html=True)
    predict_button = st.button('🎯 Predict Price Range')
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------
# Prediction and Result Display
# ------------------------------
# Mapping numeric prediction to descriptive category with emoji
price_mapping = {
    0: "📉 Low Range",
    1: "📊 Mid-Range",
    2: "💫 Mid Range-Flagship",
    3: "🚀 Flagship"
}

if model and predict_button:
    try:
        # Prepare input data as a DataFrame matching the training features order
        input_data = pd.DataFrame({
            'battery_power': [battery_power],
            'blue': [blue],
            'clock_speed': [clock_speed],
            'dual_sim': [dual_sim],
            'fc': [fc],
            'four_g': [four_g],
            'int_memory': [int_memory],
            'm_dep': [m_dep],
            'mobile_wt': [mobile_wt],
            'n_cores': [n_cores],
            'pc': [pc],
            'px_height': [px_height],
            'px_width': [px_width],
            'ram': [ram],
            'sc_h': [sc_h],
            'sc_w': [sc_w],
            'talk_time': [talk_time],
            'three_g': [three_g],
            'touch_screen': [touch_screen],
            'wifi': [wifi]
        })
        
        # Make prediction using the loaded model
        prediction_numeric = model.predict(input_data)[0]
        prediction_text = price_mapping.get(prediction_numeric, "Unknown")
        
        # Display result with attractive styling (like header-container)
        st.markdown(f"""
            <div class="result-box">
                <p class="result-title">Predicted Price Range:</p>
                <p class="result-text">The model predicts the Mobile is a <strong>{prediction_text}</strong> Mobile.</p>
            </div>
        """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")

# ------------------------------
# Additional Information Expander
# ------------------------------
with st.expander("About this predictor"):
    st.markdown("""
        This mobile price prediction model is based on a Random Forest algorithm trained on a comprehensive dataset of mobile specifications.  
        **Key Features Used:**  
        - Battery Power  
        - Clock Speed  
        - Camera Resolution (Front & Primary)  
        - Memory (Internal and RAM)  
        - Screen Resolution and Dimensions  
        - Connectivity Features (Bluetooth, Dual SIM, 3G, 4G, WiFi, Touch Screen)  
        
        **Price Range Categories:**  
        - **0: 📉 Low Range**  
        - **1: 📊 Mid-Range**  
        - **2: 💫 Mid Range-Flagship**  
        - **3: 🚀 Flagship**
    """)
    
with st.expander("About This App"):
    st.markdown("""
        **Developed by:** Dhruv Jaradi & Kevin Dave  
        **Dataset Used:** https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv  
        **Machine Learning Algorithm:** Random Forest Classifier  
        **Description:**  
        This app leverages advanced machine learning techniques to analyze mobile specifications and accurately predict the mobile's price category.
    """)
