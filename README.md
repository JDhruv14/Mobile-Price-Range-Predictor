# 📱 Mobile Price Prediction using Machine Learning

<div align="center">
  <img src="https://i.gifer.com/5hoN.gif" width="300" alt="Mobile Prediction">
</div>

## 🔍 Objective
To develop a web-based application that accurately predicts the price range of a mobile phone based on its specifications using a machine learning model.

---

## 📂 Dataset Used

- **Source**: [Kaggle - Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification?select=train.csv)
- **Description**: The dataset contains features like RAM, battery power, camera resolution, screen size, connectivity features, and more for 2000 mobile phones.  
- **Target Variable**: `price_range` with values:
  - `0` = 📉 Low Range  
  - `1` = 📊 Mid-Range  
  - `2` = 💫 Mid Range-Flagship  
  - `3` = 🚀 Flagship  

---

## 🧠 Model Chosen

- **Algorithm**: Random Forest Classifier  
- **Why Random Forest?**
  - Handles both numerical and categorical features well
  - Robust to outliers and overfitting
  - Provides high accuracy with minimal hyperparameter tuning

---

## ⚙️ Features Used

- `battery_power`, `ram`, `int_memory`, `clock_speed`, `fc`, `pc`  
- `n_cores`, `m_dep`, `mobile_wt`, `px_height`, `px_width`  
- `sc_h`, `sc_w`, `talk_time`  
- Connectivity: `blue`, `dual_sim`, `four_g`, `three_g`, `wifi`, `touch_screen`

---

## 📊 Performance Metrics

- **Accuracy**: ~89% on test set
- **Train-Test Split**: 80-20
- **Model Evaluation**:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1-Score)

---

## 🌐 Web App Overview

Built with **Streamlit**, this app allows users to:
- Input mobile specifications via interactive widgets
- Get instant predictions on the price category
- View animated GIFs and a clean, mobile-themed UI
- Learn about the model and dataset via expandable sections

---

## 🛠️ Installation & Running the App

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mobile-price-prediction.git
   cd mobile-price-prediction
