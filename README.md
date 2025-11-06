# 🩷 **Breast Cancer Prediction using Machine Learning** 🧠  
*An intelligent and interpretable ML web app for early breast cancer detection.*

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Logistic%20Regression-yellow)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🌐 **Live Demo**
👉 [**Click here to use the Breast Cancer Prediction App**](https://breastcancerprediction-ydz3dsaev3wb87pqdmy3sf.streamlit.app) 🚀  
*(Hosted on Streamlit Cloud)*

---

## 💡 **Introduction**

Breast cancer is one of the most common and life-threatening diseases affecting women globally.  
This project — **"Breast Cancer Prediction using Machine Learning"** — uses a **Logistic Regression** model to predict whether a tumor is **Benign (non-cancerous)** or **Malignant (cancerous)** based on diagnostic data.  

The web app is built using **Streamlit**, enabling users to easily input tumor characteristics and receive instant predictions along with confidence scores.

---

## 🧠 **Overview**

This project demonstrates how **Machine Learning** can assist in **early cancer detection**, providing accurate and interpretable predictions.  
The model has been trained on the **Kaggle Breast Cancer Diagnostic Dataset**, achieving high accuracy while maintaining transparency in decision-making — a key factor in medical AI applications.

---

## 📊 **Dataset**

- **Source:** [Kaggle - Breast Cancer Diagnostic Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)  
- **Description:**  
  The dataset contains **569 samples** of breast cell nuclei measurements.  
  Each record includes **30 numerical features** such as radius, texture, smoothness, and area mean — along with a **diagnosis label**:
  - `M` → Malignant  
  - `B` → Benign  

---

## ⚙️ **Tech Stack**

| Tool / Library | Purpose |
|-----------------|----------|
| 🐍 **Python** | Core programming language |
| 📘 **Scikit-learn** | ML model training & evaluation |
| 💻 **Streamlit** | Interactive web app |
| 🧮 **NumPy** | Numerical operations |
| 🧾 **Pandas** | Data handling & preprocessing |
| 💾 **Joblib** | Model & scaler serialization |
| 📊 **Plotly / Matplotlib / Seaborn** | Data visualization & insights |

---

## 🧮 **Model Used: Logistic Regression**

- **Reason for choice:**  
  Logistic Regression offers **high interpretability**, **low computational cost**, and **robust performance** for binary classification tasks like cancer prediction.  
- **Output:**  
  - Prediction: *Malignant* or *Benign*  
  - Probability: Confidence score for the classification

---

## 💻 **Project Structure**

```bash
Breast-Cancer-Prediction/
│
├── breast_cancer_model.pkl          # Trained Logistic Regression model
├── scaler.pkl                       # Scaler for input normalization
├── app.py                           # Streamlit web app
├── breast_cancer_data.csv           # Dataset (optional for local testing)
├── requirements.txt                 # Dependencies
└── README.md                        # Project documentation
