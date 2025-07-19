# 💼 AI-Powered Salary Prediction Web App

Welcome to the **AI Salary Predictor** – an intelligent web application that estimates employee salaries based on key factors such as age, gender, education level, job title, and years of experience.

> 🎯 Built using **Python, Streamlit, and Machine Learning**, this project aims to bridge data and decision-making with ease and accuracy.

---

## 🚀 Features

- 🔍 **Instant Salary Prediction**  
  Upload details like age, education, job title, and more to get real-time salary estimations.

- 🤖 **Trained ML Pipeline**  
  Powered by a highly optimized machine learning model with:
  - ✅ **R² Score**: 0.91 (Excellent performance)
  - ✅ **RMSE**: ₹15,616.67

- 📦 **Interactive Web UI**  
  Built using **Streamlit**, offering a clean and intuitive user interface.

- 💾 **Model Trained on Kaggle Dataset**  
  Uses real-world data to deliver accurate predictions.

---

## 📊 Model Performance

| Metric | Value |
|--------|--------|
| **Mean Squared Error (MSE)** | 243,880,416.72 |
| **Root Mean Squared Error (RMSE)** | ₹15,616.67 |
| **R-squared (R² Score)** | 0.91 |

---

## 🧠 Technologies Used

- **Python 3.10+**
- **Streamlit**
- **Scikit-learn**
- **Pandas, NumPy**
- **Joblib** for model saving/loading

---

## 💡 How It Works

1. User inputs:
   - Age
   - Gender
   - Education Level
   - Job Title
   - Years of Experience

2. The input is passed to a **trained ML pipeline** (`salary_prediction_pipeline.pkl`).

3. The model outputs an estimated **salary (in INR ₹)** based on historical trends.

---

## 🛠 How to Run Locally

1. **Clone the repo**  
   ```bash
   git clone https://github.com/yourusername/salary-prediction-app.git
   cd salary-prediction-app
