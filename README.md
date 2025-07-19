Salary Prediction Web App
A web application that predicts employee salaries based on personal and professional factors like age, gender, education, job title, and experience.
This project combines machine learning with an easy-to-use interface to help understand salary trends and make data-driven compensation decisions.
What it does

Real-time predictions: Enter your details and get an instant salary estimate
Machine learning backend: Uses a trained model that achieved 91% accuracy (R² score) with RMSE of ₹15,616
Clean interface: Built with Streamlit for a straightforward user experience
Real data: Model trained on actual salary data from Kaggle

Model Performance
MetricValueMean Squared Error243,880,416.72Root Mean Squared Error₹15,616.67R² Score0.91
Tech Stack

Python 3.10+
Streamlit (web interface)
Scikit-learn (machine learning)
Pandas & NumPy (data processing)
Joblib (model persistence)

How it works
The app takes five inputs from users:

Age
Gender
Education level
Job title
Years of experience

These details are fed into a pre-trained machine learning pipeline (salary_prediction_pipeline.pkl) that outputs a salary prediction in Indian Rupees.
Running locally
Clone this repository:
bashgit clone https://github.com/yourusername/salary-prediction-app.git
cd salary-prediction-app

The main changes I made:

Removed excessive emojis and marketing language
Made the tone more conversational and less promotional
Simplified the structure while keeping all important information
Used more natural phrasing throughout
Reduced the "sales pitch" feel while maintaining technical accuracy
