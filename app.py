import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os
import pickle

# Page configuration
st.set_page_config(
    page_title="💼 ML Salary Prediction App",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling and animations
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: slideDown 0.8s ease-out;
    }
    
    .main-header h1 {
        color: white;
        text-align: center;
        font-weight: 700;
        font-size: 2.5rem;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
    }
    
    .input-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin-bottom: 2rem;
        border: 1px solid #e0e0e0;
        animation: fadeInUp 0.6s ease-out;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        animation: pulse 2s infinite;
    }
    
    .prediction-amount {
        font-size: 2.5rem;
        font-weight: 700;
        color: white;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .prediction-label {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin: 0;
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 3px 15px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    .sidebar-header {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
        text-align: center;
        font-weight: 600;
    }
    
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    
    .model-status {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: 600;
    }
    
    .model-trained {
        background: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    
    .model-not-trained {
        background: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    
    @keyframes slideDown {
        from {
            transform: translateY(-100px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes fadeInUp {
        from {
            transform: translateY(50px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    .stSelectbox > div > div {
        border-radius: 10px;
    }
    
    .stNumberInput > div > div {
        border-radius: 10px;
    }
    
    .stSlider > div > div {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_sample_data():
    """Create sample training data for the ML model"""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.normal(35, 10, n_samples).clip(22, 65).astype(int),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'Education_Level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                          n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'Job_Title': np.random.choice(['Developer', 'Data Scientist', 'Manager', 'Analyst', 'Engineer'], 
                                     n_samples, p=[0.25, 0.15, 0.2, 0.2, 0.2]),
        'Years_Experience': np.random.exponential(5, n_samples).clip(0, 30).astype(int)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic salary based on features
    base_salary = 400000
    
    # Education impact
    edu_bonus = df['Education_Level'].map({
        'High School': 0,
        'Bachelor': 150000,
        'Master': 300000,
        'PhD': 500000
    })
    
    # Job title impact
    job_multiplier = df['Job_Title'].map({
        'Developer': 1.2,
        'Data Scientist': 1.5,
        'Manager': 1.8,
        'Analyst': 1.0,
        'Engineer': 1.3
    })
    
    # Experience and age impact
    exp_bonus = df['Years_Experience'] * 45000
    age_factor = np.where(df['Age'] < 30, 0.9, 
                 np.where(df['Age'] < 40, 1.1,
                 np.where(df['Age'] < 50, 1.2, 1.0)))
    
    # Gender impact (unfortunately realistic in many contexts)
    gender_factor = np.where(df['Gender'] == 'Male', 1.0, 0.95)
    
    # Calculate salary with some noise
    df['Salary'] = ((base_salary + edu_bonus + exp_bonus) * job_multiplier * age_factor * gender_factor + 
                   np.random.normal(0, 50000, n_samples)).clip(300000, 3000000)
    
    return df

def prepare_features(df):
    """Encode categorical features for ML model"""
    df_encoded = df.copy()
    
    # Label encode categorical variables
    le_gender = LabelEncoder()
    le_education = LabelEncoder()
    le_job = LabelEncoder()
    
    df_encoded['Gender_encoded'] = le_gender.fit_transform(df_encoded['Gender'])
    df_encoded['Education_encoded'] = le_education.fit_transform(df_encoded['Education_Level'])
    df_encoded['Job_encoded'] = le_job.fit_transform(df_encoded['Job_Title'])
    
    return df_encoded, le_gender, le_education, le_job

def train_model():
    """Train the ML model and save it using joblib"""
    with st.spinner("🤖 Training ML model... This may take a moment."):
        try:
            # Create sample data
            df = create_sample_data()
            
            # Prepare features
            df_encoded, le_gender, le_education, le_job = prepare_features(df)
            
            # Feature selection
            features = ['Age', 'Gender_encoded', 'Education_encoded', 'Job_encoded', 'Years_Experience']
            X = df_encoded[features]
            y = df_encoded['Salary']
            
            # Train Random Forest model
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X, y)
            
            # Save model and encoders using joblib
            model_data = {
                'model': model,
                'le_gender': le_gender,
                'le_education': le_education,
                'le_job': le_job,
                'feature_names': features,
                'training_data_shape': X.shape,
                'model_score': model.score(X, y)
            }
            
            # Save using joblib
            joblib.dump(model_data, 'salary_model.joblib')
            
            # Also save using pickle as backup
            with open('salary_model.pkl', 'wb') as f:
                pickle.dump(model_data, f)
            
            return model_data, True, f"Model trained successfully! R² Score: {model_data['model_score']:.3f}"
            
        except Exception as e:
            return None, False, f"Training failed: {str(e)}"

def load_model():
    """Load the trained model using joblib"""
    try:
        if os.path.exists('salary_model.joblib'):
            model_data = joblib.load('salary_model.joblib')
            return model_data, True, "Model loaded successfully from joblib!"
        elif os.path.exists('salary_model.pkl'):
            with open('salary_model.pkl', 'rb') as f:
                model_data = pickle.load(f)
            return model_data, True, "Model loaded successfully from pickle backup!"
        else:
            return None, False, "No trained model found. Please train the model first."
    except Exception as e:
        return None, False, f"Failed to load model: {str(e)}"

def predict_salary_ml(model_data, age, gender, education_level, job_title, experience):
    """Make salary prediction using the trained ML model"""
    try:
        model = model_data['model']
        le_gender = model_data['le_gender']
        le_education = model_data['le_education']
        le_job = model_data['le_job']
        
        # Encode inputs
        gender_encoded = le_gender.transform([gender])[0]
        education_encoded = le_education.transform([education_level])[0]
        job_encoded = le_job.transform([job_title])[0]
        
        # Create feature array
        features = np.array([[age, gender_encoded, education_encoded, job_encoded, experience]])
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        return max(prediction, 300000), True, ""
        
    except Exception as e:
        return None, False, f"Prediction failed: {str(e)}"

# Data validation functions
def validate_age(age):
    if age < 18 or age > 100:
        return False, "Age must be between 18 and 100"
    return True, ""

def validate_experience(experience, age):
    if experience < 0:
        return False, "Experience cannot be negative"
    if experience > (age - 16):
        return False, "Experience cannot exceed (Age - 16) years"
    return True, ""

def get_salary_insights(age, experience, education, job_title):
    insights = []
    
    # Experience insights
    if experience < 2:
        insights.append("📈 Entry-level position - Consider gaining more experience")
    elif experience < 5:
        insights.append("🔄 Mid-junior level - Good growth potential")
    elif experience < 10:
        insights.append("⭐ Experienced professional - Strong market position")
    else:
        insights.append("🏆 Senior expert - Premium salary expectations")
    
    # Education insights
    if education == "PhD":
        insights.append("🎓 PhD qualification adds significant value")
    elif education == "Master":
        insights.append("📚 Master's degree provides competitive advantage")
    
    # Age-experience ratio
    exp_ratio = experience / (age - 18) if age > 18 else 0
    if exp_ratio > 0.8:
        insights.append("🚀 Excellent experience-to-age ratio")
    elif exp_ratio < 0.3:
        insights.append("💡 Potential for rapid career growth")
    
    return insights

def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Machine Learning Salary Prediction</h1>
        <p>Real ML-powered salary predictions using Random Forest & Joblib</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Model management section
    st.markdown("### 🔧 ML Model Management")
    
    col_model1, col_model2, col_model3 = st.columns(3)
    
    with col_model1:
        if st.button("🎯 Train New Model", use_container_width=True):
            model_data, success, message = train_model()
            if success:
                st.session_state['model_data'] = model_data
                st.success(message)
            else:
                st.error(message)
    
    with col_model2:
        if st.button("📂 Load Saved Model", use_container_width=True):
            model_data, success, message = load_model()
            if success:
                st.session_state['model_data'] = model_data
                st.success(message)
            else:
                st.error(message)
    
    with col_model3:
        if st.button("🗑️ Clear Model", use_container_width=True):
            if 'model_data' in st.session_state:
                del st.session_state['model_data']
                st.success("Model cleared from session!")
    
    # Model status
    if 'model_data' in st.session_state:
        model_data = st.session_state['model_data']
        st.markdown(f"""
        <div class="model-status model-trained">
            ✅ Model Ready | R² Score: {model_data['model_score']:.3f} | 
            Training Data: {model_data['training_data_shape'][0]} samples
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="model-status model-not-trained">
            ❌ No Model Loaded - Please train or load a model first
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📊 Prediction Settings</div>', unsafe_allow_html=True)
        
        show_insights = st.checkbox("Show Career Insights", value=True)
        show_charts = st.checkbox("Show Visualization", value=True)
        currency = st.selectbox("Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)", "£ (GBP)"])
        
        st.markdown("""
        <div class="info-box">
            <h4>🤖 About ML Model</h4>
            <ul>
                <li>Random Forest Regressor</li>
                <li>Trained on 1000+ samples</li>
                <li>Features: Age, Gender, Education, Job, Experience</li>
                <li>Saved using Joblib for persistence</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="input-section">', unsafe_allow_html=True)
        st.markdown("### 👤 Personal Information")
        
        # Personal info in columns
        per_col1, per_col2 = st.columns(2)
        
        with per_col1:
            age = st.number_input(
                "Age",
                min_value=18,
                max_value=100,
                value=30,
                help="Your current age (18-100 years)"
            )
        
        with per_col2:
            gender = st.selectbox(
                "Gender",
                ("Male", "Female"),
                help="Select your gender"
            )
        
        # Professional info
        st.markdown("### 💼 Professional Details")
        
        prof_col1, prof_col2 = st.columns(2)
        
        with prof_col1:
            education_level = st.selectbox(
                "Education Level",
                ["High School", "Bachelor", "Master", "PhD"],
                index=1,
                help="Your highest education qualification"
            )
        
        with prof_col2:
            job_title = st.selectbox(
                "Job Title",
                ["Developer", "Data Scientist", "Manager", "Analyst", "Engineer"],
                help="Your current or desired job title"
            )
        
        # Experience with advanced slider
        st.markdown("### 📈 Experience")
        experience = st.slider(
            "Years of Experience",
            min_value=0,
            max_value=50,
            value=5,
            help="Your total years of professional experience"
        )
        
        # Progress bar for experience
        exp_progress = min(experience / 20, 1.0)
        st.progress(exp_progress)
        
        if experience <= 2:
            st.info("🌱 Entry Level")
        elif experience <= 5:
            st.info("🔄 Mid-Junior Level")
        elif experience <= 10:
            st.warning("⭐ Experienced")
        else:
            st.success("🏆 Senior Expert")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Validation
        age_valid, age_msg = validate_age(age)
        exp_valid, exp_msg = validate_experience(experience, age)
        
        if not age_valid:
            st.error(age_msg)
        if not exp_valid:
            st.error(exp_msg)
        
        # Show insights
        if show_insights and age_valid and exp_valid:
            st.markdown("### 🔍 Career Insights")
            insights = get_salary_insights(age, experience, education_level, job_title)
            for insight in insights:
                st.info(insight)
    
    # Prediction section
    st.markdown("---")
    
    if st.button("🤖 Predict Salary with ML", use_container_width=True):
        if 'model_data' not in st.session_state:
            st.error("❌ No ML model loaded! Please train or load a model first.")
        elif not (age_valid and exp_valid):
            st.error("❌ Please fix the validation errors above before predicting.")
        else:
            with st.spinner("🤖 ML model is analyzing your profile..."):
                time.sleep(1)
                
                try:
                    model_data = st.session_state['model_data']
                    prediction, success, error_msg = predict_salary_ml(
                        model_data, age, gender, education_level, job_title, experience
                    )
                    
                    if not success:
                        st.error(f"❌ {error_msg}")
                        return
                    
                    # Currency conversion (simplified)
                    currency_multipliers = {
                        "₹ (INR)": 1,
                        "$ (USD)": 0.012,
                        "€ (EUR)": 0.011,
                        "£ (GBP)": 0.0095
                    }
                    
                    symbol = currency.split()[0]
                    converted_salary = prediction * currency_multipliers[currency]
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-card">
                        <p class="prediction-label">🤖 ML Predicted Annual Salary</p>
                        <div class="prediction-amount">{symbol} {converted_salary:,.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Additional metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        monthly_salary = converted_salary / 12
                        st.metric("Monthly Salary", f"{symbol} {monthly_salary:,.2f}")
                    
                    with col2:
                        hourly_rate = converted_salary / (40 * 52)
                        st.metric("Hourly Rate", f"{symbol} {hourly_rate:.2f}")
                    
                    with col3:
                        daily_rate = converted_salary / 365
                        st.metric("Daily Earning", f"{symbol} {daily_rate:.2f}")
                    
                    # Model confidence info
                    st.info(f"🎯 Model Confidence: R² Score = {model_data['model_score']:.3f}")
                    
                    # Visualization
                    if show_charts:
                        st.markdown("### 📊 ML Prediction Analysis")
                        
                        # Feature importance (if available)
                        if hasattr(model_data['model'], 'feature_importances_'):
                            feature_names = ['Age', 'Gender', 'Education', 'Job Title', 'Experience']
                            importance_df = pd.DataFrame({
                                'Feature': feature_names,
                                'Importance': model_data['model'].feature_importances_
                            }).sort_values('Importance', ascending=True)
                            
                            fig_importance = px.bar(
                                importance_df,
                                x='Importance',
                                y='Feature',
                                orientation='h',
                                title="Feature Importance in ML Model",
                                color='Importance',
                                color_continuous_scale='viridis'
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Success message
                    st.success("✅ ML Prediction completed successfully!")
                    
                    # Download option
                    result_data = {
                        "Prediction_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Model_Type": "Random Forest Regressor",
                        "Model_Score": model_data['model_score'],
                        "Age": age,
                        "Gender": gender,
                        "Education": education_level,
                        "Job_Title": job_title,
                        "Experience": experience,
                        "Predicted_Salary_INR": prediction,
                        "Currency": currency,
                        "Converted_Salary": converted_salary
                    }
                    
                    result_df = pd.DataFrame([result_data])
                    csv = result_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download ML Prediction Report",
                        data=csv,
                        file_name=f"ml_salary_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                except Exception as e:
                    st.error(f"❌ ML Prediction failed: {str(e)}")
                    st.info("💡 Please check your input data and ensure the model is properly loaded.")

if __name__ == "__main__":
    main()
