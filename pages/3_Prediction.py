import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("Diabetes Risk Calculator")
st.markdown("Enter your (or any) values below to estimate the probability of diabetes.")

@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

model = load_model()

@st.cache_data
def load_feature_names():
    df = pd.read_csv("data/diabetes_binary_5050split.csv")
    return df.drop('Diabetes_binary', axis=1).columns.tolist()

feature_names = load_feature_names()
descriptions = {
    'HighBP': 'High blood pressure (0 = no, 1 = yes)',
    'HighChol': 'High cholesterol (0 = no, 1 = yes)',
    'CholCheck': 'Cholesterol check in last 5 years (0 = no, 1 = yes)',
    'BMI': 'Body Mass Index (numeric)',
    'Smoker': 'Have you smoked at least 100 cigarettes in your entire life? (0 = no, 1 = yes)',
    'Stroke': 'Ever had a stroke (0 = no, 1 = yes)',
    'HeartDiseaseorAttack': 'Coronary heart disease or myocardial infarction (0 = no, 1 = yes)',
    'PhysActivity': 'Physical activity in past 30 days (0 = no, 1 = yes)',
    'Fruits': 'Consume fruit daily (0 = no, 1 = yes)',
    'Veggies': 'Consume vegetables daily (0 = no, 1 = yes)',
    'HvyAlcoholConsump': 'Heavy alcohol consumption (0 = no, 1 = yes)',
    'AnyHealthcare': 'Have any kind of health insurance (0 = no, 1 = yes)',
    'NoDocbcCost': 'Could not see doctor because of cost (0 = no, 1 = yes)',
    'GenHlth': 'General health (1 = excellent, 5 = poor)',
    'MentHlth': 'Days of poor mental health in past 30 days (0–30)',
    'PhysHlth': 'Days of poor physical health in past 30 days (0–30)',
    'DiffWalk': 'Difficulty walking (0 = no, 1 = yes)',
    'Sex': 'Sex (0 = female, 1 = male)',
    'Age': 'Age category (1 = 18–24, 13 = 80+)',
    'Education': 'Education level (1 = never attended, 6 = college graduate)',
    'Income': 'Income level (1 = < $10k, 8 = > $75k)'
}

st.subheader("Enter your data:")

input_data = []
for feature in feature_names:
    desc = descriptions.get(feature, feature)
    if feature in ['BMI', 'MentHlth', 'PhysHlth']:
        if feature == 'BMI':
            value = st.slider(desc, min_value=10, max_value=60, value=25, step=1)
        else:
            value = st.slider(desc, min_value=0, max_value=30, value=0, step=1)
    elif feature in ['GenHlth', 'Age', 'Education', 'Income']:
        if feature == 'GenHlth':
            value = st.slider(desc, min_value=1, max_value=5, value=3, step=1)
        elif feature == 'Age':
            value = st.slider(desc, min_value=1, max_value=13, value=7, step=1)
        elif feature == 'Education':
            value = st.slider(desc, min_value=1, max_value=6, value=4, step=1)
        else:
            value = st.slider(desc, min_value=1, max_value=8, value=5, step=1)
    else:
        value = st.selectbox(desc, options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    input_data.append(value)

input_array = np.array(input_data).reshape(1, -1)

if st.button("Calculate Risk"):
    prediction = model.predict(input_array)[0]
    probability = model.predict_proba(input_array)[0][1]  # probability of class 1
    if prediction == 1:
        st.error(f"**Diabetes predicted** with probability {probability:.2f}")
    else:
        st.success(f"**No diabetes predicted** with probability {1-probability:.2f}")
    st.info("This is a demonstration model based on statistical data. It is not a medical diagnosis.")
