import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

# ===========================
# LOAD MODEL + DATA
# ===========================

import json

@st.cache_resource
def load_model():
    with open('../models/xgb_salary_model.pkl', 'rb') as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv('../data/processed/cleaned_salaries.csv')

@st.cache_data
def load_columns():
    with open('../models/feature_columns.json', 'r') as f:
        return json.load(f)

model   = load_model()
df      = load_data()
columns = load_columns()

# ===========================
# APP HEADER
# ===========================

st.title("ds salary prediction")
st.markdown("*Predict your Data Science salary based on role, experience, and location*")
st.divider()


st.sidebar.header("🔍 Your Profile")

experience = st.sidebar.selectbox(
    "Experience Level",
    options=['EN', 'MI', 'SE', 'EX'],
    format_func=lambda x: {
        'EN': 'Entry Level',
        'MI': 'Mid Level', 
        'SE': 'Senior Level',
        'EX': 'Executive'
    }[x]
)

job_title = st.sidebar.selectbox(
    "Job Title",
    options=sorted(df['job_title'].unique())
)

company_location = st.sidebar.selectbox(
    "Company Location",
    options=sorted(df['company_location'].unique()),
    index=list(sorted(df['company_location'].unique())).index('US')
)

company_size = st.sidebar.selectbox(
    "Company Size",
    options=['S', 'M', 'L'],
    format_func=lambda x: {
        'S': 'Small',
        'M': 'Medium',
        'L': 'Large'
    }[x]
)

remote_ratio = st.sidebar.selectbox(
    "Remote Ratio",
    options=[0, 50, 100],
    format_func=lambda x: {
        0:   'Onsite',
        50:  'Hybrid',
        100: 'Fully Remote'
    }[x]
)

work_year = st.sidebar.selectbox(
    "Salary Year Benchmark",  # ← mas clear
    options=[2020, 2021, 2022, 2023],
    index=3
)


# ===========================
# SALARY PREDICTION
# ===========================

st.header("💰 Salary Prediction")

if st.button("Predict My Salary", type="primary"):
    
    # Build input row — same structure as training data
    input_data = pd.DataFrame(columns=columns)
    input_data.loc[0] = 0  # start with all zeros

    # Fill in values
    exp_map  = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    size_map = {'S': 0, 'M': 1, 'L': 2}

    input_data['work_year']          = work_year
    input_data['remote_ratio']       = remote_ratio
    input_data['experience_encoded'] = exp_map[experience]
    input_data['size_encoded']       = size_map[company_size]

    # One-hot columns
    if f'employment_type_FT' in input_data.columns:
        input_data['employment_type_FT'] = 1  # assume full time

    if f'company_location_{company_location}' in input_data.columns:
        input_data[f'company_location_{company_location}'] = 1

    if f'job_title_{job_title}' in input_data.columns:
        input_data[f'job_title_{job_title}'] = 1

    # Predict
    prediction = model.predict(input_data.astype(float))[0]

    # Display
    st.success(f"### Predicted Salary: ${prediction:,.0f} / year")
    st.caption(f"≈ ${prediction/12:,.0f} / month | "
               f"≈ ${prediction/2080:,.0f} / hour")