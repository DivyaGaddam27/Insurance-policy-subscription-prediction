import os
os.environ["STREAMLIT_HOME"] = os.getcwd()

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ------------------ TRAINING + PREPROCESSING ------------------

@st.cache_data
def load_and_train():
    train_data = pd.read_csv('src/Insurance_Train.csv')
    test_data = pd.read_csv('src/Insurance_Test.csv')

    # Feature Engineering
    for df in [train_data, test_data]:
        df['balance_age_ratio'] = df['balance'] / df['age']
        df['duration_campaign_ratio'] = df['duration'] / df['campaign']
        df['is_retired'] = (df['job'] == 'retired').astype(int)
        df['previous_campaigns_success_rate'] = df['previous'] / df['campaign']
        df['contacted_before'] = (df['pdays'] > -1).astype(int)

    train_data = train_data.drop(columns=['id'])
    test_data = test_data.drop(columns=['id'])

    # Label Encoding
    encoder = LabelEncoder()
    categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in categorical_columns:
        train_data[col] = encoder.fit_transform(train_data[col])
        test_data[col] = encoder.transform(test_data[col])

    X = train_data.drop('y', axis=1)
    y = train_data['y']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(max_depth=5, random_state=0, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    val_accuracy = accuracy_score(y_val, model.predict(X_val_scaled))

    return model, scaler, val_accuracy, list(X.columns)

# ------------------ STREAMLIT APP ------------------

st.title("📈 Insurance Subscription ML App")
st.write("This app trains a model on `Insurance_Train.csv` and allows prediction using new user input.")

with st.spinner("Training model..."):
    model, scaler, accuracy, feature_names = load_and_train()
    st.success(f"✅ Model trained with validation accuracy: **{accuracy:.2%}**")

st.header("🎯 Predict Insurance Subscription for New Customer")

# Input fields
age = st.number_input("Age", 18, 100)
balance = st.number_input("Balance")
duration = st.number_input("Call Duration")
campaign = st.number_input("Campaign Contacts", 1)
previous = st.number_input("Previous Contacts")
pdays = st.number_input("Days Since Last Contact", -1)

job = st.selectbox("Job", ['admin.', 'technician', 'retired'])
marital = st.selectbox("Marital", ['married', 'single'])
education = st.selectbox("Education", ['secondary', 'tertiary'])
housing = st.selectbox("Housing Loan", ['yes', 'no'])
loan = st.selectbox("Personal Loan", ['yes', 'no'])
contact = st.selectbox("Contact", ['cellular', 'telephone'])
month = st.selectbox("Month", ['may', 'jul'])  # Extend as needed
poutcome = st.selectbox("Previous Outcome", ['success', 'failure'])

# Encoding maps
job_map = {'admin.': 0, 'technician': 1, 'retired': 2}
marital_map = {'married': 0, 'single': 1}
education_map = {'secondary': 0, 'tertiary': 1}
housing_map = {'yes': 1, 'no': 0}
loan_map = {'yes': 1, 'no': 0}
contact_map = {'cellular': 0, 'telephone': 1}
month_map = {'may': 0, 'jul': 1}
poutcome_map = {'success': 1, 'failure': 0}

# Feature engineering
balance_age_ratio = balance / age
duration_campaign_ratio = duration / campaign
is_retired = 1 if job == 'retired' else 0
previous_campaigns_success_rate = previous / campaign
contacted_before = 1 if pdays > -1 else 0

# Input vector
X_input = np.array([[age, job_map[job], marital_map[marital], education_map[education],
                     housing_map[housing], loan_map[loan], contact_map[contact], month_map[month],
                     pdays, previous, poutcome_map[poutcome], campaign,
                     balance, duration,
                     balance_age_ratio, duration_campaign_ratio, is_retired,
                     previous_campaigns_success_rate, contacted_before]])

input_df = pd.DataFrame(X_input, columns=feature_names)
X_scaled = scaler.transform(input_df)


if st.button("🔍 Predict"):
    prediction = model.predict(X_scaled)[0]
    result = "✅ YES: Likely to Subscribe" if prediction == 1 else "❌ NO: Unlikely to Subscribe"
    st.success(f"Prediction: {result}")
