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

job = st.selectbox("Job", [
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
    'retired', 'self-employed', 'services', 'student', 'technician',
    'unemployed', 'unknown'
])

marital = st.selectbox("Marital", ['divorced', 'married', 'single'])

education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])

housing = st.selectbox("Housing Loan", ['no', 'yes'])

loan = st.selectbox("Personal Loan", ['no', 'yes'])

contact = st.selectbox("Contact", ['cellular', 'telephone', 'unknown'])

month = st.selectbox("Month", [
    'apr', 'aug', 'dec', 'feb', 'jan', 'jul',
    'jun', 'mar', 'may', 'nov', 'oct', 'sep'
])

poutcome = st.selectbox("Previous Outcome", ['failure', 'other', 'success', 'unknown'])


# Encoding maps
job_map = {'admin.': 0, 'blue-collar': 1, 'entrepreneur': 2, 'housemaid': 3,
           'management': 4, 'retired': 5, 'self-employed': 6, 'services': 7,
           'student': 8, 'technician': 9, 'unemployed': 10, 'unknown': 11}

marital_map = {'divorced': 0, 'married': 1, 'single': 2}

education_map = {'primary': 0, 'secondary': 1, 'tertiary': 2, 'unknown': 3}

housing_map = {'no': 0, 'yes': 1}

loan_map = {'no': 0, 'yes': 1}

contact_map = {'cellular': 0, 'telephone': 1, 'unknown': 2}

month_map = {'apr': 0, 'aug': 1, 'dec': 2, 'feb': 3, 'jan': 4, 'jul': 5,
             'jun': 6, 'mar': 7, 'may': 8, 'nov': 9, 'oct': 10, 'sep': 11}

poutcome_map = {'failure': 0, 'other': 1, 'success': 2, 'unknown': 3}


# Feature engineering
balance_age_ratio = balance / age if age > 0 else 0
duration_campaign_ratio = duration / campaign if campaign > 0 else 0
previous_campaigns_success_rate = previous / campaign if campaign > 0 else 0

is_retired = 1 if job == 'retired' else 0
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
