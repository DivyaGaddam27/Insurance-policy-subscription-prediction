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

    # Drop ID column first
    train_data = train_data.drop(columns=['id'])
    test_data = test_data.drop(columns=['id'])

    # Feature Engineering
    for df in [train_data, test_data]:
        df['balance_age_ratio'] = df['balance'] / df['age']
        df['duration_campaign_ratio'] = df['duration'] / df['campaign']
        df['is_retired'] = (df['job'] == 'retired').astype(int)
        df['previous_campaigns_success_rate'] = df['previous'] / df['campaign']
        df['contacted_before'] = (df['pdays'] > -1).astype(int)

    # Store original categorical values for reference
    categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'poutcome']
    
    # Create encoders for each column
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        train_data[col] = encoder.fit_transform(train_data[col])
        test_data[col] = encoder.transform(test_data[col])
        encoders[col] = encoder

    X = train_data.drop('y', axis=1)
    y = train_data['y']

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    model = RandomForestClassifier(max_depth=5, random_state=0, class_weight='balanced')
    model.fit(X_train_scaled, y_train)

    val_accuracy = accuracy_score(y_val, model.predict(X_val_scaled))

    return model, scaler, val_accuracy, list(X.columns), encoders

# ------------------ STREAMLIT APP ------------------

st.title("📈 Insurance Subscription ML App")
st.write("This app trains a model on `Insurance_Train.csv` and allows prediction using new user input.")

with st.spinner("Training model..."):
    model, scaler, accuracy, feature_names, encoders = load_and_train()
    st.success(f"✅ Model trained with validation accuracy: **{accuracy:.2%}**")

st.header("🎯 Predict Insurance Subscription for New Customer")

# Input fields
age = st.number_input("Age", 18, 100, value=35)
balance = st.number_input("Balance", value=1000)
duration = st.number_input("Call Duration (seconds)", value=200)
campaign = st.number_input("Campaign Contacts", 1, value=2)
previous = st.number_input("Previous Contacts", value=0)
pdays = st.number_input("Days Since Last Contact (-1 if never contacted)", value=-1)

job = st.selectbox("Job", [
    'admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
    'retired', 'self-employed', 'services', 'student', 'technician',
    'unemployed', 'unknown'
])

marital = st.selectbox("Marital Status", ['divorced', 'married', 'single'])

education = st.selectbox("Education", ['primary', 'secondary', 'tertiary', 'unknown'])

housing = st.selectbox("Housing Loan", ['no', 'yes'])

loan = st.selectbox("Personal Loan", ['no', 'yes'])

contact = st.selectbox("Contact Type", ['cellular', 'telephone', 'unknown'])

month = st.selectbox("Month", [
    'apr', 'aug', 'dec', 'feb', 'jan', 'jul',
    'jun', 'mar', 'may', 'nov', 'oct', 'sep'
])

poutcome = st.selectbox("Previous Outcome", ['failure', 'other', 'success', 'unknown'])

if st.button("🔍 Predict"):
    # Create a DataFrame with the same structure as training data
    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'month': [month],
        'duration': [duration],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })
    
    # Apply the same feature engineering
    input_data['balance_age_ratio'] = input_data['balance'] / input_data['age']
    input_data['duration_campaign_ratio'] = input_data['duration'] / input_data['campaign']
    input_data['is_retired'] = (input_data['job'] == 'retired').astype(int)
    input_data['previous_campaigns_success_rate'] = input_data['previous'] / input_data['campaign']
    input_data['contacted_before'] = (input_data['pdays'] > -1).astype(int)
    
    # Apply label encoding using the same encoders
    categorical_columns = ['job', 'marital', 'education', 'housing', 'loan', 'contact', 'month', 'poutcome']
    for col in categorical_columns:
        input_data[col] = encoders[col].transform(input_data[col])
    
    # Ensure the columns are in the same order as training
    input_data = input_data[feature_names]
    
    # Scale the input
    X_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(X_scaled)[0]
    prediction_proba = model.predict_proba(X_scaled)[0]
    
    # Display results
    if prediction == 1:
        st.success(f"✅ YES: Likely to Subscribe (Probability: {prediction_proba[1]:.2%})")
    else:
        st.error(f"❌ NO: Unlikely to Subscribe (Probability: {prediction_proba[1]:.2%})")
    
    # Show feature importance for debugging
    with st.expander("🔍 Feature Values (for debugging)"):
        st.write("Input feature values:")
        feature_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': X_scaled[0]
        })
        st.dataframe(feature_df)
        
        st.write("Prediction probabilities:")
        st.write(f"Class 0 (No): {prediction_proba[0]:.2%}")
        st.write(f"Class 1 (Yes): {prediction_proba[1]:.2%}")