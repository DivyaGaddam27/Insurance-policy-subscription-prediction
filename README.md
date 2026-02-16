Insurance Policy Subscription Prediction (ML App)

A machine learning web application that predicts whether a customer is likely to subscribe to an insurance policy based on their demographic and financial details. The model is deployed as an interactive Streamlit app and hosted on Hugging Face Spaces.

🔗 Live Demo (Hugging Face Space):
https://huggingface.co/spaces/Divyag26/insurance-prediction

Features

Predicts insurance policy subscription (Yes / No)

Clean and interactive Streamlit UI

Preprocessing with feature scaling

Trained ML model integrated into a deployed web app

Dockerized for easy deployment

Model and scaler stored using Git LFS

Tech Stack

Language: Python

ML Libraries: scikit-learn, pandas, numpy

Web App: Streamlit

Deployment: Hugging Face Spaces (Docker-based)

Version Control: Git + GitHub

Model Storage: Git LFS

Model & Workflow

Data preprocessing and feature engineering

Train classification model on insurance dataset

Save trained model (model.pkl) and scaler (scaler.pkl)

Build Streamlit UI for user input

Load model & scaler for real-time prediction

Deploy app on Hugging Face Spaces using Docker

Project Structure
Insurance-policy-subscription-prediction/
│
├── src/
│   ├── streamlit_app.py     # Streamlit UI
│   ├── model.pkl            # Trained ML model
│   ├── scaler.pkl           # Feature scaler
│   ├── Insurance_Train.csv  # Training data
│   └── Insurance_Test.csv   # Test data
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitattributes

How to Run Locally
git clone https://github.com/DivyaGaddam27/Insurance-policy-subscription-prediction.git
cd Insurance-policy-subscription-prediction

pip install -r requirements.txt
streamlit run src/streamlit_app.py


Then open:

http://localhost:8501

Deployment

The app is deployed on Hugging Face Spaces using Docker.
Model files are tracked using Git LFS to handle large files efficiently.

Future Improvements

Add model performance metrics (Accuracy, ROC-AUC) to UI

Add model comparison (Logistic Regression vs Random Forest)

Improve feature engineering

Add user authentication

Add data validation and logging

👩‍💻 Author

Divya Gaddam

Hugging Face: https://huggingface.co/Divyag26
