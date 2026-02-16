# Insurance Policy Subscription Prediction (ML App)

A machine learning web application that predicts whether a customer is likely to subscribe to an insurance policy based on their demographic and financial details. The model is deployed as an interactive **Streamlit** app and hosted on **Hugging Face Spaces**.

🔗 **Live Demo (Hugging Face Space):**  
https://huggingface.co/spaces/Divyag26/insurance-prediction

---

## Features

- Predicts insurance policy subscription (Yes / No)
- Interactive Streamlit UI
- Feature scaling and preprocessing
- Deployed on Hugging Face Spaces (Docker)
- Model and scaler tracked with Git LFS

---

## Tech Stack

- **Language:** Python  
- **ML Libraries:** scikit-learn, pandas, numpy  
- **Web App:** Streamlit  
- **Deployment:** Hugging Face Spaces (Docker)  
- **Version Control:** Git + GitHub  
- **Model Storage:** Git LFS  

---

## Model & Workflow

1. Data preprocessing and feature engineering  
2. Train classification model on insurance dataset  
3. Save trained model (`model.pkl`) and scaler (`scaler.pkl`)  
4. Build Streamlit UI for user input  
5. Load model & scaler for real-time prediction  
6. Deploy app on Hugging Face Spaces using Docker  

---

## Project Structure

```text
Insurance-policy-subscription-prediction/
│
├── src/
│   ├── streamlit_app.py
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── Insurance_Train.csv
│   └── Insurance_Test.csv
│
├── Dockerfile
├── requirements.txt
├── README.md
└── .gitattributes
