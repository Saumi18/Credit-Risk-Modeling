import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
# --- Load models ---
@st.cache_resource
def load_models():
    lgb_model = joblib.load('lightgbm_model.pkl')
    mlp_model = joblib.load('mlp_model.pkl')
    return lgb_model, mlp_model

# --- Load scaler if needed ---
@st.cache_resource
def load_scaler():
    return joblib.load('scaler.pkl')  # save your scaler as 'scaler.pkl'

# --- Prediction ---
def predict(model, X, model_name):
    if model_name == 'LightGBM':
        prob = model.predict(X)
        pred = (prob > 0.5).astype(int)
    else:
        prob = model.predict_proba(X)[:, 1]
        pred = model.predict(X)
    return pred, prob

# --- App Layout ---
st.set_page_config(page_title="Credit Score Predictor", layout="centered")
st.title("📊 Credit Score Behavior Prediction")
st.markdown("Predict customer behavior using **MLP** or **LightGBM** models.")

# --- Sidebar ---
model_choice = st.sidebar.selectbox("Choose Model", ["LightGBM", "MLP"])
uploaded_file = st.sidebar.file_uploader("Upload CSV for Prediction", type=['csv'])

# --- Main Section ---
lgb_model, mlp_model = load_models()
scaler = load_scaler()

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    # Preprocess input (assumes same features as training)
    X_input = scaler.transform(df)  # must match training preprocessing

    model = lgb_model if model_choice == "LightGBM" else mlp_model
    predictions, probabilities = predict(model, X_input, model_choice)

    df['Prediction'] = predictions
    df['Probability'] = probabilities

    st.subheader("Prediction Results")
    st.dataframe(df)

    # Optional download
    csv = df.to_csv(index=False).encode()
    st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

else:
    st.info("Please upload a CSV file from sidebar to get predictions.")

# --- Test metrics display (optional) ---
st.markdown("---")
if st.checkbox("Show Model Evaluation on Test Data (for validation)"):
    # Load your test data (ensure path and preprocessing match)
    test_data = pd.read_csv("test_data.csv")  # Replace with your test set
    X_test = scaler.transform(test_data.drop(columns=["target"]))
    y_test = test_data["target"]

    model = lgb_model if model_choice == "LightGBM" else mlp_model
    y_pred, y_prob = predict(model, X_test, model_choice)

    st.markdown("**Classification Report:**")
    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    st.markdown("**Confusion Matrix:**")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

    st.markdown(f"**ROC AUC Score:** `{roc_auc_score(y_test, y_prob):.4f}`")