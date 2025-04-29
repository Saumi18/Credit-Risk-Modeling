# Credit Risk Modeling

This project builds a **Credit Behavior Score Model** to predict the creditworthiness of an individual based on financial and behavioral data.  
It uses machine learning models (**LightGBM** and **MLP**) to classify whether a customer will likely default or behave responsibly.

An **interactive Streamlit web app** is developed where the user can **upload a CSV file** (containing one customer's data) to get instant predictions.

---

## Project Overview

**Credit Risk Modeling** is critical for financial institutions like banks, NBFCs, and lending platforms to assess the probability that a borrower will default on loan obligations.  
This project builds a **behavior-based credit scoring system**, where the focus is not just on static demographic attributes (like age or income) but also on **dynamic behavioral patterns** — such as repayment history, credit utilization, outstanding balances, etc.

The model uses **supervised machine learning techniques** to estimate the likelihood of default (`target: good/bad behavior`) based on historical financial behavior data.

Key technical points:

- **Features**:  
  Include a mix of **customer demographics**, **financial attributes** (credit limit, utilization rates), and **payment behaviors** (payment delays, missed payments).

- **Target**:  
  A binary classification where the model predicts if the customer's future behavior is "good" (no default) or "bad" (default risk).

- **Modeling Approach**:
  - **LightGBM (Gradient Boosted Trees)**: Efficient for large datasets and handles feature interactions well.
  - **Multi-Layer Perceptron (MLP)**: Captures non-linear relationships between customer features and risk.

- **Real-world Relevance**:
  - Enables **credit risk scoring** for loan approvals, credit limit adjustments, or early collection interventions.
  - Supports **risk-based pricing**: adjusting interest rates based on predicted borrower risk.

The final product is wrapped in an interactive **Streamlit web app**, allowing easy single-customer scoring through CSV file uploads.

---

## Repository Structure

- `app.py` — Main Streamlit app script.
- `Credit_Score_Behaviour.ipynb` — Notebook for model development and training.
- `lightgbm_model.pkl` — Pre-trained LightGBM model.
- `mlp_model.pkl` — Pre-trained MLP model.
- `scaler.pkl` — Scaler object used for input normalization.
- `requirements.txt` — List of required packages.

---

## Getting Started

### Prerequisites

- Python 3.7 or above
- pip package manager

### Installation

1. **Clone the Repository**

```bash
git clone https://github.com/Saumi18/Credit-Risk-Modeling.git
cd Credit-Risk-Modeling
```

2. **(Optional) Create and Activate a Virtual Environment**

```bash
python -m venv venv
# For Windows
venv\Scripts\activate
# For Mac/Linux
source venv/bin/activate
```

3. **Install the Required Dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Streamlit App

After installing dependencies:

```bash
streamlit run app.py
```

Then open your browser at:

```
http://localhost:8501/
```

---

## How to Use the App

1. **Prepare a CSV File**  
   - Create a `.csv` file containing **exactly one row** representing the customer's features.
   - Ensure the **column names match** the expected input features used during model training.

   Example format:

   | feature1 | feature2 | feature3 | ... |
   |:--------:|:--------:|:--------:|:---:|
   | value1   | value2   | value3   | ... |

2. **Upload the CSV File**
   - In the Streamlit app, click **Browse files** and upload your customer's CSV file.

3. **Select the Model**
   - Choose either **LightGBM** or **MLP** from the model selection dropdown.

4. **Predict**
   - Click the **Predict** button to get the credit risk prediction.

---

## Technologies Used

- **Python** for scripting and modeling
- **LightGBM** and **MLP** for classification
- **scikit-learn** for preprocessing
- **Streamlit** for web app development
- **pandas**, **numpy**, **matplotlib**, **seaborn** for data analysis and visualization

---

## Contribution

Currently, external contributions are not open, but feedback and suggestions are welcome!

---
