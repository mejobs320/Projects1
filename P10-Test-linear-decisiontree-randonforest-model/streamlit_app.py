"""
Property Price Predictor — Streamlit Web UI
-----------------------------------------------------------
Only 3 inputs are collected from the user: size, age, bedrooms.
Everything else about the property — bathrooms, garage spaces,
location, and price — is predicted by trained models.

Three model families are trained and compared side by side:
    - Linear / Logistic Regression
    - Decision Tree
    - Random Forest

Two sections:
    1. Model performance — compare accuracy of all 3 models on test data
    2. Predict — pick a model, enter size/age/bedrooms, get every other
       property attribute predicted and shown

Run in VS Code:
    1. Install dependencies:  pip install -r requirements.txt
    2. Launch the app:        streamlit run streamlit_app.py
    3. It opens automatically at http://localhost:8501
"""

import pandas as pd
import numpy as np
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
)

# ---------------------------------------------------------
# Page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Property Price Predictor",
    page_icon="🏠",
    layout="wide",
)

DATA_PATH = "property_prices_nonlinear.csv"

# What the user provides
INPUT_FEATURES = ["size_sqft", "age_years", "bedrooms"]

# What the models predict from those inputs
NUMERIC_TARGETS = ["bathrooms", "garage_spaces", "price"]
CATEGORICAL_TARGET = "location"

RANDOM_STATE = 42

# The three model families to train and compare
MODEL_NAMES = ["Linear Regression", "Decision Tree", "Random Forest"]


# ---------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------
# Cached model training
#   Trains all 3 model families:
#     - Linear/Logistic Regression
#     - Decision Tree
#     - Random Forest
#   Regressors are multi-output (predict bathrooms, garage_spaces,
#   price together). Classifiers predict location.
# ---------------------------------------------------------
@st.cache_resource
def train_models(df: pd.DataFrame):
    X = df[INPUT_FEATURES]
    y_numeric = df[NUMERIC_TARGETS]
    y_location = df[CATEGORICAL_TARGET]

    X_train, X_test, ynum_train, ynum_test, yloc_train, yloc_test = train_test_split(
        X, y_numeric, y_location, test_size=0.2, random_state=RANDOM_STATE
    )

    regressors = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    classifiers = {
        "Linear Regression": Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
    }

    numeric_metrics = {}   # {model_name: {target: {mae, rmse, r2}}}
    location_accuracy = {}  # {model_name: accuracy}

    for name, reg in regressors.items():
        reg.fit(X_train, ynum_train)
        pred = pd.DataFrame(
            reg.predict(X_test), columns=NUMERIC_TARGETS, index=X_test.index
        )
        numeric_metrics[name] = {
            col: {
                "mae": mean_absolute_error(ynum_test[col], pred[col]),
                "rmse": np.sqrt(mean_squared_error(ynum_test[col], pred[col])),
                "r2": r2_score(ynum_test[col], pred[col]),
            }
            for col in NUMERIC_TARGETS
        }

    for name, clf in classifiers.items():
        clf.fit(X_train, yloc_train)
        pred = clf.predict(X_test)
        location_accuracy[name] = accuracy_score(yloc_test, pred)

    return regressors, classifiers, numeric_metrics, location_accuracy


# ---------------------------------------------------------
# Load data + train models
# ---------------------------------------------------------
try:
    df = load_data(DATA_PATH)
except FileNotFoundError:
    st.error(f"Couldn't find `{DATA_PATH}`. Make sure it's in the same folder as this script.")
    st.stop()

regressors, classifiers, numeric_metrics, location_accuracy = train_models(df)

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.title("🏠 Property Price Predictor")
st.caption(
    "Give the model a property's size, age, and bedroom count — "
    "it predicts bathrooms, garage spaces, location, and price. "
    "Compare Linear Regression, Decision Tree, and Random Forest below."
)

# ---------------------------------------------------------
# Section 1 — Model performance comparison
# ---------------------------------------------------------
st.header("📈 Model performance comparison")
st.caption("Measured on a held-out 20% test set none of the models saw during training.")

for target in NUMERIC_TARGETS:
    st.subheader(target.replace("_", " ").title())
    comp_df = pd.DataFrame({
        name: {
            "MAE": numeric_metrics[name][target]["mae"],
            "RMSE": numeric_metrics[name][target]["rmse"],
            "R²": numeric_metrics[name][target]["r2"],
        }
        for name in MODEL_NAMES
    }).T
    m_cols = st.columns(len(MODEL_NAMES))
    best_r2 = comp_df["R²"].idxmax()
    for i, name in enumerate(MODEL_NAMES):
        with m_cols[i]:
            label = f"{name} 🏆" if name == best_r2 else name
            st.markdown(f"**{label}**")
            st.metric("MAE", f"{comp_df.loc[name, 'MAE']:,.2f}")
            st.metric("RMSE", f"{comp_df.loc[name, 'RMSE']:,.2f}")
            st.metric("R²", f"{comp_df.loc[name, 'R²']:.3f}")

st.subheader("Location (classification)")
loc_cols = st.columns(len(MODEL_NAMES))
best_acc_model = max(location_accuracy, key=location_accuracy.get)
for i, name in enumerate(MODEL_NAMES):
    with loc_cols[i]:
        label = f"{name} 🏆" if name == best_acc_model else name
        st.markdown(f"**{label}**")
        st.metric("Accuracy", f"{location_accuracy[name] * 100:.1f}%")

st.divider()

# ---------------------------------------------------------
# Section 2 — Predict everything else
# ---------------------------------------------------------
st.header("🔮 Predict a property")
st.caption("Choose a model, enter what you know, and let it fill in the rest.")

model_choice = st.selectbox("Model", MODEL_NAMES, index=2)

in_col1, in_col2, in_col3, in_col4 = st.columns([1, 1, 1, 1])

with in_col1:
    size_sqft = st.slider("Size (sqft)", 400, 5000, 2000, step=50)
with in_col2:
    age_years = st.slider("Age (years)", 0, 60, 10)
with in_col3:
    bedrooms = st.slider("Bedrooms", 1, 6, 3)
with in_col4:
    st.write("")
    st.write("")
    predict_clicked = st.button("Predict", type="primary", use_container_width=True)

if predict_clicked:
    new_input = pd.DataFrame([{
        "size_sqft": size_sqft,
        "age_years": age_years,
        "bedrooms": bedrooms,
    }])

    regressor = regressors[model_choice]
    classifier = classifiers[model_choice]

    numeric_pred = regressor.predict(new_input)[0]
    bathrooms_pred, garage_pred, price_pred = numeric_pred
    location_pred = classifier.predict(new_input)[0]

    st.subheader(f"Predicted property details — {model_choice}")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    r_col1.metric("Bathrooms", f"{bathrooms_pred:.1f}")
    r_col2.metric("Garage spaces", f"{garage_pred:.1f}")
    r_col3.metric("Location", location_pred)
    r_col4.metric("Price", f"${price_pred:,.0f}")
