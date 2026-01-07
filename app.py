import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(page_title="Cross Validation App", layout="centered")

st.title("Cross Validation using Streamlit")
st.write("Upload a CSV file and perform K-Fold Cross Validation")

# -----------------------------------
# Upload CSV File
# -----------------------------------
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------------
    # Select numeric columns
    # -----------------------------------
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if numeric_df.shape[1] < 2:
        st.error("Dataset must contain at least one feature and one target column.")
    else:
        # Last column as target
        X = numeric_df.iloc[:, :-1]
        y = numeric_df.iloc[:, -1]

        st.subheader("Features Used")
        st.write(list(X.columns))

        st.subheader("Target Column")
        st.write(y.name)

        # -----------------------------------
        # Feature Scaling
        # -----------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # -----------------------------------
        # Select K value
        # -----------------------------------
        k = st.slider("Select number of folds (K)", min_value=2, max_value=10, value=5)

        # -----------------------------------
        # Model & Cross Validation
        # -----------------------------------
        model = LogisticRegression(max_iter=1000)

        kf = KFold(n_splits=k, shuffle=True, random_state=42)

        scores = cross_val_score(
            model,
            X_scaled,
            y,
            cv=kf,
            scoring="accuracy"
        )

        # -----------------------------------
        # Display Results
        # -----------------------------------
        st.subheader("Cross Validation Results")

        st.write("Accuracy for each fold:")
        st.write(scores)

        st.success(f"Mean Accuracy: {scores.mean():.4f}")
        st.info(f"Standard Deviation: {scores.std():.4f}")