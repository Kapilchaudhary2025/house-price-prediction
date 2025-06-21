import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Title
st.title("🏡 House Price Prediction App")
st.write(
    "A simple machine learning app to predict house prices using Linear Regression. "
    "Upload your data or use default Kaggle housing data. [Project by Kapil]"
)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("data/train.csv")
    # Drop unnecessary columns & handle missing values
    df = df.drop(['Alley', 'PoolQC', 'Fence', 'MiscFeature', 'Id'], axis=1, errors="ignore")
    for col in df.select_dtypes(include=np.number).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    df = pd.get_dummies(df, drop_first=True)
    return df

# Data
df = load_data()

# Train/test split
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Sidebar - User Input
st.sidebar.header("Enter house features for prediction:")

# For demo: Take first 10 features only (you can expand as needed)
input_features = {}
feature_names = X.columns[:10]  # Select first 10 for simplicity
for feature in feature_names:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    input_features[feature] = st.sidebar.slider(
        label=feature, min_value=min_val, max_value=max_val, value=mean_val
    )

# Convert input to DataFrame
input_df = pd.DataFrame([input_features])

# Add rest of columns as zeros (for dummy variables and unused columns)
for col in X.columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[X.columns]  # Ensure correct order

# Prediction
if st.sidebar.button("Predict House Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")

# Show some sample data
with st.expander("See sample training data"):
    st.write(df.head())