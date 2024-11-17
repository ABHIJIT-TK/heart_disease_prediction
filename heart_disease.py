import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load the dataset
df = pd.read_csv('./heart disease classification dataset.csv')

# Drop unnecessary column
df = df.drop(columns=['Unnamed: 0'])

# Map categorical values to numerical
df['target'] = df['target'].map({'yes': 1, 'no': 0})
df['sex'] = df['sex'].map({'male': 1, 'female': 0})

# Fill missing values with median
df['trestbps'].fillna(df['trestbps'].median(), inplace=True)
df['chol'].fillna(df['chol'].median(), inplace=True)
df['thalach'].fillna(df['thalach'].median(), inplace=True)

# Split features and target
x = df.drop(labels="target", axis=1)
y = df.target.to_numpy()

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.45, random_state=42)

# Logistic Regression model
model = LogisticRegression()
model.fit(x_train, y_train)

# Streamlit UI
st.title("Heart Disease Prediction")

# Collect user input
st.sidebar.header("Input Patient Data")
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.sidebar.selectbox("Sex", ["male", "female"])
cp = st.sidebar.slider("Chest Pain Type (cp)", min_value=0, max_value=3, value=0)
trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps)", value=120)
chol = st.sidebar.number_input("Serum Cholesterol (chol)", value=200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", [0, 1])
restecg = st.sidebar.slider("Resting Electrocardiographic Results (restecg)", min_value=0, max_value=2, value=0)
thalach = st.sidebar.number_input("Maximum Heart Rate Achieved (thalach)", value=150)
exang = st.sidebar.selectbox("Exercise Induced Angina (exang)", [0, 1])
oldpeak = st.sidebar.number_input("ST Depression Induced by Exercise (oldpeak)", value=0.0)
slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment (slope)", min_value=0, max_value=2, value=0)
ca = st.sidebar.slider("Number of Major Vessels (ca)", min_value=0, max_value=4, value=0)
thal = st.sidebar.slider("Thalassemia (thal)", min_value=0, max_value=3, value=0)

# Prediction function
def predict_disease(data):
    """
    Predict whether the person is diseased or not based on input data.
    """
    input_df = pd.DataFrame([data])
    input_df['sex'] = input_df['sex'].map({'male': 1, 'female': 0})
    input_df['trestbps'].fillna(df['trestbps'].median(), inplace=True)
    input_df['chol'].fillna(df['chol'].median(), inplace=True)
    input_df['thalach'].fillna(df['thalach'].median(), inplace=True)
    prediction = model.predict(input_df)
    return "Diseased" if prediction[0] == 1 else "Not Diseased"

# Prepare input data
input_data = {
    'age': age,
    'sex': sex,
    'cp': cp,
    'trestbps': trestbps,
    'chol': chol,
    'fbs': fbs,
    'restecg': restecg,
    'thalach': thalach,
    'exang': exang,
    'oldpeak': oldpeak,
    'slope': slope,
    'ca': ca,
    'thal': thal
}

# Show prediction
if st.button("Predict"):
    result = predict_disease(input_data)
    st.subheader("Prediction:")
    st.write(f"The patient is **{result}**.")
