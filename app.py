import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("logistic_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("🚢 Titanic Survival Prediction")

# Input fields
pclass = st.selectbox("Passenger Class", [1, 2, 3])
age = st.slider("Age", 0, 80, 30)
fare = st.slider("Fare", 0, 500, 50)
sex = st.selectbox("Sex", ['male', 'female'])
embarked = st.selectbox("Embarked", ['Q', 'S', 'C'])

# Convert categorical inputs into numeric (one-hot style)
ex_male = 1 if sex == 'male' else 0
embarked_Q = 1 if embarked == 'Q' else 0
embarked_S = 1 if embarked == 'S' else 0

# Arrange input data
input_data = np.array([[pclass, age, fare, ex_male, embarked_Q, embarked_S]])

# Apply scaling (important since model was trained with scaled data)
input_data_scaled = scaler.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)[0]

# Show result
st.subheader("🔍 Prediction Result:")
st.write("✅ Survived" if prediction == 1 else "❌ Did Not Survive")
