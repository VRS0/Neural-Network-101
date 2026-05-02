import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from architecture import nn

st.title("Iris Flower Classifier")
st.write("Neural Network Based ~ scratch.")

@st.cache_resource
def train_model():
    iris = load_iris()
    X, y = iris.data, pd.get_dummies(iris.target).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = nn(learning_rate=0.1)
    errors = model.training(X_train, y_train, 5000)

    return model, scaler, errors, X_train, X_test, y_train, y_test

model, scaler, errors, X_train, X_test, y_train, y_test = train_model()

# Metrics
col1, col2 = st.columns(2)
col1.metric("Train Accuracy", f"{model.accuracy(X_train, y_train):.1f}%")
col2.metric("Test Accuracy",  f"{model.accuracy(X_test,  y_test):.1f}%")

# Error curve 
st.subheader("Training Error")
st.line_chart(errors)

# Live prediction 
st.subheader("Enter a Flower's Measurements")

col1, col2, col3, col4 = st.columns(4)
sepal_len = col1.number_input("Sepal Length", 4.0, 8.0, 5.1)
sepal_wid = col2.number_input("Sepal Width",  1.5, 5.0, 3.5)
petal_len = col3.number_input("Petal Length", 1.0, 7.0, 1.4)
petal_wid = col4.number_input("Petal Width",  0.1, 3.0, 0.2)

if st.button("Predict"):
    species   = ["Setosa", "Versicolor", "Virginica"]
    sample    = scaler.transform([[sepal_len, sepal_wid, petal_len, petal_wid]])
    probs     = model.forward_pass(sample)[0]
    predicted = species[np.argmax(probs)]

    st.success(f"Predicted: **{predicted}**")
    st.bar_chart(dict(zip(species, probs)))