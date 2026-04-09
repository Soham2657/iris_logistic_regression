# import libraries
import streamlit as st          # for UI
import numpy as np              # for arrays
import pickle                  # to load saved model

# load trained model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# app title (UI heading)
st.title("🌸 Iris Flower Prediction App")

# input fields (user will enter values)
sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

# button to trigger prediction
if st.button("Predict"):

    # convert inputs into array format (model expects this)
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # scale features (IMPORTANT: same scaling as training)
    features = scaler.transform(features)

    # make prediction
    prediction = model.predict(features)

    # show result
    if prediction[0] == 0:
        st.write("🌼 This is VERSICOLOR")
    else:
        st.write("🌺 This is VERGINICA")