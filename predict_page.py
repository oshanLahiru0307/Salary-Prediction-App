from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pickle
import numpy as np

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

# Load saved model and encoders
data = load_model()
regressor_loaded = data['model']
le_country = data['le_country']
le_education = data['le_education']

def show_predict_page():
    
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = ( 
        "United States", "India", "United Kingdom", "Germany", "Canada",
        "Brazil", "France", "Spain", "Australia", "Netherlands",
        "Poland", "Italy", "Russian Federation", "Sweden"
    )

    education_levels = (
        "Bachelor’s degree",
        "Master’s degree",
        "Less than a Bachelors",
        "Post grad"
    )
    
    country = st.selectbox("Countries", countries)
    education = st.selectbox("Education Level", education_levels)
    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")

    if ok:
        try:
            # Convert input to numerical values using label encoders
            X = np.array([[country, education, experience]], dtype=object)

            X[:, 0] = le_country.transform(X[:, 0])  # Encode country
            X[:, 1] = le_education.transform(X[:, 1])  # Encode education

            X = X.astype(float)  # Ensure float dtype for prediction

            # Make prediction using the loaded model
            salary = regressor_loaded.predict(X)

            st.subheader(f"The Estimated Annual Salary is ${salary[0]:,.2f}")

        except Exception as e:
            st.error(f"Error: {e}")
