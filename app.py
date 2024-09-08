import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load your trained model
model = pickle.load(open('breast_cancer_logistic_model.pkl', 'rb'))

# Define the app title and description
st.title('Breast Cancer Prediction')
st.write('This app predicts the likelihood of breast cancer based on medical features.')

# Sidebar for user input
st.sidebar.header('User Input Features')

def user_input_features():
    mean_radius = st.sidebar.slider('Mean Radius', 6.0, 30.0, 14.0)
    mean_texture = st.sidebar.slider('Mean Texture', 9.0, 40.0, 20.0)
    mean_perimeter = st.sidebar.slider('Mean Perimeter', 40.0, 190.0, 90.0)
    mean_area = st.sidebar.slider('Mean Area', 150.0, 2500.0, 500.0)
    mean_smoothness = st.sidebar.slider('Mean Smoothness', 0.05, 0.2, 0.1)
    mean_compactness = st.sidebar.slider('Mean Compactness', 0.01, 0.35, 0.1)
    mean_concavity = st.sidebar.slider('Mean Concavity', 0.0, 0.45, 0.1)
    mean_concave_points = st.sidebar.slider('Mean Concave Points', 0.0, 0.2, 0.1)
    mean_symmetry = st.sidebar.slider('Mean Symmetry', 0.1, 0.3, 0.2)
    mean_fractal_dimension = st.sidebar.slider('Mean Fractal Dimension', 0.04, 0.1, 0.06)

    # Additional sliders for the remaining features
    radius_error = st.sidebar.slider('Radius Error', 0.1, 3.0, 0.5)
    texture_error = st.sidebar.slider('Texture Error', 0.1, 4.0, 1.0)
    perimeter_error = st.sidebar.slider('Perimeter Error', 1.0, 25.0, 5.0)
    area_error = st.sidebar.slider('Area Error', 6.0, 250.0, 30.0)
    smoothness_error = st.sidebar.slider('Smoothness Error', 0.001, 0.03, 0.007)
    compactness_error = st.sidebar.slider('Compactness Error', 0.002, 0.1, 0.02)
    concavity_error = st.sidebar.slider('Concavity Error', 0.001, 0.4, 0.02)
    concave_points_error = st.sidebar.slider('Concave Points Error', 0.001, 0.05, 0.01)
    symmetry_error = st.sidebar.slider('Symmetry Error', 0.007, 0.08, 0.02)
    fractal_dimension_error = st.sidebar.slider('Fractal Dimension Error', 0.001, 0.03, 0.003)

    worst_radius = st.sidebar.slider('Worst Radius', 7.0, 40.0, 16.0)
    worst_texture = st.sidebar.slider('Worst Texture', 12.0, 50.0, 25.0)
    worst_perimeter = st.sidebar.slider('Worst Perimeter', 50.0, 250.0, 110.0)
    worst_area = st.sidebar.slider('Worst Area', 185.0, 4000.0, 880.0)
    worst_smoothness = st.sidebar.slider('Worst Smoothness', 0.07, 0.25, 0.15)
    worst_compactness = st.sidebar.slider('Worst Compactness', 0.027, 1.5, 0.5)
    worst_concavity = st.sidebar.slider('Worst Concavity', 0.0, 1.25, 0.25)
    worst_concave_points = st.sidebar.slider('Worst Concave Points', 0.0, 0.35, 0.2)
    worst_symmetry = st.sidebar.slider('Worst Symmetry', 0.15, 0.7, 0.3)
    worst_fractal_dimension = st.sidebar.slider('Worst Fractal Dimension', 0.05, 0.25, 0.1)

    data = {
        'mean_radius': mean_radius,
        'mean_texture': mean_texture,
        'mean_perimeter': mean_perimeter,
        'mean_area': mean_area,
        'mean_smoothness': mean_smoothness,
        'mean_compactness': mean_compactness,
        'mean_concavity': mean_concavity,
        'mean_concave_points': mean_concave_points,
        'mean_symmetry': mean_symmetry,
        'mean_fractal_dimension': mean_fractal_dimension,
        'radius_error': radius_error,
        'texture_error': texture_error,
        'perimeter_error': perimeter_error,
        'area_error': area_error,
        'smoothness_error': smoothness_error,
        'compactness_error': compactness_error,
        'concavity_error': concavity_error,
        'concave_points_error': concave_points_error,
        'symmetry_error': symmetry_error,
        'fractal_dimension_error': fractal_dimension_error,
        'worst_radius': worst_radius,
        'worst_texture': worst_texture,
        'worst_perimeter': worst_perimeter,
        'worst_area': worst_area,
        'worst_smoothness': worst_smoothness,
        'worst_compactness': worst_compactness,
        'worst_concavity': worst_concavity,
        'worst_concave_points': worst_concave_points,
        'worst_symmetry': worst_symmetry,
        'worst_fractal_dimension': worst_fractal_dimension
    }
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

# No scaling, using the raw input features directly
df_scaled = df

# Display user input features
st.subheader('User Input features')
st.write(df)

# Prediction
prediction = model.predict(df_scaled)
prediction_proba = model.predict_proba(df_scaled)

# Display the prediction and prediction probability
st.subheader('Prediction')
cancer_diagnosis = np.array(['Malignant', 'Benign'])
st.write(cancer_diagnosis[prediction[0]])

st.subheader('Prediction Probability')
st.write(prediction_proba)
