import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the trained model and the column names
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('model_columns.pkl', 'rb') as file:
    model_columns = pickle.load(file)

# Set page configuration for a professional look
st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

# Sidebar menu
st.sidebar.title("Navigation")
menu = ['Predict', 'Upload & Visualize']
choice = st.sidebar.radio('Choose an option:', menu)

# App header
st.title("Breast Cancer Detection System")
st.write("This application uses a machine learning model to predict whether a tumor is malignant or benign based on input data or uploaded CSV files.")

if choice == 'Predict':
    st.subheader('Predict Cancer Diagnosis')
    st.write("Use the sliders below to input the values for the prediction:")

    # Defining input sliders for model prediction
    input_data = {}
    slider_ranges = {
        'radius_mean': (0.0, 30.0),
        'texture_mean': (0.0, 40.0),
        'perimeter_mean': (0.0, 200.0),
        'area_mean': (0.0, 2500.0),
        'smoothness_mean': (0.0, 0.2),
        'compactness_mean': (0.0, 0.35),
        'concavity_mean': (0.0, 0.45),
        'concave points_mean': (0.0, 0.2),
        'symmetry_mean': (0.0, 0.3),
        'fractal_dimension_mean': (0.0, 0.1)
    }

    for col in model_columns:
        if col in slider_ranges:
            input_data[col] = st.slider(f'{col}', *slider_ranges[col], step=0.1)
        else:
            input_data[col] = st.slider(f'{col}', 0.0, 100.0, step=0.1)

    input_df = pd.DataFrame([input_data])

    # Predict button
    if st.button('Predict'):
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        prediction = model.predict(input_df)[0]
        result = 'Malignant' if prediction == 1 else 'Benign'
        st.success(f'The prediction is: **{result}**')

elif choice == 'Upload & Visualize':
    st.subheader('Upload and Visualize Your Data')
    st.write("Upload a CSV file and explore its contents with visualizations.")

    uploaded_file = st.file_uploader('Upload your CSV file', type=['csv'])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df = df.drop(columns=['Unnamed: 32'], errors='ignore')

        st.write("### Data Preview")
        st.dataframe(df.head())

        # Visualizations
        st.write("### Data Visualizations")

        # Add a slider to limit the data size
        sample_size = st.slider("Select sample size for visualizations (recommended <= 500)", min_value=100, max_value=len(df), value=300, step=50)

        if st.button('Show Correlation Heatmap'):
            st.write("#### Correlation Heatmap")
            df_sampled = df.sample(n=sample_size, random_state=42)
            df_numeric = df_sampled.select_dtypes(include=[np.number])

            if df_numeric.empty:
                st.error("No numeric data available for correlation.")
            else:
                # Create a figure for the heatmap
                fig, ax = plt.subplots(figsize=(8, 6))
                corr = df_numeric.corr()
                sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax)
                st.pyplot(fig)

        if st.button('Show Pair Plot'):
            st.write("#### Pair Plot")
            df_sampled = df.sample(n=sample_size, random_state=42)
            if len(df_sampled.columns) > 5:
                df_sampled = df_sampled[df_sampled.columns[:5]]  # Limit to the first 5 columns

            # Create a figure for the pair plot
            sns.pairplot(df_sampled)
            st.pyplot()

        # Clear memory only if variables are defined
        if 'df_sampled' in locals():
            del df_sampled
        if 'df_numeric' in locals():
            del df_numeric
