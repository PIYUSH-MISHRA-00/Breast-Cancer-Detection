![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
# Breast Cancer Detection App

This is a simple web application built using Streamlit that predicts the likelihood of breast cancer based on several medical features. The prediction model is based on a Logistic Regression model trained with the Breast Cancer Wisconsin dataset.

## Features

- **User Input**: Users can input medical features such as mean radius, mean texture, mean perimeter, and more through an intuitive sidebar.
- **Prediction**: The application predicts whether a given input corresponds to a benign or malignant tumor.
- **Prediction Probability**: The application also displays the probability associated with each prediction.

## How to Run the Application

### Prerequisites

Before running the application, ensure you have the following installed:

- **Docker**: If you don't have Docker installed, you can download it from [here](https://www.docker.com/products/docker-desktop).

### Running the Application with Docker

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/breast-cancer-prediction.git
    cd breast-cancer-prediction
    ```

2. **Build the Docker image**:
    ```bash
    docker build -t breast-cancer-prediction-app .
    ```

3. **Run the Docker container**:
    ```bash
    docker run -p 8501:8501 breast-cancer-prediction-app
    ```

4. **Access the application**:
    Open your web browser and go to `http://localhost:8501` to access the Breast Cancer Prediction App.

### Running the Application Locally

If you prefer to run the application directly on your machine without Docker:

1. **Create a virtual environment** (optional but recommended):
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the application**:
    ```bash
    streamlit run app.py
    ```

4. **Access the application**:
    Open your web browser and go to `http://localhost:8501`.

## Project Structure

```
.
├── Dockerfile
├── README.md
├── app.py
├── breast_cancer_logistic_model.pkl
├── requirements.txt
└── .gitignore
```

Dockerfile: Docker configuration file to containerize the application.
app.py: The main application file that runs the Streamlit app.
breast_cancer_logistic_model.pkl: Pre-trained logistic regression model used for prediction.
requirements.txt: List of Python dependencies required for the project.
.gitignore: Specifies files and directories that Git should ignore.

## About the Dataset

The model was trained using the Breast Cancer Wisconsin dataset, which contains features that describe the characteristics of cell nuclei present in a breast mass.

Dataset Link: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data