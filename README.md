Diabetes Health Dashboard

A simple web app to explore diabetes risk factors and predict diabetes probability using CDC survey data.
Features

    View dataset summary and statistics

    Compare distributions of health indicators (BMI, blood pressure, smoking, etc.) between diabetic and non-diabetic groups

    Correlation heatmap

    Predict diabetes risk based on user input

Installation

    Clone this repository:
    bash

    git clone https://github.com/XeQieT/diabetes_analysis.git
    cd diabetes_analysis

    Install dependencies:
    bash

    pip install -r requirements.txt

    Download the dataset from Kaggle(https://www.kaggle.com/datasets/alexteboul/diabetes-health-indicators-dataset) and place diabetes_binary_5050split_health_indicators_BRFSS2015.csv inside a folder named data.

    Train the model:
    bash

    python train_model.py

    Run the app:
    bash

    streamlit run app.py

Project Structure

    app.py – main entry point

    pages/ – three analysis pages

    train_model.py – trains and saves the model

    model.pkl – trained model

    requirements.txt – Python dependencies

Technologies

Python, pandas, matplotlib, seaborn, scikit-learn, Streamlit.
Author

XeQieT

For educational purposes only. Not a real diagnostic tool.
