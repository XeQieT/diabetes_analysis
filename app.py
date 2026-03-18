import streamlit as st

st.set_page_config(page_title="Diabetes Health Dashboard", layout="wide")

st.title("Diabetes Risk Factors Analysis")
st.markdown("""
Welcome to the dashboard built on the **Diabetes Health Indicators** dataset (CDC BRFSS 2015).  
Here you can explore how various lifestyle and health factors are associated with diabetes.

Use the sidebar menu to navigate between pages:
- **EDA** – initial data exploration and target variable distribution
- **Factor Visualization** – detailed analysis of individual risk factors
- **Risk Calculator** – predict diabetes probability based on your inputs
""")
