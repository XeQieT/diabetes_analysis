import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Exploratory Data Analysis (EDA)")

@st.cache_data
def load_data():
    df = pd.read_csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    return df

df = load_data()

st.subheader("First 10 rows of the dataset")
st.write(df.head(10))

st.subheader("Dataset info")
st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
st.write("**Data types:**")
st.write(df.dtypes)

st.subheader("Descriptive statistics")
st.write(df.describe())

st.subheader("Target variable distribution (Diabetes_binary)")
fig, ax = plt.subplots()
sns.countplot(data=df, x='Diabetes_binary', ax=ax)
ax.set_xticklabels(['No diabetes', 'Diabetes'])
ax.set_xlabel("Diabetes")
ax.set_ylabel("Count")
st.pyplot(fig)

counts = df['Diabetes_binary'].value_counts()
st.write(f"No diabetes: {counts[0]} records")
st.write(f"Diabetes: {counts[1]} records")
st.write("The dataset is balanced (50/50), which is convenient for machine learning.")
