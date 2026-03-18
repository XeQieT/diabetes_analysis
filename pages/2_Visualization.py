import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("Risk Factor Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("data/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    return df

df = load_data()

features = [col for col in df.columns if col != 'Diabetes_binary']

selected_feature = st.selectbox("Choose a factor to analyze", features)

st.subheader(f"Distribution of '{selected_feature}' by diabetes status")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Diabetes == 0
df_healthy = df[df['Diabetes_binary'] == 0]
sns.histplot(df_healthy[selected_feature], kde=True, ax=axes[0], color='green')
axes[0].set_title("No diabetes")

# Diabetes == 1
df_diabetes = df[df['Diabetes_binary'] == 1]
sns.histplot(df_diabetes[selected_feature], kde=True, ax=axes[1], color='red')
axes[1].set_title("Diabetes")

st.pyplot(fig)

st.subheader("Average value of the factor by group")
grouped = df.groupby('Diabetes_binary')[selected_feature].mean()
st.write(grouped.rename(index={0: "No diabetes", 1: "Diabetes"}))

st.subheader("Correlation matrix of all features")
corr = df.corr()
fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='coolwarm', ax=ax_corr)
st.pyplot(fig_corr)

st.subheader("Correlation with target (Diabetes_binary)")
corr_with_target = corr['Diabetes_binary'].sort_values(ascending=False)
st.write(corr_with_target)
