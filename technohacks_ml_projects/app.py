import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import os
import io

st.sidebar.title("Titanic ML Dashboard")
page = st.sidebar.radio("Go to", ["Preprocessing", "EDA", "Linear Regression"])

if page == "Preprocessing":
    st.title("Titanic Data Preprocessing")
    uploaded_file = st.file_uploader("Upload Titanic CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Original Data Preview")
        st.dataframe(df.head())

        # Preprocess
        df['Age'] = df['Age'].fillna(df['Age'].median())
        if 'Cabin' in df.columns:
            df = df.drop('Cabin', axis=1)
        df = pd.get_dummies(df, columns=['Sex'], drop_first=True)
        scaler = MinMaxScaler()
        df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])

        st.subheader("Cleaned Data Preview")
        st.dataframe(df.head())
        st.subheader("Summary Statistics")
        st.write(df.describe())

        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Cleaned CSV", data=csv, file_name='cleaned_titanic.csv', mime='text/csv')

elif page == "EDA":
    st.title("Exploratory Data Analysis")
    uploaded_file = st.file_uploader("Upload Cleaned Titanic CSV", type=["csv"], key="eda")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Dataset Info")
        buffer = io.StringIO()
        df.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Age and Fare Distribution")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.histplot(df["Age"], kde=True, ax=axes[0])
        axes[0].set_title("Age Distribution")
        sns.histplot(df["Fare"], kde=True, ax=axes[1])
        axes[1].set_title("Fare Distribution")
        st.pyplot(fig)

        st.subheader("Survival by Sex and Pclass")
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sns.barplot(x="Sex_male", y="Survived", data=df, ax=axes[0])
        axes[0].set_title("Survival by Sex")
        sns.barplot(x="Pclass", y="Survived", data=df, ax=axes[1])
        axes[1].set_title("Survival by Pclass")
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

elif page == "Linear Regression":
    st.title("Salary Prediction via Linear Regression")
    uploaded_file = st.file_uploader("Upload Salary CSV", type=["csv"], key="lr")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        X = df[["YearsExperience"]]
        y = df["Salary"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        st.subheader("Model Performance")
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")

        st.subheader("Prediction Plot")
        fig, ax = plt.subplots(figsize=(10,6))
        ax.scatter(X_test, y_test, color="blue", label="Actual")
        ax.plot(X_test, y_pred, color="red", label="Predicted")
        ax.set_xlabel("YearsExperience")
        ax.set_ylabel("Salary")
        ax.legend()
        st.pyplot(fig)
