import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load and prepare data
@st.cache_data
def load_and_prepare_data(file_path):
    df = pd.read_csv(file_path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    df = df.drop(['CustomerID'], axis=1, errors='ignore')
    categorical_columns = df.select_dtypes(include=['object']).columns
    df = pd.get_dummies(df, columns=categorical_columns)
    return df

# Train the model
def train_churn_model(df):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return model, feature_importance, y_test, y_pred, scaler

# Visualization
def visualize_results(feature_importance, y_test, y_pred, df):
    st.subheader("Top 10 Feature Importances")
    total_importance = feature_importance['importance'].sum()
    feature_importance['percentage'] = (feature_importance['importance'] / total_importance) * 100
    fig1, ax1 = plt.subplots()
    sns.barplot(data=feature_importance.head(10), x='percentage', y='feature', palette='viridis', ax=ax1)
    ax1.set_title('Top 10 Features Impact (%)')
    st.pyplot(fig1)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig2, ax2 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=['Not Churned', 'Churned'],
                yticklabels=['Not Churned', 'Churned'], ax=ax2)
    ax2.set_title('Confusion Matrix')
    st.pyplot(fig2)

    st.subheader("Churn Distribution")
    test_dist = pd.Series(y_test).value_counts(normalize=True) * 100
    fig3, ax3 = plt.subplots()
    ax3.pie(test_dist,
            labels=[f'Not Churned\n({test_dist[0]:.1f}%)', 
                   f'Churned\n({test_dist[1]:.1f}%)'],
            colors=['lightblue', 'coral'],
            autopct='%1.1f%%',
            explode=(0, 0.1))
    ax3.set_title('Churn Distribution')
    st.pyplot(fig3)

    st.subheader("Monthly Charges by Churn Status")
    fig4, ax4 = plt.subplots()
    sns.boxplot(x=y_test, y=df['MonthlyCharges'], palette=['lightblue', 'coral'], ax=ax4)
    ax4.set_xlabel('Churn Status (0: Not Churned, 1: Churned)')
    ax4.set_ylabel('Monthly Charges ($)')
    ax4.set_title('Monthly Charges by Churn Status')
    st.pyplot(fig4)

# Streamlit UI
st.title("Customer Churn Prediction Dashboard")

uploaded_file = st.file_uploader("Upload your churn CSV file", type=["csv"])
if uploaded_file is not None:
    df = load_and_prepare_data(uploaded_file)
    st.success("Data Loaded and Prepared Successfully!")

    # Train model
    model, feature_importance, y_test, y_pred, scaler = train_churn_model(df)

    st.subheader("Model Performance")
    st.text(classification_report(y_test, y_pred))

    # Visualizations
    visualize_results(feature_importance, y_test, y_pred, df)
