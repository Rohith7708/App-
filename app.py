import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
App title

st.set_page_config(page_title="Customer Churn Prediction Dashboard", layout="wide") st.title("📉 Customer Churn Prediction with Random Forest")

Load and prepare the data

def load_and_prepare_data(df): df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce') df['Churn'] = (df['Churn'] == 'Yes').astype(int) df = df.drop(['CustomerID'], axis=1, errors='ignore') df = pd.get_dummies(df, columns=df.select_dtypes(include=['object']).columns) return df

Train the model

def train_churn_model(df): X = df.drop('Churn', axis=1) y = df['Churn'] X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) scaler = StandardScaler() X_train_scaled = scaler.fit_transform(X_train) X_test_scaled = scaler.transform(X_test) model = RandomForestClassifier(n_estimators=100, random_state=42) model.fit(X_train_scaled, y_train) y_pred = model.predict(X_test_scaled) feature_importance = pd.DataFrame({ 'feature': X.columns, 'importance': model.feature_importances_ }).sort_values('importance', ascending=False) return model, feature_importance, y_test, y_pred, X_test, scaler

Visualization

def visualize_results(feature_importance, y_test, y_pred, original_df): fig, axs = plt.subplots(2, 2, figsize=(20, 15))

# Feature Importance
total_importance = feature_importance['importance'].sum()
feature_importance['percentage'] = (feature_importance['importance'] / total_importance) * 100
sns.barplot(data=feature_importance.head(10),
            x='percentage', y='feature',
            palette='viridis', ax=axs[0, 0])
axs[0, 0].set_title('Top 10 Features Impact (%)')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
            xticklabels=['Not Churned', 'Churned'],
            yticklabels=['Not Churned', 'Churned'], ax=axs[0, 1])
axs[0, 1].set_title('Confusion Matrix')

# Churn Distribution
test_dist = pd.Series(y_test).value_counts(normalize=True) * 100
axs[1, 0].pie(test_dist,
              labels=[f'Not Churned\n({test_dist[0]:.1f}%)', f'Churned\n({test_dist[1]:.1f}%)'],
              colors=['lightblue', 'coral'],
              autopct='%1.1f%%', explode=(0, 0.1))
axs[1, 0].set_title('Churn Distribution')

# Monthly Charges Boxplot
df_subset = original_df.iloc[y_test.index]
sns.boxplot(x=y_test, y=df_subset['MonthlyCharges'],
            palette=['lightblue', 'coral'], ax=axs[1, 1])
axs[1, 1].set_title('Monthly Charges by Churn Status')
axs[1, 1].set_xlabel('Churn Status (0: Not Churned, 1: Churned)')
axs[1, 1].set_ylabel('Monthly Charges')

plt.tight_layout()
st.pyplot(fig)

Load dataset directly

try: df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv') st.success("Data successfully loaded from file.") df_prepared = load_and_prepare_data(df.copy())

# Train the model
model, feature_importance, y_test, y_pred, X_test, scaler = train_churn_model(df_prepared)

# Display model performance
st.subheader("📋 Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# Visualization
st.subheader("📊 Churn Insights Visualization")
visualize_results(feature_importance, y_test, y_pred, df)

except Exception as e: st.error(f"Error loading data: {e}")

