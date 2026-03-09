import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# --- 1. CONFIGURATION & DATA LOADING ---
# Ensure this path is correct for your local machine
path = r"C:\Users\sthut\Downloads\archive (2)\customer_churn_data.csv"

try:
    df = pd.read_csv(path)
    print("✅ Data loaded successfully!")
except FileNotFoundError:
    print("❌ Error: file not found. Check your file path.")
    exit()

# --- 2. DATA CLEANING & EXPLORATION ---
print("\n--- Dataset Overview ---")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Handle missing values
df['InternetService'] = df['InternetService'].fillna('None')

# Quick Visualization: Churn by Contract Type
sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='ContractType', hue='Churn', palette='viridis')
plt.title('Churn Rate by Contract Type')
plt.show() 

# --- 3. PREPROCESSING ---
# Drop non-predictive columns
df_ml = df.drop('CustomerID', axis=1)

# Categorical Encoding
categorical_cols = ['Gender', 'ContractType', 'InternetService', 'TechSupport']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Target Encoding (Map 'Yes' to 1 and 'No' to 0)
if df_ml['Churn'].dtype == 'object':
    df_ml['Churn'] = df_ml['Churn'].map({'Yes': 1, 'No': 0})

# --- 4. MODEL TRAINING ---
X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 5. EVALUATION & VISUALIZATION ---
y_pred = model.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Feature Importance Plot
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances, y=importances.index, hue=importances.index, legend=False)
plt.title('Feature Importances')
plt.show()

# --- 6. SAVE MODEL ---
joblib.dump(model, 'churn_model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')
print("\n✅ Model and Encoders saved as .pkl files in the current directory!")