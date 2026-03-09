import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- 1. DATA LOADING ---
path = r"C:\Users\sthut\Downloads\archive (2)\customer_churn_data.csv"

if os.path.exists(path):
    df = pd.read_csv(path)
    print("✅ Data loaded successfully!")
else:
    print(f"❌ ERROR: File not found at {path}")
    exit()

# --- 2. PREPROCESSING ---
# Fill missing InternetService values
df['InternetService'] = df['InternetService'].fillna('None')

# Drop CustomerID as it has no predictive power
df_ml = df.drop('CustomerID', axis=1)

# Encode categorical text into numbers
categorical_cols = ['Gender', 'ContractType', 'InternetService', 'TechSupport']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le

# Convert Churn text (Yes/No) to numbers (1/0)
if df_ml['Churn'].dtype == 'object':
    df_ml['Churn'] = df_ml['Churn'].map({'Yes': 1, 'No': 0})

# --- 3. MODEL TRAINING ---
X = df_ml.drop('Churn', axis=1)
y = df_ml['Churn']

# Split data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Training the Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- 4. SAVING ASSETS ---
joblib.dump(model, 'churn_model.pkl')
joblib.dump(label_encoders, 'encoders.pkl')

print("\n✅ SUCCESS!")
print("- Created 'churn_model.pkl'")
print("- Created 'encoders.pkl'")