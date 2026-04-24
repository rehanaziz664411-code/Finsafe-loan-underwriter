import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# --- 1. LOAD DATA ---
print("📂 Loading Loan Dataset (4,300 rows)...")
df = pd.read_csv('loan_approval.csv')
df.columns = df.columns.str.strip()

# --- 2. DATA CLEANING ---
df = df.dropna()

# --- 3. ENCODING ---
le = LabelEncoder()
cat_cols = df.select_dtypes(include=['object']).columns
label_encoders = {}

for col in cat_cols:
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# --- 4. FEATURE SELECTION ---
# Drop 'loan_id' because it is just a random number/string and doesn't help predict approval
if 'loan_id' in df.columns:
    df = df.drop('loan_id', axis=1)

X = df.drop('loan_status', axis=1)
y = df['loan_status']

# --- 5. SPLIT & SCALE ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 6. TRAINING (Random Forest - Optimized for Small Data) ---
print("🚀 Training Robust Random Forest Engine...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,              # Prevent overfitting on small data
    min_samples_leaf=5,       # Ensure each "leaf" has enough data
    class_weight='balanced',   # Handle potential class imbalance
    random_state=42
)

model.fit(X_train_scaled, y_train)

# --- 7. EVALUATION ---
# Using Cross-Validation to prove the 4k rows are enough
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
preds = model.predict(X_test_scaled)

print("\n" + "="*30)
print("📊 BANKING AUDIT REPORT")
print("="*30)
print(f"Cross-Val Mean Accuracy: {cv_scores.mean():.2%}")
print(f"Test Set Accuracy: {accuracy_score(y_test, preds):.2%}")
print("\nClassification Details:")
print(classification_report(y_test, preds))

# --- 8. SAVE ARTIFACTS ---
with open('loan_model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('loan_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('loan_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("\n✅ Success! Model saved for Pakistani FinTech Dashboard.")