import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# Load updated dataset
df = pd.read_csv("data/final_train_dataset.csv")

# -----------------------------
# 1. FEATURES AND TARGET
# -----------------------------
feature_cols = [
    "temperature",
    "humidity",
    "rainfall",
    "population_density",
    "cases_last_week",
    "social_sentiment",
    "resource_utilization",

    # Water Quality Features
    "ph_level",
    "turbidity",
    "contamination_index",
    "tds",
    "water_temperature"
]

X = df[feature_cols]
y = df["risk_level"]  # OUTPUT: Low, Medium, High

# Encode labels
label_map = {"Low": 0, "Medium": 1, "High": 2}
df["risk_label"] = df["risk_level"].map(label_map)
y = df["risk_label"]

# -----------------------------
# 2. TRAIN-TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, shuffle=True
)

# -----------------------------
# 3. SCALING
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 4. MODEL (Multinomial Logistic Regression)
# -----------------------------
model = LogisticRegression(
    max_iter=2000,
    multi_class="multinomial",
    solver="lbfgs"
)

model.fit(X_train_scaled, y_train)

# -----------------------------
# 5. EVALUATION
# -----------------------------
y_pred = model.predict(X_test_scaled)
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# 6. SAVE MODEL + SCALER
# -----------------------------
joblib.dump(model, "backend/models/model.pkl")
joblib.dump(scaler, "backend/models/scaler.pkl")
joblib.dump(label_map, "backend/models/label_map.pkl")

print("\n✅ Model, Scaler & Label Map Saved Successfully!")
