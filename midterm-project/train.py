#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

print("=" * 80)
print("TRAINING PREDICTIVE MAINTENANCE MODEL")
print("=" * 80)

# Load data
print("\n1. Loading data...")
df = pd.read_csv('data/ai4i2020.csv')
print(f"   Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Data preparation
print("\n2. Preparing data...")
df_model = df.copy()

# Drop unnecessary columns
columns_to_drop = ['UDI', 'Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']
df_model = df_model.drop(columns=columns_to_drop)

# Rename columns (remove brackets for XGBoost)
df_model = df_model.rename(columns={
    'Air temperature [K]': 'Air_temperature_K',
    'Process temperature [K]': 'Process_temperature_K',
    'Rotational speed [rpm]': 'Rotational_speed_rpm',
    'Torque [Nm]': 'Torque_Nm',
    'Tool wear [min]': 'Tool_wear_min'
})

# Encode Type
le = LabelEncoder()
df_model['Type_encoded'] = le.fit_transform(df_model['Type'])
df_model = df_model.drop('Type', axis=1)

# Feature engineering
df_model['Temp_diff'] = df_model['Process_temperature_K'] - df_model['Air_temperature_K']
df_model['Power_factor'] = df_model['Torque_Nm'] * df_model['Rotational_speed_rpm']

print("   Features prepared and engineered")

# Split data
X = df_model.drop('Machine failure', axis=1)
y = df_model['Machine failure']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Train set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("   Features scaled")

# Train model
print("\n3. Training XGBoost model...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("   Model trained successfully")

# Evaluate
print("\n4. Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"   Accuracy:  {accuracy:.4f}")
print(f"   Precision: {precision:.4f}")
print(f"   Recall:    {recall:.4f}")
print(f"   F1-Score:  {f1:.4f}")
print(f"   ROC-AUC:   {roc_auc:.4f}")

# Save models
print("\n5. Saving model and preprocessing objects...")
with open('models/model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("   ✓ Model saved: models/model.pkl")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("   ✓ Scaler saved: models/scaler.pkl")

with open('models/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)
print("   ✓ Label encoder saved: models/label_encoder.pkl")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE!")
print("=" * 80)
