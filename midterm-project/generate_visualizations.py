"""
Generate visualizations from the notebook and save them as PNG files
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)
plt.rcParams['font.size'] = 10

# Load data
print("Loading data...")
df = pd.read_csv('data/ai4i2020.csv')

# ============================================================================
# 1. TARGET DISTRIBUTION
# ============================================================================
print("Generating: Target Distribution...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Count plot
df['Machine failure'].value_counts().plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
axes[0].set_title('Machine Failure Distribution', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Machine Failure (0=No, 1=Yes)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['No Failure', 'Failure'], rotation=0)

# Pie chart
df['Machine failure'].value_counts().plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                                          colors=['#2ecc71', '#e74c3c'], startangle=90)
axes[1].set_title('Machine Failure Proportion', fontsize=14, fontweight='bold')
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('images/01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/01_target_distribution.png")

# ============================================================================
# 2. FAILURE TYPES DISTRIBUTION
# ============================================================================
print("Generating: Failure Types Distribution...")
failure_types = ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
failure_counts = df[failure_types].sum()

fig, ax = plt.subplots(figsize=(10, 6))
failure_counts.plot(kind='bar', ax=ax, color='#3498db')
ax.set_title('Failure Types Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Failure Type')
ax.set_ylabel('Number of Failures')
ax.set_xticklabels(failure_counts.index, rotation=45)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/02_failure_types.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/02_failure_types.png")

# ============================================================================
# 3. FEATURE CORRELATION WITH TARGET
# ============================================================================
print("Generating: Feature Correlation with Target...")
numerical_features = ['Air temperature [K]', 'Process temperature [K]', 
                     'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

target_corr = df[numerical_features + ['Machine failure']].corr()['Machine failure'].drop('Machine failure')
target_corr = target_corr.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
target_corr.plot(kind='barh', ax=ax, color=['#e74c3c' if x < 0 else '#2ecc71' for x in target_corr.values])
ax.set_title('Feature Correlation with Machine Failure', fontsize=14, fontweight='bold')
ax.set_xlabel('Correlation Coefficient')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('images/03_feature_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/03_feature_correlation.png")

# ============================================================================
# 4. TEMPERATURE ANALYSIS
# ============================================================================
print("Generating: Temperature Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Air temperature
df.boxplot(column='Air temperature [K]', by='Machine failure', ax=axes[0])
axes[0].set_title('Air Temperature by Failure Status', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Machine Failure (0=No, 1=Yes)')
axes[0].set_ylabel('Air Temperature [K]')
axes[0].get_figure().suptitle('')

# Process temperature
df.boxplot(column='Process temperature [K]', by='Machine failure', ax=axes[1])
axes[1].set_title('Process Temperature by Failure Status', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Machine Failure (0=No, 1=Yes)')
axes[1].set_ylabel('Process Temperature [K]')
axes[1].get_figure().suptitle('')

plt.tight_layout()
plt.savefig('images/04_temperature_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/04_temperature_analysis.png")

# ============================================================================
# 5. SPEED AND TORQUE ANALYSIS
# ============================================================================
print("Generating: Speed and Torque Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Rotational speed
df.boxplot(column='Rotational speed [rpm]', by='Machine failure', ax=axes[0])
axes[0].set_title('Rotational Speed by Failure Status', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Machine Failure (0=No, 1=Yes)')
axes[0].set_ylabel('Rotational Speed [rpm]')
axes[0].get_figure().suptitle('')

# Torque
df.boxplot(column='Torque [Nm]', by='Machine failure', ax=axes[1])
axes[1].set_title('Torque by Failure Status', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Machine Failure (0=No, 1=Yes)')
axes[1].set_ylabel('Torque [Nm]')
axes[1].get_figure().suptitle('')

plt.tight_layout()
plt.savefig('images/05_speed_torque_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/05_speed_torque_analysis.png")

# ============================================================================
# 6. TOOL WEAR ANALYSIS
# ============================================================================
print("Generating: Tool Wear Analysis...")
fig, ax = plt.subplots(figsize=(12, 6))

df.boxplot(column='Tool wear [min]', by='Machine failure', ax=ax)
ax.set_title('Tool Wear Distribution by Failure Status', fontsize=14, fontweight='bold')
ax.set_xlabel('Machine Failure (0=No, 1=Yes)')
ax.set_ylabel('Tool Wear [min]')
ax.get_figure().suptitle('')

plt.tight_layout()
plt.savefig('images/06_tool_wear_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/06_tool_wear_analysis.png")

# ============================================================================
# Data Preparation for Model Comparison
# ============================================================================
print("\nPreparing data for model training...")

# Encode categorical variable
le = LabelEncoder()
df['Product type'] = le.fit_transform(df['Type'])

# Rename columns to remove brackets (XGBoost compatibility)
df.columns = df.columns.str.replace('[', '').str.replace(']', '').str.lower().str.replace(' ', '_')

# Features and target
numerical_features_renamed = ['air_temperature_k', 'process_temperature_k', 
                              'rotational_speed_rpm', 'torque_nm', 'tool_wear_min']
X = df[numerical_features_renamed + ['product_type']]
y = df['machine_failure']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 7. MODEL COMPARISON - ROC CURVES
# ============================================================================
print("Generating: ROC Curves Comparison...")

# Train models
lr = LogisticRegression(random_state=42, max_iter=1000)
dt = DecisionTreeClassifier(random_state=42, max_depth=10)
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
xgb_model = XGBClassifier(n_estimators=100, random_state=42, max_depth=7, learning_rate=0.1, scale_pos_weight=28.5)

# Train on scaled data (except tree-based models which don't need scaling)
lr.fit(X_train_scaled, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

# Predictions and probabilities
y_pred_lr = lr.predict(X_test_scaled)
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

y_pred_dt = dt.predict(X_test)
y_pred_proba_dt = dt.predict_proba(X_test)[:, 1]

y_pred_rf = rf.predict(X_test)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

y_pred_xgb = xgb_model.predict(X_test)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# Calculate ROC curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
roc_auc_dt = auc(fpr_dt, tpr_dt)

fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)

# Plot ROC curves
fig, ax = plt.subplots(figsize=(10, 8))

ax.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {roc_auc_lr:.3f})', linewidth=2)
ax.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {roc_auc_dt:.3f})', linewidth=2)
ax.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.3f})', linewidth=2)
ax.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.3f})', linewidth=2, linestyle='-', color='red')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('images/07_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/07_roc_curves.png")

# ============================================================================
# 8. FEATURE IMPORTANCE - XGBOOST
# ============================================================================
print("Generating: Feature Importance (XGBoost)...")

feature_importance = xgb_model.feature_importances_
feature_names = X.columns

feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_df['Feature'], feature_df['Importance'], color='#3498db')
ax.set_title('Feature Importance - XGBoost Model', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('images/08_feature_importance_xgboost.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/08_feature_importance_xgboost.png")

# ============================================================================
# 9. CONFUSION MATRIX - XGBOOST
# ============================================================================
print("Generating: Confusion Matrix (XGBoost)...")

cm = confusion_matrix(y_test, y_pred_xgb)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=True,
            xticklabels=['No Failure', 'Failure'],
            yticklabels=['No Failure', 'Failure'])
ax.set_title('Confusion Matrix - XGBoost Model', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label')
ax.set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('images/09_confusion_matrix_xgboost.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/09_confusion_matrix_xgboost.png")

# ============================================================================
# 10. MODEL PERFORMANCE METRICS COMPARISON
# ============================================================================
print("Generating: Model Metrics Comparison...")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

models = ['Logistic Regression', 'Decision Tree', 'Random Forest', 'XGBoost']
y_preds = [y_pred_lr, y_pred_dt, y_pred_rf, y_pred_xgb]

metrics_data = []
for model_name, y_pred in zip(models, y_preds):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    metrics_data.append([model_name, acc, prec, rec, f1])

metrics_df = pd.DataFrame(metrics_data, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])

# Plot grouped bar chart
fig, ax = plt.subplots(figsize=(12, 6))
x = np.arange(len(models))
width = 0.2

ax.bar(x - 1.5*width, metrics_df['Accuracy'], width, label='Accuracy', color='#3498db')
ax.bar(x - 0.5*width, metrics_df['Precision'], width, label='Precision', color='#e74c3c')
ax.bar(x + 0.5*width, metrics_df['Recall'], width, label='Recall', color='#2ecc71')
ax.bar(x + 1.5*width, metrics_df['F1-Score'], width, label='F1-Score', color='#f39c12')

ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right')
ax.legend(fontsize=11)
ax.set_ylim([0, 1.05])
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/10_model_metrics_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Saved: images/10_model_metrics_comparison.png")

print("\n" + "="*60)
print("✓ All visualizations generated successfully!")
print("="*60)
