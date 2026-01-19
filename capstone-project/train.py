import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

# 1. Parameters
# --------------------------------------------------------
output_file = 'model.bin'
# XGBoost parameters (from your notebook tuning)
xgb_params = {
    'eta': 0.3, 
    'max_depth': 6,
    'min_child_weight': 1,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'nthread': 8,
    'seed': 1,
    'verbosity': 1,
}
num_boost_round = 100

# 2. Data Preparation
# --------------------------------------------------------
print("Loading data...")
df = pd.read_csv('data.csv')

# Clean columns
df.columns = df.columns.str.lower().str.replace(' ', '_')
df = df.fillna(0)

# Convert Target to Binary (0=Healthy, 1=Maintenance)
df['maintenance_label'] = (df['maintenance_label'] > 0).astype(int)

# Split Data
print("Splitting data...")
df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

y_train = df_full_train.maintenance_label.values
y_test = df_test.maintenance_label.values

del df_full_train['maintenance_label']
del df_test['maintenance_label']

# Feature Engineering
categorical = list(df_full_train.select_dtypes(include=['object']).columns)
numerical = list(df_full_train.select_dtypes(include=['number']).columns)

dv = DictVectorizer(sparse=False)
train_dicts = df_full_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)

test_dicts = df_test[categorical + numerical].to_dict(orient='records')
X_test = dv.transform(test_dicts)

# 3. Training
# --------------------------------------------------------
print("Training XGBoost model...")
features = dv.get_feature_names_out()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features.tolist())
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=features.tolist())

model = xgb.train(xgb_params, dtrain, num_boost_round=num_boost_round)

# 4. Evaluation
# --------------------------------------------------------
y_pred = model.predict(dtest)
auc = roc_auc_score(y_test, y_pred)
print(f"Validation AUC: {auc:.3f}")

# 5. Save Model
# --------------------------------------------------------
print(f"Saving model to {output_file}...")
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print("Training finished successfully!")