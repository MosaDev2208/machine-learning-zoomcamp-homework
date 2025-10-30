#!/usr/bin/env python
import pickle
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

print(f'pandas=={pd.__version__}')
print(f'numpy=={np.__version__}')
print(f'sklearn=={sklearn.__version__}')

def load_data():
    """Load and preprocess the telco churn dataset"""
    data_url = 'https://raw.githubusercontent.com/alexeygrigorev/mlbookcamp-code/master/chapter-03-churn-prediction/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    df = pd.read_csv(data_url)
    
    # Clean column names
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    
    # Clean categorical columns
    categorical_columns = list(df.dtypes[df.dtypes == 'object'].index)
    for c in categorical_columns:
        df[c] = df[c].str.lower().str.replace(' ', '_')
    
    # Fix totalcharges column
    df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
    df.totalcharges = df.totalcharges.fillna(0)
    
    df.churn = (df.churn == 'yes').astype(int)
    
    return df

def train_model(df):
    """Train logistic regression model"""
    numerical = ['tenure', 'monthlycharges', 'totalcharges']
    categorical = ['gender', 'seniorcitizen', 'partner', 'dependents',
                   'phoneservice', 'multiplelines', 'internetservice',
                   'onlinesecurity', 'onlinebackup', 'deviceprotection',
                   'techsupport', 'streamingtv', 'streamingmovies',
                   'contract', 'paperlessbilling', 'paymentmethod']
    
    dicts = df[numerical + categorical].to_dict(orient='records')
    
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(dicts)
    y = df.churn.values
    
    model = LogisticRegression(random_state=1, max_iter=1000)
    model.fit(X, y)
    
    pipeline = make_pipeline(dv, model)
    return pipeline

print("Loading data...")
df = load_data()
print(f"Dataset shape: {df.shape}")

print("\nTraining model...")
pipeline = train_model(df)

print("\nSaving model...")
with open('model.bin', 'wb') as f_out:
    pickle.dump(pipeline, f_out)

print("✅ Model saved to model.bin successfully!")
