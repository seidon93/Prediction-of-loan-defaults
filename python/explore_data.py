import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

features = pd.read_csv(os.path.join(RAW_DATA_DIR, "german_credit_features.csv"))
targets = pd.read_csv(os.path.join(RAW_DATA_DIR, "german_credit_targets.csv"))

df = pd.concat([features, targets], axis=1)

print(df.shape)
print(df.info())
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())