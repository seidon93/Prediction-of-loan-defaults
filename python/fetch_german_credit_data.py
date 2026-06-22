"""
Fetch German Credit Data from UCI ML Repository.

Downloads the Statlog (German Credit Data) dataset (ID=144)
and saves raw features and targets as CSV files into data/raw/.
"""

import os
from ucimlrepo import fetch_ucirepo

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, "data", "raw")

os.makedirs(RAW_DATA_DIR, exist_ok=True)

# ── Fetch dataset ────────────────────────────────────────────────────────────
print("Fetching Statlog (German Credit Data) from UCI ML Repository …")
statlog_german_credit_data = fetch_ucirepo(id=144)

# Data (as pandas DataFrames)
X = statlog_german_credit_data.data.features
y = statlog_german_credit_data.data.targets

# ── Save raw data ────────────────────────────────────────────────────────────
features_path = os.path.join(RAW_DATA_DIR, "german_credit_features.csv")
targets_path = os.path.join(RAW_DATA_DIR, "german_credit_targets.csv")

X.to_csv(features_path, index=False)
y.to_csv(targets_path, index=False)

print(f"Features saved to: {features_path}  ({X.shape[0]} rows, {X.shape[1]} columns)")
print(f"Targets  saved to: {targets_path}  ({y.shape[0]} rows, {y.shape[1]} columns)")

# -- Metadata ------------------------------------------------------------------
print("\n-- METADATA " + "-" * 50)
print(statlog_german_credit_data.metadata)

# -- Variable information ------------------------------------------------------
print("\n-- VARIABLES " + "-" * 49)
print(statlog_german_credit_data.variables)

