import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    script_dir,
    "..",
    "data",
    "cleaned",
    "german_credit_cleaned.csv"
)

df = pd.read_csv(csv_path)

numeric_cols = [
    "loan_duration",
    "credit_amount",
    "installment_rate",
    "residence_duration",
    "age",
    "existing_credits",
    "dependents",
    "loan_status"
]

corr = df[numeric_cols].corr()

print(corr["loan_status"].sort_values(ascending=False))