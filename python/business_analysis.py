import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "cleaned", "german_credit_cleaned.csv")

df = pd.read_csv(csv_path)

print(df["loan_status"].value_counts())

print()

print(
    round(
        df["loan_status"].value_counts(normalize=True) * 100,
        2
    )
)