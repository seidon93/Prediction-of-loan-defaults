import pandas as pd
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "..", "data", "cleaned", "german_credit_cleaned.csv")

df = pd.read_csv(csv_path)

for col in df.select_dtypes(include="object"):
    print("\n")
    print(col)
    print(df[col].unique())