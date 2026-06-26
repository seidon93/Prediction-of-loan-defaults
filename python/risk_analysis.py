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

checking_mapping = {
    "A11": "< 0 DM",
    "A12": "0 - 200 DM",
    "A13": "> 200 DM",
    "A14": "No Checking Account"
}

df["checking_account"] = df["checking_account"].map(checking_mapping)

result = (
    df.groupby(["checking_account", "loan_status"])
      .size()
      .unstack(fill_value=0)
)

result.columns = ["Good Credit", "Bad Credit"]

result["Default Rate (%)"] = (
    result["Bad Credit"] /
    (result["Good Credit"] + result["Bad Credit"])
    * 100
).round(2)

output_path = os.path.join(
    script_dir,
    "..",
    "data",
    "cleaned",
    "checking_account_risk.csv"
)

result.to_csv(output_path)