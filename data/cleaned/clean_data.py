import pandas as pd
from pathlib import Path

# Resolve paths relative to this script's location
SCRIPT_DIR = Path(__file__).resolve().parent
RAW_DIR = SCRIPT_DIR.parent / "raw"
CLEANED_DIR = SCRIPT_DIR

features = pd.read_csv(RAW_DIR / "german_credit_features.csv")
targets = pd.read_csv(RAW_DIR / "german_credit_targets.csv")

df = pd.concat([features, targets], axis=1)

column_mapping = {
    "Attribute1": "checking_account",
    "Attribute2": "loan_duration",
    "Attribute3": "credit_history",
    "Attribute4": "loan_purpose",
    "Attribute5": "credit_amount",
    "Attribute6": "savings_account",
    "Attribute7": "employment_duration",
    "Attribute8": "installment_rate",
    "Attribute9": "personal_status",
    "Attribute10": "other_debtors",
    "Attribute11": "residence_duration",
    "Attribute12": "property_type",
    "Attribute13": "age",
    "Attribute14": "other_installment_plans",
    "Attribute15": "housing",
    "Attribute16": "existing_credits",
    "Attribute17": "job",
    "Attribute18": "dependents",
    "Attribute19": "telephone",
    "Attribute20": "foreign_worker",
    "class": "loan_status"
}

df.rename(columns=column_mapping, inplace=True)

df.to_csv(
    CLEANED_DIR / "german_credit_cleaned.csv",
    index=False
)

print("Dataset cleaned successfully")
print(df.head())