import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

script_dir = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(
    script_dir,
    "..",
    "data",
    "cleaned",
    "german_credit_cleaned.csv"
)

df = pd.read_csv(csv_path)

print(df.head())

df.info()

print("\nInformace o datech")
print(df.info())

print("\nChybějící hodnoty")
print(df.isnull().sum())

features = [
    "credit_amount",
    "loan_duration",
    "age",
    "installment_rate"
]

target = "loan_status"

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape)
print("Test :", X_test.shape)

from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
print(predictions[:20])

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

cm = confusion_matrix(y_test, predictions)

print(cm)