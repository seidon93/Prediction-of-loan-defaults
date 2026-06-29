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

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)


df_model = df.copy()
X = df_model.drop("loan_status", axis=1)

y = df_model["loan_status"]
X = pd.get_dummies(
    X,
    drop_first=True
)
print(X.head())
print(X.shape)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Trénovací data:", X_train.shape)
print("Testovací data:", X_test.shape)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\n===== MODEL RESULTS =====")

print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions, pos_label=2))
print("Recall   :", recall_score(y_test, predictions, pos_label=2))
print("F1 Score :", f1_score(y_test, predictions, pos_label=2))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, predictions))