import pickle
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ======================================================
# Project Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "Titanic-Dataset.csv"
MODEL_PATH = BASE_DIR / "models" / "titanic_model.pkl"

# ======================================================
# Load Dataset
# ======================================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

# ======================================================
# Data Preprocessing
# ======================================================

# Drop unnecessary columns
df.drop(
    columns=["PassengerId", "Name", "Ticket", "Cabin"],
    inplace=True
)

# Fill missing values
df["Age"] = df["Age"].fillna(df["Age"].median())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encode categorical variables
df["Sex"] = df["Sex"].map({
    "male": 0,
    "female": 1
})

df["Embarked"] = df["Embarked"].map({
    "S": 0,
    "C": 1,
    "Q": 2
})

# ======================================================
# Split Dataset
# ======================================================

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ======================================================
# Train Model
# ======================================================

print("Training Random Forest Model...")

model = RandomForestClassifier(random_state=42)

model.fit(X_train, y_train)

# ======================================================
# Save Model
# ======================================================

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("\nModel trained successfully!")
print(f"Model saved at:\n{MODEL_PATH}")