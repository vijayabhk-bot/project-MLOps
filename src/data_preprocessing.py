import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os


def resolve_path(path):
    # If running locally, fallback to relative data folder
    if not os.path.exists(path):
        if path.startswith("dbfs:/"):
            # Convert dbfs:/ to /dbfs/ for Databricks
            path = path.replace("dbfs:/", "/dbfs/")
        elif "creditcard.csv" in path:
            # Use local dataset
            path = os.path.join("data", "creditcard.csv")
    return path

def load_and_preprocess_data(path="data/creditcard.csv"):
    path = resolve_path(path)
    df = pd.read_csv(path)
    
    # Features & labels
    X = df.drop("Class", axis=1)
    y = df["Class"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test
