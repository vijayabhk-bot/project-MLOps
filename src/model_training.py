
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model(X_train, y_train, model_path="models/fraud_model_vA.pkl",
                max_iter=1000, class_weight="balanced"):
    """
    Train Logistic Regression model.
    
    Args:
        X_train: Training features
        y_train: Training labels
        model_path: Path to save trained model
        max_iter: Maximum iterations for solver
        class_weight: Handle class imbalance
    """
    model = LogisticRegression(max_iter=max_iter, class_weight=class_weight)
    model.fit(X_train, y_train)

    # Ensure models/ directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model

from sklearn.ensemble import RandomForestClassifier

def train_model_B(X_train, y_train, model_path="models/fraud_model_vB.pkl",
                  n_estimators=100, max_depth=10, class_weight="balanced", random_state=42):
    """
    Train RandomForest model (Model B).
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state
    )
    model.fit(X_train, y_train)

    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    return model

