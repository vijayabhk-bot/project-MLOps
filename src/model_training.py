# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import classification_report
# import joblib

# def train_model(X_train, y_train, model_path="models/fraud_model.pkl"):
#     model = LogisticRegression(max_iter=1000, class_weight="balanced")
#     model.fit(X_train, y_train)
#     joblib.dump(model, model_path)
#     return model




from sklearn.linear_model import LogisticRegression
import joblib
import os

def train_model(X_train, y_train, model_path="models/fraud_model.pkl",
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
