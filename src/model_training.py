from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

def train_model(X_train, y_train, model_path="models/fraud_model.pkl"):
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model
