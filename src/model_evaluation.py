# from sklearn.metrics import classification_report

# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     report = classification_report(y_test, y_pred, target_names=["Not Fraud", "Fraud"])
#     print(report)
#     return report



from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    # Instead of string report, return a dict
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }

    # Optional: still print the detailed classification report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

    return metrics