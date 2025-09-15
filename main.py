# from src.data_preprocessing import load_and_preprocess_data
# from src.model_training import train_model
# from src.model_evaluation import evaluate_model

# def main():
#     X_train, X_test, y_train, y_test = load_and_preprocess_data()
#     model = train_model(X_train, y_train)
#     evaluate_model(model, X_test, y_test)

# if __name__ == "__main__":
#     main()




# import mlflow
# import mlflow.sklearn
# from src.data_preprocessing import load_and_preprocess_data
# from src.model_training import train_model
# from src.model_evaluation import evaluate_model

# def main():
#     mlflow.set_experiment("/Shared/fraud-detection-exp")

#     with mlflow.start_run():
#         X_train, X_test, y_train, y_test = load_and_preprocess_data()
#         model = train_model(X_train, y_train)
#         report = evaluate_model(model, X_test, y_test)

#         # Log model
#         mlflow.sklearn.log_model(model, "fraud_model")
#         mlflow.log_param("algorithm", "LogisticRegression")

# if __name__ == "__main__":
#     main()




import mlflow
import mlflow.sklearn
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model
import joblib
import os

def main():
    # Set MLflow experiment (Databricks compatible)
    mlflow.set_experiment("/Shared/fraud-detection-exp")

    with mlflow.start_run():
        # Load data
        X_train, X_test, y_train, y_test = load_and_preprocess_data()

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        report = evaluate_model(model, X_test, y_test)

        # ----   Log Parameters ----
        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")

        # ---- Log Metrics ----
        mlflow.log_metric("accuracy", report["accuracy"])
        mlflow.log_metric("precision", report["precision"])
        mlflow.log_metric("recall", report["recall"])
        mlflow.log_metric("f1_score", report["f1_score"])

        # ----   Log Model ----
        mlflow.sklearn.log_model(model, "fraud_model")

        # ----   Save local copy ----
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/fraud_model.pkl")

        print("\n Training Complete")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Precision: {report['precision']:.4f}")
        print(f"Recall: {report['recall']:.4f}")
        print(f"F1 Score: {report['f1_score']:.4f}")
        print("Model also saved locally in: models/fraud_model.pkl")

if __name__ == "__main__":
    main()
