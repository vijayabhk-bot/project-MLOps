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




# import mlflow
# import mlflow.sklearn
# from src.data_preprocessing import load_and_preprocess_data
# from src.model_training import train_model,train_model_B
# from src.model_evaluation import evaluate_model
# import joblib
# import os

# def main():
#     # Set MLflow experiment (Databricks compatible)
#     mlflow.set_experiment("/Shared/fraud-detection-exp")

#     with mlflow.start_run():
#         # Load data
#         X_train, X_test, y_train, y_test = load_and_preprocess_data()

#         # Train model
#         model = train_model(X_train, y_train)

#         # Evaluate model
#         report = evaluate_model(model, X_test, y_test)

#         # ----   Log Parameters ----
#         mlflow.log_param("algorithm", "LogisticRegression")
#         mlflow.log_param("max_iter", 1000)
#         mlflow.log_param("class_weight", "balanced")

#         # ---- Log Metrics ----
#         mlflow.log_metric("accuracy", report["accuracy"])
#         mlflow.log_metric("precision", report["precision"])
#         mlflow.log_metric("recall", report["recall"])
#         mlflow.log_metric("f1_score", report["f1_score"])

#         # ----   Log Model ----
#         mlflow.sklearn.log_model(model, "fraud_model")

#         # ----   Save local copy ----
#         os.makedirs("models", exist_ok=True)
#         joblib.dump(model, "models/fraud_model.pkl")

#         print("\n Training Complete")
#         print(f"Accuracy: {report['accuracy']:.4f}")
#         print(f"Precision: {report['precision']:.4f}")
#         print(f"Recall: {report['recall']:.4f}")
#         print(f"F1 Score: {report['f1_score']:.4f}")
#         print("Model also saved locally in: models/fraud_model.pkl")


#     with mlflow.start_run():
#         # Load data
#         X_train, X_test, y_train, y_test = load_and_preprocess_data()

#         # Train model
#         model = train_model(X_train, y_train)

#         # Evaluate model
#         report = evaluate_model(model, X_test, y_test)

#         # ----   Log Parameters ----
#         mlflow.log_param("algorithm", "LogisticRegression")
#         mlflow.log_param("max_iter", 1000)
#         mlflow.log_param("class_weight", "balanced")

#         # ---- Log Metrics ----
#         mlflow.log_metric("accuracy", report["accuracy"])
#         mlflow.log_metric("precision", report["precision"])
#         mlflow.log_metric("recall", report["recall"])
#         mlflow.log_metric("f1_score", report["f1_score"])

#         # ----   Log Model ----
#         mlflow.sklearn.log_model(model, "fraud_model")

#         # ----   Save local copy ----
#         os.makedirs("models", exist_ok=True)
#         joblib.dump(model, "models/fraud_model.pkl")

#         print("\n Training Complete")
#         print(f"Accuracy: {report['accuracy']:.4f}")
#         print(f"Precision: {report['precision']:.4f}")
#         print(f"Recall: {report['recall']:.4f}")
#         print(f"F1 Score: {report['f1_score']:.4f}")
#         print("Model also saved locally in: models/fraud_model.pkl")

# if __name__ == "__main__":
#     main()




import mlflow
import mlflow.sklearn
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model, train_model_B   # <-- Make sure train_model_B exists
from src.model_evaluation import evaluate_model
import joblib
import os


def main():
    # Set MLflow experiment
    mlflow.set_experiment("/Shared/fraud-detection-exp")

    # -------------------------
    # Train Model A (Logistic Regression)
    # -------------------------
    with mlflow.start_run(run_name="LogisticRegression_Model"):
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        model_A = train_model(X_train, y_train, model_path="models/fraud_model_vA.pkl")

        report_A = evaluate_model(model_A, X_test, y_test)

        mlflow.log_param("algorithm", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("accuracy", report_A["accuracy"])
        mlflow.log_metric("precision", report_A["precision"])
        mlflow.log_metric("recall", report_A["recall"])
        mlflow.log_metric("f1_score", report_A["f1_score"])

        mlflow.sklearn.log_model(model_A, "fraud_model_vA")

        print("\n[Model A - Logistic Regression]")
        print(f"Accuracy: {report_A['accuracy']:.4f}")
        print(f"Precision: {report_A['precision']:.4f}")
        print(f"Recall: {report_A['recall']:.4f}")
        print(f"F1 Score: {report_A['f1_score']:.4f}")
        print("Saved as: models/fraud_model_vA.pkl")

    # -------------------------
    # Train Model B (RandomForest)
    # -------------------------
    with mlflow.start_run(run_name="RandomForest_Model"):
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        model_B = train_model_B(X_train, y_train, model_path="models/fraud_model_vB.pkl")

        report_B = evaluate_model(model_B, X_test, y_test)

        mlflow.log_param("algorithm", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("class_weight", "balanced")

        mlflow.log_metric("accuracy", report_B["accuracy"])
        mlflow.log_metric("precision", report_B["precision"])
        mlflow.log_metric("recall", report_B["recall"])
        mlflow.log_metric("f1_score", report_B["f1_score"])

        mlflow.sklearn.log_model(model_B, "fraud_model_vB")

        print("\n[Model B - RandomForest]")
        print(f"Accuracy: {report_B['accuracy']:.4f}")
        print(f"Precision: {report_B['precision']:.4f}")
        print(f"Recall: {report_B['recall']:.4f}")
        print(f"F1 Score: {report_B['f1_score']:.4f}")
        print("Saved as: models/fraud_model_vB.pkl")


if __name__ == "__main__":
    main()
