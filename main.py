# from src.data_preprocessing import load_and_preprocess_data
# from src.model_training import train_model
# from src.model_evaluation import evaluate_model

# def main():
#     X_train, X_test, y_train, y_test = load_and_preprocess_data()
#     model = train_model(X_train, y_train)
#     evaluate_model(model, X_test, y_test)

# if __name__ == "__main__":
#     main()




import mlflow
import mlflow.sklearn
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.model_evaluation import evaluate_model

def main():
    mlflow.set_experiment("/Shared/fraud-detection-exp")

    with mlflow.start_run():
        X_train, X_test, y_train, y_test = load_and_preprocess_data()
        model = train_model(X_train, y_train)
        report = evaluate_model(model, X_test, y_test)

        # Log model
        mlflow.sklearn.log_model(model, "fraud_model")
        mlflow.log_param("algorithm", "LogisticRegression")

if __name__ == "__main__":
    main()
