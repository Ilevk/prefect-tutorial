import mlflow
from sklearn.ensemble import RandomForestClassifier

mlflow.set_tracking_uri('http://localhost:5001')

mlflow.set_experiment("MINIO_Default")

params = {"max_depth": 5, "n_estimators": 10}
metrics = {"accuracy": 0.9, "auc": 0.8}

model = RandomForestClassifier(**params)

with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "rf_model")
