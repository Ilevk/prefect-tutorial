docker-compose up -d

export AWS_ACCESS_KEY_ID=mlflow_admin
export AWS_SECRET_ACCESS_KEY=mlflow_admin
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

mlflow server --port 5000 --backend-store-uri postgresql://postgres:postgres@localhost:5432/mlflow_db --default-artifact-root s3://mlflow/mlruns

docker-compose stop
