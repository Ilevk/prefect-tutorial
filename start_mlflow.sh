docker-compose up -d

export AWS_ACCESS_KEY_ID=mlflow_admin
export AWS_SECRET_ACCESS_KEY=mlflow_admin
export MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

mlflow server --port 5001 --backend-store-uri postgresql://postgres:postgres@localhost:5433/postgres --default-artifact-root s3://mlflow/mlruns

docker-compose down