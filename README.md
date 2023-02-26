# Prefect-Tutorial

## Prerequisites
### 1. Setup Environment
```
$ pip install -f requirements.txt
```

### 2. Prepare Dataset
[Dacon Apartment Price Prediction Competition.](https://dacon.io/competitions/official/21265/overview/description)
1. Download dataset & Unzip
2. Run to_database script
```
$ cd src/database
$ python3 to_database.py
```


### 3. Run MLFlow with minio, postgres (Model Registry)
- Run mlflow server, minio and postgres are run by docker-compose.
#### Conenction Info
- mlflow
    - url: localhost:5000
- postgres
    - url: localhost:5432
    - user: postgres
    - password: postgres
    - database: postgres, mlflow_db, optuna
- minio
    - url: localhost:9000, localhost:9001
    - user: mlflow_admin
    - password: mlflow_admin
    - bucket: mlflow
```
$ bash start_mlflow.sh
```

### 4. Run Local Agent (Worker)
```
$ cd src/prefect
$ bash start_agent.sh
```

## 5. Register Flow
- Register flow to Prefect Cloud.
- Trigger the pipeline after registration.
```
$ cd src/prefect
$ python3 main.py
```
