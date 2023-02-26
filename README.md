# Prefect-Tutorial

## Setup Environment
```
$ pip install -f requirements.txt
```

## Dataset Source
- Dacon Apartment Price Prediction Competition.
- https://dacon.io/competitions/official/21265/overview/description

```
$ cd src/database
$ python3 to_database.py
```


## Run MLFlow with minio, postgres
- mlflow: localhost:5000
- postgres: localhost:5432
    - user: postgres
    - password: postgres
    - database: postgres, mlflow_db, optuna
- minio: localhost:9000
    - user: mlflow_admin
    - password: mlflow_admin
    - bucket: mlflow
```
$ bash start_mlflow.sh
```

## Run Local Agent
```
$ cd src/prefect
$ bash start_agent.sh
```

## Register Flow
```
$ cd src/prefect
$ python3 main.py
```