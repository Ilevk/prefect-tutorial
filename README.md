# Prefect-Tutorial

## Setup Environment
```
$ pip install -f requirements.txt
```

## Run MLFlow with minio, postgres
- mlflow: localhost:5000
- postgres: localhost:5432
- minio: localhost:9000
```
$ bash start_mlflow.sh
```

## Run Local Agent
```
$ cd src
$ bash start_agent.sh
```

## Register Flow
```
$ cd src
$ python3 main.py
```