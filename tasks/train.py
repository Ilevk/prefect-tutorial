from typing import Any, Dict, Tuple

import pandas as pd
from sqlalchemy import create_engine
import optuna
import prefect
from prefect import task
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.store.artifact.runs_artifact_repo import RunsArtifactRepository
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error


logger = prefect.context.get("logger")


def get_prep_pipeline(data: pd.DataFrame):
    # 데이터 타입별로 분류
    num_columns = data.select_dtypes(exclude=[object]).columns.values
    cat_columns = data.select_dtypes(include=[object]).columns.values
    
    logger.info(f"num_columns: {num_columns}")
    logger.info(f"cat_columns: {cat_columns}")

    # 전처리 파이프라인
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_columns),
        ]
    )

    return preprocessor


@task(log_stdout=True, nout=2)
def load_data_task() -> Tuple[Tuple[pd.DataFrame], Tuple[pd.DataFrame]]:
   engine = create_engine("postgresql://postgres:postgres@localhost:5432/postgres")

   sql = "SELECT * FROM public.apartments where transaction_real_price is not null"
   data  = pd.read_sql(sql, con=engine)
    
   train = data.sample(5000)

   label = train["transaction_real_price"]
   train.drop(columns=['apartment_id', 'transaction_id', 'transaction_real_price', 'jibun', 'apt', 'addr_kr', 'dong'], axis=1, inplace=True)

   x_train, x_valid, y_train, y_valid = train_test_split(
       train, label, test_size=0.7, random_state=42)

   return (x_train, y_train), (x_valid, y_valid)


@task(log_stdout=True, nout=3)
def hpo_task(train: Tuple[pd.DataFrame], valid: Tuple[pd.DataFrame],
    ) -> Tuple[Dict[str, Any], Dict[str, float]]:

    x_train, y_train = train
    x_valid, y_valid = valid

    prep_pipeline = get_prep_pipeline(x_train)

    x_train = prep_pipeline.fit_transform(x_train)
    x_valid = prep_pipeline.transform(x_valid)

    logger.info("HPO Start.")

    def objectiveModel(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 15),
            "subsample": trial.suggest_float("subsample", 0.5, 1),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
        }

        model = XGBRegressor(**params)

        model.fit(
            x_train,
            y_train,
            eval_metric="rmse",
            eval_set=[[x_train, y_train], [x_valid, y_valid]],
            early_stopping_rounds=30,
            verbose=None,
        )

        score = mean_squared_error(y_valid, model.predict(x_valid))

        return score

    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objectiveModel(trial), n_trials=1)

    logger.info(f"Best params: {study.best_params }, Best score:{study.best_value}")
    metrics = {"MSE": study.best_value}

    return prep_pipeline, study.best_trial.params, metrics


@task()
def train_task(
    prep_pipeline: Pipeline,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    params: Dict[str, Any],
):

    logger.info(f"Training Start with params: {params}")
    x_train, y_train = train
    x_valid, y_valid = valid

    x_train = prep_pipeline.transform(x_train)
    x_valid = prep_pipeline.transform(x_valid)

    model = XGBRegressor(**params)

    model.fit(
        x_train,
        y_train,
        eval_metric="rmse",
        eval_set=[[x_train, y_train], [x_valid, y_valid]],
        early_stopping_rounds=30,
        verbose=None,
    )

    logger.info(f"Training End")

    pipeline = Pipeline(steps=[("preprocessor", prep_pipeline), ("model", model)])

    return pipeline


@task(log_stdout=True, nout=2)
def log_model_task(
    model: Pipeline,
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
) -> Tuple[str, str]:
    logger.info(
        f"Log Model with params: {params} metric: {metrics} model_name: {model_name}"
    )

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        model_info = mlflow.sklearn.log_model(model, model_name)

    logger.info(f"Log Model Done: {model_info}")
    return model_info.run_id, model_info.model_uri


@task(log_stdout=True)
def create_model_version(
    model_name: str, run_id: str, model_uri: str, eval_metric: str
) -> str:
    logger.info(
        f"Create model version model_name: {model_name}, run_id: {run_id}, model_uri: {model_uri}, eval_metric: {eval_metric}"
    )

    client = MlflowClient()

    try:
        client.create_registered_model(model_name)
    except Exception as e:
        logger.info("Model already exists")

    current_metric = client.get_run(run_id).data.metrics[eval_metric]
    model_source = RunsArtifactRepository.get_underlying_uri(model_uri)
    model_version = client.create_model_version(
        model_name, model_source, run_id, description=f"{eval_metric}: {current_metric}"
    )

    logger.info(f"Done Create model version")
    return model_version.version


@task(log_stdout=True)
def transition_model_task(model_name: str, version: str, eval_metric: str) -> str:
    logger.info(f"Deploy model: {model_name} version: {version}")

    client = MlflowClient()
    production_model = None
    current_model = client.get_model_version(model_name, version)

    filter_string = f"name='{current_model.name}'"
    results = client.search_model_versions(filter_string)

    for mv in results:
        if mv.current_stage == "Production":
            production_model = mv

    if production_model is None:
        client.transition_model_version_stage(
            current_model.name, current_model.version, "Production"
        )
        production_model = current_model
    else:
        current_metric = client.get_run(current_model.run_id).data.metrics[eval_metric]
        production_metric = client.get_run(production_model.run_id).data.metrics[
            eval_metric
        ]

        if current_metric < production_metric:
            client.transition_model_version_stage(
                current_model.name,
                current_model.version,
                "Production",
                archive_existing_versions=True,
            )
            production_model = current_model

    logger.info(f"Done Deploy Production_Model: {production_model}")
    return production_model.version
