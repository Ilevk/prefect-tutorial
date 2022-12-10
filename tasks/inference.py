from datetime import datetime
from typing import List

import mlflow
import pandas as pd
import prefect
from prefect import task
from sklearn.pipeline import Pipeline

from utils import save_prediction

logger = prefect.context.get("logger")


@task(log_stdout=True)
def load_test_data() -> pd.DataFrame:
    logger.info("Data Preprocessing start")

    data = pd.read_csv("test.csv")

    return data


@task(log_stdout=True)
def load_model_task(model_name: str) -> Pipeline:
    logger.info("Load Production Model Start.")

    model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")

    logger.info(f"Done Load Model.")

    return model


@task(log_stdout=True, nout=2)
def batch_inference_task(
    model: Pipeline,
    data: pd.DataFrame,
) -> List[pd.DataFrame]:
    logger.info("Batch Inference Start")

    results = pd.read_csv("submission.csv")
    results["transaction_real_price"] = model.predict(data)
    results["predict_date"] = datetime.today().date()

    logger.info("Batch Inference Done")
    return results


@task(log_stdout=True)
def save_database(results: List[pd.DataFrame]) -> None:
    logger.info(f"Save to db Start")

    save_prediction(results)

    logger.info(f"Save Result Done")
