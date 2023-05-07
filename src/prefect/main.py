from prefect import flow

from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from tasks.train import (
    load_data_task,
    hpo_task,
    train_task,
    log_model_task,
    create_model_version,
    transition_model_task,
)

@flow(name="Train Pipeline")
def train_pipeline(eval_metric, model_name):
    train, valid = load_data_task()
    prep_pipeline, params, metrics = hpo_task(
        train, valid
    )
    model = train_task(
        prep_pipeline,
        train,
        valid,
        params
    )
    run_id, model_uri = log_model_task(
        model, model_name, params, metrics
    )
    current_version = create_model_version(
        model_name, run_id, model_uri, eval_metric
    )
    production_version = transition_model_task(
        model_name, current_version, eval_metric
    )

if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=train_pipeline,                                              # 사용할 flow 함수
        name="Example Flow Deployment",                                   # deployment 이름
        version=1,                                                        # deployment version
        work_queue_name="train_agent",                                    # 사용할 work queue 이름
        schedule=(CronSchedule(cron="* * * * *", timezone="Asia/Seoul")), # 스케줄링
        parameters=dict(eval_metric="MSE", model_name="apartment-model"), # flow 함수에 전달할 파라미터
    )

    deployment.apply()
