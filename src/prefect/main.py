from prefect import flow

from prefect.deployments import Deployment
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
        train, valid, upstream_tasks=[train, valid]
    )
    model = train_task(
        prep_pipeline,
        train,
        valid,
        params,
        upstream_tasks=[prep_pipeline, params, metrics],
    )
    run_id, model_uri = log_model_task(
        model, model_name, params, metrics, upstream_tasks=[model, params, metrics]
    )
    current_version = create_model_version(
        model_name, run_id, model_uri, eval_metric, upstream_tasks=[run_id, model_uri]
    )
    production_version = transition_model_task(
        model_name, current_version, eval_metric, upstream_tasks=[current_version]
    )

if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=train_pipeline,                   # 사용할 flow 함수
        name="Example Flow Deployment",        # deployment 이름
        version=1,                             # deployment version
        work_queue_name="train_agent",         # 사용할 work queue 이름
        eval_metric="mse",                     # flow argument 1
        model_name="apartment-model",          # flow argument 2
    )
