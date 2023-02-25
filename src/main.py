from prefect import Flow

from prefect import Parameter
from tasks.train import (
    load_data_task,
    hpo_task,
    train_task,
    log_model_task,
    create_model_version,
    transition_model_task,
)

with Flow("Model Train Flow") as flow:
    eval_metric = Parameter("Evaluation Metric", "MSE")
    model_name = Parameter("Model Registry Name", "project_name")

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
    flow.register(project_name="prefect-tutorial",
                  add_default_labels=False,
                  labels=['train_agent'])
