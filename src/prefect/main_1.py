from prefect import Flow
from prefect.schedules import CronSchedule

from tasks.train_1 import (
    load_dataset_task,
    hpo_task,
    train_task
)

with Flow(name="apartments-tutorial",
          schedule=CronSchedule("* * * * *")) as flow:
    train, valid = load_dataset_task()
    preprocesser, best_params, best_values = hpo_task(train, valid)
    pipeline = train_task(preprocesser, train, valid, best_params)


if __name__ == "__main__":
    flow.register(project_name="prefect-tutorial", 
                  add_default_labels=False,
                  labels=["train_agent"])
