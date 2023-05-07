# example.py

from prefect import task, flow, get_run_logger
from prefect.deployments import Deployment


@task
def example_task():
    logger = get_run_logger()
    logger.info("Hello, world!")

@flow(name="Example Flow")
def example_flow():
    example_task()
    

if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=example_flow,                    # 사용할 flow 함수
        name="Example Flow Deployment",       # deployment 이름
        version=1,                            # deployment version
        work_queue_name="example-flow-queue", # 사용할 work queue 이름
    )
