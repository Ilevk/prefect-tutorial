# example.py

import prefect
from prefect import task, Flow

logger = prefect.context.get("logger")

@task
def example_task():
    logger.info("Hello, world!")
    
with Flow("Example Flow") as flow:
    example_task()
    

flow.register(project_name="tutorial")

# flow.run()
