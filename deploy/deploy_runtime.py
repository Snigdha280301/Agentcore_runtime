from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session
import time

boto_session = Session()
region = boto_session.region_name

agentcore_runtime = Runtime()
agent_name = "langgraph_claude_getting_started"

response = agentcore_runtime.configure(
    entrypoint="langgraph_bedrock.py",
    auto_create_execution_role=True,
    auto_create_ecr=True,
    requirements_file="requirements.txt",
    region=region,
    agent_name=agent_name
)
print("Configured:", response)

launch_result = agentcore_runtime.launch()
print("Launched:", launch_result)

status_response = agentcore_runtime.status()
status = status_response.endpoint['status']
while status not in ['READY', 'CREATE_FAILED', 'DELETE_FAILED', 'UPDATE_FAILED']:
    time.sleep(10)
    status_response = agentcore_runtime.status()
    status = status_response.endpoint['status']
    print(status)
print("Final status:", status)

invoke_response = agentcore_runtime.invoke({"prompt": "How much is 2+2?"})
print("Invoke response:", invoke_response)
