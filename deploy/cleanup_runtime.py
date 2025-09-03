import boto3
from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session

boto_session = Session()
region = boto_session.region_name

agentcore_runtime = Runtime()
launch_result = agentcore_runtime.launch()  # or store this from deploy step

agentcore_control_client = boto3.client("bedrock-agentcore-control", region_name=region)
ecr_client = boto3.client("ecr", region_name=region)

runtime_delete_response = agentcore_control_client.delete_agent_runtime(
    agentRuntimeId=launch_result.agent_id
)
print("Deleted runtime:", runtime_delete_response)

response = ecr_client.delete_repository(
    repositoryName=launch_result.ecr_uri.split('/')[1],
    force=True
)
print("Deleted repo:", response)
