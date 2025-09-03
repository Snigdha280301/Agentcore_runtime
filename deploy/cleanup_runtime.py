import boto3, json

# Load saved launch result
with open("deploy/launch_result.json", "r") as f:
    launch_result = json.load(f)

region = boto3.session.Session().region_name
agentcore_control_client = boto3.client("bedrock-agentcore-control", region_name=region)
ecr_client = boto3.client("ecr", region_name=region)

# Delete Agent Runtime
runtime_delete_response = agentcore_control_client.delete_agent_runtime(
    agentRuntimeId=launch_result["agent_id"]
)
print("Deleted runtime:", runtime_delete_response)

# Delete ECR Repo
response = ecr_client.delete_repository(
    repositoryName=launch_result["ecr_uri"].split('/')[1],
    force=True
)
print("Deleted repo:", response)
