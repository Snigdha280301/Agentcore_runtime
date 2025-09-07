import boto3, json, uuid

client = boto3.client("bedrock-agentcore")

response = client.invoke_agent_runtime(
    agentRuntimeArn="arn:aws:bedrock-agentcore:us-east-1:356225522107:runtime/cityassistant_311_agent_3-38spwz8TKm",
    runtimeUserId="test-user",
    runtimeSessionId=str(uuid.uuid4()),  # ✅ length = 36
    payload=json.dumps({"prompt": "hello"})
)

print(response["response"].read().decode("utf-8"))
