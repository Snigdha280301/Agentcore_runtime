from bedrock_agentcore_starter_toolkit import Runtime
from boto3.session import Session
from botocore.exceptions import ClientError
import time, json, os

print(">>> deploy_runtime.py starting...", flush=True)

session = Session()
region = session.region_name or os.getenv("AWS_REGION") or "us-east-1"
agent_name = "cityassist311"
entrypoint = "langgraph_bedrock.py"
reqs = "requirements.txt"

rt = Runtime()

print(f">>> configure(region={region}, agent_name={agent_name})", flush=True)
cfg = rt.configure(
    entrypoint=entrypoint,
    requirements_file=reqs,
    auto_create_execution_role=True,
    auto_create_ecr=True,
    region=region,
    agent_name=agent_name,
)
print("Configured:", cfg, flush=True)

try:
    print(">>> launch(auto_update_on_conflict=True)", flush=True)
    launch = rt.launch(auto_update_on_conflict=True)
except ClientError as e:
    if e.response.get("Error", {}).get("Code") == "ConflictException":
        print("?? Exists; retrying with auto_update_on_conflict=True...", flush=True)
        launch = rt.launch(auto_update_on_conflict=True)
    else:
        raise
print("Launched:", launch, flush=True)

def _status(x):
    if isinstance(x, dict):
        ep = x.get("endpoint") or x
        return ep.get("status") or ep.get("Status")
    ep = getattr(x, "endpoint", None)
    if isinstance(ep, dict):
        return ep.get("status") or ep.get("Status")
    return getattr(ep, "status", None)

print(">>> polling status...", flush=True)
while True:
    s = _status(rt.status())
    print("Status:", s, flush=True)
    if s in {"READY", "CREATE_FAILED", "DELETE_FAILED", "UPDATE_FAILED"}:
        break
    time.sleep(10)

agent_id = getattr(launch, "agent_id", None) or getattr(launch, "agentRuntimeId", None)
os.makedirs("deploy", exist_ok=True)
with open("deploy/launch_result.json", "w") as f:
    json.dump(
        {"agent_id": agent_id, "agent_arn": getattr(launch, "agent_arn", None),
         "ecr_uri": getattr(launch, "ecr_uri", None), "region": region},
        f, indent=2
    )
print("Saved: deploy/launch_result.json", flush=True)
print(">>> done.", flush=True)
