# import os
# import boto3
# import json
# import requests
# from boto3.session import Session
# from bedrock_agentcore_starter_toolkit import Runtime
# from dotenv import load_dotenv

# load_dotenv()

# def require_env(name: str) -> str:
#     val = os.getenv(name)
#     if not val:
#         raise RuntimeError(f"Missing required env var: {name}")
#     return val

# def mask(s: str, keep: int = 4) -> str:
#     if not s:
#         return ""
#     return "*" * max(len(s) - keep, 0) + s[-keep:]
# # ======================================================
# # Helper: Setup Cognito Client Credentials
# # ======================================================
# def setup_cognito_user_pool():
#     boto_session = Session()
#     region = boto_session.region_name

#     # 👉 Fill in your existing Cognito values
#     pool_id = "us-east-1_XE5s66viL"   # Your existing user pool ID
#     client_id = "44vf6rtqc4c34jtr0dpp2d7rrp"
#     client_secret = "1ds4rqgas7nl0r9pdj5lome5n4pvnolvv3qf1n4crq06d4nfl96u"
#     domain = "my-domain-i22y6il8"    # Your Cognito domain prefix (App integration → Domain name)

#     try:
#         # Request a token via client credentials flow
#         token_url = f"https://{domain}.auth.{region}.amazoncognito.com/oauth2/token"
#         headers = {"Content-Type": "application/x-www-form-urlencoded"}
#         data = {
#             "grant_type": "client_credentials",
#             "client_id": client_id,
#             "client_secret": client_secret,
#         }

#         resp = requests.post(token_url, headers=headers, data=data)
#         resp.raise_for_status()
#         bearer_token = resp.json()["access_token"]

#         cognito_config = {
#             "pool_id": pool_id,
#             "client_id": client_id,
#             "client_secret": client_secret,
#             "bearer_token": bearer_token,
#             "discovery_url": f"https://cognito-idp.{region}.amazonaws.com/{pool_id}/.well-known/openid-configuration",
#             "domain": domain,
#         }

#         print(f"\n✅ Cognito setup complete (Client Credentials flow)")
#         print(f"   Pool ID: {pool_id}")
#         print(f"   Client ID: {client_id}")
#         print(f"   Bearer Token: {bearer_token}\n")

#         return cognito_config

#     except Exception as e:
#         print(f"Error retrieving Cognito token: {e}")
#         return None


# # ======================================================
# # Helper: Regenerate Bearer Token
# # ======================================================
# def get_bearer_token(client_id, client_secret, domain, region="us-east-1"):
#     token_url = f"https://{domain}.auth.{region}.amazoncognito.com/oauth2/token"
#     headers = {"Content-Type": "application/x-www-form-urlencoded"}
#     data = {
#         "grant_type": "client_credentials",
#         "client_id": client_id,
#         "client_secret": client_secret,
#     }

#     resp = requests.post(token_url, headers=headers, data=data)
#     resp.raise_for_status()
#     return resp.json()["access_token"]


# # ======================================================
# # Runtime Deployment
# # ======================================================
# if __name__ == "__main__":
#     print("Setting up Amazon Cognito user pool...")
#     cognito_config = setup_cognito_user_pool()
#     print("Cognito setup completed ✓")

#     # Example: regenerate token any time
#     fresh_token = get_bearer_token(
#         client_id=cognito_config["client_id"],
#         client_secret=cognito_config["client_secret"],
#         domain=cognito_config["domain"],
#     )
#     print(f"Fresh Bearer Token: {fresh_token}")

#     # Reuse existing execution role
#     role_name = "AgentCoreRuntimeExecutionRole"
#     iam = boto3.client("iam")
#     role = iam.get_role(RoleName=role_name)
#     execution_role_arn = role["Role"]["Arn"]
#     print(f"Using existing execution role: {execution_role_arn}")

#     boto_session = boto3.session.Session()
#     region = boto_session.region_name

#     agentcore_runtime = Runtime()

#     response = agentcore_runtime.configure(
#         entrypoint="agentcore_311.py",   # points to your agent file
#         execution_role=execution_role_arn,
#         auto_create_ecr=True,
#         requirements_file="requirements.txt",
#         region=region,
#         agent_name="cityassistant_311_agent",
#         outbound_auth_configuration={
#         "customJWTAuthorizer": {
#             # Use a fresh token whenever possible
#             "token": fresh_token,
#             "discoveryUrl": cognito_config.get("discovery_url"),
#             }
#         },
#     )

#     print("Runtime configuration completed:", response)


#!/usr/bin/env python3

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from bedrock_agentcore_starter_toolkit import Runtime

def main():
    script_dir = Path(__file__).parent
    env_file = script_dir / ".env"

    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded environment variables from {env_file}")
    else:
        raise FileNotFoundError(f"Missing .env file at {env_file}")

    runtime_name = os.getenv("AGENTCORE_RUNTIME_NAME", "cityassistant_311_agent")
    execution_role_arn = os.getenv("AGENTCORE_ROLE_ARN")
    region = os.getenv("AWS_REGION", "us-east-1")

    if not execution_role_arn:
        raise ValueError("❌ Missing AGENTCORE_ROLE_ARN in .env")

    runtime = Runtime()

    response = runtime.configure(
        agent_name=runtime_name,
        entrypoint="agentcore_311.py",    # Your agent file
        execution_role=execution_role_arn,
        auto_create_ecr=True,
        requirements_file="requirements.txt",
        region=region,
    )

    print("✅ Runtime configured successfully!")
    print(json.dumps(response, indent=2, default=str))

if __name__ == "__main__":
    main()



