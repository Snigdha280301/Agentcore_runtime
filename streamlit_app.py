# # app.py  — Streamlit frontend for AgentCore Runtime
# # (no local agent, no local tools)

# import warnings
# warnings.filterwarnings("ignore", message=r".*LangChain agents will continue to be supported.*")

# import os
# import json
# import re
# import streamlit as st
# from bedrock_agentcore_starter_toolkit import Runtime

# # -------------------- Config --------------------
# st.set_page_config(page_title="CityAssist 311", page_icon="🏙️", layout="centered")
# st.title("CityAssist 311 🏙️")

# # Load agent runtime info produced by deploy/deploy_runtime.py
# LAUNCH_JSON = os.path.join("deploy", "launch_result.json")
# with open(LAUNCH_JSON, "r") as f:
#     _lr = json.load(f)

# AGENT_ID = _lr["agent_id"]
# AWS_REGION = _lr.get("region") or os.getenv("AWS_REGION", "us-east-1")

# rt = Runtime(region=AWS_REGION)

# # (Optional) simple client-side emergency short-circuit
# _EMERG = r"(heart attack|gun|shots fired|fire in (my|the)|unconscious|not breathing|domestic violence|break[- ]?in|armed|stabbed|car crash with injuries)"
# def is_emergency(text: str) -> bool:
#     return bool(re.search(_EMERG, text or "", re.IGNORECASE))

# # -------------------- Session State --------------------
# if "history" not in st.session_state:
#     greeting = (
#         "Hi! 👋 **Welcome to 311 City Services.** How can I help you today?\n\n"
#         "• **Report an issue** (pothole, streetlight out, missed trash…)\n\n"
#         "• **Check ticket status** (paste your 8-character ticket ID)\n\n"
#         "• **Ask about city services** (missed trash, noise, potholes, streetlight etc.)\n\n"
#     )
#     st.session_state.history = [("assistant", greeting)]

# # -------------------- Render History --------------------
# for role, msg in st.session_state.history:
#     with st.chat_message(role):
#         st.markdown(msg)

# # -------------------- Chat Input --------------------
# prompt = st.chat_input("Report an issue, check a ticket, or ask about city services…")
# if prompt:
#     # Show user message
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     st.session_state.history.append(("user", prompt))

#     # Optional client-side emergency check
#     if is_emergency(prompt):
#         reply = "This sounds like an emergency. Please call **911** immediately."
#     else:
#         # Invoke your AgentCore Runtime (SigV4 via your AWS creds).
#         # If later you add Cognito inbound auth for user-scoped APIs (e.g., Gmail),
#         # you could forward an ID token via identity_token=... here.
#         try:
#             resp = rt.invoke(agent_id=AGENT_ID, payload={"prompt": prompt})
#             # resp may be a string or a dict depending on model/agent setup
#             if isinstance(resp, str):
#                 reply = resp
#             else:
#                 reply = resp.get("output") or resp.get("content") or json.dumps(resp)
#         except Exception as e:
#             reply = f"Sorry, I hit an error talking to the runtime: `{e}`"

#     # Show assistant message
#     with st.chat_message("assistant"):
#         st.markdown(reply)
#     st.session_state.history.append(("assistant", reply))
# streamlit_app.py
import os, json, uuid
import streamlit as st
import boto3
from botocore.config import Config

REGION = os.getenv("AWS_REGION", "us-east-1")
AGENT_RUNTIME_ARN = "arn:aws:bedrock-agentcore:us-east-1:356225522107:runtime/cityassist311-TwtVcB8DOk"
QUALIFIER = "DEFAULT"   # optional; omit if you didn't version it

client = boto3.client(
    "bedrock-agentcore",
    region_name=REGION,
    config=Config(read_timeout=30, retries={"max_attempts": 2})
)

st.set_page_config(page_title="CityAssist 311", layout="centered")
st.title("CityAssist 311 🧭")

if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

placeholder = "Report an issue, check a ticket, or ask about city services…"
if prompt := st.chat_input(placeholder):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        payload = {"prompt": prompt, "session_id": st.session_state.session_id}

        kwargs = dict(
            agentRuntimeArn=AGENT_RUNTIME_ARN,
            runtimeUserId=st.session_state.user_id,
            runtimeSessionId=st.session_state.session_id,
            payload=json.dumps(payload).encode("utf-8"),
            contentType="application/json",
            accept="application/json",
        )
        # If your runtime shows a versioned endpoint, include QUALIFIER; otherwise skip it.
        if QUALIFIER:
            kwargs["qualifier"] = QUALIFIER

        resp = client.invoke_agent_runtime(**kwargs)
        raw = resp["response"].read().decode("utf-8", errors="replace").strip()

        try:
            parsed = json.loads(raw)
            text = parsed if isinstance(parsed, str) else parsed.get("content", raw)
        except json.JSONDecodeError:
            text = raw

        with st.chat_message("assistant"):
            st.markdown(text)
        st.session_state.messages.append({"role": "assistant", "content": text})

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")
