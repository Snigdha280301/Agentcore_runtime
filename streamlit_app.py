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

# ---------------- Session state ----------------
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    # preload assistant greeting with service info
    greeting = (
        "👋 **Welcome to CityAssist 311!**\n\n"
        "Here’s what I can help you with:\n\n"
        "• **Report an issue** (pothole, graffiti, broken streetlight, trash pickup…)\n"
        "• **Check ticket status** (give me your 8-character ticket ID)\n"
        "• **Ask about city services** (e.g., noise complaints, parking, sanitation)\n\n"
        "Just type below to get started!"
    )
    st.session_state.messages = [{"role": "assistant", "content": greeting}]

# ---------------- Render chat history ----------------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------------- Chat input ----------------
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
