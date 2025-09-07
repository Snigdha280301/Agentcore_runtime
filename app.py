import streamlit as st
import boto3
import json
import uuid

# --- AWS Bedrock Agent Client ---
client = boto3.client("bedrock-agentcore")

# --- Minimal page setup ---
st.set_page_config(page_title="CityAssist 311", layout="centered")
st.title("CityAssist 311 🧭")

# --- Session state ---
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Show history ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat input ---
placeholder = "Report an issue, check a ticket, or ask about city services…"
if prompt := st.chat_input(placeholder):
    # user bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # call runtime
    agent_runtime_arn = "arn:aws:bedrock-agentcore:us-east-1:356225522107:runtime/cityassistant_311_agent_3-38spwz8TKm"
    try:
        payload = {
            "prompt": prompt,
            "session_id": st.session_state.session_id,  # keep multi-turn memory
        }
        resp = client.invoke_agent_runtime(
            agentRuntimeArn=agent_runtime_arn,
            runtimeUserId=st.session_state.user_id,
            runtimeSessionId=st.session_state.session_id,
            payload=json.dumps(payload).encode("utf-8"),
        )

        raw = resp["response"].read().decode("utf-8")

        # IMPORTANT: decode JSON if the runtime returned a JSON string (e.g. "\"Hello\\nWorld\"")
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, str):
                text = parsed
            elif isinstance(parsed, dict) and "content" in parsed:
                text = parsed["content"]
            else:
                text = raw
        except json.JSONDecodeError:
            text = raw

        # assistant bubble
        with st.chat_message("assistant"):
            st.markdown(text)
        st.session_state.messages.append({"role": "assistant", "content": text})

    except Exception as e:
        with st.chat_message("assistant"):
            st.error(f"Error: {e}")
