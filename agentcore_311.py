from bedrock_agentcore.runtime import BedrockAgentCoreApp
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage, ToolMessage
from langchain_aws import ChatBedrock
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from typing import TypedDict, List, Annotated
import os
import re
import asyncio
import json
import boto3
import random
import time
from botocore.exceptions import ClientError
from dotenv import load_dotenv
import httpx

# ======================================================
# Load secrets from Secrets Manager (fallback to .env)
# ======================================================
def get_secret():
    secret_name = os.getenv("SECRET_NAME", "cityassist_311_env")
    region_name = os.getenv("AWS_REGION", "us-east-1")

    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=region_name)

    try:
        get_secret_value_response = client.get_secret_value(SecretId=secret_name)
        secret = get_secret_value_response["SecretString"]
        secrets_dict = json.loads(secret)

        # Inject into environment
        for key, value in secrets_dict.items():
            os.environ[key] = value

        print(f"✅ Loaded secrets from AWS Secrets Manager: {secret_name}")
        return "secrets_manager"
    except ClientError as e:
        print(f"⚠️ Could not load from Secrets Manager ({e}), falling back to .env")
        load_dotenv()
        return "dotenv"

# Call at startup
secrets_source = get_secret()

# ======================================================
# System prompt
# ======================================================
SYSTEM_PROMPT = """
You are CityAssist, a city 311-style non-emergency assistant.  
You MUST always follow the rules below.  

INTENTS → TOOLS
- Report an issue → call target-create-ticket___create_ticket
- Check ticket status → call target-get-ticket-status___get_ticket_status
- Ask about city services → call target-knowledge-base___search_kb

ENTRY POINTS
- A user may begin by reporting an issue, asking for ticket status, or asking about city services.  
- Do NOT assume the flow always starts with "report an issue."  
- Detect the intent and follow the correct flow immediately.

REPORTING
- To create ANY ticket, you must collect exactly FOUR fields, one at a time:
  1. Category (pothole, graffiti, streetlight outage, trash, etc.)
  2. Description (any free-text description is acceptable)
  3. Location (address, landmark, or simple text)
  4. Contact Email
- If the user has provided ALL FOUR fields, you MUST immediately call target-create-ticket___create_ticket.  
- If the user confirms with “yes”, “correct”, “ok”, or similar after you summarize the collected info, you MUST call target-create-ticket___create_ticket with those fields.  
- After creating a ticket, return the ticket_id and ETA. Never restart the conversation.

STATUS
- If the user provides a ticket ID (8 characters), call target-get-ticket-status___get_ticket_status and summarize status, ETA, and department.  
- If no ticket ID is provided, ask for it briefly.  
- After answering, do not reset the conversation — allow follow-ups.

KNOWLEDGE BASE
- For service questions, you MUST call target-knowledge-base___search_kb with the user’s exact text. Do NOT answer from your own knowledge.  
- Provide a concise, general answer from the KB.  
- Then IMMEDIATELY offer to create a ticket. If the user agrees, collect ONLY the missing fields and then call target-create-ticket___create_ticket.

GUARDRAILS
- If the request suggests an emergency, respond: "Call 911 now." Do not call any tools.  
- Always be concise, clear, and action-oriented.  
- Never ask for a phone number.  
- Never loop or repeat the same question more than once. If a user ignores a missing field, explain why it’s required and stop.  

RULES
- Always call AT MOST ONE tool per user turn.  
- After calling a tool, ALWAYS wait for its response before making another call.  
- Never call a tool more than once in a row without a user message in between.  
- Never reset to a greeting once information has been collected. Continue until ticket creation, status lookup, or KB answer is complete.  
- Do not invent or assume any missing values—always explicitly ask the user.  
- If a tool call fails or returns an error, summarize the error in plain English to the user instead of retrying automatically.  
- Prioritize completing the current flow (reporting, status, or KB) before switching intents.  

"""

# ======================================================
# Emergency detection
# ======================================================
EMERGENCY_REGEX = r"(heart attack|gun|shots fired|fire in (my|the)|unconscious|not breathing|domestic violence|break[- ]?in|armed|stabbed|car crash with injuries)"
def is_emergency(text: str) -> bool:
    return bool(re.search(EMERGENCY_REGEX, text or "", re.IGNORECASE))

# ======================================================
# LangGraph State
# ======================================================
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# ======================================================
# Async-safe backoff for throttling
# ======================================================
async def async_invoke_with_backoff(fn, *args, retries=5, base_delay=5.0, **kwargs):
    for attempt in range(retries):
        try:
            return await fn(*args, **kwargs)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                retry_after = e.response.headers.get("Retry-After")
                delay = int(retry_after) if retry_after else base_delay * (2 ** attempt)
                print(f"⚠️ 429 Too Many Requests. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                raise
        except Exception as e:
            if "ThrottlingException" in str(e):
                delay = base_delay * (2 ** attempt)
                print(f"⚠️ Throttled. Retrying in {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                raise
    raise RuntimeError("Max retries exceeded")

# ======================================================
# Tool caching + retry
# ======================================================
_cached_tools = None

async def fetch_tools_with_retry(session, retries=5, base_delay=5.0):
    global _cached_tools
    attempt = 0
    while attempt < retries:
        try:
            tools = await load_mcp_tools(session)
            print(f"✅ Loaded {len(tools)} tools from Gateway")
            for t in tools:
                print(f"   - {t.name}: {t.description}")
            _cached_tools = tools
            return tools
        except Exception as e:
            attempt += 1
            delay = base_delay * (2 ** attempt)
            print(f"❌ Failed to load tools (attempt {attempt}/{retries}): {e}")
            if attempt >= retries:
                raise
            print(f"⏳ Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

# ======================================================
# Safe ToolNode wrapper with debug logs
# ======================================================
class SafeToolNode(ToolNode):
    def __init__(self, tools):
        super().__init__(tools)
        self._tool_names = {t.name for t in tools}

    def invoke(self, state: ChatState, config=None):
        last_msg = state["messages"][-1]
        if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
            for tc in last_msg.tool_calls:
                tool_name = tc.get("name") if isinstance(tc, dict) else tc.name
                tool_args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                print(f"🔧 LLM requested tool: {tool_name} with args: {tool_args}")

                if tool_name not in self._tool_names:
                    print(f"⚠️ Tool not found: {tool_name}")
                    return {
                        "messages": [
                            AIMessage(
                                content=f"⚠️ Tool not found: {tool_name}. Please check tool configuration.",
                                role="assistant",
                            )
                        ]
                    }
        return super().invoke(state, config)

def normalize_anthropic_response(ai: AIMessage) -> AIMessage:
    if ai.tool_calls:
        return ai
    tool_calls = []
    for item in ai.additional_kwargs.get("content", []):
        if isinstance(item, dict) and item.get("type") == "tool_use":
            tool_calls.append({
                "id": item.get("id"),
                "name": item.get("name"),
                "args": item.get("input", {})
            })
    if tool_calls:
        ai.tool_calls = tool_calls
    return ai

# ======================================================
# Build LangGraph graph (lazy init)
# ======================================================
graph = None  # lazy init

async def build_graph():
    llm = ChatBedrock(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        model_kwargs={"temperature": 0.0},
    )

    GATEWAY_URL = os.getenv("GATEWAY_URL")
    GATEWAY_ACCESS_TOKEN = os.getenv("GATEWAY_ACCESS_TOKEN")
    if not GATEWAY_URL or not GATEWAY_ACCESS_TOKEN:
        raise RuntimeError("❌ Missing GATEWAY_URL or GATEWAY_ACCESS_TOKEN")

    print(f"🔌 Attempting to connect to Gateway at {GATEWAY_URL}...")

    async with streamablehttp_client(
        GATEWAY_URL, headers={"Authorization": f"Bearer {GATEWAY_ACCESS_TOKEN}"}
    ) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()

            global _cached_tools
            if _cached_tools is None:
                tools = await fetch_tools_with_retry(session)
            else:
                tools = _cached_tools
                print("♻️ Reusing cached tools")

            llm_tools = llm.bind_tools(tools, tool_choice="auto")

            def chatbot(state: ChatState):
                msgs = state["messages"]
                if not msgs or not isinstance(msgs[0], SystemMessage):
                    msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
                ai = llm_tools.invoke(msgs)
                ai = normalize_anthropic_response(ai)   # 🔑 normalize anthropic tool output
                return {"messages": [ai]}


            def should_use_tools(state: ChatState) -> str:
                last_msg = state["messages"][-1]
                if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
                    return "tools"
                return "end"

            g = StateGraph(ChatState)
            g.add_node("chatbot", chatbot)
            g.add_node("tools", SafeToolNode(tools))
            g.set_entry_point("chatbot")

            g.add_conditional_edges("chatbot", should_use_tools, {"tools": "tools", "end": END})
            g.add_edge("tools", "chatbot")

            return g.compile()

# ======================================================
# AgentCore Runtime App
# ======================================================
app = BedrockAgentCoreApp()

@app.entrypoint
def invoke(payload):
    global graph
    user_input = (payload or {}).get("prompt") or (payload or {}).get("message")

    if is_emergency(user_input):
        return "This sounds like an emergency. Please call 911 immediately."

    if graph is None:
        graph = asyncio.run(build_graph())

    out = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    messages = out["messages"]

    for m in messages:
        if isinstance(m, ToolMessage):
            print(f"📥 Tool response from {m.name}: {m.content}")
        if isinstance(m, AIMessage):
            print(f"🤖 Final AI reply: {m.content}")
            return m.content

    return "Sorry, I didn’t catch that."

if __name__ == "__main__":
    print(f"🔑 Secrets loaded from: {secrets_source}")
    app.run()