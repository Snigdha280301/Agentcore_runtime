
import os
import time
import json
import re
from typing import Any, Dict, List, Annotated, TypedDict, Optional, Callable

# ----- AgentCore app -----
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ----- LangChain / LangGraph -----
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ----- MCP client (Gateway) -----
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client  # <- correct name

# ----- AWS / HTTP -----
import boto3
import requests
import anyio


# =============================
# Config & helpers
# =============================

SYSTEM_PROMPT = """
You are CityAssist, a 311-style non-emergency assistant.

YOU HAVE THESE TOOLS (via Gateway MCP) — CALL THEM WHEN RELEVANT:
- create_ticket_tool → creates a new ticket
  args: { category: string, description: string, location: string, contact_email: string }

- get_ticket_status_tool → gets the status/details for a ticket
  args: { ticket_id: string }

- search_kb_tool → searches the city knowledge base
  args: { query: string }

- send_email_tool → sends a confirmation/notice email
  args: { to_email: string, category: string, description: string, ticket_id?: string }

ROUTING RULES
- If the user gives a ticket id (like "6e63bbbe"), CALL get_ticket_status_tool exactly once with that id.
- If the user wants to report something, collect {category, description, location, contact_email} and CALL create_ticket_tool.
- For service questions, CALL search_kb_tool first; summarize briefly; offer to create a ticket if appropriate.
- Do NOT apologize for “system errors” if a tool exists; just try the tool. If a tool call fails, return a short helpful message.

OUTPUT STYLE
- Be concise and action-oriented. Do NOT show raw JSON.
- For ticket status, summarize like:
  "Ticket ID: …  Status: …  Dept: …  ETA: …  Location: …  Description: …  Last Updated: …"
- If any fields are missing, omit them gracefully.

GUARDRAILS
- If it’s an emergency: reply exactly "Call 911 now." and stop.
"""

EMERGENCY_REGEX = r"(heart attack|gun|shots fired|fire in (my|the)|unconscious|not breathing|domestic violence|break[- ]?in|armed|stabbed|car crash with injuries)"
def is_emergency(text: str) -> bool:
    return bool(re.search(EMERGENCY_REGEX, text or "", re.IGNORECASE))


GATEWAY_SECRET_NAME = os.getenv("GATEWAY_SECRET_NAME", "agentcore/cityassist311/gateway")

def _load_gateway_cfg() -> Dict[str, str]:
    env = {
        "gateway_url": os.getenv("GATEWAY_URL"),
        "token_url": os.getenv("COGNITO_TOKEN_URL"),
        "client_id": os.getenv("COGNITO_CLIENT_ID"),
        "client_secret": os.getenv("COGNITO_CLIENT_SECRET"),
    }
    if all(env.values()):
        print(f"[CFG] Loaded Gateway/Cognito config from environment.")
        return env

    sm = boto3.client("secretsmanager")
    val = sm.get_secret_value(SecretId=GATEWAY_SECRET_NAME)["SecretString"]
    cfg = json.loads(val)
    print(f"[CFG] Loaded Gateway/Cognito config from Secrets Manager: {GATEWAY_SECRET_NAME}")

    for k in ("gateway_url", "token_url", "client_id", "client_secret"):
        if not cfg.get(k):
            raise ValueError(f"Secret missing required key: {k}")
    if not cfg["token_url"].endswith("/oauth2/token"):
        raise ValueError("token_url must end with '/oauth2/token'.")
    return cfg


# ------- OAuth2: client_credentials token (cached) -------
_cached_token: Optional[str] = None
_token_expiry: float = 0.0

def _fetch_access_token(cfg: Dict[str, str]) -> str:
    global _cached_token, _token_expiry
    now = time.time()
    if _cached_token and now < _token_expiry - 60:
        return _cached_token

    r = requests.post(
        cfg["token_url"],
        data={
            "grant_type": "client_credentials",
            "client_id": cfg["client_id"],
            "client_secret": cfg["client_secret"],
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=15,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Token request failed: {r.status_code} {r.text}")
    body = r.json()
    _cached_token = body["access_token"]
    _token_expiry = now + int(body.get("expires_in", 3600))
    return _cached_token


# =============================
# MCP helpers
# =============================

class GatewayMcpClient:
    """Short-lived per-call MCP client (avoids cancel-scope/TaskGroup issues)."""
    def __init__(self, gateway_url: str):
        self.gateway_url = gateway_url

    async def _with_session(self, access_token: str, fn: Callable[[ClientSession], Any]):
        async with streamablehttp_client(
            self.gateway_url, headers={"Authorization": f"Bearer {access_token}"}
        ) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as sess:
                await sess.initialize()
                return await fn(sess)

    async def call_tool(self, access_token: str, name: str, arguments: Dict[str, Any]) -> str:
        async def _do(sess: ClientSession):
            resp = await sess.call_tool(name, arguments or {})
            parts: List[str] = []
            for item in resp.content or []:
                txt = getattr(item, "text", None) or getattr(item, "data", None) or str(item)
                if txt:
                    parts.append(str(txt))
            return "\n".join(parts) if parts else json.dumps(resp.model_dump())

        # Retry 429 a few times
        delay = 0.7
        for attempt in range(4):
            try:
                return await self._with_session(access_token, _do)
            except Exception as e:
                msg = str(e)
                if "429" in msg and attempt < 3:
                    await anyio.sleep(delay)
                    delay *= 1.8
                    continue
                raise

def _arun(coro_fn, *args, **kwargs):
    async def runner():
        return await coro_fn(*args, **kwargs)
    return anyio.run(runner)


# =============================
# Formatting the raw tool result
# =============================

def _maybe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def _normalize_tool_payload(raw: str) -> Dict[str, Any]:
    """
    Accepts whatever MCP returned and tries to normalize into a dict:
    {statusCode, body(dict)} if possible.
    """
    # 1) raw may already be JSON with statusCode/body
    j = _maybe_json_load(raw)
    if isinstance(j, dict) and "statusCode" in j:
        body = j.get("body")
        if isinstance(body, str):
            body_json = _maybe_json_load(body)
            if isinstance(body_json, dict):
                j["body"] = body_json
        return j

    # 2) raw might be just a body JSON as string
    if isinstance(j, dict):
        return {"statusCode": 200, "body": j}

    # 3) fallback: text (keep as is)
    return {"statusCode": 200, "body": {"_text": raw}}

def _format_ticket_status(body: Dict[str, Any]) -> str:
    ticket_id = body.get("ticket_id")
    status = body.get("status") or body.get("ticket_status")
    dept = body.get("dept") or body.get("department")
    eta_days = body.get("eta_days")
    updated = body.get("updated_at")
    category = body.get("category")
    desc = body.get("description")
    location = body.get("location")

    parts = []
    if ticket_id: parts.append(f"Ticket ID: {ticket_id}")
    if status:    parts.append(f"Status: {status}")
    if dept:      parts.append(f"Dept: {dept}")
    if eta_days is not None: parts.append(f"ETA: {eta_days} day(s)")
    if location:  parts.append(f"Location: {location}")
    if category:  parts.append(f"Category: {category}")
    if desc:      parts.append(f"Description: {desc}")
    if updated:   parts.append(f"Last Updated: {updated}")
    if not parts:
        # Preserve something if we have only text
        text = body.get("_text")
        if text:
            return text
        return "No details were returned."
    return "  ".join(parts)

def _format_create_ticket(body: Dict[str, Any]) -> str:
    # Attempt to output the essentials
    tid = body.get("ticket_id")
    dept = body.get("dept") or body.get("department")
    eta_days = body.get("eta_days")
    category = body.get("category")
    desc = body.get("description")
    location = body.get("location")
    parts = []
    if tid: parts.append(f"Ticket created: {tid}")
    if dept: parts.append(f"Department: {dept}")
    if eta_days is not None: parts.append(f"ETA: {eta_days} day(s)")
    if location: parts.append(f"Location: {location}")
    if category: parts.append(f"Category: {category}")
    if desc: parts.append(f"Description: {desc}")
    if not parts:
        text = body.get("_text")
        if text:
            return text
        return "Ticket created."
    return "  ".join(parts)


# =============================
# Build static tools (no discovery)
# =============================

def build_static_tools(gateway: GatewayMcpClient, cfg: Dict[str, str]) -> List[Tool]:
    """
    We bind to the known MCP tool names directly to avoid list_tools (and 429).
    """
    static = [
        ("create_ticket_tool",       "target-create-ticket__create_ticket"),
        ("get_ticket_status_tool",   "target-get-ticket-status__get_ticket_status"),
        ("search_kb_tool",           "target-knowledge-base__search_kb"),
        ("send_email_tool",          "target-email__send_email"),
    ]

    print("[TOOLS] Binding static tools:")
    for alias, name in static:
        print(f"  - {alias} → {name}")

    def _factory(alias: str, mcp_name: str):
        def _call(**kwargs):
            token = _fetch_access_token(cfg)
            print(f"[TOOLS] Calling {alias} ({mcp_name}) with args: {kwargs}")
            raw = _arun(gateway.call_tool, token, mcp_name, kwargs or {})
            print(f"[TOOLS] Raw result from {alias}: {raw!r}")

            # Pretty post-processing for the two main flows
            norm = _normalize_tool_payload(raw)
            body = norm.get("body", {})
            if alias == "get_ticket_status_tool":
                human = _format_ticket_status(body if isinstance(body, dict) else {})
                print(f"[TOOLS] Parsed ticket status: {human}")
                return human
            if alias == "create_ticket_tool":
                human = _format_create_ticket(body if isinstance(body, dict) else {})
                print(f"[TOOLS] Parsed create_ticket: {human}")
                return human

            # For search_kb / send_email or unknown, return the most readable we can
            if isinstance(body, dict) and "_text" not in body:
                try:
                    # Compact humanish dict
                    return "; ".join(f"{k}: {v}" for k, v in body.items())
                except Exception:
                    pass
            return body.get("_text") or raw
        return _call

    tools: List[Tool] = []
    for alias, name in static:
        tools.append(Tool(name=alias, description=f"Gateway tool bound to {name}", func=_factory(alias, name)))
    return tools


# =============================
# LangGraph app
# =============================

class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def build_graph(lc_tools: List[Tool]):
    llm = ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"),
        model_kwargs={"temperature": float(os.getenv("MODEL_TEMPERATURE", "0.0"))},
    )
    llm_tools = llm.bind_tools(lc_tools)

    def chatbot(state: ChatState):
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        ai = llm_tools.invoke(msgs)
        return {"messages": [ai]}

    def needs_tools(state: ChatState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else "end"

    g = StateGraph(ChatState)
    g.add_node("chatbot", chatbot)
    g.add_node("tools", ToolNode(lc_tools))
    g.add_conditional_edges("chatbot", needs_tools, {"tools": "tools", "end": END})
    g.add_edge("tools", "chatbot")
    g.set_entry_point("chatbot")
    return g.compile()


# =============================
# AgentCore app entrypoint
# =============================

app = BedrockAgentCoreApp()

_cfg = _load_gateway_cfg()
_gateway = GatewayMcpClient(_cfg["gateway_url"])
_lc_tools = build_static_tools(_gateway, _cfg)  # <- static binding (no discovery)
_graph = build_graph(_lc_tools)


@app.entrypoint
def invoke(payload: Dict[str, Any]):
    user_input = (payload or {}).get("prompt") or (payload or {}).get("inputText") or ""
    user_input = (user_input or "").strip()
    if not user_input:
        return "Please provide a message."
    if is_emergency(user_input):
        return "Call 911 now."

    out = _graph.invoke({"messages": [HumanMessage(content=user_input)]})
    for m in reversed(out["messages"]):
        if getattr(m, "role", None) == "assistant" or getattr(m, "type", None) == "ai":
            return m.content
    return "Sorry, I didn’t catch that."


if __name__ == "__main__":
    app.run()
