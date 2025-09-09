import os, json, time, re, logging
from typing import Any, Dict, List, Annotated, TypedDict, Optional

# ----- AgentCore runtime -----
from bedrock_agentcore.runtime import BedrockAgentCoreApp

# ----- LangChain / LangGraph -----
from langchain_aws import ChatBedrock
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.tools import Tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

# ----- MCP HTTP client -----
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

# ----- AWS / HTTP / Async -----
import boto3
import requests
import anyio


# =============================
# Logging
# =============================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
log = logging.getLogger("cityassist311")


# =============================
# System Prompt
# =============================
SYSTEM_PROMPT = """
You are CityAssist, a city 311-style non-emergency assistant.  

INTENTS → TOOLS
- Report an issue → call create_ticket_tool
- Check ticket status → call get_ticket_status_tool
- Ask about city services → call search_kb_tool

REPORTING
- To create ANY ticket, collect exactly FOUR fields, one at a time:
  1) Category
  2) Description
  3) Address (street address or landmark)
  4) Contact Email
- Once all are provided, call create_ticket_tool and return confirmation.

STATUS
- If user provides an 8-char ticket ID, call get_ticket_status_tool and summarize status.
- If no ID, ask briefly.

KNOWLEDGE BASE
- For service questions, call search_kb_tool with user’s text. Summarize and offer ticket creation.

GUARDRAILS
- If it’s an emergency: reply "Call 911 now." and stop.
- One tool per user turn, wait for tool response before another.
- Never loop endlessly. If a required field is ignored, explain why it’s required and stop.
- Never reset to greeting mid-convo.
"""


# =============================
# Helpers
# =============================
_EMERG_RX = r"(heart attack|gun|fire|shots fired|unconscious|stabbed|domestic violence|car crash)"
_HEXLIKE = re.compile(r"^[0-9a-fA-F-]{6,64}$")

def is_emergency(text: str) -> bool:
    return bool(re.search(_EMERG_RX, text or "", re.IGNORECASE))

def load_gateway_cfg() -> Dict[str, str]:
    secret_name = os.getenv("GATEWAY_SECRET_NAME", "agentcore/cityassist311/gateway")
    cfg = {
        "gateway_url": os.getenv("GATEWAY_URL"),
        "token_url": os.getenv("COGNITO_TOKEN_URL"),
        "client_id": os.getenv("COGNITO_CLIENT_ID"),
        "client_secret": os.getenv("COGNITO_CLIENT_SECRET"),
    }
    if all(cfg.values()):
        log.info("[CFG] Loaded Gateway/Cognito config from environment.")
        return cfg
    sm = boto3.client("secretsmanager")
    val = sm.get_secret_value(SecretId=secret_name)["SecretString"]
    cfg = json.loads(val)
    log.info(f"[CFG] Loaded Gateway/Cognito config from Secrets Manager: {secret_name}")
    return cfg


# =============================
# OAuth2 token cache
# =============================
_token: Optional[str] = None
_expiry: float = 0.0

def fetch_token(cfg: Dict[str, str]) -> str:
    global _token, _expiry
    now = time.time()
    if _token and now < _expiry - 60:
        return _token
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
    r.raise_for_status()
    body = r.json()
    _token, _expiry = body["access_token"], now + int(body.get("expires_in", 3600))
    log.info("[AUTH] Obtained new access token; expires_in=%ss", body.get("expires_in", 3600))
    return _token


# =============================
# MCP Gateway Client
# =============================
class GatewayMcpClient:
    def __init__(self, url: str):
        self.url = url

    async def _with_session(self, token: str, fn):
        async with streamablehttp_client(self.url, headers={"Authorization": f"Bearer {token}"}) as (r, w, _):
            async with ClientSession(r, w) as sess:
                await sess.initialize()
                return await fn(sess)

    async def call_tool(self, token: str, name: str, args: Dict[str, Any]):
        async def _run(sess: ClientSession):
            resp = await sess.call_tool(name, args or {})
            parts = [getattr(c, "text", None) or str(c) for c in resp.content or []]
            return "\n".join(p for p in parts if p)

        delay = 0.5
        max_attempts = 4
        for attempt in range(1, max_attempts + 1):
            try:
                log.info("[MCP] Calling tool '%s' with args=%s (attempt %d/%d)", name, args, attempt, max_attempts)
                result = await self._with_session(token, _run)
                log.info("[MCP] Tool '%s' succeeded on attempt %d", name, attempt)
                return result
            except Exception as e:
                s = str(e)
                if "429" in s and attempt < max_attempts:
                    log.warning("[MCP][429] Tool '%s' throttled; retrying in %.2fs (attempt %d/%d)",
                                name, delay, attempt, max_attempts - 1)
                    await anyio.sleep(delay)
                    delay *= 2
                    continue
                log.error("[MCP] Tool '%s' failed: %s", name, s)
                raise


def run_async(coro, *a, **kw):
    return anyio.run(lambda: coro(*a, **kw))


# =============================
# Tool arg normalization
# =============================
def merge_and_normalize_args(positional: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge positional + kwargs and normalize Anthropic '__arg1'.
    Heuristic:
      - If __arg1 looks hex-like, treat as 'ticket_id'
      - Else treat as 'query'
    """
    args = dict(kwargs or {})
    if positional is None:
        pass
    elif isinstance(positional, dict):
        args = {**positional, **args}
    else:
        args = {"__arg1": positional, **args}

    if "__arg1" in args and not any(k in args for k in ("ticket_id", "query", "category")):
        v = args.pop("__arg1")
        if isinstance(v, str) and _HEXLIKE.match(v):
            args["ticket_id"] = v
        else:
            args["query"] = v
    return args


# =============================
# Response Formatters
# =============================
def format_create_ticket(data: Dict[str, Any]) -> str:
    # Per tool outputSchema: ticket_id, status, eta_days (others may be absent)
    return (
        "✅ Ticket Created\n"
        f"- Ticket ID: {data.get('ticket_id','N/A')}\n"
        f"- Status: {data.get('status','N/A')}\n"
        f"- ETA: {data.get('eta_days','N/A')} day(s)"
    )

def format_ticket_status(data: Dict[str, Any]) -> str:
    return (
        "📋 Ticket Status\n"
        f"- Ticket ID: {data.get('ticket_id','N/A')}\n"
        f"- Status: {data.get('status') or data.get('ticket_status','N/A')}\n"
        f"- Department: {data.get('dept') or data.get('department','N/A')}\n"
        f"- Category: {data.get('category','N/A')}\n"
        f"- Last Updated: {data.get('updated_at','N/A')}\n"
        f"- ETA: {data.get('eta_days','N/A')} day(s)"
    )

def format_search_kb(data: Dict[str, Any]) -> str:
    if not isinstance(data, dict):
        return str(data)
    ans = data.get("answer")
    src = data.get("source")
    if ans and src:
        return f"{ans}\n\n— Source: {src}"
    if ans:
        return ans
    return data.get("error") or "No answer found in the knowledge base."

def format_send_email(data: Dict[str, Any]) -> str:
    if isinstance(data, dict) and data.get("message"):
        return f"📧 {data['message']}"
    return "📧 Email request processed."


# =============================
# Static Tools → LangChain
# =============================
def build_static_tools(client: GatewayMcpClient, cfg: Dict[str, str]) -> List[Tool]:
    static = [
        ("create_ticket_tool",     "target-create-ticket___create_ticket"),
        ("get_ticket_status_tool", "target-get-ticket-status___get_ticket_status"),
        ("search_kb_tool",         "target-knowledge-base___search_kb"),
        ("send_email_tool",        "target-email___send_email"),
    ]
    for alias, mcp in static:
        log.info("[TOOLS] Binding %s → %s", alias, mcp)

    def _factory(alias: str, mcp_name: str):
        def _call(input: Any = None, **kwargs):
            args = merge_and_normalize_args(input, kwargs)

            # ----- PER-TOOL ARG NORMALIZATION & VALIDATION -----
            if alias == "create_ticket_tool":
                # Aliases: allow 'location'/'email' from the LLM
                if "address" not in args and "location" in args:
                    args["address"] = args.pop("location")
                if "contact_email" not in args and "email" in args:
                    args["contact_email"] = args.pop("email")

                required = ("category", "description", "address", "contact_email")
                missing = [k for k in required if not args.get(k)]
                if missing:
                    return f"To create a ticket I still need: {', '.join(missing)}."

            elif alias == "get_ticket_status_tool":
                if "ticket_id" not in args and "id" in args:
                    args["ticket_id"] = args.pop("id")
                if not args.get("ticket_id"):
                    return "Please provide your 8-character ticket ID."

            elif alias == "search_kb_tool":
                if not args.get("query"):
                    return "Tell me what to search for (e.g., 'missed trash pickup')."

            elif alias == "send_email_tool":
                # Normalize 'to_email'
                if "to_email" not in args:
                    if "email" in args:
                        args["to_email"] = args.pop("email")
                    elif "contact_email" in args:
                        args["to_email"] = args.pop("contact_email")
                if "status" not in args and "ticket_status" in args:
                    args["status"] = args.pop("ticket_status")

                required = ("to_email", "ticket_id", "category", "description", "status")
                missing = [k for k in required if not args.get(k)]
                if missing:
                    return f"To send the email I need: {', '.join(missing)}."
            # ---------------------------------------------------

            tok = fetch_token(cfg)
            raw = run_async(client.call_tool, tok, mcp_name, args or {})

            # Try JSON parse for pretty formatting
            body = {}
            try:
                if isinstance(raw, str) and raw.strip().startswith("{"):
                    body = json.loads(raw)
            except Exception:
                pass

            if alias == "create_ticket_tool" and body:
                return format_create_ticket(body)
            if alias == "get_ticket_status_tool" and body:
                return format_ticket_status(body)
            if alias == "search_kb_tool" and body:
                return format_search_kb(body)
            if alias == "send_email_tool" and body:
                return format_send_email(body)

            return raw
        return _call

    return [Tool(name=alias, description=f"Gateway tool bound to {mcp}", func=_factory(alias, mcp))
            for alias, mcp in static]


# =============================
# LangGraph (LLM + Tools)
# =============================
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

def build_graph(tools: List[Tool]):
    llm = ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL_ID", "us.anthropic.claude-3-7-sonnet-20250219-v1:0"),
        model_kwargs={"temperature": 0.0},
    ).bind_tools(tools)

    def chatbot(state: ChatState):
        msgs = state["messages"]
        if not msgs or not isinstance(msgs[0], SystemMessage):
            msgs = [SystemMessage(content=SYSTEM_PROMPT)] + msgs
        return {"messages": [llm.invoke(msgs)]}

    def router(state: ChatState):
        return "tools" if getattr(state["messages"][-1], "tool_calls", None) else "end"

    g = StateGraph(ChatState)
    g.add_node("chatbot", chatbot)
    g.add_node("tools", ToolNode(tools))
    g.add_conditional_edges("chatbot", router, {"tools": "tools", "end": END})
    g.add_edge("tools", "chatbot")
    g.set_entry_point("chatbot")
    return g.compile()


# =============================
# Entrypoint for Runtime
# =============================
app = BedrockAgentCoreApp()
_cfg = load_gateway_cfg()
_client = GatewayMcpClient(_cfg["gateway_url"])
_tools = build_static_tools(_client, _cfg)
_graph = build_graph(_tools)

@app.entrypoint
def invoke(payload: Dict[str, Any]):
    text = (payload or {}).get("prompt") or (payload or {}).get("inputText") or ""
    text = (text or "").strip()
    if not text:
        return "Please provide a message."
    if is_emergency(text):
        return "Call 911 now."

    out = _graph.invoke({"messages": [HumanMessage(content=text)]})

    # Return the assistant/AI message content robustly
    for m in reversed(out["messages"]):
        if getattr(m, "role", None) == "assistant" or getattr(m, "type", None) == "ai":
            return m.content

    return "Sorry, I didn’t catch that."


if __name__ == "__main__":
    app.run()
