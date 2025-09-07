
## ⚙️ Prerequisites
1. Python 3.11+
2. AWS CLI configured (`aws configure`)
3. Deployed **AgentCore Runtime** on AWS (with correct IAM roles).
4. Secrets stored in **AWS Secrets Manager**:
   - `gateway_url`
   - `token_url`
   - `client_id`
   - `client_secret`

---

##  Setup Instructions

### Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Set Environment Variables

Either export locally:
```bash
export GATEWAY_SECRET_NAME="agentcore/cityassist311/gateway"
export BEDROCK_MODEL_ID="us.anthropic.claude-3-7-sonnet-20250219-v1:0"
```
Or update AWS Secrets Manager with these values.

### Run Locally
```bash
python langgraph_bedrock.py
```
```bash
python .\deploy\deploy_runtime.py  
```
```bash
agentcore launch        
```

## To check logs on cli :
```bash
aws logs tail /aws/bedrock-agentcore/runtimes/cityassist311-TwtVcB8DOk-DEFAULT --since 5m --follow
```

## How It Works
1. **System Prompt + User Input** → Sent to LLM (Claude on Bedrock).
2. **LangChain** → Decides if a tool is needed based on system prompt + user query.
3. **LangGraph** → Routes the request through a `ToolNode`.
4. **MCP Client** → Connects to Gateway via Streamable HTTP with Cognito token.
5. **Gateway** → Invokes the correct backend Lambda tool.
6. **Response** → Returned to the user.

---

## 🔍 Logs & Debugging
- Use **CloudWatch Logs** to see which Lambda tool was invoked.
- LangGraph prints available tools at cold start.
- Add more `print()` statements in `langgraph_bedrock.py` for debugging.

---


