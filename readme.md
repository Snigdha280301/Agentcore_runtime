# LangGraph + Bedrock Agent

This project demonstrates how to build and deploy an **AI Agent** using **LangGraph**, **Amazon Bedrock**, and **AgentCore Runtime**.

---

## 📂 Project Structure

```
langgraph-bedrock-agent/
│── langgraph_bedrock.py          # Main agent code (with BedrockAgentCoreApp)
│── requirements.txt               # Dependencies
│── README.md                      # Project instructions (this file)
│
├── deploy/                        # Deployment scripts
    ├── deploy_runtime.py          # Configure and launch agent in AgentCore
    └── cleanup_runtime.py         # Delete runtime + ECR repo
    

```

---

## ⚡ Requirements

- Python 3.10+
- AWS CLI configured (`aws configure`)
- Amazon Bedrock access enabled for your account
- Model access granted (Claude, Mistral, etc.)

---

## 🚀 Setup

1. Clone this repo:

   ```bash
   git clone <your-repo>
   cd langgraph-bedrock-agent
   ```

2. Create and activate virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Mac/Linux
   .venv\Scripts\activate    # Windows
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Run Locally

You can start the agent locally:

```bash
python langgraph_bedrock.py
```

It will run using the **BedrockAgentCoreApp** runtime locally.

You can test with payloads like:

```python
from langgraph_bedrock import langgraph_bedrock
print(langgraph_bedrock({"prompt": "How much is 2+2?"}))
```

---

## ☁️ Deploy to AgentCore

### Step 1: Configure runtime

Run:

```bash
python deploy/deploy_runtime.py
```

This will:

- Package your code
- Create an ECR repo
- Create an execution role
- Launch an AgentCore runtime in your AWS account

### Step 2: Monitor status

The script waits until the runtime reaches **READY** status.

### Step 3: Invoke agent

Once deployed, you can call your agent:

```python
from bedrock_agentcore_starter_toolkit import Runtime
runtime = Runtime()
response = runtime.invoke({"prompt": "What is the weather today?"})
print(response)
```

---

## 🧹 Cleanup

To delete the runtime and ECR repo:

```bash
python deploy/cleanup_runtime.py
```

This ensures you don’t leave extra AWS resources running.

---

## ✅ Notes

- Make sure your IAM user/role has the following permissions:
  - `bedrock:InvokeModel`
  - `bedrock:InvokeModelWithResponseStream`
  - `bedrock:ListFoundationModels`
  - `bedrock-agentcore:*`
  - `ecr:*`
  - `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:PassRole`
  - `logs:*`
- Enable specific foundation models (Claude, Mistral, etc.) in the **AWS Console → Bedrock → Model access**.

---
