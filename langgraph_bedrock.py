from langgraph.graph import StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from bedrock_agentcore.runtime import BedrockAgentCoreApp
import operator, math

app = BedrockAgentCoreApp()

@tool
def calculator(expression: str) -> str:
    """
    Calculate the result of a mathematical expression.
    """
    try:
        safe_dict = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "sum": sum, "pow": pow,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log10": math.log10, "exp": math.exp,
            "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
            "degrees": math.degrees, "radians": math.radians,
            "add": operator.add, "sub": operator.sub,
            "mul": operator.mul, "truediv": operator.truediv,
        }
        result = eval(expression, safe_dict)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def weather():
    """Get weather"""  # Dummy implementation
    return "sunny"


def create_agent():
    from langchain_aws import ChatBedrock
    llm = ChatBedrock(
        model_id="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
        model_kwargs={"temperature": 0.1}
    )
    tools = [calculator, weather]
    llm_with_tools = llm.bind_tools(tools)

    def chatbot(state: MessagesState):
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content="You're a helpful assistant.")] + messages
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools))
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.set_entry_point("chatbot")
    return graph_builder.compile()

agent = create_agent()

@app.entrypoint
def langgraph_bedrock(payload):
    user_input = payload.get("prompt")
    response = agent.invoke({"messages": [HumanMessage(content=user_input)]})
    return response["messages"][-1].content

if __name__ == "__main__":
    app.run()
