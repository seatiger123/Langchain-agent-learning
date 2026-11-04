import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing_extensions import Annotated,TypedDict
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from agent import llm, tools, tool_map

class AgentState(TypedDict):
    messages: Annotated[list,add_messages]

llm_with_tools = llm.bind_tools(tools)

def call_model(state: AgentState):
    ai_msg = llm_with_tools.invoke(state["messages"])
    return {"messages": [ai_msg]}

def call_tools(state: AgentState):
    last = state["messages"][-1]
    results = []
    for call in last.tool_calls:
        result = tool_map[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=result,tool_call_id=call["id"]))
    return {"messages":results}

def should_continue(state: AgentState):
    if state["messages"][-1].tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)
graph.add_node("model", call_model)
graph.add_node("tools",call_tools)
graph.add_edge(START,"model")
graph.add_conditional_edges("model", should_continue, ["tools",END])
graph.add_edge("tools", "model")
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)    
print(app.get_graph().draw_mermaid())

config = {"configurable": {"thread_id":"user-b"}}
while True:
    question = input("你 ( q退出 ) ：")
    if question.strip() == "q":
        break
    result = app.invoke({"messages":[HumanMessage(content=question)]},config)
    print("AI:",result["messages"][-1].content)