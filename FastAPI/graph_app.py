
from typing_extensions import Annotated,TypedDict
from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

from agent import llm, tools, tool_map, DANGEROUS_TOOLS
import sqlite3
from pathlib import Path
from langgraph.checkpoint.sqlite import SqliteSaver

DB_FILE = Path(__file__).parent / "checkpoints.sqlite"

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
        if call["name"] in DANGEROUS_TOOLS:
            approved = interrupt({"tool": call["name"], "args":call["args"]})
            if not approved:
                results.append(ToolMessage(
                    content="用户已拒绝本次操作，未执行，不要再重试，改为向用户确认他真正想做什么",
                    tool_call_id=call["id"]
                ))
                continue
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
checkpointer = SqliteSaver(sqlite3.connect(DB_FILE, check_same_thread=False))
app = graph.compile(checkpointer=checkpointer)  