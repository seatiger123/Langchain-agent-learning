import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing_extensions import Annotated,TypedDict
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from agent import llm, tools, tool_map, DANGEROUS_TOOLS

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
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)    
print(app.get_graph().draw_mermaid())

config = {"configurable": {"thread_id":"user-a"}}
while True:
    question = input("\n你 ( q退出 ) ：")
    if question.strip() == "q":
        break
    result = app.invoke({"messages":[HumanMessage(content=question)]},config)
    while result.get("__interrupt__"):
        info = result["__interrupt__"][0].value
        print(f"图已暂停并存档，等待确认：{info}")
        ok = input("批准吗？（y/n):").strip().lower()
        result = app.invoke(Command(resume=(ok == "y")), config)
    print("AI:",result["messages"][-1].content)