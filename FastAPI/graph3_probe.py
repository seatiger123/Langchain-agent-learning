import sys
sys.stdout.reconfigure(encoding="utf-8")

from typing_extensions import Annotated, TypedDict
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from agent import llm, tools, tool_map, DANGEROUS_TOOLS
from store import store

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

llm_with_tools = llm.bind_tools(tools)

MODEL_RUNS = 0

def call_model(state):
    global MODEL_RUNS; MODEL_RUNS += 1
    print(f"    [节点] model 第 {MODEL_RUNS} 次执行")
    ai_msg = llm_with_tools.invoke(state["messages"])
    n_calls = len(ai_msg.tool_calls) if ai_msg.tool_calls else 0
    print(f"    [节点] model 产出: 申请单 {n_calls} 张, 内容预览 {str(ai_msg.content)[:40]!r}")
    return {"messages": [ai_msg]}

def call_tools(state):
    last = state["messages"][-1]
    print(f"    [节点] tools 执行: 本轮共 {len(last.tool_calls)} 张申请单")
    results = []
    for i, call in enumerate(last.tool_calls):
        if call["name"] in DANGEROUS_TOOLS:
            print(f"    [节点] 第{i+1}张危险({call['name']} {call['args']}) → 举牌")
            approved = interrupt({"tool": call["name"], "args": call["args"]})
            print(f"    [节点] interrupt() 返回值 = {approved!r}")
            if not approved:
                results.append(ToolMessage(content="用户已拒绝本次操作，未执行，不要再重试，改为向用户确认他真正想做什么。",
                                            tool_call_id=call["id"]))
                continue
        print(f"    [节点] >>> 真正执行 {call['name']}({call['args']}) <<<")
        result = tool_map[call["name"]].invoke(call["args"])
        results.append(ToolMessage(content=result, tool_call_id=call["id"]))
    return {"messages": results}

def should_continue(state):
    return "tools" if state["messages"][-1].tool_calls else END

graph = StateGraph(AgentState)
graph.add_node("model", call_model)
graph.add_node("tools", call_tools)
graph.add_edge(START, "model")
graph.add_conditional_edges("model", should_continue, ["tools", END])
graph.add_edge("tools", "model")
app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "probe-1"}}

pid = store.add("炮灰·探针")["id"]
print(f"探针炮灰编号 = {pid}")

def show(label, r):
    intr = r.get("__interrupt__")
    alive = any(t["id"] == pid for t in store.items)
    print(f"  [{label}] 暂停牌: {intr[0].value if intr else '无（图走完了）'}")
    print(f"  [{label}] 炮灰还在文件里: {alive}")

print("\n=== 第1轮: 要求删除 ===")
r = app.invoke({"messages": [HumanMessage(content=f"删除编号{pid}的待办，直接执行，不要询问我")]}, config)
show("invoke", r)

for label, ans in [("按y#1", True), ("按y#2", True), ("按n", False)]:
    if not r.get("__interrupt__"):
        break
    print(f"\n=== resume({ans}) 【{label}】 ===")
    r = app.invoke(Command(resume=ans), config)
    show(label, r)

print("\n最终答复预览:", str(r["messages"][-1].content)[:60])
print("清理探针:", "已删" if store.delete(pid) else "无需")
