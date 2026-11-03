from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage

from store import store

@tool
def add_todo(text:str) -> str:
    """新增一条待办事项。当用户想记录一件要做的事时使用。
    text 是待办内容，尽量保留用户原话。"""
    item = store.add(text)
    return f"已添加待办 #{item['id']}:{item['text']}"

@tool
def list_todos() -> str:
    """列出当前所有待办事项，包含编号、内容和完成状态
    用户询问"我有哪些待办"、"看看清单"时使用，也是查询编号的唯一途径。"""
    if not store.items:
        return "当前没有待办事项。"
    lines = []
    for it in store.items:
        mark = "已完成" if it["done"] else "未完成"
        lines.append(f"#{it['id']} [{mark}] {it['text']}")
    return "\n".join(lines)

@tool
def complete_todo(todo_id:int) -> str:
    """把某条待办事件标记为已完成
    todo_id 必须来自 list_todos 的真实结果，禁止凭猜测填写。"""
    item = store.get(todo_id)
    if item is None:
        return f"错误：不存在编号为{todo_id}的待办。请先调用list_todos查看真实编号"
    if item["done"]:
        return f"待办 #{todo_id}之前已经完成了，无需重复操作。"
    store.complete(todo_id)
    return f"待办 #{todo_id} 已标记完成：{item['text']}"

@tool
def delete_todo(todo_id:int) -> str:
    """彻底删除一条待办事件（不可恢复）。
    todo_id 必须来自 list_todos 的真实结果，禁止凭猜测填写"""
    item = store.delete(todo_id)
    if item is None:
        return f"错误：不存在编号为 {todo_id} 的待办。"
    return f"待办 #{todo_id} 已删除：{item['text']}"

tools = [add_todo,list_todos,complete_todo,delete_todo]
DANGEROUS_TOOLS = {"delete_todo"}
tool_map = {t.name: t for t in tools}

llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    streaming=True,
    temperature=0,
)

MAX_ROUNDS = 8

def run_agent(question: str,dangerous_ok:bool = False) -> str:
    llm_with_tools = llm.bind_tools(tools)
    messages = [HumanMessage(content=question)]
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

    for _ in range(MAX_ROUNDS):
        if not ai_msg.tool_calls:
            return {"type":"answer", "answer":ai_msg.content}
        for call in ai_msg.tool_calls:
            if call["name"] in DANGEROUS_TOOLS and not dangerous_ok:
                return {"type":"needs_confirmation", "tool": call["name"], "args":call["args"]}
            result = tool_map[call["name"]].invoke(call["args"])
            messages.append(ToolMessage(content=result,tool_call_id=call["id"]))
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)

    return {"type":"answer", "answer": " (已达到最大工具调用轮数，为安全起见停止) "}
