import sys,os,json
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage

sys.stdout.reconfigure(encoding="utf-8")
DATA_FILE = Path(__file__).parent / "todos.json"

llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    streaming=True
)

class TodoStore:
    def __init__(self):
        self.items = []
        self._next = 1
        self._load()

    def add(self, text):
        item = {"id":self._next,"text": text, "done": False}
        self._next+=1
        self.items.append(item)
        self._save()
        return item

    def get(self,todo_id):
        for it in self.items:
            if it["id"] == todo_id:
                return it

        return None
    def complete(self,todo_id):
        item = self.get(todo_id)
        if item is None or item["done"]:
            return None
        item["done"] = True
        self._save()
        return item
    def delete(self,todo_id):
        item = self.get(todo_id)
        if item is None:
            return None
        self.items.remove(item)
        self._save()
        return item
    
    def _save(self):
        with open(DATA_FILE,"w",encoding="utf-8") as f:
            json.dump(self.items,f,ensure_ascii=False,indent=4)

    def _load(self):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE,"r",encoding="utf-8") as f:
                self.items = json.load(f)
            self._next = max((it["id"] for it in self.items),default=0) + 1


store = TodoStore()

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

@tool
def divide(a:float,b:float) -> str:
    """简易计算器，只能做除法。任何除法计算都必须使用本工具完成，禁止心算，除法计算必须调用该工具，无论数据是否合法。"""
    return str(a/b)

tools = [add_todo,list_todos,complete_todo,delete_todo,divide]
tool_map = {t.name:t for t in tools}
DANGEROUS_TOOLS = {"delete_todo"}
llm_with_tools = llm.bind_tools(tools)

question = input("请输入问题:")
messages = [HumanMessage(content=question)]

ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

round = 0;
while ai_msg.tool_calls:
    round += 1
    print(f">>> [第{round}轮] {len(ai_msg.tool_calls)} 张申请单")
    for call in ai_msg.tool_calls:
        print(f"     申请：{call['name']}({call['args']})")

        if call["name"] in DANGEROUS_TOOLS:
            ok = input(f"即将执行不可逆操作：{call['name']}({call['args']}),确认吗？(y/n)")
            if ok.strip().lower() != "y":
                result = "用户已拒绝本次操作，未执行，不要再重试，改为向用户确认他真正想做什么。"
                print(f"       结果：{result}")
                messages.append(ToolMessage(content=result,tool_call_id=call["id"]))
                continue
            
        result = tool_map[call["name"]].invoke(call["args"])
        print(f"     结果：{result}")
        messages.append(ToolMessage(content=result,tool_call_id=call["id"]))
    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

print("\n=== final answer ===")
print(ai_msg.content)
