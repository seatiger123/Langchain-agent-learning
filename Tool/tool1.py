import sys

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage

sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    streaming=True
)

@tool
def get_weather(city:str) -> str:
    """获取当前指定城市天气情况"""
    print(f">>>[执行] get_weather 真的执行了， city = {city}")
    return f"{city}今天晴，气温25度，东风2级"

@tool
def get_user_city(user:str) -> str:
    """查询某个用户当前所在的城市"""
    return "北京"


@tool
def get_current_time() -> str:
    """获取当前的日期和时间"""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M")

tools = [get_weather,get_user_city,get_current_time]
llm_with_tools = llm.bind_tools(tools)
question = input("请输入问题：")
messages = [HumanMessage(content=question)]


ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)

round = 0;
while ai_msg.tool_calls:
    round+=1
    print(f">>> [第{round}轮] 本条消息包含 {len(ai_msg.tool_calls)} 张申请单")
    for tool_call in ai_msg.tool_calls:
        tool_func = {t.name:t for t in tools}[tool_call["name"]]
        result = tool_func.invoke(tool_call["args"])
        print(f">>>[执行结果]{result}")

        messages.append(
            ToolMessage(content=result,tool_call_id=tool_call["id"])
        )

    ai_msg = llm_with_tools.invoke(messages)
    messages.append(ai_msg)

print("\n=== final answer ===")
print(ai_msg.content)

