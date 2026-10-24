import sys
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage,ToolMessage

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

agent = create_agent(llm,[get_weather,get_current_time,get_user_city])

result = agent.invoke(
    {"messages":[("user","用户小周所在的城市今天适合穿短袖吗？")]}
)

for m in result["messages"]:
    print(f"{type(m).__name__:12} | content: {str(m.content)[:60]} | tool_calls: {getattr(m, 'tool_calls', None)}")

    