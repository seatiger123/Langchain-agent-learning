import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    streaming=True
)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            你是一个AI助手，能够回答用户的问题。
            """
        ),
        (
            "human",
            "我喜欢吃香蕉"
            "请回答{question}"
        )
    ]
)

question = input("请输入你想得到答案的问题：")

chain = prompt | llm

for chunk in chain.stream({"question":question}):
    print(chunk.content,end="",flush=True)
