import sys

from langchain_openai import ChatOpenAI
from memory import save_history,load_history
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,AIMessage
sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    streaming=True
)

history = load_history()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个AI助手，可以回复用户的问题"),
        MessagesPlaceholder("history"),
        ("human","{question}")
    ]
)

chain = prompt | llm

while True:
    question = input("请输入你的问题：")
    ans = ""
    if question == "exit":
        print("本轮对话结束")
        break

    for chunk in chain.stream({"history":history,"question":question}):
        print(chunk.content,end="",flush=True)
        ans+=chunk.content
    print()


    history.append(
        HumanMessage(content=question)
    )
    history.append(
        AIMessage(content=ans)
    )

    save_history(history)