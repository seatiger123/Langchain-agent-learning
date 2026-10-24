import sys

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage,AIMessage

sys.stdout.reconfigure(encoding="utf-8")

llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    streaming=True
)

# prompt = PromptTemplate.from_template(
#     "你是一个AI智能体，负责回答用户的问题{topic}"
# )
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个AI智能体，能够回答用户的问题"
        ),
        MessagesPlaceholder("history"),
        (
            "human",
            "{question}"
        )
    ]
)

chain = prompt | llm

history = []

while True:
    question = input("请输入问题:")

    if question == "exit":
        print("本轮对话到此结束")
        break;
    
    for chunk in chain.stream({"question":question,"history":history}):
        print(chunk.content,end="",flush=True)

    print()

    history.append(
        HumanMessage(content=question)
    )

#只能保留一轮对话中的记忆，若重新运行则会清除记忆