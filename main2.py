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

prompt = PromptTemplate.from_template(
    "你是一名高级AI工程师，精通各种AI相关的知识与应用，请你详细介绍一下{topic}"
)

topic = input("请输入你想了解的AI知识：")
print(prompt.input_variables)
chain = prompt | llm

for chunk in chain.stream({"topic":topic}):
    print(chunk.content,end="",flush=True)