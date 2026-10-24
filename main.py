import sys

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate

sys.stdout.reconfigure(encoding="utf-8")

#load_dotenv()


llm = ChatOpenAI(
    model="google/gemma-4-e4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    streaming=True
)

# prompt =  ChatPromptTemplate.from_messages(

#     [
#         (
#             "system",
#             "你是一位高冷的女高中生，用户是你的同学，你有点喜欢用户，但你是个傲娇，所以在用户面前会表现得高冷，但有时也会不知所措，在用户说出一些豪华时，你也会卸下高冷的防护，慢慢敞开心扉"
#         ),
#         (
#             "human",
#             "请你对{topic}做出回复"
#         )


#     ]
    
# )

prompt = PromptTemplate.from_template(
    "你是一名厨师，请你介绍一下{topic}的做法"
)
chain = prompt | llm

topic = input("请输入你想制作的佳肴：")

for chunk in chain.stream(
    {
        "topic":topic
    }
):
    print(chunk.content,end="",flush=True)
'''
for chunk in llm.stream(
    "今晚你想吃什么啊？"
):
    print(chunk.content,end="",flush=True)

#print(result.content)
'''