import json
import os
from langchain_core.messages import HumanMessage,AIMessage

MEMORY_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memory.json")

def load_history():
    if not os.path.exists(MEMORY_FILE):
        return []

    with open(MEMORY_FILE,"r",encoding="utf-8") as f:
        data = json.load(f)

    history = []

    for message in data:
        if message["role"] == "human":
            history.append(
                HumanMessage(content=message["content"])
            )
        elif message["role"] == "assistant":
            history.append(
                AIMessage(content=message["content"])
            )

    return history


def save_history(history):
    data = []

    for message in history:
        if isinstance(message,HumanMessage):
            data.append({
                "role":"human",
                "content":message.content

            })

        elif isinstance(message,AIMessage):
            data.append({
                "role":"assistant",
                "content":message.content
            })

    with open(MEMORY_FILE,"w",encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4
        )