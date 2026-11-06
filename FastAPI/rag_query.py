import sys
sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from pathlib import Path
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_core.messages import HumanMessage

DB_DIR = Path(__file__).parent / "chroma_db"
col = chromadb.PersistentClient(path=str(DB_DIR)).get_collection("diary")

embedder = OpenAIEmbeddings(model="text-embedding-qwen3-embedding-0.6b",
    base_url="http://127.0.0.1:1234/v1", api_key="lm-studio",
    check_embedding_ctx_length=False)

llm = ChatOpenAI(
    model="google/gemmma-4-e4b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    temperature=0
)

PROMPT = """请仅根据下面的参考资料回答问题，并在回答末尾标注来源文件名。分三种情况作答：
1. 资料能回答：正常回答并标来源。
2. 资料与问题相关但不足以完整回答：说出资料中已知的部分，并注明"资料未提及其余部分"。
3. 资料与问题完全无关：才明确说"资料里没有相关内容"。
任何情况下禁止编造。

{context}

问题：{question}"""

def ask(question:str,k:int = 3):
    qv = embedder.embed_query(question)
    res = col.query(query_embeddings=[qv],n_results=k)
    cards,sources = res["documents"][0], res["metadatas"][0]

    context = "\n\n".join(
        f"[资料{i+1}] 来源:{sources[i]['source']}\n{cards[i]}"
        for i in range(len(cards)))

    answer = llm.invoke([HumanMessage(
        content=PROMPT.format(context=context, question=question))]).content
    return answer, sources

if __name__ == "__main__":
    while True:
        q = input("\n问点什么（q退出）：")
        if q.strip() == "q":
            break
        answer, sources = ask(q)
        print("\n" + answer)
        print("\n(实际检索到的来源：",", ".join(s["source"] for s in sources) + ")")