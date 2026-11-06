import sys
sys.stdout.reconfigure(encoding="utf-8")

import chromadb
from pathlib import Path
from langchain_openai import OpenAIEmbeddings

DOCS_DIR = Path(r"C:\Users\特别周\Desktop\obsidian\note_or_whatelse\离散数学\第3章 函数")
DB_DIR = Path(__file__).parent / "chroma_db"

embedder = OpenAIEmbeddings(
    model="text-embedding-qwen3-embedding-0.6b",
    base_url="http://127.0.0.1:1234/v1",
    api_key="lm-studio",
    check_embedding_ctx_length=False,
)

def chunk_text(text, source, max_len=500):
    cards = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if len(para) < 30:
            continue
        if len(para) <= max_len:
            cards.append({"text": para, "source": source})
        else:
            buf = ""
            for sent in para.replace("。", "。\n").split("\n"):
                if buf and len(buf) + len(sent) > max_len:
                    cards.append({"text": buf, "source": source})
                    buf = sent
                else:
                    buf += sent
            if buf:
                cards.append({"text":buf,"source":source})
    return cards

cards = []
ALLOWED = {"3.1 函数的定义和性质.md","3.2 函数的复合与反函数.md","3.3 双射函数与集合的基数.md"}
for f in sorted(DOCS_DIR.glob("*.md")):
    # if not any(c.isdigit() for c in f.stem):
    #     continue
    if f.name not in ALLOWED:
        continue
    cards += chunk_text(f.read_text(encoding="utf-8"),source=f.name)
print(f"共切出 {len(cards)}张知识卡片，前三张预览：")
for c in cards[:3]:
    print(f"  [{c['source']}] {c['text'][:50]}...")

client = chromadb.PersistentClient(path=str(DB_DIR))
try:
    client.delete_collection("diary")
except Exception:
    pass
col = client.get_or_create_collection("diary")


BATCH = 16
for i in range(0, len(cards), BATCH):
    batch = cards[i:i+BATCH]
    vecs = embedder.embed_documents([c["text"] for c in batch])
    col.add(ids=[f"card-{i+j}" for j in range(len(batch))],
            embeddings=vecs,
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"]} for c in batch])
print(f"入库完成：{col.count()} 张卡片住在 {DB_DIR}")

for q in ["输入法惹的祸是哪天破的案", "怎么让机器重启后还记得我", "接口测试和点两下有啥区别"]:
    qv = embedder.embed_query(q)
    res = col.query(query_embeddings=[qv], n_results=2)
    print(f"\n问：{q}")
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"   [距离 {dist:.3f}] ({meta['source']}) {doc[:50]}...")

while True:
    q = input("\n你想查什么（q退出）：")
    if q.strip() == "q":
        break;
    qv = embedder.embed_query(q)
    res = col.query(query_embeddings=[qv], n_results=3)
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        print(f"  [距离 {dist:}] ({meta['source']}) {doc[:50]}...")