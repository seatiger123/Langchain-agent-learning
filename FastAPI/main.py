import json,os
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from langchain_core.messages import HumanMessage
from graph_app import app as agent_app
from langgraph.types import Command
from store import store
from rag_query import ask

from fastapi import FastAPI, HTTPException, Depends
from store import store, TodoStore

def get_store() -> TodoStore:
    return store


app = FastAPI()

class AskRequest(BaseModel):
    question: str = Field(min_length=1,max_length=500)

class TodoCreate(BaseModel):
    text: str = Field(min_length=1, max_length=200, description="代办内容,1-200字")

class TodoOut(BaseModel):
    id:int
    text:str
    done:bool

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    thread_id: str = Field(min_length=1, max_length=64, description="会话ID，同一ID共享一段记忆")

class ConfirmRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=64)
    approve: bool = Field(description="是否批准该会话正在暂停中的危险操作")

@app.post("/chat")
def chat(req: ChatRequest):
    config = {"configurable":{"thread_id":req.thread_id}}
    result = agent_app.invoke({"messages":[HumanMessage(content=req.question)]},config)
    intr = result.get("__interrupt__")
    if intr:
        info = intr[0].value
        return {"type":"needs_confirmation","thread_id":req.thread_id,"tool":info["tool"],"args":info["args"]}
    return {"type":"answer","answer":result["messages"][-1].content}

@app.post("/chat/confirm")
def chat_confirm(req: ConfirmRequest):
    config = {"configurable": {"thread_id": req.thread_id}}
    snap = agent_app.get_state(config)
    if not snap.next:
        raise HTTPException(status_code=409, detail=f"会话{req.thread_id}没有等待确认的操作")
    
    result = agent_app.invoke(Command(resume=req.approve),config)
    intr = result.get("__interrupt__")
    if intr:
        info = intr[0].value
        return {"type":"needs_confirmation","thread_id":req.thread_id,"tool":info["tool"],"args":info["args"]}
    return {"type":"answer","answer":result["messages"][-1].content}

@app.post("/ask")
def ask_kb(req: AskRequest):
    answer, sources = ask(req.question)
    return {"answer":answer, "sources":[s["source"] for s in sources]}

@app.get("/")
def root():
    return {"message":"Hello FastAPI"}

@app.get("/todos",response_model=list[TodoOut])
def list_todos(done: bool | None = None, store: TodoStore = Depends(get_store)):
    if done is None:
        return store.items
    return [t for t in store.items if t["done"] == done]

@app.get("/todos/{todo_id}",response_model=TodoOut)
def get_todo(todo_id:int, store: TodoStore = Depends(get_store)):
    item = store.get(todo_id)
    if item is None:
        raise HTTPException(status_code=404,detail=f"不存在编号为{todo_id}的待办")
    return item



@app.post("/todos", status_code=201,response_model=TodoOut)
def create_todo(data: TodoCreate, store: TodoStore = Depends(get_store)):
    return store.add(data.text)

@app.patch("/todos/{todo_id}/complete",response_model=TodoOut)
def complete_todo(todo_id: int, store: TodoStore = Depends(get_store)):
    item = store.get(todo_id)
    if item is None:
        raise HTTPException(status_code=404,detail=f"不存在编号为{todo_id}的待办")
    if item["done"]:
        raise HTTPException(status_code=409,detail=f"待办 #{todo_id}已经完成过了，无需重复操作")
    store.complete(todo_id)
    return item

@app.delete("/todos/{todo_id}",status_code=204)
def delete_todo(todo_id: int, store: TodoStore = Depends(get_store)):
    item = store.delete(todo_id)
    if item is None:
        raise HTTPException(status_code=404,detail=f"不存在编号为{todo_id}的待办")
