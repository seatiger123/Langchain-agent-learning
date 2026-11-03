import json,os
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from agent import run_agent
from store import store

app = FastAPI()

class TodoCreate(BaseModel):
    text: str = Field(min_length=1, max_length=200, description="代办内容,1-200字")

class TodoOut(BaseModel):
    id:int
    text:str
    done:bool

class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户的问题")
    dangerous_ok:bool = Field(default=False, description="放行危险操作（delete）。仅当用户看到 needs_confirmation 并明确同意后才置true")


@app.post("/chat")
def chat(req: ChatRequest):
    return run_agent(req.question, req.dangerous_ok)

@app.get("/")
def root():
    return {"message":"Hello FastAPI"}

@app.get("/todos",response_model=list[TodoOut])
def list_todos(done: bool | None = None):
    if done is None:
        return store.items
    return [t for t in store.items if t["done"] == done]

@app.get("/todos/{todo_id}",response_model=TodoOut)
def get_todo(todo_id:int):
    item = store.get(todo_id)
    if item is None:
        raise HTTPException(status_code=404,detail=f"不存在编号为{todo_id}的待办")
    return item



@app.post("/todos", status_code=201,response_model=TodoOut)
def create_todo(data: TodoCreate):
    return store.add(data.text)

@app.patch("/todos/{todo_id}/complete",response_model=TodoOut)
def complete_todo(todo_id: int):
    item = store.get(todo_id)
    if item is None:
        raise HTTPException(status_code=404,detail=f"不存在编号为{todo_id}的待办")
    if item["done"]:
        raise HTTPException(status_code=409,detail=f"待办 #{todo_id}已经完成过了，无需重复操作")
    store.complete(todo_id)
    return item

@app.delete("/todos/{todo_id}",status_code=204)
def delete_todo(todo_id: int):
    item = store.delete(todo_id)
    if item is None:
        raise HTTPException(status_code=404,detail=f"不存在编号为{todo_id}的待办")
