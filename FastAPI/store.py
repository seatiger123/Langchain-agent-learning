import json,os
from pathlib import Path

DATA_FILE = Path(__file__).parent / "todos.json"
class TodoStore:
    def __init__(self):
        self.items = []
        self._next = 1
        self._load()

    def add(self, text):
        item = {"id":self._next,"text": text, "done": False}
        self._next+=1
        self.items.append(item)
        self._save()
        return item

    def get(self,todo_id):
        for it in self.items:
            if it["id"] == todo_id:
                return it

        return None
    def complete(self,todo_id):
        item = self.get(todo_id)
        if item is None or item["done"]:
            return None
        item["done"] = True
        self._save()
        return item
    def delete(self,todo_id):
        item = self.get(todo_id)
        if item is None:
            return None
        self.items.remove(item)
        self._save()
        return item
    
    def _save(self):
        with open(DATA_FILE,"w",encoding="utf-8") as f:
            json.dump(self.items,f,ensure_ascii=False,indent=4)

    def _load(self):
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE,"r",encoding="utf-8") as f:
                self.items = json.load(f)
            self._next = max((it["id"] for it in self.items),default=0) + 1


store = TodoStore()