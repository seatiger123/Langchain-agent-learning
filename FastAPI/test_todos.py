import pytest
from fastapi.testclient import TestClient
from store import TodoStore
from main import app, get_store


class FakeStore(TodoStore):
    def _load(self):
        pass
    def _save(self):
        pass


@pytest.fixture
def client():
    fake = FakeStore()                                  # 每个测试一间全新试衣间
    app.dependency_overrides[get_store] = lambda: fake  # 把插座后面的真货换成替身
    yield TestClient(app)
    app.dependency_overrides.clear()                    # 收摊：拔掉假插头，不影响下家


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200


def test_create_and_get(client):
    r = client.post("/todos", json={"text": "写测试"})
    assert r.status_code == 201
    r2 = client.get(f"/todos/{r.json()['id']}")
    assert r2.status_code == 200
    assert r2.json()["text"] == "写测试"


def test_get_missing_404(client):
    r = client.get("/todos/999")
    assert r.status_code == 404
    assert "999" in r.json()["detail"]


def test_create_empty_422(client):
    r = client.post("/todos", json={"text": ""})
    assert r.status_code == 422


def test_complete_then_conflict(client):
    tid = client.post("/todos", json={"text": "完成我"}).json()["id"]
    assert client.patch(f"/todos/{tid}/complete").status_code == 200
    assert client.patch(f"/todos/{tid}/complete").status_code == 409


def test_full_lifecycle(client):
    tid = client.post("/todos", json={"text": "一生"}).json()["id"]
    assert client.patch(f"/todos/{tid}/complete").status_code == 200
    assert client.delete(f"/todos/{tid}").status_code == 204
    assert client.get(f"/todos/{tid}").status_code == 404