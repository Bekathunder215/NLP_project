import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    os.environ["EMBEDDING_PROVIDER"] = "dummy"

    import esco_mapper.backend as backend_module

    importlib.reload(backend_module)
    with TestClient(backend_module.app) as client:
        yield client


def test_search_dense(client):
    response = client.get("/search", params={"query": "recommender systems"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "recommender systems"
    assert payload["mode"] == "dense"
    assert "results" in payload
    assert len(payload["results"]) >= 1
    assert "course_number" in payload["results"][0]
    assert "score" in payload["results"][0]


def test_course_detail_roundtrip(client):
    response = client.get("/search", params={"query": "database"})
    assert response.status_code == 200
    course_number = response.json()["results"][0]["course_number"]

    detail = client.get(f"/courses/{course_number}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["course_number"] == course_number
    assert "matched_skills" in payload
