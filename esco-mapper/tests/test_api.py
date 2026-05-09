import importlib
import os

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    os.environ["EMBEDDING_PROVIDER"] = "dummy"

    import src.esco_mapper.backend as backend_module

    importlib.reload(backend_module)
    with TestClient(backend_module.app) as client:
        yield client


def test_course_detail_api_v1(client):
    # Get a valid course_number from search
    response = client.get(
        "/api/v1/search", params={"query": "database", "mode": "sparse"}
    )
    assert response.status_code == 200

    course_number = response.json()["results"][0]["course_number"]

    # Call the correct API v1 endpoint
    detail = client.get(f"/api/v1/courses/{course_number}")
    assert detail.status_code == 200

    payload = detail.json()
    assert payload["course_number"] == course_number

    # Basic sanity checks for CourseDetail
    assert "matched_skills" in payload
    assert isinstance(payload["matched_skills"], (list, dict))


def test_course_occupations_api_v1(client):
    # Get a valid course_number from search
    response = client.get(
        "/api/v1/search", params={"query": "database", "mode": "sparse"}
    )
    assert response.status_code == 200

    course_number = response.json()["results"][0]["course_number"]

    # Call occupations endpoint
    occ = client.get(f"/api/v1/courses/{course_number}/occupations")
    assert occ.status_code == 200

    payload = occ.json()

    assert payload["course_number"] == course_number

    assert "relevant_occupations" in payload
    assert isinstance(payload["relevant_occupations"], list)

    # only validate structure if non-empty
    if len(payload["relevant_occupations"]) > 0:
        first = payload["relevant_occupations"][0]

        assert "occupation_uri" in first
        assert "label" in first
        assert "coverage_score" in first

        assert isinstance(first["occupation_uri"], str)
        assert isinstance(first["label"], str)
        assert isinstance(first["coverage_score"], (float, int))
