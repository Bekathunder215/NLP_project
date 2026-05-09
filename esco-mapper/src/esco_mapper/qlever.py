from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_QLEVER_URL = "http://localhost:7654"
DEFAULT_QLEVER_TIMEOUT = 30.0


def qlever_url() -> str:
    url = os.getenv("QLEVER_URL", DEFAULT_QLEVER_URL)
    return url.rstrip("/")


def qlever_timeout() -> float:
    value = os.getenv("QLEVER_TIMEOUT")
    if not value:
        return DEFAULT_QLEVER_TIMEOUT
    try:
        return float(value)
    except ValueError:
        return DEFAULT_QLEVER_TIMEOUT


def looks_like_sparql(text: str) -> bool:
    return text.lstrip().upper().startswith(
        ("PREFIX", "SELECT", "ASK", "CONSTRUCT", "DESCRIBE")
    )


async def run_sparql_query(sparql: str) -> dict[str, Any]:
    if not sparql.strip():
        raise ValueError("SPARQL query is empty")

    url = qlever_url()
    timeout = qlever_timeout()
    headers = {"Accept": "application/sparql-results+json"}

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, data={"query": sparql}, headers=headers)
        response.raise_for_status()
        return response.json()


def run_sparql_query_sync(sparql: str) -> dict[str, Any]:
    if not sparql.strip():
        raise ValueError("SPARQL query is empty")

    url = qlever_url()
    timeout = qlever_timeout()
    headers = {"Accept": "application/sparql-results+json"}

    with httpx.Client(timeout=timeout) as client:
        response = client.post(url, data={"query": sparql}, headers=headers)
        response.raise_for_status()
        return response.json()
