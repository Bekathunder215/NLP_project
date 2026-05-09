from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .helpers import (
    build_pipeline,
    get_course_detail,
    get_course_occupations,
    get_course_skills,
    get_occupation_courses,
    get_occupation_detail,
    get_skill_courses,
    get_skill_detail,
    handle_query,
    search_courses,
)
from .models import (
    CourseDetail,
    CourseOccupationsResponse,
    CourseSkillsResponse,
    OccupationCoursesResponse,
    OccupationDetailResponse,
    QleverQueryRequest,
    QleverQueryResponse,
    QueryRequest,
    QueryResponse,
    SearchResponse,
    SkillCoursesResponse,
    SkillDetailResponse,
)
from .qlever import looks_like_sparql, run_sparql_query

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state = build_pipeline()
    globals().update(state)
    yield


app = FastAPI(lifespan=lifespan)
# List the exact origins you want to allow
origins = [
    "http://localhost:8001",
    "http://127.0.0.1:8001",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Explicitly allow your frontend port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/courses/{course_number}", response_model=CourseDetail)
async def course_detail(course_number: str):
    data = get_course_detail(course_number, globals())
    if data is None:
        raise HTTPException(status_code=404, detail="Course not found")
    return data


@app.get("/api/v1/courses/{course_number}/skills", response_model=CourseSkillsResponse)
async def course_skills(course_number: str):
    data = get_course_skills(course_number, globals())
    if data is None:
        raise HTTPException(status_code=404, detail="Course not found")
    return data


@app.get("/api/v1/courses/{course_number}/occupations", response_model=CourseOccupationsResponse)
async def course_occupations(course_number: str):
    data = get_course_occupations(course_number, globals())
    if data is None:
        raise HTTPException(status_code=404, detail="Course not found")
    return data


@app.get("/api/v1/skills/{skill_uri}", response_model=SkillDetailResponse)
async def skill_detail(skill_uri: str):
    data = get_skill_detail(skill_uri, globals())
    if data is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return data


@app.get("/api/v1/skills/{skill_uri}/courses", response_model=SkillCoursesResponse)
async def skill_courses(skill_uri: str):
    data = get_skill_courses(skill_uri, globals())
    if data is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return data


@app.get("/api/v1/occupations/{occupation_uri}", response_model=OccupationDetailResponse)
async def occupation_detail(occupation_uri: str):
    data = get_occupation_detail(occupation_uri, globals())
    if data is None:
        raise HTTPException(status_code=404, detail="Occupation not found")
    return data


@app.get("/api/v1/occupations/{occupation_uri}/courses", response_model=OccupationCoursesResponse)
async def occupation_courses(occupation_uri: str):
    data = get_occupation_courses(occupation_uri, globals())
    if data is None:
        raise HTTPException(status_code=404, detail="Occupation not found")
    return data


@app.post("/api/v1/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    return handle_query(request, globals())


@app.post("/api/v1/query", response_model=QleverQueryResponse)
async def qlever_query(request: QleverQueryRequest):
    sparql = request.sparql
    if not sparql and request.text:
        if looks_like_sparql(request.text):
            sparql = request.text
        else:
            raise HTTPException(
                status_code=400,
                detail="Provide SPARQL in 'sparql' or pass a SPARQL query as 'text'.",
            )
    if not sparql:
        raise HTTPException(status_code=400, detail="SPARQL query is required.")

    try:
        results = await run_sparql_query(sparql)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502, detail=f"QLever request failed: {exc}"
        ) from exc

    return {"query": request.text, "sparql": sparql, "results": results}


@app.get("/api/v1/search", response_model=SearchResponse)
async def search(
    query: str,
    top_k: int = Query(10, gt=0),
    mode: str = Query("dense"),
    alpha: float = Query(0.5, ge=0.0, le=1.0),
):
    print(f"Received search request: query='{query}', top_k={top_k}, mode='{mode}', alpha={alpha}")
    return search_courses(query, globals(), top_k=top_k, mode=mode, alpha=alpha)
