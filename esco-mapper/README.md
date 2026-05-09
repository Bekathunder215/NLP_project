# ESCO Mapper (Skeleton)

Minimal FastAPI skeleton for mapping DTU courses to ESCO skills and occupations.

## Run

### Prerequisites

1. Please ensure you have Python 3.10+ installed.
2. Ensure you have an .nt file from the esco-v1.2.1.rdf if you want to use the QLever integration.

after this, you can run the backend and frontend with:

```bash
docker compose up
```

if you do not want docker, follow these steps:

3. Create and activate a virtual environment.
4. Install dependencies: `pip install -r requirements.txt`
5. Start the API: `uvicorn esco_mapper.backend:app --reload --app-dir src`
6. Start the frontend: `uvicorn esco_frontend.app:app --reload --port 8001 --app-dir src`
7. Open the UI at http://localhost:8001
8. Tests with `pytest tests`

Frontend notes:

- The frontend calls the backend directly from the browser.
- If your browser blocks requests, enable CORS on the backend or serve both on the same host.
- Override the backend URL with `BACKEND_URL=http://localhost:8000` if needed.

## Data

- Looks for data/dtu_courses.jsonl first.
- Falls back to Assignments/infoRetrieval/data/dtu_courses.jsonl if present.
- (Optional) ESCO data files:
  - data/output_esco.nt full database, that is queried by sparkql (has lists of {"skill_uri",
    "label", "occupation_uris", "skill_uris" ... })

## QLever (ESCO RDF)

- QLever config lives in data/Qleverfile and data/Qleverfile-ui.yml.
- The RDF is expected at data/esco-v1.2.1.rdf.
- Start QLever (docker runtime via qlever CLI):
  1.  `cd data`
  2.  `qlever index`
  3.  `qlever start`
- QLever endpoint: http://localhost:7654
- QLever UI: http://localhost:8176

Notes:

- The ESCO file is RDF/XML. If indexing fails, convert to N-Triples and update Qleverfile to point
  at the .nt file.
- The backend proxy uses `QLEVER_URL` (see .env.example).

## Endpoints

- GET /api/v1/courses/{course_number}
- GET /api/v1/courses/{course_number}/occupations
- GET /api/v1/skills
- GET /api/v1/skills/courses
- GET /api/v1/occupations
- GET /api/v1/search

## DOCS

you can always look for the docs at

- Backend: http://backend.esco-mapper.orb.local:8000/docs (most important)
- FrontEnd: http://localhost:8001/docs
