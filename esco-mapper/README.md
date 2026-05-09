# ESCO Mapper (Skeleton)

Minimal FastAPI skeleton for mapping DTU courses to ESCO skills and occupations.

## Run

1. Create and activate a virtual environment.
2. Install dependencies:
   `pip install -r requirements.txt`
3. Start the API:
   `uvicorn esco_mapper.backend:app --reload --app-dir src`
4. Start the frontend:
   `uvicorn esco_frontend.app:app --reload --port 8001 --app-dir src`
5. Open the UI at http://localhost:8001
6. Tests with `pytest tests`

Frontend notes:

- The frontend calls the backend directly from the browser.
- If your browser blocks requests, enable CORS on the backend or serve both on the same host.
- Override the backend URL with `BACKEND_URL=http://localhost:8000` if needed.

## Data

- Looks for data/dtu_courses.jsonl first.
- Falls back to Assignments/infoRetrieval/data/dtu_courses.jsonl if present.
- Optional ESCO data files:
  - data/esco_skills.json (list of {"skill_uri", "label"})
  - data/esco_occupations.json (list of {"occupation_uri", "label", "essential_skills", "optional_skills"})

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

- The ESCO file is RDF/XML. If indexing fails, convert to N-Triples and update
  Qleverfile to point at the .nt file.
- The backend proxy uses `QLEVER_URL` (see .env.example).

## Endpoints

- GET /courses/{course_number}
- GET /skills/{skill_uri}
- GET /occupations/{occupation_uri}
- GET /courses/{course_number}/skills
- GET /skills/{skill_uri}/courses
- GET /occupations/{occupation_uri}/courses
- GET /courses/{course_number}/occupations
- POST /query
- POST /v1/query (proxy SPARQL to QLever)
