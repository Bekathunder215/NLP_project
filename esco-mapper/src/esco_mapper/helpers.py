import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import openai
from dotenv import load_dotenv
from joblib import dump, load
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .qlever import run_sparql_query_sync

load_dotenv()

DEFAULT_SKILLS = [
    {
        "skill_uri": "http://data.europa.eu/esco/skill/505e4ef3-7ce4-437d-b7b4-5c608f71c258",
        "label": "build recommender systems",
    },
    {
        "skill_uri": "http://data.europa.eu/esco/skill/3b6f4d3a-2f0b-49b2-9cb6-5c4a0ef0f7b1",
        "label": "design database scheme",
    },
    {
        "skill_uri": "http://data.europa.eu/esco/skill/0ff5f50b-0c0b-4d54-a6d0-0a0b9a2b0b58",
        "label": "normalise data",
    },
    {
        "skill_uri": "http://data.europa.eu/esco/skill/8d1b3a34-7264-4f88-b398-82b4e40c33d6",
        "label": "mobile application development",
    },
]

DEFAULT_OCCUPATIONS = [
    {
        "occupation_uri": "http://data.europa.eu/esco/occupation/258e46f9-0075-4a2e-adae-1ff0477e0f30",
        "label": "industrial mobile devices software developer",
        "essential_skills": [
            "http://data.europa.eu/esco/skill/8d1b3a34-7264-4f88-b398-82b4e40c33d6",
            "http://data.europa.eu/esco/skill/3b6f4d3a-2f0b-49b2-9cb6-5c4a0ef0f7b1",
        ],
        "optional_skills": [
            "http://data.europa.eu/esco/skill/0ff5f50b-0c0b-4d54-a6d0-0a0b9a2b0b58",
        ],
    },
    {
        "occupation_uri": "http://data.europa.eu/esco/occupation/1b9b3e6e-9d18-4e3a-8b2a-0d8a7a7b64a4",
        "label": "data scientist",
        "essential_skills": [
            "http://data.europa.eu/esco/skill/505e4ef3-7ce4-437d-b7b4-5c608f71c258",
            "http://data.europa.eu/esco/skill/3b6f4d3a-2f0b-49b2-9cb6-5c4a0ef0f7b1",
            "http://data.europa.eu/esco/skill/0ff5f50b-0c0b-4d54-a6d0-0a0b9a2b0b58",
        ],
        "optional_skills": [],
    },
]

NOMIC_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
DEFAULT_EMBEDDING_PROVIDER = "campusai"

ESCO_SKILL_URI_PREFIX = "http://data.europa.eu/esco/skill/"
ESCO_OCCUPATION_URI_PREFIX = "http://data.europa.eu/esco/occupation/"


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _workspace_root(root: Path) -> Path:
    return root.parents[1]


def _vectorizer_cache_dir(root: Path) -> Path:
    return root / "vectorizer_cache"


def _chroma_dir(root: Path) -> Path:
    return root / "chroma_db"


def _find_first(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None

def _esco_cache_dir(root: Path) -> Path:
    return root / "data" / "esco"

def _esco_indexes_path(root: Path) -> Path:
    return _esco_cache_dir(root) / "esco_indexes.joblib"

def _load_esco_indexes(root: Path) -> dict[str, Any] | None:
    path = _esco_indexes_path(root)

    if path.exists():
        return load(path)

    return None

def _save_esco_indexes(indexes: dict[str, Any], root: Path) -> None:
    cache_dir = _esco_cache_dir(root)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dump(indexes, _esco_indexes_path(root))

def load_courses_jsonl() -> tuple[list[dict[str, Any]], Path | None]:
    root = _project_root()
    workspace = _workspace_root(root)
    candidates = [
        root / "data" / "dtu_courses.jsonl",
        workspace / "Assignments" / "infoRetrieval" / "data" / "dtu_courses.jsonl",
        workspace / "Assignments" / "rag" / "data" / "dtu_courses.jsonl",
    ]
    path = _find_first(candidates)
    if path is None:
        return (
            [
                {
                    "course_code": "42578",
                    "title": "Advanced Business Analytics",
                    "fields": {"Point( ECTS )": 5},
                    "learning_objectives": [
                        "build recommender systems",
                        "design database scheme",
                    ],
                }
            ],
            None,
        )

    courses: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            courses.append(json.loads(line))
    return courses, path


def _get_nomic_model() -> SentenceTransformer:
    model_name = os.getenv("NOMIC_EMBED_MODEL", NOMIC_MODEL_NAME)
    return SentenceTransformer(model_name)


def campusai_embed(texts: list[str]) -> np.ndarray:
    api_key = os.getenv("CAMPUSAI_API_KEY")
    api_base = os.getenv(
        "CAMPUSAI_BASE_URL", "https://chat.campusai.compute.dtu.dk/api"
    )
    model = os.getenv("CAMPUSAI_EMBED_MODEL", "nomic-embed-text")
    if not api_key:
        raise ValueError("CAMPUSAI_API_KEY is not set")

    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = client.embeddings.create(model=model, input=texts)
    embeddings = np.array([item.embedding for item in response.data], dtype=float)
    return embeddings


def nomic_embed_local(texts: list[str]) -> np.ndarray:
    model = _get_nomic_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=float)


def select_embedder() -> tuple[str, callable]:
    provider = os.getenv("EMBEDDING_PROVIDER", DEFAULT_EMBEDDING_PROVIDER).lower()
    if provider == "dummy":
        return "dummy", dummy_embed
    if provider == "campusai":
        try:
            _ = os.getenv("CAMPUSAI_API_KEY")
            if not _:
                raise ValueError("missing key")
            return provider, campusai_embed
        except ValueError:
            provider = "nomic"
    if provider in ("nomic", "nomic-local"):
        return "nomic", nomic_embed_local
    return "nomic", nomic_embed_local


def _sparql_bindings(results: dict[str, Any]) -> list[dict[str, Any]]:
    return results.get("results", {}).get("bindings", [])


def _sparql_value(binding: dict[str, Any], key: str) -> str | None:
    value = binding.get(key)
    if not isinstance(value, dict):
        return None
    return value.get("value")


def _env_int(name: str) -> int | None:
    value = os.getenv(name)
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_predicates(name: str, default: list[str]) -> list[str]:
    value = os.getenv(name)
    if not value:
        return default
    items = [item.strip() for item in value.split(",")]
    return [item for item in items if item]


def _sparql_property_path(uris: list[str]) -> str:
    if not uris:
        return ""
    return "|".join(f"<{uri}>" for uri in uris)

def _load_esco_skills_from_qlever(limit: int | None = None) -> list[dict[str, Any]]:
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    # We combine the "prefLabel" node and the "literalForm" into one path.
    # We use LANGMATCHES because it is more robust than LANG() == "en".
    sparql = f"""
PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>

SELECT DISTINCT ?skill ?label
WHERE {{
    ?skill skosxl:prefLabel ?labelNode .
    
    # Try to get the literal form from the label node
    ?labelNode skosxl:literalForm ?label .
    
    # FILTER(STRSTARTS(STR(?skill), "{ESCO_SKILL_URI_PREFIX}"))
    # FILTER(LANGMATCHES(LANG(?label), "en"))
}}
{limit_clause}
"""

    results = run_sparql_query_sync(sparql)
    print(f"-- QLever returned {len(_sparql_bindings(results))} skills from the first query.")
    skills = []
    
    for binding in _sparql_bindings(results):
        skill_uri = _sparql_value(binding, "skill")
        label = _sparql_value(binding, "label")
        if skill_uri and label:
            skills.append({"skill_uri": skill_uri, "label": label})

    # FALLBACK: If the triple-hop (skill -> node -> literal) returned nothing,
    # it's likely your endpoint has "flattened" labels (skill -> literal).
    if not skills:
        sparql_flattened = f"""
PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#>
SELECT DISTINCT ?skill ?label
WHERE {{
    ?skill skosxl:prefLabel ?label .
    FILTER(STRSTARTS(STR(?skill), "{ESCO_SKILL_URI_PREFIX}"))
    FILTER(LANGMATCHES(LANG(?label), "en"))
}}
{limit_clause}
"""
        results = run_sparql_query_sync(sparql_flattened)

        for binding in _sparql_bindings(results):
            skill_uri = _sparql_value(binding, "skill")
            label = _sparql_value(binding, "label")
            if skill_uri and label:
                skills.append({"skill_uri": skill_uri, "label": label})

    return skills


def _load_esco_occupations_from_qlever(
    limit: int | None = None,
) -> list[dict[str, Any]]:
    limit_clause = f"LIMIT {limit}" if limit else ""
    
    # 1. Primary Query: Try the flattened SKOS-XL version first 
    # (Matching your successful CURL test)
    sparql = f"""
PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#>
SELECT DISTINCT ?occupation ?label
WHERE {{
    ?occupation skosxl:prefLabel ?label .
    FILTER(STRSTARTS(STR(?occupation), "{ESCO_OCCUPATION_URI_PREFIX}"))
    FILTER(LANGMATCHES(LANG(?label), "en"))
}}
{limit_clause}
"""

    results = run_sparql_query_sync(sparql)
    print(f"-- QLever returned {len(_sparql_bindings(results))} occupations from the flattened query.")

    occupations_by_uri: dict[str, dict[str, Any]] = {}
    
    for binding in _sparql_bindings(results):
        occ_uri = _sparql_value(binding, "occupation")
        label = _sparql_value(binding, "label")
        if occ_uri and label:
            occupations_by_uri[occ_uri] = {
                "occupation_uri": occ_uri,
                "label": label,
                "essential_skills": [],
                "optional_skills": []
            }

    # 2. Fallback: If 0 results, try the SKOS-XL Node path or standard SKOS
    if not occupations_by_uri:
        sparql_fallback = f"""
PREFIX skosxl: <http://www.w3.org/2008/05/skos-xl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
SELECT DISTINCT ?occupation ?label
WHERE {{
    {{
       ?occupation skosxl:prefLabel ?node .
       ?node skosxl:literalForm ?label .
    }} UNION {{
       ?occupation skos:prefLabel ?label .
    }}
    FILTER(STRSTARTS(STR(?occupation), "{ESCO_OCCUPATION_URI_PREFIX}"))
    FILTER(LANGMATCHES(LANG(?label), "en"))
}}
{limit_clause}
"""
        results = run_sparql_query_sync(sparql_fallback)
        for binding in _sparql_bindings(results):
            occ_uri = _sparql_value(binding, "occupation")
            label = _sparql_value(binding, "label")
            if occ_uri and label:
                occupations_by_uri[occ_uri] = {
                    "occupation_uri": occ_uri,
                    "label": label,
                    "essential_skills": [],
                    "optional_skills": []
                }

    if not occupations_by_uri:
        return []

    # 3. Load Essential Skills
    essential_predicates = _env_predicates(
        "ESCO_ESSENTIAL_PREDICATES",
        ["http://data.europa.eu/esco/model#relatedEssentialSkill"],
    )
    if essential_predicates:
        path = _sparql_property_path(essential_predicates)
        sparql_ess = f"""
SELECT DISTINCT ?occupation ?skill WHERE {{
  ?occupation {path} ?skill .
  FILTER(STRSTARTS(STR(?occupation), "{ESCO_OCCUPATION_URI_PREFIX}"))
  FILTER(STRSTARTS(STR(?skill), "{ESCO_SKILL_URI_PREFIX}"))
}}"""
        res_ess = run_sparql_query_sync(sparql_ess)
        for b in _sparql_bindings(res_ess):
            o_uri, s_uri = _sparql_value(b, "occupation"), _sparql_value(b, "skill")
            if o_uri in occupations_by_uri:
                occupations_by_uri[o_uri]["essential_skills"].append(s_uri)

    # 4. Load Optional Skills
    optional_predicates = _env_predicates(
        "ESCO_OPTIONAL_PREDICATES",
        ["http://data.europa.eu/esco/model#relatedOptionalSkill"],
    )
    if optional_predicates:
        path = _sparql_property_path(optional_predicates)
        sparql_opt = f"""
SELECT DISTINCT ?occupation ?skill WHERE {{
  ?occupation {path} ?skill .
  FILTER(STRSTARTS(STR(?occupation), "{ESCO_OCCUPATION_URI_PREFIX}"))
  FILTER(STRSTARTS(STR(?skill), "{ESCO_SKILL_URI_PREFIX}"))
}}"""
        res_opt = run_sparql_query_sync(sparql_opt)
        for b in _sparql_bindings(res_opt):
            o_uri, s_uri = _sparql_value(b, "occupation"), _sparql_value(b, "skill")
            if o_uri in occupations_by_uri:
                occupations_by_uri[o_uri]["optional_skills"].append(s_uri)

    return list(occupations_by_uri.values())


def load_esco_skills() -> list[dict[str, Any]]:
    source = os.getenv("ESCO_SOURCE", "qlever").lower()
    limit = _env_int("ESCO_SKILLS_LIMIT")

    if source in {"qlever", "auto"}:
        try:
            skills = _load_esco_skills_from_qlever(limit)
            if skills:
                return skills
        except Exception as exc:
            print(f"QLever skills load failed, falling back: {exc}")

    root = _project_root()
    path = root / "data" / "esco_skills.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return DEFAULT_SKILLS


def load_esco_occupations() -> list[dict[str, Any]]:
    source = os.getenv("ESCO_SOURCE", "qlever").lower()
    limit = _env_int("ESCO_OCCUPATIONS_LIMIT")

    if source in {"qlever", "auto"}:
        try:
            occupations = _load_esco_occupations_from_qlever(limit)
            if occupations:
                return occupations
        except Exception as exc:
            print(f"QLever occupations load failed, falling back: {exc}")

    root = _project_root()
    path = root / "data" / "esco_occupations.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return DEFAULT_OCCUPATIONS


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def dummy_embed(texts: list[str]) -> np.ndarray:
    embeddings = np.zeros((len(texts), 8), dtype=float)
    for i, text in enumerate(texts):
        for token in _tokenize(text):
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest, 16) % embeddings.shape[1]
            embeddings[i, idx] += 1.0

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return embeddings / norms


def _overlap_score(label_tokens: list[str], text_tokens: set[str]) -> float:
    if not label_tokens:
        return 0.0
    matches = sum(1 for token in label_tokens if token in text_tokens)
    return matches / len(label_tokens)


def course_to_text(course: dict[str, Any]) -> str:
    parts: list[str] = []
    title = course.get("title")
    if title:
        parts.append(str(title))

    objectives = course.get("learning_objectives") or course.get("learning-objectives")
    if objectives:
        parts.append(" ".join(objectives))

    fields = course.get("fields", {})
    for key in (
        "Course content",
        "Scope and form",
        "Course type",
        "Department",
        "Language of instruction",
    ):
        value = fields.get(key)
        if value:
            parts.append(str(value))

    return "\n".join(parts)


def _course_metadata(course: dict[str, Any]) -> dict[str, Any]:
    return {
        "course_number": course.get("course_code"),
        "title": course.get("title"),
    }


def _llm_stub_matches(
    course_text: str, skills: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    api_key = os.getenv("CAMPUSAI_API_KEY")
    if not api_key:
        raise ValueError("CAMPUSAI_API_KEY is not set")

    api_base = os.getenv(
        "CAMPUSAI_BASE_URL", "https://chat.campusai.compute.dtu.dk/api"
    )
    model = os.getenv("CAMPUSAI_MODEL")

    skills_list = "\n".join(
        f"- {skill['skill_uri']} | {skill['label']}" for skill in skills
    )

    system_prompt = (
        "You are mapping DTU course descriptions to ESCO skills. "
        "Select up to 8 relevant skills from the provided list. "
        'Return JSON only with the format: {"matches": ['
        '{"skill_uri": "...", "label": "...", "confidence": 0.0}]} '
        "Confidence must be between 0 and 1. Only use skills from the list."
    )
    user_prompt = f"Course text:\n{course_text}\n\nSkills:\n{skills_list}"

    client = openai.OpenAI(api_key=api_key, base_url=api_base)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    content = response.choices[0].message.content or ""

    matches_raw = _parse_llm_matches(content)
    return _normalize_llm_matches(matches_raw, skills)


def _parse_llm_matches(content: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            return []
        try:
            data = json.loads(content[start : end + 1])
        except json.JSONDecodeError:
            return []

    if isinstance(data, dict):
        matches = data.get("matches", [])
    elif isinstance(data, list):
        matches = data
    else:
        return []

    if not isinstance(matches, list):
        return []
    return [m for m in matches if isinstance(m, dict)]


def _normalize_llm_matches(
    matches: list[dict[str, Any]], skills: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    skills_by_uri = {skill["skill_uri"]: skill["label"] for skill in skills}
    skills_by_label = {skill["label"].lower(): skill["skill_uri"] for skill in skills}
    merged: dict[str, dict[str, Any]] = {}

    for item in matches:
        skill_uri = item.get("skill_uri") or item.get("uri")
        label = item.get("label")

        if skill_uri not in skills_by_uri and label:
            skill_uri = skills_by_label.get(str(label).lower())

        if skill_uri not in skills_by_uri:
            continue

        label = skills_by_uri[skill_uri]
        confidence = item.get("confidence", 0.5)
        try:
            confidence = float(confidence)
        except (TypeError, ValueError):
            confidence = 0.5
        confidence = max(0.0, min(1.0, confidence))

        current = merged.get(skill_uri)
        if current is None or confidence > current["confidence"]:
            merged[skill_uri] = {
                "skill_uri": skill_uri,
                "label": label,
                "confidence": round(confidence, 2),
            }

    return list(merged.values())


def _load_sparse_index(
    texts: list[str], cache_dir: Path
) -> tuple[TfidfVectorizer, Any]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    vectorizer_path = cache_dir / "tfidf_vectorizer.joblib"
    matrix_path = cache_dir / "tfidf_matrix.joblib"

    if not texts:
        return TfidfVectorizer(lowercase=True), np.zeros((0, 0))

    if vectorizer_path.exists() and matrix_path.exists():
        vectorizer = load(vectorizer_path)
        matrix = load(matrix_path)
        if getattr(matrix, "shape", (0,))[0] == len(texts):
            return vectorizer, matrix

    vectorizer = TfidfVectorizer(lowercase=True)
    matrix = vectorizer.fit_transform(texts)
    dump(vectorizer, vectorizer_path)
    dump(matrix, matrix_path)
    return vectorizer, matrix


def _init_chroma_collection(
    course_ids: list[str],
    course_texts: list[str],
    metadatas: list[dict[str, Any]],
    embed_fn: callable,
    chroma_dir: Path,
) -> tuple[Any, bool]:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name="courses", metadata={"hnsw:space": "cosine"}
    )

    if not course_ids:
        return collection, False

    if collection.count() == len(course_ids):
        return collection, False

    embeddings = embed_fn(course_texts)
    collection.upsert(
        ids=course_ids,
        documents=course_texts,
        embeddings=embeddings.tolist(),
        metadatas=metadatas,
    )
    return collection, True


def _course_ects(course: dict[str, Any]) -> float | None:
    fields = course.get("fields", {})
    ects = fields.get("Point( ECTS )")
    if ects is None:
        return None
    try:
        return float(ects)
    except (TypeError, ValueError):
        return None


def _course_description(course: dict[str, Any]) -> str | None:
    fields = course.get("fields", {})
    for key in ("Course content", "Scope and form"):
        value = fields.get(key)
        if value:
            return str(value)
    return None




def ensure_esco_loaded(state: dict[str, Any]) -> None:
    if state.get("esco_loaded"):
        return

    skills = load_esco_skills()
    occupations = load_esco_occupations()
    print(f"---Data loaded:\n\t {len(skills)} skills,\n\t {len(occupations)} occupations")
    esco_state = compute_esco_indexes(
        state["courses"],
        state["course_map"],
        skills,
        occupations,
    )
    # print(esco_state)
    # state.update(esco_state)
    # state["esco_loaded"] = True
    return esco_state


def build_pipeline() -> dict[str, Any]:
    print("Building data pipeline and loading resources...")
    courses, courses_path = load_courses_jsonl()
    print(f"Loaded {len(courses)} courses from {courses_path or 'default data'}")
    lazy_esco = _env_flag("ESCO_LAZY_LOAD", True)
    if lazy_esco:
        skills: list[dict[str, Any]] = []
        occupations: list[dict[str, Any]] = []
        print("Skipping ESCO load on startup (lazy mode enabled)")
    else:
        skills = load_esco_skills()
        print(f"Loaded {len(skills)} skills")
        occupations = load_esco_occupations()
        print(f"Loaded {len(occupations)} occupations")

    root = _project_root()
    course_texts: list[str] = []
    course_ids: list[str] = []
    course_metadatas: list[dict[str, Any]] = []
    for course in courses:
        course_id = course.get("course_code")
        if not course_id:
            continue
        course_ids.append(course_id)
        course_texts.append(course_to_text(course))
        course_metadatas.append(_course_metadata(course))

    embedding_provider, embed_fn = select_embedder()
    chroma_dir = _chroma_dir(root)
    collection, reindexed = _init_chroma_collection(
        course_ids,
        course_texts,
        course_metadatas,
        embed_fn,
        chroma_dir,
    )

    vectorizer, tfidf_matrix = _load_sparse_index(
        course_texts, _vectorizer_cache_dir(root)
    )

    course_map = {course.get("course_code"): course for course in courses}
    esco_state: dict[str, Any] = {
        "skills": skills,
        "occupations": occupations,
        "skill_map": {skill["skill_uri"]: skill for skill in skills},
        "occupation_map": {
            occupation["occupation_uri"]: occupation for occupation in occupations
        },
        "course_skill_matches": {},
        "skill_course_matches": {},
        "occupation_course_matches": {},
        "course_occupation_matches": {course_number: [] for course_number in course_map},
        "skill_occupation_map": {},
    }

    if not lazy_esco:
        cached = _load_esco_indexes(root)

        if cached is not None:
            print("Loaded ESCO indexes from cache")
            esco_state = cached
        else:
            print("Computing ESCO indexes...")
            esco_state = compute_esco_indexes(
                courses,
                course_map,
                skills,
                occupations,
            )
            _save_esco_indexes(esco_state, root)
            print("ESCO indexes saved to cache")
    return {
        "courses": courses,
        "courses_path": courses_path,
        "course_texts": course_texts,
        "course_ids": course_ids,
        "skills": esco_state["skills"],
        "occupations": esco_state["occupations"],
        "course_map": course_map,
        "skill_map": esco_state["skill_map"],
        "occupation_map": esco_state["occupation_map"],
        "chroma_collection": collection,
        "embedding_provider": embedding_provider,
        "embedding_fn": embed_fn,
        "chroma_reindexed": reindexed,
        "tfidf_vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "course_skill_matches": esco_state["course_skill_matches"],
        "skill_course_matches": esco_state["skill_course_matches"],
        "occupation_course_matches": esco_state["occupation_course_matches"],
        "course_occupation_matches": esco_state["course_occupation_matches"],
        "skill_occupation_map": esco_state["skill_occupation_map"],
        "esco_loaded": not lazy_esco,
    }


def get_course_detail(
    course_number: str, state: dict[str, Any]
) -> dict[str, Any] | None:
    esco_state = ensure_esco_loaded(state)
    course = state["course_map"].get(course_number)
    print(f"Getting details for course {course_number}: {'found' if course else 'not found'}")
    if course is None:
        return None
    return {
        "course_number": course_number,
        "title": course.get("title"),
        "description": _course_description(course),
        "ects": _course_ects(course),
        "matched_skills": esco_state["course_skill_matches"].get(course_number, []),
    }


def get_course_skills(
    course_number: str, state: dict[str, Any]
) -> dict[str, Any] | None:
    esco_loaded = ensure_esco_loaded(state)
    course = state["course_map"].get(course_number)
    if course is None:
        return None
    return {
        "course_number": course_number,
        "matched_skills": esco_loaded["course_skill_matches"].get(course_number, []),
    }


def get_skill_detail(skill_uri: str, state: dict[str, Any]) -> dict[str, Any] | None:
    esco_loaded = ensure_esco_loaded(state)
    skill = esco_loaded["skill_map"].get(skill_uri)
    print(f"Getting details for skill {skill_uri}: {'found' if skill else 'not found'}")
    print(f'skill is {skill}')
    if skill is None:
        return None
    return {
        "skill_uri": skill_uri,
        "label": skill.get("label"),
        "related_occupations": esco_loaded["skill_occupation_map"].get(skill_uri, []),
    }


def get_skill_courses(skill_uri: str, state: dict[str, Any]) -> dict[str, Any] | None:
    esco_loaded = ensure_esco_loaded(state)
    skill = esco_loaded["skill_map"].get(skill_uri)
    print(f"Getting courses for skill {skill_uri}: {'found' if skill else 'not found'}")
    print(f'skill is {skill}')
    if skill is None:
        return None
    return {
        "skill_uri": skill_uri,
        "label": skill.get("label"),
        "courses": esco_loaded["skill_course_matches"].get(skill_uri, []),
    }


def get_course_occupations(
    course_number: str, state: dict[str, Any]
) -> dict[str, Any] | None:
    esco_loaded = ensure_esco_loaded(state)
    if course_number not in state["course_map"]:
        print(f"Course number {course_number} not found in course map.")
        return None
    return {
        "course_number": course_number,
        "relevant_occupations": esco_loaded["course_occupation_matches"].get(
            course_number, []
        ),
    }


def get_occupation_detail(
    occupation_uri: str, state: dict[str, Any]
) -> dict[str, Any] | None:
    esco_loaded = ensure_esco_loaded(state)
    occupation = esco_loaded["occupation_map"].get(occupation_uri)
    if occupation is None:
        return None

    skill_map = esco_loaded["skill_map"]
    print(skill_map.get('http://data.europa.eu/esco/concept-scheme/6c930acd-c104-4ece-acf7-f44fd7333036'))
    essential = [
        {
            "skill_uri": uri,
            "label": skill_map.get(uri, {}).get("label", 'Unknown Skill'),
        }
        for uri in occupation.get("essential_skills", [])
    ]
    optional = [
        {
            "skill_uri": uri,
            "label": skill_map.get(uri, {}).get("label", 'Unknown Skill'),
        }
        for uri in occupation.get("optional_skills", [])
    ]
    recommended = esco_loaded["occupation_course_matches"].get(occupation_uri, [])

    return {
        "occupation": occupation.get("label"),
        "essential_skills": essential,
        "optional_skills": optional,
        "recommended_courses": recommended,
    }


def get_occupation_courses(
    occupation_uri: str, state: dict[str, Any]
) -> dict[str, Any] | None:
    esco_loaded = ensure_esco_loaded(state)
    occupation = state["occupation_map"].get(occupation_uri)
    if occupation is None:
        return None
    return {
        "occupation_uri": occupation_uri,
        "label": occupation.get("label"),
        "courses": esco_loaded["occupation_course_matches"].get(occupation_uri, []),
    }


def _dense_search(
    query: str, state: dict[str, Any], top_k: int
) -> list[dict[str, Any]]:
    collection = state["chroma_collection"]
    embed_fn = state["embedding_fn"]
    query_embedding = embed_fn([query]).tolist()
    result = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
        include=["distances", "metadatas"],#, "ids"],
    )

    ids = result.get("ids", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]

    items: list[dict[str, Any]] = []
    for course_id, distance, meta in zip(ids, distances, metadatas):
        score = 1.0 - float(distance)
        items.append(
            {
                "course_number": course_id,
                "title": (meta or {}).get("title"),
                "score": round(score, 4),
            }
        )
    return items


def _sparse_search(
    query: str, state: dict[str, Any], top_k: int
) -> list[dict[str, Any]]:
    vectorizer = state["tfidf_vectorizer"]
    matrix = state["tfidf_matrix"]
    course_ids = state["course_ids"]
    course_map = state["course_map"]

    if getattr(matrix, "shape", (0, 0))[0] == 0:
        return []

    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(matrix, query_vec).ravel()
    indices = np.argsort(-scores)[:top_k]

    items: list[dict[str, Any]] = []
    for idx in indices:
        course_id = course_ids[idx]
        course = course_map.get(course_id, {})
        items.append(
            {
                "course_number": course_id,
                "title": course.get("title"),
                "score": round(float(scores[idx]), 4),
            }
        )
    return items


def search_courses(
    query: str,
    state: dict[str, Any],
    top_k: int = 10,
    mode: str = "dense",
    alpha: float = 0.5,
) -> dict[str, Any]:
    mode = mode.lower()
    if mode not in {"dense", "sparse", "hybrid"}:
        mode = "dense"

    dense_results: list[dict[str, Any]] = []
    sparse_results: list[dict[str, Any]] = []

    if mode in {"dense", "hybrid"}:
        dense_results = _dense_search(query, state, top_k)
    if mode in {"sparse", "hybrid"}:
        sparse_results = _sparse_search(query, state, top_k)

    if mode == "dense":
        results = dense_results
    elif mode == "sparse":
        results = sparse_results
    else:
        combined: dict[str, dict[str, Any]] = {}
        for item in dense_results:
            combined[item["course_number"]] = {
                **item,
                "score": alpha * item["score"],
            }
        for item in sparse_results:
            entry = combined.get(item["course_number"])
            if entry is None:
                combined[item["course_number"]] = {
                    **item,
                    "score": (1 - alpha) * item["score"],
                }
            else:
                entry["score"] += (1 - alpha) * item["score"]

        results = sorted(
            combined.values(), key=lambda item: item["score"], reverse=True
        )[:top_k]

    return {"query": query, "mode": mode, "results": results}


def _occupation_similarity(question: str, label: str) -> float:
    question_tokens = set(_tokenize(question))
    label_tokens = set(_tokenize(label))
    if not label_tokens:
        return 0.0
    overlap = len(question_tokens & label_tokens)
    return overlap / len(label_tokens)


def match_occupation(question: str, state: dict[str, Any]) -> tuple[str, float]:
    best_uri = ""
    best_score = 0.0
    for occupation_uri, occupation in state["occupation_map"].items():
        score = _occupation_similarity(question, occupation.get("label", ""))
        if score > best_score:
            best_uri = occupation_uri
            best_score = score
    if not best_uri:
        fallback = next(iter(state["occupation_map"].keys()), "")
        return fallback, 0.0
    return best_uri, round(best_score, 2)


def recommend_courses_for_occupation(
    occupation_uri: str,
    state: dict[str, Any],
    completed_courses: list[str] | None = None,
    max_ects: float | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    completed = set(completed_courses or [])
    recommendations = []
    total_ects = 0.0

    for course in state["occupation_course_matches"].get(occupation_uri, []):
        course_number = course["course_number"]
        if course_number in completed:
            continue
        course_info = state["course_map"].get(course_number, {})
        ects = _course_ects(course_info) or 0.0
        if max_ects is not None and total_ects + ects > max_ects:
            continue
        recommendations.append(course)
        total_ects += ects
        if len(recommendations) >= limit:
            break
    return recommendations


def handle_query(request: Any, state: dict[str, Any]) -> dict[str, Any]:
    ensure_esco_loaded(state)
    occupation_uri, confidence = match_occupation(request.question, state)
    occupation = state["occupation_map"].get(occupation_uri, {})

    constraints = request.constraints or {}
    max_ects = None
    if isinstance(constraints, dict):
        try:
            max_ects = float(constraints.get("max_ects"))
        except (TypeError, ValueError):
            max_ects = None

    recommended = recommend_courses_for_occupation(
        occupation_uri,
        state,
        completed_courses=request.completed_courses,
        max_ects=max_ects,
    )

    return {
        "interpreted_intent": "occupation_planning",
        "matched_occupation": {
            "uri": occupation_uri,
            "label": occupation.get("label", ""),
            "confidence": confidence,
        },
        "recommended_courses": recommended,
        "explanation": "Courses selected based on essential skill coverage.",
    }


def compute_esco_indexes(
    courses: list[dict[str, Any]],
    course_map: dict[str, dict[str, Any]],
    skills: list[dict[str, Any]],
    occupations: list[dict[str, Any]],
) -> dict[str, Any]:
    skill_map = {skill["skill_uri"]: skill for skill in skills}
    occupation_map = {
        occupation["occupation_uri"]: occupation for occupation in occupations
    }
    print(f"---Computing skill matches for {len(courses)} courses and {len(skills)} skills...")
    course_skill_matches: dict[str, list[dict[str, Any]]] = {}
    course_skill_uris: dict[str, set[str]] = {}
    skill_course_matches: dict[str, list[dict[str, Any]]] = {
        skill["skill_uri"]: [] for skill in skills
    }

    for course in courses:
        course_number = course.get("course_code")
        if not course_number:
            continue
        matches = match_skills_for_course(course, skills)
        course_skill_matches[course_number] = matches
        course_skill_uris[course_number] = {match["skill_uri"] for match in matches}
        for match in matches:
            skill_course_matches[match["skill_uri"]].append(
                {
                    "course_number": course_number,
                    "title": course.get("title"),
                    "confidence": match["confidence"],
                }
            )

    for courses_list in skill_course_matches.values():
        courses_list.sort(key=lambda item: item["confidence"], reverse=True)

    skill_occupation_map: dict[str, list[str]] = {
        skill["skill_uri"]: [] for skill in skills
    }
    for occupation in occupations:
        for skill_uri in occupation.get("essential_skills", []) + occupation.get(
            "optional_skills", []
        ):
            skill_occupation_map.setdefault(skill_uri, []).append(
                occupation["occupation_uri"]
            )

    occupation_course_matches: dict[str, list[dict[str, Any]]] = {}
    course_occupation_matches: dict[str, list[dict[str, Any]]] = {
        course_number: [] for course_number in course_map
    }

    for occupation in occupations:
        occupation_uri = occupation["occupation_uri"]
        label = occupation.get("label")
        essential = occupation.get("essential_skills", [])
        if not essential:
            occupation_course_matches[occupation_uri] = []
            continue

        occ_courses: list[dict[str, Any]] = []
        for course_number, course in course_map.items():
            matched_uris = course_skill_uris.get(course_number, set())
            covered = [uri for uri in essential if uri in matched_uris]
            if not covered:
                continue
            coverage = round(len(covered) / len(essential), 2)
            covered_skills = [
                {
                    "skill_uri": uri,
                    "label": skill_map.get(uri, {}).get("label", uri),
                }
                for uri in covered
            ]
            occ_courses.append(
                {
                    "course_number": course_number,
                    "skill_coverage": coverage,
                    "covered_skills": covered_skills,
                }
            )
            course_occupation_matches[course_number].append(
                {
                    "occupation_uri": occupation_uri,
                    "label": label,
                    "coverage_score": coverage,
                }
            )

        occ_courses.sort(key=lambda item: item["skill_coverage"], reverse=True)
        occupation_course_matches[occupation_uri] = occ_courses

    return {
        "skills": skills,
        "occupations": occupations,
        "skill_map": skill_map,
        "occupation_map": occupation_map,
        "course_skill_matches": course_skill_matches,
        "skill_course_matches": skill_course_matches,
        "occupation_course_matches": occupation_course_matches,
        "course_occupation_matches": course_occupation_matches,
        "skill_occupation_map": skill_occupation_map,
    }


def match_skills_for_course(
    course: dict[str, Any],
    skills: list[dict[str, Any]],
    max_matches: int = 10,
) -> list[dict[str, Any]]:
    course_text = course_to_text(course)
    text_tokens = set(_tokenize(course_text))

    keyword_matches: list[dict[str, Any]] = []
    for skill in skills:
        label_tokens = _tokenize(skill["label"])
        score = _overlap_score(label_tokens, text_tokens)
        if score <= 0:
            continue
        keyword_matches.append(
            {
                "skill_uri": skill["skill_uri"],
                "label": skill["label"],
                "confidence": round(score, 2),
            }
        )

    llm_matches: list[dict[str, Any]] = []
    if _env_flag("ENABLE_LLM_MATCHING", False):
        try:
            llm_matches = _llm_stub_matches(course_text, skills)
        except Exception as exc:
            print(f"LLM matching skipped due to error: {exc}")

    merged: dict[str, dict[str, Any]] = {}
    for match in keyword_matches + llm_matches:
        existing = merged.get(match["skill_uri"])
        if existing is None or match["confidence"] > existing["confidence"]:
            merged[match["skill_uri"]] = match

    matches = sorted(merged.values(), key=lambda item: item["confidence"], reverse=True)
    return matches[:max_matches]




{
  "course_code": "02451",
  "url": "https://kurser.dtu.dk/course/02451",
  "title": "02451 Introduction to Machine Learning",
  "academic_year": "2025/2026",
  "fields": {
    "Danish title": "Introduktion til machine learning",
    "Language of instruction": "English",
    "Point( ECTS )": 5,
    "Course type": "BScOffered as a single course",
    "Schedule": "Spring F4A (Tues 13-17)",
    "Location": "Campus Lyngby",
    "Scope and form": "The activities alternate between lectures, problem classes and hands-on Python exercises.",
    "Duration of Course": "13 weeks",
    "Date of examination": "The exam will be held on a special day: Click \"Date of examination\" to the left to see the date see DT",
    "Type of assessment": [
      "Written examination and exercises",
      "Approval of assignments is a prerequisite for passing the course."
    ],
    "Exam duration": "Written exam: 4 hours",
    "Aid": [
      "No Aid : Multiple choice.",
      "No electronic aids (e.g., calculators).",
      "Only allowed to bring two A4 sheets of handwritten notes."
    ],
    "Evaluation": "7 step scale , external examiner",
    "Previous Course": 2450,
    "Not applicable together with": "02450/02452",
    "Academic prerequisites": "(01001/01002/01003/01004/01005).­(02402/02403).­(02002/02101/02102/02525/02631/02632/02633/02692) , Basic course in linear algebra and calculus, basic knowledge of probability theory or statistics, basic knowledge of Python.",
    "Responsible": "Morten Mørup , Ph. (+45) 4525 3900 , mmor@dtu.dk",
    "Course co-responsible": [
      "Bjørn Sand Jensen (Primary contact person) , bjje@dtu.dk",
      "Georgios Arvanitidis , Lyngby Campus, Building 321, Ph. (+45) 4525 5241 , gear@dtu.dk"
    ],
    "Department": "01 Department of Applied Mathematics and Computer Science",
    "Home page": "http://www.compute.dtu.dk/courses/02450",
    "Registration Sign up": "At the Studyplanner",
    "Green challenge participation": "Please contact the teacher for information on whether this course gives the student the opportunity to prepare a project that may participate in DTU´s Study Conference on sustainability, climate technology, and the environment (GRØN DYST). More infor http://www.groendyst.dtu.dk/english"
  },
  "learning_objectives": [
    "Explain the major steps involved in data modeling from preparing the data, modeling the data to evaluating and disseminating the results.",
    "Discuss key machine learning concepts such as feature extraction, cross-validation, generalization and over-fitting, prediction, curse of dimensionality, and the bias-variance trade-off.",
    "Match practical problems to standard data modeling problems such as dimensionality reduction, regression, classification, density estimation and clustering.",
    "Explain how a relevant set of machine learning methods works.",
    "Describe assumptions, strengths, and limitations of relevant machine learning methods.",
    "Apply, modify, and implement central aspects of machine learning algorithms in Python",
    "Apply visualization techniques and statistics to evaluate model performance, identify patterns and data issues.",
    "Select, combine and modify data modeling tools in order to analyze data and disseminate the results of the analysis."
  ]
}