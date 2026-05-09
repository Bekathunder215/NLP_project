# Project proposal

================

## Match DTU courses with ESCO ontology

Modified from originally suggestion by Maria Morais Veiga Madeira Vendas

From European Union Website: "ESCO (European Skills, Competences, Qualifications and Occupations) is
the European multilingual classification of Skills, Competences and Occupations. ESCO works as a
dictionary, describing, identifying and classifying professional occupations and skills relevant for
the EU labour market and education and training."

The idea with the project is to map which (DTU) courses provide which ESCO skills. For instance, In
ESCO, 'data scientist' has among its essential skills and competence: 'build recommender systems',
'design database scheme' and 'normalise data'. If you search the DTU course database for 'build
recommender systems' the single course 42578 Advanced Business Analytics comes up. In its current
course description, 'recommender systems' are mentioned. A suitably prompted LLM might be able to
identify that the 42578 course to some degree provide a ESCO data scientist skill. Running this
across all DTU course and all ESCO occupations you can build a complete mapping which allows you to
answer a question such as 'What courses should I take to become a 'industrial mobile devices
software developer' or an inverse question "Which ESCO occupations do the course 02451 provide
skills for".

How to do the mapping is not trivial. One method is to compare all courses with all ESCO occupation
and their skills. This approach could require millions of requests to an LLM. The number of ESCO
occupations could be limited to just occupations relevant for engineers. The number of courses
examined could also be limited by restricting to certain DTU departments.

### Endpoints

There are various API endpoints possible with base path, say '/api/v1'.

Request: `GET /courses/{course_number}`

Example response

```{
  "course_number": "42578",
  "title": "Advanced Business Analytics",
  "description": "...",
  "ects": 5,
  "matched_skills": [
    {
      "skill_uri": "http://data.europa.eu/esco/skill/505e4ef3-7ce4-437d-b7b4-5c608f71c258",
      "label": "build recommender systems",
      "confidence": 0.87
    }
  ]
}
```

`GET /skills/{skill_uri}`

`GET /occupations/{occupation_uri}`

`GET /courses/{course_number}/skills`

`GET /skills/{skill_uri}/courses`

`GET /occupations/{occupation_uri}/courses`

Example response:

```
{
  "occupation": "data scientist",
  "essential_skills": [...],
  "optional_skills": [...],
  "recommended_courses": [
    {
      "course_number": "42578",
      "skill_coverage": 0.42,
      "covered_skills": [...]
    }
  ]
}
```

`POST /planner/occupation`

Example request

```
{
  "occupation_uri": "http://data.europa.eu/esco/occupation/258e46f9-0075-4a2e-adae-1ff0477e0f30",
  "completed_courses": ["02450"],
  "max_ects": 30,
  "strategy": "maximize_skill_coverage"
}
```

Example response:

```
{
  "target_occupation": "industrial mobile devices software developer",
  "skill_gap": [...],
  "recommended_courses": [...],
  "coverage_after_plan": 0.78
}
```

`GET /courses/{course_number}/occupations`

Example response:

```
{
  "course_number": "02451",
  "relevant_occupations": [
    {
      "occupation_uri": "...",
      "label": "data scientist",
      "coverage_score": 0.31
    }
  ]
}
```

`POST /query`

Example request

```
{
  "question": "What courses should I take to become an industrial mobile devices software developer?",
  "completed_courses": ["02451"],
  "constraints": {
    "max_ects": 25
  }
}
```

Example response:

```
{
  "interpreted_intent": "occupation_planning",
  "matched_occupation": {
    "uri": "...",
    "label": "industrial mobile devices software developer",
    "confidence": 0.91
  },
  "recommended_courses": [...],
  "explanation": "Courses selected based on essential skill coverage."
}
```
