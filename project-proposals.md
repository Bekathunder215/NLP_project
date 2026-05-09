Project proposals
=================

Match DTU courses with ESCO ontology
------------------------------------
Modified from originally suggestion by Maria Morais Veiga Madeira Vendas 

From European Union Website: "ESCO (European Skills, Competences,
Qualifications and Occupations) is the European multilingual
classification of Skills, Competences and Occupations. ESCO works as a
dictionary, describing, identifying and classifying professional
occupations and skills relevant for the EU labour market and education
and training."

The idea with the project is to map which (DTU) courses provide which
ESCO skills. For instance, In ESCO, 'data scientist' has among its
essential skills and competence: 'build recommender systems', 'design
database scheme' and 'normalise data'. If you search the DTU course
database for 'build recommender systems' the single course 42578
Advanced Business Analytics comes up. In its current course description,
'recommender systems' are mentioned. A suitably prompted LLM might
be able to identify that the 42578 course to some degree provide a
ESCO data scientist skill. Running this across all DTU course and all
ESCO occupations you can build a complete mapping which allows you to
answer a question such as 'What courses should I take to become a
'industrial mobile devices software developer' or an inverse
question "Which ESCO occupations do the course 02451 provide skills
for".

How to do the mapping is not trivial. One method is to compare all
courses with all ESCO occupation and their skills. This approach could
require millions of requests to an LLM. The number of ESCO
occupations could be limited to just occupations relevant for
engineers. The number of courses examined could also be limited by
restricting to certain DTU departments.


### Endpoints

There are various API endpoints possible with base path, say '/api/v1'.

Request:
`GET /courses/{course_number}`

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

Academic CV generation with LLMs and knowledge graphs
-----------------------------------------------------
Generation of resumé/CV for an academic is often required in connection with
job applications, research grant applications and promotions. One CV
cannot necessarily be reused unmodified between applications, due to
updates and possible restrictions on the format of the CV, e.g., with
respect to number of pages and number of publications that can be
listed. There may be further customizations wanted, file format and
targeted emphasis based on who the CV is for. Some information for a
CV may be readily available in a knowledge graph (Wikidata) while other
information may be necessary to provide in some other form. Large
language models (LLM) has been used for the creation of resumés.

This project should make a Web service that is based on data from a
knowledge graph and some specification about the CV should generate a
CV with an LLM. It is suggested to get the data from Wikidata.

### Scoping questions
- Input: Wikidata, previous CV, target.
- Fielded or free-form restrictions
- Templating
- Output format to support: LaTeX, Markdown, PDF
- Web application: front end for humans to access the Web service.
- Iterative chat or one-off.

### References
* RealCV - An AI-Powered Resume Generator. https://scholia.toolforge.org/work/Q136309845
* CVTool: Automating Content Variants of CVs. https://scholia.toolforge.org/work/Q136309530
* https://resume.io/


Large language models for text-to-query
---------------------------------------
SPARQL is a query language for knowledge graphs. A user might not want
to formulate a query in the SPARQL language but instead in a natural
language and we want a system that can generate a SPARQL query based
on a query in natural language. 

This is an extension of the text-to-query exercise. Extensions could
be set up a DSPy GEPA or MIPROv2 prompt optimization (probably
difficult), implement better tools for the ReAct tool calling,
particularly schema discovery (probably improve the performance),
implement dynamic few-shot learning where related SPARQL query are
used in the prompt (probably a good idea, requires an RAG-like system
for the questions associated with the SPARQL queries), refinement
based on natural language SPARQL verbalization.

### References
* SPINACH, Monica Lam et al.
* GRASP, Hannah Bast et al.
* Investigating Large Language Models for Text-to-SPARQL Generation. https://scholia.toolforge.org/work/Q136222467


Retrieval-augmented generation for the DTU course database
----------------------------------------------------------
This project is very closely related to the last exercise.
The idea is to build further on the system in one way or another.
For instance, by extracting course dependencies, so as to make a
complete study line proposal so a student user can query "I am a
master student interested in [in certain your interests] with two
years of study. I have taken the following courses [course list],
please provide a suggestion for further study that fits.
It could also combine the course list with information about research
from the publications represented in Wikidata.

Note that the course dependency specification at DTU is somewhat
complex. 


Automatic summarization and sentiment scoring of student feedback
-----------------------------------------------------------------
The study board of DTU Compute reads all the anonymous feedback for
all the courses at DTU. For members of the the study board, board
teachers and students, an XLSX is usually prepared by an adminstrator.
The project should construct a Web service that allow the user to upload
a file and for each course generate sentiment and summarization together
with quotes from individual student feedback. 

The data of interest can unfortunately not be shared! And the data
format is challenging to describe, so this project is not quite
relevant. If you are in a study board you will have access to the
anonymized feedback, but otherwise 


Find DTU researcher given a text
--------------------------------
Given a text, that could represent a (future) project title, an
article title or an project or article abstract find relevant DTU
researchers. This could be useful as a supervisor suggester.
It is envisioned that the publications of the researcher can be embedded
and the embeddings of all articles from DTU can be stored in a matrix.
The query string is compared with the embedding matrix and ordered
results are returns.
The publications of DTU researchers can be identified from one or more
SPARQL queries to triple stores that store Wikidata data. The articles
can be fetched and the title of the article is also available, - but not
the abstract. I have found around 50.000 articles-author pairs.
As part of the Web service you should return some information about
the DTU researcher.

A SPARQL query to get the topics of papers published by a DTU employee
is
```sparql
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
SELECT ?researcher ?paper ?topic WHERE {
  ?researcher wdt:P108 wd:Q1269766 .
  ?paper wdt:P50 ?researcher .
  ?paper wdt:P921 ?topic .
}
```
This can be issued against a QLever Wikidata endpoint https://qlever.dev/wikidata

With ChatGPT, I have vibe coded a Web app that does this relying on my
SPARQL query, the QLever endpoint and a dense vector representation via
CampusAI embedding. It works reasonably well.


Find relevant DTU courses based on a query of scientific document
-----------------------------------------------------------------
Given a scientific article in the form of a PDF file return relevant
DTU courses that supports and are in the area of the article as well
as a summary of the list.


LLM-based exam
--------------
Given a curriculum generate an oral-like exam.

### References
* https://finnaarupnielsen.wordpress.com/2025/10/08/ai-based-oral-exams/


Grade a repository for a project exercise
-----------------------------------------
Automatically grade a repository based on the course curriculum and 
project specification.


Learning objectives miner
-------------------------
Learning Objectives are short sentences that are listed in course
descriptions. In the DTU Course database there usually a set of
learning objectives. 

The project should make a Web service that could help teachers and
students alike in handling of the learning objects. Suggested
endpoints: 

One endpoint could analyse a specific learning objective and check
that formulated correctly according to guidelines, e.g., it should
have an action verb and the verb must be measurable and observable,
so, e.g., verbs that are hard to measure such as 'understand' should
be avoided. Lists exists in the literature with these verbs:
Understand, recognize, acquaint, examine, think, realize, inquire,
obtain, ... Learning objectives should be separate, so two action
verbs should not combined in one learning objective. The spelling and
grammar of learning objectives could also be checked.

Another endpoint could act as a similarity service listing learning
objective from the same an other courses that relates to a specific
learning object.  

Endpoint for a course, e.g. checking each individual learning
objective. (here is a problem what a course is? Identified by a course
number where you fetch the course text, a list of string in JSON,
...?) This could also check against the "course content".

There could also be a learning object suggester.

Endpoint

`POST /learning-objectives/analyse`

Request
```
{
  "text": "Understand the principles of machine learning and apply them to classification problems."
}
```
Example response
```
{
  "text": "Understand the principles of machine learning and apply them to classification problems.",
  "valid": false,
  "issues": [
    {
      "type": "non_measurable_verb",
      "verb": "understand",
      "message": "The verb 'understand' is difficult to measure or observe."
    },
    {
      "type": "multiple_verbs",
      "verbs": ["understand", "apply"],
      "message": "Learning objectives should contain a single action verb."
    }
  ],
  "detected_verbs": [
    {
      "verb": "understand",
      "classification": "non_measurable"
    },
    {
      "verb": "apply",
      "classification": "measurable"
    }
  ],
  "suggestions": [
    "Explain the principles of machine learning.",
    "Apply machine learning methods to classification problems."
  ],
  "score": 0.42
}
```

Request example

`GET /api/v1/courses/42578/learning-objectives/similar?limit=5`

Example response
```
{
  "course_number": "42578",
  "results": [
    {
      "course_number": "02456",
      "learning_objective": "Implement deep learning architectures.",
      "similarity": 0.90
    },
    {
      "course_number": "42186",
      "learning_objective": "Develop predictive models using machine learning.",
      "similarity": 0.87
    }
  ]
}
```

### References
- An analysis of verbs used in the course outcomes of outcome-based
integrated courses at a medical school based on the taxonomy of
educational objectives, https://pmc.ncbi.nlm.nih.gov/articles/PMC6715899/
- How to Write Well-Defined Learning Objectives, https://pmc.ncbi.nlm.nih.gov/articles/PMC5944406/


Podcast
-------
Two "people" chat about the content of the course notes. The two
people are chatbots with different personas that that each have access
to the curriculum (the PDFs) via RAG, ReAct or other means. They pose
questions and answer them. 

The Text-to-speech of CampusAI is poor, so the podcast could be a
written conversation.