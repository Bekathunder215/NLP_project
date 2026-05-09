from pydantic import BaseModel, Field


class SkillRef(BaseModel):
    skill_uri: str
    label: str


class SkillMatch(SkillRef):
    confidence: float


class CourseMatch(BaseModel):
    course_number: str
    title: str | None = None
    confidence: float | None = None


class CourseDetail(BaseModel):
    course_number: str
    title: str | None = None
    description: str | None = None
    ects: float | None = None
    matched_skills: list[SkillMatch] = Field(default_factory=list)


class CourseSkillsResponse(BaseModel):
    course_number: str
    matched_skills: list[SkillMatch] = Field(default_factory=list)


class SkillDetailResponse(BaseModel):
    skill_uri: str
    label: str
    related_occupations: list[str] = Field(default_factory=list)


class SkillCoursesResponse(BaseModel):
    skill_uri: str
    label: str
    courses: list[CourseMatch] = Field(default_factory=list)


class OccupationMatch(BaseModel):
    occupation_uri: str
    label: str
    coverage_score: float


class CourseOccupationsResponse(BaseModel):
    course_number: str
    relevant_occupations: list[OccupationMatch] = Field(default_factory=list)


class RecommendedCourse(BaseModel):
    course_number: str
    skill_coverage: float
    covered_skills: list[SkillRef] = Field(default_factory=list)


class OccupationDetailResponse(BaseModel):
    occupation: str
    essential_skills: list[SkillRef] = Field(default_factory=list)
    optional_skills: list[SkillRef] = Field(default_factory=list)
    recommended_courses: list[RecommendedCourse] = Field(default_factory=list)


class OccupationCoursesResponse(BaseModel):
    occupation_uri: str
    label: str
    courses: list[RecommendedCourse] = Field(default_factory=list)


class QueryRequest(BaseModel):
    question: str
    completed_courses: list[str] = Field(default_factory=list)
    constraints: dict | None = None


class MatchedOccupation(BaseModel):
    uri: str
    label: str
    confidence: float


class QueryResponse(BaseModel):
    interpreted_intent: str
    matched_occupation: MatchedOccupation
    recommended_courses: list[RecommendedCourse] = Field(default_factory=list)
    explanation: str


class QleverQueryRequest(BaseModel):
    text: str | None = None
    sparql: str | None = None


class QleverQueryResponse(BaseModel):
    query: str | None = None
    sparql: str
    results: dict


class SearchResult(BaseModel):
    course_number: str
    title: str | None = None
    score: float


class SearchResponse(BaseModel):
    query: str
    mode: str
    results: list[SearchResult] = Field(default_factory=list)
