from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    """A single test case in an evaluation dataset."""

    query_id: str
    question: str
    user_id: str | None = Field(
        default=None,
        description="Customer user_id for scene classification via order data. If None, falls back to hardcoded scene.",
    )
    expected_scene: str | None = Field(
        default=None,
        description="Expected scene (presale/insale/aftersale) for assertion. Optional.",
    )
    relevant_source_patterns: list[str] = Field(
        default_factory=list,
        description="Substrings that match a relevant document's 'source' field (case-insensitive).",
    )
    noise_contexts: list[str] = Field(
        default_factory=list,
        description="Irrelevant text snippets injected into the judge prompt to test noise sensitivity.",
    )
    ground_truth_answer: str | None = Field(
        default=None,
        description="Optional reference answer for future metrics (not used by current judges).",
    )


class RetrievedDoc(BaseModel):
    """Normalized representation of a single retrieval result."""

    source: str
    score: float
    snippet: str
    file_name: str = ""
    page_number: int = 0
    rank: int = 0


class RetrievalEvalResult(BaseModel):
    """Per-query retrieval metrics."""

    query_id: str
    recall_at_k: float
    hit_rate_at_k: int  # 0 or 1
    mrr: float
    precision_at_k: float
    retrieved_count: int
    relevant_retrieved_count: int
    total_relevant: int


class GenerationEvalResult(BaseModel):
    """Per-query LLM-as-a-judge scores."""

    query_id: str
    faithfulness_score: int
    faithfulness_justification: str
    answer_relevance_score: int
    answer_relevance_justification: str
    context_usage_score: int
    context_usage_justification: str
    noise_sensitivity_score: int | None = None
    noise_sensitivity_justification: str | None = None


class EvalReport(BaseModel):
    """Aggregated evaluation report."""

    dataset_name: str
    num_queries: int
    k: int

    avg_recall_at_k: float
    avg_hit_rate_at_k: float
    avg_mrr: float
    avg_precision_at_k: float

    avg_faithfulness: float | None = None
    avg_answer_relevance: float | None = None
    avg_context_usage: float | None = None
    avg_noise_sensitivity: float | None = None

    retrieval_details: list[RetrievalEvalResult] = Field(default_factory=list)
    generation_details: list[GenerationEvalResult] | None = None
