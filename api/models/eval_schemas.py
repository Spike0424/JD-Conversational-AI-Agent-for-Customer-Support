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
        description=(
            "Patterns used to identify relevant database knowledge rows by aliases "
            "(case-insensitive substring match)."
        ),
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

    document_id: int | str | None = None
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
    f1_score: float = 0.0
    retrieved_count: int
    relevant_retrieved_count: int
    # Number of enabled, relevant knowledge rows in the database candidate scope.
    total_relevant: int
    # Raw retrieval output for debugging + JSON dump inspection.
    # Each entry: {source, score, snippet, file_name, page_number, rank}
    retrieved_docs: list[dict] = Field(default_factory=list)
    relevant_source_patterns: list[str] = Field(default_factory=list)
    # query: the raw user question, verbatim. search_terms: the tokenized
    # terms BM25-like scoring actually consumed (incl. phrase/synonym expansion).
    query: str | None = None
    search_terms: list[str] = Field(default_factory=list)


class GenerationEvalResult(BaseModel):
    """Per-query LLM-as-a-judge scores.

    Scores are None when the entry is answers-only (judge skipped) —
    the raw agent_answer is still recorded for manual inspection.
    """

    query_id: str
    faithfulness_score: int | None = None
    faithfulness_justification: str | None = None
    answer_relevance_score: int | None = None
    answer_relevance_justification: str | None = None
    context_usage_score: int | None = None
    context_usage_justification: str | None = None
    noise_sensitivity_score: int | None = None
    noise_sensitivity_justification: str | None = None
    unparseable_metric_count: int = 0
    # Raw agent output + reference answer for diff inspection in JSON dumps.
    agent_answer: str | None = None
    ground_truth_answer: str | None = None


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

    unparseable_metric_count: int = 0

    retrieval_details: list[RetrievalEvalResult] = Field(default_factory=list)
    generation_details: list[GenerationEvalResult] | None = None
