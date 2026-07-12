from app.evaluation.schemas import (
    EvalQuery,
    EvalReport,
    GenerationEvalResult,
    RetrievedDoc,
    RetrievalEvalResult,
)
from app.evaluation.retrieval_metrics import aggregate_retrieval, compute_retrieval_metrics, normalize_docs
from app.evaluation.generation_metrics import aggregate_generation, build_judge_llm

__all__ = [
    "EvalQuery",
    "EvalReport",
    "GenerationEvalResult",
    "RetrievedDoc",
    "RetrievalEvalResult",
    "aggregate_retrieval",
    "aggregate_generation",
    "build_judge_llm",
    "compute_retrieval_metrics",
    "normalize_docs",
    "EvalRunner",
]


def __getattr__(name: str):
    if name == "EvalRunner":
        from app.evaluation.runner import EvalRunner

        return EvalRunner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
