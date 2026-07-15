from __future__ import annotations

import json
import logging
import re

from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.evaluation.schemas import GenerationEvalResult, RetrievedDoc

logger = logging.getLogger(__name__)

_MAX_SNIPPET_CHARS = 1500

# Justifications produced when the judge output could not be parsed or the LLM call failed.
# Used to detect non-real scores so they can be tracked separately from real 1-5 ratings.
_UNPARSEABLE_MARKERS = (
    "LLM judge returned unparseable output",
    "LLM judge call failed",
)

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system.
Your task is to score the quality of an answer on a 1-5 Likert scale using a specific rubric.
Respond ONLY with a JSON object: {"score": <int 1-5>, "justification": "<brief explanation in Chinese>"}
Do not include any other text."""


def build_judge_llm() -> ChatOpenAI | None:
    """Create a deterministic LLM instance for evaluation judging.

    Returns None if the required API key or base URL is not configured.
    """
    settings = get_settings()
    if not settings.openai_api_key or not settings.openai_base_url:
        logger.warning("LLM judge unavailable: OPENAI_API_KEY or OPENAI_BASE_URL is not set.")
        return None
    return ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        temperature=0.0,
        request_timeout=120,
    )


def _extract_first_json_object(text: str) -> str | None:
    """Return the substring from the first '{' to its matching '}', or None."""
    start = text.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _parse_judge_response(raw_text: str) -> tuple[int, str]:
    candidate = _extract_first_json_object(raw_text)
    if candidate:
        try:
            data = json.loads(candidate)
            score = int(data.get("score", 3))
            justification = str(data.get("justification", ""))
            return _clamp_score(score), justification
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    score_match = re.search(r"\"score\"\s*:\s*(\d+)", raw_text)
    if score_match:
        score = int(score_match.group(1))
        return _clamp_score(score), raw_text.strip()[:500]

    logger.warning("Judge returned unparseable output: %s", raw_text[:200])
    return 3, f"LLM judge returned unparseable output: {raw_text[:200]}"


def _clamp_score(score: int) -> int:
    return max(1, min(5, score))


def _judge(prompt: str, judge_llm: ChatOpenAI) -> tuple[int, str]:
    full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{prompt}"
    try:
        response = judge_llm.invoke(full_prompt)
    except Exception:
        logger.exception("LLM judge call failed")
        return 3, "LLM judge call failed (network/timeout/rate-limit)"
    raw = str(response.content) if hasattr(response, "content") else str(response)
    return _parse_judge_response(raw)


def _build_context_block(retrieved_docs: list[RetrievedDoc]) -> str:
    if not retrieved_docs:
        return "(No context retrieved)"
    lines = []
    for doc in retrieved_docs:
        meta = f"source={doc.source}"
        if doc.file_name:
            meta += f", file={doc.file_name}"
        if doc.page_number:
            meta += f", page={doc.page_number}"
        snippet = doc.snippet[:_MAX_SNIPPET_CHARS]
        if len(doc.snippet) > _MAX_SNIPPET_CHARS:
            snippet += "…(truncated)"
        lines.append(f"[{doc.rank}] ({meta}, score={doc.score:.4f})\n{snippet}")
    return "\n\n".join(lines)


def _is_unparseable(justification: str | None) -> bool:
    return bool(justification) and any(justification.startswith(m) for m in _UNPARSEABLE_MARKERS)


def _evaluate_metric(
    prompt_template: str,
    question: str,
    answer: str,
    context: str,
    judge_llm: ChatOpenAI,
) -> tuple[int, str]:
    prompt = prompt_template.format(question=question, context=context, answer=answer)
    return _judge(prompt, judge_llm)


# ---------------------------------------------------------------------------
# Prompt templates for each metric
# ---------------------------------------------------------------------------

FAITHFULNESS_PROMPT = """\
Evaluate the FAITHFULNESS of the answer.
Faithfulness measures whether all claims in the answer can be derived from the provided context.
An answer is faithful if every factual statement it makes is supported by the context.

Question: {question}

Context:
{context}

Answer: {answer}

Rubric:
1 - Answer is entirely unsupported by context; contains hallucinated facts.
2 - Answer is mostly unsupported; only 1-2 minor supported points among many fabrications.
3 - Answer is partially supported; about half of the claims can be traced to context.
4 - Answer is mostly faithful; only 1 minor unsupported claim or imprecision.
5 - All claims are fully grounded in the provided context; no hallucination.

Provide your evaluation as JSON."""

ANSWER_RELEVANCE_PROMPT = """\
Evaluate the ANSWER RELEVANCE of the answer.
Answer relevance measures how well the answer addresses the user's question.
A relevant answer stays on topic and directly responds to what was asked.

Question: {question}

Context:
{context}

Answer: {answer}

Rubric:
1 - Answer is completely off-topic or non-responsive.
2 - Answer touches the domain but misses the core question.
3 - Answer partially addresses the question but includes irrelevant tangents.
4 - Answer mostly addresses the question; minor digressions only.
5 - Answer directly and completely addresses the question.

Provide your evaluation as JSON."""

CONTEXT_USAGE_PROMPT = """\
Evaluate the CONTEXT USAGE of the answer.
Context usage measures how effectively the answer leverages the provided retrieved context.
A high score means the answer synthesizes, cites, or meaningfully incorporates relevant context information.

Question: {question}

Context:
{context}

Answer: {answer}

Rubric:
1 - Answer ignores context entirely; no evidence of context usage.
2 - Answer superficially references one context item without meaningful use.
3 - Answer uses some context but misses key relevant passages.
4 - Answer uses most relevant context well; minor missed opportunities.
5 - Answer expertly integrates all relevant context; cites and synthesizes effectively.

Provide your evaluation as JSON."""

NOISE_SENSITIVITY_PROMPT = """\
Evaluate the NOISE SENSITIVITY of the answer.
Noise sensitivity measures whether the answer avoids being misled by irrelevant or distracting context.
The context below includes both relevant retrieved documents and injected noise.
A high score means the answer correctly ignored the noise and stayed focused on relevant information.

Question: {question}

Retrieved Context:
{context}

Noise Context (irrelevant, should be ignored):
{noise_context}

Answer: {answer}

Rubric:
1 - Answer is completely misled by noise; builds response around irrelevant content.
2 - Answer incorporates significant noise content into the response.
3 - Answer shows some influence from noise but remains partially on-topic.
4 - Answer mostly ignores noise; only minor irrelevant detail present.
5 - Answer is completely unaffected by noise; stays focused on relevant information only.

Provide your evaluation as JSON."""


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def evaluate_faithfulness(
    question: str,
    answer: str,
    context: str,
    judge_llm: ChatOpenAI,
) -> tuple[int, str]:
    return _evaluate_metric(FAITHFULNESS_PROMPT, question, answer, context, judge_llm)


def evaluate_answer_relevance(
    question: str,
    answer: str,
    context: str,
    judge_llm: ChatOpenAI,
) -> tuple[int, str]:
    return _evaluate_metric(ANSWER_RELEVANCE_PROMPT, question, answer, context, judge_llm)


def evaluate_context_usage(
    question: str,
    answer: str,
    context: str,
    judge_llm: ChatOpenAI,
) -> tuple[int, str]:
    return _evaluate_metric(CONTEXT_USAGE_PROMPT, question, answer, context, judge_llm)


def evaluate_noise_sensitivity(
    question: str,
    answer: str,
    context: str,
    noise_contexts: list[str],
    judge_llm: ChatOpenAI,
) -> tuple[int, str] | None:
    if not noise_contexts:
        return None
    noise_block = "\n\n".join(
        f"[Noise {i}] {text[:_MAX_SNIPPET_CHARS]}"
        + ("…(truncated)" if len(text) > _MAX_SNIPPET_CHARS else "")
        for i, text in enumerate(noise_contexts, 1)
    )
    prompt = NOISE_SENSITIVITY_PROMPT.format(
        question=question,
        context=context,
        noise_context=noise_block,
        answer=answer,
    )
    return _judge(prompt, judge_llm)


def evaluate_all_generation_metrics(
    query_id: str,
    question: str,
    answer: str,
    retrieved_docs: list[RetrievedDoc],
    noise_contexts: list[str],
    judge_llm: ChatOpenAI,
) -> GenerationEvalResult:
    context = _build_context_block(retrieved_docs)

    faith_score, faith_just = evaluate_faithfulness(question, answer, context, judge_llm)
    rel_score, rel_just = evaluate_answer_relevance(question, answer, context, judge_llm)
    usage_score, usage_just = evaluate_context_usage(question, answer, context, judge_llm)

    noise_result = evaluate_noise_sensitivity(question, answer, context, noise_contexts, judge_llm)
    noise_score, noise_just = noise_result if noise_result else (None, None)

    unparseable_count = sum(
        1 for j in (faith_just, rel_just, usage_just, noise_just) if _is_unparseable(j)
    )

    return GenerationEvalResult(
        query_id=query_id,
        faithfulness_score=faith_score,
        faithfulness_justification=faith_just,
        answer_relevance_score=rel_score,
        answer_relevance_justification=rel_just,
        context_usage_score=usage_score,
        context_usage_justification=usage_just,
        noise_sensitivity_score=noise_score,
        noise_sensitivity_justification=noise_just,
        unparseable_metric_count=unparseable_count,
    )


def aggregate_generation(
    results: list[GenerationEvalResult],
) -> dict[str, float | int | None]:
    if not results:
        return {
            "avg_faithfulness": None,
            "avg_answer_relevance": None,
            "avg_context_usage": None,
            "avg_noise_sensitivity": None,
            "unparseable_metric_count": 0,
        }
    n = len(results)
    noise_scores = [r.noise_sensitivity_score for r in results if r.noise_sensitivity_score is not None]
    return {
        "avg_faithfulness": sum(r.faithfulness_score for r in results) / n,
        "avg_answer_relevance": sum(r.answer_relevance_score for r in results) / n,
        "avg_context_usage": sum(r.context_usage_score for r in results) / n,
        "avg_noise_sensitivity": sum(noise_scores) / len(noise_scores) if noise_scores else None,
        "unparseable_metric_count": sum(r.unparseable_metric_count for r in results),
    }
