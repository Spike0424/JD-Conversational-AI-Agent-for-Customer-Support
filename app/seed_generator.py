"""Generate seed script data from prompt/ directory using LLM."""

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_openai import ChatOpenAI

from app.config import get_settings
from app.db import create_session as create_seed_session, init_db as init_seed_db
from app.seed_models import SeedScript, load_seed_index

logger = logging.getLogger(__name__)

PROMPT_DIR = Path(__file__).resolve().parent.parent / "prompt"

SEED_GENERATION_PROMPT = """你是一个电商客服话术专家。以下是 {platform} 平台的客服话术模板。

请为每个话术模板生成 3 个用户可能说的触发话术（自然口语化的用户提问），以及保留原始话术模板作为回复。

话术内容：
{script_content}

返回 JSON 数组，每个元素包含：
[
  {{"trigger_text": "用户可能说的触发话术", "response_template": "对应的话术模板原文"}}
]

只返回 JSON 数组，不要其他内容。"""


def _get_scripts_from_md(platform: str) -> list[dict[str, str]]:
    """Parse prompt/{platform}/*.md files for script templates."""
    platform_dir = PROMPT_DIR / platform
    if not platform_dir.exists():
        logger.warning("Prompt directory not found: %s", platform_dir)
        return []

    results = []
    for md_file in sorted(platform_dir.glob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        scenario = md_file.stem  # aftersales / presales / sales
        results.append({"scenario": scenario, "content": content, "platform": platform})
    return results


def generate_seeds(platform: str = "", max_per_template: int = 3) -> dict[str, Any]:
    """Generate seed scripts from prompt/ files using LLM.

    Args:
        platform: '京东' or empty = all.
        max_per_template: max trigger utterances per script template.

    Returns: {"generated": int, "errors": list}
    """
    init_seed_db()
    settings = get_settings()

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        model=settings.model_name,
        temperature=0.3,
        request_timeout=120,
    )
    embeddings = FastEmbedEmbeddings(model_name=settings.local_embedding_model_name)

    platforms = [platform] if platform else ["京东"]
    total = 0
    errors = []

    for plat in platforms:
        docs = _get_scripts_from_md(plat)
        if not docs:
            continue

        for doc in docs:
            prompt = SEED_GENERATION_PROMPT.format(
                platform=plat,
                script_content=doc["content"][:4000],
            )
            logger.info("Generating seeds for %s/%s", plat, doc["scenario"])

            try:
                response = llm.invoke(prompt)
                raw = str(response.content) if hasattr(response, "content") else str(response)
                match = re.search(r"\[.*\]", raw, re.DOTALL)
                if not match:
                    errors.append(f"{plat}/{doc['scenario']}: no JSON array in response")
                    continue
                items = json.loads(match.group(0))
            except Exception as exc:
                errors.append(f"{plat}/{doc['scenario']}: {exc}")
                continue

            session = create_seed_session()
            count = 0
            for item in items[:max_per_template * 5]:  # safety cap
                trigger = str(item.get("trigger_text", "")).strip()
                template = str(item.get("response_template", "")).strip()
                if not trigger or not template:
                    continue
                try:
                    vec = np.array(embeddings.embed_query(trigger), dtype=np.float32)
                    seed = SeedScript(
                        platform=plat,
                        scenario=doc["scenario"],
                        trigger_text=trigger,
                        response_template=template,
                        embedding_bytes=vec.tobytes(),
                    )
                    session.add(seed)
                    count += 1
                except Exception as exc:
                    errors.append(f"embed {plat}/{doc['scenario']}: {exc}")
            session.commit()
            session.close()
            total += count
            logger.info("  %s/%s: %d seeds", plat, doc["scenario"], count)

    load_seed_index()
    return {"generated": total, "errors": errors}
