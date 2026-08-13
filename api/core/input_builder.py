"""InputBuilder — constructs LLM input messages from session state and dependencies.

Extracted from ReActQAAgent to separate message-building concerns from
agent execution concerns.  Owns: scene prompt loading, session-info formatting,
history compression, user persistence, and the combined message construction.
"""

from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from typing import Any

from api.core.config import Settings
from api.core.retry import ainvoke_with_network_retry
from api.core.token_counter import count_tokens
from api.models.session_store import SessionStore

logger = logging.getLogger(__name__)

E_COMMERCE_PLATFORMS = "京东"

REACT_SYSTEM_PROMPT = f"""You are an intelligent question-answering assistant for a Chinese e-commerce customer service scenario.
Use a ReAct style workflow internally:
1) Reason about the question.
2) Call search_knowledge to retrieve relevant information from the knowledge base about {E_COMMERCE_PLATFORMS} products, policies, shipping, and after-sales procedures.
3) Observe tool outputs.
4) Answer the question based on the retrieved information.

Never expose chain-of-thought in your final response.
Return only a concise and helpful final answer in Chinese unless user asks otherwise."""

AGENS_SYSTEM_PROMPT = "你是一个对话摘要助手，简明总结对话要点。"


class InputBuilder:
    """Builds and manages LLM input messages: system prompt, history, compression."""

    _SCENE_PROMPT_FILE: dict[str, str] = {
        "presale": "presales.md",
        "insale": "sales.md",
        "aftersale": "aftersales.md",
        "mixed": "mixed.md",
    }

    def __init__(
        self,
        session_store: SessionStore,
        settings: Settings,
        agens_llm: Any | None = None,
    ) -> None:
        self._session_store = session_store
        self._settings = settings
        self._agens_llm = agens_llm
        self._prompt_cache: dict[str, str] = {}

    # ── Scene prompt loading ──────────────────────────────────────────

    def load_scene_prompt(self, scene: str) -> str:
        """Load system prompt = _base.md + {scene}.md, fallback to REACT_SYSTEM_PROMPT.

        _base.md is always loaded (role / strategy / tool rules / anti-injection).
        The scene file is appended after it; if scene is empty or file missing,
        only base is used.  Cached per scene after first load.
        """
        cache_key = (scene or "").strip()
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        base_path = Path(self._settings.prompt_dir) / "_base.md"
        try:
            base = base_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            logger.warning("Base prompt not found: %s, using REACT_SYSTEM_PROMPT", base_path)
            return REACT_SYSTEM_PROMPT

        filename = self._SCENE_PROMPT_FILE.get(cache_key) if cache_key else None
        if not filename:
            self._prompt_cache[cache_key] = base
            return base

        scene_path = Path(self._settings.prompt_dir) / filename
        try:
            scene_content = scene_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, OSError):
            logger.warning("Scene prompt not found: %s, using base only", scene_path)
            self._prompt_cache[cache_key] = base
            return base

        if not scene_content:
            self._prompt_cache[cache_key] = base
            return base
        result = base + "\n\n---\n\n" + scene_content
        self._prompt_cache[cache_key] = result
        return result

    # ── Prompt-injection-safe formatting ──────────────────────────────

    @staticmethod
    def safe_value(value: object) -> str:
        """Escape user-supplied values before they're pasted into the system prompt.

        Wraps in <input> tags so the LLM treats it as data, not instructions,
        and doubles braces so it can't break out of an outer f-string template.
        """
        if value is None:
            return ""
        text = str(value).replace("{", "{{").replace("}", "}}")
        text = text.replace("\n", " ").replace("\r", " ")
        return f"<input>{text}</input>"

    @staticmethod
    def format_session_info(dependencies: dict | None) -> str:
        """Build 【当前会话信息】 block, attached at end of system prompt."""
        if not dependencies:
            return ""

        def _line(label: str, value: object) -> str | None:
            if value in (None, ""):
                return None
            return f"- {label}: {InputBuilder.safe_value(value)}"

        lines: list[str] = []
        for key, label in (
            ("shop_id", "shop_id"),
            ("shop_name", "shop_name"),
            ("user_id", "user_id"),
            ("recipient_uid", "recipient_uid"),
            ("customer_uid", "customer_uid"),
            ("context_type", "context_type"),
            ("channel_type", "channel_type"),
        ):
            line = _line(label, dependencies.get(key))
            if line:
                lines.append(line)

        if dependencies.get("goods_id"):
            lines.append(
                f"- goods_id: {InputBuilder.safe_value(dependencies['goods_id'])}"
                "（当前商品，商品知识优先）"
            )
        if dependencies.get("goods_name"):
            lines.append(f"- goods_name: {InputBuilder.safe_value(dependencies['goods_name'])}")
        if dependencies.get("order_sn"):
            lines.append(f"- order_sn: {InputBuilder.safe_value(dependencies['order_sn'])}")

        if not lines:
            return ""
        return "\n\n【当前会话信息】\n" + "\n".join(lines)

    # ── Message construction ──────────────────────────────────────────

    async def build_input_messages(
        self,
        session_id: str,
        question: str,
        scene: str = "",
        dependencies: dict | None = None,
    ) -> list[dict[str, str]]:
        """Assemble the full message list sent to the LLM on each turn."""
        system_prompt = self.load_scene_prompt(scene)
        system_prompt = await self._fill_db_placeholders(system_prompt, scene, dependencies)
        system_prompt += self.format_session_info(dependencies)
        history = self._session_store.load(session_id=session_id)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            *history,
            {"role": "user", "content": question},
        ]
        result = await self._compress_history(messages, session_id)
        logger.info(
            "【LLM 输入】session=%s scene=%s system=%d chars history=%d条 user=%s",
            session_id, scene, len(system_prompt), len(history), question,
        )
        return result

    async def _fill_db_placeholders(
        self, prompt: str, scene: str, dependencies: dict | None
    ) -> str:
        """Replace [DB_*] placeholders with actual KB data before sending to LLM.

        Empty placeholders are blanked out so the LLM doesn't try to fill them
        by calling search_knowledge.  KB queries run in parallel via to_thread.
        """
        if "[DB_" not in prompt:
            return prompt

        shop_id = (dependencies or {}).get("shop_id")
        goods_id = (dependencies or {}).get("goods_id")

        shop_text = ""
        product_text = ""
        if shop_id and scene:
            from api.services.knowledge import get_search_knowledge

            sk = get_search_knowledge()
            if sk:
                tasks = []
                if goods_id:
                    tasks.append(asyncio.to_thread(sk.fetch_product_knowledge, shop_id, scene, goods_id))
                else:
                    tasks.append(asyncio.to_thread(lambda: ""))
                tasks.append(asyncio.to_thread(sk.fetch_shop_advantages, shop_id, scene))
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    if goods_id:
                        product_text = results[0] if isinstance(results[0], str) else ""
                        shop_text = results[1] if isinstance(results[1], str) else ""
                    else:
                        shop_text = results[0] if isinstance(results[0], str) else ""
                except Exception:
                    logger.exception("KB placeholder fill failed shop=%s scene=%s", shop_id, scene)

        return (
            prompt
            .replace("[DB_SHOP_ADVANTAGES]", shop_text)
            .replace("[DB_PRODUCT_KNOWLEDGE]", product_text)
            .replace("[DB_SHIPPING_POLICY]", "")
            .replace("[DB_HOT_MODELS_RECOMMENDATION]", "")
        )

    async def _compress_history(
        self, messages: list[dict[str, str]], session_id: str
    ) -> list[dict[str, str]]:
        """Compress old messages into a summary when token count exceeds threshold.

        The system prompt (first message with role=system) is always preserved
        in the output and excluded from compression scope.  The compressed summary
        is returned inline but NOT persisted to the session store — persisting it
        would create duplicate system-role content on subsequent turns.
        """
        if not self._agens_llm:
            return messages

        token_count = count_tokens(messages, self._settings.model_name)
        threshold = int(
            self._settings.history_window_tokens * self._settings.history_compress_threshold
        )

        if token_count <= threshold:
            return messages

        logger.info(
            "Compressing history session_id=%s tokens=%s threshold=%s",
            session_id, token_count, threshold,
        )

        keep_count = self._settings.history_keep_recent_turns * 2

        # Separate system prompt from the rest — never compress the system prompt
        system_msg = messages[0] if messages and messages[0].get("role") == "system" else None
        rest = messages[1:] if system_msg else messages

        if len(rest) <= keep_count:
            return messages

        old_messages = rest[:-keep_count]
        keep_messages = rest[-keep_count:]

        old_text = "\n".join(f"{m['role']}: {m['content']}" for m in old_messages)
        compress_prompt = f"请总结以下对话历史，保留关键信息：\n\n{old_text}"

        try:
            summary_msg = await ainvoke_with_network_retry(
                self._agens_llm,
                [
                    {"role": "system", "content": AGENS_SYSTEM_PROMPT},
                    {"role": "user", "content": compress_prompt},
                ],
                self._settings.network_retries,
                "Agens",
            )
            summary = (
                getattr(summary_msg, "content", "")
                if hasattr(summary_msg, "content")
                else str(summary_msg)
            )
        except Exception:
            logger.exception("History compression failed session_id=%s", session_id)
            return messages

        if not summary.strip():
            return messages

        logger.info(
            "History compressed session_id=%s old_messages=%s summary_len=%s",
            session_id, len(old_messages), len(summary),
        )

        # Return system prompt + summary + recent messages.
        # DO NOT persist the summary to the session store — it's already
        # included inline for this turn, and persisting it would cause
        # duplicate system-role content on the next turn.
        result = [system_msg] if system_msg else []
        result.append({"role": "system", "content": f"[历史摘要] {summary.strip()}"})
        result.extend(keep_messages)
        return result

    # ── User messages persistence ─────────────────────────────────────

    def persist_user(self, session_id: str, question: str) -> None:
        self._session_store.append(session_id=session_id, role="user", content=question)

    # ── Output normalization ──────────────────────────────────────────

    @staticmethod
    def normalize_text(text: str) -> str:
        return re.sub(r"\s+", " ", text.strip())

    def normalize_answer(self, raw: str, session_id: str) -> str:
        """Filter internal terms, dedup against last assistant message, persist to history.

        L4: empty answer is NOT persisted to history (avoids polluting future context).
        """
        filtered = raw
        for word in self._settings.output_filter_words:
            filtered = filtered.replace(word, "")

        if not filtered.strip():
            return ""

        normalized = self.normalize_text(filtered)
        last = self._session_store.load_last_assistant(session_id)
        if last is not None and self.normalize_text(last) == normalized:
            logger.info("Dedup hit session_id=%s — skipping history write", session_id)
            return filtered

        self._session_store.append(
            session_id=session_id, role="assistant", content=filtered,
        )
        return filtered