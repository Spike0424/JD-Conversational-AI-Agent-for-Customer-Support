# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
uv sync                                    # install deps
uv run uvicorn main:app --reload           # start FastAPI on :8000
uv run chainlit run chainlit_app.py -w     # start Chainlit chat UI
uv run .venv/bin/pytest -q                 # full test suite
uv run .venv/bin/pytest tests/test_X.py -q # single test file
uv run .venv/bin/pytest -k name -q         # tests matching name
PGPASSWORD=pg2024 psql -h 127.0.0.1 -p 5434 -U postgres -d jd_agent -c "..."  # DB inspect
uv run python script/seed_*.py             # seed data (orders, knowledge, mock)
```

No separate lint/format step (project doesn't configure one).

## Architecture

### Request flow

```
HTTP (FastAPI)
  → routes_chat.py / routes_admin.py / routes_qa.py
  → ChatOrchestrator (app/orchestrator/service.py)
    ├─ scene_classifier.py      # presale / insale / aftersale / mixed via order data
    ├─ turn_context.py          # parse raw turn → ProductCard / OrderCard / MediaInfo
    ├─ agent_runtime.py         # ReAct graph (LangChain create_agent)
    │    └─ tools/search_knowledge.py   # hybrid retrieval (alias + keyword + vector)
    └─ session_store.py         # agent_messages table (SQLModel)
```

The orchestrator (`ChatOrchestrator.chat`) branches on `Context.type` before scene classification:
- `is_silent` (withdraw / auth / system_biz / system_status / mall_system_msg) → empty answer
- `requires_human` (image / video) → `{actions: ["transfer_to_human"], need_handoff: True}`
- `goods_card` + no customer text → "您想了解这款商品的哪方面呢？"
- otherwise → scene classifier + agent

### Embedded shop page model

The agent is designed to be embedded in product shop pages. Frontend sends a `Context` payload (`app/context_models.py`) alongside each turn with: `shop_id`, `shop_name`, `goods_id`, `goods_name`, `order_sn`, `media_url`, `context_type`, `channel_type`. The orchestrator parses raw text via `parse_turn_context()` to extract the clean customer message, then injects a `【当前会话信息】` block into the LLM system prompt (`agent_runtime._format_session_info`).

All platform-provided values are run through `_safe_value()` which wraps them in `<input>...</input>` tags and doubles braces — this is prompt-injection defense, not decoration. Don't bypass it.

### Hybrid retrieval

`SearchKnowledge` (in `app/tools/search_knowledge.py`) ranks candidates by combining:
1. **Alias match** (`_alias_match_score`, highest weight ~240) — splits aliases on `/|;；\n\r` and matches against normalized query
2. **Keyword score** (`_keyword_match_score`) — BM25-like scoring across `aliases + answer + tags + product_family + section_title`
3. **Vector similarity** (BAAI/bge-small-zh-v15 via fastembed) — `>=0.45` threshold promotes to "hybrid" match_type
4. **Goods ID bonus** — `+50` if entry's `goods_id == current`, `-30` if mismatch

Set context via `set_context(shop_id, scene, goods_id)` before each call. Result is also written to `RAGLogger` (see `app/evaluation/rag_logger.py`).

### Scene classification

`SceneClassifier` (in `app/orchestrator/scene_classifier.py`) queries the `orders` table (via psycopg2 connection pool) for the user's recent orders, then derives scene:
- no orders → `presale`
- all orders signed (`arr_time IS NOT NULL`) → `aftersale`
- mix of signed + unsigned → `scene_hint=mixed_orders`, scene=`insale`, but the orchestrator then re-routes to `mixed` prompt that asks for order number
- 30-minute in-memory cache keyed by `session_id`

`SceneClassifier.last_scene_hint` is the side-channel for orchestrator to detect mixed.

### Empty-answer fallback (4-layer)

In `ReActQAAgent.ask()` / `ask_stream()`:
1. **L1 silent retry** — re-call with exponential backoff if answer is empty
2. **L2 knowledge-base fallback** — call `SearchKnowledge.search()` directly, prefix with "以下是根据知识库检索到的相关信息"
3. **L3 default reply** — "抱歉，我暂时无法回答，请换个问题或稍后再试"
4. **L4 history purification** — empty answer is NOT persisted to `agent_messages` to avoid polluting future context

`_judge()` wraps `judge_llm.invoke()` in try/except; one transient API failure does not abort the batch.

### Database

SQLModel ORM (`app/business_models.py`) on PostgreSQL with pgvector extension. Key tables:
- `shops`, `accounts`, `channels`, `keywords` — basic entities
- `presale_knowledge` / `insale_knowledge` / `aftersale_knowledge` — three scene-specific knowledge tables (each with `SceneKnowledgeMixin` columns: `shop_id`, `goods_id`, `sub_intent`, `aliases`, `answer`, `tags`, `section_title`, `product_family`, `priority`, `enabled`)
- `scene_knowledge_embeddings` — pgvector HNSW index for vector search
- `aftersale_chunks` — chunked + vectorized aftersale docs
- `agent_messages` — chat history (session_id, role, content, timestamp)
- `orders` — merged orders+delivery+skus (use `extend_existing=True` flag)

`app/db.py::create_session()` returns SQLModel session; `init_db()` registers pgvector extension via `pgvector.psycopg2.register.register_vector` (NOT `pgvector.sqlalchemy.psycopg2`, which doesn't exist).

### Admin API

`app/api/routes_admin.py` exposes 5 endpoints under `/v1/admin/*` (mounted at this prefix in the router). All require `Authorization: Bearer <ADMIN_API_TOKEN>` when `ADMIN_API_TOKEN` env var is set; auth is disabled when unset (dev convenience). Use `hmac.compare_digest` for the check.

Endpoints: `turn-context/dry-run` (POST), `context/validate` (POST), `sessions/{sid}` (GET), `scene-cache/clear` (POST), `context-types` (GET).

## Conventions & gotchas

- **LLM is OpenAI-compatible** — default is DeepSeek (`deepseek-v4-flash` via `api.deepseek.com`). For ReAct/tool agents, `CHAT_THINKING_MODE=disabled` is required (DeepSeek's reasoning_content otherwise 400s).
- **Embedding model**: `EMBEDDING_PROVIDER=local` → FastEmbed downloads `BAAI/bge-small-zh-v1.5` (~400MB, cached in `~/.cache/fastembed`). Set `FASTEMBED_CACHE_DIR=/home/spike/.cache/fastembed` to override.
- **Scene prompts**: `prompt/{presales,sales,aftersales,mixed}.md` — these are LLM system prompts, not user prompts. Loaded by `_load_scene_prompt` based on scene name.
- **Tool registration**: `util/agent_tool.py::agent_tool` decorator registers tools at import time. `search_knowledge` is the only registered tool; it's a singleton (`get_search_knowledge()`).
- **`extend_existing=True`** is used on several tables (`orders`, `rag_chunks`) to suppress "Multiple classes found" errors from SQLModel metadata reflection. Don't remove this.
- **`output_filter_words`** in Settings — words stripped from assistant output before persistence (configured in `.env`). L4 history purification check skips persistence when filter empties the answer.
- **`ChatRequest.question` is now optional** (default `""`) — embedded flows can send empty question + non-text `context_type` (image, video, withdraw, etc.).
- **Test fixture pattern**: `tests/conftest.py` provides `api_client` and `orch_stub` fixtures; `orch_stub.chat` / `.stream_chat` are `AsyncMock`-style (the orchestrator methods are `async def`). Use `AsyncMock` not `MagicMock` when stubbing them in tests.
- **History compression**: Agens LLM (separate `ChatOpenAI` instance, optional via env) summarizes old messages when token count exceeds `history_window_tokens × history_compress_threshold`. Skipped if Agens not configured.

## File map (entry points for common tasks)

| Want to... | Edit |
|------------|------|
| Add a new contextType | `app/context_models.py` (enum) → service.py branch |
| Add a new knowledge column | `app/business_models.py` (SceneKnowledgeMixin) + migration |
| Change scene routing | `app/orchestrator/scene_classifier.py` |
| Change system prompt for a scene | `prompt/{scene}.md` |
| Add a tool for the agent | `app/tools/*.py` with `@agent_tool` decorator |
| Add an admin endpoint | `app/api/routes_admin.py` (auto-protected by bearer token) |
| Tune 4-layer fallback | `app/llm/agent_runtime.py` `ask()` / `ask_stream()` |
| Add seed data | `script/seed_*.py` (uses `init_db()` + `create_session()`) |
| Add a golden test case | `tests/golden_*.py` (DATASET list) + `tests/test_*.py` |