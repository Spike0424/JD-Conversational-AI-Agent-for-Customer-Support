# LangChain ReAct QA Agent

This project provides a FastAPI-based intelligent QA agent using LangChain ReAct (Reasoning + Acting) with an OpenAI-compatible API.
It now supports streaming output (SSE), RAG vector retrieval, and persistent sessions via Redis or SQLite database.

## 1) Install dependencies

```bash
uv sync
```

## 2) Configure environment

```bash
cp .env.example .env
```

Then update `.env` with your model provider settings.

## 3) Start service

FastAPI API service:

```bash
uv run uvicorn main:app --reload
```

Chainlit web chat:

```bash
uv run chainlit run chainlit_app.py -w --headless
```

## 4) Test endpoints

Health check:

```bash
curl -s http://127.0.0.1:8000/health
```

Chat:

```bash
curl -s -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session",
    "question": "请帮我计算 (12 + 8) * 3"
  }'
```

Streaming chat (SSE):

```bash
curl -N -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session",
    "question": "总结一下项目里关于 LangChain 的内容"
  }'
```

Add document to RAG vector index:

```bash
curl -s -X POST http://127.0.0.1:8000/rag/documents \
  -H "Content-Type: application/json" \
  -d '{
    "source": "faq.md",
    "content": "这里放需要进入向量库的文档内容"
  }'
```

In Chainlit, paste documents with:

```text
/ingest faq.md
这里放需要进入向量库的文档内容
```

Normal Chainlit chat does not build embeddings or touch RAG. Only `/ingest` embeds and stores the provided document text.

Handoff to human agent:

```bash
curl -s -X POST http://127.0.0.1:8000/v1/handoff \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "demo-session",
    "reason": "用户要求人工处理退款争议",
    "priority": "high"
  }'
```

## Built-in tools

- `calculator`: evaluates safe arithmetic expressions
- `search_docs`: searches local text files under `DOCS_ROOT`
- `search_knowledge_base`: semantic retrieval over local vector index
- `search_products` / `get_order_status` / `check_warranty` / `create_after_sale_ticket`: mock business tools for 3C customer service

### Tool modes

- `ENABLE_BUSINESS_TOOLS=true` enables customer-service business tools (recommended for production).
- `ENABLE_DEV_TOOLS=false` keeps file/bash developer tools disabled by default.
- Turn on `ENABLE_DEV_TOOLS=true` only in trusted debugging environments.
- Business tools return structured JSON with `status/code/data/message`.
- Configure real backend adapters via `OMS_BASE_URL`, `CRM_BASE_URL`, `AFTERSALE_BASE_URL`.
- `BUSINESS_API_TIMEOUT_SECONDS` and `BUSINESS_API_RETRIES` control timeout/retry behavior.

## API error format

- Non-streaming error responses use:
  - `error_code`
  - `message`
  - `trace_id`
- Streaming (`/chat/stream`) sends `meta` event first with:
  - `trace_id`, `intent`, `citations`, `actions`
  - each citation item includes `source`, `snippet`, `score`

## Citation traceability

- `/v1/chat` now returns structured citations with `source/snippet/score`.
- Normal chat does not build embeddings or touch RAG.
- Use `/rag/documents` or `/v1/rag/documents` to embed user-provided documents into the vector index.
- The final `answer` appends an `依据来源` section only when the user explicitly requests RAG/knowledge-base retrieval and citations exist.

## Notes

- ReAct agent performs reasoning and tool use internally, and only returns the final answer.
- Session persistence uses Redis first (when `REDIS_URL` is valid), otherwise falls back to SQLite (`DATABASE_URL`).
- RAG index is built from files under `DOCS_ROOT` using OpenAI embeddings.

### DeepSeek thinking mode (`reasoning_content` 400)

If you use **DeepSeek** models with **thinking mode** enabled, the API requires every **tool-calling** follow-up request to include the assistant message’s `reasoning_content`. LangChain’s agent loop often does not replay that field, which surfaces as:

`The reasoning_content in the thinking mode must be passed back to the API.`

This project sends `extra_body={"thinking": {"type": "disabled"}}` when `CHAT_THINKING_MODE` is `disabled`, or when it is `auto` and `OPENAI_BASE_URL` points at `deepseek.com`. Set `CHAT_THINKING_MODE=enabled` only if you need visible chain-of-thought and accept that tool-heavy flows may still need provider-specific message handling.

![mock数据](./png/image.png)