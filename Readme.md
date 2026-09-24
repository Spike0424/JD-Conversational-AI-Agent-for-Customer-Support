# 京东第三方商家 AI 客服 Agent

> 凌晨一点，一位买家盯着刚下单的手机页面问："屏幕碎了，保内修要钱吗？"
> 三秒后他收到了带具体政策的回复——回复他的不是熬夜的客服，而是这个仓库里的 Agent：
> 它认得买家正在看哪个商品、下过哪张订单，记得你店里的每一条售后政策，
> 并且只用你**喂给它的文档**说话。

## 这是什么

面向京东第三方商家的智能客服系统：**FastAPI 后端 + React 前端**。
Agent 通过页面上下文（shop_id / goods_id / order_sn）感知买家"在哪、买了什么"，
用 **混合检索（别名精确匹配 + 关键词 + 语义向量）+ LLM 单轮工具调用** 回答售前 / 售中 / 售后问题。

## 核心亮点：把文档"喂"给客服，一条命令

把店铺的售后政策 **PDF**、运营笔记 **Markdown** 丢进 `doc/` 目录，剩下全自动：

```
doc/*.pdf, *.md
   │  解析（扫描件自动 OCR）→ 切块 → LLM 生成"买家可能怎么问"
   ▼
写入 PostgreSQL
   ├── {scene}_knowledge 表  → 别名/关键词精确匹配（一层检索）
   └── aftersale_chunks 表   → 512 维 pgvector 向量 + HNSW 索引（二层语义检索）
```

导入完成，Agent 立刻能答"收到货 15 天坏了怎么办"——问法再刁钻也能被语义检索兜住。重复执行自动去重。

## 安装

需要下载的东西都在这里，逐段复制执行即可：

```bash
# Python 依赖
uv sync

# 数据库（Docker：PostgreSQL + pgvector，一条命令拉起）
# 已有 PostgreSQL 的话跳过这步，把连接串填到 .env 的 DATABASE_URL 即可
docker run -d --name jd-agent-pg -p 5434:5432 \
  -e POSTGRES_PASSWORD=pg2024 -e POSTGRES_DB=jd_agent \
  pgvector/pgvector:pg16

# 配置（填入你的 LLM API key；embedding 默认本地 fastembed，无需下载服务）
cp .env.example .env

# 前端依赖
cd web && npm install && cd ..
```

## 使用

```bash
# 1. 启动后端 :8000（首次启动自动建表）
uv run uvicorn api.main:app --reload

# 2. 注册店铺（只需一次）
PGPASSWORD=pg2024 psql -h 127.0.0.1 -p 5434 -U postgres -d jd_agent -c \
  "INSERT INTO shops (channel_id, shop_id, shop_name) VALUES (1, 'JD-DEMO', '演示店铺');"

# 3. 导入知识：PDF/MD 放进 doc/，选 scene（presale|insale|aftersale）
uv run python util/import_docs.py --scene aftersale --shop-id 7

# 4. 启动前端 :5173（代理 /v1 → :8000）
cd web && npm run dev
```

打开页面，填店铺 / 商品 / 订单号开始对话。`npm run build` 产物由 FastAPI 挂载到 `/`，生产环境同源零 CORS。

## 架构一览

```
HTTP (FastAPI)
  -> api/controllers/{chat,admin,shop}.py
  -> ChatOrchestrator (api/core/orchestrator.py)
    ├─ turn_context.py      解析商品卡/订单卡/媒体上下文
    ├─ scene_classifier.py  售前/售中/售后场景路由
    ├─ agent_runtime.py     LLM + bind_tools 单轮调用
    │    ├─ input_builder.py   场景 prompt + pre-RAG 占位符填充
    │    └─ knowledge.py       混合检索（alias + keyword + vector）
    └─ session_store.py     会话历史
```

- **流式回复**：`POST /v1/chat/stream` 用 SSE；非流式走 `POST /v1/chat`。
- **稳定性**：4 层空答兜底 + 3 层 API 错误兜底（4xx 快速失败 / 5xx 换模型重试 / 网络指数退避）。
- **安全**：prompt 注入防御全部代码级实现（`api/core/input_builder.py`）。

## 端点速查

| 类型 | 路由 | 说明 |
|------|------|------|
| Chat | `POST /v1/chat` / `/v1/chat/stream` | 流式用 SSE |
| Public | `GET /v1/shops` / `/v1/products` | 前端表单，IP 限流 |
| Admin | `/v1/admin/*` | bearer token 鉴权 |

## 评测

内置检索/生成双维评测：`tests/golden_aftersale.py` golden 数据集，`script/run_eval.py` 跑批，
指标含 hit_rate@k / MRR / precision@k / F1 与 LLM-as-a-judge 打分。

## 更多

开发约定、目录结构、改代码去哪——见 [CLAUDE.md](CLAUDE.md)。
