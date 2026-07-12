from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_base_url: str = Field(..., alias="OPENAI_BASE_URL")
    # Chat uses OPENAI_*; RAG embeddings need a provider that implements /v1/embeddings (DeepSeek does not).
    embedding_openai_api_key: str = Field(default="", alias="EMBEDDING_OPENAI_API_KEY")
    embedding_openai_base_url: str = Field(default="", alias="EMBEDDING_OPENAI_BASE_URL")
    embedding_model_name: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL_NAME")
    embedding_provider: str = Field(default="local", alias="EMBEDDING_PROVIDER")
    local_embedding_model_name: str = Field(
        default="BAAI/bge-small-zh-v1.5",
        alias="LOCAL_EMBEDDING_MODEL_NAME",
    )
    # 单次向量化文档条数。聚合网关（如中转站）大批量易 500，可调小甚至为 1。
    embedding_batch_size: int = Field(default=16, alias="EMBEDDING_BATCH_SIZE")
    embedding_max_retries: int = Field(default=1, alias="EMBEDDING_MAX_RETRIES")
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    temperature: float = Field(default=0.2, alias="TEMPERATURE")
    max_history_turns: int = Field(default=6, alias="MAX_HISTORY_TURNS")
    docs_root: str = Field(default=".", alias="DOCS_ROOT")
    redis_url: str = Field(default="", alias="REDIS_URL")
    database_url: str = Field(
        default="postgresql://postgres:pg2024@127.0.0.1:5434/jd_agent",
        alias="DATABASE_URL",
    )
    rag_enabled: bool = Field(default=True, alias="RAG_ENABLED")
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")
    rag_chunk_size: int = Field(default=800, alias="RAG_CHUNK_SIZE")
    rag_chunk_overlap: int = Field(default=80, alias="RAG_CHUNK_OVERLAP")
    rag_index_dir: str = Field(default=".rag/faiss", alias="RAG_INDEX_DIR")
    # FAISS returns distance score (smaller means more similar). Negative disables threshold filtering.
    rag_score_threshold: float = Field(default=-1.0, alias="RAG_SCORE_THRESHOLD")
    rag_deduplicate_results: bool = Field(default=True, alias="RAG_DEDUPLICATE_RESULTS")
    enable_dev_tools: bool = Field(default=False, alias="ENABLE_DEV_TOOLS")
    enable_business_tools: bool = Field(default=True, alias="ENABLE_BUSINESS_TOOLS")
    business_api_timeout_seconds: int = Field(default=8, alias="BUSINESS_API_TIMEOUT_SECONDS")
    business_api_retries: int = Field(default=2, alias="BUSINESS_API_RETRIES")
    business_api_key: str = Field(default="", alias="BUSINESS_API_KEY")
    oms_base_url: str = Field(default="", alias="OMS_BASE_URL")
    crm_base_url: str = Field(default="", alias="CRM_BASE_URL")
    aftersale_base_url: str = Field(default="", alias="AFTERSALE_BASE_URL")
    # DeepSeek thinking mode: tool-calling agents must pass back `reasoning_content` on every
    # follow-up request; LangChain often does not. Use "disabled" for ReAct (see README).
    # Values: auto | enabled | disabled
    chat_thinking_mode: str = Field(default="auto", alias="CHAT_THINKING_MODE")
    chat_timeout_seconds: int = Field(default=60, alias="CHAT_TIMEOUT_SECONDS")
    chat_retries: int = Field(default=1, alias="CHAT_RETRIES")
    agent_recursion_limit: int = Field(default=6, alias="AGENT_RECURSION_LIMIT")
    # ── 混合检索搜索词配置 ────────────────────────────────────────────
    search_phrase_candidates: list[str] = Field(
        default_factory=lambda: [
            "商品参数", "参数", "规格", "型号", "款式", "尺寸", "尺码",
            "重量", "容量", "功率", "电压", "材质", "面料", "成分",
            "功能", "使用方法", "安装", "配件", "赠品", "颜色",
            "有货", "现货", "库存", "什么快递", "发货地", "质保",
            "保修", "退货包运费", "七天无理由", "7天无理由",
        ],
    )
    search_synonym_expansions: dict[str, list[str]] = Field(
        default_factory=dict,
    )
    # ── 输出规范化配置 ──────────────────────────────────────────────
    output_filter_words: list[str] = Field(
        default_factory=lambda: [
            "rag", "RAG", "检索召回", "向量检索", "知识库检索",
        ],
    )
    # ── Agens 摘要 LLM（独立配置）───────────────────────────────────
    agens_api_key: str = Field(default="", alias="AGENS_API_KEY")
    agens_base_url: str = Field(default="", alias="AGENS_BASE_URL")
    agens_model_name: str = Field(default="gpt-4o-mini", alias="AGENS_MODEL_NAME")
    agens_temperature: float = Field(default=0.0, alias="AGENS_TEMPERATURE")
    # ── 历史压缩配置 ──────────────────────────────────────────────
    history_window_tokens: int = Field(default=128_000, alias="HISTORY_WINDOW_TOKENS")
    history_compress_threshold: float = Field(default=0.7, alias="HISTORY_COMPRESS_THRESHOLD")
    history_keep_recent_turns: int = Field(default=4, alias="HISTORY_KEEP_RECENT_TURNS")
    # ── 场景提示词目录 ────────────────────────────────────────────
    prompt_dir: str = Field(default="./prompt", alias="PROMPT_DIR")
    log_rotation_mb: int = Field(default=10, alias="LOG_ROTATION_MB")
    log_retention_days: int = Field(default=7, alias="LOG_RETENTION_DAYS")


@lru_cache
def get_settings() -> Settings:
    return Settings()


# ── Sub-scene keyword rules ────────────────────────────────────────

SUB_SCENE_RULES: dict[str, tuple[str, ...]] = {
    "product_attribute": (
        "参数", "规格", "型号", "尺寸", "尺码", "重量", "容量", "功率", "材质", "面料", "颜色", "库存",
    ),
    "product_usage": (
        "怎么用", "使用教程", "使用方法", "说明书", "安装",
    ),
    "shipping": (
        "快递", "发货", "物流", "到货", "发货地", "从哪发", "从哪里发",
    ),
    "aftersale": (
        "质保", "保修", "退货", "退款", "运费", "运费险", "质量问题", "坏了",
    ),
}
