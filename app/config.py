from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = Field(..., alias="OPENAI_API_KEY")
    openai_base_url: str = Field(..., alias="OPENAI_BASE_URL")
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    temperature: float = Field(default=0.2, alias="TEMPERATURE")
    max_history_turns: int = Field(default=6, alias="MAX_HISTORY_TURNS")
    docs_root: str = Field(default=".", alias="DOCS_ROOT")
    redis_url: str = Field(default="", alias="REDIS_URL")
    database_url: str = Field(default="sqlite:///./agent_sessions.db", alias="DATABASE_URL")
    rag_enabled: bool = Field(default=True, alias="RAG_ENABLED")
    rag_top_k: int = Field(default=4, alias="RAG_TOP_K")
    rag_chunk_size: int = Field(default=800, alias="RAG_CHUNK_SIZE")
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


@lru_cache
def get_settings() -> Settings:
    return Settings()
