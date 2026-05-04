from functools import lru_cache

from app.orchestrator import ChatOrchestrator


@lru_cache
def get_orchestrator() -> ChatOrchestrator:
    return ChatOrchestrator()
