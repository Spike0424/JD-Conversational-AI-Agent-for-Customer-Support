from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import get_orchestrator
from app.api.routes_admin import router as admin_router
from app.api.routes_chat import router as chat_router
from util.logger_setup import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时预热 orchestrator + agent，避免首次请求承担冷启动开销。"""
    orchestrator = get_orchestrator()
    orchestrator._agent.prewarm()
    yield


app = FastAPI(title="LangChain ReAct QA Agent", version="0.1.0", lifespan=lifespan)
app.include_router(chat_router)
app.include_router(admin_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}