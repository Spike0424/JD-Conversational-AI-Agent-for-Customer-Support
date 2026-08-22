from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.controllers.admin import router as admin_router
from api.controllers.auth import router as auth_router
from api.controllers.chat import router as chat_router
from api.controllers.sessions import router as sessions_router
from api.controllers.deps import get_orchestrator
from api.controllers.rate_limit import limiter
from api.controllers.shop import router as shop_router
from api.models.db import apply_migrations, init_db
from util.logger_setup import setup_logging

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时建表 + 跑幂等迁移 + 预热 orchestrator + agent。"""
    init_db()
    apply_migrations()
    orchestrator = get_orchestrator()
    orchestrator._agent.prewarm()
    yield


app = FastAPI(title="LangChain ReAct QA Agent", version="0.1.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.include_router(chat_router)
app.include_router(admin_router)
app.include_router(shop_router)
app.include_router(auth_router)
app.include_router(sessions_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Serve frontend SPA (must be after all /v1/* routes to avoid shadowing them)
_frontend_dist = Path(__file__).parent.parent / "web" / "dist"
if _frontend_dist.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")