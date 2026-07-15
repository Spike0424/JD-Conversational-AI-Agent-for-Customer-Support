from fastapi import FastAPI

from app.api.routes_admin import router as admin_router
from app.api.routes_chat import router as chat_router
from app.api.routes_qa import router as qa_router
from util.logger_setup import setup_logging

setup_logging()

app = FastAPI(title="LangChain ReAct QA Agent", version="0.1.0")
app.include_router(chat_router)
app.include_router(qa_router)
app.include_router(admin_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
