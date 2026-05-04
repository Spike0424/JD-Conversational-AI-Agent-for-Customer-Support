from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session identifier.")
    question: str = Field(..., min_length=1, description="User question.")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    trace_id: str | None = None
    intent: str = "general_qa"
    citations: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    need_handoff: bool = False


class HandoffRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session identifier.")
    reason: str = Field(..., min_length=1, description="Reason to transfer to human support.")
    priority: str = Field(default="normal", description="Ticket priority: low/normal/high.")


class HandoffResponse(BaseModel):
    session_id: str
    ticket_id: str
    queue: str
    status: str = "queued"


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    trace_id: str | None = None
