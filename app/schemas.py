from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Conversation session identifier.")
    question: str = Field(..., min_length=1, description="User question.")


class Citation(BaseModel):
    source: str
    snippet: str
    score: float
    file_name: str | None = None
    page_number: int | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    trace_id: str | None = None
    intent: str = "general_qa"
    citations: list[Citation] = Field(default_factory=list)
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


class DocumentIngestRequest(BaseModel):
    source: str = Field(..., min_length=1, description="Document source name, file name, or user-provided label.")
    content: str = Field(..., min_length=1, description="Document content to embed and add to the vector index.")


class DocumentIngestResponse(BaseModel):
    source: str
    chunks_added: int
    status: str = "indexed"


class PdfPathIngestRequest(BaseModel):
    pdf_path: str = Field(..., min_length=1, description="Absolute or relative PDF path on server.")
    source: str | None = Field(default=None, description="Optional source label shown in citations.")


class ErrorResponse(BaseModel):
    error_code: str
    message: str
    trace_id: str | None = None
