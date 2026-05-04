from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

from app.config import Settings

TEXT_FILE_EXTENSIONS = {".md", ".txt", ".py", ".json", ".yaml", ".yml", ".toml"}
IGNORED_DIR_NAMES = {".git", ".venv", "__pycache__", ".jd_browser_profile", ".cursor"}


class RAGIndex:
    def __init__(self, settings: Settings) -> None:
        self._top_k = settings.rag_top_k
        self._enabled = settings.rag_enabled
        self._vector_store: InMemoryVectorStore | None = None

        if not self._enabled:
            return

        documents = self._load_documents(Path(settings.docs_root).resolve(), settings.rag_chunk_size)
        if not documents:
            return

        embeddings = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model="text-embedding-3-small",
        )
        self._vector_store = InMemoryVectorStore(embeddings)
        self._vector_store.add_documents(documents)

    def _load_documents(self, root: Path, chunk_size: int) -> list[Document]:
        docs: list[Document] = []
        for path in root.rglob("*"):
            if any(part in IGNORED_DIR_NAMES for part in path.parts):
                continue
            if not path.is_file() or path.suffix.lower() not in TEXT_FILE_EXTENSIONS:
                continue

            try:
                content = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue

            text = content.strip()
            if not text:
                continue

            rel = str(path.relative_to(root))
            for i in range(0, len(text), chunk_size):
                chunk = text[i : i + chunk_size].strip()
                if chunk:
                    docs.append(Document(page_content=chunk, metadata={"source": rel}))
        return docs

    def query(self, user_query: str) -> str:
        if not self._enabled:
            return "RAG is disabled."
        if not self._vector_store:
            return "RAG index is empty."

        matches = self._vector_store.similarity_search_with_score(user_query, k=self._top_k)
        if not matches:
            return "No relevant context found."

        lines = []
        for doc, score in matches:
            source = str(doc.metadata.get("source", "unknown"))
            snippet = " ".join(doc.page_content.splitlines()[:3]).strip()
            lines.append(f"[{source}] score={score:.4f} | {snippet}")
        return "\n".join(lines)
