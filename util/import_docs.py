#!/usr/bin/env python3
"""Import PDF / Markdown docs into scene knowledge tables (+ aftersale chunks).

Pipeline: parse (pypdf, OCR fallback for scanned PDFs) → chunk (RecursiveCharacterTextSplitter)
→ LLM generates natural-language alias questions per chunk → insert into
presale/insale/aftersale_knowledge. For the aftersale scene, chunks are also
embedded (fastembed, 384-dim) into aftersale_chunks for pgvector layer-2 retrieval.

Usage:
    uv run python util/import_docs.py --scene aftersale --shop-id 7
    uv run python util/import_docs.py --scene presale --shop-id 7 --doc-dir doc --no-llm
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from sqlalchemy import select

from api.core.config import get_settings
from api.core.embedding import get_embeddings
from api.models.business import (
    AftersaleChunk,
    AftersaleKnowledge,
    InsaleKnowledge,
    PresaleKnowledge,
    Shop,
)
from api.models.db import create_session, init_db

# Below this many chars per page a PDF is treated as scanned and OCR kicks in.
_MIN_CHARS_PER_PAGE = 30
_MIN_CHUNK_LEN = 20

_SCENE_MODELS = {
    "presale": PresaleKnowledge,
    "insale": InsaleKnowledge,
    "aftersale": AftersaleKnowledge,
}

ALIAS_PROMPT = """你是京东客服知识库管理员。以下是一段规则文本片段，请生成 3-5 个买家可能提问的自然语言问题（每个不超过15个字）。

规则片段：
{chunk}

只返回 JSON 数组，例如：["收到货15天坏了怎么办", "性能故障怎么退", "保修期内维修要钱吗"]
只返回 JSON 数组，不要其他内容。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import PDF/MD docs into knowledge DB")
    parser.add_argument("--scene", choices=sorted(_SCENE_MODELS), default="aftersale")
    parser.add_argument("--shop-id", type=int, default=7, help="shops.id (primary key)")
    parser.add_argument("--goods-id", type=int, default=None, help="Bind to a product; default shop-general")
    parser.add_argument("--doc-dir", type=str, default="doc")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM alias generation (offline import)")
    return parser.parse_args()


def _llm() -> ChatOpenAI:
    s = get_settings()
    return ChatOpenAI(
        api_key=s.openai_api_key, base_url=s.openai_base_url,
        model=s.model_name, temperature=0.3,
    )


def _parse_json_array(text: str) -> list[str]:
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text)
    m = re.search(r"\[.*\]", text, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    return [str(a).strip() for a in data if str(a).strip()] if isinstance(data, list) else []


def extract_pdf_text(pdf_path: Path) -> tuple[str, bool]:
    """Return (text, used_ocr). Falls back to OCR when pages have too little text."""
    reader = PdfReader(str(pdf_path))
    pages = [(p.extract_text() or "") for p in reader.pages]
    if pages and sum(len(t.strip()) for t in pages) / len(pages) < _MIN_CHARS_PER_PAGE:
        print(f"    little text ({len(pages)} pages) → OCR fallback")
        return _ocr_pdf(pdf_path), True
    return "\n\n".join(t for t in pages if t.strip()), False


def _ocr_pdf(pdf_path: Path) -> str:
    from pdf2image import convert_from_path
    import pytesseract

    texts: list[str] = []
    images = convert_from_path(str(pdf_path), dpi=300)
    for i, img in enumerate(images, 1):
        page_text = pytesseract.image_to_string(img, lang="chi_sim+eng")
        if page_text.strip():
            texts.append(page_text.strip())
        print(f"    page {i}/{len(images)}: {len(page_text)} chars")
    return "\n\n".join(texts)


def chunk_text(text: str) -> list[str]:
    s = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(1, s.rag_chunk_size),
        chunk_overlap=max(0, s.rag_chunk_overlap),
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
        keep_separator=True,
    )
    return [c.strip() for c in splitter.split_text(text) if len(c.strip()) >= _MIN_CHUNK_LEN]


def generate_aliases(llm: ChatOpenAI | None, chunk: str, fallback: str) -> str:
    """LLM question aliases, ';'-joined. Falls back to the doc/section name."""
    if llm is not None:
        try:
            resp = llm.invoke(ALIAS_PROMPT.format(chunk=chunk[:1500]))
            aliases = _parse_json_array(str(resp.content))
            if aliases:
                return ";".join(aliases)
        except Exception as exc:
            print(f"    alias LLM failed ({exc}), using fallback")
    return fallback


def main() -> None:
    args = parse_args()
    model = _SCENE_MODELS[args.scene]
    doc_dir = Path(args.doc_dir)
    if not doc_dir.is_dir():
        sys.exit(f"doc dir not found: {doc_dir.resolve()}")

    init_db()

    session = create_session()
    try:
        shop = session.get(Shop, args.shop_id)
        if shop is None:
            sys.exit(
                f"Shop id={args.shop_id} not found in shops table. "
                "Create it first, e.g. INSERT INTO shops (channel_id, shop_id, shop_name) VALUES (1, 'JD-DEMO', '演示店铺');"
            )
    finally:
        session.close()

    llm = None if args.no_llm else _llm()
    files = sorted(p for p in doc_dir.iterdir() if p.suffix.lower() in {".pdf", ".md"})
    print(f"{len(files)} files in {doc_dir} → scene={args.scene} shop={args.shop_id}\n")

    stats = {"rows": 0, "chunks": 0, "skipped": 0}
    for path in files:
        source_type = path.suffix.lower().lstrip(".")
        section_title = path.stem[:255]
        print(f"[{source_type.upper()}] {path.name}")

        try:
            if source_type == "pdf":
                text, _ = extract_pdf_text(path)
            else:
                text = path.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"  ✗ extract failed: {exc}")
            continue
        if not text.strip():
            print("  ✗ no text extracted")
            continue

        chunks = chunk_text(text)
        print(f"  {len(chunks)} chunks")

        # Existing answers for this shop+source, for skip-on-rerun dedup.
        session = create_session()
        try:
            existing = {
                answer
                for (answer,) in session.exec(
                    select(model.answer).where(
                        model.shop_id == args.shop_id,
                        model.source_type == source_type,
                    )
                ).all()
            }

            added = 0
            for chunk in chunks:
                if chunk in existing:
                    stats["skipped"] += 1
                    continue
                aliases = generate_aliases(llm, chunk, fallback=section_title)
                row = model(
                    shop_id=args.shop_id,
                    goods_id=args.goods_id,
                    aliases=aliases,
                    answer=chunk,
                    section_title=section_title,
                    source_type=source_type,
                    enabled=True,
                )
                session.add(row)
                session.flush()  # assign row.id for chunk FK
                if args.scene == "aftersale":
                    vec = get_embeddings().embed_query(chunk)
                    session.add(AftersaleChunk(
                        knowledge_id=row.id,
                        chunk_content=chunk,
                        embedding=vec,
                        aliases={"questions": aliases.split(";")},
                    ))
                    stats["chunks"] += 1
                existing.add(chunk)
                added += 1
            session.commit()
            print(f"  ✓ rows_added={added}")
            stats["rows"] += added
        except Exception as exc:
            session.rollback()
            print(f"  ✗ import failed: {exc}")
        finally:
            session.close()

    print(f"\nDone: {stats['rows']} knowledge rows, "
          f"{stats['chunks']} embedded chunks, {stats['skipped']} duplicates skipped")


if __name__ == "__main__":
    main()
