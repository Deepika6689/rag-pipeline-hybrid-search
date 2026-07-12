"""
Phase 1.1: Multi-format document loader.
Normalizes markdown / text / HTML / PDF into clean plaintext + metadata,
and keeps raw + processed versions on disk so re-indexing never requires
re-uploading source files.
"""
import os
import json
import hashlib
from dataclasses import dataclass, asdict
from typing import List

from bs4 import BeautifulSoup
from pypdf import PdfReader

from app import config


@dataclass
class LoadedDocument:
    doc_id: str
    source_file: str
    file_type: str
    title: str
    text: str
    sections: List[dict]  # [{"heading": str, "text": str, "page": int|None}]


def _doc_id_for(path: str) -> str:
    return hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]


def _load_markdown_or_text(path: str) -> LoadedDocument:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        raw = f.read()

    sections = []
    current_heading = "Document Start"
    current_lines = []

    for line in raw.splitlines():
        if line.strip().startswith("#"):
            if current_lines:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_lines).strip(),
                    "page": None,
                })
            current_heading = line.strip("# ").strip() or current_heading
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_lines).strip(),
            "page": None,
        })

    full_text = "\n\n".join(s["text"] for s in sections if s["text"])
    title = os.path.basename(path)

    return LoadedDocument(
        doc_id=_doc_id_for(path),
        source_file=path,
        file_type="markdown" if path.endswith(".md") else "text",
        title=title,
        text=full_text,
        sections=[s for s in sections if s["text"]],
    )


def _load_html(path: str) -> LoadedDocument:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    sections = []
    current_heading = "Document Start"
    current_chunks = []

    for el in soup.find_all(["h1", "h2", "h3", "p", "li"]):
        if el.name in ("h1", "h2", "h3"):
            if current_chunks:
                sections.append({
                    "heading": current_heading,
                    "text": "\n".join(current_chunks).strip(),
                    "page": None,
                })
            current_heading = el.get_text(strip=True) or current_heading
            current_chunks = []
        else:
            text = el.get_text(strip=True)
            if text:
                current_chunks.append(text)

    if current_chunks:
        sections.append({
            "heading": current_heading,
            "text": "\n".join(current_chunks).strip(),
            "page": None,
        })

    full_text = "\n\n".join(s["text"] for s in sections if s["text"])
    title = soup.title.get_text(strip=True) if soup.title else os.path.basename(path)

    return LoadedDocument(
        doc_id=_doc_id_for(path),
        source_file=path,
        file_type="html",
        title=title,
        text=full_text,
        sections=[s for s in sections if s["text"]],
    )


def _load_pdf(path: str) -> LoadedDocument:
    reader = PdfReader(path)
    sections = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if text:
            sections.append({
                "heading": f"Page {i + 1}",
                "text": text,
                "page": i + 1,
            })

    full_text = "\n\n".join(s["text"] for s in sections)
    title = os.path.basename(path)

    return LoadedDocument(
        doc_id=_doc_id_for(path),
        source_file=path,
        file_type="pdf",
        title=title,
        text=full_text,
        sections=sections,
    )


def load_document(path: str) -> LoadedDocument:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".md", ".markdown"):
        return _load_markdown_or_text(path)
    if ext in (".txt",):
        return _load_markdown_or_text(path)
    if ext in (".html", ".htm"):
        return _load_html(path)
    if ext == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"Unsupported file type: {ext}")


def load_directory(directory: str) -> List[LoadedDocument]:
    docs = []
    for fname in sorted(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        if not os.path.isfile(fpath):
            continue
        ext = os.path.splitext(fname)[1].lower()
        if ext not in (".md", ".markdown", ".txt", ".html", ".htm", ".pdf"):
            continue
        docs.append(load_document(fpath))
    return docs


def persist_processed(doc: LoadedDocument, out_dir: str = config.PROCESSED_DOCS_DIR):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{doc.doc_id}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(asdict(doc), f, indent=2)
    return out_path
