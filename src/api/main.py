"""FastAPI application entrypoint."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pmo_agent import memory
from src.pmo_agent.llm_client import generate_response

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Task Management PMO Agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB
ALLOWED_EXTENSIONS = {"txt", "md", "csv", "docx"}
MAX_NOTE_LENGTH = 10_000


def _json_error(status_code: int, message: str) -> JSONResponse:
    """Return a standardized JSON error response."""

    logger.warning("Returning error %s: %s", status_code, message)
    return JSONResponse(status_code=status_code, content={"error": message})


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    logger.warning("HTTPException at %s: %s", request.url.path, message)
    return JSONResponse(status_code=exc.status_code, content={"error": message})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("Validation error at %s: %s", request.url.path, exc.errors())
    return JSONResponse(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, content={"error": "Invalid request payload."})


class LLMRequest(BaseModel):
    """Schema for LLM prompt requests."""

    prompt: str


class LLMResponse(BaseModel):
    """Schema for LLM responses."""

    response: str


class MemoryAddRequest(BaseModel):
    """Schema for adding free-form notes to memory."""

    label: str
    content: str


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@app.post("/llm", response_model=LLMResponse)
def call_llm(payload: LLMRequest) -> LLMResponse:
    """Invoke the Gemini LLM with the provided prompt."""

    logger.info("/llm endpoint called")

    try:
        text = generate_response(payload.prompt)
    except EnvironmentError as exc:
        logger.error("LLM call failed due to missing configuration", exc_info=exc)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - external API errors
        logger.error("LLM call failed", exc_info=exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return LLMResponse(response=text)


@app.get("/chat")
def get_chat_page() -> FileResponse:
    """Serve the minimal chat UI for interacting with the LLM endpoint."""

    return FileResponse(STATIC_DIR / "chat.html")


@app.get("/upload-ui")
def get_upload_page() -> FileResponse:
    """Serve the upload and memory management UI."""

    return FileResponse(STATIC_DIR / "upload.html")


def _get_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


async def _read_file(file: UploadFile) -> bytes:
    data = await file.read()
    logger.debug("Read %s bytes from uploaded file %s", len(data), file.filename)
    return data


def _handle_text_file(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - depends on user input
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to decode file as UTF-8.") from exc


def _handle_csv_file(data: bytes) -> tuple[str, dict[str, Any]]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:  # pragma: no cover - depends on user input
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unable to decode CSV as UTF-8.") from exc

    preview_df = None
    try:
        df = pd.read_csv(io.StringIO(text))
        preview_df = df.head(50)
    except Exception as exc:  # pragma: no cover - depends on pandas parsing
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to parse CSV file.") from exc

    csv_preview = {
        "columns": preview_df.columns.tolist(),
        "rows": preview_df.astype(str).values.tolist(),
    }
    summary_csv = preview_df.to_csv(index=False)
    return summary_csv, csv_preview


def _handle_docx_file(data: bytes) -> str:
    try:
        document = Document(io.BytesIO(data))
    except Exception as exc:  # pragma: no cover - depends on python-docx internals
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to read DOCX file.") from exc

    paragraphs = [para.text.strip() for para in document.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a single file upload, extract text, and store it in memory."""

    if not file.filename:
        return _json_error(status.HTTP_400_BAD_REQUEST, "Filename is required.")

    extension = _get_extension(file.filename)
    if extension not in ALLOWED_EXTENSIONS:
        return _json_error(status.HTTP_400_BAD_REQUEST, "Unsupported file type. Allowed: txt, md, csv, docx.")

    data = await _read_file(file)

    if len(data) > MAX_FILE_SIZE:
        return _json_error(status.HTTP_400_BAD_REQUEST, "File too large. Limit is 1 MB.")
    if not data:
        return _json_error(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")

    logger.info("Processing uploaded file %s", file.filename)

    text_content: str
    csv_preview: dict[str, Any] | None = None

    if extension in {"txt", "md"}:
        text_content = _handle_text_file(data)
    elif extension == "csv":
        text_content, csv_preview = _handle_csv_file(data)
    else:  # docx
        text_content = _handle_docx_file(data)

    label = f"doc:{Path(file.filename).name}"

    try:
        memory.add_text(label=label, content=text_content)
    except ValueError as exc:
        return _json_error(status.HTTP_400_BAD_REQUEST, str(exc))

    response_payload: dict[str, Any] = {
        "stored_label": label,
        "chars_stored": len(text_content),
    }
    if csv_preview is not None:
        response_payload["csv_preview"] = csv_preview

    logger.info("Stored uploaded file %s in memory", file.filename)
    return JSONResponse(status_code=status.HTTP_200_OK, content=response_payload)


@app.get("/memory")
def list_memory() -> dict[str, Any]:
    """Return the current memory items."""

    items = memory.list_items()
    logger.debug("Listing %s memory items", len(items))
    return {"items": items}


@app.post("/memory/reset")
def reset_memory() -> dict[str, bool]:
    """Reset the in-memory store."""

    memory.reset()
    logger.info("Memory reset via API request")
    return {"ok": True}


@app.post("/memory/add_text")
def add_memory_text(payload: MemoryAddRequest) -> JSONResponse:
    """Add a free-form note to memory."""

    if len(payload.content or "") > MAX_NOTE_LENGTH:
        return _json_error(status.HTTP_400_BAD_REQUEST, "Content exceeds maximum length of 10k characters.")

    try:
        memory.add_text(label=payload.label, content=payload.content)
    except ValueError as exc:
        return _json_error(status.HTTP_400_BAD_REQUEST, str(exc))

    logger.info("Added manual memory note with label %s", payload.label)
    return JSONResponse(status_code=status.HTTP_200_OK, content={"ok": True})
