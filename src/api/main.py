"""FastAPI application entrypoint."""
from __future__ import annotations

import io
import logging
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from docx import Document
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pmo_agent import session_store
from src.pmo_agent.llm_client import chat_complete
from src.pmo_agent.prompts import get_system_prompt

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Task Management PMO Agent")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB
ALLOWED_EXTENSIONS = {"txt", "md", "csv", "docx"}
SESSION_COOKIE_NAME = "chat_session_id"
_CONTEXT_SNIPPET = 2_000
_TEXT_PREVIEW_LIMIT = 500


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


class ChatSendRequest(BaseModel):
    """Schema for chat messages."""

    message: str


class MemoryAddRequest(BaseModel):
    """Schema for adding free-form notes to memory."""

    label: str
    content: str


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@app.get("/workspace")
def get_workspace_page() -> FileResponse:
    """Serve the combined upload/chat workspace UI."""

    return FileResponse(STATIC_DIR / "workspace.html")


@app.middleware("http")
async def ensure_session_cookie(request: Request, call_next):
    """Ensure every request has a stable chat session identifier."""

    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    new_session = False

    if not session_id:
        session_id = str(uuid.uuid4())
        new_session = True
        logger.debug("Generated new chat session id %s", session_id)

    request.state.chat_session_id = session_id

    response = await call_next(request)

    if new_session:
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            httponly=True,
            samesite="lax",
        )
        logger.info("Assigned chat session cookie %s", session_id)

    return response


def _get_session_id(request: Request) -> str:
    session_id = getattr(request.state, "chat_session_id", None)
    if not session_id:
        session_id = request.cookies.get(SESSION_COOKIE_NAME) or str(uuid.uuid4())
        request.state.chat_session_id = session_id
    return session_id


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
async def upload_file(request: Request, file: UploadFile = File(...)) -> JSONResponse:
    """Accept a single file upload, extract text, and store it for the session."""

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

    session_id = _get_session_id(request)
    session_store.set_context(session_id, text_content)

    response_payload: dict[str, Any] = {
        "ok": True,
        "chars": len(text_content),
        "has_context": True,
    }

    if csv_preview is not None:
        response_payload["csv_preview"] = csv_preview
    else:
        response_payload["text_preview"] = text_content[:_TEXT_PREVIEW_LIMIT]

    logger.info("Stored uploaded file %s for session %s", file.filename, session_id)
    return JSONResponse(status_code=status.HTTP_200_OK, content=response_payload)


@app.post("/chat/send")
async def chat_send(payload: ChatSendRequest, request: Request) -> JSONResponse:
    """Send a chat message to Gemini using session-scoped context."""

    message = (payload.message or "").strip()
    if not message:
        return _json_error(status.HTTP_400_BAD_REQUEST, "Message must not be empty.")

    session_id = _get_session_id(request)
    history = session_store.get_history(session_id)
    context = session_store.get_context(session_id)

    system_prompt = get_system_prompt()
    used_context = False
    if context:
        context_snippet = context[:_CONTEXT_SNIPPET]
        system_prompt = (
            f"{system_prompt}\n\nContext from latest uploaded file (may be truncated):\n{context_snippet}"
        )
        used_context = True

    messages = history + [{"role": "user", "content": message}]

    try:
        reply = chat_complete(system_prompt, messages)
    except ValueError as exc:
        logger.error("Invalid chat payload", exc_info=exc)
        return _json_error(status.HTTP_400_BAD_REQUEST, str(exc))
    except EnvironmentError as exc:
        logger.error("Chat failed due to missing configuration", exc_info=exc)
        return _json_error(status.HTTP_400_BAD_REQUEST, str(exc))
    except Exception as exc:  # pragma: no cover - external API errors
        logger.error("Gemini chat request failed", exc_info=exc)
        return _json_error(status.HTTP_500_INTERNAL_SERVER_ERROR, "Gemini API request failed.")

    session_store.append_user(session_id, message)
    session_store.append_assistant(session_id, reply)

    logger.info("Chat message processed for session %s (used_context=%s)", session_id, used_context)
    return JSONResponse(status_code=status.HTTP_200_OK, content={"reply": reply, "used_context": used_context})
