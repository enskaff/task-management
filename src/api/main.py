"""FastAPI application entrypoint."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.pmo_agent.llm_client import generate_response

app = FastAPI(title="Task Management PMO Agent")


class LLMRequest(BaseModel):
    """Schema for LLM prompt requests."""

    prompt: str


class LLMResponse(BaseModel):
    """Schema for LLM responses."""

    response: str


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}


@app.post("/llm", response_model=LLMResponse)
def call_llm(payload: LLMRequest) -> LLMResponse:
    """Invoke the Gemini LLM with the provided prompt."""

    try:
        text = generate_response(payload.prompt)
    except EnvironmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - external API errors
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return LLMResponse(response=text)
