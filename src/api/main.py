"""FastAPI application entrypoint."""

from fastapi import FastAPI

app = FastAPI(title="Task Management PMO Agent")


@app.get("/")
def read_root() -> dict[str, str]:
    """Basic health endpoint."""

    return {"status": "ok"}
