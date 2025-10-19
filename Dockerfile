FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy project metadata and code before installing
COPY pyproject.toml ./
COPY src ./src

# Install dependencies from pyproject.toml
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 7860

# Use the port directly (Render injects $PORT if needed)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
