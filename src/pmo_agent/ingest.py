"""Data ingestion utilities for the PMO agent."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from .schemas import ProjectPlan, Task


class CSVIngestor:
    """Load project plans from CSV files."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def read_tasks(self) -> Iterable[Task]:
        df = pd.read_csv(self.path)
        for row in df.to_dict(orient="records"):
            yield Task(**row)

    def read_project_plan(self, name: str, description: str | None = None) -> ProjectPlan:
        return ProjectPlan(name=name, description=description, tasks=list(self.read_tasks()))
