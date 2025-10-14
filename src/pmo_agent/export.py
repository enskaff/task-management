"""Export utilities for the PMO agent."""

from pathlib import Path
from typing import Iterable

import pandas as pd

from .schemas import Task


def export_tasks_to_csv(tasks: Iterable[Task], path: Path) -> None:
    df = pd.DataFrame([task.model_dump() for task in tasks])
    df.to_csv(path, index=False)
