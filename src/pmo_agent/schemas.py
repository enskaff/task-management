"""Pydantic schemas for PMO agent data structures."""

from datetime import date
from typing import List, Optional

from pydantic import BaseModel


class Task(BaseModel):
    """Represents a single task in a project plan."""

    id: str
    name: str
    owner: Optional[str] = None
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    status: Optional[str] = None


class ProjectPlan(BaseModel):
    """Collection of tasks comprising a project plan."""

    name: str
    description: Optional[str] = None
    tasks: List[Task] = []
