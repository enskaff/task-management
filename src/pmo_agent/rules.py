"""Rule evaluation for project plans."""

from collections.abc import Iterable

from .schemas import ProjectPlan, Task


class RuleViolation(Exception):
    """Raised when a project plan violates predefined rules."""


def ensure_unique_task_ids(plan: ProjectPlan) -> None:
    seen: set[str] = set()
    for task in plan.tasks:
        if task.id in seen:
            raise RuleViolation(f"Duplicate task id detected: {task.id}")
        seen.add(task.id)


def validate_plan(plan: ProjectPlan, rules: Iterable) -> None:
    for rule in rules:
        rule(plan)


def default_rules() -> list:
    return [ensure_unique_task_ids]
