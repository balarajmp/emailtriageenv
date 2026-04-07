from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from .graders.email_triage_grader import grade_single_decision
    from .models import EmailItem, EmailTriageAction, EmailTriageObservation
except ImportError:
    from graders.email_triage_grader import grade_single_decision
    from models import EmailItem, EmailTriageAction, EmailTriageObservation

TASKS = [
    {
        "type": "easy",
        "email": {
            "id": "e-001",
            "sender": "promo@spam.com",
            "subject": "You won a free iPhone",
            "body": "Click now",
            "is_vip": False,
            "timestamp": "2026-01-01T10:00:00Z",
        },
    },
    {
        "type": "medium",
        "email": {
            "id": "e-002",
            "sender": "team@company.com",
            "subject": "Meeting tomorrow",
            "body": "Let's meet at 10 AM",
            "is_vip": False,
            "timestamp": "2026-01-01T11:00:00Z",
        },
    },
    {
        "type": "hard",
        "email": {
            "id": "e-003",
            "sender": "ceo@company.com",
            "subject": "URGENT: send report",
            "body": "Need financial report ASAP",
            "is_vip": True,
            "timestamp": "2026-01-01T12:00:00Z",
        },
    },
]


class EmailTriageEnvironment(Environment):
    """Intelligent Email Triage + Action environment (industry-grade)."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_bank = self._load_tasks()
        self._active_task: Optional[Dict[str, Any]] = None
        self.current_task: Optional[Dict[str, Any]] = None
        self.task_index = 0
        self._emails: Dict[str, Dict[str, Any]] = {}
        self._gold: Dict[str, Dict[str, Any]] = {}
        self._processed: Dict[str, Dict[str, Any]] = {}
        self._total_reward = 0.0
        self._task_id: Optional[str] = None

    def _load_tasks(self) -> Dict[str, Dict[str, Any]]:
        tasks_path = Path(__file__).parent / "tasks"
        task_files = [tasks_path / "easy.json", tasks_path / "medium.json", tasks_path / "hard.json"]
        tasks: Dict[str, Dict[str, Any]] = {}

        for task_file in task_files:
            data = json.loads(task_file.read_text(encoding="utf-8"))
            tasks[data["task_id"]] = data

        return tasks

    def _select_task(self, task_id: Optional[str] = None, difficulty: Optional[str] = None) -> Dict[str, Any]:
        if task_id is not None:
            return self._task_bank[task_id]

        if difficulty is not None:
            for task in self._task_bank.values():
                if task["difficulty"] == difficulty:
                    return task

        return self._task_bank["email-triage-easy"]

    def reset(self):
        self.current_task = TASKS[self.task_index % len(TASKS)]
        self.task_index += 1

        email = self.current_task["email"]

        return {
            "observation": {
                "emails": [email],
                "processed_email_ids": [],
                "pending_email_ids": [email["id"]],
                "per_email_scores": {},
                "message": "New task",
                "metadata": {},
            }
        }

    def step(self, action):
        if not hasattr(self, "current_task"):
            raise RuntimeError("Call reset() first.")

        email = self.current_task["email"]
        task_type = self.current_task["type"]

        classification = action["classification"]
        priority = action["priority"]

        correct = False

        if task_type == "easy":
            correct = classification == "spam"

        elif task_type == "medium":
            correct = classification == "normal"

        elif task_type == "hard":
            correct = classification == "urgent" and priority == 3

        # IMPORTANT: reward must be strictly between 0 and 1
        reward = 0.8 if correct else 0.2

        return {
            "observation": {
                "emails": [email],
                "processed_email_ids": [email["id"]],
                "pending_email_ids": [],
                "per_email_scores": {email["id"]: reward},
                "message": "Correct" if correct else "Incorrect",
                "metadata": {},
            },
            "reward": reward,
            "done": True,
            "info": {},
        }

    def _build_observation(self, reward, done, message, metadata_extra=None):
        pending = [eid for eid in self._emails if eid not in self._processed]

        metadata = {
            "task_id": self._task_id,
            "processed": len(self._processed),
            "total": len(self._emails),
        }

        if metadata_extra:
            metadata.update(metadata_extra)

        return EmailTriageObservation(
            emails=[EmailItem(**e) for e in self._emails.values()],
            processed_email_ids=list(self._processed.keys()),
            pending_email_ids=pending,
            per_email_scores={k: v["score"] for k, v in self._processed.items()},
            done=done,
            reward=reward,
            metadata=metadata,
            message=message,
        )

    def episode_score(self) -> float:
        if not self._processed:
            return 0.0
        return sum(v["score"] for v in self._processed.values()) / len(self._emails)

    @property
    def state(self) -> State:
        return self._state


MyEnvironment = EmailTriageEnvironment
