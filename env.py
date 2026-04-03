from __future__ import annotations

import json
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


class EmailTriageEnvironment(Environment):
    """Intelligent Email Triage + Action environment (industry-grade)."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._task_bank = self._load_tasks()
        self._active_task: Optional[Dict[str, Any]] = None
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

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs: Any):
        selected = self._select_task(task_id=kwargs.get("task_id"), difficulty=kwargs.get("difficulty"))
        self._task_id = selected["task_id"]

        self._state = State(episode_id=episode_id or str(uuid4()), step_count=0)
        self._active_task = selected
        self._emails = {email["id"]: email for email in selected["emails"]}
        self._gold = {label["email_id"]: label for label in selected["labels"]}
        self._processed = {}
        self._total_reward = 0.0

        return EmailTriageObservation(
            emails=[EmailItem(**email) for email in selected["emails"]],
            processed_email_ids=[],
            pending_email_ids=list(self._emails.keys()),
            per_email_scores={},
            done=False,
            reward=0.0,
            metadata={
                "task_id": self._task_id,
                "difficulty": selected["difficulty"],
                "description": selected["description"],
                "real_world": "enterprise_email_triage_with_sla",
            },
            message="Environment reset. Start triaging emails.",
        )

    def step(self, action: EmailTriageAction, timeout_s: Optional[float] = None, **kwargs: Any):
        if self._active_task is None:
            raise RuntimeError("Call reset() first.")

        self._state.step_count += 1

        if action.email_id not in self._emails:
            return self._build_observation(-0.3, False, "Invalid email_id")

        if action.email_id in self._processed:
            return self._build_observation(-0.2, False, "Duplicate action")

        expected = self._gold[action.email_id]

        score, details = grade_single_decision(
            prediction={
                "classification": action.classification,
                "priority": action.priority,
                "action": action.action,
            },
            expected={
                "classification": expected["classification"],
                "priority": expected["priority"],
                "action": expected["action"],
            },
        )

        reward = details["reward"]


        email_data = self._emails[action.email_id]
        if email_data.get("is_vip") and score < 0.5:
            reward -= 0.3

        if email_data.get("deadline"):
            if action.classification != "urgent" and action.action != "escalate":
                reward -= 0.2


        if len(self._processed) + 1 == len(self._emails):
            if self._state.step_count <= len(self._emails):
                reward += 0.1

        self._total_reward += reward

        self._processed[action.email_id] = {
            "score": score,
            "prediction": action.model_dump(),
        }

        done = len(self._processed) == len(self._emails)

        return self._build_observation(
            reward,
            done,
            f"Processed {action.email_id}",
            metadata_extra={
                "episode_total_reward": round(self._total_reward, 4)
            },
        )

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