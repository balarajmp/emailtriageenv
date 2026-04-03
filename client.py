from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State


from .models import EmailItem, EmailTriageAction, EmailTriageObservation

class EmailTriageEnvClient(EnvClient[EmailTriageAction, EmailTriageObservation, State]):
    """Client for the EmailTriageEnv environment."""

    def _step_payload(self, action: EmailTriageAction) -> Dict:
        return action.model_dump()

    def _parse_result(self, payload: Dict) -> StepResult[EmailTriageObservation]:
        obs_data = payload.get("observation", {})
        observation = EmailTriageObservation(
            emails=[EmailItem(**email) for email in obs_data.get("emails", [])],
            processed_email_ids=obs_data.get("processed_email_ids", []),
            pending_email_ids=obs_data.get("pending_email_ids", []),
            per_email_scores=obs_data.get("per_email_scores", {}),
            message=obs_data.get("message", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )


MyEnv = EmailTriageEnvClient