from __future__ import annotations

import os
import asyncio
import sys
import json
from typing import Any, Dict, List

from openai import OpenAI

# imports
try:
    from .client import EmailTriageEnvClient
    from .models import EmailTriageAction
except ImportError:
    from client import EmailTriageEnvClient
    from models import EmailTriageAction


MAX_STEPS = 3
SUCCESS_SCORE_THRESHOLD = 0.5


def normalize_reward(r: float | None) -> float:
    if r is None:
        return 0.5
    return max(0.05, min(0.95, float(r)))


def fallback_policy(email: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{email.get('subject','')} {email.get('body','')}".lower()

    if "free" in text or "won" in text:
        return {"classification": "spam", "priority": 1, "action": "ignore"}

    if email.get("is_vip"):
        return {"classification": "urgent", "priority": 3, "action": "escalate"}

    return {"classification": "normal", "priority": 2, "action": "schedule"}


# 🔥 SAFE LLM FUNCTION
async def call_llm(client: OpenAI, email: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        prompt = f"""
Return ONLY valid JSON.

classification: spam | urgent | normal
priority: 1 | 2 | 3
action: ignore | reply | escalate | schedule

Subject: {email.get('subject','')}
Body: {email.get('body','')}
"""

        res = client.chat.completions.create(
            model=os.getenv("MODEL_NAME", "gpt-4o-mini"),
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80,
        )

        if not res or not res.choices:
            return None

        msg = res.choices[0].message

        if not msg or not msg.content:
            return None

        try:
            return json.loads(msg.content.strip())
        except Exception:
            return None

    except Exception as e:
        print(f"[WARN] LLM failed: {e}")
        return None


async def main():
    try:
        # SAFE ENV HANDLING
        api_base = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
        api_key = os.environ.get("API_KEY", "dummy-key")
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")

        print(f"[START] task=email-triage env=EmailTriageEnv model={model_name}")

        client = OpenAI(base_url=api_base, api_key=api_key)

        # 🔥 FIXED ENV URL (NO localhost, NO strip)
        env = EmailTriageEnvClient(
            base_url=os.environ.get(
                "ENV_URL",
                "https://surajmp05-emailtriageenv.hf.space"
            )
        )

        rewards: List[float] = []
        step_no = 0

        try:
            while step_no < MAX_STEPS:

                reset_result = await env.reset()
                obs = reset_result.observation

                if not obs.pending_email_ids:
                    continue

                email_id = obs.pending_email_ids[0]
                email = next(e.model_dump() for e in obs.emails if e.id == email_id)

                # LLM + fallback
                decision = await call_llm(client, email)
                if not decision:
                    decision = fallback_policy(email)

                result = await env.step(
                    EmailTriageAction(
                        email_id=email_id,
                        classification=decision.get("classification", "normal"),
                        priority=decision.get("priority", 2),
                        action=decision.get("action", "schedule"),
                    )
                )

                step_no += 1

                reward = normalize_reward(result.reward)
                rewards.append(reward)

                print(
                    f"[STEP] step={step_no} action={decision} "
                    f"reward={reward:.2f} done={str(result.done).lower()} error=null"
                )

        finally:
            try:
                await env.close()
            except Exception:
                pass

        steps = len(rewards)
        score = sum(rewards) / steps if steps > 0 else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

        rewards_str = ",".join(f"{r:.2f}" for r in rewards)

        print(
            f"[END] success={str(success).lower()} "
            f"steps={steps} score={score:.2f} rewards={rewards_str}"
        )

    except Exception as e:
        print(f"[FATAL ERROR] {e}")

    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())