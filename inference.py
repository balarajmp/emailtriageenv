from __future__ import annotations

import json
import os
import asyncio
import sys
from typing import Any, Dict, List

from openai import OpenAI

# ✅ Support BOTH execution modes
try:
    from .client import EmailTriageEnvClient
    from .models import EmailTriageAction
except ImportError:
    from client import EmailTriageEnvClient
    from models import EmailTriageAction


MAX_STEPS = 6
SUCCESS_SCORE_THRESHOLD = 0.7


def _fallback_policy(email: Dict[str, Any]) -> Dict[str, Any]:
    text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
    is_vip = bool(email.get("is_vip", False))

    if any(word in text for word in ["free", "verify", "won", "payout", "click", "lottery", "offer"]):
        return {"classification": "spam", "priority": 1, "action": "ignore"}

    if is_vip or any(word in text for word in ["urgent", "deadline", "today", "asap", "eod", "important"]):
        return {
            "classification": "urgent",
            "priority": 3,
            "action": "escalate" if is_vip else "reply",
        }

    return {"classification": "normal", "priority": 2, "action": "schedule"}


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start:end + 1])
    raise ValueError("Invalid JSON from LLM")


async def call_llm(llm_client: OpenAI, email: Dict[str, Any]) -> Dict[str, Any] | None:
    try:
        prompt = f"""
Classify this email strictly into JSON:

Options:
- classification: spam / urgent / normal
- priority: 1 (low), 2 (medium), 3 (high)
- action: ignore / reply / escalate / schedule

Email:
Subject: {email.get('subject', '')}
Body: {email.get('body', '')}

Return ONLY JSON.
"""

        response = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
        )

        content = response.choices[0].message.content
        return _extract_json(content)

    except Exception as e:
        print(f"[WARN] LLM failed: {e}")
        return None


async def main() -> None:
    try:
        # ✅ MUST use injected variables (CRITICAL)
        api_base_url = os.environ["API_BASE_URL"]
        api_key = os.environ["API_KEY"]

        env_url = os.getenv("ENV_URL", "https://surajmp05-emailtriageenv.hf.space")

        print("[START] task=email-triage env=EmailTriageEnv")

        llm_client = OpenAI(base_url=api_base_url, api_key=api_key)
        env_client = EmailTriageEnvClient(base_url=env_url)

        rewards: List[float] = []

        try:
            # 🔥 RETRY RESET
            for attempt in range(3):
                try:
                    reset_result = await env_client.reset()
                    obs = reset_result.observation
                    break
                except Exception as e:
                    print(f"[WARN] reset attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(2)
            else:
                print("[ERROR] reset failed")
                return

            step_count = 0

            while getattr(obs, "pending_email_ids", []) and step_count < MAX_STEPS:
                try:
                    email_id = obs.pending_email_ids[0]
                    email = next(
                        item.model_dump()
                        for item in obs.emails
                        if item.id == email_id
                    )
                except Exception as e:
                    print(f"[ERROR] email extraction failed: {e}")
                    break

                # 🔥 USE LLM (MANDATORY FOR PHASE 2)
                decision = await call_llm(llm_client, email)

                # fallback safety
                if not decision:
                    decision = _fallback_policy(email)

                decision["priority"] = max(1, min(3, decision.get("priority", 2)))

                # 🔥 RETRY STEP
                for attempt in range(3):
                    try:
                        result = await env_client.step(
                            EmailTriageAction(
                                email_id=email_id,
                                classification=decision.get("classification", "normal"),
                                priority=decision["priority"],
                                action=decision.get("action", "schedule"),
                            )
                        )
                        break
                    except Exception as e:
                        print(f"[WARN] step attempt {attempt+1} failed: {e}")
                        await asyncio.sleep(2)
                else:
                    print("[ERROR] step failed")
                    break

                obs = result.observation
                step_count += 1
                rewards.append(result.reward)

                print(
                    f"[STEP] step={step_count} action={decision} "
                    f"reward={result.reward:.2f} done={str(result.done).lower()} error=null"
                )

                if result.done:
                    break

            success = sum(rewards) >= SUCCESS_SCORE_THRESHOLD
            reward_str = ",".join(f"{r:.2f}" for r in rewards)

            print(f"[END] success={str(success).lower()} steps={step_count} rewards={reward_str}")

        finally:
            try:
                await env_client.close()
            except Exception:
                pass

    except Exception as e:
        print(f"[FATAL ERROR] {e}")

    sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())