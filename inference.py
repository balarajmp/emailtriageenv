from __future__ import annotations

import json
import os
import asyncio
import sys
from typing import Any, Dict, List

from openai import OpenAI

# ✅ Support BOTH execution modes (script + package)
try:
    from .client import EmailTriageEnvClient
    from .models import EmailTriageAction
except ImportError:
    from client import EmailTriageEnvClient
    from models import EmailTriageAction


MAX_STEPS = 6
TEMPERATURE = 0.0
MAX_TOKENS = 100
SUCCESS_SCORE_THRESHOLD = 0.7


def _fallback_policy(email: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based safe fallback policy"""
    text = f"{email.get('subject', '')} {email.get('body', '')}".lower()
    is_vip = bool(email.get("is_vip", False))

    if any(word in text for word in ["free", "verify", "won", "payout", "click"]):
        return {"classification": "spam", "priority": 1, "action": "ignore"}

    if is_vip or any(word in text for word in ["urgent", "deadline", "today", "asap", "eod"]):
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
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start:end + 1])
    raise ValueError("Model response did not contain valid JSON")


async def main() -> None:
    try:
        api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
        hf_token = os.getenv("HF_TOKEN", "")
        env_url = os.getenv("ENV_URL", "https://surajmp05-emailtriageenv.hf.space")

        print(f"[START] task=email-triage env=EmailTriageEnv model={model_name}")

        # ✅ Safe LLM init
        try:
            llm_client = OpenAI(base_url=api_base_url, api_key=hf_token or "no-token")
        except Exception as e:
            print(f"[ERROR] LLM init failed: {e}")
            return

        # ✅ Safe Env init
        try:
            env_client = EmailTriageEnvClient(base_url=env_url)
        except Exception as e:
            print(f"[ERROR] Env client init failed: {e}")
            return

        rewards: List[float] = []

        try:
            # 🔥 RETRY LOGIC FOR RESET
            for attempt in range(3):
                try:
                    reset_result = await env_client.reset()
                    obs = reset_result.observation
                    break
                except Exception as e:
                    print(f"[WARN] reset attempt {attempt+1} failed: {e}")
                    await asyncio.sleep(2)
            else:
                print("[ERROR] reset failed after retries")
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

                decision = _fallback_policy(email)
                decision["priority"] = max(1, min(3, decision["priority"]))

                # 🔥 RETRY LOGIC FOR STEP
                for attempt in range(3):
                    try:
                        result = await env_client.step(
                            EmailTriageAction(
                                email_id=email_id,
                                classification=decision["classification"],
                                priority=decision["priority"],
                                action=decision["action"],
                            )
                        )
                        break
                    except Exception as e:
                        print(f"[WARN] step attempt {attempt+1} failed: {e}")
                        await asyncio.sleep(2)
                else:
                    print("[ERROR] step failed after retries")
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

            print(
                f"[END] success={str(success).lower()} steps={step_count} rewards={reward_str}"
            )

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