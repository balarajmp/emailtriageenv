from __future__ import annotations

import json
import os
import asyncio
from typing import Any, Dict

from openai import OpenAI

from client import EmailTriageEnvClient
from models import EmailTriageAction



MAX_STEPS = 6
TEMPERATURE = 0.0
MAX_TOKENS = 100
SUCCESS_SCORE_THRESHOLD = 0.7


def _fallback_policy(email: Dict[str, Any]) -> Dict[str, Any]:
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

    return {"classification": "normal", "priority": 1, "action": "schedule"}


def _extract_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
    raise ValueError("Model response did not contain valid JSON")


def _llm_decision(client: OpenAI, model_name: str, email: Dict[str, Any]) -> Dict[str, Any]:
    prompt = {
        "email": email,
        "valid_classification": ["spam", "urgent", "normal"],
        "valid_priority": [1, 2, 3],
        "valid_action": ["reply", "ignore", "escalate", "schedule"],
        "instruction": "Return JSON only with keys: classification, priority, action.",
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are an accurate email triage assistant. Output JSON only."},
            {"role": "user", "content": json.dumps(prompt)},
        ],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
    )

    raw = response.choices[0].message.content or "{}"
    return _extract_json(raw)


# ✅ ASYNC MAIN
async def main() -> None:
    api_base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN", "")
    image_name = os.getenv("LOCAL_IMAGE_NAME", "emailtriageenv:latest")

    # 🔥 STRICT LOG FORMAT
    print(f"[START] task=email-triage env=EmailTriageEnv model={model_name}")

    llm_client = OpenAI(base_url=api_base_url, api_key=hf_token or "no-token")

    env_client = await EmailTriageEnvClient.from_docker_image(image_name)

    try:
        reset_result = await env_client.reset()
        obs = reset_result.observation

        step_count = 0
        total_reward = 0.0

        while obs.pending_email_ids and step_count < MAX_STEPS:
            email_id = obs.pending_email_ids[0]
            email = next(item.model_dump() for item in obs.emails if item.id == email_id)

            # 🔥 Fully deterministic policy
            decision = _fallback_policy(email)

            result = await env_client.step(
                EmailTriageAction(
                    email_id=email_id,
                    classification=decision["classification"],
                    priority=decision["priority"],
                    action=decision["action"],
                )
            )

            obs = result.observation
            step_count += 1
            total_reward += result.reward

            print(
                f"[STEP] step={step_count} action={decision} reward={result.reward:.2f} done={result.done} error=null"
            )

            if result.done:
                break

        success = total_reward >= SUCCESS_SCORE_THRESHOLD

        print(
            f"[END] success={str(success).lower()} steps={step_count} rewards={round(total_reward, 4)}"
        )

    finally:
        await env_client.close()


# ✅ RUN
if __name__ == "__main__":
    asyncio.run(main())