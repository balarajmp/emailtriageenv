from __future__ import annotations

from typing import Dict, List, Tuple

WEIGHTS = {
    "classification": 0.4,
    "priority": 0.3,
    "action": 0.3,
}

PENALTIES = {
    "classification": -0.2,
    "priority": -0.15,
    "action": -0.15,
}


def grade_single_decision(prediction: Dict, expected: Dict) -> Tuple[float, Dict[str, float]]:
    """Return weighted score [0,1] and reward with penalties."""
    component_correct = {
        "classification": float(prediction.get("classification") == expected.get("classification")),
        "priority": float(prediction.get("priority") == expected.get("priority")),
        "action": float(prediction.get("action") == expected.get("action")),
    }

    weighted_score = sum(component_correct[k] * WEIGHTS[k] for k in WEIGHTS)
    reward = sum(
        WEIGHTS[k] if component_correct[k] == 1.0 else PENALTIES[k]
        for k in WEIGHTS
    )

    return round(weighted_score, 4), {
        "classification": component_correct["classification"],
        "priority": component_correct["priority"],
        "action": component_correct["action"],
        "score": round(weighted_score, 4),
        "reward": round(reward, 4),
    }


def grade_batch(predictions: List[Dict], expected: List[Dict]) -> float:
    """Return average score across all emails, in [0.0, 1.0]."""
    if len(predictions) != len(expected) or not expected:
        return 0.0

    expected_map = {item["email_id"]: item for item in expected}
    per_email_scores = []

    for pred in predictions:
        email_id = pred.get("email_id")
        if email_id not in expected_map:
            per_email_scores.append(0.0)
            continue

        score, _ = grade_single_decision(
            prediction=pred,
            expected=expected_map[email_id],
        )
        per_email_scores.append(score)

    return round(sum(per_email_scores) / len(expected), 4)