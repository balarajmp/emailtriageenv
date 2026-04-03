# 📧 EmailTriageEnv — Intelligent Email Triage & Action Environment

## 🚀 Overview

EmailTriageEnv is a **real-world OpenEnv environment** that simulates enterprise email triage workflows used in:

* Customer support systems
* Corporate inbox management
* SLA-driven escalation pipelines

The environment evaluates an AI agent’s ability to:

* Classify emails (spam / urgent / normal)
* Assign priorities (1–3)
* Choose appropriate actions (reply / ignore / escalate / schedule)

This goes beyond simple classification by modeling **multi-step decision-making under constraints**, closely reflecting real operational systems.

---

## 🎯 Motivation

Modern organizations receive high volumes of emails requiring:

* Rapid prioritization
* Correct escalation decisions
* Deadline awareness (SLA compliance)

This environment enables benchmarking of agents on **realistic productivity tasks**, rather than synthetic or game-based scenarios.

---

## 🧠 Environment Design

### 🔹 Observation Space

Each observation contains:

* List of emails with:

  * `id`, `sender`, `subject`, `body`
  * `is_vip`, `timestamp`, optional `deadline`
* `pending_email_ids`
* `processed_email_ids`
* `per_email_scores`
* metadata (task, difficulty, progress)

---

### 🔹 Action Space

Agent must output:

```json
{
  "email_id": "string",
  "classification": "spam | urgent | normal",
  "priority": 1 | 2 | 3,
  "action": "reply | ignore | escalate | schedule"
}
```

---

### 🔹 Environment API

Implements full OpenEnv spec:

* `reset()` → initial observation
* `step(action)` → observation, reward, done, info
* `state()` → current state

---

## 🧩 Tasks

### 🟢 Easy

* Single email classification
* Basic decision making

---

### 🟡 Medium

* Multiple emails
* Mixed priorities
* Requires correct sequencing

---

### 🔴 Hard (Enterprise Scenario)

* Multiple VIP stakeholders (CEO, CTO, HR)
* Conflicting deadlines (ASAP, EOD, fixed time)
* Production-critical incidents
* Ambiguous priority decisions

👉 Designed to challenge **state-of-the-art agents**

---

## 🎯 Grading System

Each decision is evaluated using a deterministic grader:

| Component      | Weight |
| -------------- | ------ |
| Classification | 0.4    |
| Priority       | 0.3    |
| Action         | 0.3    |

Score range:

```text
0.0 → 1.0
```

---

## 💰 Reward Function

Provides **dense, meaningful feedback**:

✔ Positive reward for correct decisions
✔ Partial credit for partially correct outputs
✔ Penalties for:

* Incorrect VIP handling
* Missing deadlines (SLA violations)
* Duplicate or invalid actions

✔ Efficiency bonus for completing tasks optimally

---

## ⚙️ Key Features

* ✅ Real-world task simulation (enterprise email workflows)
* ✅ Deterministic scoring (reproducible results)
* ✅ Multi-step decision environment
* ✅ SLA-aware reward shaping
* ✅ VIP priority modeling
* ✅ Fully containerized deployment

---

## 🤖 Baseline Inference

The provided `inference.py`:

* Uses OpenAI-compatible client
* Produces **structured logs**:

```text
[START]
[STEP]
[END]
```

* Ensures **reproducible performance** across runs

---

## 🐳 Setup & Usage

### 🔹 Install dependencies

```bash
pip install openenv-core huggingface_hub openai
```

---

### 🔹 Run locally

```bash
openenv validate
uv run server
```

---

### 🔹 Build Docker

```bash
docker build -t emailtriageenv:latest .
docker run -p 8000:8000 emailtriageenv:latest
```

---

### 🔹 Run inference

```bash
python inference.py
```

---

## 🚀 Deployment

Deployed via Hugging Face Spaces:

```bash
openenv push --repo-id <your-username>/emailtriageenv
```

---

## 📊 Baseline Results

* Easy: near-perfect performance
* Medium: moderate complexity
* Hard: requires nuanced decision-making

👉 Designed to expose gaps in agent reasoning and prioritization

---

## 🏁 Conclusion

EmailTriageEnv provides a **high-fidelity, real-world benchmark** for evaluating AI agents in productivity workflows.

It emphasizes:

* Decision quality
* Priority management
* Real-world constraints

making it a valuable tool for **agent evaluation and training**.

---
