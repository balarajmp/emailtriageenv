---

title: EmailTriageEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# 📧 EmailTriageEnv — AI-Powered Email Decision Environment

## 🚀 Overview

**EmailTriageEnv** is a production-grade OpenEnv environment that simulates **real-world enterprise email workflows** such as:

* Customer support triage systems
* SLA-driven escalation pipelines
* Corporate inbox management

It evaluates how well AI agents can:

* 📩 Understand email context
* ⚡ Prioritize tasks intelligently
* 🧠 Make correct operational decisions under constraints

> 🔥 Built for **real-world productivity AI**, not toy problems.

---

## 🎯 Why This Matters

Modern organizations handle:

* Thousands of emails daily
* Strict SLA deadlines ⏱️
* VIP stakeholders 👑
* Critical production incidents 🚨

👉 A wrong decision can cause **real business impact**

This environment benchmarks whether AI can **think and act like a human operator**.

---

## 🧠 What the Agent Must Do

For each email, the agent outputs:

```json
{
  "email_id": "string",
  "classification": "spam | urgent | normal",
  "priority": 1 | 2 | 3,
  "action": "reply | ignore | escalate | schedule"
}
```

### ✅ Constraints

* Priority must be strictly: **1 (low), 2 (medium), 3 (high)**
* Invalid actions or duplicate handling are penalized

---

## 🏗️ Environment Design

### 📥 Observation Space

* Email list:

  * sender, subject, body
  * VIP flag 👑
  * timestamps & deadlines ⏳
* `pending_email_ids`
* `processed_email_ids`
* `per_email_scores`
* Metadata & progress

---

### ⚙️ Action Space

* Multi-step decision making
* Context-aware actions
* SLA-sensitive prioritization

---

### 🔁 API

* `reset()` → returns initial observation
* `step(action)` → returns observation, reward, done, info
* `state()` → returns current state

---

## 🧩 Task Levels

### 🟢 Easy

* Single email
* Basic classification

### 🟡 Medium

* Multiple emails
* Mixed priorities

### 🔴 Hard (Enterprise Scenario)

* CEO / CTO / HR stakeholders 👑
* Conflicting deadlines (ASAP vs EOD)
* Production incidents 🚨
* Ambiguous decision-making

> 💡 Designed to challenge **advanced AI agents**

---

## 📊 Scoring System

| Component      | Weight |
| -------------- | ------ |
| Classification | 0.4    |
| Priority       | 0.3    |
| Action         | 0.3    |

Score Range: **0.0 → 1.0**

---

## 🎯 Reward Design

✔ Correct decisions → positive reward
✔ Partial correctness → proportional reward
❌ Mistakes → penalties

Penalizes:

* ❌ Incorrect VIP handling
* ❌ Missed deadlines (SLA violations)
* ❌ Duplicate or invalid actions

🎁 Efficiency bonus for optimal decisions

---

## ✨ Key Features

* 🏢 Real enterprise workflow simulation
* 🎯 Deterministic scoring
* 🔄 Multi-step reasoning
* ⏱️ SLA-aware decisions
* 👑 VIP prioritization
* 🐳 Docker-ready deployment
* ☁️ Hugging Face Space deployment

---

## ⚡ Quick Start

### Install

```bash
pip install openenv-core huggingface_hub openai
```

---

### Run Locally

```bash
openenv validate
uv run server
```

---

### Docker (HF Compatible)

```bash
docker build -t emailtriageenv:latest .
docker run -p 7860:7860 emailtriageenv:latest
```

---

### Inference

```bash
python -m my_env.inference
```

---

## 🌐 Deployment

```bash
openenv push --repo-id surajmp05/emailtriageenv
```

Live Space:
👉 https://surajmp05-emailtriageenv.hf.space

---

## 📈 Example Inference Output

```text
[START] task=email-triage env=EmailTriageEnv model=gpt-4o-mini
[STEP] step=1 action={'classification': 'spam', 'priority': 1, 'action': 'ignore'} reward=0.50 done=false error=null
[STEP] step=2 action={'classification': 'urgent', 'priority': 3, 'action': 'reply'} reward=1.00 done=true error=null
[END] success=true steps=2 rewards=0.50,1.00
```

---

## 📈 Baseline Performance

* 🟢 Easy → Near perfect
* 🟡 Medium → Moderate
* 🔴 Hard → Requires reasoning

👉 Exposes gaps in agent decision-making

---

## 🏆 Why This Project Stands Out

* Not just classification → **decision intelligence**
* Real-world constraints → **SLA + VIP handling**
* Multi-step reasoning → **agent evaluation benchmark**
* Enforces strict action validity and priority constraints

---

## 🔮 Future Improvements

* 📊 Decision monitoring dashboard
* 🧠 Memory-enabled agents
* 🤖 Multi-agent collaboration
* 📡 Real email integrations

---

## 🤝 Contributing

Pull requests welcome!
Let’s build the future of **AI productivity systems** 🚀

---

## 📜 License

MIT License

---

## 💡 Final Thought

> The future of AI is not just answering questions —
> it’s making the **right decisions at the right time**.

**EmailTriageEnv is a step toward that future.**
