# CLAUDE.md — Clinical Trial Patient Matcher (OpenEnv Hackathon)

## Project Context

**Team:** pseudo su (Tejas Varshney, Juhil Modi, Harshit Tiwari)  
**Hackathon:** OpenEnv Round 1 — Scaler × Hugging Face  
**Deadline:** April 8, 11:59 PM  
**Goal:** Build a flawless, real-world RL environment using the OpenEnv framework, deployed as a Hugging Face Space.

The environment simulates a **Clinical Research Coordinator (CRC)** who must match patients to active clinical trials by parsing unstructured Electronic Health Record (EHR) summaries against strict inclusion/exclusion criteria.

---

## Folder Structure

```
clinical-trial-matcher/
├── CLAUDE.md                  # This file
├── README.md
├── openenv.yaml               # OpenEnv spec config (REQUIRED)
├── Dockerfile                 # Must build and run cleanly
├── inference.py               # Baseline script (root, uses OpenAI client)
├── requirements.txt
├── env/
│   ├── __init__.py
│   ├── env.py                 # Core environment: step(), reset(), state()
│   ├── models.py              # Pydantic models: Observation, Action, Reward
│   ├── grader.py              # Deterministic graders for all 3 tasks
│   └── data/
│       ├── ground_truth.json  # Hidden mapping: patient_id → trial_id | 'reject'
│       ├── task1_data.json    # 10 patients, 1 trial (Easy)
│       ├── task2_data.json    # 20 patients, 3 trials (Medium)
│       └── task3_data.json    # 30 patients, 5 trials (Hard)
└── tests/
    ├── test_env.py
    └── test_grader.py
```

---

## Strict OpenEnv Requirements (Disqualification if missed)

- [ ] `step(action)` → returns `(observation, reward, done, info)`
- [ ] `reset()` → returns initial `Observation`
- [ ] `state()` → returns current environment state
- [ ] Pydantic models for `Observation`, `Action`, `Reward` (strictly typed, documented)
- [ ] Valid `openenv.yaml` at the root
- [ ] Exactly **3 tasks** with deterministic graders returning scores in `[0.0, 1.0]`
- [ ] Continuous reward shaping (not binary pass/fail)
- [ ] Working `Dockerfile` (deployable as a Hugging Face Space)
- [ ] `inference.py` at root using OpenAI client; reads `API_BASE_URL`, `MODEL_NAME`, `HF_TOKEN`, `OPENAI_API_KEY`
- [ ] Inference script emits strict stdout logs: `[START]`, `[STEP]`, `[END]`
- [ ] Inference runtime < 20 minutes on 2 vCPU / 8 GB RAM

---

## Pydantic Models (`env/models.py`)

### Observation
```python
class TrialInfo(BaseModel):
    trial_id: str
    inclusion_criteria: str
    exclusion_criteria: str

class Observation(BaseModel):
    active_patient_ehr: str        # Unstructured doctor's note
    available_trials: list[TrialInfo]
    patients_remaining: int
```

### Action (Union type — agent picks one)
```python
class Enroll(BaseModel):
    action_type: Literal["enroll"]
    trial_id: str                  # Must match a trial_id in available_trials

class Reject(BaseModel):
    action_type: Literal["reject"]
    reason: str                    # Free-text rationale

class RequestLabResult(BaseModel):
    action_type: Literal["request_lab"]
    test_name: str                 # e.g. "HbA1c", "eGFR", "LDL"

Action = Union[Enroll, Reject, RequestLabResult]
```

### Reward
```python
class Reward(BaseModel):
    value: float                   # Always in [-1.0, +1.0]
    reason: str                    # Human-readable explanation
```

---

## The 3 Tasks

### Task 1 — Easy: Explicit Matching
- **Patients:** 10 | **Trials:** 1
- **Logic:** EHR language exactly mirrors trial protocol terminology.  
  _Example: Trial excludes "Asthma" → EHR says "Patient has Asthma."_
- **Target agent behavior:** Direct keyword matching, no inference needed.

### Task 2 — Medium: Ontology & Inference
- **Patients:** 20 | **Trials:** 3
- **Logic:** Agent must infer conditions from clinical proxies.  
  _Example: Trial requires "Hypertension" → EHR lists "Lisinopril" + BP "150/95 mmHg."_
- **Target agent behavior:** Medical reasoning, drug-indication mapping.

### Task 3 — Hard: Incomplete Data
- **Patients:** 30 | **Trials:** 5
- **Logic:** Some EHRs are deliberately missing required lab values. Agent must call `RequestLabResult` before enrolling. Blind enrollment triggers a heavy penalty.  
  _Example: Trial requires HbA1c < 7.5% → EHR doesn't include it → Agent must request "HbA1c"._
- **Target agent behavior:** Recognise missing data, request it, then decide.

---

## Reward Logic (`env/grader.py`)

| Event | Reward |
|---|---|
| Correct enrollment OR accurate rejection | `+1.0` |
| Enrollment violating an Exclusion Criterion | `-1.0` |
| Unnecessary `RequestLabResult` (data was already present) | `-0.1` |

**End-of-episode normalization:**
```
final_score = clamp(total_reward / max_possible_reward, 0.0, 1.0)
```

Ground truth lives in `env/data/ground_truth.json`:
```json
{
  "P001": "TRIAL_A",
  "P002": "reject",
  "P003": "TRIAL_B",
  ...
}
```

Graders are **deterministic** — no randomness, no LLM calls inside the grader.

---

## Environment State Machine

```
reset()
  └─> loads task data, shuffles patient queue, returns Observation

step(Enroll | Reject)
  └─> grades action against ground truth
  └─> advances patient queue
  └─> returns next Observation (or done=True if queue empty)

step(RequestLabResult)
  └─> returns same Observation + hidden lab value appended to EHR
  └─> does NOT advance queue
  └─> penalizes if lab was already present in EHR

state()
  └─> returns current patient index, remaining queue, cumulative reward
```

---

## `inference.py` — Required Stdout Format

```
[START] {"task": "...", "env": "clinical-trial-matcher", "model": "..."}
[STEP]  {"step": 1, "action": "...", "reward": 1.0, "done": false, "error": null}
[STEP]  {"step": 2, "action": "...", "reward": -0.1, "done": false, "error": null}
...
[END]   {"success": true, "steps": 10, "score": 0.85, "rewards": [...]}
```

Environment variables read by `inference.py`:
```bash
API_BASE_URL     # LLM endpoint
MODEL_NAME       # e.g. "nvidia/Llama-3.1-Nemotron-70B-Instruct"
HF_TOKEN         # Hugging Face token
OPENAI_API_KEY   # API key (may alias HF_TOKEN)
```

---

## Code Style Rules

1. **Python 3.10+** — use `X | Y` union syntax, `match` statements where appropriate.
2. **Pydantic v2** — all models use `model_validator`, `field_validator` (not v1 `@validator`).
3. **No randomness in graders** — state is seeded and deterministic; same task always produces same ground truth.
4. **No LLM calls inside the environment** — only in `inference.py`.
5. **Type everything** — no `Any`, no bare `dict`. Every function has return type annotations.
6. **Document all Pydantic fields** — agents read the schema; good docstrings directly improve agent performance.
7. **Reward reasons are human-readable** — helps with debugging and scoring transparency.

---

## Submission Checklist

```bash
# 1. Validate spec
pip install openenv-core
openenv validate

# 2. Build Docker
docker build -t clinical-trial-matcher .
docker run -p 7860:7860 clinical-trial-matcher

# 3. Run baseline inference
export API_BASE_URL=...
export MODEL_NAME=...
export HF_TOKEN=...
python inference.py

# 4. Verify logs contain [START], [STEP]×n, [END]
# 5. Confirm score in [0.0, 1.0]
# 6. Submit via team lead (Juhil Modi) on Scaler portal
```

---

## AI Code Generation Notes

When generating code for this repo:
- Follow the Pydantic models above exactly — do not rename fields.
- Graders must only reference `ground_truth.json`; never compute ground truth dynamically.
- `RequestLabResult` penalty is only applied if the test was **already present** in the EHR text.
- The `available_trials` list in `Observation` must always reflect the full trial pool for the current task, not just eligible ones — the agent must do the filtering.
- Keep `env.py` and `grader.py` decoupled — grader takes `(patient_id, action)` and returns `Reward`, nothing else.