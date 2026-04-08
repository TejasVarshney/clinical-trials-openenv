"""
Baseline inference script for Clinical Trial Patient Matcher.

Runs an LLM agent through all 3 tasks via the environment HTTP API.
Emits strictly formatted [START], [STEP], [END] logs to stdout.

Environment variables:
    API_BASE_URL   — LLM endpoint (e.g. https://api-inference.huggingface.co/v1)
    MODEL_NAME     — model identifier
    HF_TOKEN       — Hugging Face token
    OPENAI_API_KEY — API key (may alias HF_TOKEN)
    ENV_URL        — environment server URL (default: http://localhost:8000)
"""

from __future__ import annotations

import json
import os
import sys
import traceback

from openai import OpenAI

from clinical_trial_env import ClinicalTrialAction, ClinicalTrialEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "") or HF_TOKEN
ENV_URL = os.environ.get("ENV_URL", "http://localhost:8000")

TASKS = ["task1", "task2", "task3"]
MAX_LAB_REQUESTS_PER_PATIENT = 3  # safety cap

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(tag: str, payload: dict) -> None:
    """Print a log line in the required format."""
    print(f"[{tag}] {json.dumps(payload)}", flush=True)


def env_reset(env: ClinicalTrialEnv, task_id: str) -> dict:
    """Reset the persistent environment and return a response-shaped dict."""
    result = env.reset(task_id=task_id)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
    }


def env_step(env: ClinicalTrialEnv, action: dict) -> dict:
    """Execute one step in the persistent environment and return a response-shaped dict."""
    action_obj = ClinicalTrialAction(**action)
    result = env.step(action_obj)
    return {
        "observation": result.observation.model_dump(),
        "reward": result.reward,
        "done": result.done,
    }


def build_system_prompt(trials: list[dict]) -> str:
    """Create the system prompt that teaches the LLM how to act as a CRC."""
    trial_descriptions = ""
    for t in trials:
        trial_descriptions += (
            f"\n--- Trial {t['trial_id']}: {t['title']} ---\n"
            f"Inclusion: {t['inclusion_criteria']}\n"
            f"Exclusion: {t['exclusion_criteria']}\n"
        )

    return f"""You are a Clinical Research Coordinator (CRC). Your job is to
review a patient's Electronic Health Record (EHR) and decide whether they
qualify for one of the available clinical trials.

AVAILABLE TRIALS:
{trial_descriptions}

For each patient you MUST respond with EXACTLY ONE JSON object (no markdown,
no extra text). Choose one of these actions:

1. Enroll the patient in a trial:
   {{"action_type": "enroll", "trial_id": "<TRIAL_ID>"}}

2. Reject the patient (does not qualify for any trial):
   {{"action_type": "reject", "reason": "<brief reason>"}}

3. Request a missing lab result before deciding:
   {{"action_type": "request_lab", "test_name": "<lab test name>"}}
   Only use this if the EHR is missing a lab value required by a trial's
   inclusion/exclusion criteria. Common labs: eGFR, HbA1c, ALT, Platelet count,
   CrCl, FEV1, PHQ-9.

RULES:
- Check ALL inclusion criteria AND exclusion criteria carefully.
- If the patient violates ANY exclusion criterion, reject them.
- If the patient doesn't meet ALL inclusion criteria for any trial, reject them.
- If a required lab value is missing from the EHR, request it first.
- Respond with raw JSON only. No explanation, no markdown fences."""


def build_user_prompt(ehr: str, patients_remaining: int) -> str:
    """Build the user message with the current patient's EHR."""
    return (
        f"Patients remaining: {patients_remaining}\n\n"
        f"PATIENT EHR:\n{ehr}\n\n"
        "Respond with your action as a JSON object."
    )


def parse_llm_response(text: str) -> dict:
    """Extract a JSON action from the LLM response text."""
    text = text.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    return json.loads(text)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_task(client: OpenAI, task_id: str) -> dict:
    """Run one task end-to-end and return the summary."""
    with ClinicalTrialEnv(base_url=ENV_URL).sync() as env:
        # Reset environment within a persistent session.
        reset_resp = env_reset(env, task_id)
        obs = reset_resp.get("observation", reset_resp)

        trials = obs.get("available_trials", [])
        system_prompt = build_system_prompt(trials)

        log("START", {"task": task_id, "env": "clinical-trial-matcher", "model": MODEL_NAME})

        step_num = 0
        rewards: list[float] = []
        done = False
        lab_requests_this_patient = 0

        while not done:
            step_num += 1
            ehr = obs.get("active_patient_ehr", "")
            remaining = obs.get("patients_remaining", 0)

            user_prompt = build_user_prompt(ehr, remaining)

            try:
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=256,
                    temperature=0.0,
                )
                llm_text = completion.choices[0].message.content or ""
                action = parse_llm_response(llm_text)
            except Exception as e:
                # Fallback: reject if LLM fails
                action = {"action_type": "reject", "reason": f"LLM error: {e}"}

            # Safety: cap lab requests per patient
            if action.get("action_type") == "request_lab":
                lab_requests_this_patient += 1
                if lab_requests_this_patient > MAX_LAB_REQUESTS_PER_PATIENT:
                    action = {"action_type": "reject", "reason": "Max lab requests exceeded"}
            else:
                lab_requests_this_patient = 0

            try:
                step_result = env_step(env, action)
            except Exception as e:
                log("STEP", {
                    "step": step_num,
                    "request": action,
                    "response": None,
                    "reward": 0.0,
                    "done": False,
                    "error": str(e),
                })
                continue

            response_body = step_result
            obs = response_body.get("observation", response_body)
            reward = response_body.get("reward") or obs.get("reward", 0.0)
            done = response_body.get("done", False) or obs.get("done", False)
            error = response_body.get("error")
            rewards.append(reward if reward is not None else 0.0)

            log("STEP", {
                "step": step_num,
                "request": action,
                "response": response_body,
                "reward": reward,
                "done": done,
                "error": error,
            })

        # Extract final score
        metadata = obs.get("metadata", {})
        score = metadata.get("final_score", 0.0)

        summary = {
            "success": True,
            "steps": step_num,
            "score": score,
            "rewards": rewards,
        }
        log("END", summary)
        return summary


def main() -> None:
    if not API_BASE_URL:
        print("ERROR: API_BASE_URL environment variable not set", file=sys.stderr)
        sys.exit(1)
    if not MODEL_NAME:
        print("ERROR: MODEL_NAME environment variable not set", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(base_url=API_BASE_URL, api_key=OPENAI_API_KEY)

    for task_id in TASKS:
        try:
            run_task(client, task_id)
        except Exception:
            traceback.print_exc(file=sys.stderr)
            log("END", {"success": False, "task": task_id, "error": traceback.format_exc()})


if __name__ == "__main__":
    main()
