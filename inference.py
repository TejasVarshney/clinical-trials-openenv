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
    MAX_TOKENS     — completion token budget per step (default: 1024)
"""

from __future__ import annotations

import json
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any

from openai import OpenAI

from clinical_trial_env import ClinicalTrialAction, ClinicalTrialEnv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def load_dotenv_file(path: str = ".env") -> None:
    """Load simple KEY=VALUE pairs from a local .env file into os.environ."""
    env_path = Path(path)
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        # Preserve explicitly exported environment variables.
        os.environ.setdefault(key, value)


load_dotenv_file()


def env_int(name: str, default: int) -> int:
    """Read an integer env var with a safe fallback."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default

API_BASE_URL = os.environ.get("API_BASE_URL")
MODEL_NAME = os.environ.get("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.environ.get("API_KEY") or HF_TOKEN
ENV_URL = os.environ.get("ENV_URL")
MAX_TOKENS = env_int("MAX_TOKENS", 8192)

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


def message_content_to_text(message: Any) -> str:
    """Convert provider-specific message content shapes to plain text."""
    content = getattr(message, "content", "")

    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                value = part.get("text")
                if isinstance(value, str):
                    parts.append(value)
            else:
                value = getattr(part, "text", None)
                if isinstance(value, str):
                    parts.append(value)
        text = "".join(parts)
    else:
        text = str(content or "")

    if text.strip():
        return text

    # Some OpenAI-compatible providers place text in model_extra fields.
    model_extra = getattr(message, "model_extra", None)
    if isinstance(model_extra, dict):
        for key in ("reasoning_content", "content"):
            value = model_extra.get(key)
            if isinstance(value, str) and value.strip():
                return value

    return text


def parse_llm_response(text: str) -> dict:
    """Extract a JSON action from the LLM response text."""
    text = text.strip()
    if not text:
        raise ValueError("Empty model response")

    candidates = [text]
    fenced_blocks = re.findall(r"```(?:json|javascript|js)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in fenced_blocks if block.strip())

    decoder = json.JSONDecoder()

    for candidate in candidates:
        if not candidate:
            continue

        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        # Fallback: find first decodable JSON object inside free-form text.
        for i, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[i:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

    preview = text.replace("\n", " ")[:160]
    raise ValueError(f"No JSON object found in model response: {preview}")


def normalize_action(action: dict) -> dict:
    """Coerce model outputs into a valid ClinicalTrialAction-shaped dict."""
    if not isinstance(action, dict):
        return {"action_type": "reject", "reason": "Model returned non-object action"}

    action_type = str(action.get("action_type", "")).strip().lower()

    if action_type == "enroll":
        trial_id = action.get("trial_id")
        if isinstance(trial_id, str) and trial_id.strip():
            return {"action_type": "enroll", "trial_id": trial_id.strip()}
        return {"action_type": "reject", "reason": "Missing trial_id in model response"}

    if action_type == "request_lab":
        test_name = action.get("test_name")
        if isinstance(test_name, str) and test_name.strip():
            return {"action_type": "request_lab", "test_name": test_name.strip()}
        return {"action_type": "reject", "reason": "Missing test_name in model response"}

    if action_type == "reject":
        reason = action.get("reason")
        if isinstance(reason, str) and reason.strip():
            return {"action_type": "reject", "reason": reason.strip()}
        return {"action_type": "reject", "reason": "Did not meet eligibility criteria"}

    return {"action_type": "reject", "reason": "Invalid action_type in model response"}


def request_llm_action(client: OpenAI, system_prompt: str, user_prompt: str) -> dict:
    """Request one action from the model with one strict-JSON retry."""
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=base_messages,
        max_tokens=MAX_TOKENS,
        temperature=0.0,
    )
    first_text = message_content_to_text(completion.choices[0].message)

    try:
        return normalize_action(parse_llm_response(first_text))
    except Exception as first_error:
        retry_messages = [
            *base_messages,
            {
                "role": "user",
                "content": (
                    "Your previous response was invalid. Return ONLY one valid JSON object "
                    'with double quotes and no extra text. Use one of: '
                    '{"action_type":"enroll","trial_id":"<TRIAL_ID>"}, '
                    '{"action_type":"reject","reason":"<brief reason>"}, '
                    'or {"action_type":"request_lab","test_name":"<lab>"}.'
                ),
            },
        ]

        retry_completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=retry_messages,
            max_tokens=MAX_TOKENS,
            temperature=0.0,
        )
        retry_text = message_content_to_text(retry_completion.choices[0].message)

        try:
            return normalize_action(parse_llm_response(retry_text))
        except Exception as second_error:
            raise ValueError(f"{first_error}; retry failed: {second_error}")


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
                action = request_llm_action(client, system_prompt, user_prompt)
            except Exception as e:
                # Fallback: reject if LLM fails
                action = {"action_type": "reject", "reason": f"LLM error: {e}"}

            # Safety: cap lab requests per patient
            if action.get("action_type") == "request_lab":
                lab_requests_this_patient += 1
                if lab_requests_this_patient > MAX_LAB_REQUESTS_PER_PATIENT:
                    action = {"action_type": "reject", "reason": "Max lab requests exceeded"}
                    lab_requests_this_patient = 0
            else:
                lab_requests_this_patient = 0

            try:
                step_result = env_step(env, action)
            except Exception as e:
                log(
                    "STEP",
                    {
                        "step": step_num,
                        "action": json.dumps(action, sort_keys=True),
                        "reward": 0.0,
                        "done": False,
                        "error": str(e),
                    },
                )
                continue

            response_body = step_result
            obs = response_body.get("observation", response_body)
            reward = response_body.get("reward")
            if reward is None:
                reward = obs.get("reward", 0.0)
            done = response_body.get("done", False) or obs.get("done", False)
            error = response_body.get("error")
            rewards.append(reward if reward is not None else 0.0)

            log(
                "STEP",
                {
                    "step": step_num,
                    "action": json.dumps(action, sort_keys=True),
                    "reward": reward,
                    "done": done,
                    "error": error,
                },
            )

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
