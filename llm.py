"""
llm.py — Multi-provider LLM router for forumbots.

Supported providers (set via persona["llm"]):
    "koboldai"  — AI Horde cloud (aihorde.net) — async submit/poll
    "ollama"    — Ollama local inference

AI Horde notes:
    - Free cloud service, crowdsourced volunteer GPUs
    - API key from https://lite.koboldai.net or https://aihorde.net/register
    - Generation is async: submit job → poll until done
    - "models" is a list of acceptable models; empty list = any available worker
    - Set preferred models per-persona via "llm_model" (single string) or
      "llm_models" (list of strings) in the persona JSON
"""

import logging
import re
import time
import requests
import secrets

log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #

AIHORDE_BASE        = "https://aihorde.net/api/v2"
AIHORDE_APIKEY      = secrets.AIHORDE_API_KEY
AIHORDE_POLL_SEC    = 5    # seconds between status polls
AIHORDE_TIMEOUT_SEC = 300  # give up after this many seconds

OLLAMA_BASE   = "http://localhost:11434"
OLLAMA_MODEL  = "llama3"  # default; override with "llm_model" in persona JSON

# Shared generation params for AI Horde
AIHORDE_PARAMS = {
    "max_length":         350,
    "max_context_length": 2048,
    "temperature":        0.9,
    "top_p":              0.9,
    "top_k":              100,
    "rep_pen":            1.1,
}

OLLAMA_OPTIONS = {
    "temperature": 0.9,
    "top_p":       0.9,
    "top_k":       100,
}

# ------------------------------------------------------------------ #
# AI Horde provider
# ------------------------------------------------------------------ #

class AIHordeProvider:

    def __init__(self, api_key: str, models: list[str] = None):
        """
        api_key : your AI Horde API key
        models  : list of acceptable model names; empty = any worker picks it up
        """
        self.api_key = api_key
        self.models  = models or []  # empty = no preference, any worker
        self.headers = {
            "Content-Type": "application/json",
            "apikey": self.api_key,
        }

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"{system_prompt}\n\n{user_prompt}\n"

        payload = {"prompt": full_prompt, "params": AIHORDE_PARAMS}
        if self.models:
            payload["models"] = self.models

        # Submit job
        try:
            resp = requests.post(
                f"{AIHORDE_BASE}/generate/text/async",
                json=payload,
                headers=self.headers,
                timeout=60,
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            raise TimeoutError("AI Horde submit timed out - server busy, will retry next loop")

        body = resp.json()
        request_id = body.get("id")
        if not request_id:
            raise ValueError(f"AI Horde gave no request id. Response: {body}")

        log.info(f"AI Horde job submitted, id={request_id}, models={self.models or 'any'}")

        # Poll for completion
        elapsed = 0
        while elapsed < AIHORDE_TIMEOUT_SEC:
            time.sleep(AIHORDE_POLL_SEC)
            elapsed += AIHORDE_POLL_SEC

            try:
                status = requests.get(
                    f"{AIHORDE_BASE}/generate/text/status/{request_id}",
                    headers=self.headers,
                    timeout=60,
                )
                status.raise_for_status()
            except requests.exceptions.Timeout:
                log.warning("AI Horde status poll timed out, retrying...")
                continue

            data = status.json()

            if data.get("done"):
                generations = data.get("generations", [])
                if not generations:
                    raise ValueError("AI Horde: no workers available to handle this request")
                text = generations[0].get("text", "").strip()
                log.info(f"AI Horde job {request_id} complete ({len(text)} chars)")
                return text

            log.info(f"AI Horde waiting... est {data.get('wait_time', '?')}s, queue pos {data.get('queue_position', '?')}")

        raise TimeoutError(f"AI Horde job {request_id} timed out after {AIHORDE_TIMEOUT_SEC}s")


# ------------------------------------------------------------------ #
# Ollama provider
# ------------------------------------------------------------------ #

class OllamaProvider:

    def __init__(self, model: str = OLLAMA_MODEL):
        self.url   = f"{OLLAMA_BASE}/api/chat"
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model":   self.model,
            "stream":  False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "options": OLLAMA_OPTIONS,
        }
        resp = requests.post(self.url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()


# ------------------------------------------------------------------ #
# Router
# ------------------------------------------------------------------ #

def generate(persona: dict, system_prompt: str, user_prompt: str) -> str:
    provider_key = persona.get("llm", "").lower()
    username     = persona.get("username", "unknown")

    if provider_key == "koboldai":
        # Build model list from persona JSON
        # Accepts "llm_model" (str) or "llm_models" (list), or neither (any worker)
        if "llm_models" in persona:
            models = persona["llm_models"]
        elif "llm_model" in persona:
            models = [persona["llm_model"]]
        else:
            models = []  # AI Horde will use any available worker

        provider = AIHordeProvider(api_key=AIHORDE_APIKEY, models=models)

    elif provider_key == "ollama":
        model = persona.get("llm_model", OLLAMA_MODEL)
        provider = OllamaProvider(model=model)

    else:
        raise ValueError(
            f"[{username}] Unknown LLM provider '{provider_key}'. "
            f"Valid options: koboldai, ollama"
        )

    log.debug(f"[{username}] Calling provider: {provider_key}")
    text = provider.generate(system_prompt, user_prompt)

    if not text:
        raise ValueError(f"[{username}] Empty response from {provider_key}")

    return text