#!/usr/bin/env python3
"""
llm.py - every model call in jobscan goes through here.

Three jobs use a model, and each can point at any backend:

  rubric   score.py's prose-rubric scorer (the "old scorer"). Generative:
           batches of postings in, a JSON array of scores out.
  jev      jev.py's band / yes-no classifier. Wants a PROBABILITY per option,
           not text.
  tailor   tailor.py's viability analysis, letters and resume advice.
           Splits further into tailor_analyze / tailor_letter /
           tailor_resume if you want a cheaper model for one of them.

config.yaml carries named model profiles under `models:` and a role map
under `ai:` that says which profile each job uses:

    models:
      haiku:   {provider: anthropic, model: claude-haiku-4-5-20251001}
      qwen:    {provider: ollama,    model: "qwen3:8b"}
    ai:
      rubric: haiku
      jev:    {career: qwen, bridge: jevk5}
      tailor: haiku

A config with neither block behaves exactly as before: the roles fall back to
Anthropic for rubric/tailor and the `jev:` block's llama.cpp server for jev.

One-off overrides without editing the file, from any script or CI:

    JOBSCAN_AI="rubric=qwen,tailor=sonnet" python main.py

PROVIDERS. Everything is plain HTTP through `requests`, so no vendor SDK is
required. `kind` is the wire protocol; the rest are presets that fill in a
base_url and the environment variable the key lives in.

    anthropic            Anthropic Messages API
    openai               OpenAI
    openai_compatible    anything speaking /v1/chat/completions (set base_url)
    ollama lmstudio llamacpp vllm                         local servers
    openrouter groq together deepseek mistral gemini xai  hosted
    jevk5                the JevK5 calibrated classifier on llama.cpp
    litellm              pip install litellm; Bedrock, Vertex, Azure, Cohere...

HOW `jev` WORKS ON A MODEL THAT IS NOT JEVK5. A question becomes a lettered
multiple choice (same JSON prompt JevK5 was trained on) and the answer is read
one of three ways, set per profile with `readout:`:

    logprobs    softmax over the option letters' first-token log-probs.
                Calibrated-ish, one call, no generated text. Needs an endpoint
                that returns top_logprobs: llama.cpp, vLLM, OpenAI's
                non-reasoning models, Ollama (read through its native
                /api/chat, since its OpenAI endpoint has none), and others.
                Run `preflight.py --probe` to see what a given model does.
    verbalized  the model writes {"A": 0.7, "B": 0.3}. Works everywhere,
                including Anthropic. Probabilities are the model's own guess
                and are noticeably less calibrated.
    vote        N samples at temperature 1, counted. Slow and N times the
                cost; steadier than verbalized on small models.
    auto        (default) logprobs where the endpoint supports it, falling
                back to verbalized the first time it does not.
"""
import copy
import json
import math
import os
import re
import subprocess
import sys
import threading
import time
from urllib.parse import urlparse

ROLES = ("rubric", "jev", "tailor", "tailor_analyze", "tailor_letter",
         "tailor_resume")
TRACK_KEYS = ("career", "bridge")

# --------------------------------------------------------------------------
# Presets. `kind` is the protocol; everything else is a default the profile
# can override.
# --------------------------------------------------------------------------
_LOCAL = {"cost": "free"}
PRESETS = {
    "anthropic": {"kind": "anthropic", "base_url": "https://api.anthropic.com",
                  "api_key_env": "ANTHROPIC_API_KEY", "readout": "verbalized"},
    "openai": {"kind": "openai", "base_url": "https://api.openai.com/v1",
               "api_key_env": "OPENAI_API_KEY",
               # OpenAI's reasoning models reject max_tokens outright.
               "max_tokens_param": "max_completion_tokens"},
    "openai_compatible": {"kind": "openai"},
    "ollama": {"kind": "openai", "base_url": "http://127.0.0.1:11434/v1",
               "api_key": "ollama", **_LOCAL},
    "lmstudio": {"kind": "openai", "base_url": "http://127.0.0.1:1234/v1",
                 "api_key": "lm-studio", **_LOCAL},
    "llamacpp": {"kind": "openai", "base_url": "http://127.0.0.1:8080/v1",
                 "api_key": "none", **_LOCAL},
    "vllm": {"kind": "openai", "base_url": "http://127.0.0.1:8000/v1",
             "api_key": "none", **_LOCAL},
    "openrouter": {"kind": "openai", "base_url": "https://openrouter.ai/api/v1",
                   "api_key_env": "OPENROUTER_API_KEY"},
    "groq": {"kind": "openai", "base_url": "https://api.groq.com/openai/v1",
             "api_key_env": "GROQ_API_KEY"},
    "together": {"kind": "openai", "base_url": "https://api.together.xyz/v1",
                 "api_key_env": "TOGETHER_API_KEY"},
    "deepseek": {"kind": "openai", "base_url": "https://api.deepseek.com/v1",
                 "api_key_env": "DEEPSEEK_API_KEY"},
    "mistral": {"kind": "openai", "base_url": "https://api.mistral.ai/v1",
                "api_key_env": "MISTRAL_API_KEY"},
    "gemini": {"kind": "openai",
               "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
               "api_key_env": "GEMINI_API_KEY"},
    "xai": {"kind": "openai", "base_url": "https://api.x.ai/v1",
            "api_key_env": "XAI_API_KEY"},
    # Calibration values from JevK5 v0.3's jevk5_config.json, as jev.py
    # shipped them. A wrong value silently miscalibrates every probability.
    "jevk5": {"kind": "jevk5", "endpoint": "http://127.0.0.1:8090",
              "readout": "jevk5", "calibration_temperature": 1.22,
              "knockout_temperature": 0.93, "ctx": 8192, "ngl": 99,
              "startup_timeout": 180, **_LOCAL},
    "litellm": {"kind": "litellm"},
}

PROFILE_DEFAULTS = {
    "timeout": 600,            # the Anthropic SDK's default, kept
    "max_retries": 3,
    "concurrency": 1,          # parallel jev postings / rubric batches
    "readout": "auto",
    "top_logprobs": 20,        # OpenAI's ceiling; llama.cpp allows more
    "calibration_temperature": 1.0,
    "vote_samples": 5,
    "decide_max_tokens": 400,  # verbalized/vote budget; raise for reasoners
    "max_tokens_param": "max_tokens",
}

# Built-in models when config.yaml names none. Matches what the code shipped
# with before profiles existed, so an old config keeps working unchanged.
DEFAULT_RUBRIC_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TAILOR_MODEL = "claude-sonnet-4-6"


class LLMError(Exception):
    """
    Any failure from any backend, normalised.

    `permanent` means retrying the same call this run is pointless: a bad key,
    no credit, a model that does not exist, a local server that is not there.
    score.py stops asking after one of these instead of failing 24 batches.
    """
    def __init__(self, message, status=None, permanent=False):
        super().__init__(message)
        self.status = status
        self.status_code = status        # what score.py used to read
        self.permanent = permanent


# ==========================================================================
# Profiles and roles
# ==========================================================================
def _legacy_jev_profile(root):
    """The `jev:` block's runtime keys, as a profile. Old configs put them there."""
    j = (root or {}).get("jev") or {}
    p = {"provider": "jevk5", "name": "jevk5"}
    for k in ("endpoint", "ctx", "ngl", "startup_timeout"):
        if j.get(k) not in (None, ""):
            p[k] = j[k]
    if j.get("temperature") is not None:
        p["calibration_temperature"] = j["temperature"]
    if j.get("knockout_temperature") is not None:
        p["knockout_temperature"] = j["knockout_temperature"]
    if j.get("server_exe") or j.get("model_path"):
        p["server"] = {"exe": j.get("server_exe") or "",
                       "model_path": j.get("model_path") or ""}
    return p


def _default_jev(root):
    """
    jev's default model: the local JevK5 you already run. The `jevk5` profile
    under models: if there is one (that is where its llama-server path and
    GGUF live now); otherwise the same settings from the old `jev:` block.
    Never a hosted model, so a missing ai.jev line cannot start billing.
    """
    models = (root or {}).get("models") or {}
    if isinstance(models.get("jevk5"), dict):
        return {**models["jevk5"], "name": "jevk5"}
    return _legacy_jev_profile(root)


def _builtin(root, role):
    root = root or {}
    if role == "jev":
        return _default_jev(root)
    if role == "rubric":
        model = (root.get("scoring") or {}).get("model") or DEFAULT_RUBRIC_MODEL
        return {"provider": "anthropic", "model": model, "name": "anthropic"}
    model = (root.get("tailoring") or {}).get("model") or DEFAULT_TAILOR_MODEL
    return {"provider": "anthropic", "model": model, "name": "anthropic"}


def _env_overrides():
    """JOBSCAN_AI="rubric=qwen,jev.career=gpt" -> {"rubric": "qwen", ...}."""
    out = {}
    for part in (os.environ.get("JOBSCAN_AI") or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _lookup(root, ref, role):
    """A profile name or an inline dict -> a complete profile dict."""
    models = (root or {}).get("models") or {}
    if isinstance(ref, dict):
        raw, name = ref, ref.get("name") or ref.get("model") or "inline"
    elif ref in models:
        raw, name = models[ref] or {}, ref
    elif ref in ("builtin", "default", None, ""):
        raw = _builtin(root, role)
        name = raw.get("name")
    elif ref == "jevk5":
        raw, name = _default_jev(root), "jevk5"
    else:
        raise LLMError(f"ai.{role} names model profile {ref!r}, which is not "
                       f"defined under models: in config.yaml "
                       f"(defined: {', '.join(models) or 'none'})",
                       permanent=True)
    return finalize(raw, name)


def finalize(raw, name="inline"):
    """Preset + defaults + the profile's own keys. Validates the provider."""
    raw = dict(raw or {})
    provider = (raw.get("provider") or "").strip().lower()
    if provider not in PRESETS:
        raise LLMError(f"model profile {name!r}: unknown provider "
                       f"{provider or '(none)'!r}. Use one of: "
                       f"{', '.join(PRESETS)}", permanent=True)
    p = {**PROFILE_DEFAULTS, **PRESETS[provider], **raw}
    p["provider"] = provider
    if provider == "anthropic" and "base_url" not in raw and os.environ.get(
            "ANTHROPIC_BASE_URL"):
        # The SDK honoured this; keep honouring it.
        p["base_url"] = os.environ["ANTHROPIC_BASE_URL"]
    p["name"] = name
    if p["kind"] == "openai" and not p.get("base_url"):
        raise LLMError(f"model profile {name!r}: provider {provider} needs a "
                       f"base_url", permanent=True)
    # Free only by the preset (ollama, lmstudio, llamacpp, vllm, jevk5) or by
    # saying so. A localhost URL alone proves nothing: a LiteLLM or other
    # proxy on localhost can be billing a hosted API behind it, and treating
    # that as free would let it past scoring.max_per_run.
    p.setdefault("cost", "paid")
    return p


def resolve(root, role, track=None):
    """
    The profile for one job, and optionally one track.

    `ai.<role>` may be a profile name, an inline profile, or a mapping of
    {career: ..., bridge: ...}. tailor_analyze / tailor_letter / tailor_resume
    fall back to `ai.tailor`. JOBSCAN_AI overrides any of it, with
    "jev.bridge=..." addressing one track.
    """
    if role not in ROLES:
        raise ValueError(f"unknown AI role {role!r}")
    ai = dict((root or {}).get("ai") or {})
    env = _env_overrides()

    chain = [role] + (["tailor"] if role.startswith("tailor_") else [])
    for r in chain:
        if track and f"{r}.{track}" in env:
            return _lookup(root, env[f"{r}.{track}"], role)
        if r in env:
            return _lookup(root, env[r], role)
        ref = ai.get(r)
        if isinstance(ref, dict) and "provider" not in ref and (
                set(ref) & set(TRACK_KEYS) or "default" in ref):
            ref = ref.get(track) if track in ref else ref.get("default")
        if ref:
            return _lookup(root, ref, role)
    return _lookup(root, None, role)


def label(p):
    """Short, stable identifier stored with each score: name:model."""
    model = p.get("model") or ("jevk5" if p["kind"] == "jevk5" else "")
    name = p.get("name") or p["provider"]
    return name if (not model or model == name) else f"{name}:{model}"


def is_free(p):
    return str(p.get("cost", "paid")).lower() == "free"


def _is_local(p):
    url = p.get("base_url") or p.get("endpoint") or ""
    host = (urlparse(url).hostname or "").lower()
    return (host in ("localhost", "127.0.0.1", "::1", "0.0.0.0")
            or host.endswith(".local") or host.startswith("192.168.")
            or host.startswith("10."))


def check(p):
    """A problem string if this profile cannot possibly work, else None."""
    if p["kind"] == "litellm":
        try:
            import litellm  # noqa: F401
        except ImportError:
            return "pip install litellm (needed for provider: litellm)"
    if p["kind"] == "jevk5":
        try:
            import jevk5  # noqa: F401
        except ImportError:
            return ("jevk5 package not installed: pip install --no-deps "
                    "\"jevk5 @ git+https://github.com/allebee/jevk5@v0.3.0\"")
        return None
    if p["kind"] == "anthropic" and not _api_key(p) and os.environ.get(
            "ANTHROPIC_AUTH_TOKEN"):
        pass
    elif p["kind"] in ("anthropic", "openai") and not _api_key(p):
        env = p.get("api_key_env")
        if env:
            return f"no API key: set {env}"
        if not _is_local(p):
            return "no api_key or api_key_env configured"
    if p["kind"] != "jevk5" and not p.get("model"):
        return f"model profile {p['name']!r} has no model"
    return None


def _api_key(p):
    if p.get("api_key"):
        return str(p["api_key"])
    env = p.get("api_key_env")
    return os.environ.get(env, "") if env else ""


# ==========================================================================
# HTTP with retries
# ==========================================================================
_PERMANENT_400 = ("credit balance", "billing", "quota", "not_found",
                  "not found", "invalid api key", "invalid_api_key",
                  "permission", "does not exist", "unknown model",
                  "model_not_found", "insufficient")


def _post(p, url, payload, headers):
    import requests
    tries = max(1, int(p.get("max_retries", 3)) + 1)
    last = None
    for attempt in range(tries):
        try:
            r = requests.post(url, json=payload, headers=headers,
                              timeout=float(p.get("timeout", 600)))
        except requests.exceptions.ConnectionError as e:
            # A local server that refuses connections is gone for this run.
            # A hosted API that drops one is a network blip: the next batch
            # gets its own retries, as it did under the Anthropic SDK.
            last = LLMError(f"cannot reach {url}: {e.__class__.__name__}",
                            permanent=_is_local(p))
        except requests.exceptions.Timeout:
            last = LLMError(f"timed out after {p.get('timeout')}s: {url}")
        else:
            if r.status_code < 400:
                try:
                    return r.json()
                except ValueError:
                    raise LLMError(f"non-JSON response from {url}",
                                   status=r.status_code)
            text = r.text[:600]
            low = text.lower()
            permanent = (r.status_code in (401, 403, 404)
                         or (r.status_code == 400
                             and any(k in low for k in _PERMANENT_400)))
            last = LLMError(f"HTTP {r.status_code} from {p['name']}: {text}",
                            status=r.status_code, permanent=permanent)
            if r.status_code not in (408, 409, 429, 500, 502, 503, 504, 529):
                raise last
            wait = r.headers.get("retry-after")
            if attempt + 1 < tries:
                try:
                    time.sleep(min(60.0, float(wait)) if wait
                               else 2.0 * (2 ** attempt))
                except ValueError:
                    time.sleep(2.0 * (2 ** attempt))
                continue
        if attempt + 1 < tries:
            time.sleep(2.0 * (2 ** attempt))
    raise last


# ==========================================================================
# Generation
# ==========================================================================
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


def _system_text(system):
    if isinstance(system, str):
        return system
    return "\n\n".join(b["text"] for b in system or [])


def _openai_payload(p, messages, max_tokens, extra=None):
    body = {"model": p["model"], "messages": messages,
            p.get("max_tokens_param", "max_tokens"): int(max_tokens)}
    if p.get("temperature") is not None:
        body["temperature"] = p["temperature"]
    body.update(p.get("extra_body") or {})
    body.update(extra or {})
    return body


def _openai_headers(p):
    h = {"Content-Type": "application/json"}
    key = _api_key(p)
    if key:
        h["Authorization"] = f"Bearer {key}"
    h.update(p.get("headers") or {})
    return h


def _chat(p, messages, max_tokens, extra=None):
    """One OpenAI-shaped chat call; returns the raw response dict."""
    if p["kind"] == "litellm":
        import litellm
        kw = {k: v for k, v in _openai_payload(p, messages, max_tokens, extra)
              .items() if k not in ("model", "messages")}
        if p.get("max_tokens_param") != "max_tokens":
            kw["max_tokens"] = kw.pop(p["max_tokens_param"], max_tokens)
        for k in ("api_key", "base_url"):
            if p.get(k):
                kw["api_base" if k == "base_url" else k] = p[k]
        try:
            resp = litellm.completion(model=p["model"], messages=messages,
                                      timeout=p.get("timeout"),
                                      num_retries=p.get("max_retries", 3), **kw)
        except Exception as e:
            status = getattr(e, "status_code", None)
            raise LLMError(f"{type(e).__name__}: {e}", status=status,
                           permanent=status in (400, 401, 403, 404))
        return resp.model_dump() if hasattr(resp, "model_dump") else dict(resp)
    url = p["base_url"].rstrip("/") + "/chat/completions"
    return _post(p, url, _openai_payload(p, messages, max_tokens, extra),
                 _openai_headers(p))


def complete(p, system, user, max_tokens=2000):
    """
    One generative call. Returns the text.

    `system` is a string or a list of {"text": ..., "cache": bool} blocks.
    Blocks marked cache get Anthropic prompt caching; other providers just
    see the joined text (OpenAI and DeepSeek cache identical prefixes
    automatically; local servers reuse the KV cache on their own).
    """
    problem = check(p)
    if problem:
        raise LLMError(f"{p['name']}: {problem}", permanent=True)
    if p["kind"] == "jevk5":
        raise LLMError(f"{p['name']}: JevK5 is a classifier and cannot write "
                       f"text; point this role at a generative model",
                       permanent=True)

    if p["kind"] == "anthropic":
        blocks = ([{"type": "text", "text": system}]
                  if isinstance(system, str) else
                  [dict({"type": "text", "text": b["text"]},
                        **({"cache_control": {"type": "ephemeral"}}
                           if b.get("cache") else {}))
                   for b in (system or [])])
        body = {"model": p["model"], "max_tokens": int(max_tokens),
                "messages": [{"role": "user", "content": user}]}
        if any(b["text"] for b in blocks):
            body["system"] = blocks
        if p.get("temperature") is not None:
            body["temperature"] = p["temperature"]
        body.update(p.get("extra_body") or {})
        headers = {"anthropic-version": p.get("api_version", "2023-06-01"),
                   "content-type": "application/json"}
        if _api_key(p):
            headers["x-api-key"] = _api_key(p)
        elif os.environ.get("ANTHROPIC_AUTH_TOKEN"):
            # The SDK's bearer-token alternative, e.g. behind a gateway.
            headers["Authorization"] = (
                f"Bearer {os.environ['ANTHROPIC_AUTH_TOKEN']}")
        headers.update(p.get("headers") or {})
        out = _post(p, p["base_url"].rstrip("/") + "/v1/messages", body,
                    headers)
        return "".join(b.get("text", "") for b in out.get("content") or []
                       if b.get("type") == "text")

    sys_text = _system_text(system)
    messages = ([{"role": "system", "content": sys_text}] if sys_text else [])
    messages.append({"role": "user", "content": user})
    out = _chat(p, messages, max_tokens)
    try:
        text = out["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        raise LLMError(f"{p['name']}: unexpected response shape: "
                       f"{str(out)[:300]}")
    if isinstance(text, list):          # some servers return content parts
        text = "".join(c.get("text", "") for c in text if isinstance(c, dict))
    return _THINK.sub("", text).strip()


# ==========================================================================
# Decisions: a probability per option, for jev.py
# ==========================================================================
LETTERS = "ABCDEFGHIJKLMNOP"
# Same instruction and JSON layout JevK5 was trained on, so a question reads
# identically to every backend and --compare measures the model, not the
# prompt.
DECIDE_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one "
    "listed option. Respond with only its uppercase letter, with no "
    "explanation or reasoning.")
VERBAL_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. For EVERY listed "
    "option, estimate the probability that it is the correct choice. Respond "
    "with ONLY a JSON object mapping each option letter to a probability, the "
    "values summing to 1, for example {\"A\": 0.1, \"B\": 0.7, \"C\": 0.2}. "
    "No other text.")


def _options(question):
    """(id, text) per option, matching jevk5.prompt.decision_options."""
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.")
                 for k in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


def _user(state, question, texts):
    return json.dumps({
        "evidence": state,
        "criterion": question["instructions"],
        "options": [{"letter": LETTERS[i], "description": d}
                    for i, d in enumerate(texts)]}, ensure_ascii=False)


def _answer(question, probs):
    out = {"type": question["type"], "confidence": max(probs.values())}
    if question["type"] == "noul":
        out["noul"] = probs["true"]
    elif question["type"] == "choice":
        out.update(choice=max(probs, key=probs.get), probabilities=probs)
    else:
        out.update(score=sum(int(k) * v for k, v in probs.items()),
                   probabilities=probs)
    return out


def _sharpen(ps, temperature):
    """p^(1/T), renormalised. T > 1 flattens an overconfident model."""
    t = float(temperature or 1.0)
    ps = [max(q, 1e-9) ** (1.0 / t) for q in ps]
    s = sum(ps)
    return [q / s for q in ps]


class Decider:
    """
    .decide(state, question) -> the same dict shape JevK5GGUF returns, so
    jev.compose() cannot tell which backend produced it.
    """

    def __init__(self, p):
        self.p = p
        self.readout = p.get("readout", "auto")
        if self.readout == "auto" and p["kind"] == "anthropic":
            self.readout = "verbalized"
        self.missing = 0
        self.switched = None          # set when auto fell back to verbalized
        self._lock = threading.Lock()

    def decide(self, state, question):
        opts = _options(question)
        if len(opts) > len(LETTERS):
            raise LLMError(f"{len(opts)} options; at most {len(LETTERS)} "
                           f"are supported outside JevK5", permanent=True)
        texts = [t for _, t in opts]
        mode = self.readout
        if mode in ("auto", "logprobs"):
            try:
                ps = self._logprobs(state, question, texts)
            except _NoLogprobs as e:
                if mode == "logprobs":
                    raise LLMError(f"{self.p['name']}: {e}", permanent=True)
                with self._lock:
                    if self.readout == "auto":
                        self.readout = "verbalized"
                        self.switched = str(e)
                        print(f"  ! {self.p['name']}: no usable logprobs "
                              f"({e}); using verbalized probabilities")
                ps = self._verbalized(state, question, texts)
        elif mode == "verbalized":
            ps = self._verbalized(state, question, texts)
        elif mode == "vote":
            ps = self._vote(state, question, texts)
        else:
            raise LLMError(f"{self.p['name']}: unknown readout {mode!r}",
                           permanent=True)
        ps = _sharpen(ps, self.p.get("calibration_temperature", 1.0))
        return _answer(question, {k: v for (k, _), v in zip(opts, ps)})

    # ---- readouts ------------------------------------------------------
    def _messages(self, system, state, question, texts):
        return [{"role": "system", "content": system},
                {"role": "user", "content": _user(state, question, texts)}]

    def _ollama_native(self, state, question, texts):
        """
        Ollama's OpenAI endpoint does not return logprobs; its native
        /api/chat does (top 20), and takes think:false directly.
        Returns (top list, first token).
        """
        root = re.sub(r"/v1/?$", "", self.p["base_url"].rstrip("/"))
        body = {"model": self.p["model"], "stream": False,
                "messages": self._messages(DECIDE_SYSTEM, state, question,
                                           texts),
                "logprobs": True,
                "top_logprobs": min(20, int(self.p.get("top_logprobs", 20))),
                "options": {"num_predict": 1, "temperature": 0}}
        if self.p.get("think", False) is not None:
            body["think"] = self.p.get("think", False)
        headers = {"Content-Type": "application/json",
                   **(self.p.get("headers") or {})}
        try:
            out = _post(self.p, root + "/api/chat", body, headers)
        except LLMError as e:
            # A model with no thinking mode may refuse the think field.
            if e.status == 400 and "think" in str(e).lower() and "think" in body:
                body.pop("think")
                out = _post(self.p, root + "/api/chat", body, headers)
            else:
                raise
        lp = out.get("logprobs") or []
        if not lp:
            raise _NoLogprobs("this Ollama returned no logprobs (needs a "
                              "version with logprobs on /api/chat)")
        return lp[0].get("top_logprobs") or [], lp[0]

    def _logprobs(self, state, question, texts):
        if self.p["kind"] == "anthropic":
            raise _NoLogprobs("Anthropic does not return logprobs")
        if self.p["provider"] == "ollama":
            top, first = self._ollama_native(state, question, texts)
            return self._read_letters(top, first, texts)
        extra = {"logprobs": True,
                 "top_logprobs": int(self.p.get("top_logprobs", 20)),
                 "temperature": 0}
        try:
            out = _chat(self.p, self._messages(DECIDE_SYSTEM, state, question,
                                               texts), 1, extra)
        except LLMError as e:
            if e.status == 400 and "logprob" in str(e).lower():
                raise _NoLogprobs("endpoint rejected the logprobs parameter")
            raise
        try:
            lp = out["choices"][0].get("logprobs") or {}
            first = (lp.get("content") or [])[0]
            top = first.get("top_logprobs") or []
        except (KeyError, IndexError, TypeError, AttributeError):
            raise _NoLogprobs("response carried no top_logprobs")
        if not top:
            raise _NoLogprobs("response carried no top_logprobs")
        return self._read_letters(top, first, texts)

    def _read_letters(self, top, first, texts):
        """Softmax over the option letters found among the top tokens."""
        want = LETTERS[:len(texts)]
        seen = {}
        for e in top:
            tok = str(e.get("token", "")).strip().strip("*(\"'`.:)")
            if tok in want:
                # " A" and "A" are different tokens for the same answer.
                lp_ = float(e["logprob"])
                seen[tok] = (lp_ if tok not in seen else
                             max(seen[tok], lp_) + math.log1p(
                                 math.exp(-abs(seen[tok] - lp_))))
        if not seen:
            first_tok = str(first.get("token", ""))
            raise _NoLogprobs(
                f"no option letter in the top {len(top)} first tokens "
                f"(model began with {first_tok!r}; if that is a thinking tag, "
                f"turn thinking off via extra_body)")
        floor = min(float(e["logprob"]) for e in top) - 2.0
        if len(seen) < len(want):
            self.missing += 1
        logs = [seen.get(L, floor) for L in want]
        m = max(logs)
        w = [math.exp(z - m) for z in logs]
        s = sum(w)
        return [x / s for x in w]

    def _verbalized(self, state, question, texts):
        text = self._gen(VERBAL_SYSTEM, state, question, texts)
        want = LETTERS[:len(texts)]
        probs = None
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                raw = json.loads(m.group(0))
                probs = [max(0.0, float(raw.get(L, raw.get(L.lower(), 0.0))))
                         for L in want]
            except (ValueError, TypeError, AttributeError):
                probs = None
        if not probs or sum(probs) <= 0:
            # A bare letter is still an answer; treat it as confident.
            hit = re.search(r"\b([A-P])\b", text)
            if not hit or hit.group(1) not in want:
                raise LLMError(f"{self.p['name']}: could not read an answer "
                               f"from {text[:120]!r}")
            probs = [0.9 if L == hit.group(1) else 0.1 / max(1, len(want) - 1)
                     for L in want]
        s = sum(probs)
        return [q / s for q in probs]

    def _vote(self, state, question, texts):
        want = LETTERS[:len(texts)]
        n = max(1, int(self.p.get("vote_samples", 5)))
        counts = {L: 0.0 for L in want}
        for _ in range(n):
            text = self._gen(DECIDE_SYSTEM, state, question, texts,
                             temperature=1.0)
            hit = re.search(r"\b([A-P])\b", text)
            if hit and hit.group(1) in counts:
                counts[hit.group(1)] += 1
        # Half a vote each so an unchosen option is unlikely, not impossible.
        ps = [counts[L] + 0.5 for L in want]
        s = sum(ps)
        return [q / s for q in ps]

    def _gen(self, system, state, question, texts, temperature=None):
        p = self.p if temperature is None else {**self.p,
                                                "temperature": temperature}
        return complete(p, system, _user(state, question, texts),
                        int(self.p.get("decide_max_tokens", 400)))


class _NoLogprobs(Exception):
    pass


def decider(p):
    """A .decide() object for this profile. JevK5 keeps its own package."""
    problem = check(p)
    if problem:
        raise LLMError(f"{p['name']}: {problem}", permanent=True)
    if p["kind"] == "jevk5":
        from jevk5 import JevK5GGUF
        kw = {"temperature": p.get("calibration_temperature")}
        if p.get("knockout_temperature") is not None:
            kw["knockout_temperature"] = p["knockout_temperature"]
        if p.get("config"):
            kw["config"] = p["config"]
        return JevK5GGUF(p["endpoint"], **kw)
    return Decider(p)


# ==========================================================================
# Local server lifecycle
# ==========================================================================
def _health_url(p):
    if p.get("health_url"):
        return p["health_url"]
    if p["kind"] == "jevk5":
        return p["endpoint"].rstrip("/") + "/health"
    if p.get("base_url"):
        # /models answers on llama.cpp, Ollama, LM Studio and vLLM alike.
        return p["base_url"].rstrip("/") + "/models"
    return None


def _alive(url, timeout=2.0):
    import urllib.error
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def ensure_ready(p, quiet=False):
    """
    (ok, started_by_us). Only local profiles are checked.

    A server already listening is reused and never shut down; you may be
    using it elsewhere. If nothing answers and the profile has a `server:`
    block, one is started and left running for the next call.
    """
    say = (lambda *a: None) if quiet else print
    if p["kind"] not in ("jevk5", "openai") or not _is_local(p):
        return True, False
    url = _health_url(p)
    if not url or _alive(url):
        if url and not quiet:
            say(f"  - {p['name']}: server already listening, reusing it")
        return True, False

    srv = p.get("server") or {}
    cmd = srv.get("command")
    if not cmd and srv.get("exe") and srv.get("model_path"):
        base = p.get("endpoint") or p.get("base_url") or ""
        port = urlparse(base).port or 8080
        cmd = [srv["exe"], "-m", srv["model_path"],
               "-c", str(srv.get("ctx", p.get("ctx", 8192))),
               "-ngl", str(srv.get("ngl", p.get("ngl", 99))),
               "--port", str(port)] + list(srv.get("args") or [])
    if not cmd:
        say(f"  ! {p['name']}: nothing answering at {url} and no server: "
            f"block configured to start one")
        return False, False
    if isinstance(cmd, str):
        import shlex
        cmd = shlex.split(cmd, posix=sys.platform != "win32")

    say(f"  - {p['name']}: starting {os.path.basename(str(cmd[0]))}")
    try:
        kw = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
        if sys.platform == "win32":
            kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        subprocess.Popen(cmd, **kw)
    except (OSError, ValueError) as e:
        say(f"  ! {p['name']}: could not start server: "
            f"{type(e).__name__}: {e}")
        return False, False

    limit = float(srv.get("startup_timeout", p.get("startup_timeout", 180)))
    t0 = time.time()
    while time.time() - t0 < limit:
        if _alive(url):
            say(f"  - {p['name']}: ready after {time.time() - t0:.0f}s")
            return True, True
        time.sleep(2)
    say(f"  ! {p['name']}: server did not answer within {limit:.0f}s")
    return False, False


def describe(p):
    """One line for preflight and run logs."""
    where = p.get("base_url") or p.get("endpoint") or ""
    bits = [p["provider"]]
    if p.get("model"):
        bits.append(p["model"])
    if where and p["kind"] != "anthropic":
        bits.append(where)
    bits.append("free" if is_free(p) else "paid")
    return f"{p['name']} ({', '.join(bits)})"


if __name__ == "__main__":
    # python llm.py PROFILE "prompt"   - quick manual check of one profile
    import yaml
    import env
    env.load()
    root = yaml.safe_load(open("config.yaml", encoding="utf-8")) or {}
    if len(sys.argv) < 2:
        for r in ROLES:
            try:
                print(f"{r:<15} {describe(resolve(root, r))}")
            except LLMError as e:
                print(f"{r:<15} ! {e}")
        sys.exit(0)
    prof = _lookup(root, sys.argv[1], "rubric")
    print(complete(prof, "Answer briefly.",
                   sys.argv[2] if len(sys.argv) > 2 else "Say hello."))
