"""
Load key=value pairs from a .env file into os.environ.

No dependency required. Uses python-dotenv if it happens to be installed,
otherwise falls back to a small parser that handles the same basics.

Two rules that matter:

  1. Real environment variables ALWAYS win over .env. GitHub Actions injects
     secrets as real env vars and there is no .env file in CI, so the same
     code path works locally and in the cron job with no branching.

  2. Surrounding quotes are stripped. On Windows `set VAR="value"` stores the
     quote marks as part of the value, and people carry that habit into .env
     files. A quoted USAJOBS_EMAIL produces a User-Agent header that does not
     match the registered address, which USAJOBS answers with a 401.

Import and call load() at the top of any entry point:

    import env; env.load()
"""

import os
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name(".env")


def _strip_quotes(v):
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
        return v[1:-1]
    return v


def _parse(path):
    pairs = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if key:
            pairs[key] = _strip_quotes(val)
    return pairs


def load(path=None, verbose=False):
    """
    Returns the list of key names that were loaded. A missing file is fine —
    that's the normal case in CI.
    """
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        alt = Path.cwd() / ".env"
        if alt.exists():
            p = alt
        else:
            return []

    try:
        from dotenv import dotenv_values
        pairs = {k: _strip_quotes(v) for k, v in dotenv_values(p).items()
                 if v is not None}
    except ImportError:
        pairs = _parse(p)
    except Exception as e:
        print(f"  ! could not read {p}: {type(e).__name__}: {e}")
        return []

    loaded = []
    for k, v in pairs.items():
        # Real env vars win. Do not clobber what CI injected.
        if os.environ.get(k):
            continue
        os.environ[k] = v
        loaded.append(k)

    if verbose and loaded:
        print(f"  loaded {len(loaded)} vars from {p.name}")
    return loaded


# ---------------------------------------------------------------------------
# python env.py  -> show what would be loaded, values masked
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = DEFAULT_PATH if DEFAULT_PATH.exists() else Path.cwd() / ".env"
    if not p.exists():
        raise SystemExit(
            f"No .env found. Looked in:\n  {DEFAULT_PATH}\n  {Path.cwd() / '.env'}")

    print(f"Reading {p}\n")
    raw = p.read_text(encoding="utf-8-sig")
    pairs = _parse(p)
    if not pairs:
        raise SystemExit("File parsed but contains no key=value lines.")

    EXPECTED = ["ADZUNA_APP_ID", "ADZUNA_APP_KEY", "USAJOBS_EMAIL",
                "USAJOBS_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
                "OPENROUTER_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY",
                "DEEPSEEK_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY",
                "XAI_API_KEY", "JOBSCAN_AI", "SMTP_USER",
                "SMTP_PASSWORD", "DIGEST_TO", "SOCRATA_APP_TOKEN"]

    for k, v in pairs.items():
        if not v:
            state = "EMPTY — not filled in"
        elif len(v) > 8:
            state = f"ok, {len(v)} chars, ends ...{v[-4:]}"
        else:
            state = f"ok, {len(v)} chars"
        mark = " " if k in EXPECTED else "?"
        print(f" {mark} {k:<24} {state}")

    unknown = [k for k in pairs if k not in EXPECTED]
    if unknown:
        print(f"\n  ? = name not used by jobscan; check spelling: "
              f"{', '.join(unknown)}")

    missing = [k for k in EXPECTED[:4] if not pairs.get(k)]
    if missing:
        print(f"\n  MISSING (required): {', '.join(missing)}")

    quoted = [l.split("=")[0].strip() for l in raw.splitlines()
              if "=" in l and not l.strip().startswith("#")
              and l.split("=", 1)[1].strip()[:1] in "\"'"]
    if quoted:
        print(f"\n  note: stripped quotes from {', '.join(quoted)} — handled, "
              f"but cleaner to remove them")
