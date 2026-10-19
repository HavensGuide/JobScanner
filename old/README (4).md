# jobscan

Pulls job listings from free APIs, remembers what it's already shown you, emails a digest of only the new ones. Plus a cover letter drafter.

Runs on GitHub Actions cron. Cost: $0 for the scanner, a few cents per letter.

---

## Setup (about an hour)

### 1. Get the free API keys

| Service | Where | Time | Notes |
|---|---|---|---|
| **Adzuna** | developer.adzuna.com | 5 min | Instant. Gives `app_id` + `app_key`. ~1,000 calls/month free. |
| **USAJOBS** | developer.usajobs.gov/apirequest | 5 min | Self-serve. Your `User-Agent` is the email you registered. |
| **Socrata** (NYC) | data.cityofnewyork.us — sign up, then Developer Settings | 5 min | Optional. Works without a token at low volume; the token just raises your rate limit. |
| **Gmail App Password** | myaccount.google.com/apppasswords | 5 min | Requires 2FA enabled. Do **not** use your actual password. |
| Company ATS boards | — | — | No key at all. Just need each company's board slug. |

### 2. Find your company board slugs

Look at any company's job board URL:

- `boards.greenhouse.io/**datadog**` → Greenhouse token `datadog`
- `jobs.lever.co/**spotify**` → Lever token `spotify`
- `jobs.ashbyhq.com/**ramp**` → Ashby token `ramp`

Test one before adding it:

```bash
curl -s "https://boards-api.greenhouse.io/v1/boards/datadog/jobs" | head -c 300
```

Put 15–30 in `config.yaml`. This is the highest-value part of the whole setup — these are the freshest, least-crowded listings you'll see.

### 3. Run it locally first

```bash
pip install -r requirements.txt

export ADZUNA_APP_ID=... ADZUNA_APP_KEY=...
export USAJOBS_EMAIL=you@example.com USAJOBS_API_KEY=...

python main.py --dry-run     # fetches, filters, prints. Records nothing.
```

Tune `title_keywords` and `title_blocklist` in `config.yaml` until the dry-run output is mostly jobs you'd actually click. **Do this before the first real run** — otherwise the first run marks a few hundred jobs as "seen" and you never see them again.

When it looks right:

```bash
python main.py               # sends email, writes seen.db
```

### 4. Put it on GitHub Actions

1. Push to a **private** repo.
2. Settings → Secrets and variables → Actions → add each key as a secret. Names must match the `env:` block in `.github/workflows/jobscan.yml`.
3. Actions tab → jobscan → "Run workflow" to test immediately.

It runs 7am and 5pm weekdays. The workflow commits `seen.db` back to the repo after each run — that's how it remembers across runs with no database server.

### 5. Optional: Google Sheets mirror

Uncomment `gspread` and `google-auth` in `requirements.txt`. Then: Google Cloud Console → create a service account → download the JSON key → **share your sheet with the service account's email address** (this is the step everyone forgets). Set `GOOGLE_SHEET_ID` and paste the whole JSON blob as `GOOGLE_SERVICE_ACCOUNT_JSON`.

---

## Cover letters

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...

pbpaste > posting.txt          # or just paste into a file
python tailor.py posting.txt
```

Two API calls: one extracts duties, requirements, keywords, and **honest gaps**; the second writes the letter. Output goes to `out/`.

The gap list prints to your terminal. Read it. If it says "posting wants 3 years, you have 2," that's real information about whether to apply and what to address.

The constraint that matters is in `tailor.py`: the model may only use facts from `resume.yaml`. Keep that file rich and accurate. If you add a skill there that you don't have, the model will use it — the honesty of the output is entirely downstream of the honesty of that file.

Cost: roughly 3–6 cents per letter on Sonnet. Sixty applications a month lands around $2–3.

---

## Maintenance

Things that break, roughly in order of likelihood:

- **A company board 404s.** They switched ATS or renamed. The script logs it and continues; fix the slug when you notice.
- **Adzuna quota.** One call per query per run. Three queries × 2 runs/day × 22 weekdays ≈ 132/month. Well under 1,000, but adding queries multiplies fast.
- **Cron drift.** GitHub cron is UTC and ignores DST, so your 7am becomes 6am in winter.
- **Nothing arrives.** Check the Actions log first. Usually an expired Gmail app password.

## What this deliberately doesn't do

No LinkedIn or Indeed scraping — no usable API, and GitHub Actions runs on cloud IPs they block outright. No auto-apply. No auto-send of anything to a human.
