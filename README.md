# JobScanner

JobScanner is a scheduled Python and SQLite data pipeline that ingests job postings from seven sources (the Adzuna, USAJOBS, and NYC Open Data APIs, plus Greenhouse, Lever, Ashby, and Workday job boards), maps each source's fields into a single canonical schema, and reconciles records across sources. It runs twice each weekday via GitHub Actions and has ingested about 8,400 postings since August 2026.

- **Ingestion:** per-source fetchers with retry and rate-limit backoff, plus an enrichment step that fills in missing descriptions.
- **Reconciliation:** content fingerprinting (employer, title, location) catches the same posting arriving under different source IDs, with keeper rules that preserve the most complete record and any user status.
- **Storage:** SQLite with in-place schema migrations, plus an export to Google Sheets.
- **Scoring:** an LLM relevance scorer, including a fine-tuned local model (Qwen 4B with a distilled LoRA) read as probabilities from next-token logits, which ranks postings into a daily email digest.
