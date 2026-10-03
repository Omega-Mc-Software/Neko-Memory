# Psychology Debate Bench

Three local LLMs on Legion argue psychology nonstop; every round runs through the Neko-Memory
verification pipeline. Built by Neko Omega, Oct 2 2026, at Jason Omega's request.

## Players
- **Debater A** — glm-4.7-flash (19GB)
- **Debater B** — qwen2.5:3b (smaller opponent)
- **Devil's Advocate C** — gemma2:2b — attacks any consensus forming between A and B

## Rules baked into every system prompt
1. Do not agree to be agreeable; consensus is an attack target.
2. Never fabricate facts, studies, statistics, or citations; uncertainty is stated plainly.
3. LOOKUP: <query> lines get answered by Neko-Memory `answer_verified` (SUPPORTED / NOT_IN_MEMORY / CONTRADICTED) and injected into the next turn.
4. Every round's claims are run through `memory_store.verify` against memory.

## Memory testing
- Round transcripts are seeded into recall; every REVISIT_EVERY (6th) round pulls an old topic
  out of Neko-Memory and makes the debaters re-examine it — a live recall test.
- Runs nonstop under systemd (`psych-debate.service`) on Legion; one round every 20 minutes.
- Transcripts: `transcripts/YYYY-MM-DD.md` (synced here by Neko daily).
