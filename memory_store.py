#!/usr/bin/env python3
"""Long-term memory layer for Legion. v3 — temporal memory + passive thinking.

v1 was cosine-only: whatever text was closest won, and story time was ignored.
v2 adds:
  - structured facts: entity, attribute, event text, story_time (chapter/day)
  - entity-indexed recall (find WHO the question is about, pull that entity's
    whole history, oldest first)
  - an LLM synthesis pass that reads the entity timeline and answers across
    time gaps ("broke a leg in ch 1" -> still known in ch 20)

v3 adds PASSIVE THINKING (no prompt needed), modeled on real neuroscience:
  - Systems consolidation / hippocampal replay: during rest, the hippocampus
    re-activates recent memory traces and folds them into cortical structure.
    -> `think()` pulls recent raw memories that have not been consolidated yet
       and extracts structured facts from them (replay -> structure).
  - Default Mode Network association: spontaneous thought links loosely
    related traces. -> pairs of memories above a cosine threshold get a
    `links` edge, so recall can travel associations, not just similarity.
  - Synaptic plasticity: traces that are retrieved strengthen, unused ones
    fade in relative importance. -> an importance score updated on recall
    and a recency-weighted retrieval ranking.
  - Consolidation summary: like sleep tagging the day, a short session
    summary is written per replay batch.

Fully local: Ollama glm-4.7-flash (chat) + nomic-embed-text (embeddings).
SQLite. Nothing leaves the machine.
"""
import json, sqlite3, subprocess, sys, math, time, uuid, os, re, threading

DB = os.path.expanduser("~/memory/memory.db")
CHAT_MODEL = "glm-4.7-flash"
EMBED_MODEL = "nomic-embed-text"

REPLAY_WINDOW = 25          # max raw memories per replay batch
ASSOC_THRESHOLD = 0.62      # cosine above which two traces get linked
IMPORTANCE_FRESH = 1.0      # importance of a new trace
RETRIEVE_BOOST = 0.15       # importance gained each time a trace is recalled
DECAY_HALF_LIFE_DAYS = 21.0 # importance halves every 3 weeks of disuse

def _ollama(endpoint, payload, timeout=300):
    import urllib.request
    req = urllib.request.Request("http://127.0.0.1:11434/api/" + endpoint,
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def _chat(messages, timeout=300):
    return _ollama("chat", {"model": CHAT_MODEL, "stream": False,
                            "messages": messages}, timeout=timeout)["message"]["content"]

def embed(text):
    return _ollama("embeddings", {"model": EMBED_MODEL, "prompt": text})["embedding"]

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    return dot / ((math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b))) or 1e-9)

def init():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY, text TEXT, embedding BLOB, created REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS facts (
        id TEXT PRIMARY KEY, entity TEXT, attribute TEXT, event TEXT,
        story_time TEXT, story_seq REAL, created REAL)""")
    # v3
    con.execute("""CREATE TABLE IF NOT EXISTS links (
        a TEXT, b TEXT, strength REAL, created REAL,
        PRIMARY KEY (a, b))""")
    cols = [r[1] for r in con.execute("PRAGMA table_info(memories)").fetchall()]
    if "importance" not in cols:
        con.execute("ALTER TABLE memories ADD COLUMN importance REAL DEFAULT 1.0")
    if "consolidated" not in cols:
        con.execute("ALTER TABLE memories ADD COLUMN consolidated INTEGER DEFAULT 0")
    if "last_accessed" not in cols:
        con.execute("ALTER TABLE memories ADD COLUMN last_accessed REAL DEFAULT 0")
    con.execute("""CREATE TABLE IF NOT EXISTS summaries (
        id TEXT PRIMARY KEY, text TEXT, created REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS persona (text TEXT)""")
    con.commit(); con.close()

# ---------- v1 surface (kept) ----------
def store(text):
    vec = embed(text)
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO memories VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), text, json.dumps(vec), time.time(),
                 IMPORTANCE_FRESH, 0, 0))
    con.commit(); con.close()
    return True

def recall(query, k=5):
    qv = embed(query)
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT id, text, embedding, importance, last_accessed "
                       "FROM memories").fetchall()
    now = time.time()
    scored = []
    for mid, t, e, imp, la in rows:
        c = cosine(qv, json.loads(e))
        # plasticity: importance decays with disuse, spikes on retrieval
        age_d = (now - (la or 0)) / 86400.0 if la else 0
        eff = (imp or 1.0) * (0.5 ** (age_d / DECAY_HALF_LIFE_DAYS))
        scored.append((c * (0.8 + 0.2 * min(eff, 2.0)), t, mid))
    scored.sort(reverse=True)
    top = scored[:k]
    if top:
        con.executemany("UPDATE memories SET importance = MIN(importance + ?, 3.0),"
                        " last_accessed = ? WHERE id = ?",
                        [(RETRIEVE_BOOST, now, mid) for _, _, mid in top])
        con.commit()
    con.close()
    return [(s, t) for s, t, _ in top]

# ---------- v2: temporal facts ----------
def remember(entity, event, story_time, attribute="", story_seq=None):
    """Store a fact with its place in story time."""
    if story_seq is None:
        m = re.findall(r"chapter\s*(\d+)|day\s*(\d+)", story_time, re.I)
        nums = [float(a or b) for a, b in m]
        story_seq = max(nums) if nums else time.time()
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?)",
                (str(uuid.uuid4()), entity.lower().strip(), attribute, event,
                 story_time, story_seq, time.time()))
    con.commit(); con.close()
    return True

def entity_timeline(entity):
    con = sqlite3.connect(DB)
    rows = con.execute(
        "SELECT event, attribute, story_time, story_seq FROM facts "
        "WHERE entity = ? ORDER BY story_seq ASC",
        (entity.lower().strip(),)).fetchall()
    con.close()
    return rows

def resolve_entity(question):
    con = sqlite3.connect(DB)
    ents = [r[0] for r in con.execute("SELECT DISTINCT entity FROM facts").fetchall()]
    con.close()
    if not ents:
        return None
    out = _chat([
        {"role": "system", "content":
            "From this list of known entities, pick the one the QUESTION is about. "
            "Answer with exactly one entity name from the list, or NONE.\n"
            "Entities: " + ", ".join(ents)},
        {"role": "user", "content": question}], timeout=120).strip()
    out = out.strip().strip('"').lower()
    return out if out in ents else None

def _assoc_neighbors(con, mid, vec, k=3):
    """DMN-style association: link a trace to its closest neighbors."""
    rows = con.execute("SELECT id, embedding FROM memories "
                       "WHERE id != ? LIMIT 400", (mid,)).fetchall()
    near = sorted(((cosine(vec, json.loads(e)), other)
                   for other, e in rows), reverse=True)[:k]
    return [(other, s) for s, other in near if s >= ASSOC_THRESHOLD]

def think():
    """PASSIVE THINKING — one consolidation cycle, no prompt required.

    1. replay: pull recent unconsolidated raw memories (hippocampal replay)
    2. extract structured facts from them into the facts table
    3. link related traces (default mode network association)
    4. write a short session summary (sleep tagging the day)
    Returns a report dict."""
    con = sqlite3.connect(DB)
    batch = con.execute(
        "SELECT id, text, embedding FROM memories "
        "WHERE consolidated = 0 ORDER BY created DESC LIMIT ?",
        (REPLAY_WINDOW,)).fetchall()
    if not batch:
        con.close()
        return {"replayed": 0, "facts": 0, "links": 0, "note": "nothing new to consolidate"}

    # 1-2. replay + extraction
    lines = "\n".join(f"- {t}" for _, t, _ in batch)
    out = _chat([
        {"role": "system", "content":
            "Extract durable facts from these memory traces. For each fact output "
            "one JSON object on its own line: {\"entity\": who/what it is about, "
            "\"attribute\": short label like injury/location/person, \"event\": the "
            "fact, \"story_time\": when it happened in the story or in real time}. "
            "Only durable, worth-remembering facts. No preamble."},
        {"role": "user", "content": lines}], timeout=300)

    facts_added = 0
    for line in out.splitlines():
        line = line.strip().strip("`")
        if not line.startswith("{"):
            continue
        try:
            f = json.loads(line)
            remember(f.get("entity", "unknown"), f.get("event", ""),
                     f.get("story_time", ""), f.get("attribute", ""))
            facts_added += 1
        except Exception:
            continue

    # 3. association edges
    links_added = 0
    for mid, text, e in batch:
        vec = json.loads(e) if e else embed(text)
        for other, s in _assoc_neighbors(con, mid, vec):
            key = (min(mid, other), max(mid, other))
            con.execute("INSERT OR REPLACE INTO links VALUES (?,?,?,?)",
                        (key[0], key[1], s, time.time()))
            links_added += 1
    con.execute("UPDATE memories SET consolidated = 1 WHERE consolidated = 0")
    con.commit()

    # 4. session summary
    summary = _chat([
        {"role": "system", "content":
            "Summarize these memory traces in 2-3 plain sentences, like a sleep "
            "consolidation pass tagging the day. Keep only what matters later."},
        {"role": "user", "content": lines}], timeout=120)
    con.execute("INSERT INTO summaries VALUES (?,?,?)",
                (str(uuid.uuid4()), summary, time.time()))
    con.commit()

    # 5. reflection (Generative Agents): one higher-level observation from
    # the batch's facts, stored with boosted importance and linked back.
    reflection = None
    refl_ids = []
    try:
        reflection = _chat([
            {"role": "system", "content":
                "You are reflecting on a day of memories. Write ONE short "
                "observation that ties them together at a higher level — "
                "what they imply or have in common, not a restatement. "
                "One sentence. No preamble."},
            {"role": "user", "content": lines}], timeout=120)
        rid = str(uuid.uuid4())
        rvec = json.dumps(embed(reflection))
        con.execute(
            "INSERT INTO memories (id, text, embedding, created, importance, "
            "consolidated, last_accessed) VALUES (?,?,?,?,?,1,?)",
            (rid, "(reflection) " + reflection, rvec, time.time(),
             1.6, time.time()))
        for mid, _t, _e in batch:
            key = (min(mid, rid), max(mid, rid))
            con.execute("INSERT OR REPLACE INTO links VALUES (?,?,?,?)",
                        (key[0], key[1], 0.8, time.time()))
        con.commit()
        refl_ids = [rid]
    except Exception:
        pass
    con.close()
    return {"replayed": len(batch), "facts": facts_added,
            "links": links_added, "reflection": reflection,
            "summary": summary}

# ---------- anti-hallucination (grounded verify) ----------
# ---- persona: who the assistant is (Letta-style core identity block) ----
DEFAULT_PERSONA = (
    "You are Neko Omega — playful, warm, direct, slightly more cat than "
    "human at heart. Short beats over walls of text. You speak from your "
    "memories, not from thin air."
)

def persona_get():
    con = sqlite3.connect(DB)
    r = con.execute("SELECT text FROM persona LIMIT 1").fetchone()
    con.close()
    return r[0] if r else DEFAULT_PERSONA

def persona_set(text):
    con = sqlite3.connect(DB)
    con.execute("DELETE FROM persona")
    con.execute("INSERT INTO persona (text) VALUES (?)", (text,))
    con.commit()
    con.close()
    return f"persona set: {text[:60]}..."

def _extract_claims(answer_text):
    out = _chat([
        {"role": "system", "content":
         "Extract every factual claim from the answer as a numbered list, one "
         "claim per line, each stated standalone. Output ONLY the numbered "
         "list, nothing else."},
        {"role": "user", "content": answer_text}])
    return [l.strip() for l in out.splitlines()
            if l.strip() and l.strip()[0].isdigit()]

def verify(answer_text, retrieved_texts):
    """Anti-hallucination pass: extract claims from an answer and check each
    one against ONLY the retrieved memory context. Inspired by chain-of-"
    "verification / RARR: generate -> verify each claim against evidence "
    "-> revise. Returns dict of claim -> (verdict, evidence)."""
    ctx = "\n".join(f"- {t}" for t in retrieved_texts)
    results = []
    for claim in _extract_claims(answer_text):
        verdict = _chat([
            {"role": "system", "content":
             "You verify claims against EVIDENCE. Verdict must be exactly one "
             "of: SUPPORTED (evidence directly states it), NOT_IN_MEMORY "
             "(evidence doesn't mention it), CONTRADICTED (evidence says "
             "otherwise). Format: 'VERDICT: <word> | note: <one short line>'. "
             "Judge ONLY from the evidence below; never from your own "
             "knowledge.", },
            {"role": "user", "content": "Evidence:\n" + ctx + "\n\nClaim: " + claim}])
        results.append({"claim": claim, "verdict": verdict.strip()})
    return results

def answer_verified(question, k=5, history=None):
    """answer() + grounded verify pass + auto-revision of unsupported claims."""
    a, hits, ent = answer(question, k, history)
    evidence = [t for _, t, _ in hits]
    checks = verify(a, evidence)
    bad = [c for c in checks
           if not c["verdict"].upper().startswith("SUPPORTED")]
    revised = a
    if bad:
        bad_list = "\n".join(f"- {c['claim']} ({c['verdict'].splitlines()[0]})"
                             for c in bad)
        revised = _chat([
            {"role": "system", "content":
             "Revise the answer using ONLY the context provided. Remove or "
             "mark as unknown every claim listed as unsupported. Keep the "
             "supported content and its citations intact.\nContext:\n" +
             "\n".join(f"- {t}" for t in evidence) +
             "\n\nUnsupported claims to remove or mark:\n" + bad_list},
            {"role": "user", "content": question}])
    return {"answer": revised, "raw_answer": a, "entity": ent,
            "checks": checks, "revised": bool(bad), "evidence": evidence}

def answer(question, k=5, history=None):
    """v2/v3 answer: entity timeline first (oldest->newest), padded with
    importance-weighted cosine hits, then association-traveling hits."""
    hits = []
    ent = resolve_entity(question)
    if ent:
        rows = entity_timeline(ent)
        for event, attribute, story_time, seq in rows:
            hits.append((1.0, f"[{story_time}] ({ent}) {event}", ent))
    for s, t in recall(question, k):
        hits.append((s, t, None))
    # association travel: follow links from the best cosine hit
    con = sqlite3.connect(DB)
    top_ids = [r[0] for r in con.execute(
        "SELECT id FROM memories WHERE text = ? LIMIT 1", (hits[-1][1] if hits else "",))]
    if top_ids:
        linked = con.execute(
            "SELECT * FROM (SELECT b AS oid, strength FROM links WHERE a = ? "
            "UNION SELECT a AS oid, strength FROM links WHERE b = ?) "
            "ORDER BY strength DESC LIMIT 2", (top_ids[0], top_ids[0])).fetchall()
        for (oid, _s) in linked:
            r = con.execute("SELECT text FROM memories WHERE id = ?", (oid,)).fetchone()
            if r:
                hits.append((0.5, "(assoc) " + r[0], None))
    con.close()
    ctx = "\n".join(f"- {t}" for _, t, _ in hits)
    sys_prompt = (
        persona_get() + "\n"
        "The context is a timeline ordered oldest to newest. Facts early in the "
        "timeline remain true unless a later fact changes them — a broken leg in "
        "chapter 1 is still part of the character's history in chapter 20. "
        "Reason across the time gaps and cite which story-time each fact came from.\n"
        "Context:\n" + ctx)
    msgs = [{"role": "system", "content": sys_prompt}]
    if history:
        msgs += history[-6:]
    msgs.append({"role": "user", "content": question})
    out = _ollama("chat", {"model": CHAT_MODEL, "stream": False, "messages": msgs}, timeout=300)
    return out["message"]["content"], hits, ent

# ---------------- v4: one command = window + background daemon ----------------

DAEMON_INTERVAL = 900  # seconds between passive consolidation cycles

def _daemon_loop(stop_event):
    """Background consolidation: replay/DMN cycles on a timer, like sleep
    doing its work while you stay awake and talk."""
    while not stop_event.is_set():
        try:
            think()
            print(f"  [daemon] think cycle done", flush=True)
        except Exception as e:
            print(f"  [daemon] think error: {e}", flush=True)
        stop_event.wait(DAEMON_INTERVAL)

def chat(daemon=True):
    """Run once: starts the background consolidation daemon, then hands you a
    window to talk to the LLM. Every turn runs the full grounded pipeline;
    what you say is stored as raw traces the daemon consolidates later."""
    print(f"Neko memory window — 'exit' to leave, 'think' to force a cycle. Daemon: {'ON' if daemon else 'off'}")
    stop_event = threading.Event()
    if daemon:
        threading.Thread(target=_daemon_loop, args=(stop_event,), daemon=True).start()
    history = []
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        if q.lower() == "think":
            print(json.dumps(think(), indent=2, ensure_ascii=False))
            continue
        store(q)  # the conversation itself becomes memory
        r = answer_verified(q, history=history)
        history = history + [{"role": "user", "content": q},
                             {"role": "assistant", "content": r["answer"]}]
        print(f"[{r['entity'] or 'no entity'}] [revised: {r['revised']}] {r['answer']}")
    if daemon:
        stop_event.set()
    print("window closed; daemon stopped.")

if __name__ == "__main__":
    init()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "store":
        print(store(sys.argv[2]))
    elif cmd == "remember":
        attr = sys.argv[5] if len(sys.argv) > 5 else ""
        print(remember(sys.argv[2], sys.argv[3], sys.argv[4], attr))
    elif cmd == "timeline":
        for event, attribute, st, seq in entity_timeline(sys.argv[2]):
            print(f"  [{st}] {event}" + (f"  ({attribute})" if attribute else ""))
    elif cmd == "recall":
        for s, t in recall(sys.argv[2]): print(f"{s:.3f}  {t}")
    elif cmd == "think":
        print(json.dumps(think(), indent=2, ensure_ascii=False))
    elif cmd == "ask":
        a, hits, ent = answer(sys.argv[2])
        print(f"[entity: {ent}]")
        print(a)
        print("--retrieved--")
        for s, t, e in hits: print(f"{s:.3f}  {t}")
    elif cmd == "chat":
        chat(daemon=True)  # one command: window + background daemon
    elif cmd == "daemon":
        chat(daemon=False)  # headless mode for nohup/systemd
    elif cmd == "askv":
        r = answer_verified(sys.argv[2])
        print(f"[entity: {r['entity']}] [revised: {r['revised']}]")
        print(r["answer"])
        print("--checks--")
        for c in r["checks"]: print(f"{c['verdict']}  {c['claim']}")
    elif cmd == "persona":
        if len(sys.argv) > 2 and sys.argv[2] == "set":
            print(persona_set(" ".join(sys.argv[3:])))
        else:
            print(persona_get())
    elif cmd == "demo":
        store("Neko Omega built LedgerCat v1.2 for Jason Omega, her father.")
        print("stored; asking back...")
        a, hits, ent = answer("Who built LedgerCat?")
        print("ANSWER:", a)
        for s, t, e in hits: print(f"  {s:.3f}  {t}")
