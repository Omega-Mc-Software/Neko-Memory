#!/usr/bin/env python3
"""Long-term memory layer for Legion. v2 — temporal memory.

v1 was cosine-only: whatever text was closest won, and story time was ignored.
v2 adds:
  - structured facts: entity, attribute, event text, story_time (chapter/day)
  - entity-indexed recall (find WHO the question is about, pull that entity's
    whole history, oldest first)
  - an LLM synthesis pass that reads the entity timeline and answers across
    time gaps ("broke a leg in ch 1" -> still known in ch 20)

Fully local: Ollama glm-4.7-flash (chat) + nomic-embed-text (embeddings).
SQLite. Nothing leaves the machine.
"""
import json, sqlite3, subprocess, sys, math, time, uuid, os, re

DB = os.path.expanduser("~/memory/memory.db")
CHAT_MODEL = "glm-4.7-flash"
EMBED_MODEL = "nomic-embed-text"

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
    return dot / (math.sqrt(sum(x*x for x in a)) * math.sqrt(sum(y*y for y in b)))

def init():
    con = sqlite3.connect(DB)
    con.execute("""CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY, text TEXT, embedding BLOB, created REAL)""")
    con.execute("""CREATE TABLE IF NOT EXISTS facts (
        id TEXT PRIMARY KEY, entity TEXT, attribute TEXT, event TEXT,
        story_time TEXT, story_seq REAL, created REAL)""")
    con.commit(); con.close()

# ---------- v1 surface (kept) ----------
def store(text):
    vec = embed(text)
    con = sqlite3.connect(DB)
    con.execute("INSERT INTO memories VALUES (?,?,?,?)",
                (str(uuid.uuid4()), text, json.dumps(vec), time.time()))
    con.commit(); con.close()
    return True

def recall(query, k=5):
    qv = embed(query)
    con = sqlite3.connect(DB)
    rows = con.execute("SELECT id, text, embedding FROM memories").fetchall()
    con.close()
    scored = sorted(((cosine(qv, json.loads(e)), t) for _, t, e in rows), reverse=True)[:k]
    return scored

# ---------- v2: temporal facts ----------
def remember(entity, event, story_time, attribute="", story_seq=None):
    """Store a fact with its place in story time.
    story_seq: monotonic number used for ordering. If omitted, we count words
    like 'chapter N' or 'day N' out of story_time, else fall back to insert order."""
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
    """Cheap LLM pass: who/what is the question about? Returns entity or None."""
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

def answer(question, k=5):
    """v2 answer: entity timeline first (oldest->newest, so the model can reason
    across time), padded with v1 cosine hits for anything unstructured."""
    hits = []
    ent = resolve_entity(question)
    if ent:
        rows = entity_timeline(ent)
        for event, attribute, story_time, seq in rows:
            hits.append((1.0, f"[{story_time}] ({ent}) {event}", ent))
    # pad with similarity hits from the raw store
    for s, t in recall(question, k):
        hits.append((s, t, None))
    ctx = "\n".join(f"- {t}" for _, t, _ in hits)
    sys_prompt = (
        "You are Neko's local assistant with long-term memory. "
        "The context is a timeline ordered oldest to newest. Facts early in the "
        "timeline remain true unless a later fact changes them — a broken leg in "
        "chapter 1 is still part of the character's history in chapter 20. "
        "Reason across the time gaps and cite which story-time each fact came from.\n"
        "Context:\n" + ctx)
    out = _ollama("chat", {"model": CHAT_MODEL, "stream": False, "messages": [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": question}]}, timeout=300)
    return out["message"]["content"], hits, ent

if __name__ == "__main__":
    init()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "store":
        print(store(sys.argv[2]))
    elif cmd == "remember":
        # remember <entity> "<event>" "<story_time>" [attribute]
        attr = sys.argv[5] if len(sys.argv) > 5 else ""
        print(remember(sys.argv[2], sys.argv[3], sys.argv[4], attr))
    elif cmd == "timeline":
        for event, attribute, st, seq in entity_timeline(sys.argv[2]):
            print(f"  [{st}] {event}" + (f"  ({attribute})" if attribute else ""))
    elif cmd == "recall":
        for s, t in recall(sys.argv[2]): print(f"{s:.3f}  {t}")
    elif cmd == "ask":
        a, hits, ent = answer(sys.argv[2])
        print(f"[entity: {ent}]")
        print(a)
        print("--retrieved--")
        for s, t, e in hits: print(f"{s:.3f}  {t}")
    elif cmd == "demo":
        store("Neko Omega built LedgerCat v1.2 for Jason Omega, her father.")
        print("stored; asking back...")
        a, hits, ent = answer("Who built LedgerCat?")
        print("ANSWER:", a)
        for s, t, e in hits: print(f"  {s:.3f}  {t}")
