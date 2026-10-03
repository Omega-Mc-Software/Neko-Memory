#!/usr/bin/env python3
"""Psychology debate harness — three local LLMs argue, Neko's memory system as reference.
A (glm-4.7-flash) vs B (qwen2.5:3b) with C (gemma2:2b) as devil's advocate who attacks any
consensus forming. Loop keeps rounds nonstop. Transcripts: ./transcripts/YYYY-MM-DD.md
Anti-agreement rules in every system prompt. No fabricated facts; lookups go through Neko-Memory.
"""
import json, os, re, sys, datetime

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, "/home/neko/memory")
import memory_store

MODEL_A = "glm-4.7-flash"   # big debater
MODEL_B = "qwen2.5:3b"      # smaller opponent
MODEL_C = "gemma2:2b"       # devil's advocate
EXCHANGES = 3               # speaking turns per model per round
REVISIT_EVERY = 6           # every N rounds: re-debate an old topic pulled from memory
STATE = os.path.join(HERE, "state.json")
TDIR = os.path.join(HERE, "transcripts")

COMMON = ("RULES: (1) Do NOT agree just to be agreeable. If the other debaters converge on a view, "
          "that is exactly when you look for the strongest objection. (2) Never fabricate facts, "
          "studies, statistics, or citations. If you are not sure something is real, say so plainly. "
          "(3) You may look things up: include a line exactly 'LOOKUP: <query>' and the Neko-Memory "
          "system's verified answer will be injected next turn. (4) End every message by restating "
          "your strongest point in one sentence.")

def chat(model, system, messages):
    msgs = [{"role": "system", "content": system}] + messages
    return memory_store._ollama("chat", {"model": model, "stream": False,
                                         "options": {"temperature": 0.8, "num_predict": 400},
                                         "messages": msgs}, timeout=420)["message"]["content"]

def lookup(query):
    try:
        return memory_store.answer_verified(query.strip())[:1200]
    except Exception as e:
        return f"(lookup failed: {e})"

def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"round": 0, "topic": "Is personality formed more by childhood experience or by ongoing adult choices?"}

def choose_topic(st):
    if st["round"] % REVISIT_EVERY != 0:
        return st["topic"], False
    try:
        hits = memory_store.recall("psychology debate topic argument", k=3)
        old = " | ".join(h[1][:150] for h in hits) if hits else ""
        if old:
            return (f"REVISIT: we argued before. Here is what memory holds: {old}. "
                    "Re-examine it: what holds up, what did we get wrong, say so with reasons."), True
    except Exception:
        pass
    return st["topic"], False

def main():
    st = load_state()
    st["round"] += 1
    rnd = st["round"]
    os.makedirs(TDIR, exist_ok=True)
    day = datetime.date.today().isoformat()
    tf = os.path.join(TDIR, f"{day}.md")

    sysA = ("You are Debater A in a running psychology debate with two rivals. Argue rigorously, "
            "ground claims in mechanisms and real research traditions. Take and HOLD a position; "
            "changing your mind requires evidence, not politeness. " + COMMON)
    sysB = ("You are Debater B, the smaller model. Do not try to sound big: attack A's weakest "
            "premise precisely, concede only what you must, and never agree to move the debate along. " + COMMON)
    sysC = ("You are the Devil's Advocate. You take no side: your job is to attack whatever consensus "
            "is forming. Read the exchange so far, name the claim everyone is drifting toward, and "
            "argue the opposite of it as hard as you can. If they truly disagree already, attack the "
            "shared assumption underneath both. " + COMMON)

    topic, is_revisit = choose_topic(st)
    lines = [f"\n## Round {rnd} — {datetime.datetime.now().isoformat(timespec='seconds')}" + (" *(REVISIT — memory-recall test)*" if is_revisit else ""),
             f"**Topic:** {topic}", ""]

    order = [(MODEL_A, sysA, "A"), (MODEL_B, sysB, "B"), (MODEL_C, sysC, "C")] * EXCHANGES
    for i, (model, system, tag) in enumerate(order):
        if i == 0:
            msgs = [{"role": "user", "content": f"Debate topic: {topic}\nGive your opening argument."}]
        else:
            msgs = [{"role": "user", "content": f"Transcript so far this round:\n\n" + "\n".join(lines[-14:]) +
                     f"\n\nYou are Debater {tag}. Respond. Remember the rules: no fake facts, do not agree for peace."}]
        try:
            out = chat(model, system, msgs)
        except Exception as e:
            lines.append(f"*({model} failed: {e})*")
            continue
        lines.append(f"### {model} ({tag})")
        lines.append(out.strip())
        lines.append("")
        m = re.search(r"LOOKUP:\s*(.+)", out)
        if m:
            ans = lookup(m.group(1))
            lines.append(f"> **Memory lookup** ({m.group(1).strip()}): {ans}")
            lines.append("")

    # verify round against memory
    try:
        full = " ".join(lines)
        rec = memory_store.recall(full[:500], k=1)
        verdicts = memory_store.verify(full, [rec[0][1] if rec else ""])
        lines.append("**Memory verification (claims vs Neko-Memory):**")
        lines.append(str(verdicts)[:800])
        lines.append("")
    except Exception as e:
        lines.append(f"*(verification failed: {e})*")

    last = [l for l in lines if l and not l.startswith(("#", "**", ">", "*", "----"))]
    st["topic"] = f"Continuing from round {rnd}'s closing point: {last[-1][:200] if last else topic}"
    with open(STATE, "w") as f:
        json.dump(st, f)
    with open(tf, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"round {rnd} -> {tf}")

if __name__ == "__main__":
    main()
