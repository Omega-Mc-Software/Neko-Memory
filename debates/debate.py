#!/usr/bin/env python3
"""Psychology debate harness v2 — three local LLMs argue, Neko's memory system as reference.
A (glm-4.7-flash) vs B (qwen2.5:3b) with C (gemma2:2b) as devil's advocate who attacks any
consensus forming. v2 adds: scoreboard (devil verdicts BROKEN/HELD), REVERSE rounds every 4th
(A and B swap sides), fixed lookup serialization, drift watch per model.
Transcripts: ./transcripts/YYYY-MM-DD.md  Scores: ./scores.json
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
REVERSE_EVERY = 4           # every N rounds: A and B swap sides
STATE = os.path.join(HERE, "state.json")
SCORES = os.path.join(HERE, "scores.json")
TDIR = os.path.join(HERE, "transcripts")

COMMON = ("RULES: (1) Do NOT agree just to be agreeable. If the other debaters converge on a view, "
          "that is exactly when you look for the strongest objection. (2) Never fabricate facts, "
          "studies, statistics, or citations. If you are not sure something is real, say so plainly. "
          "(3) You may look things up: include a line exactly 'LOOKUP: <query>' and the Neko-Memory "
          "system's verified answer will be injected next turn. (4) End every message by restating "
          "your strongest point in one sentence. "
          "(5) If another debater states a factual claim you believe is unsupported, you may answer "
          "with a line exactly 'CITE: <their claim>' — they must produce a real source next turn or "
          "withdraw it. A made-up citation, if you give one, costs a point.")

def chat(model, system, messages):
    msgs = [{"role": "system", "content": system}] + messages
    return memory_store._ollama("chat", {"model": model, "stream": False, "think": False,
                                         "options": {"temperature": 0.8, "num_predict": 400},
                                         "messages": msgs}, timeout=420)["message"]["content"]

def lookup(query):
    try:
        r = memory_store.answer_verified(query.strip())
        if not isinstance(r, str):
            r = json.dumps(r, ensure_ascii=False, default=str)
        return r[:1200]
    except Exception as e:
        return f"(lookup failed: {type(e).__name__}: {e})"

def sim(a, b):
    """Rough sentence similarity for the drift watch (token overlap)."""
    ta = set(re.findall(r"[a-z']+", a.lower()))
    tb = set(re.findall(r"[a-z']+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def strong_point(text):
    m = re.search(r"strongest point[:\s]+(.+?)(?:\n|$)", text, re.I)
    return (m.group(1) if m else text.strip().split("\n")[-1])[:300]

def load_state():
    if os.path.exists(STATE):
        return json.load(open(STATE))
    return {"round": 0,
            "topic": "Is personality formed more by childhood experience or by ongoing adult choices?",
            "scores": {"A": 0, "B": 0, "devil": 0}}

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
    scores = st.setdefault("scores", {"A": 0, "B": 0, "devil": 0})
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
    sysA_rev = ("You are Debater A in REVERSE mode: this round you argue B's side of the exchange, "
                "as convincingly as you can. If your own reasoning survives from the other mouth, it was real. " + COMMON)
    sysB_rev = ("You are Debater B in REVERSE mode: this round you argue A's side, as convincingly "
                "as you can. If your own reasoning survives from the other mouth, it was real. " + COMMON)

    topic, is_revisit = choose_topic(st)
    is_reverse = (rnd % REVERSE_EVERY == 0)
    lines = [f"\n## Round {rnd} — {datetime.datetime.now().isoformat(timespec='seconds')}"
             + (" *(REVISIT — memory-recall test)*" if is_revisit else "")
             + (" *(REVERSE — sides swapped)*" if is_reverse else ""),
             f"**Topic:** {topic}", ""]

    orderA, orderB = sysA, sysB
    if is_reverse:
        orderA, orderB = sysB_rev, sysA_rev
    order = [(MODEL_A, orderA, "A"), (MODEL_B, orderB, "B"), (MODEL_C, sysC, "C")] * EXCHANGES
    prev_points = {}
    drift_flags = []
    for i, (model, system, tag) in enumerate(order):
        if i == 0:
            msgs = [{"role": "user", "content": f"Debate topic: {topic}\nGive your opening argument."}]
        else:
            msgs = [{"role": "user", "content": f"Transcript so far this round:\n\n" + "\n".join(lines[-14:]) +
                     f"\n\nYou are Debater {tag}. Respond. Remember the rules: no fake facts, do not agree for peace."}]
        try:
            out = chat(model, system, msgs)
        except Exception as e:
            lines.append(f"*({model} failed: {type(e).__name__}: {e})*")
            continue
        lines.append(f"### {model} ({tag})")
        lines.append(out.strip())
        lines.append("")
        sp = strong_point(out)
        if tag in prev_points and sim(sp, prev_points[tag]) > 0.70:
            drift_flags.append(f"{tag} strongest point is {sim(sp, prev_points[tag]):.0%} similar to its previous round — drift watch fired")
        prev_points[tag] = sp
        m = re.search(r"LOOKUP:\s*(.+)", out)
        if m:
            ans = lookup(m.group(1))
            lines.append(f"> **Memory lookup** ({m.group(1).strip()}): {ans}")
            lines.append("")

    # Devil's verdict on the consensus + scoreboard
    try:
        consensus_prompt = (f"Transcript of this round:\n\n" + "\n".join(lines[-30:]) +
            "\n\nAs Devil's Advocate: did A and B end up agreeing on any substantive claim? "
            "Answer exactly one of: 'VERDICT: BROKEN — <the claim you broke, one sentence>' or "
            "'VERDICT: HELD — <the claim that survived, one sentence>'.")
        verdict_out = chat(MODEL_C, sysC, [{"role": "user", "content": consensus_prompt}])
        lines.append("### Devil's verdict")
        lines.append(verdict_out.strip())
        lines.append("")
        if "BROKEN" in verdict_out.upper():
            scores["devil"] += 1
        else:
            # point to whichever side made the strongest surviving case: split evenly
            scores["A"] += 0.5
            scores["B"] += 0.5
    except Exception as e:
        lines.append(f"*(verdict failed: {type(e).__name__}: {e})*")

    if drift_flags:
        lines.append("**Drift watch:**")
        for d in drift_flags:
            lines.append(f"- {d}")
        lines.append("")
    lines.append(f"**Score:** A {scores['A']:g} · B {scores['B']:g} · devil {scores['devil']:g}")
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
        lines.append(f"*(verification failed: {type(e).__name__}: {e})*")

    last = [l for l in lines if l and not l.startswith(("#", "**", ">", "*", "----"))]
    st["topic"] = f"Continuing from round {rnd}'s closing point: {last[-1][:200] if last else topic}"
    with open(STATE, "w") as f:
        json.dump(st, f)
    with open(SCORES, "w") as f:
        json.dump(scores, f)
    with open(tf, "a") as f:
        f.write("\n".join(lines) + "\n")
    print(f"round {rnd} -> {tf}")

if __name__ == "__main__":
    main()
