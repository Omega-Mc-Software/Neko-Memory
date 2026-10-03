# Memory-system thinking test on each debate model, once per cycle.
import json, time, traceback, sys
sys.path.insert(0, "/home/neko/memory")
import memory_store

MODELS = ["glm-4.7-flash", "qwen2.5:3b", "gemma2:2b"]

def main():
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"== memtest {stamp} ==", flush=True)
    for m in MODELS:
        t0 = time.time()
        tag = "memtest-" + m.split(":")[0].replace(".", "") + "-" + time.strftime("%Y%m%d%H%M", time.gmtime())
        entry = {"model": m, "probe_tag": tag}
        try:
            memory_store.CHAT_MODEL = m
            memory_store.remember(tag, f"During the {stamp} memory test, the {m} model wrote a probe fact into Neko-Memory.", "now")
            entry["remember"] = "ok"
        except Exception as e:
            entry["remember"] = f"FAIL: {e}"
        try:
            r = memory_store.think()
            entry["think"] = (json.dumps(r, default=str)[:300] if not isinstance(r, str) else r[:300])
        except Exception as e:
            entry["think"] = f"FAIL: {e}"
        try:
            av = memory_store.answer_verified(f"What probe fact did the {m} model write during the memory test?")
            if not isinstance(av, str):
                av = json.dumps(av, default=str)
            entry["answer_verified"] = av[:500]
        except Exception as e:
            entry["answer_verified"] = f"FAIL: {e}"
        entry["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(entry, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
