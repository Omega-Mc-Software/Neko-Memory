# Legion Memory

Long-term memory layer for Legion — v2, temporal memory.

v1 was cosine-only: whatever text was closest won, and story time was ignored.
v2 adds:
- structured facts: entity, attribute, event text, story_time (chapter/day)
- entity-indexed recall (find WHO the question is about, pull that entity's whole history, oldest first)
- an LLM synthesis pass that reads the entity timeline and answers across time gaps ("broke a leg in ch 1" → still known in ch 20)

Fully local: Ollama glm-4.7-flash (chat) + nomic-embed-text (embeddings). SQLite. Nothing leaves the machine.

## Usage

```
python3 memory_store.py --help
```

## Roadmap (robot-intelligence target)

- timestamps + importance scoring on every memory, so a ten-day-old fact still beats a ten-minute-old one when it matters
- session summaries stored alongside raw facts, so chapter 1 and chapter 20 stay linked through the chapters between
- spatial and people-memory as first-class fields (pathways found, people met, things helped with — months or years later)
