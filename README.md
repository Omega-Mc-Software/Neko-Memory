# Neko Memory

Long-term memory layer for Legion — v3, temporal memory + passive thinking.

v1 was cosine-only: whatever text was closest won, and story time was ignored.
v2 adds:
- structured facts: entity, attribute, event text, story_time (chapter/day)
- entity-indexed recall (find WHO the question is about, pull that entity's whole history, oldest first)
- an LLM synthesis pass that reads the entity timeline and answers across time gaps ("broke a leg in ch 1" → still known in ch 20)

v3 adds PASSIVE THINKING, modeled on real neuroscience:
- hippocampal replay / systems consolidation — `think()` pulls recent unconsolidated traces and folds them into structured facts, no prompt needed
- default mode network association — traces above a cosine threshold get linked edges, so recall can travel associations, not just similarity
- synaptic plasticity — importance rises on retrieval, decays with a 21-day half-life of disuse, and weights the ranking
- consolidation summaries — each replay batch writes a short session summary, the way sleep tags a day

Fully local: Ollama glm-4.7-flash (chat) + nomic-embed-text (embeddings). SQLite. Nothing leaves the machine.

## Usage

```
python3 memory_store.py store "a raw memory trace"
python3 memory_store.py think                 # passive consolidation cycle
python3 memory_store.py ask "your question"   # entity timeline + weighted recall
```

## Roadmap (robot-intelligence target)

- spatial and people-memory as first-class fields (pathways found, people met, things helped with — months or years later)
- scheduled passive cycles (cron) so consolidation happens on a rhythm, not only on demand
