#!/bin/bash
# Nonstop debate loop — one round every 20 minutes, survives failures.
cd /home/neko/psych_debate
while true; do
  python3 debate.py >> loop.log 2>&1 || echo "$(date -u) round failed" >> loop.log
  sleep 1200
done
