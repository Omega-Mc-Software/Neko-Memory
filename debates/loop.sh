#!/bin/bash
# Debate cycle: round, 20-minute break, memory-system thinking test on each model, reconvene.
cd /home/neko/psych_debate
while true; do
  python3 debate.py >> loop.log 2>&1 || echo "$(date -u) round failed" >> loop.log
  echo "$(date -u) break 20min" >> loop.log
  sleep 1200
  python3 memtest.py >> memtest.log 2>&1 || echo "$(date -u) memtest failed" >> loop.log
  echo "$(date -u) reconvene" >> loop.log
done