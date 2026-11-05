#!/usr/bin/env bash
# After the xling2 translation has finished, translate the v3 candidates (extra SQuAD-train questions -> Arabic). Gemma on :8001, concurrency 2 (never two translators at once).
cd ~/projects/qubit-ai-lab
L=runs/retrieval/v2/logs
while ps -eo args= | awk '/[l]ab\.retrieval\.v2\.translate_v2/ {f=1} END{exit !f}'; do sleep 30; done
echo "$(date +%T) xling2 translation done ($(wc -l < runs/retrieval/v2/data/xling2_translations.jsonl) lines); starting xling3" >> $L/drive_tr3.log
PYTHONPATH=. .venv/bin/python -m lab.retrieval.v2.translate_v2 xling3 > $L/translate_v3.log 2>&1
echo "$(date +%T) xling3 translation done ($(wc -l < runs/retrieval/v2/data/xling3_translations.jsonl) lines)" >> $L/drive_tr3.log
