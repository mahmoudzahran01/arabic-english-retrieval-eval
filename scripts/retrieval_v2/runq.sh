#!/usr/bin/env bash
# usage: runq.sh QUEUEFILE   (each non-comment line: NAME train_v2-args...).  Skips a run whose results/NAME.dev.json exists.
# Memory guard: the Spark's 121 GB are shared with the llama-servers and other agents' evals (unified memory); do not start a run while < 25 GB are available.
Q=$(readlink -f "$1"); cd ~/projects/qubit-ai-lab
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  name=${line%% *}
  [ -f runs/retrieval/v2/results/$name.dev.json ] && continue
  while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt 25 ]; do sleep 30; done
  runs/retrieval/v2/scripts/dk.sh -m lab.retrieval.v2.train_v2 --name $line > runs/retrieval/v2/logs/$name.log 2>&1
done < "$Q"
echo "QUEUE_DONE $1" >> runs/retrieval/v2/logs/queues.done
