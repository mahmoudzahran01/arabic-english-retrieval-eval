#!/usr/bin/env bash
# gate.sh CMD...  - run CMD only when the machine is quiet (coordinator rules, S_TIER.md "Resource rules"):
#   * my own jobs are serialised by a flock (two of my drivers can never pass the gate together),
#   * no run_eval.py (MTEB agent) process, * no other qubit-lab-retrieval container running (ONE GPU job at a time machine-wide),
#   * MemAvailable >= 25 GB,
#   * ALL of the above continuously for QUIET seconds (default 600) so that the gap between two MTEB-agent runs is not mistaken for "finished" -
#     except right after one of MY jobs ended (< 180 s ago): then nobody else came in and the window is skipped.
exec 9>/tmp/v2_one_gpu_job.lock
flock 9
QUIET=${QUIET:-600}; since=0
last=$(cat /tmp/v2_gate_last_end 2>/dev/null || echo 0)
[ $(( $(date +%s) - last )) -lt 180 ] && QUIET=0
while true; do
  ev=$(ps -eo args= | awk '/[r]un_eval\.py/ {n++} END{print n+0}')
  ct=$(docker ps --filter ancestor=qubit-lab-retrieval --format '{{.ID}}' 2>/dev/null | wc -l)
  mem=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
  if [ "$ev" -eq 0 ] && [ "$ct" -eq 0 ] && [ "$mem" -ge 25 ]; then
    [ "$since" -eq 0 ] && since=$(date +%s)
    [ $(( $(date +%s) - since )) -ge "$QUIET" ] && break
  else
    since=0
  fi
  sleep 20
done
"$@"; rc=$?
date +%s > /tmp/v2_gate_last_end
exit $rc
