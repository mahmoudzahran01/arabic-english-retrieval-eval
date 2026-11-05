# lib_drive.sh - shared helpers for queue drivers (source it AFTER defining L, G and the log name LOGF).
# step NAME CMD...   runs CMD through the gate, logs START/END with the exit code, and records failures in $L/failed_steps.
#                    It does NOT abort the queue (later steps may be independent); callers still verify outputs with need().
step() { local name=$1; shift; echo "$(date +%T) START $name" >> "$LOGF"; "$G" "$@" > "$L/$name.log" 2>&1; local rc=$?
  echo "$(date +%T) END $name rc=$rc" >> "$LOGF"; [ "$rc" -eq 0 ] || echo "$(date -Is) $name rc=$rc" >> "$L/failed_steps"; return $rc; }
need() { for f in "$@"; do [ -s "$f" ] || { echo "$(date +%T) FAIL: missing $f" >> "$LOGF"; exit 1; }; done; }
# count_mteb DIR_UNDER_RESULTS  - number of task result files (excluding model_meta.json) written for one result name
count_mteb() { find "$MC/$1" -name "*.json" ! -name model_meta.json 2>/dev/null | wc -l; }
