#!/usr/bin/env bash
# usage: dk.sh <python args...>   (runs python inside the retrieval image through the shared 2-slot GPU lock)
# PYTHONPATH has the repo (/lab) and the private mteb install; banner is stripped by callers if needed.
exec /home/mzahran1/projects/qubit-ai-lab/lab/retrieval/run_docker.sh env PYTHONPATH=/lab:/lab/runs/retrieval/v2/pylibs:/lab/runs/retrieval/v2/scripts python "$@"
