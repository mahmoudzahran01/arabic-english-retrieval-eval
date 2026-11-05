#!/usr/bin/env bash
# usage: memwait.sh GB   - block until MemAvailable >= GB
while [ "$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)" -lt "$1" ]; do sleep 30; done
