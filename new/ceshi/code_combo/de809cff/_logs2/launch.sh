#!/usr/bin/env bash
set -euo pipefail

base=/home/lixinyu/snap/new
task_id=de809cff
logdir="$base/ceshi/code_combo/$task_id/_logs2"
mkdir -p "$logdir" "$base/ceshi/tem3"
state="$logdir/state.tsv"
: > "$state"

concurrency="${1:-4}"

cleanup_and_build_prompts() {
  for idx in $(seq 1 42); do
    rm -f "$base/ceshi/tem3/combo_prompt_${task_id}_${idx}.txt"
    rm -f "$base/ceshi/code_combo/$task_id/$idx/code.py"
    python "$base/combo_prompt_builder.py" "$task_id" "$idx"
  done
}

launch_one() {
  local idx="$1"
  local prompt_file="$base/ceshi/tem3/combo_prompt_${task_id}_${idx}.txt"
  local log_file="$logdir/combo_${idx}.log"
  (
    echo "INDEX $idx"
    echo "PROMPT $prompt_file"
    echo "START $(date '+%F %T')"
    python - "$prompt_file" <<'PY' | codex exec --skip-git-repo-check --disable apps --disable plugins --disable tool_suggest --disable tool_search --disable browser_use -C /home/lixinyu/snap/new -c model='"gpt-5.4"' -c reasoning.effort='"high"' -
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text()
for src, dst in [
    ("Claude Code", "Codex CLI"),
    ("Claude专属", "Codex CLI专属"),
    ("Claude", "Codex"),
    ("claude", "codex"),
    ("subagent", "agent"),
]:
    text = text.replace(src, dst)
print(text, end="")
PY
    status=$?
    echo "END $(date '+%F %T') status=$status"
    exit "$status"
  ) >"$log_file" 2>&1 &
  local pid=$!
  printf '%s\t%s\t%s\n' "$idx" "$pid" "$log_file" >> "$state"
  echo "launched idx=$idx pid=$pid log=$log_file"
}

cleanup_and_build_prompts

for idx in $(seq 1 42); do
  while [ "$(jobs -pr | wc -l)" -ge "$concurrency" ]; do
    sleep 5
  done
  launch_one "$idx"
done

wait
