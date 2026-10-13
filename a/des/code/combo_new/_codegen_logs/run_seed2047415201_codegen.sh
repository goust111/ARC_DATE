#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/lixinyu/snap/a/des/code/combo_new/seed2047415201"
LOG_ROOT="/home/lixinyu/snap/a/des/code/combo_new/_codegen_logs"
CODEX_WORKDIR="/home/lixinyu/snap/a/des/code/combo_new"
MODEL="${MODEL:-gpt-5.5}"
EFFORT="${EFFORT:-low}"
PARALLEL="${PARALLEL:-4}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-900}"

RUN_DIR="$LOG_ROOT/run_$(date +%Y%m%d_%H%M%S)_seed2047415201"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$LOG_ROOT/latest_seed2047415201_run.txt"

find "$ROOT" -path '*/codegen_prompts/*.md' -type f | sort -V > "$RUN_DIR/prompts.txt"
: > "$RUN_DIR/status.tsv"

run_one() {
  local prompt="$1"
  local out_dir target rel log
  out_dir="$(dirname "$(dirname "$prompt")")"
  target="$out_dir/$(basename "$prompt" .md).py"
  rel="${target#"$ROOT"/}"
  log="$RUN_DIR/${rel//\//__}.log"

  if [[ -s "$target" ]]; then
    printf '[SKIP_EXISTING]\t%s\t%s\n' "$prompt" "$target" >> "$RUN_DIR/status.tsv"
    return 0
  fi

  printf '[START]\t%s\t%s\n' "$prompt" "$target" >> "$RUN_DIR/status.tsv"

  if { printf '请严格执行下面的 ARC 新题代码生成提示词。必须把代码写入 prompt 指定的目标路径，并运行 prompt 里的自检命令。\n\n'; cat "$prompt"; } | timeout "$TIMEOUT_SECONDS" codex exec \
    --skip-git-repo-check \
    --cd "$CODEX_WORKDIR" \
    --sandbox danger-full-access \
    -c model="\"$MODEL\"" \
    -c reasoning.effort="\"$EFFORT\"" \
    - > "$log" 2>&1; then
    if [[ -s "$target" ]]; then
      printf '[OK]\t%s\t%s\n' "$prompt" "$target" >> "$RUN_DIR/status.tsv"
    else
      printf '[FAILED_NO_TARGET]\t%s\t%s\n' "$prompt" "$target" >> "$RUN_DIR/status.tsv"
      return 1
    fi
  else
    local rc=$?
    if [[ "$rc" == "124" ]]; then
      printf '[TIMEOUT]\t%s\t%s\n' "$prompt" "$target" >> "$RUN_DIR/status.tsv"
    else
      printf '[FAILED]\t%s\t%s\trc=%s\n' "$prompt" "$target" "$rc" >> "$RUN_DIR/status.tsv"
    fi
    return "$rc"
  fi
}

export ROOT RUN_DIR CODEX_WORKDIR MODEL EFFORT TIMEOUT_SECONDS
export -f run_one

printf '[RUN]\tmodel=%s\teffort=%s\tparallel=%s\ttimeout=%s\tstarted_at=%s\n' \
  "$MODEL" "$EFFORT" "$PARALLEL" "$TIMEOUT_SECONDS" "$(date -Is)" | tee "$RUN_DIR/meta.tsv"

xargs -r -n 1 -P "$PARALLEL" bash -lc 'run_one "$0"' < "$RUN_DIR/prompts.txt"

printf '[DONE]\tfinished_at=%s\n' "$(date -Is)" | tee -a "$RUN_DIR/meta.tsv"
