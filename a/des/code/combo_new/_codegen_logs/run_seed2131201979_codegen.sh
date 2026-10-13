#!/usr/bin/env bash
set -u

ROOT="/home/lixinyu/snap/a/des/code/combo_new"
SEED="seed2131201979"
RUN_DIR="$ROOT/_codegen_logs/run_$(date +%Y%m%d_%H%M%S)_$SEED"
PROMPT_LIST="$RUN_DIR/prompts.txt"
MISSING_LIST="$RUN_DIR/missing_prompts.txt"
STOP_FILE="$RUN_DIR/stop_usage_limit"
PARALLEL="${PARALLEL:-5}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-2400}"
MODEL="${MODEL:-gpt-5.4}"
EFFORT="${EFFORT:-high}"

if [[ "$MODEL" != "gpt-5.4" && "$MODEL" != "gpt5.4" ]]; then
  printf '[CONFIG_ERROR]\tMODEL must be gpt-5.4/gpt5.4, got %s\n' "$MODEL" >&2
  exit 2
fi

if [[ "$EFFORT" != "high" ]]; then
  printf '[CONFIG_ERROR]\tEFFORT must be high, got %s\n' "$EFFORT" >&2
  exit 2
fi

mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$ROOT/_codegen_logs/latest_${SEED}_run.txt"

find "$ROOT/$SEED" -path "*/codegen_prompts/*.md" -type f | sort -V > "$PROMPT_LIST"

while IFS= read -r p; do
  target="$(awk -F'`' '/^- 目标代码路径：`/ { print $2; exit }' "$p")"
  if [[ -z "$target" ]]; then
    printf '[BAD_PROMPT]\t%s\tmissing target path\n' "$p" >> "$RUN_DIR/status.tsv"
    continue
  fi
  if [[ ! -f "$target" ]]; then
    printf '%s\n' "$p"
  else
    printf '[SKIP_EXISTING]\t%s\t%s\n' "$p" "$target" >> "$RUN_DIR/status.tsv"
  fi
done < "$PROMPT_LIST" > "$MISSING_LIST"

printf '[START]\ttotal=%s\tmissing=%s\tparallel=%s\tmodel=%s\teffort=%s\ttimeout=%s\n' \
  "$(wc -l < "$PROMPT_LIST")" "$(wc -l < "$MISSING_LIST")" "$PARALLEL" "$MODEL" "$EFFORT" "$TIMEOUT_SECONDS" \
  | tee -a "$RUN_DIR/launcher.log"

export ROOT SEED RUN_DIR MODEL EFFORT TIMEOUT_SECONDS STOP_FILE

xargs -r -n1 -P "$PARALLEL" bash -c '
  set -u
  p="$1"
  target="$(awk -F'"'"'`'"'"' '"'"'/^- 目标代码路径：`/ { print $2; exit }'"'"' "$p")"
  rel="${p#"$ROOT"/}"
  safe="${rel//\//__}"
  safe="${safe%.md}"
  log="$RUN_DIR/${safe}.log"
  last="$RUN_DIR/${safe}.last.md"
  exit_file="$RUN_DIR/${safe}.exit"

  if [[ -f "$target" ]]; then
    printf "[SKIP_EXISTING]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
    exit 0
  fi
  if [[ -f "$STOP_FILE" ]]; then
    printf "[DEFER_USAGE_LIMIT]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
    exit 0
  fi

  printf "[START]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
  {
    printf "请严格执行下面的 ARC 新题代码生成提示词。必须把代码写入 prompt 指定的目标路径，并运行 prompt 里的自检命令。\n\n"
    printf "额外要求：不要在最终答复里粘贴完整代码、完整 diff 或长日志；最终只简要报告目标文件和自检结果。\n\n"
    cat "$p"
  } | timeout "$TIMEOUT_SECONDS"s codex exec \
      --skip-git-repo-check \
      --dangerously-bypass-approvals-and-sandbox \
      --ephemeral \
      --json \
      --color never \
      -C /home/lixinyu/snap \
      -m "$MODEL" \
      -c "reasoning.effort=\"${EFFORT}\"" \
      --output-last-message "$last" \
      - > "$log" 2>&1
  rc=$?
  printf "%s\n" "$rc" > "$exit_file"

  if [[ "$rc" -eq 0 && -f "$target" ]]; then
    printf "[OK]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
  elif grep -qi "usage limit" "$log"; then
    touch "$STOP_FILE"
    printf "[USAGE_LIMIT]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
  elif [[ "$rc" -eq 124 ]]; then
    printf "[TIMEOUT]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
  else
    printf "[FAILED:%s]\t%s\t%s\n" "$rc" "$p" "$target" >> "$RUN_DIR/status.tsv"
  fi
' bash < "$MISSING_LIST"

printf '[DONE]\tfinished_at=%s\n' "$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/launcher.log"
