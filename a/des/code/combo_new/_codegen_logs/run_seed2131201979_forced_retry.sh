#!/usr/bin/env bash
set -u

ROOT="/home/lixinyu/snap/a/des/code/combo_new"
SEED="seed2131201979"
ORIGINAL_RUN="${ORIGINAL_RUN:-$(cat "$ROOT/_codegen_logs/latest_${SEED}_run.txt")}"
RUN_DIR="$ROOT/_codegen_logs/run_$(date +%Y%m%d_%H%M%S)_${SEED}_forced_retry"
RETRY_TSV="$RUN_DIR/retry.tsv"
PARALLEL="${PARALLEL:-4}"
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
printf '%s\n' "$RUN_DIR" > "$ROOT/_codegen_logs/latest_${SEED}_forced_retry_run.txt"

awk -F'\t' '$1=="[SKIP_EXISTING]" || $1=="[TIMEOUT]" || $1 ~ /^\[FAILED:/ {print $2 "\t" $3}' \
  "$ORIGINAL_RUN/status.tsv" | sort -V > "$RETRY_TSV"

printf '[START]\toriginal=%s\tretry=%s\tparallel=%s\tmodel=%s\teffort=%s\ttimeout=%s\n' \
  "$ORIGINAL_RUN" "$(wc -l < "$RETRY_TSV")" "$PARALLEL" "$MODEL" "$EFFORT" "$TIMEOUT_SECONDS" \
  | tee -a "$RUN_DIR/launcher.log"

export ROOT SEED RUN_DIR MODEL EFFORT TIMEOUT_SECONDS

xargs -r -n2 -P "$PARALLEL" bash -c '
  set -u
  p="$1"
  target="$2"
  rel="${p#"$ROOT"/}"
  safe="${rel//\//__}"
  safe="${safe%.md}"
  log="$RUN_DIR/${safe}.log"
  last="$RUN_DIR/${safe}.last.md"
  exit_file="$RUN_DIR/${safe}.exit"

  case "$target" in
    "$ROOT/$SEED"/*/*.py) ;;
    *)
      printf "[BAD_TARGET]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
      exit 2
      ;;
  esac

  mkdir -p "$(dirname "$target")"
  if [[ -f "$target" ]]; then
    backup="$RUN_DIR/before_retry/${target#"$ROOT/$SEED"/}"
    mkdir -p "$(dirname "$backup")"
    cp -p "$target" "$backup"
    rm -f "$target"
  fi

  printf "[START]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
  {
    printf "请严格执行下面的 ARC 新题代码生成提示词。必须把代码写入 prompt 指定的目标路径，并运行 prompt 里的自检命令。\n\n"
    printf "额外强制要求：即使目标文件之前存在，也必须重新生成并覆盖该目标文件；只能写 prompt 指定的这个目标代码路径，不要创建、修改或删除其它编号目录里的 .py 文件。\n\n"
    printf "最终答复只简要报告目标文件和自检结果，不要粘贴完整代码、完整 diff 或长日志。\n\n"
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
    printf "[USAGE_LIMIT]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
  elif [[ "$rc" -eq 124 ]]; then
    printf "[TIMEOUT]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
  else
    printf "[FAILED:%s]\t%s\t%s\n" "$rc" "$p" "$target" >> "$RUN_DIR/status.tsv"
  fi
' bash < "$RETRY_TSV"

printf '[DONE]\tfinished_at=%s\n' "$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/launcher.log"
