#!/usr/bin/env bash
set -u

ROOT="/home/lixinyu/snap/a/des/code/eval_new"
RUN_DIR="$ROOT/_codegen_logs/run_$(date +%Y%m%d_%H%M%S)_requested"
PROMPT_LIST="$RUN_DIR/prompts.txt"
MISSING_LIST="$RUN_DIR/missing_prompts.txt"
PARALLEL="${PARALLEL:-2}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-2400}"
MODEL="${MODEL:-gpt-5.4}"
EFFORT="${EFFORT:-high}"
TRIM_REFERENCES="${TRIM_REFERENCES:-1}"
MAX_MISSING="${MAX_MISSING:-0}"

mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > "$ROOT/_codegen_logs/latest_requested_run.txt"

find "$ROOT" \
  \( -path "$ROOT/seed558458950/*/codegen_prompts/*.md" \
  -o -path "$ROOT/seed1967842053/*/codegen_prompts/*.md" \
  -o -path "$ROOT/seed1572225882/*/codegen_prompts/*.md" \
  -o -path "$ROOT/seed202524619/*/codegen_prompts/*.md" \) \
  -type f | sort -V > "$PROMPT_LIST"

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

if [[ "$MAX_MISSING" =~ ^[0-9]+$ && "$MAX_MISSING" -gt 0 ]]; then
  head -n "$MAX_MISSING" "$MISSING_LIST" > "$MISSING_LIST.tmp"
  mv "$MISSING_LIST.tmp" "$MISSING_LIST"
fi

printf '[START]\ttotal=%s\tmissing=%s\tparallel=%s\tmodel=%s\teffort=%s\ttimeout=%s\tmax_missing=%s\n' \
  "$(wc -l < "$PROMPT_LIST")" "$(wc -l < "$MISSING_LIST")" "$PARALLEL" "$MODEL" "$EFFORT" "$TIMEOUT_SECONDS" "$MAX_MISSING" \
  | tee -a "$RUN_DIR/launcher.log"

printf '[PROMPT_MODE]\ttrim_references=%s\n' "$TRIM_REFERENCES" | tee -a "$RUN_DIR/launcher.log"

export ROOT RUN_DIR MODEL EFFORT TIMEOUT_SECONDS TRIM_REFERENCES

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

  printf "[START]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
  {
    printf "请严格执行下面的 ARC 新题代码生成提示词。必须把代码写入 prompt 指定的目标路径，并运行 prompt 里的自检命令。\n\n"
    printf "额外要求：这是批量自动化任务，速度优先。不要在最终答复里粘贴完整代码、完整 diff 或长日志。不要读取任何本地参考代码、annotation、prompt.md、codegen_prompts、similarity_reports、目录列表或其它文件；当前 stdin 已包含目标路径、新题描述和必要参考摘要。只允许执行两类 shell 操作：1) 写入/修改目标代码文件；2) 运行 prompt 中的自检和 py_compile。请直接实现目标文件并自检，最终只简要报告目标文件和自检结果。\n\n"
    if [[ "${TRIM_REFERENCES:-1}" == "1" ]]; then
      awk '"'"'
        /^## 机制相似参考题/ { skip = 1; next }
        /^## 实现要求/ { skip = 0 }
        !skip { print }
      '"'"' "$p"
    else
      cat "$p"
    fi
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
  elif [[ "$rc" -eq 124 ]]; then
    printf "[TIMEOUT]\t%s\t%s\n" "$p" "$target" >> "$RUN_DIR/status.tsv"
  else
    printf "[FAILED:%s]\t%s\t%s\n" "$rc" "$p" "$target" >> "$RUN_DIR/status.tsv"
  fi
' bash < "$MISSING_LIST"

printf '[DONE]\tfinished_at=%s\n' "$(date --iso-8601=seconds)" | tee -a "$RUN_DIR/launcher.log"
