#!/usr/bin/env bash
set -u

LIST=/tmp/fable10_001_020_missing_prompts.txt
ROOT=/home/lixinyu/snap/a/key_combo/code/fable/10
RUN_LOG="$ROOT/codegen_001_020_run.log"

python - <<'PY'
from pathlib import Path

missing = []
for i in range(1, 21):
    root = Path(f"/home/lixinyu/snap/a/key_combo/code/fable/10/{i:03d}")
    for prompt in sorted((root / "codegen_prompts").glob("*.md"), key=lambda p: int(p.stem)):
        target = root / f"{prompt.stem}.py"
        if not target.exists():
            missing.append(prompt)

Path("/tmp/fable10_001_020_missing_prompts.txt").write_text(
    "\n".join(str(p) for p in missing) + ("\n" if missing else "")
)
print(len(missing))
PY

echo "RUN_START $(date -Is)" >> "$RUN_LOG"

xargs -P 5 -n 1 bash -c '
  p="$1"
  base="${p%/codegen_prompts/*}"
  n="$(basename "$p" .md)"
  target="$base/$n.py"
  mkdir -p "$base/codegen_logs"
  log="$base/codegen_logs/$n.log"

  if [ -s "$target" ]; then
    echo "SKIP existing $(date -Is) $p" >> "$log"
    exit 0
  fi

  echo "START $(date -Is) $p" > "$log"
  inst="请严格执行下面的 ARC 新题代码生成提示词。必须把代码写入 prompt 指定的目标路径，并运行 prompt 里的自检命令。生成代码前必须把 \`/home/lixinyu/snap/new/solvers/arc_utils.py\` 作为公共参考工具箱快速阅读，并阅读 prompt 中各参考题的 \`<generation_spec>\`；spec 只用于借鉴输入生成策略、随机维度、分支覆盖和拒绝条件，不能覆盖新题规则。若直接 import 工具箱，必须保证目标文件可被 \`SourceFileLoader\` 直接加载。

执行每个代码生成 prompt 时，还必须遵守以下语义一致性硬约束：
1. 输出必须唯一可判定。输入中的线索必须足以唯一确定目标对象、目标操作和结果。不允许依赖含糊解释，或让同一可见线索在不同语境下产生不同答案。注意不可多个不同样例的输入网格处理后的输出网格都相同。
2. 组合步骤必须不可替代。题目描述中的关键步骤都应对结果产生必要影响，不能让某一步只是表面存在，实际可被更简单的捷径完全替代。
3. 实现必须遵守题面定义的参照系。所有判断、匹配、距离、对称、排序、分区等，都必须使用题面指定的对象范围和坐标参照，不得为了实现方便换成另一套参照系。
4. 生成样例要覆盖反例。生成器要主动产生能排除错误捷径的样例，不只生成最容易通过的标准形态，还要覆盖边界情况、分支情况和易混淆情况。
5. 验证要检查语义而不只是自洽。不能只验证 main(generate_input()) 能闭环，还要检查实现是否真的满足题面规则、关键依赖、唯一性和非退化性。"

  if codex exec --skip-git-repo-check -C /home/lixinyu/snap/a -c model="\"gpt-5.4\"" -c reasoning.effort="\"high\"" "$inst

$(cat "$p")" >> "$log" 2>&1; then
    echo "DONE $(date -Is) $p" >> "$log"
  else
    rc=$?
    echo "FAIL rc=$rc $(date -Is) $p" >> "$log"
    exit "$rc"
  fi
' _ < "$LIST"

rc=$?
echo "RUN_END rc=$rc $(date -Is)" >> "$RUN_LOG"
exit "$rc"
