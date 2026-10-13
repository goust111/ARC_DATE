你正在为 ARC-AGI-2 任务 `{task_id}` 编写一个 `generate_input()` 函数。

## 你的输入

1. **规格说明**（经人工审核）：
```yaml
{spec_yaml}
```

2. **求解器代码**：阅读 `{solver_path}` —— 包含 `main(grid)` 函数。你的 `generate_input()` 将被追加到该文件末尾。

3. **评估输入/输出数据**：阅读 `{eval_path}` —— 用于理解未指定细节的参考示例（如有必要），但通常很少需要。

4. **`arc_utils` 中的工具函数** —— **请使用这些函数，不要自行重新实现**：

**放置类**（用于在网格上放置精灵图）：
- `paste(dst, patch, r0, c0, *, transparent=None, mask=None, overwrite=True, clip=False)` —— 将 patch 放置在 (r0,c0) 位置
- `random_free_location(canvas, patch, *, transparent, border=0, padding=0, padding_connectivity=8) -> (r,c)` —— 随机找一个不重叠的位置；找不到时抛出 ValueError
- `first_free_location(canvas, patch, *, transparent, border=0, padding=0, padding_connectivity=8) -> (r,c)|None`

**连通分量类**：
- `find_components(grid, *, bg=None, connectivity=4, monochromatic=True, return_format="dict") -> list[dict]` —— 每个字典包含键：color, coords, bbox, area
- `components_mask(mask, connectivity=4) -> list[np.ndarray]` —— 从布尔掩码中得到 (N,2) 坐标数组列表
- `label_mask(mask, connectivity=4) -> (labeled_array, num_components)`

**颜色与背景类**：
- `get_bg(grid) -> int` —— 最常见的颜色
- `colors_present(grid, *, exclude=None) -> list[int]`

**边界框类**（`BBox = (r0, c0, r1, c1)` 包含端点）：
- `bbox(mask) -> BBox|None`、`bbox_coords(coords) -> BBox`、`crop_bbox(grid, bbox) -> np.ndarray`

**几何类**：
- `draw_line(grid, r, c, *, end_r=None, end_c=None, length=None, direction=None, color, stop_at=None, clip=False) -> (r,c)`
- `flood_fill(grid, r, c, color, *, connectivity=4)` —— 原地操作
- `is_contiguous(bitmask, background=0, connectivity=4) -> bool`

完整源代码：`{arc_utils_path}`

## 预检查：在评估数据上验证求解器

在编写任何代码之前，先验证求解器在所有评估示例上是否正确运行：

```bash
cat > /tmp/pre_check.py << 'EOF'
import json, numpy as np
from solvers.arc2 import load_pid
mod = load_pid('{task_id}')
data = json.load(open('{eval_path}'))
for split in ('train', 'test'):
    for i, ex in enumerate(data[split]):
        inp = np.array(ex['input'], dtype=int)
        expected = np.array(ex['output'], dtype=int)
        actual = mod.main(inp)
        ok = np.array_equal(actual, expected)
        print(f'{split}[{i}]: {"OK" if ok else "FAIL"}')
        if not ok:
            raise SystemExit(f'Solver fails on {split}[{i}] — aborting')
print('All eval examples pass.')
EOF
PYTHONPATH=/home/xd/projects/BARC /home/xd/miniconda3/envs/tune/bin/python /tmp/pre_check.py
```

**如果任何示例失败，请立即停止。不要尝试修复求解器或其他文件。报告失败并退出。**

## 你的任务

编写一个 `generate_input() -> np.ndarray` 函数，遵循规格说明中的 `generation_strategy`。

## 要求

1. **忠实遵循规格说明**。规格说明中的 `generation_strategy` 描述了方法 —— 照此实现，不要重新设计。
2. **构造式，而非拒绝式**：生成逻辑必须直接构造满足约束的对象，而不是随机生成候选者再筛选。例如，如果一个精灵图需要 2 像素的接触边，就从这 2 个像素开始生长 —— 不要随机生成精灵图并期望某个恰好匹配。唯一可接受的拒绝循环是外层的 `_gen_attempt` 重试 + 最终的 `main(inp) == out` 验证检查。任何内部循环如果拒绝率超过 50%，说明方法有问题 —— 应重新构造以直接生成所需输出。
3. **"输出优先"在代码结构中也意味着输出优先**：当规格说明中的 generation_strategy 说用"输出优先"方法时，你的代码必须按字面意思遵循 generation_strategy 的步骤，先构建完整输出（通常是约束更多的结构），然后通过逆转 `main()` 中描述的求解器变换来推导输入。最后用 `np.array_equal(main(input), output)` 进行验证。如果你的代码先放置输入对象再期望求解器通过随机试验产生良好输出，那就做错了 —— 这种方式的接受率往往接近于零。
4. **阅读求解器代码**以理解精确约束、平局处理逻辑和边界情况（如有必要）。
5. **尽可能使用 `arc_utils` 工具函数**。
6. **代码结构**：拆分为 `_gen_attempt() -> np.ndarray | None`（单次尝试，成功返回网格，失败返回 None）和 `generate_input()`（重试循环，最多 300 次尝试）。这样可以保持生成逻辑简洁可读。控制难度或场景类型的配置参数（如网格大小、对象数量、是否可达/卡住、变换变体）必须在 `generate_input()` 中只采样一次，然后作为参数传递给 `_gen_attempt()`。不要在 `_gen_attempt()` 内部重新采样——如果每次重试都重新抽签，容易成功的配置会在输出中占绝大多数，而难生成的配置几乎不会出现。
7. **不要修改求解器文件中任何已有的代码或注释**。将你的函数追加到文件末尾。
8. **规格说明符合性检查**：在测试之前，逐行审查规格说明中的每一个属性和约束（`spec:` 和 `generation_strategy:` 两个部分）。对每一条，找到实现它的确切代码位置。如果某个规格说明条目没有对应的代码，先实现它再进行测试。需要特别注意：
   - 条件变换规则（规格说明中的 if-then 分支）
   - 防止求解器歧义的约束（如匹配的唯一性）
   - 验证检查 `main(input)==output` 无法检测为缺失的特性（如某个变换变体是合法的但从未被生成）

## 测试

编写函数后进行测试。**重要提示**：始终将测试/审计脚本写入临时文件再运行，不要使用带有多行代码的 `python -c "..."`（会触发沙箱权限提示从而阻塞执行）。

```bash
cat > /tmp/test_gen.py << 'EOF'
import numpy as np, time
from solvers.arc2 import load_pid
mod = load_pid('{task_id}')
main, generate_input = mod.main, mod.generate_input
ok, fail = 0, 0
t0 = time.time()
for i in range(200):
    try:
        inp = generate_input()
        out = main(inp)
        ok += 1
    except Exception as e:
        fail += 1
        if fail <= 5: print(f'Trial {i}: {e}')
elapsed = time.time() - t0
print(f'Success: {ok}/200, Time: {elapsed:.1f}s ({elapsed/200*1000:.0f}ms/call)')
EOF
PYTHONPATH=/home/xd/projects/BARC /home/xd/miniconda3/envs/tune/bin/python /tmp/test_gen.py
```

如果成功率低于 90%，请诊断失败原因并修复代码。反复测试直到达到 ≥ 90%。

### 质量审计

在成功率达到 ≥ 90% 后，对 100 个生成样本进行规格说明符合性审计：

1. **范围约束**（如 `count: 2-5`、`size: 3-7 pixels`）：测量 100 个样本中的实际分布。计算生成分布与规格说明范围上**均匀分布**之间的 **Jensen-Shannon 距离**（`scipy.spatial.distance.jensenshannon(p, q, base=2)`）。**标记**任何 JS 距离 > 0.5 的属性。目标是**多样性** —— 生成数据应探索完整的规格说明范围，而非聚集在某个值上。
2. **绝对约束**（如 `count: 3`、`单一连通分量`、`无重叠`）：从规格说明中提取所有非 `[observed]` 的绝对约束，验证 **100% 的样本**都满足每一条。标记任何违规。这些是硬性要求 —— 即使一个违规也意味着生成器有 bug。
3. **视觉自检（必须执行，不可跳过）**：保存一张 PNG 图片，将 4-5 个生成的输入-输出对与所有评估示例进行对比（使用 matplotlib imshow，颜色映射为 `['#000','#0074D9','#FF4136','#2ECC40','#FFDC00','#AAA','#F012BE','#FF851B','#7FDBFF','#870C25']` 对应值 0-9，评估数据对在上方行，生成数据对在下方行）。查看图像并检查形状、位置、空间关系等 —— 生成数据应看起来像属于与评估数据同一任务。这能捕获程序化检查遗漏的问题。尽可能修改代码以减少显著的视觉差异，并报告任何剩余问题。
4. 打印汇总表（属性、类型、结果、标记状态）作为**自检报告**。
5. **变换变体覆盖率**：如果规格说明描述了多种变换路径（如"仅旋转" vs "翻转 + 颜色交换"，或"水平" vs "垂直"），统计生成样本中每种路径的出现比例。标记任何覆盖率为 0% 的路径 —— 这通常意味着生成器代码完全缺失该分支。每种变体应至少出现在 10% 的样本中，除非规格说明将其标记为罕见情况。

### 处理审计失败

当某个属性被标记（JS > 0.5）时，判断属于以下哪种情况：

- **代码缺陷**：代码未能覆盖合理的规格说明范围 → 修复代码。
- **规格说明错误**：规格说明范围与任务的结构约束矛盾（如对称性排除了偶数维度）→ 修正或删除该规格说明条目。
- **规格说明过度约束**：该约束与任务核心逻辑无关，且满足它会增加不成比例的代码复杂度 → 修正或删除该规格说明条目。

规格说明条目遵循奥卡姆剃刀原则：只保留对求解器正确性或核心逻辑覆盖必要的约束。无法证明其存在合理性的约束应被删除，而非修补。

**`[observed]` 约束**：标记为 `[observed]` 的规格说明条目是在评估数据中观察到但求解器并不要求的模式。如果成本低（如向已有的辅助函数传递 `symmetry` 参数），可以满足它们，但不要编写大量额外代码（> ~50 行）来强制满足。它们不应被审计 —— 在 JS 距离检查中排除它们。

修改规格说明后：输出规格说明的差异及理由，按更新后的规格说明重新审计，反复迭代直到所有属性通过。

## 常见提示/陷阱

- 如果求解器通过频率推断颜色（如背景色 = 最常见颜色），请确保生成的网格保持相同的频率排序。
- 避免存在多个有效解的场景 —— 求解器的启发式方法可能选择与你预期输出不同的解。
- 如果求解器假设每个对象都参与变换，请验证所有生成的对象都是可达/活跃的。

## 输出

将 `generate_input()`（及其所需的任何辅助函数）追加到 `{solver_path}`。
