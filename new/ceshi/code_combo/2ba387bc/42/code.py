# Combo variant of 2ba387bc (index 42)
# combo_spec: D5/1+D6/3+D9/1
# Name: 正方形输出+不对称分组+分隔列

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
for helper_dir in (ROOT / "arc2_solver_select", ROOT / "solvers"):
    if helper_dir.exists() and str(helper_dir) not in sys.path:
        sys.path.insert(0, str(helper_dir))

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location


"""
Design analysis for combo D5/1 + D6/3 + D9/1
=============================================

a) 具体化与行为追踪
- 原题输入：若干彼此分离的 4x4 物体散落在背景上。空心框架的 2x2 内部是背景色；
  实心方块则整块填满。`main()` 找出这些 4x4 对象后，分别按输入中的 `(row, col)`
  排序，再把框架放到左侧 4 列、实心放到右侧 4 列，输出宽度固定 8，高度是较长组的
  `4 * count`。
- 本融合把三个变化耦合起来：
  1. D5/1：左右两列之间插入 1 列背景色分隔，因此基础宽度从 8 变成 9，框架列仍在
     `cols 0..3`，分隔列是 `col 4`，实心列移到 `cols 5..8`。
  2. D6/3：输出不再只开 9x高，而是把这个“9 宽、两列顶部对齐”的布局嵌入左上角，
     再把整个画布补成正方形。边长 `side = max(9, 4 * max_group_count)`。
  3. D9/1：生成器强制一组只有 1 个，另一组有 4 或 5 个。于是较长列决定 `side`
     一定是 16 或 20；较短列只占最顶端 4 行，下面保留大片背景。
- 具体例子：
  若输入里按行排序后得到 `frames=[F1]`，`solids=[S1,S2,S3,S4]`。
  原题输出是 16x8：左列顶端放 `F1`，右列自上而下放 4 个实心。
  本变体先做成 16x9：`F1` 在 `(0,0)`，分隔列 `col 4` 全背景，`S1..S4` 在 `col 5`。
  然后 D6/3 检查 `max(9,16)=16`，最终输出 16x16，右侧额外 7 列全部背景。
  如果较长组有 5 个，则输出从 20x9 再补成 20x20。

b) 复杂度校准
- 三个变化都肉眼可见，而且自由度有限：
  不对称数量只取 `1 vs 4` 或 `1 vs 5`；
  分隔列固定 1 像素；
  正方形边长只会是 16 或 20。
- 如果只做 D5/1 而不做 D9/1，多数样本仍可能看起来像普通双列堆叠，只是中间空一列；
  如果只做 D6/3 而数量接近，正方形只表现为底部或右侧少量填充，辨识度不足。
  D9/1 让一列很短、一列很长，正好放大 D5/1 的分组效果，也让 D6/3 的“大量右侧空白”
  稳定出现。
- 颜色不承担新语义，仍只用于区分对象个体，避免把规律转移到不可见的颜色编号。

c) 隐式假设审查（对原 main() 的相关代码逐段检查）
- `bg = get_bg(g)`：仍然成立。生成器保持大面积均匀背景。
- `find_components(... monochromatic=False)`：仍然成立，但它依赖所有对象在 4 连通下彼此
  分离。生成器继续使用 `padding=1`，保证对象不接触、不合并。
- `bbox_h(b) != 4 or bbox_w(b) != 4: continue`：仍然成立。所有有效对象仍严格是 4x4。
- 原题把“不是框架”的 4x4 直接当实心。这里改成显式检测 `_is_solid(...)`，避免把意外
  图案静默归到实心组。这是新的 solver 假设：有效对象只能是“标准框架”或“标准实心”，
  生成器会保证这一点。
- `frames.sort(...)` / `solids.sort(...)`：仍然按输入 `(row,col)` 排序，不变。
- `outH = 4 * max(len(frames), len(solids))`：保留为“基础布局高度”，但不再是最终输出高；
  新逻辑先得到 `base_h = 4 * max_group`、`base_w = 9`，再取 `side = max(base_w, base_h)`。
- `outW = 8` 与 `paste(..., 4)` 的旧假设被 D5/1 破坏。新列起点必须是 `0` 和 `5`，并且
  `col 4` 留空。
- 原题默认“高度不足部分填充背景色”只发生在底部；本融合仍保持左上角锚定，因此 D6/3
  增加的正方形填充只出现在右侧和/或底部，不引入额外居中对齐语义。

d) 放置合法性
- 生成器所有对象都通过 `random_free_location(... border=1, padding=1, padding_connectivity=4)`
  放置，保证：
  1. 不重叠；
  2. 不越界；
  3. 不会与其他对象 4 连通接触。
- 框架用 `transparent=bg` 粘贴，保证 2x2 内部保持背景，不会被误填。
- 生成器额外要求所有对象的顶行两两不同，减少“同一行只靠列打破排序”的样本占比，
  让人类更容易看出组内是按从上到下排序。

e) 视觉可观察性
- 人类在 2-4 个样例中能同时看到三件事：
  1. 输出总是正方形；
  2. 两组之间总有一列背景色分隔；
  3. 一列很短、另一列很长，且短列只占顶部 4 行。
- 生成器随机决定哪一组是 minority（框架或实心），避免出现“左列总短”或“右列总短”
  这种可被错误捷径解释的偏差。
- 还会拒绝输入中“所有框架都在所有实心上方”或反过来的样本，避免空间分布本身伪装成
  分类规则。

f) 可视化整洁度
- 输入继续是干净的离散 4x4 组件，没有额外标记或重叠层。
- 输出仍是规则网格堆叠，只新增固定 1 列分隔和右/下背景补齐，不会造成局部拥挤。

g) 变化的可推断性
- 本融合没有颜色映射或重着色；每个对象在输入输出中保持原色。
- 规则完全依赖可见的结构线索：空心/实心形状、纵向先后顺序、列位置、分隔列、正方形边界。

h) 退化审查与量化指标
- 若生成器没真正触发融合，可能退化为：
  1. 两组数量接近，D9/1 视觉上不明显；
  2. 输出恰好不是靠较长列决定边长，D6/3 不显著；
  3. 去掉分隔列后布局几乎无差别。
- 因此样本必须满足：
  `minority_count == 1`，`majority_count in {4,5}`，`side in {16,20}`，且
  分隔列 `col 4` 全背景、右侧存在至少 7 或 11 列纯背景补白。
  这些指标会在测试后统计。

i) 新 solver 的隐式假设与生成器保障
- 新 solver 假设所有有效对象都能被 `_is_frame` 或 `_is_solid` 准确识别；
  生成器只构造这两类。
- 新 solver 假设至少存在 1 个有效对象；生成器始终生成 5 或 6 个。
- 新 solver 假设按 `(row,col)` 排序足以确定稳定堆叠顺序；生成器保证对象互不接触且
  顶行不同，使这一顺序清晰可读。
"""


def _is_frame(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    return len(set(map(int, border.flatten()))) == 1 and bool((sub[1:3, 1:3] == int(bg)).all())


def _is_solid(sub: np.ndarray, bg: int) -> bool:
    if sub.shape != (4, 4):
        return False
    vals = set(map(int, sub.flatten()))
    return len(vals) == 1 and int(next(iter(vals))) != int(bg)


def _extract_objects(
    grid: np.ndarray,
) -> tuple[int, list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")

    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = map(int, b)
        sub = crop_bbox(g, b)
        if _is_frame(sub, bg):
            frames.append((int(r0), int(c0), sub))
        elif _is_solid(sub, bg):
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return int(bg), frames, solids


def _assemble_seed_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    max_group = max(1, int(len(frames)), int(len(solids)))
    out = np.full((4 * max_group, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def _assemble_combo_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    max_group = max(1, int(len(frames)), int(len(solids)))
    base_h = 4 * max_group
    base_w = 9
    side = max(base_h, base_w)
    out = np.full((side, side), int(bg), dtype=int)

    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 5)

    return out


def _output_metrics(
    grid: np.ndarray,
) -> dict[str, int]:
    bg, frames, solids = _extract_objects(np.array(grid, dtype=int))
    combo = _assemble_combo_output(bg, frames, solids)
    max_group = max(int(len(frames)), int(len(solids)))
    minority = min(int(len(frames)), int(len(solids)))
    return {
        "n_frames": int(len(frames)),
        "n_solids": int(len(solids)),
        "minority_count": int(minority),
        "majority_count": int(max_group),
        "side": int(combo.shape[0]),
        "separator_bg_cells": int(np.sum(combo[:, 4] == int(bg))),
        "right_padding_cols": int(combo.shape[1] - 9),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 frames/solids into two separated columns, then square-pad.

    - Frames go to cols 0..3.
    - Col 4 is a background separator.
    - Solids go to cols 5..8.
    - Both groups keep the original top-to-bottom ordering from the input.
    - The whole result is padded to a square with side max(9, 4 * max_group_count).
    """
    bg, frames, solids = _extract_objects(np.array(grid, dtype=int))
    return _assemble_combo_output(bg, frames, solids)


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    arr = np.full((4, 4), int(col), dtype=int)
    arr[1:3, 1:3] = int(bg)
    return arr


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _choose_counts() -> tuple[int, int]:
    majority = int(np.random.choice([4, 5]))
    if bool(np.random.randint(0, 2)):
        return 1, majority
    return majority, 1


def _place_with_unique_row(
    grid: np.ndarray,
    patch: np.ndarray,
    bg: int,
    used_rows: set[int],
) -> tuple[int, int]:
    for _ in range(120):
        r0, c0 = random_free_location(
            grid,
            patch,
            transparent=bg,
            border=1,
            padding=1,
            padding_connectivity=4,
        )
        if int(r0) not in used_rows:
            return int(r0), int(c0)
    raise ValueError("failed to place object on a fresh row")


def _rows_are_interleaved(frames: list[tuple[int, int, np.ndarray]], solids: list[tuple[int, int, np.ndarray]]) -> bool:
    if not frames or not solids:
        return False
    frame_rows = [int(r0) for r0, _c0, _sub in frames]
    solid_rows = [int(r0) for r0, _c0, _sub in solids]
    return (
        min(frame_rows) < max(solid_rows) and min(solid_rows) < max(frame_rows)
    )


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n_frames, n_solids = _choose_counts()
    total = int(n_frames + n_solids)
    h = int(np.random.randint(28, 38))
    w = int(np.random.randint(26, 38))
    g = np.full((h, w), int(bg), dtype=int)

    colors = list(map(int, np.random.permutation(np.arange(1, 10))[:total]))
    objects: list[tuple[str, np.ndarray]] = []
    for i in range(n_frames):
        objects.append(("frame", _make_frame(colors[i], bg)))
    for i in range(n_solids):
        objects.append(("solid", _make_solid(colors[n_frames + i])))
    np.random.shuffle(objects)

    used_rows: set[int] = set()
    try:
        for kind, patch in objects:
            _ = kind
            r0, c0 = _place_with_unique_row(g, patch, bg, used_rows)
            paste(g, patch, int(r0), int(c0), transparent=bg)
            used_rows.add(int(r0))

        parsed_bg, frames, solids = _extract_objects(g)
        if parsed_bg != bg:
            return None
        if len(frames) != n_frames or len(solids) != n_solids:
            return None
        if not _rows_are_interleaved(frames, solids):
            return None

        combo = _assemble_combo_output(parsed_bg, frames, solids)
        seed = _assemble_seed_output(parsed_bg, frames, solids)
        if combo.shape[0] != combo.shape[1]:
            return None
        if combo.shape[1] < 9:
            return None
        if not bool((combo[:, 4] == int(parsed_bg)).all()):
            return None
        if max(len(frames), len(solids)) not in {4, 5}:
            return None
        if min(len(frames), len(solids)) != 1:
            return None
        if combo.shape[0] not in {16, 20}:
            return None
        if combo.shape == seed.shape and np.array_equal(combo, seed):
            return None
        if not np.array_equal(main(g), combo):
            return None
        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
