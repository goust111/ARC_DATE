# Combo variant of 65b59efc (index 16)
# combo_spec: D4/1+D18/0
# Name: 填充后裁剪输出

"""
design_analysis:
  fusion_resolution:
    - D4/1 改的是输出版式：原题先在隐藏的 `N x N` 块画布上铺满所有可能槽位，再直接输出整幅
      `(N*N) x (N*N)` 大图；现在只能输出“实际被 marker 触发的块”的最小外接矩形。
    - D18/0 改的是空槽内容：原题未被触发的块保持全局背景；现在这些块若落在裁剪矩形内部，
      必须显示成固定默认图案。
    - 两者必须一起实现，顺序是：先从 marker 恢复哪些块被盖章，再用这些块求裁剪 bbox，
      然后只在 bbox 内构造输出，并把未盖章块填成默认图案。若先在整幅隐藏画布上把所有空块都
      填默认图案，再按非背景像素紧裁剪，就会因为默认图案遍布全画布而完全失效。
  concretization_and_behavior_trace:
    - 例：`N=3`，marker 最终命中块槽位 `(0,0)`, `(0,2)`, `(1,0)`。
    - 原题：
      1. 生成 `9x9` 输出。
      2. 这三个 `3x3` 块贴模板，其余六块全是背景色。
    - 融合后：
      1. 先看命中块的块级 bbox，为行 `0..1`、列 `0..2`，所以只保留 `2 x 3` 个块。
      2. 在这个 `6x9` 输出里，三块命中位置贴模板，另外三块空位填成默认图案。
      3. 被裁掉的是隐藏画布上 bbox 外侧的块，而不是“所有默认图案块”。
    - 差异点：
      1. 输出尺寸必须由 marker 槽位集合决定，不能固定写死为 `(N*N, N*N)`。
      2. bbox 内部的空块不再留黑，而要明显区别于盖章块。
  complexity_calibration:
    - 输入端仍保留原题的三带结构、模板色选模板、局部坐标选槽位，不额外引入第二套图例。
    - 默认图案固定为“整块分隔色 `sep`”，取值少、辨识强，也能借输入中的分隔线给人类提示。
    - 生成器强制：
      1. 裁剪 bbox 至少在一个维度上小于 `N`，避免退化回整幅输出。
      2. bbox 内至少留一个未盖章块，确保 D18/0 在样例里稳定可见。
      3. 每个模板至少被使用一次，避免无支撑颜色。
  implicit_assumptions_review_main:
    - `sep_cols[0] == N` 仍依赖首条竖向分隔线在第 `N` 列；生成器保留原版式。
    - 顶带每列模板仍假设“单一模板色 + 背景”组成二值 mask；生成器保持这一点，并拒绝全满模板。
    - 底带仍假设每个模板列只出现一种 palette 输出色；生成器继续只放一个 swatch 颜色。
    - 中带仍假设每个 tile 列只使用一种 marker/template 色；生成器继续做模板到 tile 列的 1:1 指派。
    - 新假设 1：至少存在一个 marker；生成器总会放置不少于 `K` 个 marker。
    - 新假设 2：裁剪依据是“命中槽位集合”而非像素前景；生成器保证 bbox 外没有 marker。
    - 新假设 3：默认图案与模板印章可区分；生成器保证 palette 颜色不等于 `sep`。
  placement_legality:
    - 继续用 output-first：先选块级 bbox，再在其中挑若干唯一槽位放 marker，随后反推输入中带像素。
    - 同一 `(lr, lc)` 槽位至多属于一个模板，避免不同模板抢同一输出块。
    - bbox 选在 `0..N-1` 的块坐标范围内，不越界；模板、marker、palette、分隔线仍分层放置且互不重叠。
  visual_observability:
    - 人类能直接看到：输出不再总是方阵；同时裁剪后矩形内部仍保留若干整块灰色区域。
    - 因为灰色块颜色与输入分隔线一致，且模板块使用其他 palette 颜色，空块/印章块的角色差异清楚。
    - 多组样例会重复出现“有 marker 的最外层决定输出边界，边界内空位也保留”的现象，可从肉眼归纳。
  visual_cleanliness_and_inferability:
    - 默认图案采用整块单色 `sep`，避免棋盘格这类更高频纹理抢过模板本身。
    - 输出中每个 `N x N` 块只属于“印章块”或“默认块”之一，没有叠色覆盖。
    - 颜色种类仍受原题约束，主要新增的只是输出里更多 `sep` 块，不会把规则藏进不可见数值排序。
  non_degeneracy_checks_used_in_generator:
    - 生成器拒绝 `bbox_h == N and bbox_w == N` 的样本。
    - 生成器拒绝 `mark_count == bbox_area` 的样本，保证裁剪矩形内部有默认块。
    - 生成器用 oracle 校验 `main(input) == expected`。
  new_main_assumptions_to_guarantee:
    - `main()` 新增了“根据命中块集合求 bbox”的循环；生成器保证边界行列上至少有 marker，
      这样裁剪矩形确实对应设计时采样的 bbox。
    - `main()` 用 `sep` 整块填默认图案；生成器保证 `sep` 不是 palette 颜色，默认块不会伪装成印章块。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste, random_sprite  # noqa: E402


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the tiled stamp legend, but render only the occupied block window and fill its unused blocks with the separator color.

    Input structure:
    - The grid is still partitioned into K tile columns of width N, separated by dashed 1-cell separator columns.
    - The top band stores K binary template masks, each drawn in a unique template color.
    - The middle band stores marker pixels. A marker's color still chooses the template, and its local `(lr, lc)` position inside a tile column still names a block slot in a hidden `N x N` output tiling.
    - The bottom band still stores one palette swatch per template column, and the two dashed separator rows still split the three bands.

    Rule:
    - Recover templates, palette colors, and all occupied block slots exactly as in the base task.
    - The occupied slots define which blocks receive stamps and also define the smallest block-aligned rectangle that remains visible in the output.
    - Inside that cropped rectangle, stamped blocks use the recolored template; every unstamped block is filled with a fixed default tile made entirely of the separator color.

    Key insight: marker color and local marker coordinates still decode template choice and block position, but the final canvas is cropped by the set of occupied slots, not by rendered foreground pixels, because empty blocks now carry their own visible default pattern.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flat))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert sep_cols and sep_rows

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols if c + 1 + n <= w]
    assert starts

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        mask = sub == col
        assert mask.any() and (~mask).any()
        tcols.append(col)
        tmasks.append(mask)
    assert len(set(tcols)) == len(tcols)
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    mid_r0 = n + 1
    slot_owner: dict[tuple[int, int], int] = {}
    for st in starts:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                j = tcol_to_j[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j, "different templates must occupy different slots"
                slot_owner[(lr, lc)] = j

    assert slot_owner
    rows = [rr for rr, _cc in slot_owner]
    cols = [cc for _rr, cc in slot_owner]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    out = np.full(((r1 - r0 + 1) * n, (c1 - c0 + 1) * n), sep, dtype=int)
    for (rr, cc), j in slot_owner.items():
        tile = np.where(tmasks[j], pcols[j], bg)
        paste(out, tile, (rr - r0) * n, (cc - c0) * n)

    return out


# --------------- generator (combo D4/1 + D18/0) ---------------

_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_BBOX_COVERAGE = 0.55
_MAX_RETRIES = 400


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _sample_template(n: int, color: int) -> np.ndarray:
    for _ in range(80):
        sprite = random_sprite(
            n,
            n,
            density=random.uniform(*_DENSITY_RANGE),
            symmetry=random.choice(_SYMMETRY_OPTIONS),
            color_palette=[color],
            connectivity=_TEMPLATE_CONNECTIVITY,
            background=_BG,
        )
        mask = sprite == color
        if mask.any() and (~mask).any():
            return mask
    raise RuntimeError("failed to sample non-degenerate template")


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _sample_active_bbox(n: int, k: int) -> tuple[int, int, int, int] | None:
    candidates = [
        (bh, bw)
        for bh in range(2, n + 1)
        for bw in range(2, n + 1)
        if bh * bw >= k + 1 and not (bh == n and bw == n)
    ]
    if not candidates:
        return None
    bh, bw = random.choice(candidates)
    top = random.randint(0, n - bh)
    left = random.randint(0, n - bw)
    return top, left, bh, bw


def _build_slot_set(
    top: int,
    left: int,
    bh: int,
    bw: int,
    k: int,
) -> list[tuple[int, int]] | None:
    area = bh * bw
    max_marks = area - 1
    min_marks = max(k, 3, int(np.ceil(_MIN_BBOX_COVERAGE * area)))
    if min_marks > max_marks:
        return None

    mark_count = random.randint(min_marks, max_marks)

    seed = [(top, left), (top, left + bw - 1), (top + bh - 1, left)]
    seed = list(dict.fromkeys(seed))
    if len(seed) > mark_count:
        return None

    cells = [(top + dr, left + dc) for dr in range(bh) for dc in range(bw)]
    remaining = [cell for cell in cells if cell not in seed]
    random.shuffle(remaining)
    slots = seed + remaining[: mark_count - len(seed)]

    rows = [r for r, _c in slots]
    cols = [c for _r, c in slots]
    if min(rows) != top or max(rows) != top + bh - 1 or min(cols) != left or max(cols) != left + bw - 1:
        return None
    if len(slots) == area:
        return None
    return slots


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    rows = [rr for _j, rr, _cc in placements]
    cols = [cc for _j, _rr, cc in placements]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    expected = np.full(((r1 - r0 + 1) * n, (c1 - c0 + 1) * n), _SEP, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], pcols[j], _BG)
        paste(expected, tile, (rr - r0) * n, (cc - c0) * n)
    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    bbox = _sample_active_bbox(n, k)
    if bbox is None:
        return None
    top, left, bh, bw = bbox

    used_slots = _build_slot_set(top, left, bh, bw, k)
    if used_slots is None:
        return None

    tcols = random.sample(_AVAILABLE_COLORS, k)
    pcols = random.sample(_AVAILABLE_COLORS, k)
    templates = [_sample_template(n, col) for col in tcols]

    assign_pool = list(range(k))
    assign_pool.extend(random.randrange(k) for _ in range(len(used_slots) - k))
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]
    if set(assign_pool) != set(range(k)):
        return None

    expected = _build_expected(n, templates, pcols, placements)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < 0.35 else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
