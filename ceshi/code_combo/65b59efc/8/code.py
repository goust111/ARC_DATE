# Combo variant of 65b59efc (index 8)
# combo_spec: D5/1+D8/0
# Name: 颜色对驱动双色重绘

"""
design_analysis:
  fusion_resolution:
    - D5/1 明确要求模板块使用两种输出色：模板活跃像素一色、模板空白像素另一色。
    - D8/0 明确要求调色板不再是直接单色样，而是“键色 + 值色”的颜色对，需要先分辨哪一个是键。
    - 为了把两者真正融合而不是并排叠加，这里将底部带做成两行颜色对：
      上方一行的无序颜色对解出“模板背景重绘色”，下方一行的无序颜色对解出“模板前景重绘色”。
      每个 pair 内都包含该模板的键色（即模板色/标记色）和对应输出值色，左右顺序随机。
  concretization_and_behavior_trace:
    - 原题：某模板列底部只有一个直接色样，marker 色直接选模板，输出块是 `mask ? palette : 全局背景`。
    - 融合后：同一模板列底部有两条 pair 记录，例如上行出现 {模板色 3, 输出色 8}，下行出现 {模板色 3, 输出色 1}。
      solver 必须先知道模板键色是 3，于是把上行另一色 8 当作“块背景色”，下行另一色 1 当作“块前景色”。
      最终输出块变成 `mask ? 1 : 8`，不再是 `mask ? 1 : 0`。
    - 差异点：
      1. 调色板不能按固定左右位置读取，必须识别键值角色。
      2. 输出块内部不再露出全局背景，而是完整填满双色模板块。
  complexity_calibration:
    - 仍保留原题的三带结构、tile 列、marker 定位、模板色选模板这套主干，避免同时改动太多维度。
    - 新自由度只集中在两处：pair 内左右顺序随机、以及每个模板有 fg/bg 两个输出色。
    - generator 强制模板既有活跃像素也有空白像素，且背景重绘色从不等于全局背景，从而保证双色效果可见。
    - generator 还强制样例内同时出现“键在左”和“键在右”两种 pair，避免退化为固定位置读取。
  implicit_assumptions_review_main:
    - `N = sep_cols[0]` 仍依赖首个竖向分隔线出现在第 N 列；生成器保留原始栅格版式。
    - 顶部模板读取仍依赖每个模板 tile 只有一种模板色；生成器继续用单色二值模板。
    - 中带 marker 仍依赖“每个 tile 列只出现一种 marker/template 色”；生成器继续对模板和 tile 列做 1 对 1 指派。
    - 新增假设 1：底部带恰有两条含色样的 palette 行；生成器只在这两行落色，其余 palette 带和尾部 padding 保持背景。
    - 新增假设 2：每条 palette 行、每个模板列都恰好出现 2 个不同颜色，且其中一个必为该模板键色；生成器显式保证。
    - 新增假设 3：palette 值色不与任何模板键色重合，这样“哪个是键色”在视觉上稳定可见；生成器显式保证。
  placement_legality:
    - 继续采用 output-first：先决定输出块使用哪个模板和位置，再反推 marker 布局。
    - marker 仍使用互不冲突的 `(lr, lc)` 槽位，不会出现不同模板争用同一输出块。
    - 两个 palette pair 都放在各自模板列内部的相邻两格，不跨越分隔列，也不与模板/marker 带重叠。
  visual_observability:
    - 人类可直接看到：底部每列有两行“模板色 + 另一色”的配对；而输出中每个盖章块内部也明显只有两色。
    - 因为 pair 左右顺序在同一样例内会混用，解题者必须通过“模板色重复出现”来识别键，而不能靠固定位置猜。
    - 因为模板本身保留空洞，输出块中空洞位置被一致地改成另一种颜色，双色重绘规律可从 2-3 个样例归纳。
  visual_cleanliness_and_inferability:
    - 画面仍然只有模板带、marker 带、palette 带三层，不引入额外提示对象。
    - palette 值色不与模板键色混用，降低颜色角色歧义。
    - 每列 pair 只占 2 个相邻像素，信息密度高但不杂乱。
  non_degeneracy_checks_used_in_generator:
    - 每个模板都含有至少一个 1 和至少一个 0。
    - 每个模板的 `palette_bg != palette_fg` 且 `palette_bg != 全局背景`。
    - 同一输入上，融合输出必须与“原题单色重绘输出”不同。
    - palette pair 的左右顺序必须同时包含两种方向。
  new_main_assumptions_to_guarantee:
    - palette 带两条数据行都能通过“该模板色在 pair 中出现一次”恢复值色。
    - 同一模板若被盖章，其模板空白区域必须真实存在，否则 D5 分量会弱化；生成器已拒绝全填充模板。
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
    Decode a tiled stamp legend whose palette now uses unordered color pairs and two-tone block fills.

    Input structure:
    - The grid is split into K template columns of width N, with dashed separator rows and separator columns.
    - The top band stores K binary templates, each drawn in its own template color.
    - The middle band stores marker pixels; a marker's color still identifies the template, and its local (row, col) inside a tile column still identifies the output block slot.
    - The bottom band contains exactly two palette rows with colored pairs. In each template column and on each palette row, one pixel is the template key color and the other is the output color value. The upper pair row encodes block-background color, the lower pair row encodes block-foreground color.

    Rule:
    - Recover the template masks from the top band and map marker colors to template indices exactly as before.
    - For each template column, read the two palette rows; within each row, find the pixel matching that template's key color and treat the other pixel in the pair as the encoded repaint color.
    - Use the lower decoded color for template-active cells and the upper decoded color for template-inactive cells when stamping the chosen template into the output block indicated by each marker's local position.

    Key insight: local marker coordinates still choose where to stamp, but the palette legend must now be decoded as two unordered key-value pairs per template column, yielding a foreground color and a background color for every stamped block.
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
    k = len(starts)

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        key = vals.pop()
        mask = sub == key
        assert mask.any() and (~mask).any()
        tcols.append(key)
        tmasks.append(mask)
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pal_r0 = sep_rows[-1] + 1
    palette_rows = [
        r
        for r in range(pal_r0, h)
        if any(int(v) not in {bg, sep} for v in g[r, :])
    ]
    assert len(palette_rows) == 2
    pbg_rows = palette_rows[0]
    pfg_rows = palette_rows[1]

    pbg: list[int] = []
    pfg: list[int] = []
    for j, st in enumerate(starts):
        key = tcols[j]
        upper = {int(v) for v in g[pbg_rows, st : st + n].flat} - {bg, sep}
        lower = {int(v) for v in g[pfg_rows, st : st + n].flat} - {bg, sep}
        assert len(upper) == 2 and key in upper
        assert len(lower) == 2 and key in lower
        pbg_val = next(v for v in upper if v != key)
        pfg_val = next(v for v in lower if v != key)
        pbg.append(pbg_val)
        pfg.append(pfg_val)

    mid_r0 = n + 1
    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}

    for st in starts:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "tile column must use one marker color"
                tile_color = v
                j = tcol_to_j[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j, "different templates cannot share one slot"
                slot_owner[(lr, lc)] = j
                tile = np.where(tmasks[j], pfg[j], pbg[j])
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


# --------------- generator (combo D5/1 + D8/0) ---------------

_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3]
_PAL_H_WEIGHTS = [4, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.45, 0.82)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.60
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


def _render_tile(mask: np.ndarray, fg: int, bg_fill: int) -> np.ndarray:
    return np.where(mask, fg, bg_fill)


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    pfg: list[int],
    pbg: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        paste(expected, _render_tile(templates[j], pfg[j], pbg[j]), rr * n, cc * n)
    return expected


def _build_original_like_expected(
    n: int,
    templates: list[np.ndarray],
    pfg: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        paste(expected, np.where(templates[j], pfg[j], _BG), rr * n, cc * n)
    return expected


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


def _pair_cols(n: int) -> tuple[int, int]:
    left = min(n - 2, n // 2)
    right = left + 1
    return left, right


def _draw_pair(grid: np.ndarray, row: int, st: int, n: int, key: int, value: int, key_left: bool) -> None:
    c0, c1 = _pair_cols(n)
    if key_left:
        grid[row, st + c0] = key
        grid[row, st + c1] = value
    else:
        grid[row, st + c0] = value
        grid[row, st + c1] = key


def _sample_palette_colors(k: int, template_colors: list[int]) -> tuple[list[int], list[int]]:
    pool = [c for c in _AVAILABLE_COLORS if c not in template_colors]
    pfg: list[int] = []
    pbg: list[int] = []
    for _ in range(k):
        fg, bg_fill = random.sample(pool, 2)
        pfg.append(fg)
        pbg.append(bg_fill)
    return pfg, pbg


def _mixed_pair_orders(k: int) -> tuple[list[bool], list[bool]] | None:
    bg_bits = [random.choice([False, True]) for _ in range(k)]
    fg_bits = [random.choice([False, True]) for _ in range(k)]
    if all(bg_bits) or not any(bg_bits):
        return None
    if all(fg_bits) or not any(fg_bits):
        return None
    return bg_bits, fg_bits


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    tcols = random.sample(_AVAILABLE_COLORS, k)
    pfg, pbg = _sample_palette_colors(k, tcols)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    templates = [_sample_template(n, col) for col in tcols]

    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    min_marks = max(k, int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    mark_count = random.randint(min_marks, n * n)
    used_slots = all_slots[:mark_count]
    assign_pool = list(range(k))
    assign_pool.extend(random.randrange(k) for _ in range(mark_count - k))
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]
    if set(assign_pool) != set(range(k)):
        return None

    expected = _build_expected(n, templates, pfg, pbg, placements)
    old_like = _build_original_like_expected(n, templates, pfg, placements)
    if np.array_equal(expected, old_like):
        return None

    pair_orders = _mixed_pair_orders(k)
    if pair_orders is None:
        return None
    bg_bits, fg_bits = pair_orders

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

    pair_bg_r = 2 * n + pal_h
    pair_fg_r = pair_bg_r + 1
    for j in range(k):
        st = j * (n + 1)
        _draw_pair(grid, pair_bg_r, st, n, tcols[j], pbg[j], bg_bits[j])
        _draw_pair(grid, pair_fg_r, st, n, tcols[j], pfg[j], fg_bits[j])

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
