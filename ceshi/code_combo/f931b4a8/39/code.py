# Combo variant of f931b4a8 (index 39)
# combo_spec: D4/3+D5/2+D6/1
# Name: 对角条纹+砖墙铺贴+行循环颜色

from __future__ import annotations

from math import ceil
from pathlib import Path
import random
import sys

import numpy as np

SOLVER_UTILS = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_UTILS) not in sys.path:
    sys.path.append(str(SOLVER_UTILS))

from arc_utils import Color


# Design analysis for the combo variant
# -------------------------------------
# combo_behavior:
#   concretization:
#     - 输入仍是偶数边长正方形，严格均分为四个象限；上半部只负责计数，下半部负责纹理。
#     - 右下模板不再是任意图案，而是“对角条纹模板”：非黑前景沿左上到右下方向成带状分布，
#       黑色位置是待填充的洞。
#     - 模板平铺时不再全局对齐；每跨过一个高度为 Q 的模板行块，下一块在水平方向右移 Q/2，
#       因而连续对角带会在块边界处折成锯齿。
#     - 黑洞填色不再来自 coarse block 坐标 LL[r//Q, c//Q]；改为读取左下象限的“行颜色序列”
#       sequence = LL[:, 0]，并在输出第 r 行统一使用 sequence[r % Q]。
#   behavior_trace:
#     - 例：Q=4，LL 行序列为 [2, 5, 8, 5]，模板主对角带为每行在列 r 处的前景 7，其余为黑。
#     - 原题在输出位置 (5, 1) 读取 stencil[1,1]，若为黑则填 LL[1,0]，因此颜色由 coarse tile row/col 决定。
#     - 融合后先看 r//Q=1 为奇数块行，所以模板列改为 (1 - 2) % 4 = 3，读取 stencil[1,3]；
#       若该处为黑，则填 sequence[5 % 4] = sequence[1] = 5。
#     - 差异有两层：模板采样位置随块行奇偶发生半宽错位；填洞颜色只跟绝对输出行号循环有关。
#       结果是：同一条对角前景在块边界处接成锯齿，而锯齿之间的背景每一行换色。
#   complexity_calibration:
#     - 若模板为空、LL 纯色、或模板对 Q/2 水平平移不敏感，则组合会退化；generator 会拒绝这类样本。
#     - 非空样本强制输出高度 > Q、宽度 > Q+Q/2，使“行循环”和“砖墙错位”都至少出现一次。
#     - LL 限制成每行纯色且至少两色重复，这样 sequence 的来源对人类可见，且每个颜色会在多行重复支撑。
#     - 模板限制为 33%-67% 密度的对角条纹，避免太稀像噪点、太密像实心块。
#   implicit_assumption_audit:
#     - 原 solver 的 `stencil[r % q, c % q]` 假设所有模板副本列对齐；融合后必须改成依赖 `tile_row` 的列索引。
#     - 原 solver 的 `fill_tile[r // q, c // q]` 假设黑洞颜色来自 coarse block；融合后这个前提失效，
#       必须改为按 `r % q` 读取行循环颜色。
#     - 新 solver 额外依赖 Q 为偶数，才能定义半瓦片错位。原题家族 Q 来自 {2,4,8}，generator 维持这一点。
#     - 新 solver 还假设 LL 第一列是非黑的行序列代表。原题本就要求 LL 全非黑，generator 进一步保证每行纯色，
#       因而第一列能稳定暴露整行颜色。
#   placement_legality:
#     - 四象限的空间布局完全不变，不引入新对象，因此输入构造不存在重叠与越界风险。
#     - 输出采样时，模板索引始终取模到 [0, Q)，行循环索引始终取 `r % Q`，都天然有界。
#   visual_observability:
#     - 解题者能在右下直接看到对角条纹模板，在左下直接看到按行排列的颜色序列。
#     - 输出里若对比两段相邻模板行块，会发现对角线在块边界处横向跳半格，排除普通周期平铺。
#     - 同一输出行里的所有黑洞都变成同色，而相邻行颜色按 LL 的行顺序循环，排除 coarse block 填色。
#   visual_cleanliness:
#     - 角色分工清楚：上半部计数、左下给行颜色、右下给对角骨架；输出仅叠加两层信息。
#     - 颜色数控制在 LL 2-3 色、模板 1-3 色，且两者调色板互不重叠，减少视觉混淆。
#   inferability:
#     - 颜色变化不依赖颜色 id 大小或排序，只依赖肉眼可见的行位置与模板空洞位置。
#     - generator 用对照 oracle 拒绝“融合输出 == 原题输出 / D5-only 输出 / D6-only 输出”的样本，
#       保证训练样本真的需要联合推断三项变化。


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.0
STENCIL_SINGLE_COLOR_PROB = 0.82
STENCIL_MULTI_COLOR_RANGE = (2, 3)
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    values = list(pool)
    random.shuffle(values)
    return values[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by sampling colors independently from palette."""
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    """Return all coordinates of a q-by-q grid in a layout-biased order."""
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0 = random.randint(0, q - 1)
        c0 = random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows = list(range(q))
        cols = list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar = random.randrange(q)
        ac = random.randrange(q)
        random.shuffle(coords)
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


def _make_lower_left_general(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Fallback builder for the empty-stencil branch."""
    k = len(palette)
    tile = np.empty((q, q), dtype=int)
    mode = "solid" if k == 1 else random.choice(["rows", "cols", "rows", "cols", "diagonal"])

    if mode == "solid":
        tile[:] = palette[0]
    elif mode == "rows":
        offset = random.randrange(k)
        for r in range(q):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(q):
            tile[:, c] = palette[(c + offset) % k]
    else:
        offset = random.randrange(k)
        rr, cc = np.mgrid[:q, :q]
        idx = (rr + cc + offset) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    existing = set(map(int, np.unique(tile)))
    missing = [color for color in palette if color not in existing]
    if missing:
        fix_coords = (
            list(required_coords)
            if required_coords is not None
            else [(r, c) for r in range(q) for c in range(q)]
        )
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_lower_left_row_sequence(q: int, palette: list[int]) -> np.ndarray:
    """Build an all-non-black tile whose visible structure is a row color cycle."""
    tile = np.empty((q, q), dtype=int)
    offset = random.randrange(len(palette))
    for r in range(q):
        tile[r, :] = palette[(r + offset) % len(palette)]

    if len(set(map(int, tile[:, 0]))) < 2:
        tile[0, :] = palette[0]
        tile[1 % q, :] = palette[min(1, len(palette) - 1)]
    assert np.all(tile != BLACK)
    return tile


def _shifted_stencil_col(c: int, q: int, tile_row: int) -> int:
    """Apply the brick-wall half-tile offset on odd stencil block rows."""
    shift = (q // 2) if tile_row % 2 == 1 else 0
    return (c - shift) % q


def _row_fill_sequence(fill_tile: np.ndarray) -> np.ndarray:
    """Use the first column as the visible row-color sequence."""
    seq = np.asarray(fill_tile[:, 0], dtype=int)
    assert np.all(seq != BLACK)
    return seq


def _is_offset_sensitive(tile: np.ndarray) -> bool:
    """True iff half-width shifting changes the stencil tile."""
    shift = tile.shape[1] // 2
    return not np.array_equal(tile, np.roll(tile, shift=shift, axis=1))


def _make_diagonal_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a non-empty diagonal-stripe stencil with visible black holes."""
    rr, cc = np.mgrid[:q, :q]

    if q == 2:
        period_choices = [2]
        width_choices = [1]
    elif q == 4:
        period_choices = [3]
        width_choices = [1, 2]
    else:
        period_choices = [3, 5]
        width_choices = [1, 2, 3]

    for _ in range(120):
        period = random.choice(period_choices)
        widths = [w for w in width_choices if 0.30 <= w / period <= 0.75]
        if not widths:
            widths = width_choices
        width = random.choice(widths)
        offset = random.randrange(period)
        mask = ((cc - rr - offset) % period) < width
        density = float(mask.mean())
        if not (0.30 <= density <= 0.75):
            continue

        tile = np.zeros((q, q), dtype=int)
        if len(palette) == 1:
            tile[mask] = palette[0]
        else:
            color_offset = random.randrange(len(palette))
            color_idx = (rr + color_offset) % len(palette)
            for idx, color in enumerate(palette):
                tile[mask & (color_idx == idx)] = color

        if np.any(tile == BLACK) and np.any(tile != BLACK) and _is_offset_sensitive(tile):
            return tile

    raise RuntimeError("failed to build a diagonal offset-sensitive stencil")


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = (
        1
        if random.random() < CONTROL_SINGLE_COLOR_PROB
        else random.randint(2, min(max_colors, count, len(pool)))
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        hh, ww = random.choice(factors)
        r0 = random.randint(0, q - hh)
        c0 = random.randint(0, q - ww)
        tile[r0:r0 + hh, c0:c0 + ww] = _random_color_array((hh, ww), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top quadrants as output dimensions and render a diagonal stencil with brick-offset copies over row-cycled colors.

    Input structure:
    - The input is an even square divided into four equal quadrants, with black as background.
    - The upper-left and upper-right quadrants are control regions whose non-black counts set the output height and width.
    - The lower-left quadrant is a fully non-black color tile whose rows expose a repeating color sequence.
    - The lower-right quadrant is either an empty all-black tile or a stencil tile whose foreground lies on diagonal stripe bands and whose black cells are fill holes.

    Rule:
    - Output height equals the number of non-black pixels in the upper-left quadrant.
    - Output width equals the number of non-black pixels in the upper-right quadrant.
    - If the lower-right quadrant is empty, tile the lower-left quadrant directly over the output.
    - Otherwise, tile the stencil in blocks of height Q; odd block rows are shifted right by half a tile width, stencil foreground stays unchanged, and every sampled black stencil cell on output row r is replaced by the cyclic row color from the lower-left quadrant at row r % Q.

    Key insight:
    - The lower half now encodes two coupled periodicities: the right quadrant gives a fine diagonal stripe skeleton whose copies alternate alignment like brickwork, while the left quadrant gives a row-wise color cycle that paints the skeleton's gaps, producing colored zigzag bands.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw and qh % 2 == 0

    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_tile = g[qh:, :qw]
    stencil = g[qh:, qw:]

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % qh, c % qw])
                assert value != bg
                out[r, c] = value
        return out

    row_sequence = _row_fill_sequence(fill_tile)
    for r in range(out_h):
        tile_row = r // qh
        stencil_r = r % qh
        stripe_color = int(row_sequence[r % qh])
        for c in range(out_w):
            stencil_c = _shifted_stencil_col(c, qw, tile_row)
            value = int(stencil[stencil_r, stencil_c])
            if value == bg:
                value = stripe_color
            out[r, c] = value

    return out


def _expected_output_original(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Original task behavior: periodic stencil with coarse LL hole filling."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]

    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _expected_output_d5_only(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Single-variant reference: brick-offset stencil with original coarse fill."""
    q = fill_tile.shape[0]
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        rr, cc = np.mgrid[:out_h, :out_w]
        return fill_tile[rr % q, cc % q]

    for r in range(out_h):
        tile_row = r // q
        stencil_r = r % q
        for c in range(out_w):
            stencil_c = _shifted_stencil_col(c, q, tile_row)
            value = int(stencil[stencil_r, stencil_c])
            if value == BLACK:
                value = int(fill_tile[r // q, c // q])
            out[r, c] = value
    return out


def _expected_output_d6_only(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Single-variant reference: periodic stencil with row-cycled hole filling."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]

    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    row_sequence = _row_fill_sequence(fill_tile)
    tiled[bg_mask] = row_sequence[rr[bg_mask] % q]
    return tiled


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Fused behavior: brick-offset stencil with row-cycled hole filling."""
    q = fill_tile.shape[0]
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        rr, cc = np.mgrid[:out_h, :out_w]
        return fill_tile[rr % q, cc % q]

    row_sequence = _row_fill_sequence(fill_tile)
    for r in range(out_h):
        tile_row = r // q
        stencil_r = r % q
        stripe_color = int(row_sequence[r % q])
        for c in range(out_w):
            stencil_c = _shifted_stencil_col(c, q, tile_row)
            value = int(stencil[stencil_r, stencil_c])
            if value == BLACK:
                value = stripe_color
            out[r, c] = value
    return out


def _behavior_metrics(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> dict[str, int]:
    """Quantify whether D4, D5, and D6 are all visibly active in a sample."""
    fused = _expected_output(fill_tile, stencil, out_h, out_w)
    original = _expected_output_original(fill_tile, stencil, out_h, out_w)
    d5_only = _expected_output_d5_only(fill_tile, stencil, out_h, out_w)
    d6_only = _expected_output_d6_only(fill_tile, stencil, out_h, out_w)

    fg_count = int(np.count_nonzero(stencil != BLACK))
    bg_count = int(np.count_nonzero(stencil == BLACK))
    return {
        "fg_count": fg_count,
        "bg_count": bg_count,
        "orig_diff": int(np.count_nonzero(fused != original)),
        "d5_only_diff": int(np.count_nonzero(fused != d5_only)),
        "d6_only_diff": int(np.count_nonzero(fused != d6_only)),
    }


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid fused sample."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    if stencil_empty:
        ll_k = 1 if random.random() < 0.70 else random.randint(2, min(3, q))
        ll_palette = _palette(ll_k)
        stencil_palette: list[int] = []
    else:
        ll_k = random.randint(2, min(3, q))
        ll_palette = _palette(ll_k)
        st_k = (
            1
            if random.random() < STENCIL_SINGLE_COLOR_PROB
            else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        )
        stencil_pool = [color for color in COLORS if color not in ll_palette]
        if len(stencil_pool) < st_k:
            return None
        stencil_palette = _palette(st_k, stencil_pool)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [color for color in COLORS if color not in output_colors]
    if len(control_pool) < 3:
        return None

    min_h = max(OUTPUT_DIM_MIN, q if stencil_empty else q + 1)
    min_w = max(OUTPUT_DIM_MIN, q if stencil_empty else q + (q // 2) + 1)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(min_h, hi + 1)
        for out_w in range(min_w, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    if stencil_empty:
        used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
        fill_tile = _make_lower_left_general(q, ll_palette, used_fill)
        stencil = np.zeros((q, q), dtype=int)
    else:
        fill_tile = _make_lower_left_row_sequence(q, ll_palette)
        stencil = _make_diagonal_stencil(q, stencil_palette)

    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    metrics = _behavior_metrics(fill_tile, stencil, out_h, out_w)
    if not stencil_empty:
        if metrics["bg_count"] == 0 or metrics["fg_count"] == 0:
            return None
        if metrics["orig_diff"] == 0 or metrics["d5_only_diff"] == 0 or metrics["d6_only_diff"] == 0:
            return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
