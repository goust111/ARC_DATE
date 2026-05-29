# Variant of f931b4a8
# Dimension: D6 — 背景替换规则 (transform)
# Change: 模板中的零值像素被替换为颜色源在粗粒度坐标（瓦片重复索引）处的颜色：fill_color = LL[tile_row][... → 背景替换颜色按输出行索引循环

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


# Design analysis for the D6 variant
# ----------------------------------
# spec:
#   unchanged_structure:
#     - 输入仍是偶数边长正方形，均分成四个象限。
#     - 左上/右上仍只通过非黑像素数控制输出高宽。
#     - 左下仍是颜色源，右下仍是模板；右下全黑时仍退回为直接平铺左下。
#   d6_core_change:
#     - 仅在右下模板非空且某个模板像素为黑色时触发新规则。
#     - 旧规则把该黑洞替换为 LL[输出粗粒度块行][输出粗粒度块列]。
#     - 新规则改为：先从 LL 取出“按行排列的颜色序列”，这里具体定义为 LL 第一列自上而下的颜色；
#       然后输出中第 r 行的所有模板黑洞统一填成 sequence[r % Q]。
#   concrete_pixel_behavior:
#     - 为了让人眼能读出这条规则，generator 在非空模板样本中把 LL 做成“每行纯色的横条序列”。
#     - 这样 sequence 就直接可见为 LL 的横条颜色顺序，输出中的模板黑洞会形成按行循环的水平条纹。
#   behavior_trace:
#     - 例：Q=2，LL=[[2,2],[5,5]]，模板=[[7,0],[0,7]]，输出高宽为 4x4。
#     - 原题黑洞填色：前两行黑洞都取 LL[0,*]=2，后两行都取 LL[1,*]=5，得到按 2 行分块的背景色。
#     - 变体黑洞填色：第 0/1/2/3 行分别取 2/5/2/5，得到逐行循环的水平条纹。
#     - 差异点在于：旧逻辑依赖 coarse tile row/col；新逻辑只依赖绝对输出行号。
#   complexity_calibration:
#     - 若 LL 是纯色，变体会退化得不明显；因此非空模板样本强制 LL 至少两种颜色且按行分布。
#     - 若模板没有黑洞，新规则无从观察；因此非空模板 builder 始终保留黑洞，并把模板为空的概率压低。
#     - 输出高度在非空模板样本中至少为 Q+1（Q=2 时至少 3），让行序列出现重复，更容易看出“循环”。
#   implicit_assumption_audit:
#     - main() 新增假设：模板黑洞用到的行颜色序列可从 LL 第一列读取，且该列非黑。
#     - 这一假设对 solver 是安全的，因为原题本就要求 LL 全非黑；generator 进一步保证非空模板时 LL 每行纯色，
#       因而“第一列代表整行颜色”对人类观察也成立。
#     - 原 solver 中 fill_r/fill_c 的 coarse 索引断言不再适用；若保留会把新规则错误地限制回旧任务。
#   placement_legality:
#     - 四象限仍严格放在固定位置，无额外对象，因此不存在重叠问题。
#     - control 区只编码计数，颜色与输出隔离；LL 和模板仍各占一个完整象限。
#   observability_and_tidiness:
#     - 非空模板样本的输出由“模板前景图案 + 行循环背景条纹”组成，区域职责清楚，不依赖颜色 id 排序。
#     - 颜色种类控制在 2-3 个 LL 颜色和 1-4 个模板颜色内，足以看出规律而不至于杂乱。


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)
STENCIL_EMPTY_PROB = 0.08
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))],
                    dtype=int).reshape(shape)


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0, c0 = random.randint(0, q - 1), random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows, cols = list(range(q)), list(range(q))
        random.shuffle(rows)
        random.shuffle(cols)
        return [(r, c) for r in rows for c in cols]
    if mode == "compact":
        ar, ac = random.randrange(q), random.randrange(q)
        random.shuffle(coords)
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _make_lower_left_general(q: int, palette: list[int],
                             required_coords: list[tuple[int, int]] | None = None) -> np.ndarray:
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
    missing = [c for c in palette if c not in existing]
    if missing:
        fix_coords = list(required_coords) if required_coords is not None else [
            (r, c) for r in range(q) for c in range(q)
        ]
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    return tile


def _make_lower_left_row_sequence(q: int, palette: list[int]) -> np.ndarray:
    tile = np.empty((q, q), dtype=int)
    offset = random.randrange(len(palette))
    for r in range(q):
        tile[r, :] = palette[(r + offset) % len(palette)]

    if len(set(map(int, tile[:, 0]))) < 2:
        tile[0, :] = palette[0]
        tile[1 % q, :] = palette[min(1, len(palette) - 1)]
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    tile = np.zeros((q, q), dtype=int)
    modes = ["checker", "row_stripe", "col_stripe", "random", "random"]
    if q >= STENCIL_FRAME_MIN_Q:
        modes.append("frame")
    mode = random.choice(modes)

    if mode == "checker":
        parity = random.randrange(2)
        mask = (np.arange(q)[:, None] + np.arange(q)[None, :]) % 2 == parity
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "frame":
        t = random.randint(1, max(1, q // 4))
        mask = np.ones((q, q), dtype=bool)
        mask[t:q - t, t:q - t] = False
        tile[mask] = _random_color_array((int(mask.sum()),), palette)
    elif mode == "row_stripe":
        parity = random.randrange(2)
        for r in range(q):
            if r % 2 == parity:
                tile[r, :] = _random_color_array((q,), palette)
    elif mode == "col_stripe":
        parity = random.randrange(2)
        for c in range(q):
            if c % 2 == parity:
                tile[:, c] = _random_color_array((q,), palette)
    else:
        area = q * q
        lo = max(1, int(ceil(STENCIL_FG_FRAC[0] * area)))
        hi = max(lo, int(STENCIL_FG_FRAC[1] * area))
        fg_count = random.randint(lo, hi)
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for r, c in coords[:fg_count]:
            tile[r, c] = random.choice(palette)
    assert np.any(tile != BLACK)
    assert np.any(tile == BLACK)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else \
        random.randint(2, min(max_colors, count, len(pool)))
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    factors = [(h, w) for h in range(1, q + 1) for w in range(1, q + 1) if h * w == count]
    if factors and random.random() < CONTROL_RECT_PROB:
        h, w = random.choice(factors)
        r0 = random.randint(0, q - h)
        c0 = random.randint(0, q - w)
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
    else:
        mode = random.choice(["rect", "compact", "rows"])
        coords = _ordered_coords(q, mode)[:count]
        for r, c in coords:
            tile[r, c] = random.choice(pal)
    return tile


def _row_fill_sequence(fill_tile: np.ndarray) -> np.ndarray:
    seq = np.asarray(fill_tile[:, 0], dtype=int)
    assert np.all(seq != BLACK)
    return seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top quadrants as output dimensions and use the bottom quadrants to build a tiled pattern.

    Input structure:
    - The input is an even square divided into four equal quadrants.
    - The upper-left and upper-right quadrants are control regions counted against black background.
    - The lower-left quadrant is a non-black color source tile.
    - The lower-right quadrant is either an empty all-black tile or a stencil tile with black holes.

    Rule:
    - Output height equals the number of non-black pixels in the upper-left quadrant.
    - Output width equals the number of non-black pixels in the upper-right quadrant.
    - If the lower-right quadrant is empty, repeat the lower-left tile directly across the output.
    - Otherwise, repeat the stencil; stencil foreground stays unchanged, and each stencil black hole on output row r is replaced by the r-th cyclic color from the lower-left tile's row sequence.

    Key insight:
    - The bottom-left quadrant no longer provides coarse block colors for stencil holes; instead it exposes a row-wise color cycle, so the filled background becomes horizontal stripes underneath the repeated stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
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
        stripe_color = int(row_sequence[r % qh])
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            if value == bg:
                value = stripe_color
            out[r, c] = value

    return out


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]

    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    row_sequence = _row_fill_sequence(fill_tile)
    tiled[bg_mask] = row_sequence[rr[bg_mask] % q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
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
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else \
            random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    min_h = max(OUTPUT_DIM_MIN, q + (0 if stencil_empty else 1))
    min_w = max(OUTPUT_DIM_MIN, q)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (h, w) for h in range(min_h, hi + 1) for w in range(min_w, hi + 1)
        if h * w <= OUTPUT_AREA_MAX
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
        stencil = _make_stencil(q, stencil_palette)

    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
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
