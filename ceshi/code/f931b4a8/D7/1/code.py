# Variant of f931b4a8
# Dimension: D7 — 输出网格形状 (grid_shape)
# Change: 输出尺寸由两个控制象限的前景像素计数严格确定：height = count(UL_fg), width = count(... → 输出为模板尺寸的整数倍
#
# Design analysis
# spec:
#   variant_core:
#     - 输入仍然是偶数边长的正方形，并按四象限切分；左下是颜色源瓦片，右下是模板瓦片。
#     - 上方两个控制象限不再直接给出输出的像素高宽，而是分别给出模板沿竖直/水平方向的重复次数。
#     - 若象限边长为 q，则输出尺寸固定为 (UL_count * q, UR_count * q)。
#   concrete_behavior:
#     - 右下模板仍按 qxq 的细粒度周期平铺到整个输出。
#     - 模板中的黑色洞仍改由左下瓦片提供颜色，但索引方式变成按“第几个模板块”取值：
#       output[r, c] = fill_tile[r // q, c // q] when stencil[r % q, c % q] is black.
#     - 因此左下瓦片现在承担 qxq 个“粗块颜色标签”的角色，每个标签控制一个整块 qxq 模板副本中的洞。
#   behavior_trace:
#     - 例：q = 4，UL 有 3 个前景像素，UR 有 2 个前景像素。
#     - 原题：输出尺寸是 3x2；模板只会截取一个很小的左上角区域，粗粒度索引只访问 fill_tile[0,0]。
#     - 变体：输出尺寸是 12x8；模板完整重复为 3 行 2 列，粗粒度索引会访问 fill_tile[0:3, 0:2]。
#     - 差异：计数的语义从“最终像素数”改成了“模板块个数”，使输出天然按模板尺寸整齐分块。
#   complexity_calibration:
#     - 为避免退化成只输出单个模板，生成器禁止 (repeat_h, repeat_w) 同时为 1。
#     - 为避免输出过大，重复次数限制在 1..q，且总面积受 OUTPUT_AREA_MAX 约束。
#     - 重复次数保持低自由度整数计数，解题者能从少量样例看出“每多一个控制像素，就多一整个模板块”。
#   implicit_assumptions_review:
#     - 原 main 中 `out_h = count_nonzero(upper_left != bg)` 依赖“计数即像素高”；变体下必须改成 `repeat_h * q`。
#     - 原 main 中 `fill_r = r // qh` 和 `assert fill_r < qh` 在原题通常无害，因为 out_h <= q^2。
#       变体下若重复次数超过 q，这里会越界，所以 solver 与 generator 都必须保证 repeat_h <= q、repeat_w <= q。
#     - 原 empty-stencil 分支直接按像素周期铺左下瓦片；变体仍可沿用，只是输出尺寸改成模板整倍数。
#   placement_validity:
#     - 四象限布局未变，不存在对象重叠问题。
#     - 控制象限只需精确放置少量前景像素；它们的颜色仍不参与输出，且从输出调色板中避开，保证区域角色清晰。
#     - 左下保持全非黑，右下非空模板始终保留黑洞，确保两层纹理都可见。
#   visual_observability:
#     - 输出总是由完整 qxq 模板块拼成，边界清晰；相邻模板块中的黑洞颜色会随左下粗粒度坐标改变。
#     - 因为控制计数远小于输出像素数，观察者更容易归纳出“上方是在数块数”，而不是“数输出边长像素数”。
#   visual_cleanliness:
#     - 结构仍分四个大区块，没有新增标记叠加。
#     - 左下使用简单实色/条纹/对角线，右下使用棋盘/边框/条纹/散点，视觉职责分明。
#   color_inference:
#     - 颜色仅沿用原题中的“模板色 + 填充色”组合，不引入基于颜色 id 的排序或映射。

from __future__ import annotations

import random

import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the two upper quadrants as repeat counters for a blockwise pattern built from the lower quadrants.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants are control regions whose foreground counts specify how many full template copies appear vertically and horizontally.
    - The lower-left quadrant is a non-black fill tile, and the lower-right quadrant is either a stencil tile or an all-black empty template.

    Rule:
    - Let q be the quadrant size. Output height is `count(upper_left foreground) * q`, and output width is `count(upper_right foreground) * q`.
    - If the lower-right quadrant is empty, repeat the lower-left tile directly across the enlarged output.
    - Otherwise, repeat the lower-right stencil over the output; whenever a stencil cell is black, replace it with the lower-left color at the coarse block coordinate `(r // q, c // q)`.

    Key insight:
    - The top quadrants no longer encode pixel dimensions; they encode how many whole qxq template blocks to place, while the bottom quadrants still define the fine stencil and the coarse block colors.
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

    repeat_h = int(np.count_nonzero(upper_left != bg))
    repeat_w = int(np.count_nonzero(upper_right != bg))
    assert 0 < repeat_h <= qh and 0 < repeat_w <= qw

    out_h = repeat_h * qh
    out_w = repeat_w * qw

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(fill_tile[r % qh, c % qw])
                assert value != bg
                out[r, c] = value
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = int(stencil[r % qh, c % qw])
            if value == bg:
                fill_r = r // qh
                fill_c = c // qw
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


# --------------- generator (o5x) ---------------

from math import ceil

# ── configuration constants ──────────────────────────────────────────
BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)                 # allowed input grid side-lengths N
STENCIL_EMPTY_PROB = 0.16               # chance lower-right is all-black
LL_SOLID_PROB = 0.70                    # chance fill tile is single-color
LL_MULTI_COLOR_RANGE = (2, 3)           # color count when fill tile is multi-color
STENCIL_SINGLE_COLOR_PROB = 0.75        # chance stencil uses one fg color
STENCIL_MULTI_COLOR_RANGE = (2, 4)      # color count when stencil is multi-color
STENCIL_FG_FRAC = (0.30, 0.75)          # fg fraction range for random stencil mode
STENCIL_FRAME_MIN_Q = 4                 # min quadrant size to allow frame stencil
CONTROL_SINGLE_COLOR_PROB = 0.85        # chance control quadrant uses one color
CONTROL_RECT_PROB = 0.85                # chance control pixels form an axis-aligned rect
UL_MAX_COLORS = 3                       # max distinct colors in upper-left control
UR_MAX_COLORS = 2                       # max distinct colors in upper-right control
REPEAT_MIN = 1                          # minimum number of template copies per axis
OUTPUT_AREA_MAX = 256                   # maximum output area
GEN_MAX_RETRIES = 300                   # retry budget for generate_input


# ── palette helpers ──────────────────────────────────────────────────

def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


# ── coordinate orderings for control quadrant ────────────────────────

def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    """Return all (r,c) in a qxq grid in a layout-dependent order."""
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
        return sorted(
            coords,
            key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()),
        )
    random.shuffle(coords)
    return coords


# ── quadrant builders ────────────────────────────────────────────────

def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the fill tile (all non-black) using stripe/solid/diagonal patterns."""
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
    elif mode == "diagonal":
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


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black = holes filled from fill tile)."""
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
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    max_k = min(max_colors, count, len(pool))
    if max_k == 1 or random.random() < CONTROL_SINGLE_COLOR_PROB:
        k = 1
    else:
        k = random.randint(2, max_k)
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


# ── oracle / assembly ────────────────────────────────────────────────

def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    repeat_h: int,
    repeat_w: int,
) -> np.ndarray:
    """Reproduce the solver's tiling logic for oracle validation."""
    q = fill_tile.shape[0]
    out_h, out_w = repeat_h * q, repeat_w * q
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _valid_repeats(q: int, ll_k: int) -> list[tuple[int, int]]:
    """Enumerate repeat counts that fit the area cap and expose enough coarse cells."""
    pairs = []
    for repeat_h in range(REPEAT_MIN, q + 1):
        for repeat_w in range(REPEAT_MIN, q + 1):
            out_h, out_w = repeat_h * q, repeat_w * q
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            if repeat_h * repeat_w < ll_k:
                continue
            pairs.append((repeat_h, repeat_w))
    return pairs


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    valid_repeats = _valid_repeats(q, ll_k)
    if not valid_repeats:
        return None
    visible_repeats = [pair for pair in valid_repeats if pair != (1, 1)]
    repeat_h, repeat_w = random.choice(visible_repeats or valid_repeats)

    used_fill = [(r, c) for r in range(repeat_h) for c in range(repeat_w)]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, repeat_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, repeat_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, repeat_h, repeat_w)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if actual.shape == (repeat_h, repeat_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
