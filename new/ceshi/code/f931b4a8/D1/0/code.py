# Variant of f931b4a8
# Dimension: D1 — 输出尺寸的编码方式 (symbol_instructions)
# Change: 左上象限前景像素计数 = 输出行数，右上象限前景像素计数 = 输出列数... → 前景色块的外接矩形尺寸决定输出大小

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color


# Design analysis:
# spec_variant:
#   unchanged_core:
#     - 输入仍是偶数边长的正方形，按四个等大象限分区。
#     - 左下象限仍是不含黑色的颜色源瓦片；右下象限仍是模板瓦片，黑色表示洞。
#     - 输出仍由下方两块共同决定：若模板全黑则复制左下瓦片；否则取右下模板，模板洞位由左下瓦片补色。
#   d1_change:
#     - 旧规则读取上方控制区的前景数量；新规则读取前景联合外接框的尺寸。
#     - 左上只关心外接框高度，右上只关心外接框宽度。
#   concrete_encoding:
#     - 左上控制块生成一个“纵向稀疏带”：它覆盖 H 行和 2-4 列，但每行只放 1-若干个像素。
#       因此 bbox 高度 = H，而像素总数通常大于 H。
#     - 右上控制块生成一个“横向稀疏带”：它覆盖 2-4 行和 W 列，但每列只放 1-若干个像素。
#       因此 bbox 宽度 = W，而像素总数通常大于 W。
#   behavior_trace:
#     - 例: Q=4 时，左上前景坐标为 {(0,1),(0,2),(1,2),(2,1),(2,2)}。
#       旧逻辑得到行数 5；新逻辑只看外接框行跨度 0..2，所以行数是 3。
#       右上同理把“列数 = 像素个数”改成“列数 = 最左/最右前景之间的跨度”。
#   complexity_calibration:
#     - 控制区始终让像素数严格大于目标跨度，避免样本悄悄退化回原题。
#     - 同时把非关键轴限制在较窄的 2-4 格内，保证外观整洁，不会像满屏散点那样难以读出外接框。
#   hidden_assumptions_audit:
#     - 原 solver 中 `count_nonzero` 依赖“数量就是尺寸”，这里必须替换为 bbox 尺寸提取。
#     - 原 solver 对下方瓦片的读取逻辑依赖输出尺寸为正且不超过可索引范围；新 generator 保证
#       `1 <= out_h <= Q` 且 `1 <= out_w <= Q`，因此这些断言仍成立。
#     - 原 generator 的 `_make_control(..., count=...)` 已不适用，因为它只控制数量，不控制外接框跨度。
#       这里改成按轴向分别构造，并显式验证 bbox 尺寸和像素数量的差距。
#   placement_legality:
#     - 控制区的 bbox 起点在象限内随机选取，且宽高不超过象限边长，不会越界。
#     - 左下保持全非黑，右下允许黑洞；组合后 `main()` 的所有索引都在边界内。
#   observability_cleanliness:
#     - 上方是黑底上的单个稀疏色带，下方仍是完整瓦片，四块区域职责清楚。
#     - 禁止利用颜色 id 排序；颜色仅做分区装饰，真正的控制信号完全由可见空间跨度承担。


def _bbox_dims(tile: np.ndarray, bg: int) -> tuple[int, int]:
    coords = np.argwhere(tile != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the spatial span of the two upper control patterns, then cut an output from the lower pattern pair.

    Input structure:
    - The input is an even square split into four equal quadrants with black as background.
    - The upper-left and upper-right quadrants are control panels made of scattered foreground pixels.
    - The lower-left quadrant is a non-black color tile, and the lower-right quadrant is a stencil tile that may contain holes.

    Rule:
    - Output height is the height of the foreground bounding box in the upper-left quadrant.
    - Output width is the width of the foreground bounding box in the upper-right quadrant.
    - If the lower-right quadrant is entirely black, copy the lower-left tile over the output window using its usual periodic indexing.
    - Otherwise, read colors from the lower-right stencil; whenever the stencil cell is black, replace it with the coarse color from the lower-left tile at the corresponding tile-block position.

    Key insight:
    - The top half no longer encodes size by counting pixels; it encodes size by how far the scattered pixels stretch across rows or columns.
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

    out_h, _ = _bbox_dims(upper_left, bg)
    _, out_w = _bbox_dims(upper_right, bg)
    assert out_h > 0 and out_w > 0

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

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)                   # keep quadrants large enough for the bbox cue to be visible
STENCIL_EMPTY_PROB = 0.18              # chance lower-right is all-black
LL_SOLID_PROB = 0.70                   # chance fill tile is single-color
LL_MULTI_COLOR_RANGE = (2, 3)          # color count when fill tile is multi-color
STENCIL_SINGLE_COLOR_PROB = 0.75       # chance stencil uses one fg color
STENCIL_MULTI_COLOR_RANGE = (2, 4)     # color count when stencil is multi-color
STENCIL_FG_FRAC = (0.30, 0.75)         # fg fraction range for random stencil mode
STENCIL_FRAME_MIN_Q = 4                # min quadrant size to allow frame stencil
CONTROL_SINGLE_COLOR_PROB = 0.85       # chance control quadrant uses one color
CONTROL_BAND_SPAN_MAX = 4              # non-measured axis span for the sparse control band
UL_MAX_COLORS = 3                      # max distinct colors in upper-left control
UR_MAX_COLORS = 2                      # max distinct colors in upper-right control
OUTPUT_DIM_MIN = 3                     # minimum output height/width
OUTPUT_AREA_MAX = 256                  # retained for symmetry with the original task
GEN_MAX_RETRIES = 300                  # retry budget for generate_input


def _palette(k: int, pool=COLORS) -> list[int]:
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
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
    return tile


def _control_palette(max_colors: int, pool: list[int]) -> list[int]:
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(2, min(max_colors, len(pool)))
    return _palette(k, pool)


def _make_vertical_control(q: int, height: int, max_colors: int, pool: list[int]) -> np.ndarray:
    assert 1 <= height <= q
    pal = _control_palette(max_colors, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    span_w = random.randint(2, min(q, CONTROL_BAND_SPAN_MAX))
    r0 = random.randint(0, q - height)
    c0 = random.randint(0, q - span_w)
    rows = list(range(r0, r0 + height))
    cols = list(range(c0, c0 + span_w))

    for r in rows:
        picks = random.randint(1, span_w)
        for c in random.sample(cols, picks):
            tile[r, c] = random.choice(pal)

    tile[r0, c0] = random.choice(pal)
    tile[r0 + height - 1, c0 + span_w - 1] = random.choice(pal)

    if np.count_nonzero(tile != BLACK) <= height:
        extras = [(r, c) for r in rows for c in cols if tile[r, c] == BLACK]
        assert extras
        r, c = random.choice(extras)
        tile[r, c] = random.choice(pal)

    bbox_h, bbox_w = _bbox_dims(tile, BLACK)
    assert bbox_h == height and bbox_w == span_w
    assert np.count_nonzero(tile != BLACK) > height
    return tile


def _make_horizontal_control(q: int, width: int, max_colors: int, pool: list[int]) -> np.ndarray:
    assert 1 <= width <= q
    pal = _control_palette(max_colors, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    span_h = random.randint(2, min(q, CONTROL_BAND_SPAN_MAX))
    r0 = random.randint(0, q - span_h)
    c0 = random.randint(0, q - width)
    rows = list(range(r0, r0 + span_h))
    cols = list(range(c0, c0 + width))

    for c in cols:
        picks = random.randint(1, span_h)
        for r in random.sample(rows, picks):
            tile[r, c] = random.choice(pal)

    tile[r0, c0] = random.choice(pal)
    tile[r0 + span_h - 1, c0 + width - 1] = random.choice(pal)

    if np.count_nonzero(tile != BLACK) <= width:
        extras = [(r, c) for r in rows for c in cols if tile[r, c] == BLACK]
        assert extras
        r, c = random.choice(extras)
        tile[r, c] = random.choice(pal)

    bbox_h, bbox_w = _bbox_dims(tile, BLACK)
    assert bbox_h == span_h and bbox_w == width
    assert np.count_nonzero(tile != BLACK) > width
    return tile


def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = min(OUTPUT_DIM_MIN, q)
    valid_dims = [
        (h, w)
        for h in range(lo, q + 1)
        for w in range(lo, q + 1)
        if h * w <= OUTPUT_AREA_MAX
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    visible_fill = [(r, c) for r in range(out_h) for c in range(out_w)]
    fill_tile = _make_lower_left(q, ll_palette, visible_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_vertical_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_horizontal_control(q, out_w, UR_MAX_COLORS, control_pool)

    visible_stencil = stencil[:out_h, :out_w]
    if not stencil_empty and not np.any(visible_stencil != BLACK):
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    ul_h, _ = _bbox_dims(upper_left, BLACK)
    _, ur_w = _bbox_dims(upper_right, BLACK)
    if ul_h != out_h or ur_w != out_w:
        return None
    if np.count_nonzero(upper_left != BLACK) <= out_h:
        return None
    if np.count_nonzero(upper_right != BLACK) <= out_w:
        return None

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
