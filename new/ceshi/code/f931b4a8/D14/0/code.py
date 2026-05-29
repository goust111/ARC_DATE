from __future__ import annotations

# Variant of f931b4a8
# Dimension: D14 — 颜色源与模板的尺寸关系 (meta)
# Change: 颜色源和模板的尺寸相同（均为半网格大小），两者一一对应地参与铺贴... → 颜色源尺寸小于模板（需放大对齐）

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis:
# variant:
#   source_task: b10624e5
#   borrowed_mechanism: |
#     参考题 b10624e5 的关键不是“任意复杂缩放”，而是“从输入里读出一个离散比例，
#     再用这个比例把一个较小结构对齐到较大结构”。这里沿用同样的复杂度校准：
#     只使用可见、稳定的整数 2x 放大，不引入多档浮动比例。
#   concrete_layout: |
#     输入仍然是 8x8，被分成四个 4x4 象限。上左和上右仍是控制区；右下仍是 4x4 模板。
#     变化只发生在左下：它不再是完整 4x4 颜色表，而是嵌在黑背景中的一个 2x2 非黑小瓦片。
#     solver 需要先在左下找到这个 2x2 外接框，再把它按 2x 放大成 4x4 的 coarse 颜色图。
#   behavior_trace: |
#     例：左下小瓦片是 [[1,2],[3,4]]，放大后得到
#       [[1,1,2,2],
#        [1,1,2,2],
#        [3,3,4,4],
#        [3,3,4,4]]
#     若输出像素位于第 (r=10,c=13)，因为模板大小 q=4，所以它属于第 (2,3) 个模板重复块。
#     原题在这里会直接读取 fill_tile[2,3]；变体则必须读取放大后的 coarse 图 [2,3]，
#     也就是小瓦片中的 [1,1]，得到颜色 4。差异发生在“黑洞补色”的 coarse lookup 阶段。
#   complexity_guardrails: |
#     为了让 D14 的变化稳定可见，生成器固定输入边长为 8，使 q=4，左下源瓦片固定为 2x2。
#     这样 2x 放大后恰好覆盖 4x4 模板坐标；若继续允许 q=8，则在原输出尺寸上限 16 下，
#     coarse lookup 往往只会采到放大图的左上角，变体会在样例里不够明显。
#   hidden_assumptions_checked: |
#     原 solver 依赖“左下整块 4x4 都是非黑且可直接按 coarse 坐标索引”。这个前提现在失效：
#     1. 左下大部分位置会变黑，因此不能再把整个象限当成 fill tile。
#     2. coarse 索引仍是 0..3，但颜色源只有 2x2，所以必须先按整除比例扩成 4x4。
#     3. 只有当左下非黑区域恰好形成一个实心矩形时，bbox 提取才无歧义；generator 会保证这一点。
#   legality_and_visibility: |
#     左下只放一个 2x2 实心非黑矩形，其余留黑，不重叠、不越界，视觉上也能一眼看出
#     “颜色源比模板更小”。输出高宽固定在 9..16，保证至少跨 3 个模板块，四个源单元都会参与。


def _extract_source_tile(fill_quadrant: np.ndarray, bg: int) -> np.ndarray:
    """Find the unique solid non-background rectangle inside the lower-left quadrant."""
    coords = np.argwhere(fill_quadrant != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    tile = fill_quadrant[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(tile != bg)

    mask = np.zeros(fill_quadrant.shape, dtype=bool)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert not np.any((fill_quadrant != bg) & ~mask)
    assert tile.shape[0] < fill_quadrant.shape[0] and tile.shape[1] < fill_quadrant.shape[1]
    return tile


def _expand_source_tile(source_tile: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Nearest-neighbor expand a smaller source tile to the stencil's coarse resolution."""
    sh, sw = source_tile.shape
    th, tw = target_shape
    assert th % sh == 0 and tw % sw == 0
    scale_r = th // sh
    scale_c = tw // sw
    return np.repeat(np.repeat(source_tile, scale_r, axis=0), scale_c, axis=1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top quadrants to size an output whose repeated stencil is filled by an upscaled lower-left source tile.

    Input structure:
    - The input is an even square cut into four equal quadrants.
    - The two upper quadrants are control regions; only their non-black counts matter.
    - The lower-right quadrant is a full-size stencil tile.
    - The lower-left quadrant contains a smaller non-black source rectangle embedded in black.

    Rule:
    - Output height is the non-black count in the upper-left quadrant.
    - Output width is the non-black count in the upper-right quadrant.
    - First enlarge the small lower-left source so it matches the lower-right quadrant's size by repeating each source cell over an equal block.
    - If the stencil is empty, tile this enlarged source directly over the output; otherwise tile the stencil and use the enlarged source only for stencil background cells, indexed by the repeated-stencil block position.

    Key insight:
    - The bottom quadrants still describe a two-level pattern, but the color source now lives at a coarser resolution and must be upscaled before it can align with the stencil's coordinate system.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    upper_left = g[:qh, :qw]
    upper_right = g[:qh, qw:]
    fill_quadrant = g[qh:, :qw]
    stencil = g[qh:, qw:]

    source_tile = _extract_source_tile(fill_quadrant, bg)
    coarse_fill = _expand_source_tile(source_tile, (qh, qw))

    out_h = int(np.count_nonzero(upper_left != bg))
    out_w = int(np.count_nonzero(upper_right != bg))
    assert out_h > 0 and out_w > 0

    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != bg):
        for r in range(out_h):
            for c in range(out_w):
                value = int(coarse_fill[r % qh, c % qw])
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
                value = int(coarse_fill[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


# --------------- generator (o5x) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIDE = 8
QUADRANT = GRID_SIDE // 2
SOURCE_SIZE = QUADRANT // 2
SOURCE_SCALE = QUADRANT // SOURCE_SIZE

STENCIL_EMPTY_PROB = 0.10
LL_SOLID_PROB = 0.30
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.70
STENCIL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 9
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _palette(k: int, pool=COLORS) -> list[int]:
    """Return k distinct colors sampled from pool."""
    pool = list(pool)
    random.shuffle(pool)
    return pool[:k]


def _random_color_array(shape: tuple[int, ...], palette: list[int]) -> np.ndarray:
    """Fill an array by drawing each cell independently from palette."""
    return np.array([random.choice(palette) for _ in range(int(np.prod(shape)))],
                    dtype=int).reshape(shape)


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
        return sorted(coords, key=lambda rc: (abs(rc[0] - ar) + abs(rc[1] - ac), random.random()))
    random.shuffle(coords)
    return coords


def _make_small_source(size: int, palette: list[int]) -> np.ndarray:
    """Build the embedded source tile (all non-black) with simple visible patterns."""
    k = len(palette)
    tile = np.empty((size, size), dtype=int)
    mode = "solid" if k == 1 else random.choice(["rows", "cols", "checker", "diagonal"])

    if mode == "solid":
        tile[:] = palette[0]
    elif mode == "rows":
        offset = random.randrange(k)
        for r in range(size):
            tile[r, :] = palette[(r + offset) % k]
    elif mode == "cols":
        offset = random.randrange(k)
        for c in range(size):
            tile[:, c] = palette[(c + offset) % k]
    elif mode == "checker":
        rr, cc = np.mgrid[:size, :size]
        idx = (rr + cc + random.randrange(k)) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color
    else:
        rr, cc = np.mgrid[:size, :size]
        idx = (rr * size + cc + random.randrange(k)) % k
        for i, color in enumerate(palette):
            tile[idx == i] = color

    existing = set(map(int, np.unique(tile)))
    missing = [c for c in palette if c not in existing]
    if missing:
        fix_coords = [(r, c) for r in range(size) for c in range(size)]
        random.shuffle(fix_coords)
        for color, (r, c) in zip(missing, fix_coords):
            tile[r, c] = color
    assert np.all(tile != BLACK)
    return tile


def _make_fill_quadrant(q: int, source_tile: np.ndarray) -> np.ndarray:
    """Place the smaller source tile inside a black quadrant at a non-trivial offset."""
    size = source_tile.shape[0]
    quadrant = np.zeros((q, q), dtype=int)
    offsets = [(r, c) for r in range(q - size + 1) for c in range(q - size + 1) if (r, c) != (0, 0)]
    r0, c0 = random.choice(offsets)
    quadrant[r0:r0 + size, c0:c0 + size] = source_tile
    return quadrant


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build the stencil tile (black = holes filled from the upscaled source)."""
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

    assert np.any(tile == BLACK)
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
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


def _expected_output(source_tile: np.ndarray, stencil: np.ndarray,
                     out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the solver's tiling logic for oracle validation."""
    q = stencil.shape[0]
    coarse_fill = _expand_source_tile(source_tile, (q, q))
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return coarse_fill[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = coarse_fill[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _used_source_cell_count(source_tile: np.ndarray, stencil: np.ndarray,
                            out_h: int, out_w: int) -> int:
    """Count how many source cells actually influence the output."""
    q = stencil.shape[0]
    sh, sw = source_tile.shape
    scale_r = q // sh
    scale_c = q // sw
    used: set[tuple[int, int]] = set()

    if not np.any(stencil != BLACK):
        for r in range(q):
            for c in range(q):
                used.add((r // scale_r, c // scale_c))
        return len(used)

    block_rows = ceil(out_h / q)
    block_cols = ceil(out_w / q)
    for br in range(block_rows):
        row_span = min(q, out_h - br * q)
        for bc in range(block_cols):
            col_span = min(q, out_w - bc * q)
            if np.any(stencil[:row_span, :col_span] == BLACK):
                used.add((br // scale_r, bc // scale_c))
    return len(used)


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    n = GRID_SIDE
    q = QUADRANT
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else \
            random.randint(*STENCIL_MULTI_COLOR_RANGE)
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    valid_dims = [
        (h, w) for h in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        for w in range(OUTPUT_DIM_MIN, OUTPUT_DIM_MAX + 1)
        if h * w <= OUTPUT_AREA_MAX
    ]
    out_h, out_w = random.choice(valid_dims)

    source_tile = _make_small_source(SOURCE_SIZE, ll_palette)
    fill_quadrant = _make_fill_quadrant(q, source_tile)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    if _used_source_cell_count(source_tile, stencil, out_h, out_w) < SOURCE_SIZE * SOURCE_SIZE:
        return None
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_quadrant
    grid[q:, q:] = stencil

    expected = _expected_output(source_tile, stencil, out_h, out_w)
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
