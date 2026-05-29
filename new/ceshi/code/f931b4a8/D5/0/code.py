# Variant of f931b4a8
# Dimension: D5 — 铺贴方式 (transform)
# Change: 模板以周期铺贴方式填充输出矩形：output[r][c] = stencil[r % H_stencil][c % W_stencil] → 镜像铺贴（相邻瓦片翻转）

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis
# ===============
# a) 具体化与行为追踪
#    输入仍是一个偶数边长正方形，分成四个象限。上方两个象限只负责计数，左下给粗粒度颜色，
#    右下给细粒度 stencil。原题在输出中重复 stencil 时始终使用同一朝向；本变体改成按块镜像：
#    第 0 个块保持原样，第 1 个块沿共享边翻转，第 2 个块再翻回，如此交替。
#    具体索引是：块行 br = r // Q，块列 bc = c // Q；局部行 lr = r % Q，局部列 lc = c % Q。
#    若 br 为奇数则使用 Q-1-lr，若 bc 为奇数则使用 Q-1-lc。这样第二行块会竖直翻转，
#    第二列块会水平翻转，右下角块则双翻转。
#    小例子：Q=2，stencil = [[1,0],[2,3]]。原题前两块列是 [1,0 | 1,0]，
#    本变体变成 [1,0 | 0,1]；下一块行又会竖直翻成 [[2,3],[1,0]]。变化只发生在
#    “把 stencil 投影到输出”的环节，其余象限含义不变。
#
# b) 复杂度校准
#    只改 D5 会有退化风险：如果输出只覆盖一个 stencil 块、或 stencil 对横翻/竖翻都对称、
#    或 fill/stencil 颜色碰巧相同，人眼会把它看成原题的普通平铺。为避免这个问题，
#    生成器强制非空 stencil、输出至少跨 2x2 个块、stencil 同时破坏横向与纵向镜像对称，
#    并让 fill 与 stencil 使用不重叠的调色板。
#
# c) 对原 main() 的逐段隐式假设审查
#    - out_h/out_w 的计算仍只依赖上方两个控制象限，这一前提保持成立。
#    - 空 stencil 分支原本直接周期铺贴 fill_tile；该分支与 D5 无关，因此保持不变。
#    - 原非空 stencil 分支假设“模板局部坐标 = 取模坐标”。这是本变体唯一失效的前提，
#      必须改成“模板局部坐标 = 取模坐标再按块奇偶翻转”。
#    - 原 black-hole 回填逻辑依赖 fill_r = r // Q, fill_c = c // Q；镜像只改变 stencil 局部朝向，
#      不改变 coarse block 编号，因此这里仍然正确，不能误改成镜像后的局部坐标。
#
# d) 放置合法性
#    生成器仍然直接构造四个象限，没有对象相互遮挡的问题。新的合法性重点是：
#    out_h,out_w <= Q*Q，保证 coarse block 坐标 fill_r/fill_c 始终落在左下 QxQ 范围内；
#    同时至少让 ceil(out_h/Q) >= 2 且 ceil(out_w/Q) >= 2，确保镜像块真实出现。
#
# e) 视觉可观察性
#    观察者需要从少量样例里看到“隔壁块是翻过来的”。因此训练样本不能长期落在
#    对称 stencil、单块输出、或空 stencil 上。这里用结构约束而不是颜色 id 规则来实现：
#    所有判断都基于几何翻转后的图案是否不同。
#
# f) 可视化整洁度
#    输出仍由一个规则网格重复生成，没有额外叠层。左下颜色块负责黑洞填色，右下 stencil
#    负责细纹理，职责清晰；控制象限颜色不出现在输出，避免上方噪声干扰判断。
#
# g) 颜色变化可推断性
#    本变体不引入新的颜色映射。唯一变化是同一 stencil 在相邻块中的朝向变化，所以生成器只需
#    保证 fill/stencil 颜色集尽量分离、并让差异在翻转后可见即可。


def _mirrored_local_index(idx: int, tile_size: int, block_idx: int) -> int:
    """Return the local coordinate inside one repeated block under mirror tiling."""
    local = int(idx % tile_size)
    if int(block_idx) % 2 == 1:
        return int(tile_size - 1 - local)
    return local


def _mirrored_stencil_lookup(stencil: np.ndarray, r: int, c: int) -> int:
    """Read stencil[r,c] after applying the mirrored-repeat transform."""
    q = int(stencil.shape[0])
    block_r = int(r // q)
    block_c = int(c // q)
    local_r = _mirrored_local_index(int(r), q, block_r)
    local_c = _mirrored_local_index(int(c), q, block_c)
    return int(stencil[local_r, local_c])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the two upper control quadrants to size an output driven by the lower tiles.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants encode output height and width by pixel count.
    - The lower-left quadrant is a coarse fill tile, and the lower-right quadrant is either a stencil tile or empty.

    Rule:
    - Output height is the number of non-black pixels in the upper-left quadrant.
    - Output width is the number of non-black pixels in the upper-right quadrant.
    - If the lower-right stencil is empty, repeat the lower-left tile directly over the output.
    - Otherwise, repeat the stencil with mirror tiling: odd repeated block rows are vertically flipped, odd block columns are horizontally flipped; black stencil cells are replaced by the lower-left coarse color at the repeated block coordinate.

    Key insight:
    - The top half still supplies only dimensions, but the lower-right tile is no longer copied in one fixed orientation: neighboring repeats alternate by reflection, so the solver must track repeated-block parity instead of simple modulo repetition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 0 and w % 2 == 0

    bg = int(Color.BLACK)
    qh, qw = h // 2, w // 2
    assert qh == qw
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

    for r in range(out_h):
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == bg:
                fill_r = int(r // qh)
                fill_c = int(c // qw)
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


# --------------- generator (o5x, D5 mirrored-tiling variant) ---------------

# ── configuration constants ──────────────────────────────────────────
BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (4, 8, 16)                 # allowed input grid side-lengths N
STENCIL_EMPTY_PROB = 0.0                # force the D5 change to actually appear
LL_SOLID_PROB = 0.35                    # lower to make coarse colors more visible through holes
LL_MULTI_COLOR_RANGE = (2, 3)           # color count when fill tile is multi-color
STENCIL_SINGLE_COLOR_PROB = 0.45        # lower to encourage visible asymmetric texture
STENCIL_MULTI_COLOR_RANGE = (2, 4)      # color count when stencil is multi-color
STENCIL_FG_FRAC = (0.30, 0.75)          # fg fraction range for random stencil mode
STENCIL_FRAME_MIN_Q = 4                 # min quadrant size to allow frame stencil
CONTROL_SINGLE_COLOR_PROB = 0.85        # chance control quadrant uses one color
CONTROL_RECT_PROB = 0.85                # chance control pixels form an axis-aligned rect
UL_MAX_COLORS = 3                       # max distinct colors in upper-left control
UR_MAX_COLORS = 2                       # max distinct colors in upper-right control
OUTPUT_DIM_MIN = 3                      # minimum output height/width
OUTPUT_DIM_MAX = 16                     # maximum output height/width
OUTPUT_AREA_MAX = 256                   # maximum output area
GEN_MAX_RETRIES = 300                   # retry budget for generate_input
STENCIL_ATTEMPTS = 40                   # retries to obtain an informative stencil


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


def _make_stencil_base(q: int, palette: list[int]) -> np.ndarray:
    """Build a candidate stencil tile (black = holes filled from fill tile)."""
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


def _stencil_is_informative(stencil: np.ndarray, need_v: bool, need_h: bool) -> bool:
    """Require active flip directions to change the stencil geometry or colors."""
    if not np.any(stencil != BLACK):
        return False
    if need_v and np.array_equal(stencil, stencil[::-1, :]):
        return False
    if need_h and np.array_equal(stencil, stencil[:, ::-1]):
        return False
    return True


def _make_stencil(q: int, palette: list[int], *, need_v: bool, need_h: bool) -> np.ndarray:
    """Build a stencil whose mirrored repeats differ along every active axis."""
    for _ in range(STENCIL_ATTEMPTS):
        tile = _make_stencil_base(q, palette)
        if _stencil_is_informative(tile, need_v, need_h):
            return tile

    # Fallback: forcibly break both symmetries while keeping the tile non-empty.
    tile = np.zeros((q, q), dtype=int)
    r0, c0 = 0, 0
    tile[r0, c0] = palette[0]
    if need_h and q > 1:
        tile[r0, q - 1] = palette[min(1, len(palette) - 1)]
    if need_v and q > 1:
        tile[q - 1, c0] = palette[min(2, len(palette) - 1)]
    if need_v and need_h and q > 1:
        tile[q - 1, q - 1] = BLACK
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly `count` non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
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

def _expected_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Reproduce the solver's mirrored-tiling logic for oracle validation."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % q, c % q])
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = _mirrored_stencil_lookup(stencil, r, c)
            if value == BLACK:
                value = int(fill_tile[r // q, c // q])
            out[r, c] = value
    return out


def _periodic_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Original task output, used only to reject degenerate variant samples."""
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    tiled = tiled.copy()
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid non-degenerate mirrored-tiling input."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB
    if stencil_empty:
        return None

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    remaining_for_stencil = [c for c in COLORS if c not in ll_palette]
    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
    if len(remaining_for_stencil) < st_k:
        return None
    stencil_palette = _palette(st_k, remaining_for_stencil)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    lo = max(OUTPUT_DIM_MIN, q + 1)
    hi = min(OUTPUT_DIM_MAX, q * q)
    valid_dims = [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
        and ceil(out_h / q) >= 2
        and ceil(out_w / q) >= 2
        and ceil(out_h / q) * ceil(out_w / q) >= ll_k
    ]
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)
    tile_rows = int(ceil(out_h / q))
    tile_cols = int(ceil(out_w / q))

    used_fill = [(r, c) for r in range(tile_rows) for c in range(tile_cols)]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = _make_stencil(q, stencil_palette, need_v=tile_rows >= 2, need_h=tile_cols >= 2)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    periodic = _periodic_output(fill_tile, stencil, out_h, out_w)
    diff_pixels = int(np.count_nonzero(expected != periodic))
    if diff_pixels == 0:
        return None

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

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
