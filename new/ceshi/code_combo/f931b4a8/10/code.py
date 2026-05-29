# Combo variant of f931b4a8 (index 10)
# combo_spec: D5/0+D15/2
# Name: 镜像铺贴+边界镜像补全

from __future__ import annotations

import random
from math import ceil

import numpy as np

from arc_utils import Color

# Design analysis
# ===============
# a) 具体化与行为追踪
#    原题输入仍是四象限：左上/右上只负责计数，左下提供 coarse fill 颜色，右下提供 stencil。
#    原题在输出里把 stencil 做普通取模平铺；D5/0 改成按块交替镜像：第 0 个块保持原向，
#    第 1 个块沿共享边翻转，第 2 个块再翻回。这里再叠加 D15/2：如果最后一个块只露出一部分，
#    这部分不再读取“靠左上侧的前缀”，而是读取“靠外边界的后缀”，等价于把残块也从外边界
#    往内镜像补完。
#    具体小例子：Q=4，某一行的 stencil 列序是 [a,b,c,d]，输出宽度设为 6。
#    - 原题周期铺贴：         [a,b,c,d, a,b]
#    - 仅 D5 块间镜像：       [a,b,c,d, d,c]
#    - 本组合 D5 + D15：      [a,b,c,d, c,d]
#    前 4 列体现“相邻块翻转”，最后 2 列体现“残块从外边界朝里取样”。二维情形对行、列两个轴
#    同时独立执行这一规则，右下角残块会同时承受两轴镜像。
#
# b) 复杂度校准
#    若输出只跨 1 个块、或最后残块只有 1 行/1 列、或 stencil 横竖镜像都对称，那么 D15/2
#    的边界补全就会退化得几乎不可见。因此生成器强制：
#    - 只使用 Q>=4 的样本，避免 Q=2 时残块只能宽/高为 1；
#    - 输出尺寸严格位于 (Q, 2Q) 之间且不是 Q+1，从而恰好跨 2x2 个块且残块宽高至少为 2；
#    - stencil 非空且同时破坏横向/纵向镜像对称；
#    - fill/stencil 使用不重叠颜色集，防止镜像差异被颜色巧合掩盖。
#
# c) 对原 main() 的逐段隐式假设审查
#    - 上半部分用非黑像素个数决定 out_h/out_w，这一前提不变。
#    - 空 stencil 分支原本直接平铺 fill_tile；D5/D15 都只修改 stencil 投影逻辑，因此该分支保留。
#    - 原非空分支默认“局部坐标 = 取模坐标”；D5 先把它改成“按块奇偶镜像”，本组合进一步要求：
#      若当前格子位于最后一个不完整块内，则局部坐标要改成从残块外边界向内读取。
#    - 黑洞回填仍然依赖 coarse block 坐标 fill_r = r // Q, fill_c = c // Q。新的镜像只影响
#      stencil 的局部采样，不改变 coarse block 身份，所以这里不能跟着镜像。
#
# d) 放置合法性
#    生成器仍然直接构造四象限，无对象重叠问题。新的关键合法性约束是：
#    - 选择 out_h,out_w 使 ceil(out_dim / Q) == 2，保证 coarse 坐标只访问 fill_tile 的前 2x2。
#    - 残块宽高至少为 2，保证边界镜像在像素层面可见。
#    - out_dim <= 16 且 area <= 256，继续满足原题物理限制。
#
# e) 视觉可观察性
#    观察者需要同时看到两种信号：中间接缝两侧互为镜像，以及最右/最下残块不是简单截断。
#    因此样本会稳定出现 2x2 块布局，并让 stencil 在横翻/竖翻后明显变样。这样人眼从少量样例
#    就能看到“内部块翻转 + 边界残块再翻一次”的双层规律。
#
# f) 可视化整洁度
#    输出仍只由 coarse fill 与 fine stencil 两层组成，没有新增标记。控制象限颜色不进入输出；
#    fill 负责黑洞补色，stencil 负责细纹理，职责与原题一致，所以虽然规则更复杂，视觉层次仍清楚。
#
# g) 变化的可推断性
#    本组合不引入新的颜色映射，变化只来自几何采样坐标。为了让这种几何差异可推断，生成器限制
#    颜色数在少量范围内，并量化拒绝退化样本：组合输出必须同时区别于原题的周期平铺和 D5-only 的
#    普通镜像平铺，否则重采样。


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)                    # use Q>=4 so boundary mirroring remains visible
STENCIL_EMPTY_PROB = 0.0                # combo should always exercise stencil logic
LL_SOLID_PROB = 0.30
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.40
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 400
STENCIL_ATTEMPTS = 50
MIN_DIFF_VS_D5 = 4
MIN_DIFF_VS_PERIODIC = 8


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


def _ordered_coords(q: int, mode: str) -> list[tuple[int, int]]:
    """Return all coordinates in a qxq grid in a layout-dependent order."""
    coords = [(r, c) for r in range(q) for c in range(q)]
    if mode == "rect":
        r0, c0 = random.randint(0, q - 1), random.randint(0, q - 1)
        return [((r + r0) % q, (c + c0) % q) for r, c in coords]
    if mode == "rows":
        rows = list(range(q))
        cols = list(range(q))
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


def _make_lower_left(
    q: int,
    palette: list[int],
    required_coords: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Build the fill tile (all non-black) using simple low-frequency patterns."""
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


def _stencil_is_informative(stencil: np.ndarray) -> bool:
    """Require the stencil to change under both horizontal and vertical reflection."""
    if not np.any(stencil != BLACK):
        return False
    if np.array_equal(stencil, stencil[::-1, :]):
        return False
    if np.array_equal(stencil, stencil[:, ::-1]):
        return False
    return True


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a non-empty asymmetric stencil so both mirror dimensions remain visible."""
    for _ in range(STENCIL_ATTEMPTS):
        tile = _make_stencil_base(q, palette)
        if _stencil_is_informative(tile):
            return tile

    tile = np.zeros((q, q), dtype=int)
    tile[0, 0] = palette[0]
    tile[0, q - 1] = palette[min(1, len(palette) - 1)]
    tile[q - 1, 0] = palette[min(2, len(palette) - 1)]
    tile[q - 1, q - 1] = BLACK
    return tile


def _make_control(q: int, count: int, max_colors: int, pool: list[int]) -> np.ndarray:
    """Build a control quadrant with exactly count non-black pixels."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2,
        min(max_colors, count, len(pool)),
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


def _d5_axis_index(idx: int, tile_size: int) -> int:
    """Mirror-tiled local index used by the D5-only variant."""
    block = int(idx // tile_size)
    local = int(idx % tile_size)
    if block % 2 == 1:
        return int(tile_size - 1 - local)
    return local


def _combo_axis_index(idx: int, out_len: int, tile_size: int) -> int:
    """Local index for D5 mirrored tiles plus D15 reflection of the residual visible interval."""
    block = int(idx // tile_size)
    local = int(idx % tile_size)
    tile_count = int(ceil(out_len / tile_size))
    remainder = int(out_len % tile_size)
    visible = tile_size if remainder == 0 else remainder
    is_last_partial = remainder != 0 and block == tile_count - 1

    local_after_boundary_reflection = int(visible - 1 - local) if is_last_partial else local
    if block % 2 == 1:
        return int(tile_size - 1 - local_after_boundary_reflection)
    return int(local_after_boundary_reflection)


def _d5_stencil_lookup(stencil: np.ndarray, r: int, c: int) -> int:
    """Read stencil under shared-edge mirror tiling only."""
    q = int(stencil.shape[0])
    return int(stencil[_d5_axis_index(r, q), _d5_axis_index(c, q)])


def _combo_stencil_lookup(stencil: np.ndarray, r: int, c: int, out_h: int, out_w: int) -> int:
    """Read stencil under the fused D5+D15 mirror rule."""
    q = int(stencil.shape[0])
    rr = _combo_axis_index(r, out_h, q)
    cc = _combo_axis_index(c, out_w, q)
    return int(stencil[rr, cc])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the upper control quadrants and synthesize a reflected output from the lower tiles.

    Input structure:
    - The grid is an even square split into four equal quadrants.
    - Black is the background.
    - The upper-left and upper-right quadrants encode output height and width by counting non-black pixels.
    - The lower-left quadrant is a coarse fill tile, and the lower-right quadrant is either a stencil tile or empty.

    Rule:
    - Output height is the number of non-black pixels in the upper-left quadrant.
    - Output width is the number of non-black pixels in the upper-right quadrant.
    - If the lower-right stencil is empty, repeat the lower-left tile directly over the output.
    - Otherwise, tile the stencil with two reflection layers: neighboring repeated blocks alternate by mirror flip, and if the final block on an axis is incomplete, that visible fragment is read from the outer edge inward rather than as a simple prefix; black stencil cells are replaced by the lower-left coarse color at the repeated-block coordinate.

    Key insight:
    - The upper half still supplies only dimensions, but the lower-right tile must be sampled with coordinate reflections instead of plain modulo: first by repeated-block parity, then again at the output boundary when the last repeated block is only partial.
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
            value = _combo_stencil_lookup(stencil, r, c, out_h, out_w)
            if value == bg:
                fill_r = int(r // qh)
                fill_c = int(c // qw)
                assert fill_r < qh and fill_c < qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


def _combo_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Oracle output for the fused D5+D15 rule."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % q, c % q])
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = _combo_stencil_lookup(stencil, r, c, out_h, out_w)
            if value == BLACK:
                value = int(fill_tile[r // q, c // q])
            out[r, c] = value
    return out


def _d5_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Output for the D5-only mirrored-tiling variant."""
    q = int(fill_tile.shape[0])
    out = np.empty((out_h, out_w), dtype=int)
    if not np.any(stencil != BLACK):
        for r in range(out_h):
            for c in range(out_w):
                out[r, c] = int(fill_tile[r % q, c % q])
        return out

    for r in range(out_h):
        for c in range(out_w):
            value = _d5_stencil_lookup(stencil, r, c)
            if value == BLACK:
                value = int(fill_tile[r // q, c // q])
            out[r, c] = value
    return out


def _periodic_output(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Output for the original periodic-tiling task."""
    q = int(fill_tile.shape[0])
    rr, cc = np.mgrid[:out_h, :out_w]
    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    tiled = tiled.copy()
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _boundary_mask(out_h: int, out_w: int, q: int) -> np.ndarray:
    """Mask cells that lie in the bottom or right partial mirrored completion zone."""
    mask = np.zeros((out_h, out_w), dtype=bool)
    rem_h = out_h % q
    rem_w = out_w % q
    if rem_h:
        mask[out_h - rem_h :, :] = True
    if rem_w:
        mask[:, out_w - rem_w :] = True
    return mask


def _behavior_metrics(fill_tile: np.ndarray, stencil: np.ndarray, out_h: int, out_w: int) -> dict[str, int]:
    """Quantify whether both mirror layers materially affect the sample."""
    combo = _combo_output(fill_tile, stencil, out_h, out_w)
    d5 = _d5_output(fill_tile, stencil, out_h, out_w)
    periodic = _periodic_output(fill_tile, stencil, out_h, out_w)
    boundary = _boundary_mask(out_h, out_w, int(fill_tile.shape[0]))
    return {
        "diff_vs_d5": int(np.count_nonzero(combo != d5)),
        "diff_vs_periodic": int(np.count_nonzero(combo != periodic)),
        "boundary_diff_vs_d5": int(np.count_nonzero((combo != d5) & boundary)),
    }


def _valid_output_dims(q: int) -> list[tuple[int, int]]:
    """Return dimensions that force exactly 2x2 blocks and visible residual boundaries."""
    lo = q + 2
    hi = min(2 * q - 1, OUTPUT_DIM_MAX)
    return [
        (out_h, out_w)
        for out_h in range(lo, hi + 1)
        for out_w in range(lo, hi + 1)
        if out_h * out_w <= OUTPUT_AREA_MAX
        and ceil(out_h / q) == 2
        and ceil(out_w / q) == 2
        and out_h % q >= 2
        and out_w % q >= 2
    ]


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a non-degenerate fused-mirror sample."""
    n = random.choice(GRID_SIZES)
    q = n // 2

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)

    stencil_color_pool = [c for c in COLORS if c not in ll_palette]
    st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(*STENCIL_MULTI_COLOR_RANGE)
    if len(stencil_color_pool) < st_k:
        return None
    stencil_palette = _palette(st_k, stencil_color_pool)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    valid_dims = _valid_output_dims(q)
    if not valid_dims:
        return None
    out_h, out_w = random.choice(valid_dims)

    used_fill = [(r, c) for r in range(2) for c in range(2)]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)

    metrics = _behavior_metrics(fill_tile, stencil, out_h, out_w)
    if metrics["diff_vs_d5"] < MIN_DIFF_VS_D5:
        return None
    if metrics["boundary_diff_vs_d5"] < MIN_DIFF_VS_D5:
        return None
    if metrics["diff_vs_periodic"] < MIN_DIFF_VS_PERIODIC:
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
    expected = _combo_output(fill_tile, stencil, out_h, out_w)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused-mirror input")
