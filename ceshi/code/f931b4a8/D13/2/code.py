# Variant of f931b4a8
# Dimension: D13 — 多步处理与迭代 (multi_step)
# Change: 单步处理：读取四个象限 -> 计算输出尺寸 -> 铺贴并替换 -> 输出... → 先铺贴再裁剪到控制尺寸

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color

# Design analysis
# ----------------
# variant_spec:
#   unchanged_structure:
#     - 输入仍是偶数边长正方形，分成四个等大象限。
#     - UL/UR 仍然只做控制；LL 仍然是不含黑色的颜色源；LR 仍然是模板/镂空瓦片。
#   new_visible_rule:
#     - 原题直接从无限铺贴图案的原点 (0,0) 取一个 out_h x out_w 的前缀窗口。
#     - 变体先在概念上生成整个无限平面，再裁剪一个带起点偏移的窗口。
#     - 输出高 = UL 非黑像素数；输出宽 = UR 非黑像素数。
#     - 窗口起点行偏移 = UL 前景包围盒的 top；窗口起点列偏移 = UR 前景包围盒的 left。
#     - 因此控制区同时编码“窗口大小”和“窗口起点”，形成两步推理：
#       先从控制象限恢复窗口参数，再把底部图案当作无限平面来裁剪。
#   concrete_small_example:
#     - 设 q=4，UL 有 6 个前景且最上方前景位于第 1 行，所以 out_h=6, row_offset=1。
#     - UR 有 5 个前景且最左前景位于第 2 列，所以 out_w=5, col_offset=2。
#     - 原题会读取绝对坐标 [0:6, 0:5]；变体读取的是 [1:7, 2:7]。
#     - 如果 LR 是棋盘模板，LL 是行条纹，那么输出首格会从不同模板相位开始，肉眼可见。
#   complexity_calibration:
#     - 只新增一个低自由度窗口定位机制：行偏移只来自 UL 的 top，列偏移只来自 UR 的 left。
#     - 不引入颜色排序、对象匹配或多种子规则，避免 2-4 个样例无法归纳。
#     - 为了让偏移可见，generator 强制大多数样例使用非零偏移，并拒绝与原题零偏移输出完全相同的样本。
#   hidden_assumptions_from_original_main:
#     - `fill_r = r // qh`, `fill_c = c // qw` 原先依赖窗口从 0 开始，所以索引不会超过 q-1。
#     - 变体下应改为先加窗口偏移，再对粗粒度索引取模：
#       `fill_r = ((row_offset + r) // q) % q`, `fill_c = ((col_offset + c) // q) % q`。
#     - `stencil[r % q, c % q]` 也必须改成使用绝对坐标的相位：
#       `stencil[(row_offset + r) % q, (col_offset + c) % q]`。
#   placement_legality:
#     - UL 的所有前景必须位于 `r >= row_offset`，且至少有一个落在 `r == row_offset`，从而 bbox top 可恢复。
#     - UR 的所有前景必须位于 `c >= col_offset`，且至少有一个落在 `c == col_offset`，从而 bbox left 可恢复。
#     - 这要求 `out_h <= q * (q - row_offset)`、`out_w <= q * (q - col_offset)`，generator 会显式筛选。
#   observability_and_cleanliness:
#     - 控制象限继续只承担计数/定位职责，不与底部图案重叠。
#     - 通过保留简单矩形或紧凑团块，bbox 顶边/左边能直接看出来，不会产生凌乱标记。
#     - 禁止借助颜色 id 做排序或映射；颜色只用于填充，不参与隐藏控制。
#   new_solver_assumptions_checked_against_generator:
#     - `main()` 假设 UL/UR 各至少有一个前景像素；generator 通过 out_h/out_w >= 3 保证。
#     - `main()` 假设 LL 无黑色；generator 仍保持这一点。
#     - `main()` 假设 LR 若非空则是前景/黑洞混合模板；generator 仍只生成这些模式。
#     - `main()` 假设 bbox top/left 的编码唯一可读；generator 保证所有控制像素都在对应边界之后。


def _control_offsets(upper_left: np.ndarray, upper_right: np.ndarray, bg: int) -> tuple[int, int]:
    """Recover the row/col crop offsets from the control quadrants."""
    ul_coords = np.argwhere(upper_left != bg)
    ur_coords = np.argwhere(upper_right != bg)
    assert ul_coords.size > 0 and ur_coords.size > 0
    row_offset = int(ul_coords[:, 0].min())
    col_offset = int(ur_coords[:, 1].min())
    return row_offset, col_offset


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the top quadrants as a crop-size plus crop-offset code, then cut a window from a repeated lower pattern.

    Input structure:
    - The input is an even square split into four equal quadrants.
    - Black is the background throughout the control regions and the stencil holes.
    - The upper-left and upper-right quadrants are control panels whose foreground counts still encode output height and width.
    - The lower-left quadrant is a non-black coarse color tile, and the lower-right quadrant is a repeated stencil tile that may also be empty.

    Rule:
    - Output height is the number of non-black cells in the upper-left quadrant, and output width is the number of non-black cells in the upper-right quadrant.
    - The top edge of the upper-left foreground bbox gives the row offset of a crop window; the left edge of the upper-right foreground bbox gives the column offset.
    - Conceptually extend the lower pattern over the whole plane, then crop the `out_h x out_w` window starting at that offset.
    - If the stencil is empty, the cropped window comes directly from the repeated lower-left tile; otherwise stencil foreground stays visible and stencil background reveals the coarse lower-left color chosen by absolute super-tile coordinates.

    Key insight:
    - The task is no longer "size then tile from the origin"; the controls first define a shifted window, so both stencil phase and coarse-fill lookup must be computed in absolute coordinates before cropping.
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

    row_offset, col_offset = _control_offsets(upper_left, upper_right, bg)

    out = np.empty((out_h, out_w), dtype=int)
    stencil_empty = not np.any(stencil != bg)
    for r in range(out_h):
        abs_r = row_offset + r
        for c in range(out_w):
            abs_c = col_offset + c
            if stencil_empty:
                value = int(fill_tile[abs_r % qh, abs_c % qw])
                assert value != bg
                out[r, c] = value
                continue

            value = int(stencil[abs_r % qh, abs_c % qw])
            if value == bg:
                fill_r = (abs_r // qh) % qh
                fill_c = (abs_c // qw) % qw
                value = int(fill_tile[fill_r, fill_c])
                assert value != bg
            out[r, c] = value

    return out


# --------------- generator (o5x) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)                   # q=2 cannot express a visible non-zero offset with out_dim >= 3
STENCIL_EMPTY_PROB = 0.18
LL_SOLID_PROB = 0.45
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
CONTROL_RECT_PROB = 0.80
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
SHIFTED_WINDOW_PROB = 0.90
GEN_MAX_RETRIES = 400


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


def _make_lower_left(q: int, palette: list[int]) -> np.ndarray:
    """Build the coarse color tile (must contain no black pixels)."""
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
        coords = [(r, c) for r in range(q) for c in range(q)]
        random.shuffle(coords)
        for color, (r, c) in zip(missing, coords):
            tile[r, c] = color
    return tile


def _make_stencil(q: int, palette: list[int]) -> np.ndarray:
    """Build a mixed stencil tile (black = holes, non-black = visible foreground)."""
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


def _compact_order(coords: list[tuple[int, int]], anchor: tuple[int, int]) -> list[tuple[int, int]]:
    """Sort coords by distance from an anchor, with slight randomness to avoid rigid repeats."""
    random.shuffle(coords)
    return sorted(
        coords,
        key=lambda rc: (abs(rc[0] - anchor[0]) + abs(rc[1] - anchor[1]), random.random()),
    )


def _make_control(
    q: int,
    count: int,
    max_colors: int,
    pool: list[int],
    *,
    fixed_top: int | None = None,
    fixed_left: int | None = None,
) -> np.ndarray:
    """Build a control quadrant with exact count and a recoverable bbox top/left."""
    k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
        2, min(max_colors, count, len(pool))
    )
    pal = _palette(k, pool)
    tile = np.full((q, q), BLACK, dtype=int)

    allowed = [
        (r, c)
        for r in range(q)
        for c in range(q)
        if (fixed_top is None or r >= fixed_top) and (fixed_left is None or c >= fixed_left)
    ]
    assert count <= len(allowed)

    rects: list[tuple[int, int, int, int]] = []
    for h in range(1, q + 1):
        for w in range(1, q + 1):
            if h * w != count:
                continue
            row_starts = [fixed_top] if fixed_top is not None else list(range(q - h + 1))
            col_starts = [fixed_left] if fixed_left is not None else list(range(q - w + 1))
            for r0 in row_starts:
                for c0 in col_starts:
                    if r0 + h <= q and c0 + w <= q:
                        rects.append((r0, c0, h, w))

    if rects and random.random() < CONTROL_RECT_PROB:
        r0, c0, h, w = random.choice(rects)
        tile[r0:r0 + h, c0:c0 + w] = _random_color_array((h, w), pal)
        return tile

    if fixed_top is not None:
        boundary = [(fixed_top, c) for c in range(q)]
        anchor = random.choice(boundary)
    elif fixed_left is not None:
        boundary = [(r, fixed_left) for r in range(q)]
        anchor = random.choice(boundary)
    else:
        anchor = random.choice(allowed)

    ordered = [anchor]
    ordered.extend(_compact_order([coord for coord in allowed if coord != anchor], anchor))
    for r, c in ordered[:count]:
        tile[r, c] = random.choice(pal)
    return tile


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
    row_offset: int,
    col_offset: int,
) -> np.ndarray:
    """Reproduce the shifted-window solver logic for oracle validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    abs_r = rr + row_offset
    abs_c = cc + col_offset

    if not np.any(stencil != BLACK):
        return fill_tile[abs_r % q, abs_c % q]

    tiled = stencil[abs_r % q, abs_c % q].copy()
    bg_mask = tiled == BLACK
    coarse_r = (abs_r // q) % q
    coarse_c = (abs_c // q) % q
    tiled[bg_mask] = fill_tile[coarse_r[bg_mask], coarse_c[bg_mask]]
    return tiled


def _valid_window_configs(q: int) -> list[tuple[int, int, int, int]]:
    """Enumerate offset/size combinations that fit their control quadrants."""
    hi = min(OUTPUT_DIM_MAX, q * q)
    configs: list[tuple[int, int, int, int]] = []
    for row_offset in range(q):
        max_h = min(hi, q * (q - row_offset))
        if max_h < OUTPUT_DIM_MIN:
            continue
        for col_offset in range(q):
            max_w = min(hi, q * (q - col_offset))
            if max_w < OUTPUT_DIM_MIN:
                continue
            for out_h in range(OUTPUT_DIM_MIN, max_h + 1):
                for out_w in range(OUTPUT_DIM_MIN, max_w + 1):
                    if out_h * out_w <= OUTPUT_AREA_MAX:
                        configs.append((row_offset, col_offset, out_h, out_w))
    return configs


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid non-degenerate D13 sample; return None on failure."""
    n = random.choice(GRID_SIZES)
    q = n // 2
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = (
        random.randint(*LL_MULTI_COLOR_RANGE)
        if stencil_empty
        else (1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE))
    )
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

    configs = _valid_window_configs(q)
    shifted = [cfg for cfg in configs if cfg[0] > 0 or cfg[1] > 0]
    if not shifted:
        return None
    if random.random() < SHIFTED_WINDOW_PROB:
        row_offset, col_offset, out_h, out_w = random.choice(shifted)
    else:
        row_offset, col_offset, out_h, out_w = random.choice(configs)

    fill_tile = _make_lower_left(q, ll_palette)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)
    upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool, fixed_top=row_offset)
    upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool, fixed_left=col_offset)

    grid = np.zeros((n, n), dtype=int)
    grid[:q, :q] = upper_left
    grid[:q, q:] = upper_right
    grid[q:, :q] = fill_tile
    grid[q:, q:] = stencil

    expected = _expected_output(fill_tile, stencil, out_h, out_w, row_offset, col_offset)
    baseline = _expected_output(fill_tile, stencil, out_h, out_w, 0, 0)
    if (row_offset == 0 and col_offset == 0) or np.array_equal(expected, baseline):
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
