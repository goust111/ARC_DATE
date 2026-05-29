# Combo variant of f931b4a8 (index 2)
# combo_spec: D1/2+D7/1
# Name: 连通分量计数+模板倍数输出

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color, find_components


# Design analysis for the combo variant
# =====================================
# combo_behavior:
#   concrete_behavior_and_trace:
#     - 输入仍然是四象限偶数正方形：左下提供全非黑颜色源瓦片，右下提供模板瓦片或空模板。
#     - 融合点 1：上方两个控制象限不再数前景像素，而是数 4 邻接前景连通分量。
#     - 融合点 2：这些计数不再直接等于输出像素高宽，而是等于模板沿纵向/横向重复的次数。
#     - 例：q=4 时，若左上有 3 个分离色块，总共占 5 个像素；右上有 2 个分离色块，总共占 3 个像素。
#       原题会输出 5x3；仅做 D1 会输出 3x2；本融合题应理解为 repeat_h=3、repeat_w=2，
#       所以输出是 (3*4)x(2*4)=12x8，再按右下模板细粒度平铺，并在模板黑洞处用左下的粗粒度块色回填。
#   complexity_calibration:
#     - 若控制象限所有分量都退化成单像素，人类可能把“组件数”误看成“像素数”。
#     - 因此生成器强制左右控制区各至少有一个多像素分量，并验证像素数严格大于分量数。
#     - 若 repeat_h=repeat_w=1，则输出只是一块模板，D7 的“整块倍数”语义不明显；因此生成器禁掉 (1,1)。
#     - 为避免 `fill_tile[r // q, c // q]` 越界，重复次数限制为 1..q；这也让输出面积保持在 256 以内。
#   implicit_assumptions_review:
#     - 原 `np.count_nonzero(upper_left != bg)` 假设“像素数就是尺寸”，融合后失效，必须改为组件计数。
#     - 原 `out_h/out_w` 直接当最终尺寸，融合后改成 `repeat * q`；否则无法体现 D7 的整模板输出。
#     - 原回填逻辑 `fill_r = r // qh`, `fill_c = c // qw` 现在依赖 `repeat_h <= qh`、`repeat_w <= qw`；
#       生成器显式满足该条件，因此 `fill_r` 和 `fill_c` 始终落在左下 qxq 瓦片内。
#     - `find_components(..., monochromatic=False, connectivity=4)` 假设不同颜色但正交接触的像素属于同一块；
#       生成器通过黑色间隔分开组件，不依赖“颜色不同就不连通”的隐含规则。
#   placement_legality:
#     - 控制组件先放在同一棋盘格奇偶性的锚点上，天然不会 4 邻接。
#     - 之后只允许把组件向“不会碰到别的组件”的安全邻格扩展 1 格，保证不会越界、不会合并。
#     - 下半部分沿用原题的清晰四象限布局，无重叠；左下始终全非黑，右下非空模板始终保留黑洞。
#   visual_observability:
#     - 人类可以直接看到上方是“几个断开的块”，并且这些块的面积和输出倍数不一致，迫使解题者先分割再计数。
#     - 输出总是由完整 qxq 模板块拼成，而不是被裁成任意尺寸；相邻模板块黑洞的颜色取自左下不同粗块坐标。
#     - 不使用颜色编号大小、排序或其他不可视因素；颜色只用于视觉分区，不参与数值映射。
#   visual_cleanliness_and_inference:
#     - 上半区是黑底控制块，下半区是颜色源与模板，两层职责仍然分明，没有叠加标记。
#     - 左下使用实色/条纹/对角线，右下使用棋盘/边框/条纹/散点，便于肉眼区分“粗粒度颜色源”和“细粒度模板”。
#     - 颜色种类保持少量，且输出仅由左下和右下颜色组成，因此规律能在少数样例中稳定归纳。
#   new_main_assumptions_audited:
#     - `main()` 新增假设 1：控制区至少各有 1 个前景组件；生成器只枚举正重复次数。
#     - `main()` 新增假设 2：组件数不会超过 q；生成器只挑选 `repeat <= q` 的样本。
#     - `main()` 新增假设 3：左右控制区的像素块即使多色也按整体前景连通计数；生成器允许多色但不靠颜色断块。


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)                   # q=4 or 8; q=2 太小，不利于“多像素分量 + 可见倍数输出”
STENCIL_EMPTY_PROB = 0.16
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_SINGLE_COLOR_PROB = 0.85
UL_MAX_COLORS = 3
UR_MAX_COLORS = 2
CONTROL_EXTRA_EXPAND_PROB = 0.45
CONTROL_BUILD_RETRIES = 150
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 300


def _count_control_components(tile: np.ndarray, bg: int) -> int:
    """Count 4-connected foreground components in one control quadrant."""
    comps = find_components(
        tile,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    return len(comps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Interpret the upper quadrants as disconnected control blobs whose counts scale a templated output built from the lower quadrants.

    Input structure:
    - The grid is an even square split into four equal quadrants on a black background.
    - The upper-left and upper-right quadrants contain separated foreground blobs that act as vertical and horizontal repeat controls.
    - The lower-left quadrant is a non-black color-source tile, and the lower-right quadrant is either a stencil tile or an all-black empty template.

    Rule:
    - Let q be the quadrant side length. Count 4-connected foreground components in the upper-left and upper-right quadrants to get `repeat_h` and `repeat_w`.
    - The output size is `(repeat_h * q) x (repeat_w * q)`, so the output is always an integer number of full template copies.
    - If the lower-right quadrant is empty, repeat the lower-left tile directly across the output.
    - Otherwise tile the stencil over the output; whenever a stencil cell is black, replace it with the lower-left color chosen by the coarse template-block coordinate `(r // q, c // q)`.

    Key insight:
    - The bottom half still defines the same two-level texture as the original task, but the top half must be parsed in two steps: segment disconnected blobs, then treat those blob counts as template repeat multipliers rather than literal pixel dimensions.
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

    repeat_h = _count_control_components(upper_left, bg)
    repeat_w = _count_control_components(upper_right, bg)
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


def _checkerboard_coords(q: int, parity: int) -> list[tuple[int, int]]:
    """Cells on one checkerboard parity; no two are 4-neighbors."""
    coords = [(r, c) for r in range(q) for c in range(q) if (r + c) % 2 == parity]
    random.shuffle(coords)
    return coords


def _neighbors4_in_bounds(q: int, r: int, c: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr, cc = r + dr, c + dc
        if 0 <= rr < q and 0 <= cc < q:
            out.append((rr, cc))
    return out


def _safe_extension_cells(
    component_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    q: int,
) -> list[tuple[int, int]]:
    """
    Return cells that can be added to this component without merging into another one.

    A candidate extension may only touch occupied cells that already belong to the same component.
    """
    options: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in component_cells:
        for rr, cc in _neighbors4_in_bounds(q, r, c):
            cand = (rr, cc)
            if cand in seen or cand in occupied:
                continue
            seen.add(cand)
            if any(
                nb in occupied and nb not in component_cells
                for nb in _neighbors4_in_bounds(q, rr, cc)
            ):
                continue
            options.append(cand)
    return options


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
    """
    Build a control quadrant with exactly `count` disconnected foreground components.

    At least one component is expanded beyond a single pixel so component-count != pixel-count.
    """
    assert 1 <= count <= q
    for _ in range(CONTROL_BUILD_RETRIES):
        parity = random.randrange(2)
        anchors = _checkerboard_coords(q, parity)
        chosen = set(anchors[:count])
        component_map = {anchor: {anchor} for anchor in chosen}
        occupied = set(chosen)

        expansions = 0
        expansion_goal = 1
        if count >= 3 and random.random() < CONTROL_EXTRA_EXPAND_PROB:
            expansion_goal += 1

        while expansions < expansion_goal:
            expandable = []
            for anchor, cells in component_map.items():
                if len(cells) > 1:
                    continue
                options = _safe_extension_cells(cells, occupied, q)
                if options:
                    expandable.append((anchor, options))
            if not expandable:
                break
            anchor, options = random.choice(expandable)
            extra = random.choice(options)
            component_map[anchor].add(extra)
            occupied.add(extra)
            expansions += 1

        if expansions < 1:
            continue

        max_k = min(max_colors, count, len(pool))
        if max_k == 1 or random.random() < CONTROL_SINGLE_COLOR_PROB:
            k = 1
        else:
            k = random.randint(2, max_k)
        pal = _palette(k, pool)
        tile = np.full((q, q), BLACK, dtype=int)

        anchors_order = list(component_map)
        random.shuffle(anchors_order)
        comp_colors: dict[tuple[int, int], int] = {}
        for i, anchor in enumerate(anchors_order):
            if i < len(pal):
                comp_colors[anchor] = pal[i]
            else:
                comp_colors[anchor] = random.choice(pal)

        for anchor, cells in component_map.items():
            color = comp_colors[anchor]
            for r, c in cells:
                tile[r, c] = color

        if _count_control_components(tile, BLACK) != count:
            continue
        if int(np.count_nonzero(tile != BLACK)) <= count:
            continue
        return tile

    raise ValueError(f"failed to build a control quadrant with {count} components in q={q}")


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


def _valid_repeats(q: int, stencil_empty: bool, ll_k: int) -> list[tuple[int, int]]:
    """Enumerate repeat pairs that fit the area cap and keep coarse fill lookups legal."""
    pairs: list[tuple[int, int]] = []
    for repeat_h in range(1, q + 1):
        for repeat_w in range(1, q + 1):
            out_h, out_w = repeat_h * q, repeat_w * q
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            if (repeat_h, repeat_w) == (1, 1):
                continue
            if not stencil_empty and repeat_h * repeat_w < ll_k:
                continue
            pairs.append((repeat_h, repeat_w))
    return pairs


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid grid; return None on failure."""
    stencil_empty = random.random() < STENCIL_EMPTY_PROB

    ll_k = 1 if random.random() < LL_SOLID_PROB else random.randint(*LL_MULTI_COLOR_RANGE)
    ll_palette = _palette(ll_k)
    if stencil_empty:
        stencil_palette: list[int] = []
    else:
        st_k = 1 if random.random() < STENCIL_SINGLE_COLOR_PROB else random.randint(
            *STENCIL_MULTI_COLOR_RANGE
        )
        stencil_palette = _palette(st_k)

    output_colors = set(ll_palette) | set(stencil_palette)
    control_pool = [c for c in COLORS if c not in output_colors]
    if not control_pool:
        return None

    valid_by_q: dict[int, list[tuple[int, int]]] = {}
    for n in GRID_SIZES:
        q = n // 2
        repeats = _valid_repeats(q, stencil_empty, ll_k)
        if repeats:
            valid_by_q[q] = repeats
    if not valid_by_q:
        return None

    q = random.choice(list(valid_by_q))
    repeat_h, repeat_w = random.choice(valid_by_q[q])
    n = 2 * q

    used_fill = [(r, c) for r in range(repeat_h) for c in range(repeat_w)]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)

    try:
        upper_left = _make_control(q, repeat_h, UL_MAX_COLORS, control_pool)
        upper_right = _make_control(q, repeat_w, UR_MAX_COLORS, control_pool)
    except ValueError:
        return None

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

    ul_pixels = int(np.count_nonzero(upper_left != BLACK))
    ur_pixels = int(np.count_nonzero(upper_right != BLACK))
    if ul_pixels <= repeat_h or ur_pixels <= repeat_w:
        return None
    if actual.shape == (ul_pixels, ur_pixels):
        return None
    if actual.shape != (repeat_h * q, repeat_w * q):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
