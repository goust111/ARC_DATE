# Variant of f931b4a8
# Dimension: D1 — 输出尺寸的编码方式 (symbol_instructions)
# Change: 左上象限前景像素计数 = 输出行数，右上象限前景像素计数 = 输出列数
#   → 用连通分量数量编码尺寸

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color, find_components


# Design analysis for the D1 variant
# ==================================
# variant_spec:
#   concrete_behavior:
#     - 输入仍然是偶数边长的四象限正方形；下半部分的颜色源瓦片与模板瓦片规则不变。
#     - 变化只发生在上方两个控制象限：它们不再通过“前景像素总数”编码尺寸，而是通过
#       “彼此断开的前景色块数量”编码尺寸。这里按 4 邻接把非黑像素分成离散群组。
#     - 例如一个 4x4 控制象限里，若出现两个横向 1x2 小条和一个单像素点，则旧题会把
#       2+2+1=5 当作尺寸；本变体应把 3 个分离色块当作尺寸。
#   complexity_calibration:
#     - 若所有控制块都退化成单像素，连通分量数就等于像素数，样例会退化回原题。
#     - 因此生成器强制每个控制象限至少有一个多像素连通块，其余块可为单像素或少量
#       第二个扩展块；这样人类必须先做分割，再计数，且新旧规则在多数样例上会给出不同尺寸。
#     - 控制色块之间始终留黑色正交间隔，不依赖“颜色不同所以算不同块”这种隐性规则。
#   implicit_assumptions_from_original_main:
#     - `upper_left` / `upper_right` 的角色不变，仍然只负责输出高宽。
#     - 原代码中 `np.count_nonzero(...)` 依赖“尺寸 = 像素数”；该前提已失效，必须改为组件计数。
#     - 下方两个象限的切分、模板平铺、模板黑洞回填、以及回填使用粗粒度坐标
#       `fill_tile[r // q, c // q]` 的前提都保持成立，因此相关循环可原样保留。
#   placement_legality:
#     - 控制象限的组件先放在棋盘格锚点上，天然避免不同组件正交接触；随后只给少量组件
#       向安全邻格扩展 1 个像素，保证不会越界、不会与别的组件合并。
#     - 输出尺寸仍限制在 3..16 且面积不超过 256，回填时 `ceil(out_h / q)` 与 `ceil(out_w / q)`
#       对应的 lower-left 粗粒度坐标始终落在瓦片内。
#   visual_observability:
#     - 人类从样例中能直接看到“上方是几个断开的块”而不是“上方涂了多少格”；
#       多像素块保证这一差异在视觉上可观察。
#     - 不使用颜色编号做排序或变换；控制区颜色只起到分隔视觉块的作用。
#   cleanliness:
#     - 四象限分工仍然清晰：上方黑底控制块、左下纯非黑填色、右下模板或空白模板。
#     - 不把新标记叠加到下方功能区，避免把 D1 变体和原始底部规则混在一起。
#   new_main_assumptions_audited:
#     - `main()` 现在假设上方组件以 4 邻接定义；生成器显式按该约束构造。
#     - `main()` 不要求组件单色或单像素；生成器虽然主要生成单色小块，但保留多色控制板的自由度。


def _count_control_components(tile: np.ndarray, bg: int) -> int:
    """Count 4-connected non-background groups inside one control quadrant."""
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
    Read the two upper control quadrants as counts of disconnected blobs, then size an output built from the lower tiles.

    Input structure:
    - The grid is an even square divided into four equal quadrants on a black background.
    - The upper-left and upper-right quadrants contain separated foreground control blobs.
    - The lower-left quadrant is an all-nonblack fill tile, and the lower-right quadrant is either a stencil tile or fully black.

    Rule:
    - Output height is the number of 4-connected foreground components in the upper-left quadrant.
    - Output width is the number of 4-connected foreground components in the upper-right quadrant.
    - If the lower-right stencil is empty, repeat the lower-left tile directly across the output.
    - Otherwise tile the stencil over the output; whenever the stencil contributes black, replace that cell with the lower-left color selected by the coarse tile position.

    Key insight:
    - The lower half still defines the repeated texture exactly as before, but the top half now encodes size through segmentation: you must count disconnected blobs rather than foreground area.
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

    out_h = _count_control_components(upper_left, bg)
    out_w = _count_control_components(upper_right, bg)
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


# --------------- generator (D1 variant) ---------------

BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)                    # use q>=4 so component-count dimensions 3..16 stay feasible
MAX_COMPONENTS_BY_Q = {4: 6, 8: 16}     # chosen to keep room for at least one multi-cell component
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
OUTPUT_DIM_MIN = 3
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
    return np.array(
        [random.choice(palette) for _ in range(int(np.prod(shape)))],
        dtype=int,
    ).reshape(shape)


def _checkerboard_coords(q: int, parity: int) -> list[tuple[int, int]]:
    """Cells on one checkerboard parity; these anchors are never 4-neighbors."""
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

    A candidate extension cell may touch occupied cells only if those occupied cells already
    belong to the same component.
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
    """
    Build a control quadrant with exactly `count` disconnected foreground components.

    At least one component is expanded beyond a single pixel so the sample does not collapse
    back to the old pixel-count rule.
    """
    assert 1 <= count <= MAX_COMPONENTS_BY_Q[q]
    for _ in range(CONTROL_BUILD_RETRIES):
        parity = random.randrange(2)
        anchors = _checkerboard_coords(q, parity)
        chosen = set(anchors[:count])
        component_map = {anchor: {anchor} for anchor in chosen}
        occupied = set(chosen)

        expansions = 0
        expansion_goal = 1
        if count >= 4 and random.random() < CONTROL_EXTRA_EXPAND_PROB:
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

        k = 1 if random.random() < CONTROL_SINGLE_COLOR_PROB else random.randint(
            2, min(max_colors, count, len(pool))
        )
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

        assert _count_control_components(tile, BLACK) == count
        assert int(np.count_nonzero(tile != BLACK)) > count
        return tile

    raise ValueError(f"failed to build a control quadrant with {count} components in q={q}")


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Reproduce the solver's tiling logic for oracle validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q]
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _valid_dims_for_q(q: int, stencil_empty: bool, ll_k: int) -> list[tuple[int, int]]:
    max_dim = min(OUTPUT_DIM_MAX, MAX_COMPONENTS_BY_Q[q])
    valid: list[tuple[int, int]] = []
    for out_h in range(OUTPUT_DIM_MIN, max_dim + 1):
        for out_w in range(OUTPUT_DIM_MIN, max_dim + 1):
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            if not stencil_empty and ceil(out_h / q) * ceil(out_w / q) < ll_k:
                continue
            valid.append((out_h, out_w))
    return valid


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
        dims = _valid_dims_for_q(q, stencil_empty, ll_k)
        if dims:
            valid_by_q[q] = dims
    if not valid_by_q:
        return None

    q = random.choice(list(valid_by_q))
    out_h, out_w = random.choice(valid_by_q[q])
    n = 2 * q

    used_fill = [(r, c) for r in range(ceil(out_h / q)) for c in range(ceil(out_w / q))]
    fill_tile = _make_lower_left(q, ll_palette, used_fill)
    stencil = np.zeros((q, q), dtype=int) if stencil_empty else _make_stencil(q, stencil_palette)

    try:
        upper_left = _make_control(q, out_h, UL_MAX_COLORS, control_pool)
        upper_right = _make_control(q, out_w, UR_MAX_COLORS, control_pool)
    except ValueError:
        return None

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
    if int(np.count_nonzero(upper_left != BLACK)) == out_h:
        return None
    if int(np.count_nonzero(upper_right != BLACK)) == out_w:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
