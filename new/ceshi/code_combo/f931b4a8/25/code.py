# Combo variant of f931b4a8 (index 25)
# combo_spec: D1/2+D2/0+D10/1
# Name: 连通分量计数+行列互换+多色控制通道

from __future__ import annotations

from math import ceil
import random

import numpy as np

from arc_utils import Color, find_components


# Design analysis for the combo variant
# =====================================
# combo_design_analysis:
#   concrete_behavior:
#     - 底部两象限保持原题机制：左下是全非黑颜色源瓦片，右下是模板瓦片或全黑空模板；
#       输出仍然通过“模板平铺 + 黑洞位用 coarse fill 回填”得到。
#     - 改动全部集中在上方控制区，并且三项变化彼此耦合：
#       1) D10/1: 每个控制象限恰有两种非黑颜色，其中只有一种“共享颜色”同时出现在左右两个控制象限；
#          另一个颜色只在本象限出现，充当干扰通道。
#       2) D1/2: 真正控制尺寸的不是像素数，而是这条共享颜色通道在该象限里的 4 邻接连通分量数。
#       3) D2/0: 左上不再控制高度，而是控制宽度；右上控制高度。
#     - 具体小例子（q=4）：
#       左上有红色 5 个断开块 + 黄色 2 个断开块；右上有红色 3 个断开块 + 蓝色 1 个断开块。
#       原题会把左上/右上所有非黑像素数直接读成高/宽；本融合版必须先看出红色是唯一跨两侧共享
#       的控制色，再分别数红色块数，最后执行交叉映射：输出高度=右上红块数=3，输出宽度=左上红块数=5。
#   complexity_calibration:
#     - 若控制色块全部退化成单像素，则“连通分量数=像素数”，D1 会失效；因此生成器强制共享颜色
#       在每个控制象限至少有一个多像素连通块。
#     - 若左右输出尺寸相等，则 D2 的互换不可见；因此生成器强制 out_h != out_w。
#     - 若左右控制区都有许多共同颜色，人类难以判断目标通道；因此限制为“恰有一个共享颜色 +
#       每侧一个本地干扰色”，既有多色语义，又不至于失控。
#   implicit_assumptions_from_original_main:
#     - 原来的 `np.count_nonzero(control != bg)` 同时依赖了“颜色无语义”和“像素数即尺寸”两条前提；
#       现在这两条都不成立，必须先跨象限比对颜色，再按单色组件计数。
#     - 原来 upper-left/upper-right 各自独立解码；现在要同时查看两个控制象限，找出唯一共享颜色。
#     - 原来 upper-left 控高、upper-right 控宽；现在这条映射要反转。
#     - 下半部分的模板平铺、空洞回填、`fill_tile[r // q, c // q]` 的 coarse lookup 仍然成立，
#       所以相关输出构造逻辑可以保留。
#   placement_legality:
#     - 控制组件先放在同一棋盘奇偶性的锚点上，保证不同组件之间不会 4 邻接；随后只给少量共享色
#       组件向安全邻格扩展 1 个像素，既能制造多像素分量，又不会并块或越界。
#     - 输出尺寸仍限制在 3..16 且面积不超过 256；因此 coarse fill 所需的
#       `ceil(out_h / q)`、`ceil(out_w / q)` 坐标始终落在左下瓦片内部。
#   visual_observability:
#     - 人类能直接看到：上方每侧都有两种颜色，但只有一种颜色在左右两侧都出现；这给出了
#       “先分离颜色通道”的结构线索。
#     - 共享颜色被做成断开的多像素块，保证“数块”与“数格子”在视觉上不同。
#     - 生成器拒绝退化样本：如果按原题像素计数、按共享色但不交换行列、或按全部前景组件计数，
#       恰好得到同一输出，就丢弃重生。
#   cleanliness:
#     - 顶部仍然只是黑底控制区，底部仍然只是纹理来源区；没有把额外标记叠加到输出纹理或模板上。
#     - 每个控制象限只用两种非黑色，避免上方通道过多导致画面嘈杂。
#   new_main_assumptions_audited:
#     - `_shared_signal_color()` 假设左右控制象限恰好共享一种非黑颜色；生成器显式使用
#       `shared_color + left_noise + right_noise` 三个互异颜色来保证这一点。
#     - `_count_color_components()` 按 4 邻接、单色分量计数；生成器用黑色正交间隔隔开组件，
#       并验证计数结果与目标值一致。
#     - `main()` 依赖左下瓦片全非黑、右下可空或非空；生成器沿用原题下半区约束并在组装后回跑
#       `main()` 自校验。


BLACK = 0
COLORS = tuple(range(1, 10))

GRID_SIZES = (8, 16)
MAX_SIGNAL_COMPONENTS_BY_Q = {4: 5, 8: 16}
MAX_DISTRACTOR_COMPONENTS_BY_Q = {4: 2, 8: 4}
STENCIL_EMPTY_PROB = 0.16
LL_SOLID_PROB = 0.70
LL_MULTI_COLOR_RANGE = (2, 3)
STENCIL_SINGLE_COLOR_PROB = 0.75
STENCIL_MULTI_COLOR_RANGE = (2, 4)
STENCIL_FG_FRAC = (0.30, 0.75)
STENCIL_FRAME_MIN_Q = 4
CONTROL_BUILD_RETRIES = 180
SIGNAL_EXTRA_EXPAND_PROB = 0.45
OUTPUT_DIM_MIN = 3
OUTPUT_DIM_MAX = 16
OUTPUT_AREA_MAX = 256
GEN_MAX_RETRIES = 500


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
    """Cells that can extend one component without touching another component."""
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


def _count_color_components(tile: np.ndarray, color: int, bg: int) -> int:
    """Count 4-connected components of one selected control color."""
    comps = find_components(
        tile,
        bg=bg,
        colors=[int(color)],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    return len(comps)


def _count_all_foreground_components(tile: np.ndarray, bg: int) -> int:
    """Count 4-connected foreground components while ignoring color labels."""
    comps = find_components(
        tile,
        bg=bg,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    return len(comps)


def _shared_signal_color(left_control: np.ndarray, right_control: np.ndarray, bg: int) -> int:
    """Return the unique non-black color that appears in both upper control quadrants."""
    left_colors = set(map(int, np.unique(left_control))) - {bg}
    right_colors = set(map(int, np.unique(right_control))) - {bg}
    shared = left_colors & right_colors
    assert len(shared) == 1
    return int(next(iter(shared)))


def _expected_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    out_h: int,
    out_w: int,
) -> np.ndarray:
    """Reproduce the lower-half tiling logic for oracle validation."""
    q = fill_tile.shape[0]
    rr, cc = np.mgrid[:out_h, :out_w]
    if not np.any(stencil != BLACK):
        return fill_tile[rr % q, cc % q]
    tiled = stencil[rr % q, cc % q].copy()
    bg_mask = tiled == BLACK
    tiled[bg_mask] = fill_tile[rr[bg_mask] // q, cc[bg_mask] // q]
    return tiled


def _legacy_pixel_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    upper_left: np.ndarray,
    upper_right: np.ndarray,
) -> np.ndarray:
    """Original-task output: all non-black pixels, no color semantics, no swap."""
    out_h = int(np.count_nonzero(upper_left != BLACK))
    out_w = int(np.count_nonzero(upper_right != BLACK))
    return _expected_output(fill_tile, stencil, out_h, out_w)


def _shared_unswapped_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    upper_left: np.ndarray,
    upper_right: np.ndarray,
    bg: int,
) -> np.ndarray:
    """A near-miss baseline that uses the right color channel but forgets the swap."""
    signal = _shared_signal_color(upper_left, upper_right, bg)
    out_h = _count_color_components(upper_left, signal, bg)
    out_w = _count_color_components(upper_right, signal, bg)
    return _expected_output(fill_tile, stencil, out_h, out_w)


def _all_components_swapped_output(
    fill_tile: np.ndarray,
    stencil: np.ndarray,
    upper_left: np.ndarray,
    upper_right: np.ndarray,
    bg: int,
) -> np.ndarray:
    """A near-miss baseline that swaps axes but ignores the color-channel split."""
    out_h = _count_all_foreground_components(upper_right, bg)
    out_w = _count_all_foreground_components(upper_left, bg)
    return _expected_output(fill_tile, stencil, out_h, out_w)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a shared control color from the upper quadrants, count its blobs with swapped axis roles, then tile the lower pattern.

    Input structure:
    - The grid is an even square divided into four equal quadrants on a black background.
    - Each upper control quadrant contains multiple non-black colors, and exactly one of those colors is shared between the two upper quadrants.
    - The lower-left quadrant is an all-nonblack fill tile, and the lower-right quadrant is either a stencil tile or a fully black empty stencil.

    Rule:
    - Find the unique non-black color that appears in both upper control quadrants.
    - Output height is the number of 4-connected components of that shared color in the upper-right quadrant.
    - Output width is the number of 4-connected components of that shared color in the upper-left quadrant.
    - If the lower-right stencil is empty, repeat the lower-left tile directly across the output.
    - Otherwise tile the stencil over the output; whenever the stencil contributes black, replace that cell with the lower-left color selected by the coarse tile position.

    Key insight:
    - The top half now encodes dimensions through a three-step chain: separate color channels, isolate the only color shared across both controls, count that color's disconnected blobs, and then remember that the left/right controls have been swapped with respect to width/height.
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

    signal = _shared_signal_color(upper_left, upper_right, bg)
    out_h = _count_color_components(upper_right, signal, bg)
    out_w = _count_color_components(upper_left, signal, bg)
    assert out_h > 0 and out_w > 0
    assert np.all(fill_tile != bg)

    return _expected_output(fill_tile, stencil, out_h, out_w)


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


def _make_control(
    q: int,
    signal_components: int,
    distractor_components: int,
    signal_color: int,
    distractor_color: int,
) -> np.ndarray:
    """
    Build a control quadrant with one shared signal color and one local distractor color.

    The shared signal color encodes the dimension via component count and is forced to contain
    at least one multi-cell component so the puzzle does not collapse back to pixel counting.
    """
    assert 1 <= signal_components <= MAX_SIGNAL_COMPONENTS_BY_Q[q]
    assert 1 <= distractor_components <= MAX_DISTRACTOR_COMPONENTS_BY_Q[q]
    total_components = signal_components + distractor_components

    for _ in range(CONTROL_BUILD_RETRIES):
        parity = random.randrange(2)
        anchors = _checkerboard_coords(q, parity)
        if total_components > len(anchors):
            continue
        random.shuffle(anchors)
        signal_anchor_list = anchors[:signal_components]
        distractor_anchor_list = anchors[signal_components:total_components]

        component_map: dict[tuple[int, int], set[tuple[int, int]]] = {
            anchor: {anchor} for anchor in signal_anchor_list + distractor_anchor_list
        }
        signal_anchors = set(signal_anchor_list)
        distractor_anchors = set(distractor_anchor_list)
        occupied = set(signal_anchor_list + distractor_anchor_list)

        target_expansions = 1
        if signal_components >= 4 and random.random() < SIGNAL_EXTRA_EXPAND_PROB:
            target_expansions += 1

        expansions = 0
        while expansions < target_expansions:
            expandable = []
            for anchor in signal_anchor_list:
                cells = component_map[anchor]
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

        tile = np.full((q, q), BLACK, dtype=int)
        for anchor in signal_anchors:
            for r, c in component_map[anchor]:
                tile[r, c] = signal_color
        for anchor in distractor_anchors:
            for r, c in component_map[anchor]:
                tile[r, c] = distractor_color

        if _count_color_components(tile, signal_color, BLACK) != signal_components:
            continue
        if _count_color_components(tile, distractor_color, BLACK) != distractor_components:
            continue
        if int(np.count_nonzero(tile == signal_color)) <= signal_components:
            continue
        if _count_all_foreground_components(tile, BLACK) != total_components:
            continue
        return tile

    raise ValueError(
        f"failed to build control q={q} with signal={signal_components}, "
        f"distractor={distractor_components}"
    )


def _valid_dims_for_q(q: int, stencil_empty: bool, ll_k: int) -> list[tuple[int, int]]:
    max_dim = min(OUTPUT_DIM_MAX, MAX_SIGNAL_COMPONENTS_BY_Q[q])
    valid: list[tuple[int, int]] = []
    for out_h in range(OUTPUT_DIM_MIN, max_dim + 1):
        for out_w in range(OUTPUT_DIM_MIN, max_dim + 1):
            if out_h == out_w:
                continue
            if out_h * out_w > OUTPUT_AREA_MAX:
                continue
            if not stencil_empty and ceil(out_h / q) * ceil(out_w / q) < ll_k:
                continue
            valid.append((out_h, out_w))
    return valid


def _choose_distractor_components(q: int, signal_components: int) -> int:
    max_distractor = min(
        MAX_DISTRACTOR_COMPONENTS_BY_Q[q],
        (q * q + 1) // 2 - signal_components,
    )
    candidates = [d for d in range(1, max_distractor + 1) if d != signal_components]
    assert candidates
    return random.choice(candidates)


def _gen_attempt() -> np.ndarray | None:
    """Try once to produce a valid non-degenerate combo example."""
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
    if len(control_pool) < 3:
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

    shared_color, left_noise_color, right_noise_color = random.sample(control_pool, 3)
    left_distractor = _choose_distractor_components(q, out_w)
    right_distractor = _choose_distractor_components(q, out_h)

    try:
        upper_left = _make_control(
            q=q,
            signal_components=out_w,
            distractor_components=left_distractor,
            signal_color=shared_color,
            distractor_color=left_noise_color,
        )
        upper_right = _make_control(
            q=q,
            signal_components=out_h,
            distractor_components=right_distractor,
            signal_color=shared_color,
            distractor_color=right_noise_color,
        )
    except ValueError:
        return None

    expected = _expected_output(fill_tile, stencil, out_h, out_w)
    if np.array_equal(expected, _legacy_pixel_output(fill_tile, stencil, upper_left, upper_right)):
        return None
    if np.array_equal(expected, _shared_unswapped_output(fill_tile, stencil, upper_left, upper_right, BLACK)):
        return None
    if np.array_equal(expected, _all_components_swapped_output(fill_tile, stencil, upper_left, upper_right, BLACK)):
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
    if _shared_signal_color(upper_left, upper_right, BLACK) != shared_color:
        return None
    if _count_color_components(upper_left, shared_color, BLACK) != out_w:
        return None
    if _count_color_components(upper_right, shared_color, BLACK) != out_h:
        return None
    if _count_all_foreground_components(upper_left, BLACK) <= out_w:
        return None
    if _count_all_foreground_components(upper_right, BLACK) <= out_h:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
