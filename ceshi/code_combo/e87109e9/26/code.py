# Combo variant of e87109e9 (index 26)
# combo_spec: D11/0+D15/1
# Name: 多背景分区加穿透转弯

import random

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask
except ImportError:
    from solvers.arc_utils import bbox_coords, components_mask


# Design analysis for the combo variant
# -------------------------------------
# spec:
#   invariant_structure:
#     - 输入仍由 6x24 的顶部图例和下方主区域组成，图例仍只编码“障碍颜色 -> 左/右转”。
#     - 主区域仍有唯一的青色实心方块，射线仍以 sz x sz 的滑动方块形式沿四个基方向发出。
#     - 障碍仍是实心单色矩形；障碍颜色来自图例色集合。
#   fused_variant:
#     - 主区域不再是单一背景，而是 3 个纵向背景带：左带 / 中带 / 右带，各带颜色不同。
#     - 左右两侧背景带中的障碍遵循 D11：射线先穿透整个障碍，在远侧出口格现身后才按图例转弯。
#     - 中间背景带中的障碍改为“完全阻挡”：射线在第一次接触该障碍前就终止，不进入也不转弯。
#     - 为了让“分区而非颜色”这一点对人眼可见，生成器固定让至少一对“同色障碍”跨带出现：
#       同一种颜色会在中带当阻挡墙、在侧带当穿透转门。
#   trace_example:
#     - 设 sz=2，主区域被切成 8|8|8 三个竖带。北向射线在中带撞上横向障碍：
#       原题会在近侧转弯；D11 单变体会穿过去再转；本融合版则直接停在障碍前。
#     - 同时，东向射线在右带撞上同色或异色竖向障碍：
#       它会穿过该障碍，把障碍内部也刷成青色，并在远侧出口格依据图例左/右转后继续走。
#     - 因而同一张输出里同时出现两类证据：中带障碍保持原色并截断光束，侧带障碍内部出现青色通道。
#   complexity_calibration:
#     - 固定为 3 个整齐竖带，避免“区域是谁”本身变成额外谜题；行为档位也只保留两类：阻挡 / 穿透后转弯。
#     - 侧带障碍厚度固定在 3-4，确保“远侧才转弯”的几何位移肉眼可见。
#     - 四束光里固定包含 2 次阻挡和 2 次穿透，样例不会退化成“全阻挡”或“全穿透”。
#   hidden_assumptions_review:
#     - 原 solver 假设主区域只有一个背景色，paint_block() 只在该颜色上涂色；融合后必须把任意背景色都视作可涂区域。
#     - 原 solver 用输出网格 out 来探测障碍；D11 穿透会覆盖障碍颜色，因此碰撞检测必须改读静态 base。
#     - 原 solver 把“障碍颜色”当成“障碍实例”的代号；融合后同色障碍可在不同带里承担不同角色，
#       所以需要改成按连通块识别具体障碍，再用颜色只去查图例方向。
#     - 新 solver 用“非障碍、非青色单元的列众数”来恢复背景带，因此默认背景色不与图例障碍色重合；
#       生成器通过 bg_choices = BG_COLOR_POOL - legend_colors 显式保证这一点。
#     - 原 solver 遇障碍时只有“立刻转弯”这一分支；融合后需要分成：
#       中带 -> 立即停止；侧带 -> 连续穿透同一连通块，出障碍后再转。
#   placement_legality:
#     - 生成器把背景带固定成 8|8|8，所有障碍都完整地落在某一带内部，并与带边界保留 1 格背景缓冲。
#     - 中带上下各放一条横障碍，左右带各放一条竖障碍；四者互不重叠，且与青色方块至少隔 2 格。
#     - 侧带障碍远侧总留有完整出口 footprint，避免“想穿透但没有出口格”的非法样本。
#   observability_and_cleanliness:
#     - 区域差异由整列背景色直接标出，不依赖颜色 id 的数值大小。
#     - 阻挡障碍不被覆盖，穿透障碍内部会出现整段青色通道，两类视觉结果非常干净。
#     - 通过同色跨带复用，解题者能从少量样例排除“某个障碍颜色总是阻挡/总是穿透”的错误归纳。

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

# Color pools
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

# Fixed layout
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]
BAND_W = 8

# Source square
TEAL_SZ2_PROB = 0.75

# Obstacles
OBS_THICKNESS_RANGE = (3, 4)
SPAN_MIN_LEN = 6
SPAN_MAX_EXTRA = 7
MAX_RETRIES = 500


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, counts = np.unique(legend, return_counts=True)
    bg = int(vals[counts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turn: dict[int, str] = {}
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        color = int(next(iter(colors)))
        xs = np.where(interior == color)[1]
        x = int(xs[0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_teal_square(grid: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(grid == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        if int(np.sum(grid[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        candidates.append((int(r0), int(c0), int(hh)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _choose_covering_span(
    anchor0: int,
    anchor1: int,
    allowed0: int,
    allowed1: int,
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    if allowed0 > anchor0 or allowed1 < anchor1:
        return None
    max_len = min(max_len, allowed1 - allowed0 + 1)
    min_len = max(min_len, anchor1 - anchor0 + 1)
    if max_len < min_len:
        return None
    choices: list[tuple[int, int]] = []
    for length in range(min_len, max_len + 1):
        start_lo = max(allowed0, anchor1 - length + 1)
        start_hi = min(anchor0, allowed1 - length + 1)
        for start in range(start_lo, start_hi + 1):
            choices.append((start, start + length - 1))
    if not choices:
        return None
    return random.choice(choices)


def _infer_bands(
    base: np.ndarray, obstacle_colors: set[int]
) -> tuple[list[tuple[int, int, int]], np.ndarray]:
    h, w = base.shape
    col_colors: list[int] = []
    mask = ~np.isin(base, list(obstacle_colors) + [TEAL])
    for c in range(w):
        vals, counts = np.unique(base[mask[:, c], c], return_counts=True)
        assert vals.size, "each column must retain visible background cells"
        col_colors.append(int(vals[counts.argmax()]))

    bands: list[tuple[int, int, int]] = []
    start = 0
    for c in range(1, w + 1):
        if c == w or col_colors[c] != col_colors[start]:
            bands.append((start, c - 1, col_colors[start]))
            start = c
    assert len(bands) == 3, "combo generator must produce three vertical background bands"

    col_to_band = np.empty(w, dtype=int)
    for idx, (c0, c1, _) in enumerate(bands):
        col_to_band[c0 : c1 + 1] = idx
    return bands, col_to_band


def _collect_obstacles(
    base: np.ndarray, obstacle_colors: set[int], col_to_band: np.ndarray
) -> tuple[np.ndarray, dict[int, dict[str, int | tuple[int, int, int, int]]]]:
    comp_grid = np.zeros(base.shape, dtype=int)
    info: dict[int, dict[str, int | tuple[int, int, int, int]]] = {}
    obstacle_mask = np.isin(base, list(obstacle_colors))
    for comp_id, cells in enumerate(components_mask(obstacle_mask, connectivity=4), start=1):
        vals = set(map(int, np.unique(base[tuple(cells.T)])))
        assert len(vals) == 1, "obstacle component must be monochrome"
        color = int(next(iter(vals)))
        r0, c0, r1, c1 = bbox_coords(cells)
        band_ids = {int(col_to_band[int(c)]) for c in cells[:, 1]}
        assert len(band_ids) == 1, "each obstacle must lie inside one background band"
        band = int(next(iter(band_ids)))
        comp_grid[tuple(cells.T)] = comp_id
        info[comp_id] = {"color": color, "band": band, "bbox": (r0, c0, r1, c1)}
    assert len(info) == 4, "combo generator uses four obstacle components"
    return comp_grid, info


def _obstacle_component_at(comp_grid: np.ndarray, r: int, c: int, sz: int) -> int | None:
    ids = set(map(int, np.unique(comp_grid[r : r + sz, c : c + sz])))
    ids.discard(0)
    if not ids:
        return None
    assert len(ids) == 1, "beam footprint must not hit multiple obstacle components"
    return int(next(iter(ids)))


def _solve_cropped(
    base: np.ndarray, legend_turn: dict[int, str]
) -> tuple[np.ndarray, list[list[tuple[str, int, int, int, int, int, int]]]]:
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    obstacle_colors = set(map(int, legend_turn))
    bands, col_to_band = _infer_bands(base, obstacle_colors)
    middle_band = len(bands) // 2
    comp_grid, obstacle_info = _collect_obstacles(base, obstacle_colors, col_to_band)

    out = base.copy()
    h, w = out.shape
    traces: list[list[tuple[str, int, int, int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        out[r : r + sz, c : c + sz] = TEAL

    for d0 in range(4):
        r, c, d = sr, sc, d0
        beam_trace: list[tuple[str, int, int, int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            comp_id = _obstacle_component_at(comp_grid, nr, nc, sz)
            if comp_id is None:
                r, c = nr, nc
                paint(r, c)
                continue

            info = obstacle_info[comp_id]
            color = int(info["color"])
            band = int(info["band"])
            if band == middle_band:
                beam_trace.append(("block", color, comp_id, nr, nc, nr, nc))
                break

            exited = False
            entry_r, entry_c = nr, nc
            while True:
                r, c = nr, nc
                paint(r, c)
                nnr, nnc = r + dr, c + dc
                if not in_bounds(nnr, nnc):
                    break
                next_comp = _obstacle_component_at(comp_grid, nnr, nnc, sz)
                if next_comp == comp_id:
                    nr, nc = nnr, nnc
                    continue
                assert next_comp is None, "transparent path must exit into empty background"
                r, c = nnr, nnc
                paint(r, c)
                beam_trace.append(("pass", color, comp_id, entry_r, entry_c, r, c))
                exited = True
                break
            if not exited:
                break
            d = _rot(d, legend_turn[color])
        traces.append(beam_trace)

    return out, traces


def _solve_edge_turn_all(base: np.ndarray, legend_turn: dict[int, str]) -> np.ndarray:
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    obstacle_colors = set(map(int, legend_turn))
    _, col_to_band = _infer_bands(base, obstacle_colors)
    comp_grid, obstacle_info = _collect_obstacles(base, obstacle_colors, col_to_band)

    out = base.copy()
    h, w = out.shape

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint_background(r: int, c: int) -> None:
        blk = base[r : r + sz, c : c + sz]
        dst = out[r : r + sz, c : c + sz]
        dst[~np.isin(blk, list(obstacle_colors) + [TEAL])] = TEAL
        out[r : r + sz, c : c + sz] = dst

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break
            comp_id = _obstacle_component_at(comp_grid, nr, nc, sz)
            if comp_id is None:
                r, c = nr, nc
                paint_background(r, c)
                continue
            color = int(obstacle_info[comp_id]["color"])
            d = _rot(d, legend_turn[color])

    return out


def _solve_transparent_all(base: np.ndarray, legend_turn: dict[int, str]) -> np.ndarray:
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    obstacle_colors = set(map(int, legend_turn))
    _, col_to_band = _infer_bands(base, obstacle_colors)
    comp_grid, obstacle_info = _collect_obstacles(base, obstacle_colors, col_to_band)

    out = base.copy()
    h, w = out.shape

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        out[r : r + sz, c : c + sz] = TEAL

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break
            comp_id = _obstacle_component_at(comp_grid, nr, nc, sz)
            if comp_id is None:
                r, c = nr, nc
                paint(r, c)
                continue
            color = int(obstacle_info[comp_id]["color"])
            exited = False
            while True:
                r, c = nr, nc
                paint(r, c)
                nnr, nnc = r + dr, c + dc
                if not in_bounds(nnr, nnc):
                    break
                next_comp = _obstacle_component_at(comp_grid, nnr, nnc, sz)
                if next_comp == comp_id:
                    nr, nc = nnr, nnc
                    continue
                assert next_comp is None
                r, c = nnr, nnc
                paint(r, c)
                exited = True
                break
            if not exited:
                break
            d = _rot(d, legend_turn[color])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal square-beams across a partitioned board whose bands change obstacle behavior.

    Input structure:
    - Top band: a grey legend with four cells; each colored vertical bar sits on the left or right side of its cell to encode counterclockwise vs clockwise turning for that obstacle color.
    - Below: a main board split into three vertical background-color bands, containing one solid teal source square and several solid colored obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - Launch one sz-by-sz teal beam from the source square in each cardinal direction and paint every footprint the beam actually visits.
    - Obstacles in the middle background band are opaque walls: a beam that would enter one stops immediately and does not paint the wall.
    - Obstacles in the left or right background bands are semi-transparent turn gates: the beam passes through the whole obstacle, paints its interior, steps out on the far side, and only then turns according to the legend entry for that obstacle color.
    - After any turn, the beam continues under the same regional rule until the next step would leave the board or a middle-band wall blocks it.

    Key insight: Two signals must be fused: the legend still maps obstacle color to left/right turning, but the background band containing the obstacle decides whether that turn happens at all or whether the beam is simply blocked.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend must be confined to the top band"

    legend_turn = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    out, _ = _solve_cropped(base, legend_turn)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    turns = {color: random.choice(["L", "R"]) for color in legend_colors}

    bg_choices = [color for color in BG_COLOR_POOL if color not in legend_colors]
    assert len(bg_choices) >= 3
    left_bg, middle_bg, right_bg = random.sample(bg_choices, 3)

    tn = random.randint(*OBS_THICKNESS_RANGE)
    ts = random.randint(*OBS_THICKNESS_RANGE)
    tw = random.randint(*OBS_THICKNESS_RANGE)
    te = random.randint(*OBS_THICKNESS_RANGE)

    left_start, left_end = 0, BAND_W - 1
    center_start, center_end = BAND_W, 2 * BAND_W - 1
    right_start, right_end = 2 * BAND_W, GRID_W - 1

    sr_lo = tn + 3
    sr_hi = h - ts - sz - 3
    sc_lo = center_start + 2
    sc_hi = center_end - sz - 1
    if sr_lo > sr_hi or sc_lo > sc_hi:
        return None
    sr = random.randint(sr_lo, sr_hi)
    sc = random.randint(sc_lo, sc_hi)

    north_span = _choose_covering_span(
        sc,
        sc + sz - 1,
        center_start + 1,
        center_end - 1,
        SPAN_MIN_LEN,
        center_end - center_start - 1,
    )
    south_span = _choose_covering_span(
        sc,
        sc + sz - 1,
        center_start + 1,
        center_end - 1,
        SPAN_MIN_LEN,
        center_end - center_start - 1,
    )
    west_span = _choose_covering_span(
        sr,
        sr + sz - 1,
        1,
        h - 2,
        SPAN_MIN_LEN,
        min(h - 2, sz + SPAN_MAX_EXTRA),
    )
    east_span = _choose_covering_span(
        sr,
        sr + sz - 1,
        1,
        h - 2,
        SPAN_MIN_LEN,
        min(h - 2, sz + SPAN_MAX_EXTRA),
    )
    if None in (north_span, south_span, west_span, east_span):
        return None
    n_c0, n_c1 = north_span
    s_c0, s_c1 = south_span
    w_r0, w_r1 = west_span
    e_r0, e_r1 = east_span

    north_top_hi = sr - tn - 2
    south_top_lo = sr + sz + 2
    west_col_lo = sz
    west_col_hi = left_end - tw
    east_col_lo = right_start + 1
    east_col_hi = min(right_end - te, GRID_W - sz - te)
    if north_top_hi < 1 or south_top_lo > h - ts - 2:
        return None
    if west_col_lo > west_col_hi or east_col_lo > east_col_hi:
        return None

    north_top = random.randint(1, north_top_hi)
    south_top = random.randint(south_top_lo, h - ts - 2)
    west_col = random.randint(west_col_lo, west_col_hi)
    east_col = random.randint(east_col_lo, east_col_hi)

    shared_block = random.choice([_N, _S])
    shared_outer = random.choice([_W, _E])
    shared_color, color_b, color_c = random.sample(legend_colors, 3)
    side_color = {
        shared_block: shared_color,
        shared_outer: shared_color,
    }
    remaining = [side for side in (_N, _S, _W, _E) if side not in side_color]
    side_color[remaining[0]] = color_b
    side_color[remaining[1]] = color_c

    base = np.empty((h, GRID_W), dtype=int)
    base[:, :BAND_W] = left_bg
    base[:, BAND_W : 2 * BAND_W] = middle_bg
    base[:, 2 * BAND_W :] = right_bg
    base[sr : sr + sz, sc : sc + sz] = TEAL

    rects = {
        _N: (north_top, north_top + tn - 1, n_c0, n_c1),
        _S: (south_top, south_top + ts - 1, s_c0, s_c1),
        _W: (w_r0, w_r1, west_col, west_col + tw - 1),
        _E: (e_r0, e_r1, east_col, east_col + te - 1),
    }

    occupied = np.zeros_like(base, dtype=bool)
    occupied[sr : sr + sz, sc : sc + sz] = True
    for side in (_N, _S, _W, _E):
        r0, r1, c0, c1 = rects[side]
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    try:
        combo_out, traces = _solve_cropped(base, turns)
        edge_out = _solve_edge_turn_all(base, turns)
        transparent_out = _solve_transparent_all(base, turns)
    except AssertionError:
        return None

    flat_events = [event for beam in traces for event in beam]
    block_events = [event for event in flat_events if event[0] == "block"]
    pass_events = [event for event in flat_events if event[0] == "pass"]
    if len(flat_events) != 4 or len(block_events) != 2 or len(pass_events) != 2:
        return None

    obstacle_colors = set(legend_colors)
    bands, col_to_band = _infer_bands(base, obstacle_colors)
    middle_band = len(bands) // 2
    comp_grid, obstacle_info = _collect_obstacles(base, obstacle_colors, col_to_band)
    comp_modes = {cid: ("block" if int(info["band"]) == middle_band else "pass") for cid, info in obstacle_info.items()}
    if sorted(comp_modes.values()) != ["block", "block", "pass", "pass"]:
        return None

    outer_teal = 0
    middle_teal = 0
    for comp_id, info in obstacle_info.items():
        r0, c0, r1, c1 = info["bbox"]
        teal_count = int(np.count_nonzero(combo_out[r0 : r1 + 1, c0 : c1 + 1] == TEAL))
        if comp_modes[comp_id] == "pass":
            outer_teal += teal_count
        else:
            middle_teal += teal_count
    if outer_teal < 2 * sz or middle_teal != 0:
        return None

    diff_vs_edge = int(np.count_nonzero(combo_out != edge_out))
    diff_vs_transparent = int(np.count_nonzero(combo_out != transparent_out))
    if diff_vs_edge < 3 * sz or diff_vs_transparent < 4 * sz:
        return None

    shared_pairs = [
        (side_a, side_b)
        for side_a in (_N, _S)
        for side_b in (_W, _E)
        if side_color[side_a] == side_color[side_b]
    ]
    if len(shared_pairs) != 1:
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, combo_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
