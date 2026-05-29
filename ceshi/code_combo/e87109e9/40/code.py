# Combo variant of e87109e9 (index 40)
# combo_spec: D11/0+D12/1+D14/0
# Name: 穿透转弯加障碍重涂加顺序发射

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for the combo variant
# -------------------------------------
# concrete_trace:
#   - 保留原题的 6x24 图例、唯一青色源方块、以及若干单色矩形障碍。
#   - 与原题不同，射线现在按上→右→下→左依次发射；每束都是 sz x sz 的滑动方块。
#   - 以一个 3 障碍的 hook 样例说明：
#       1. 上射线先撞到北侧横条 A。原题会在 A 的近侧立即转弯；本变体会穿过 A，并把整块 A 都重涂成青色，
#          然后在 A 远侧出口格转向。
#       2. 这束上射线再撞到西侧竖条 B，同样先整块重涂 B，再穿过后于远侧左出口转弯向下。
#       3. 它继续撞到南侧横条 C，重涂 C 后在远侧出口左转向右。
#       4. 之后轮到下射线；它撞到的 C 已不是原色，而是青色，所以不能再查图例。这里固定约定：
#          “已经变成青色的障碍”在穿透后总是顺时针转向。
#       5. 最后左射线撞到已被上射线染青的 B，也按“青色障碍 = 穿透后顺时针”处理。
#   - 因而同一块障碍在不同时间会先按图例作用、再按“青色固定顺时针”作用；这是顺序依赖的核心。
# complexity_calibration:
#   - 若仍使用贴边障碍，D11 的“穿透后才转弯”几乎看不见，所以这里统一使用悬空矩形并把厚度收紧到 3-4。
#   - 若允许任意顺序依赖，样本会太乱；这里把 generator 限制成一种稳定的上→左→下 hook 模板，
#     让前一束必定先把 B、C 两块障碍染青，再由后两束重复撞上，形成可重复观察的证据。
#   - 只新增一个额外语义档位：青色障碍固定顺时针。这样人类只需归纳“原色看图例、青色看固定规则”。
# hidden_assumptions_review:
#   - 原 solver 用可变输出网格做障碍检测；融合后不成立，因为障碍会被改写成青色，必须先提取静态障碍几何。
#   - 原 solver 假设“看到障碍就原地改方向”；D11 破坏了这一点，必须显式沿障碍内部逐步前进到出口。
#   - 原 solver 默认 8 色只是源和轨迹颜色，可安全忽略；融合后不成立，因为障碍会动态变成 8，并改变后续转向。
#   - D14 原始单变体把所有旧轨迹都当障碍，这里故意不照搬。融合描述强调的是“先重涂障碍，再让后续束看到新颜色”，
#     若把普通轨迹也提升为障碍，会引入额外不可见状态并削弱“障碍颜色变化”这一主线。
# placement_legality:
#   - generator 固定放置 3 个互不重叠的悬空矩形：北横条 A、西竖条 B、南横条 C。
#   - A 必须覆盖源方块的列车道；B 必须同时覆盖 A 的北出口车道和源方块的行车道；
#     C 必须同时覆盖 B 的西出口车道和源方块的列车道。
#   - A 与 B 在列方向保留至少 1 格间隔，B 与 C 在行方向保留至少 1 格间隔，保证穿透出口不会直接撞上另一块。
# observability_and_cleanliness:
#   - 输出里会同时出现三种可见证据：障碍整块变青、转弯点从近侧移到远侧、后续束在同一块障碍上走出不同弯法。
#   - 颜色语义不依赖数值排序；“青色特殊”来自视觉上确实被前一束染青，而不是隐式 id 规则。
#   - 模板化布局让三块障碍与三段主要路径分居不同区域，不会出现大量交叠造成的阅读混乱。

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

TEAL_SZ2_PROB = 0.75
SQUARE_ROW_MARGIN = 7
SQUARE_COL_MARGIN_LEFT = 9
SQUARE_COL_MARGIN_RIGHT = 5

OBS_THICKNESS_RANGE = (3, 4)
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.55

MAX_RETRIES = 500


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Decode obstacle-turn direction from the legend strip."""
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


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square."""
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        candidates.append((int(r0), int(c0), int(hh)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _find_obstacles(
    base: np.ndarray, bg: int, teal: int = TEAL
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int]]:
    """Extract static obstacle geometry as rectangle ids, boxes, and original colors."""
    obstacle_mask = (base != bg) & (base != teal)
    obstacle_ids = np.full(base.shape, -1, dtype=int)
    boxes: list[tuple[int, int, int, int]] = []
    colors: list[int] = []
    for obs_id, cells in enumerate(components_mask(obstacle_mask, connectivity=4)):
        r0, c0, r1, c1 = bbox_coords(cells)
        color_vals = set(map(int, np.unique(base[r0 : r1 + 1, c0 : c1 + 1])))
        color_vals.discard(bg)
        color_vals.discard(teal)
        assert len(color_vals) == 1, "each obstacle must be single-colored"
        color = int(next(iter(color_vals)))
        rect = base[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(rect == color), "each obstacle must be a solid rectangle"
        obstacle_ids[r0 : r1 + 1, c0 : c1 + 1] = obs_id
        boxes.append((int(r0), int(c0), int(r1), int(c1)))
        colors.append(color)
    assert boxes, "main grid must contain at least one obstacle"
    return obstacle_ids, boxes, colors


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _simulate_combo(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Sequential combo oracle: pass through, recolor whole obstacles, then reuse teal state."""
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    obstacle_ids, boxes, original_colors = _find_obstacles(base, bg, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    recolored_mask = 0
    beam_stats: list[dict[str, object]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint_block(r: int, c: int) -> None:
        out[r : r + sz, c : c + sz] = TEAL

    def obstacle_at(r: int, c: int) -> int | None:
        ids = set(map(int, np.unique(obstacle_ids[r : r + sz, c : c + sz])))
        ids.discard(-1)
        if not ids:
            return None
        assert len(ids) == 1, "beam footprint must not hit mixed obstacles"
        return int(next(iter(ids)))

    def recolor_obstacle(obs_id: int) -> None:
        r0, c0, r1, c1 = boxes[obs_id]
        out[r0 : r1 + 1, c0 : c1 + 1] = TEAL

    for d0 in range(4):
        r, c, d = sr, sc, d0
        hits: list[tuple[int, int, int, int, int, int, int, str]] = []
        seen: set[tuple[int, int, int, int]] = set()
        while True:
            state = (r, c, d, recolored_mask)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            obs_id = obstacle_at(nr, nc)
            if obs_id is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            was_teal = bool(recolored_mask & (1 << obs_id))
            color_before = TEAL if was_teal else original_colors[obs_id]
            turn = "R" if was_teal else turns[color_before]
            assert was_teal or color_before in turns, "obstacle color must be encoded by legend"

            recolored_mask |= 1 << obs_id
            recolor_obstacle(obs_id)

            entry_r, entry_c = nr, nc
            exited = False
            while True:
                r, c = nr, nc
                paint_block(r, c)
                nnr, nnc = r + dr, c + dc
                if not in_bounds(nnr, nnc):
                    break
                obs2 = obstacle_at(nnr, nnc)
                if obs2 == obs_id:
                    nr, nc = nnr, nnc
                    continue
                assert obs2 is None, "different obstacles must not touch along a traversed lane"
                r, c = nnr, nnc
                paint_block(r, c)
                exited = True
                break
            if not exited:
                break

            hits.append((obs_id, color_before, entry_r, entry_c, r, c, d, turn))
            d = _rot(d, turn)

        beam_stats.append({"dir": d0, "hits": hits})

    return out, beam_stats


def _simulate_static_colors(base: np.ndarray, turns: dict[int, str], bg: int) -> np.ndarray:
    """Baseline oracle used only to reject degenerate samples with no sequential effect."""
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    obstacle_ids, boxes, original_colors = _find_obstacles(base, bg, teal=TEAL)
    out = base.copy()
    h, w = out.shape

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint_block(r: int, c: int) -> None:
        out[r : r + sz, c : c + sz] = TEAL

    def obstacle_at(r: int, c: int) -> int | None:
        ids = set(map(int, np.unique(obstacle_ids[r : r + sz, c : c + sz])))
        ids.discard(-1)
        if not ids:
            return None
        assert len(ids) == 1
        return int(next(iter(ids)))

    def recolor_obstacle(obs_id: int) -> None:
        r0, c0, r1, c1 = boxes[obs_id]
        out[r0 : r1 + 1, c0 : c1 + 1] = TEAL

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

            obs_id = obstacle_at(nr, nc)
            if obs_id is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            recolor_obstacle(obs_id)
            turn = turns[original_colors[obs_id]]
            exited = False
            while True:
                r, c = nr, nc
                paint_block(r, c)
                nnr, nnc = r + dr, c + dc
                if not in_bounds(nnr, nnc):
                    break
                obs2 = obstacle_at(nnr, nnc)
                if obs2 == obs_id:
                    nr, nc = nnr, nnc
                    continue
                assert obs2 is None
                r, c = nnr, nnc
                paint_block(r, c)
                exited = True
                break
            if not exited:
                break
            d = _rot(d, turn)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire teal square-beams in sequence; obstacles are traversed, recolored, and later reinterpreted.

    Input structure:
    - The top strip is a 6x24 legend: each of the four cells contains one colored vertical bar, and the bar's left/right placement encodes counterclockwise vs clockwise turning for that color.
    - The lower board contains one solid teal source square on a uniform background plus several solid colored rectangular obstacles.

    Rule:
    - Remove the legend strip in the output.
    - Emit four sz x sz beams from the source in the fixed order up, right, down, left.
    - When a beam moves through background, it paints teal normally. When it first meets a colored obstacle, it immediately repaints that whole obstacle teal, keeps moving straight through the obstacle, steps out on the far side, and only then turns.
    - If the obstacle was still in its original color, use the legend to decide left vs right. If an earlier beam had already turned that obstacle teal, then this later beam treats teal as a special case: after exiting, it always turns clockwise.
    - Beams stop only when their next sz x sz step would leave the board.

    Key insight: The legend is no longer the whole story. You must combine three cues at once: turns happen after crossing an obstacle, collisions recolor the whole obstacle to teal, and the firing order matters because later beams may hit an already-teal obstacle that now obeys a different fixed turn rule.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_turn = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
    out, _stats = _simulate_combo(base, legend_turn, bg)
    return out


def _pick_cover_span(
    axis_len: int,
    required: list[tuple[int, int]],
    low: int,
    high: int,
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a 1D span covering all required intervals and staying inside [low, high]."""
    if low > high:
        return None
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    if req0 < low or req1 > high:
        return None

    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, high - low + 1, axis_len)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(low, req1 - length + 1)
        start1 = min(req0, high - length + 1)
        for start in range(start0, start1 + 1):
            end = start + length - 1
            choices.append((length, start, end))
    if not choices:
        return None

    lengths = sorted({length for length, _, _ in choices})
    if random.random() < SPAN_COMPACT_PROB:
        cutoff = lengths[min(len(lengths) - 1, len(lengths) // 2)]
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _rect_mask(
    h: int, w: int, rect: tuple[int, int, int, int]
) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = rect
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    t_n = random.randint(*OBS_THICKNESS_RANGE)
    t_w = random.randint(*OBS_THICKNESS_RANGE)
    t_s = random.randint(*OBS_THICKNESS_RANGE)

    min_sr = SQUARE_ROW_MARGIN
    max_sr = h - sz - SQUARE_ROW_MARGIN
    min_sc = SQUARE_COL_MARGIN_LEFT
    max_sc = w - sz - SQUARE_COL_MARGIN_RIGHT
    if min_sr > max_sr or min_sc > max_sc:
        return None
    sr = random.randint(min_sr, max_sr)
    sc = random.randint(min_sc, max_sc)

    n0_lo, n0_hi = 2, sr - t_n - 3
    w0_lo, w0_hi = 2, sc - t_w - 3
    s0_lo, s0_hi = sr + sz + 2, h - t_s - 2
    if n0_lo > n0_hi or w0_lo > w0_hi or s0_lo > s0_hi:
        return None
    n0 = random.randint(n0_lo, n0_hi)
    w0 = random.randint(w0_lo, w0_hi)
    s0 = random.randint(s0_lo, s0_hi)

    north_exit_row = n0 - 1
    west_exit_col = w0 - 1
    min_len = max(5, SPAN_MIN_MULT * sz)
    max_len_h = min(SPAN_MAX_MULT * sz, h)
    max_len_w = min(SPAN_MAX_MULT * sz, w)

    west_span = _pick_cover_span(
        h,
        [(north_exit_row, north_exit_row + sz - 1), (sr, sr + sz - 1)],
        0,
        s0 - 2,
        min_len,
        max_len_h,
    )
    if west_span is None:
        return None

    west_right = w0 + t_w - 1
    north_span = _pick_cover_span(
        w,
        [(sc, sc + sz - 1)],
        west_right + 2,
        w - 2,
        min_len,
        max_len_w,
    )
    if north_span is None:
        return None

    south_span = _pick_cover_span(
        w,
        [(west_exit_col, west_exit_col + sz - 1), (sc, sc + sz - 1)],
        1,
        w - 2,
        min_len,
        max_len_w,
    )
    if south_span is None:
        return None

    rect_n = (n0, north_span[0], n0 + t_n - 1, north_span[1])
    rect_w = (west_span[0], w0, west_span[1], w0 + t_w - 1)
    rect_s = (s0, south_span[0], s0 + t_s - 1, south_span[1])
    rects = [rect_n, rect_w, rect_s]

    occupied = np.zeros((h, w), dtype=bool)
    for rect in rects:
        r0, c0, r1, c1 = rect
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    source_rect = (sr, sc, sr + sz - 1, sc + sz - 1)
    src_mask = _rect_mask(h, w, source_rect)
    if (occupied & src_mask).any():
        return None

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    obstacle_colors = random.sample(legend_colors, 3)
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    for rect, color in zip(rects, obstacle_colors):
        r0, c0, r1, c1 = rect
        base[r0 : r1 + 1, c0 : c1 + 1] = color

    vals, counts = np.unique(base, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    turns = {color: "L" for color in obstacle_colors}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        expected, stats = _simulate_combo(base, turns, bg)
        static = _simulate_static_colors(base, turns, bg)
    except AssertionError:
        return None

    _obstacle_ids, boxes, _colors = _find_obstacles(base, bg, teal=TEAL)
    north_id = next(
        (obs_id for obs_id, (r0, _c0, r1, _c1) in enumerate(boxes) if r1 < sr),
        None,
    )
    west_id = next(
        (obs_id for obs_id, (_r0, c0, _r1, c1) in enumerate(boxes) if c1 < sc),
        None,
    )
    south_id = next(
        (obs_id for obs_id, (r0, _c0, _r1, _c1) in enumerate(boxes) if r0 > sr + sz - 1),
        None,
    )
    if north_id is None or west_id is None or south_id is None:
        return None

    hit_ids = [[hit[0] for hit in info["hits"]] for info in stats]
    hit_colors = [[hit[1] for hit in info["hits"]] for info in stats]
    if hit_ids[0] != [north_id, west_id, south_id]:
        return None
    if not hit_ids[2] or hit_ids[2][0] != south_id or hit_colors[2][0] != TEAL:
        return None
    if not hit_ids[3] or hit_ids[3][0] != west_id or hit_colors[3][0] != TEAL:
        return None
    if hit_ids[1]:
        return None

    later_teal_hits = sum(
        1 for info in stats[1:] for hit in info["hits"] if hit[1] == TEAL
    )
    if later_teal_hits < 2:
        return None

    diff_vs_static = int(np.count_nonzero(expected != static))
    recolored_obstacle_cells = int(
        np.count_nonzero((expected == TEAL) & (base != bg) & (base != TEAL))
    )
    if diff_vs_static < 3 * sz:
        return None
    if recolored_obstacle_cells < sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rects):
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
