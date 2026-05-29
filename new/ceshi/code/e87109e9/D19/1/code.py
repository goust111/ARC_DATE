# Variant of e87109e9
# Dimension: D19 — 转弯后射线方向的确定 (path)
# Change: 转弯严格为90度——遇到障碍物后射线方向精确转动90度（顺时针或逆时针）... → 转弯角度由碰撞位置决定

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# ---------------------------------------------------------------------------
# Design analysis for the D19 variant
#
# spec_variant:
#   preserved_structure:
#     legend:
#       same as source task: 6x24 top band, bg=5, 4 cells, each cell contains one
#       colored vertical bar whose left/right position still encodes CCW/CW.
#     main_grid:
#       same overall canvas family: 19-24 rows x 24 cols, one solid teal square,
#       four solid obstacle rectangles, legend cropped away in output.
#   core_change:
#     original:
#       every obstacle hit used only the obstacle color -> legend says L or R ->
#       beam always rotates exactly 90 degrees.
#     variant:
#       the obstacle color is no longer sufficient. First detect where the next
#       beam footprint meets the obstacle. If the overlap sits in the obstacle's
#       upper half, use the legend and rotate 90 degrees. If the overlap sits in
#       the lower half, ignore the legend's angle and reverse 180 degrees.
#   concrete_pixel_realization:
#     - Obstacles are still solid rectangles.
#     - The solver inspects the actual overlap between the next sz x sz beam block
#       and the obstacle component.
#     - "Upper half / lower half" is measured by comparing the overlap rows to the
#       obstacle bbox rows. The generator keeps hits far from the midpoint so this
#       split is visually stable rather than tie-sensitive.
#   behavior_trace:
#     example:
#       - Suppose the source square emits a west-going beam.
#       - In the source task, the first obstacle color alone would force either
#         W -> N or W -> S.
#       - In this variant, if that same west-going beam strikes the lower half of
#         the left obstacle, it instead becomes W -> E (a U-turn), travels back
#         across the source, and can hit another obstacle.
#       - Therefore the changed stage is exactly the collision-resolution step:
#         the path before impact is unchanged, but the post-impact direction now
#         depends on both color and impact height.
#   complexity_calibration:
#     - Each generated grid guarantees at least one upper-half collision and one
#       lower-half collision, so both branches of the new rule appear repeatedly.
#     - One side obstacle is arranged for a lower hit and the opposite side for an
#       upper hit. This keeps the rule visible without introducing many unrelated
#       degrees of freedom.
#     - Top/bottom obstacles remain as same-style rectangles but are placed off all
#       active lanes so they do not clutter the causal pattern.
#   implicit_assumption_review_for_main:
#     - Source code originally used obstacle_color_at_next(...): that assumed color
#       alone determines the turn. This is no longer valid, so the new solver must
#       recover the obstacle component and its bbox before choosing the new heading.
#     - The original code could ignore geometry after finding a color. In this
#       variant, doing so would silently collapse every lower-half hit back to the
#       old 90-degree behavior.
#     - The new solver assumes that a candidate next footprint overlaps at most one
#       obstacle component. The generator guarantees non-overlapping solid bars with
#       clear gaps, so this assumption is safe.
#     - The new solver also assumes the hit is not balanced exactly on the obstacle
#       midpoint. The generator enforces extra row margin so every meaningful hit is
#       clearly upper or clearly lower.
#   placement_legality:
#     - Side obstacles touch only left/right borders and are vertically positioned
#       around the source so the horizontal beams hit them at planned heights.
#     - Top/bottom obstacles touch only top/bottom borders and are forbidden from
#       covering the source column lane or the turned beam lane.
#     - Obstacles never overlap each other or the teal square, and the chosen
#       left/right pairing prevents 180-degree cycles: the lower-hit side sends the
#       beam across the board, and the opposite upper-hit side ejects it vertically.
#   observability:
#     - Human solvers can see that two side bars of the same visual type are hit at
#       different vertical positions relative to their own heights, and the output
#       path reacts differently.
#     - The distinguishing signal is geometric and visible in the pixels; it does
#       not rely on hidden ordering, color ids, or any non-visual metadata.
# ---------------------------------------------------------------------------


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
SQUARE_EDGE_GAP = 6

SIDE_THICKNESS_RANGE = (2, 4)
SIDE_HEIGHT_RANGE = (6, 12)
SIDE_CLEAR_MARGIN = 5

TOPBOT_THICKNESS_RANGE = (2, 4)
TOPBOT_WIDTH_RANGE = (6, 12)

MAX_RETRIES = 400


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


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
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


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot_by_legend(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _obstacle_components(base: np.ndarray, bg: int) -> list[tuple[int, int, int, int, int]]:
    mask = (base != bg) & (base != TEAL)
    obstacles: list[tuple[int, int, int, int, int]] = []
    for cells in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        block = base[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(block)))
        colors.discard(bg)
        colors.discard(TEAL)
        assert len(colors) == 1, "each obstacle component must be monochrome"
        color = int(next(iter(colors)))
        assert np.all(block == color), "each obstacle must be a solid rectangle"
        obstacles.append((int(r0), int(c0), int(r1), int(c1), color))
    return obstacles


def _hit_at_next(
    obstacles: list[tuple[int, int, int, int, int]],
    nr: int,
    nc: int,
    sz: int,
) -> tuple[int, bool] | None:
    hits: list[tuple[int, int, int, int, int]] = []
    rr1 = nr + sz - 1
    cc1 = nc + sz - 1
    for obs in obstacles:
        r0, c0, r1, c1, _color = obs
        if r1 < nr or r0 > rr1 or c1 < nc or c0 > cc1:
            continue
        hits.append(obs)
    if not hits:
        return None
    assert len(hits) == 1, "beam footprint must not hit multiple obstacles at once"
    r0, _c0, r1, _c1, color = hits[0]
    ov_r0 = max(nr, r0)
    ov_r1 = min(rr1, r1)
    is_upper = (ov_r0 + ov_r1) <= (r0 + r1)
    return color, is_upper


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four teal beams from a square, but let impact height decide whether a turn stays quarter-turn or becomes a reversal.

    Input structure:
    - The top band is still a 6x24 grey legend with four colored vertical bars; the bar's left/right placement still encodes counterclockwise vs clockwise.
    - Below it is the main grid: one solid teal square on a plain background plus several solid colored obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - Launch one sz x sz teal beam footprint in each cardinal direction from the source square, painting only background cells as it slides.
    - When the next footprint would hit an obstacle, first judge whether the overlap lands in that obstacle's upper half or lower half.
    - Upper-half hit: use the legend color and rotate 90 degrees left or right.
    - Lower-half hit: ignore the 90-degree choice and reverse 180 degrees.
    - Beams stop when the next move would leave the cropped main grid.

    Key insight: obstacle color still matters, but only after locating the collision on the obstacle itself; geometry of the hit is now part of the decoding rule.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = LEGEND_BG
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_turn = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    obstacles = _obstacle_components(base, bg)

    out = base.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            hit = _hit_at_next(obstacles, nr, nc, sz)
            if hit is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            color, is_upper = hit
            assert color in legend_turn, "obstacle color must be encoded by legend"
            if is_upper:
                d = _rot_by_legend(d, legend_turn[color])
            else:
                d = (d + 2) % 4

    return out


def _simulate_paths(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, bool]]]]:
    out = base.copy()
    obstacles = _obstacle_components(base, bg)

    teal_locs = np.argwhere(base == TEAL)
    assert teal_locs.size
    sr, sc = teal_locs.min(axis=0)
    r1, c1 = teal_locs.max(axis=0)
    sz = int(r1 - sr + 1)
    assert sz == int(c1 - sc + 1)

    h, w = out.shape
    hits: list[list[tuple[int, int, int, int, bool]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        beam_hits: list[tuple[int, int, int, int, bool]] = []
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen
            seen.add(state)
            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break
            hit = _hit_at_next(obstacles, nr, nc, sz)
            if hit is None:
                r, c = nr, nc
                paint(r, c)
                continue
            color, is_upper = hit
            beam_hits.append((d, color, nr, nc, is_upper))
            d = _rot_by_legend(d, turns[color]) if is_upper else (d + 2) % 4
        hits.append(beam_hits)

    return out, hits


def _simulate_old_rule(base: np.ndarray, turns: dict[int, str], bg: int) -> np.ndarray:
    out = base.copy()
    obstacles = _obstacle_components(base, bg)

    teal_locs = np.argwhere(base == TEAL)
    sr, sc = teal_locs.min(axis=0)
    r1, c1 = teal_locs.max(axis=0)
    sz = int(r1 - sr + 1)
    h, w = out.shape

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen
            seen.add(state)
            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break
            hit = _hit_at_next(obstacles, nr, nc, sz)
            if hit is None:
                r, c = nr, nc
                paint(r, c)
                continue
            color, _is_upper = hit
            d = _rot_by_legend(d, turns[color])
    return out


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _choose_side_rows(h: int, sr: int, sz: int, kind: str) -> tuple[int, int] | None:
    options: list[tuple[int, int]] = []
    for above in range(1, 6):
        for below in range(1, 6):
            height = above + sz + below
            if not (SIDE_HEIGHT_RANGE[0] <= height <= SIDE_HEIGHT_RANGE[1]):
                continue
            if kind == "upper" and below < above + 2:
                continue
            if kind == "lower" and above < below + 2:
                continue
            r0 = sr - above
            r1 = sr + sz - 1 + below
            if r0 < SIDE_CLEAR_MARGIN or r1 >= h - SIDE_CLEAR_MARGIN:
                continue
            options.append((r0, r1))
    return random.choice(options) if options else None


def _choose_clear_span(
    lo: int,
    hi: int,
    width_range: tuple[int, int],
    forbidden: list[tuple[int, int]],
) -> tuple[int, int] | None:
    options: list[tuple[int, int]] = []
    axis_len = hi - lo + 1
    for width in range(width_range[0], min(width_range[1], axis_len) + 1):
        for start in range(lo, hi - width + 2):
            end = start + width - 1
            if all(end < a or start > b for a, b in forbidden):
                options.append((start, end))
    return random.choice(options) if options else None


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3

    t_left = random.randint(*SIDE_THICKNESS_RANGE)
    t_right = random.randint(*SIDE_THICKNESS_RANGE)
    t_top = random.randint(*TOPBOT_THICKNESS_RANGE)
    t_bottom = random.randint(*TOPBOT_THICKNESS_RANGE)

    min_r = max(SQUARE_EDGE_GAP, SIDE_CLEAR_MARGIN + 3)
    max_r = min(h - sz - SQUARE_EDGE_GAP, h - SIDE_CLEAR_MARGIN - sz - 3)
    min_c = max(SQUARE_EDGE_GAP, t_left + 4)
    max_c = min(w - sz - SQUARE_EDGE_GAP, w - t_right - sz - 4)
    if min_r > max_r or min_c > max_c:
        return None
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    lower_side = random.choice(["left", "right"])
    left_kind = "lower" if lower_side == "left" else "upper"
    right_kind = "upper" if lower_side == "left" else "lower"

    left_rows = _choose_side_rows(h, sr, sz, left_kind)
    right_rows = _choose_side_rows(h, sr, sz, right_kind)
    if left_rows is None or right_rows is None:
        return None
    lr0, lr1 = left_rows
    rr0, rr1 = right_rows

    left_lane = t_left
    right_lane = w - t_right - sz
    active_vertical_lanes = [left_lane] if left_kind == "upper" else [right_lane]

    forbidden_cols = [(sc, sc + sz - 1)]
    forbidden_cols.extend((lane, lane + sz - 1) for lane in active_vertical_lanes)
    interior_lo = t_left + 1
    interior_hi = w - t_right - 2
    if interior_lo > interior_hi:
        return None
    top_span = _choose_clear_span(interior_lo, interior_hi, TOPBOT_WIDTH_RANGE, forbidden_cols)
    bottom_span = _choose_clear_span(interior_lo, interior_hi, TOPBOT_WIDTH_RANGE, forbidden_cols)
    if top_span is None or bottom_span is None:
        return None

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    side_names = ["left", "right", "top", "bottom"]
    side_color = dict(zip(side_names, legend_colors))
    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    bg = random.choice(bg_choices)

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    base[lr0 : lr1 + 1, 0:t_left] = side_color["left"]
    base[rr0 : rr1 + 1, w - t_right : w] = side_color["right"]
    ts0, ts1 = top_span
    bs0, bs1 = bottom_span
    base[0:t_top, ts0 : ts1 + 1] = side_color["top"]
    base[h - t_bottom : h, bs0 : bs1 + 1] = side_color["bottom"]

    vals, counts = np.unique(base, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    turns = {color: random.choice(["L", "R"]) for color in legend_colors}
    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])

    try:
        expected, hits = _simulate_paths(base, turns, bg)
    except AssertionError:
        return None

    hit_branches = [is_upper for beam in hits for *_rest, is_upper in beam]
    if not hit_branches:
        return None
    if not any(hit_branches):
        return None
    if not any(not x for x in hit_branches):
        return None

    total_hits = sum(len(beam) for beam in hits)
    if total_hits < 3:
        return None

    try:
        old_out = _simulate_old_rule(base, turns, bg)
    except AssertionError:
        return None
    if np.array_equal(old_out, expected):
        return None

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
    raise RuntimeError("failed to generate e87109e9 D19 variant input")
