# Combo variant of e87109e9 (index 3)
# combo_spec: D4/0+D14/1
# Name: 链式射线颜色渐变

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - The input keeps the original two-zone layout: a 6x24 legend on top and
#      one main grid below. The legend still maps obstacle color -> turn
#      direction by placing a vertical bar on the left or right of each cell.
#    - The main grid now contains two teal objects: one solid sz x sz source
#      square and one detached teal starter strip of length sz. The strip sits
#      exactly one background cell away from one side of the square and marks
#      the only launch direction.
#    - Example: if the strip is above the square, the ray starts north. It
#      paints teal until the next sz x sz footprint would hit a red obstacle.
#      The red legend entry says left or right, so the ray rotates in place.
#      The next free step is painted red. If that red segment later hits a blue
#      obstacle, the ray turns again and the following segment becomes blue.
#      The output is therefore one continuous polyline with color inheritance:
#      teal -> red -> blue -> ...
#    - Relative to the base task, two things changed together: the source no
#      longer launches four independent rays, and the trail color is no longer
#      fixed. The ray carries state across turns, so the solver must track both
#      direction and current paint color.
# b) Complexity calibration:
#    - To keep the variant legible from 2-4 examples, the generator always
#      produces a chain with 3 or 4 obstacle hits. That makes the "single ray"
#      behavior and the "segment recolors after each hit" behavior visible in
#      every sample.
#    - Obstacle geometry, legend size, main-grid size, obstacle thickness, and
#      edge-touching placement all stay close to the original task so the new
#      difficulty comes from rule fusion rather than extra clutter.
#    - The turn schedule is globally clockwise or counterclockwise per sample.
#      This keeps path construction stable while the color sequence still varies
#      because obstacle colors are shuffled independently of side order.
# c) Hidden-assumption audit:
#    - Base solver assumption broken by D14: "all teal cells are one square".
#      The combo solver must distinguish the square from the starter strip.
#    - Base solver assumption broken by D4: "checking the partially painted
#      output is safe for obstacle detection". That fails once trail cells can be
#      non-teal. The combo solver must detect obstacles from the immutable cropped
#      input grid (`base`), not from the evolving output.
#    - Base solver assumption broken by both together: "state is only (r,c,d)".
#      Once the same geometric state could carry a different inherited color, the
#      cycle guard should include paint color when color inheritance is active.
#    - Painting must only overwrite background cells from the input geometry.
#      This preserves obstacles, source square, and starter strip while still
#      allowing the new segment color to overwrite local elbow overlap cells.
# d) Placement legality:
#    - The source square stays well inside the main grid and leaves space for the
#      starter strip on the launch side.
#    - Obstacles remain solid monochrome rectangles touching one boundary, never
#      overlap the square or starter strip, and are sized so the intended lane
#      hits them with an sz-wide footprint.
#    - For four-hit chains, the first obstacle explicitly excludes the final
#      wraparound lane so the path exits instead of looping back to hit again.
# e) Visual observability:
#    - A human can see the unique launch direction from the detached starter
#      strip, then confirm in the output that only one polyline is drawn.
#    - A human can also see that segment colors are not arbitrary: each new
#      segment matches the obstacle color at the previous bounce.
# f) Cleanliness:
#    - Only one ray is drawn, so there is no beam-beam interference. The only
#      overlap is the local sz-wide elbow when a segment turns, which remains
#      visually readable because it is a single continuous path.
#    - Obstacles stay on the perimeter, keeping the interior mostly open.
# g) Color-change inferability:
#    - Obstacle colors come from four legend colors, but only 3-4 are used on
#      the path. Every used obstacle color must also appear on some trail cells;
#      this prevents degenerate samples where a bounce happens but its inherited
#      color is never visible.
# 9) New solver assumptions guaranteed by the generator:
#    - Exactly two teal connected components exist: one solid square and one
#      aligned starter strip of matching length.
#    - Every obstacle footprint encountered by the ray is monochrome.
#    - The legend is the unique top band using color 5.
#    - The generated chain is non-branching and does not rely on ambiguous
#      self-intersections beyond the intended elbow overwrite.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

# Color palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

# Canvas dimensions
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

# Teal source
TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5
STARTER_GAP = 1

# Obstacles
OBS_COUNT_4_PROB = 0.6
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Retry budget
MAX_RETRIES = 600


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Read the top legend and map obstacle color -> turn direction."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

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
        col = int(next(iter(colors)))
        xs = np.where(interior == col)[1]
        x = int(xs[0])
        turn[col] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_source(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int, int]:
    """
    Return (r0, c0, size, start_dir) for the solid teal square plus starter strip.

    The strip is detached by one background cell and aligned to exactly one side
    of the square, so it unambiguously selects the only launch direction.
    """
    comps = list(components_mask(g == teal, connectivity=4))
    assert len(comps) == 2, "main grid must contain one square and one starter strip"

    square_candidates: list[tuple[int, int, int]] = []
    for cells in comps:
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        if hh != ww or hh < 2:
            continue
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        square_candidates.append((int(r0), int(c0), hh))
    assert len(square_candidates) == 1, "there must be exactly one solid teal source square"

    sr, sc, sz = square_candidates[0]
    for cells in comps:
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = int(r1 - r0 + 1)
        ww = int(c1 - c0 + 1)
        area = int(cells.shape[0])
        if (int(r0), int(c0), hh) == square_candidates[0]:
            continue
        assert area == sz, "starter strip must match the square side length"
        if hh == 1 and ww == sz and int(r0) == sr - 2 and int(c0) == sc:
            return sr, sc, sz, _N
        if hh == 1 and ww == sz and int(r0) == sr + sz + 1 and int(c0) == sc:
            return sr, sc, sz, _S
        if hh == sz and ww == 1 and int(c0) == sc - 2 and int(r0) == sr:
            return sr, sc, sz, _W
        if hh == sz and ww == 1 and int(c0) == sc + sz + 1 and int(r0) == sr:
            return sr, sc, sz, _E
    raise AssertionError("starter strip must align with exactly one side of the source square")


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _starter_cells(sr: int, sc: int, sz: int, start_dir: int) -> list[tuple[int, int]]:
    if start_dir == _N:
        return [(sr - 2, c) for c in range(sc, sc + sz)]
    if start_dir == _S:
        return [(sr + sz + 1, c) for c in range(sc, sc + sz)]
    if start_dir == _W:
        return [(r, sc - 2) for r in range(sr, sr + sz)]
    return [(r, sc + sz + 1) for r in range(sr, sr + sz)]


def _sequence(start_side: int, turn: str, count: int) -> list[int]:
    delta = -1 if turn == "L" else 1
    return [int((start_side + delta * i) % 4) for i in range(count)]


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(
    axis: int, sz: int, a_present: bool, b_present: bool
) -> tuple[int | None, int | None] | None:
    lo, hi_cap = OBS_THICKNESS_RANGE
    if a_present and b_present:
        room = axis - sz - OBS_BOTH_GAP
        if room < 2 * lo:
            return None
        a = random.randint(lo, min(hi_cap, room - lo))
        b = _rand_thickness(lo, min(hi_cap, room - a))
        return (a, b) if b is not None else None
    if a_present:
        t = _rand_thickness(lo, min(hi_cap, axis - sz - OBS_SINGLE_MARGIN))
        return (t, None) if t is not None else None
    if b_present:
        t = _rand_thickness(lo, min(hi_cap, axis - sz - OBS_SINGLE_MARGIN))
        return (None, t) if t is not None else None
    return None, None


def _obstacle_rect(
    side: int, thickness: int, span: tuple[int, int], h: int, w: int
) -> tuple[int, int, int, int]:
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _lane_after_hit(side: int, thickness: dict[int, int], h: int, w: int, sz: int) -> int:
    if side == _N:
        return thickness[_N]
    if side == _S:
        return h - thickness[_S] - sz
    if side == _W:
        return thickness[_W]
    return w - thickness[_E] - sz


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals and none of the excluded ones."""
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, axis_len)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(0, req1 - length + 1)
        start1 = min(req0, axis_len - length)
        for start in range(start0, start1 + 1):
            end = start + length - 1
            if all(end < a or start > b for a, b in excluded):
                choices.append((length, start, end))
    if not choices:
        return None

    lengths = sorted({length for length, _, _ in choices})
    if random.random() < SPAN_COMPACT_PROB:
        cutoff = lengths[min(len(lengths) - 1, max(0, len(lengths) // 2))]
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _simulate_chain(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    inherit_color: bool,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], dict[tuple[int, int], int], int]:
    """
    Simulate the chained ray on the cropped main grid.

    Returns:
      - painted output
      - hit list as (incoming_dir, obstacle_color, next_r, next_c)
      - final trail colors for all background cells painted by the ray
      - parsed start direction
    """
    sr, sc, sz, start_dir = _find_source(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[tuple[int, int, int, int]] = []
    trail_cells: dict[tuple[int, int], int] = {}

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def obstacle_at(r: int, c: int) -> int | None:
        blk = base[r : r + sz, c : c + sz]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1, "ray footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    def paint_block(r: int, c: int, color: int) -> None:
        blk_base = base[r : r + sz, c : c + sz]
        bg_rows, bg_cols = np.where(blk_base == bg)
        for roff, coff in zip(bg_rows, bg_cols):
            cell = (int(r + roff), int(c + coff))
            trail_cells[cell] = int(color)
            out[cell] = int(color)

    r, c, d = int(sr), int(sc), int(start_dir)
    paint_color = TEAL
    seen: set[tuple[int, int, int, int]] = set()
    while True:
        state = (r, c, d, paint_color if inherit_color else TEAL)
        assert state not in seen, "ray path must not cycle"
        seen.add(state)

        dr, dc = _DIRS[d]
        nr, nc = r + dr, c + dc
        if not in_bounds(nr, nc):
            break

        obs = obstacle_at(nr, nc)
        if obs is None:
            r, c = nr, nc
            paint_block(r, c, paint_color if inherit_color else TEAL)
            continue

        assert obs in turns, "obstacle color must be encoded by the legend"
        hits.append((d, obs, nr, nc))
        d = _rot(d, turns[obs])
        if inherit_color:
            paint_color = int(obs)

    return out, hits, trail_cells, start_dir


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace one chained square ray whose segment color changes after each bounce.

    Input structure:
    - Top band: a 6-row legend on color-5 background with four colored vertical bars; each bar's left/right placement encodes CCW/CW turning for that color.
    - Below the legend: one main grid containing a solid teal source square, one detached teal starter strip that marks the only launch side, background cells, and 3-4 colored obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - Launch exactly one sz-wide ray from the square in the direction indicated by the starter strip.
    - The first segment paints teal on background cells only.
    - Whenever the next sz-wide step would hit obstacle color C, turn according to C's legend entry and continue following the same ray.
    - After each such turn, all later trail cells use color C until another obstacle is hit.
    - Obstacles, the source square, and the starter strip are not erased; the ray stops when its next step would leave the grid.

    Key insight: The starter strip chooses the single initial direction, and the obstacle that caused the latest turn also supplies the color of the next segment, so the output is one continuous multi-color chain rather than four independent teal beams.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must stay in the top band"

    legend_turn = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    out, _hits, _trail, _start_dir = _simulate_chain(base, legend_turn, bg, inherit_color=True)
    return out


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3
    turn_dir = random.choice(["L", "R"])
    start_side = random.choice([_N, _E, _S, _W])
    sequence = _sequence(start_side, turn_dir, obstacle_count)
    present = set(sequence)

    ns = _thickness_pair(h, sz, _N in present, _S in present)
    ew = _thickness_pair(w, sz, _W in present, _E in present)
    if ns is None or ew is None:
        return None
    thickness = {s: t for s, t in zip((_N, _S, _W, _E), (*ns, *ew)) if t is not None}

    min_r, max_r = SQUARE_EDGE_GAP, h - sz - SQUARE_EDGE_GAP
    min_c, max_c = SQUARE_EDGE_GAP, w - sz - SQUARE_EDGE_GAP
    if _N in present:
        min_r = max(min_r, thickness[_N] + 1)
    if _S in present:
        max_r = min(max_r, h - thickness[_S] - sz - 1)
    if _W in present:
        min_c = max(min_c, thickness[_W] + 1)
    if _E in present:
        max_c = min(max_c, w - thickness[_E] - sz - 1)
    if start_side == _N:
        min_r = max(min_r, thickness.get(_N, 0) + 2 + STARTER_GAP)
    if start_side == _S:
        max_r = min(max_r, h - thickness.get(_S, 0) - sz - 2 - STARTER_GAP)
    if start_side == _W:
        min_c = max(min_c, thickness.get(_W, 0) + 2 + STARTER_GAP)
    if start_side == _E:
        max_c = min(max_c, w - thickness.get(_E, 0) - sz - 2 - STARTER_GAP)
    if min_r > max_r or min_c > max_c:
        return None

    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    lane = {side: _lane_after_hit(side, thickness, h, w, sz) for side in present}
    required: dict[int, list[tuple[int, int]]] = {side: [] for side in present}
    excluded: dict[int, list[tuple[int, int]]] = {side: [] for side in present}

    for i, side in enumerate(sequence):
        if i == 0:
            base_coord = sc if side in (_N, _S) else sr
            required[side].append((base_coord, base_coord + sz - 1))
        else:
            prev = sequence[i - 1]
            required[side].append((lane[prev], lane[prev] + sz - 1))

    if obstacle_count == 4:
        first = sequence[0]
        last = sequence[-1]
        excluded[first].append((lane[last], lane[last] + sz - 1))

    spans: dict[int, tuple[int, int]] = {}
    for side in present:
        axis_len = w if side in (_N, _S) else h
        span = _choose_span(
            axis_len,
            required[side],
            excluded[side],
            SPAN_MIN_MULT * sz,
            min(SPAN_MAX_MULT * sz, axis_len),
        )
        if span is None:
            return None
        spans[side] = span

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    obstacle_colors = random.sample(legend_colors, obstacle_count)
    side_color = dict(zip(sequence, obstacle_colors))
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    for r, c in _starter_cells(sr, sc, sz, start_side):
        base[r, c] = TEAL

    for side in present:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        if np.any(base[r0 : r1 + 1, c0 : c1 + 1] != bg):
            return None
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    turns = {side_color[side]: turn_dir for side in present}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        combo_out, hits, trail_cells, parsed_start = _simulate_chain(
            base, turns, bg, inherit_color=True
        )
        mono_out, mono_hits, mono_trail, mono_start = _simulate_chain(
            base, turns, bg, inherit_color=False
        )
    except AssertionError:
        return None

    if parsed_start != start_side or mono_start != start_side:
        return None
    if len(hits) != obstacle_count or len(mono_hits) != obstacle_count:
        return None
    if [d for d, _, _, _ in hits] != sequence:
        return None
    if [d for d, _, _, _ in mono_hits] != sequence:
        return None

    hit_colors = {color for _, color, _, _ in hits}
    used_colors = set(side_color.values())
    visible_non_teal = {color for color in trail_cells.values() if color != TEAL}
    diff_cells = int(np.sum(combo_out != mono_out))
    non_teal_trail_cells = sum(1 for color in trail_cells.values() if color != TEAL)

    if used_colors != hit_colors:
        return None
    if visible_non_teal != used_colors:
        return None
    if non_teal_trail_cells < obstacle_count * sz:
        return None
    if diff_cells < 3 * sz:
        return None
    if not mono_trail:
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
    raise RuntimeError("failed to generate combo input for e87109e9")
