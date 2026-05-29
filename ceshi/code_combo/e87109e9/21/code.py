# Combo variant of e87109e9 (index 21)
# combo_spec: D1/3+D3/0
# Name: 图例顺序编码方向加箭头图例

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for combo D1/3 + D3/0
# a) Concrete behavior and trace
#    The source task has a 6x24 legend band on top and a main board below. A solid
#    teal square launches four sz-wide rays; colored edge obstacles trigger left or
#    right turns by consulting the legend.
#    This combo changes two specific stages together:
#    1. D3/0 replaces "bar on left/right" with a micro-symbol legend. Each of the
#       four legend entries is now a 4x4 black cell inside the grey band. Its
#       top-left 2x2 patch is a color sample identifying one obstacle color, and
#       its bottom-right 2x2 patch contains a 3-pixel mirrored L glyph. The two
#       glyph mirrors encode the two turn options.
#    2. D1/3 makes legend entry order operational. The leftmost entry launches the
#       up-going ray, the second launches right, the third launches down, and the
#       fourth launches left.
#    Small trace:
#    - Original: all four rays start at once, so legend order is irrelevant; only
#      color -> left/right matters.
#    - Naive fusion: if we merely "launch them sequentially" but still let later
#      rays walk through earlier teal paint, the output is identical to the source
#      task because the same four directions are still all used once.
#    - Implemented fusion: rays do launch sequentially by legend order, and the
#      already-painted teal frontier becomes occupied for later rays. A later ray
#      may therefore stop when its next incoming strip would enter an earlier teal
#      trail. This is the minimal way to make "order" visible on the grid while
#      keeping the original geometry and turn dynamics.
# b) Complexity calibration
#    The obstacle family stays narrow and readable: 3-4 solid edge bars, 2-4 cell
#    thickness, 4 legend entries, one teal source square of size 2 or 3. The new
#    symbolic freedom is only binary per legend entry (two mirrored elbows), and
#    the launch-order freedom is fixed to the human-visible left-to-right order of
#    those entries. The generator also requires both turn glyphs to appear in a
#    sample and requires at least one ray to be blocked by an earlier trail, so the
#    fusion is visible rather than collapsing into the base task.
# c) Hidden-assumption audit against the source solver
#    - `_parse_legend` in the source code assumed each cell had one vertical bar and
#      inferred turn from horizontal placement. That is now false; parsing must look
#      for a fixed color swatch plus one of two 3-pixel elbow glyphs.
#    - The source beam loop assumed starting direction was simply `for d0 in range(4)`.
#      That loses the D1/3 signal. The new loop must first read legend entries in
#      order and then launch directions in that same order.
#    - The source obstacle test ignored teal entirely, which made sequential launch
#      degenerate. The new step logic must distinguish immutable obstacle colors
#      (read from the clean base grid) from already-painted teal frontier cells
#      (read from the evolving output grid).
#    - Cycle detection in the source used `(r,c,d)`. With trail blocking, cycles are
#      already structurally discouraged, but the solver still keeps this guard.
# d) Placement legality
#    The generator keeps the source task's path-first obstacle planning so obstacle
#    rectangles remain disjoint, edge-touching, and separated from the source
#    square. The legend uses fixed cell slots and fixed glyph footprints, so legend
#    symbols never overlap. After obstacles are placed, oracle simulation rejects
#    any case where a beam footprint would see mixed obstacle colors or where the
#    new trail-blocking rule creates invalid ambiguity.
# e) Visual observability
#    Humans can see two independent clues in a few examples:
#    - every legend entry has the same two-part structure: a color swatch and a tiny
#      mirrored elbow symbol;
#    - output trails differ from the source task because earlier rays can choke off
#      later ones, so the left-to-right entry order actually matters.
#    This does not rely on numeric color ids or hidden sorting; only visible slot
#    order and visible glyph orientation matter.
# f) Visual cleanliness
#    Legend symbols stay inside the cropped top strip and use only one obstacle
#    color per cell against black on grey, so they are easy to segment. The main
#    grid remains one teal square plus large obstacle bars plus teal trails. There
#    is no extra decoration inside the playfield.
# g) Color-rule audit
#    Colors do not rank or sort anything. Legend slot position, not color value,
#    determines launch order. Color is used only to associate an obstacle with its
#    turn glyph.
# h) New solver assumptions introduced here and guaranteed by the generator
#    - The only grey (5) cells are in the top legend band, so legend cropping by
#      contiguous top rows is safe.
#    - Each legend cell contains exactly one non-background, non-black color and a
#      fixed 2x2 swatch in its top-left corner.
#    - The bottom-right 2x2 patch of each legend cell matches exactly one of the two
#      mirrored 3-pixel elbow templates.
#    - Obstacles are solid rectangles hit via a monochrome frontier strip.
#    - At least one later ray is blocked by an earlier teal trail, so the combo does
#      not silently degenerate to the base task.

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
SQUARE_EDGE_GAP = 5

OBS_COUNT_4_PROB = 0.65
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

MAX_RETRIES = 500

_SWATCH_ROWS = slice(0, 2)
_SWATCH_COLS = slice(0, 2)
_ARROW_ROWS = slice(2, 4)
_ARROW_COLS = slice(2, 4)
_ARROW_MASKS = {
    "L": np.array([[1, 0], [1, 1]], dtype=bool),
    "R": np.array([[0, 1], [1, 1]], dtype=bool),
}


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


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


def _legend_cell_turn(cell: np.ndarray, color: int) -> str:
    """Decode the mirrored 3-pixel elbow glyph from one 4x4 legend cell."""
    mask = cell == color
    assert mask.shape == (4, 4), "legend cell must be 4x4"
    assert np.all(mask[_SWATCH_ROWS, _SWATCH_COLS]), "legend color swatch missing"
    arrow_patch = mask[_ARROW_ROWS, _ARROW_COLS]
    for turn, template in _ARROW_MASKS.items():
        if np.array_equal(arrow_patch, template):
            return turn
    raise AssertionError(f"unrecognized legend arrow patch:\n{arrow_patch.astype(int)}")


def _parse_legend(grid: np.ndarray, legend_h: int) -> list[tuple[int, str]]:
    """Return legend entries in visible left-to-right order as (color, turn)."""
    legend = grid[:legend_h, :]
    assert legend.shape == (LEGEND_H, GRID_W), "legend must keep the fixed 6x24 layout"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])
    assert bg == LEGEND_BG, "legend background must be grey"

    entries: list[tuple[int, str]] = []
    for start in LEGEND_CELL_STARTS:
        cell = legend[1:5, start : start + 4]
        colors = set(map(int, np.unique(cell)))
        colors.discard(0)
        colors.discard(bg)
        assert len(colors) == 1, "each legend cell must encode exactly one obstacle color"
        color = int(next(iter(colors)))
        turn = _legend_cell_turn(cell, color)
        entries.append((color, turn))
    assert len(entries) == 4
    return entries


def _frontier_coords(nr: int, nc: int, d: int, sz: int) -> list[tuple[int, int]]:
    """Cells newly entered when a sz x sz square advances one step in direction d."""
    if d == _N:
        return [(nr, c) for c in range(nc, nc + sz)]
    if d == _S:
        return [(nr + sz - 1, c) for c in range(nc, nc + sz)]
    if d == _W:
        return [(r, nc) for r in range(nr, nr + sz)]
    return [(r, nc + sz - 1) for r in range(nr, nr + sz)]


def _simulate_paths(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    trail_blocks: bool,
    launch_dirs: tuple[int, ...] = (_N, _E, _S, _W),
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]], list[str]]:
    """
    Simulate the four legend-ordered square beams on the main grid.

    If `trail_blocks` is True, an already-painted teal frontier strip blocks later
    movement. This is the operational part of the D1/3 fusion.
    """
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []
    stops: list[str] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint_block(r: int, c: int) -> None:
        base_blk = base[r : r + sz, c : c + sz]
        out_blk = out[r : r + sz, c : c + sz]
        out_blk[(base_blk == bg) & (out_blk == bg)] = TEAL
        out[r : r + sz, c : c + sz] = out_blk

    for d0 in launch_dirs:
        r, c, d = sr, sc, d0
        beam_hits: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        stop_reason = "edge"
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                stop_reason = "edge"
                break

            frontier = _frontier_coords(nr, nc, d, sz)
            frontier_base = {int(base[rr, cc]) for rr, cc in frontier}
            obs_colors = {v for v in frontier_base if v not in (bg, TEAL)}
            assert len(obs_colors) <= 1, "frontier strip must not meet mixed obstacle colors"

            if obs_colors:
                obs = int(next(iter(obs_colors)))
                assert obs in turns, "obstacle color must be encoded by the legend"
                beam_hits.append((d, obs, nr, nc))
                d = _rot(d, turns[obs])
                continue

            if trail_blocks and any(int(out[rr, cc]) == TEAL and int(base[rr, cc]) == bg for rr, cc in frontier):
                stop_reason = "trail"
                break

            r, c = nr, nc
            paint_block(r, c)

        hits.append(beam_hits)
        stops.append(stop_reason)

    return out, hits, stops


def _fusion_metrics(base: np.ndarray, turns: dict[int, str], bg: int) -> dict[str, int]:
    """Quantify whether the order-sensitive trail blocking actually matters."""
    launch_dirs = (_N, _E, _S, _W)
    fused_out, fused_hits, fused_stops = _simulate_paths(
        base, turns, bg, trail_blocks=True, launch_dirs=launch_dirs
    )
    relaxed_out, relaxed_hits, _ = _simulate_paths(
        base, turns, bg, trail_blocks=False, launch_dirs=launch_dirs
    )
    changed = int(np.sum(fused_out != relaxed_out))
    blocked = sum(reason == "trail" for reason in fused_stops)
    hit_delta = sum(len(a) - len(b) for a, b in zip(relaxed_hits, fused_hits))
    return {
        "changed_cells": changed,
        "blocked_beams": int(blocked),
        "suppressed_hits": int(hit_delta),
    }


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    """Build the fixed 6x24 legend band using swatches plus mirrored elbow arrows."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        cell = legend[1:5, start : start + 4]
        cell[:, :] = 0
        cell[_SWATCH_ROWS, _SWATCH_COLS] = int(color)
        cell[_ARROW_ROWS, _ARROW_COLS] = _ARROW_MASKS[turns[color]].astype(int) * int(color)
    return legend


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals while missing all excluded intervals."""
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


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(
    axis: int, sz: int, a_present: bool, b_present: bool
) -> tuple[int | None, int | None] | None:
    """Compute obstacle thicknesses for the two ends of an axis (N/S or W/E)."""
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
    """Return (r0, r1, c0, c1) for an edge-touching obstacle rectangle."""
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _select_chains(present: set[int], second_hits: int) -> dict[int, int] | None:
    """Pick non-overlapping (source -> adjacent target) pairs for 2-hit beams."""
    candidates = [
        (side, nxt)
        for side in present
        for nxt in ((side - 1) % 4, (side + 1) % 4)
        if nxt in present
    ]
    random.shuffle(candidates)
    selected: list[tuple[int, int]] = []
    used: set[int] = set()
    for side, nxt in candidates:
        if len(selected) == second_hits:
            break
        if side in used or nxt in used:
            continue
        selected.append((side, nxt))
        used.update((side, nxt))
    return dict(selected) if len(selected) == second_hits else None


def _assign_turns(present: set[int], chains: dict[int, int]) -> dict[int, str]:
    """Assign L/R turn direction for each obstacle side."""
    side_turn: dict[int, str] = {}
    for side in present:
        if side in chains:
            nxt = chains[side]
            side_turn[side] = "L" if nxt == (side - 1) % 4 else "R"
        else:
            outward = [t for t in ("L", "R") if _rot(side, t) not in present]
            side_turn[side] = random.choice(outward if outward else ["L", "R"])
    return side_turn


def _compute_span_constraints(
    present: set[int],
    side_turn: dict[int, str],
    chains: dict[int, int],
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Build required/excluded intervals for each obstacle span."""
    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}

    for side in present:
        base_coord = sc if side in (_N, _S) else sr
        required[side].append((base_coord, base_coord + sz - 1))

    def lane_interval(side: int) -> tuple[int, int]:
        return lane[side], lane[side] + sz - 1

    for side in present:
        nxt = _rot(side, side_turn[side])
        if nxt not in present:
            continue
        if chains.get(side) == nxt:
            required[nxt].append(lane_interval(side))
            after = _rot(nxt, side_turn[nxt])
            if after in present:
                excluded[after].append(lane_interval(nxt))
        else:
            excluded[nxt].append(lane_interval(side))

    return required, excluded


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3
    present = set(random.sample([_N, _E, _S, _W], obstacle_count))
    target_hits = random.choice([4, 5])
    second_hits = target_hits - obstacle_count
    if second_hits < 0 or second_hits > 2:
        return None

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
    if min_r > max_r or min_c > max_c:
        return None
    sr, sc = random.randint(min_r, max_r), random.randint(min_c, max_c)

    lane: dict[int, int] = {}
    if _N in present:
        lane[_N] = thickness[_N] + 1
    if _S in present:
        lane[_S] = h - thickness[_S] - sz
    if _W in present:
        lane[_W] = thickness[_W] + 1
    if _E in present:
        lane[_E] = w - thickness[_E] - sz

    chains = _select_chains(present, second_hits)
    if chains is None:
        return None
    side_turn = _assign_turns(present, chains)
    required, excluded = _compute_span_constraints(present, side_turn, chains, sr, sc, sz, lane)

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
    side_color = dict(zip(sorted(present), obstacle_colors))
    bg_candidates = [c for c in BG_COLOR_POOL if c not in legend_colors]
    bg = random.choice(bg_candidates)

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    occupied = np.zeros((h, w), dtype=bool)
    for side in present:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    turns = {side_color[s]: side_turn[s] for s in present}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))
    if len(set(turns.values())) < 2:
        return None

    try:
        launch_dirs = (_N, _E, _S, _W)
        relaxed_out, relaxed_hits, _ = _simulate_paths(
            base, turns, bg, trail_blocks=False, launch_dirs=launch_dirs
        )
        fused_out, fused_hits, fused_stops = _simulate_paths(
            base, turns, bg, trail_blocks=True, launch_dirs=launch_dirs
        )
    except AssertionError:
        return None

    hit_counts_relaxed = [len(beam) for beam in relaxed_hits]
    hit_colors_relaxed = {color for beam in relaxed_hits for _, color, _, _ in beam}
    if sum(hit_counts_relaxed) != target_hits or any(c > 2 for c in hit_counts_relaxed):
        return None
    if set(side_color.values()) - hit_colors_relaxed:
        return None

    metrics = {
        "changed_cells": int(np.sum(fused_out != relaxed_out)),
        "blocked_beams": sum(reason == "trail" for reason in fused_stops),
        "suppressed_hits": sum(len(a) - len(b) for a, b in zip(relaxed_hits, fused_hits)),
    }
    if metrics["blocked_beams"] < 1 or metrics["changed_cells"] < sz:
        return None
    if sum(len(beam) for beam in fused_hits) < 3:
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, fused_out):
        return None
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace legend-ordered teal rays whose turn rules come from arrow glyphs.

    Input structure:
    - The top 6 rows form a grey legend band split into four left-to-right entries.
      Inside each 4x4 black cell, a fixed color swatch identifies one obstacle
      color and a tiny mirrored L glyph encodes that color's turn behavior.
    - Below the legend is the main board with one solid teal square, a single
      background color, and 3-4 solid colored obstacle rectangles touching grid
      edges.

    Rule:
    - Remove the legend from the output.
    - Read legend entries from left to right; they define the launch order
      up, right, down, left.
    - Launch one sz-wide moving teal square for each of those directions in that
      order. When the next incoming strip would hit an obstacle color, turn by the
      glyph assigned to that color and keep going.
    - Earlier painted teal trails remain on the board and block later rays if the
      later ray's next incoming strip would enter those trail cells.
    - Rays only paint background cells and stop at the outer border or at an
      earlier trail.

    Key insight: the legend now encodes two different controls at once: the micro
    arrow glyph gives each obstacle color's turn direction, and the visible
    left-to-right order of those legend entries determines which of the four
    cardinal launches happens first.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert legend_h == LEGEND_H, "legend height must remain fixed"
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend grey must be confined to the top band"

    entries = _parse_legend(g, legend_h)
    turns = {color: turn for color, turn in entries}
    launch_dirs = tuple(i for i, _ in enumerate(entries))

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"
    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    _find_teal_square(base, teal=TEAL)

    out, _hits, _stops = _simulate_paths(
        base, turns, bg, trail_blocks=True, launch_dirs=launch_dirs
    )
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate fused e87109e9 input")
