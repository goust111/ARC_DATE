# Combo variant of e87109e9 (index 28)
# combo_spec: D5/3+D16/2
# Name: 边缘源加边缘滑行

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for the combo variant
# spec:
#   concretization:
#     unchanged_core:
#       - input still has the same 6x24 legend band on top
#       - obstacle colors are still read from the legend, and left-vs-right bar placement
#         still means turn-left vs turn-right
#       - output still drops the legend and paints teal trails only onto background cells
#     D5_edge_sources:
#       - the interior teal square is removed completely
#       - instead, the playfield contains one horizontal teal 1x2 source on the top or
#         bottom edge and one vertical teal 2x1 source on the left or right edge
#       - each source emits exactly one beam inward, so the puzzle now has two beams rather
#         than the original four
#       - the beam footprint is a translated copy of that source pair and keeps the same
#         orientation for the whole trace; only its direction changes
#     D16_edge_sliding:
#       - when a beam reaches the opposite edge, the next out-of-bounds attempt is no longer
#         terminal
#       - instead the beam turns 90 degrees clockwise in place and slides along that border
#       - if a beam is already at a corner, the border interaction stops there instead of
#         wrapping around a second edge
#     fused_trace_example:
#       - suppose a top-edge source occupies (0,9)-(0,10)
#       - original task logic does not apply because there is no interior teal square
#       - in this combo, that 1x2 source moves south until it sits on the bottom edge,
#         then the next south step fails, so it turns clockwise to move west along the bottom
#       - after a few westward edge steps it hits a bottom obstacle whose legend entry is on
#         the right side, so it turns clockwise again and heads north across the interior
#       - on reaching the top edge it turns clockwise once more and slides east until a corner
#       - the visible difference from the base task is both the edge origin and the long
#         border-hugging teal runs
#   complexity_calibration:
#     - two sources give repeated evidence for the new source rule without producing the
#       clutter of four simultaneous border traces
#     - the generator forces both beams to perform a genuine opposite-edge slide and to hit
#       at least one obstacle while sliding, so the combo is visually distinct from both the
#       base task and a trivial "straight-to-edge" simplification
#     - obstacle turns used on the first sliding edge are chosen so the beam re-enters the
#       board rather than bouncing uselessly outward; this keeps the legend behavior visible
#   hidden_assumptions_review:
#     - the original `_find_teal_square(...)` assumption is invalid and is replaced by
#       `_extract_sources(...)`, which expects 1x2 or 2x1 teal components lying wholly on one
#       border and not around corners
#     - the original square-footprint `in_bounds` assumption becomes shape-dependent because
#       a source can be horizontal or vertical; solver and generator therefore track each
#       source's bbox shape explicitly
#     - the original `if not in_bounds(...): break` boundary rule is invalid; it is replaced
#       with a deterministic clockwise border turn that only fails at corners
#     - obstacle lookup still assumes a footprint never overlaps multiple obstacle colors at
#       once; generator keeps rectangles disjoint to preserve that invariant
#     - a new solver assumption is that every beam state can be keyed by `(r,c,h,w,d)`; the
#       fixed-orientation source-pair model makes this sufficient for cycle detection
#   placement_legality:
#     - sources are placed away from corners and on different axes, so their connected
#       components are unambiguous
#     - each obstacle rectangle is edge-touching, non-overlapping, and positioned away from
#       the straight inward lane of the beam that must first reach the opposite edge
#     - the generator validates every sample with the shared simulator and rejects any case
#       where the no-slide baseline matches the combo output
#   visual_observability:
#     - the only teal in the input is on the border, so a human can immediately see that the
#       source no longer lives inside the board
#     - the output contains long teal tracks on the frame, which makes the edge-sliding rule
#       legible from a few examples
#     - the legend still supplies a small repeated binary code (left bar vs right bar), so
#       the color-to-turn mapping remains human-inferable
#   cleanliness:
#     - only two real obstacles are used, one for each beam's first sliding edge, keeping the
#       board readable
#     - corner wrapping is forbidden, which avoids noisy perimeter loops
#   color_inference:
#     - only two legend colors are behaviorally active in a given sample; the other two are
#       harmless fillers, so the learner does not need to infer many unrelated color roles

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

SOURCE_LEN = 2
SOURCE_MARGIN = 5
SLIDE_GAP_MIN = 3
OBS_THICKNESS_RANGE = (2, 3)
OBS_SPAN_LEN_RANGE = (5, 9)
MAX_RETRIES = 700


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Read left/right legend bars as CCW/CW turn instructions."""
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


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _source_direction(edge: int) -> int:
    return {_N: _S, _S: _N, _W: _E, _E: _W}[edge]


def _touching_edges(r: int, c: int, h: int, w: int, H: int, W: int) -> list[int]:
    edges: list[int] = []
    if r == 0:
        edges.append(_N)
    if c + w - 1 == W - 1:
        edges.append(_E)
    if r + h - 1 == H - 1:
        edges.append(_S)
    if c == 0:
        edges.append(_W)
    return edges


def _in_bounds(r: int, c: int, h: int, w: int, H: int, W: int) -> bool:
    return 0 <= r and 0 <= c and r + h - 1 < H and c + w - 1 < W


def _extract_sources(base: np.ndarray) -> list[dict[str, int]]:
    """Return edge-source descriptors for all teal 2-cell border components."""
    H, W = base.shape
    sources: list[dict[str, int]] = []
    for cells in components_mask(base == TEAL, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        area = int(np.sum(base[r0 : r1 + 1, c0 : c1 + 1] == TEAL))
        assert area == 2 and (h, w) in {(1, 2), (2, 1)}, "teal objects must be source pixel pairs"

        if r0 == 0 and r1 == 0:
            edge = _N
        elif r0 == H - 1 and r1 == H - 1:
            edge = _S
        elif c0 == 0 and c1 == 0:
            edge = _W
        elif c0 == W - 1 and c1 == W - 1:
            edge = _E
        else:
            raise AssertionError("each source pair must lie wholly on one border")

        assert 0 < r0 < H - 2 or edge in (_N, _S), "vertical sources must avoid corners"
        assert 0 < c0 < W - 2 or edge in (_W, _E), "horizontal sources must avoid corners"
        sources.append(
            {
                "r": int(r0),
                "c": int(c0),
                "h": h,
                "w": w,
                "edge": edge,
                "d": _source_direction(edge),
            }
        )
    assert 1 <= len(sources) <= 2, "this combo expects one or two edge source pairs"
    return sorted(sources, key=lambda src: (src["edge"], src["r"], src["c"]))


def _simulate_paths(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    slide_at_edge: bool,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]], dict[str, object]]:
    """Simulate all source-pair beams and collect statistics used by generator checks."""
    sources = _extract_sources(base)
    out = base.copy()
    H, W = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []
    beam_slide_steps: list[int] = []
    beam_slide_events: list[int] = []

    def paint(r: int, c: int, h: int, w: int) -> None:
        blk = out[r : r + h, c : c + w]
        blk[blk == bg] = TEAL
        out[r : r + h, c : c + w] = blk

    def obstacle_at(r: int, c: int, h: int, w: int) -> int | None:
        blk = out[r : r + h, c : c + w]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    def boundary_turn(r: int, c: int, h: int, w: int, d: int) -> int | None:
        edges = _touching_edges(r, c, h, w, H, W)
        assert edges, "out-of-bounds step must start from a border-touching footprint"
        if len(edges) >= 2:
            return None
        assert edges[0] == d, "beam must leave through the border it is touching"
        return _rot_cw(d)

    for src in sources:
        r, c, h, w, d = src["r"], src["c"], src["h"], src["w"], src["d"]
        beam_hits: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int, int, int]] = set()
        sliding = False
        slide_steps = 0
        slide_events = 0

        while True:
            state = (r, c, h, w, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not _in_bounds(nr, nc, h, w, H, W):
                if not slide_at_edge:
                    break
                nd = boundary_turn(r, c, h, w, d)
                if nd is None:
                    break
                d = nd
                sliding = True
                slide_events += 1
                continue

            obs = obstacle_at(nr, nc, h, w)
            if obs is None:
                r, c = nr, nc
                paint(r, c, h, w)
                if sliding:
                    slide_steps += 1
                continue

            assert obs in turns, "obstacle color must be encoded by legend"
            beam_hits.append((d, obs, nr, nc))
            d = _rot(d, turns[obs])
            sliding = False

        hits.append(beam_hits)
        beam_slide_steps.append(slide_steps)
        beam_slide_events.append(slide_events)

    stats: dict[str, object] = {
        "source_count": len(sources),
        "slide_steps_total": int(sum(beam_slide_steps)),
        "slide_events_total": int(sum(beam_slide_events)),
        "beam_slide_steps": beam_slide_steps,
        "beam_slide_events": beam_slide_events,
        "sources": [(src["edge"], src["r"], src["c"], src["h"], src["w"]) for src in sources],
    }
    return out, hits, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch teal beams from edge pixel-pairs, then let opposite-edge hits continue as border slides.

    Input structure:
    - The top band is still a 6-row legend: each colored vertical bar identifies an obstacle color,
      and the bar's left/right placement inside its cell encodes turn-left vs turn-right.
    - The lower playfield contains only background, edge-touching obstacle rectangles, and one or two
      teal source pairs that sit directly on the border instead of an interior square.

    Rule:
    - Remove the legend before producing the output.
    - Each teal 2-cell border component emits one beam inward; the beam is a translated copy of that
      source pair and paints teal only onto background cells.
    - When a beam would overlap an obstacle color, consult the legend and rotate 90 degrees left or
      right according to that color's bar position.
    - When a beam would leave the grid through a single border, do not stop: turn 90 degrees clockwise
      in place and slide along that border. If the beam has already reached a corner, stop there.

    Key insight: The source is encoded by where the teal pair touches the frame, while obstacle colors
    still encode the voluntary turns; the frame itself contributes an extra compulsory clockwise turn.
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
    _extract_sources(base)

    out, _, _ = _simulate_paths(base, legend_turn, bg, slide_at_edge=True)
    return out


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick an edge span covering required intervals, avoiding excluded ones and corner clutter."""
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, axis_len - 4)
    if hi < lo:
        return None

    choices: list[tuple[int, int]] = []
    for length in range(lo, hi + 1):
        start_lo = max(2, req1 - length + 1)
        start_hi = min(req0, axis_len - length - 2)
        for start in range(start_lo, start_hi + 1):
            end = start + length - 1
            if all(end < a or start > b for a, b in excluded):
                choices.append((start, end))
    if not choices:
        return None
    return random.choice(choices)


def _obstacle_rect(
    side: int,
    thickness: int,
    span: tuple[int, int],
    h: int,
    w: int,
) -> tuple[int, int, int, int]:
    s0, s1 = span
    if side == _N:
        return 0, thickness - 1, s0, s1
    if side == _S:
        return h - thickness, h - 1, s0, s1
    if side == _W:
        return s0, s1, 0, thickness - 1
    return s0, s1, w - thickness, w - 1


def _pick_horizontal_source(h: int, w: int) -> dict[str, int]:
    edge = random.choice([_N, _S])
    c = random.randint(SOURCE_MARGIN, w - SOURCE_MARGIN - SOURCE_LEN)
    r = 0 if edge == _N else h - 1
    return {"edge": edge, "r": r, "c": c, "h": 1, "w": SOURCE_LEN, "d": _source_direction(edge)}


def _pick_vertical_source(h: int, w: int) -> dict[str, int]:
    edge = random.choice([_W, _E])
    r = random.randint(SOURCE_MARGIN, h - SOURCE_MARGIN - SOURCE_LEN)
    c = 0 if edge == _W else w - 1
    return {"edge": edge, "r": r, "c": c, "h": SOURCE_LEN, "w": 1, "d": _source_direction(edge)}


def _horizontal_obstacle_plan(src: dict[str, int], h: int, w: int) -> tuple[int, int]:
    c0 = src["c"]
    if src["edge"] == _N:
        target = random.randint(2, c0 - SLIDE_GAP_MIN)
    else:
        target = random.randint(c0 + SLIDE_GAP_MIN, w - SOURCE_LEN - 2)
    return target, target + SOURCE_LEN - 1


def _vertical_obstacle_plan(src: dict[str, int], h: int, w: int) -> tuple[int, int]:
    r0 = src["r"]
    if src["edge"] == _W:
        target = random.randint(r0 + SLIDE_GAP_MIN, h - SOURCE_LEN - 2)
    else:
        target = random.randint(2, r0 - SLIDE_GAP_MIN)
    return target, target + SOURCE_LEN - 1


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    bg = random.choice(BG_COLOR_POOL)
    legend_colors = random.sample([c for c in LEGEND_COLOR_POOL if c != bg], 4)

    h_src = _pick_horizontal_source(h, w)
    v_src = _pick_vertical_source(h, w)

    required_h = _horizontal_obstacle_plan(h_src, h, w)
    required_v = _vertical_obstacle_plan(v_src, h, w)
    first_side_h = _S if h_src["edge"] == _N else _N
    first_side_v = _E if v_src["edge"] == _W else _W

    thickness_h = random.randint(*OBS_THICKNESS_RANGE)
    thickness_v = random.randint(*OBS_THICKNESS_RANGE)
    span_h = _choose_span(
        w,
        [required_h],
        [(h_src["c"], h_src["c"] + SOURCE_LEN - 1)],
        OBS_SPAN_LEN_RANGE[0],
        OBS_SPAN_LEN_RANGE[1],
    )
    span_v = _choose_span(
        h,
        [required_v],
        [(v_src["r"], v_src["r"] + SOURCE_LEN - 1)],
        OBS_SPAN_LEN_RANGE[0],
        OBS_SPAN_LEN_RANGE[1],
    )
    if span_h is None or span_v is None:
        return None

    used_colors = random.sample(legend_colors, 2)
    turns = {color: random.choice(["L", "R"]) for color in legend_colors}
    turns[used_colors[0]] = "R"
    turns[used_colors[1]] = "R"

    side_color = {first_side_h: used_colors[0], first_side_v: used_colors[1]}

    base = np.full((h, w), bg, dtype=int)
    base[h_src["r"] : h_src["r"] + h_src["h"], h_src["c"] : h_src["c"] + h_src["w"]] = TEAL
    base[v_src["r"] : v_src["r"] + v_src["h"], v_src["c"] : v_src["c"] + v_src["w"]] = TEAL

    occupied = np.zeros((h, w), dtype=bool)
    occupied[h_src["r"] : h_src["r"] + h_src["h"], h_src["c"] : h_src["c"] + h_src["w"]] = True
    occupied[v_src["r"] : v_src["r"] + v_src["h"], v_src["c"] : v_src["c"] + v_src["w"]] = True

    rects = [
        (first_side_h, thickness_h, span_h),
        (first_side_v, thickness_v, span_v),
    ]
    for side, thickness, span in rects:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness, span, h, w)
        if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        baseline, _, base_stats = _simulate_paths(base, turns, bg, slide_at_edge=False)
        expected, hits, stats = _simulate_paths(base, turns, bg, slide_at_edge=True)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    if hit_counts != [1, 1]:
        return None
    if int(stats["source_count"]) != 2:
        return None
    if int(stats["slide_events_total"]) < 4:
        return None
    if int(stats["slide_steps_total"]) < 6:
        return None
    if np.array_equal(expected, baseline):
        return None
    if int(base_stats["slide_events_total"]) != 0:
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
    raise RuntimeError("failed to generate combo e87109e9 input")
