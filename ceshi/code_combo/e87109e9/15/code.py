# Combo variant of e87109e9 (index 15)
# combo_spec: D4/0+D18/2
# Name: 多色源加颜色继承轨迹

from __future__ import annotations

import random

import numpy as np

# Design analysis for combo D4/0 + D18/2
# a) Concretization + behavior trace
#    - The input still has the same two-zone layout as the base task: a 6x24 legend
#      on top and one main grid below. The legend semantics do not change: each of
#      four colors gets a left-bar or right-bar code, meaning CCW or CW turn.
#    - The source is no longer a solid monochrome square. It is always one isolated
#      2x2 block whose four cells are all different. Their positions encode launch
#      colors:
#        NW -> upward beam, NE -> rightward beam, SE -> downward beam, SW -> leftward beam.
#    - The beams still move as a sliding 2x2 footprint. The fusion change is that
#      color is now stateful twice over:
#        1. The initial segment color comes from the source corner for that direction.
#        2. After a beam hits obstacle color C and turns, all later painted cells use C
#           until the next obstacle hit changes the color again.
#    - Concrete trace:
#        Suppose the source block is [[2, 7], [9, 4]].
#        The upward beam starts in color 2. If its next blocked step meets obstacle 6,
#        it turns per the legend and the next clear step paints 6, not 2. If that 6-segment
#        later meets obstacle 3, it turns again and subsequent cells paint 3.
#        Compared with the base task, the launch color is no longer globally fixed.
#        Compared with D18 alone, the color is no longer fixed per beam after launch.
# b) Complexity calibration
#    - Four distinct source colors make the D18 part visually obvious in 2-3 examples.
#    - Obstacle counts, edge-touching rectangles, legend encoding, and 4-5 total hits
#      stay close to the original task so the geometry is still recognizable.
#    - The generator forces at least one 2-hit beam and rejects samples where inherited
#      colors are not visible, so the D4 part is also observable rather than latent.
#    - Color roles are disjoint inside each sample:
#        source colors != obstacle/legend colors != background color.
#      This prevents accidental ambiguity between "beam payload" and "thing that causes a turn".
# c) Hidden assumptions audited against the old solver
#    - Old source detection searched for one solid teal square. That fails here; the new
#      solver must find one unique 2x2 block of four singleton colors.
#    - Old painting always wrote color 8. That erases both fusion dimensions, so beam state
#      now carries a mutable `paint_color`.
#    - Old obstacle detection read from the partially painted output and ignored only bg/teal.
#      That breaks badly here because trails may use source colors and later obstacle colors.
#      The fused solver therefore detects blockers only from the immutable cropped input grid.
#    - Old generator could tolerate beam overlaps because all trails were effectively one color.
#      Here cross-beam overlap would create order-dependent mixed-color corridors. The generator
#      rejects layouts where different beams would paint the same output cells.
# d) Placement legality
#    - The source remains isolated from all obstacle rectangles by at least one cell.
#    - Obstacles remain solid, monochrome, edge-touching rectangles with original-style
#      thickness/span constraints and no mutual overlap.
#    - The source is placed with enough margin that every beam paints at least one visible
#      first-step strip before any possible turn or boundary stop.
# e) Visual observability
#    - A human can read the 2x2 source locally as a lookup table for the four launch colors.
#    - A human can read the legend locally as a lookup table for turn directions.
#    - The output then combines both: direction comes from geometry, turn side from the legend,
#      and the trail color after each bend visibly matches the obstacle that caused that bend.
# f) Visual cleanliness
#    - There is only one multicolor object in the input main grid: the 2x2 source.
#    - Obstacles stay monochrome and edge-anchored, so they remain easy to identify.
#    - Different beams are required to stay disjoint in the generated samples, avoiding
#      messy shared corridors with conflicting colors.
# g) Color inference
#    - No rule depends on numeric color ordering or privileged treatment of black.
#    - Every color change has a structural cue: source-corner position for launch color,
#      obstacle contact for inherited color.
# New solver assumptions guaranteed by the generator
#    - Exactly one 2x2 block has four distinct colors and each of those colors occurs once.
#    - Obstacle colors are a subset of legend colors and disjoint from the source palette.
#    - Different beams never require resolving conflicting overwrites on the same cells.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)
_SOURCE_SZ = 2

# Main-grid colors: all ARC colors except the dedicated legend background 5.
_ALL_MAIN_COLORS = [0, 1, 2, 3, 4, 6, 7, 8, 9]
_LEGEND_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

# Canvas dimensions
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_BG = 5
LEGEND_CELL_STARTS = [1, 7, 13, 19]

# Source placement
SQUARE_EDGE_GAP = 5

# Obstacles
OBS_COUNT_4_PROB = 0.65
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Retry budget
MAX_RETRIES = 1000


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Decode obstacle-color turn directions from the legend band."""
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
        color = int(next(iter(colors)))
        xs = np.where(interior == color)[1]
        x = int(xs[0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"

    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_source_block(
    grid: np.ndarray,
) -> tuple[int, int, dict[int, int], set[int]]:
    """Return top-left coordinate plus direction->launch-color map for the unique 2x2 source."""
    vals, cnts = np.unique(grid, return_counts=True)
    counts = {int(v): int(c) for v, c in zip(vals, cnts, strict=False)}

    candidates: list[tuple[int, int, dict[int, int], set[int]]] = []
    h, w = grid.shape
    for r in range(h - 1):
        for c in range(w - 1):
            blk = grid[r : r + 2, c : c + 2]
            flat = [int(x) for x in blk.flatten()]
            if len(set(flat)) != 4:
                continue
            if any(counts[v] != 1 for v in flat):
                continue
            beam_colors = {
                _N: int(blk[0, 0]),
                _E: int(blk[0, 1]),
                _S: int(blk[1, 1]),
                _W: int(blk[1, 0]),
            }
            candidates.append((int(r), int(c), beam_colors, set(flat)))

    assert len(candidates) == 1, "main grid must contain exactly one multicolor 2x2 source"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _simulate_paths(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    inherit_color: bool,
    enforce_disjoint: bool,
) -> tuple[
    np.ndarray,
    list[list[tuple[int, int, int, int]]],
    list[dict[tuple[int, int], int]],
]:
    """
    Simulate the four 2x2 beams on the cropped main grid.

    `inherit_color=False` yields the D18-only behavior where each beam keeps its source color.
    `inherit_color=True` yields the fused behavior where each bounce updates the paint color.
    """
    sr, sc, beam_colors, beam_palette = _find_source_block(base)
    assert not (beam_palette & set(turns)), "source colors and obstacle colors must be disjoint"

    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []
    beam_visible: list[dict[tuple[int, int], int]] = []
    owner = np.full((h, w), -1, dtype=int)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + _SOURCE_SZ - 1 < h and c + _SOURCE_SZ - 1 < w

    def obstacle_at(r: int, c: int) -> int | None:
        blk = base[r : r + _SOURCE_SZ, c : c + _SOURCE_SZ]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors -= beam_palette
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    def paint(r: int, c: int, color: int, beam_id: int) -> dict[tuple[int, int], int]:
        blk = out[r : r + _SOURCE_SZ, c : c + _SOURCE_SZ]
        own = owner[r : r + _SOURCE_SZ, c : c + _SOURCE_SZ]
        if enforce_disjoint:
            assert not np.any((own >= 0) & (own != beam_id)), "different beams must stay disjoint"
        mask = blk == bg
        blk[mask] = int(color)
        own[mask] = int(beam_id)

        painted: dict[tuple[int, int], int] = {}
        rows, cols = np.where(mask)
        for rr, cc in zip(rows, cols, strict=False):
            painted[(int(r + rr), int(c + cc))] = int(color)
        return painted

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        paint_color = int(beam_colors[d0])
        beam_hits: list[tuple[int, int, int, int]] = []
        visible_cells: dict[tuple[int, int], int] = {}
        seen: set[tuple[int, int, int, int]] = set()

        while True:
            state = (r, c, d, paint_color)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                visible_cells.update(paint(r, c, paint_color, d0))
                continue

            assert obs in turns, "obstacle color must be encoded by legend"
            beam_hits.append((d, obs, nr, nc))
            d = _rot(d, turns[obs])
            if inherit_color:
                paint_color = int(obs)

        hits.append(beam_hits)
        beam_visible.append(visible_cells)

    return out, hits, beam_visible


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace four 2x2 beams whose launch colors come from the source and whose later colors come from collisions.

    Input structure:
    - Top band: a 6-row legend on color-5 background with four colored vertical bars; each bar's left/right placement encodes CCW/CW turning for that obstacle color.
    - Below the legend: one main grid containing a single isolated 2x2 multicolor source block, background cells, and 3-4 monochrome obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - Read the source as a directional key: NW launches the upward beam, NE the rightward beam, SE the downward beam, SW the leftward beam.
    - Each beam moves as the original sliding 2x2 footprint and paints only currently background cells.
    - Before the first obstacle, the beam paints in its own source-derived color. Whenever the next step would hit obstacle color C, the beam turns according to the legend and all later painted cells use color C until another obstacle changes it again.
    - Obstacles are not overwritten, and a beam stops when the next 2x2 footprint would leave the main grid.

    Key insight:
    - Solving now requires fusing two local encodings: the source corners tell you the four launch colors, while the legend still tells you only how each obstacle color redirects the beam. The visible trail color after each bend is inherited from the obstacle that caused that bend.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_turn = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    _find_source_block(base)

    out, _hits, _beam_visible = _simulate_paths(
        base,
        legend_turn,
        bg,
        inherit_color=True,
        enforce_disjoint=False,
    )
    return out


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering every required interval while avoiding every excluded interval."""
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


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors, strict=False):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = int(color)
    return legend


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


def _select_chains(
    present: set[int], second_hits: int
) -> dict[int, int] | None:
    """Pick non-overlapping adjacent source->target pairs for the beams that bounce twice."""
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


def _assign_turns(
    present: set[int], chains: dict[int, int]
) -> dict[int, str]:
    """Assign legend turns per obstacle side so the chosen chains are realized."""
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


def _sample_color_roles() -> tuple[dict[int, int], np.ndarray, list[int], int]:
    """Choose disjoint source colors, legend colors, and background."""
    source_colors = random.sample(_ALL_MAIN_COLORS, 4)
    beam_colors = {
        _N: int(source_colors[0]),
        _E: int(source_colors[1]),
        _S: int(source_colors[2]),
        _W: int(source_colors[3]),
    }
    source_block = np.array(
        [
            [beam_colors[_N], beam_colors[_E]],
            [beam_colors[_W], beam_colors[_S]],
        ],
        dtype=int,
    )

    remaining_for_legend = [c for c in _LEGEND_COLOR_POOL if c not in source_colors]
    assert len(remaining_for_legend) >= 4
    legend_colors = random.sample(remaining_for_legend, 4)

    remaining_bg = [
        c for c in _ALL_MAIN_COLORS
        if c not in source_colors and c not in legend_colors
    ]
    assert len(remaining_bg) == 1
    bg = int(remaining_bg[0])
    return beam_colors, source_block, legend_colors, bg


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = _SOURCE_SZ
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
    thickness = {
        side: t
        for side, t in zip((_N, _S, _W, _E), (*ns, *ew), strict=False)
        if t is not None
    }

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
    required, excluded = _compute_span_constraints(
        present,
        side_turn,
        chains,
        sr,
        sc,
        sz,
        lane,
    )

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

    beam_colors, source_block, legend_colors, bg = _sample_color_roles()
    source_palette = set(int(x) for x in source_block.flatten())
    obstacle_colors = random.sample(legend_colors, obstacle_count)
    side_color = dict(zip(sorted(present), obstacle_colors, strict=False))

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = source_block
    occupied = np.zeros((h, w), dtype=bool)
    for side in present:
        r0, r1, c0, c1 = _obstacle_rect(side, thickness[side], spans[side], h, w)
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return None
        if not (r1 < sr - 1 or r0 > sr + sz or c1 < sc - 1 or c0 > sc + sz):
            return None
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
        base[r0 : r1 + 1, c0 : c1 + 1] = int(side_color[side])

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    turns = {int(side_color[s]): side_turn[s] for s in present}
    for color in legend_colors:
        turns.setdefault(int(color), random.choice(["L", "R"]))

    try:
        expected, hits, beam_visible = _simulate_paths(
            base,
            turns,
            bg,
            inherit_color=True,
            enforce_disjoint=True,
        )
        d18_only, _d18_hits, _d18_visible = _simulate_paths(
            base,
            turns,
            bg,
            inherit_color=False,
            enforce_disjoint=True,
        )
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _ in beam}
    inherited_visible_colors = {
        color
        for paints in beam_visible
        for color in paints.values()
        if color not in source_palette
    }
    inherited_visible_cells = sum(
        1
        for paints in beam_visible
        for color in paints.values()
        if color not in source_palette
    )
    visible_launch_colors = {
        beam_colors[i]
        for i, paints in enumerate(beam_visible)
        if any(color == beam_colors[i] for color in paints.values())
    }

    if sum(hit_counts) != target_hits or any(count > 2 for count in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None
    if len(source_palette) != 4:
        return None
    if len(visible_launch_colors) != 4:
        return None
    if len(inherited_visible_colors) < 1 or inherited_visible_cells < sz:
        return None
    if np.array_equal(expected, d18_only):
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
