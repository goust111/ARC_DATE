# Combo variant of e87109e9 (index 29)
# combo_spec: D4/0+D12/1
# Name: 障碍物重涂加颜色继承

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for the combo variant:
# - Concrete behavior and traced example:
#   Keep the original two-zone layout: a 6x24 legend on top and one cropped main
#   grid below with a single teal source square plus 3-4 solid obstacle
#   rectangles. The fused rule changes the collision semantics. Suppose the north
#   beam first hits a red obstacle while still teal: the red obstacle is repainted
#   teal, the beam turns by the legend, and its next segment becomes red. If that
#   red segment later hits a blue obstacle, the whole blue obstacle is repainted
#   red and the beam's following segment becomes blue. This is a true color swap:
#   obstacle gets the incoming beam color, while the outgoing segment gets the
#   obstacle color from immediately before the hit.
# - Complexity calibration:
#   The generator keeps the original rectangle geometry, legend encoding, and
#   square-width beam motion. To make the fusion legible without becoming noisy,
#   every sample contains exactly one ordered 2-hit chain: an earlier beam hits a
#   later obstacle first, so a later direct beam reads a recolored obstacle. That
#   gives one history-dependent collision per grid, enough to expose the fused
#   rule while avoiding many re-hit objects or long color cascades.
# - Hidden-assumption audit of the original solver:
#   * Looking for blockers inside the painted output no longer works: recolored
#     trail cells and recolored obstacles would be conflated. The solver therefore
#     builds fixed obstacle ids from the cropped input and keeps obstacle colors in
#     a separate mutable state table.
#   * Painting is no longer "always teal"; each beam carries its own current color,
#     starting at teal and updating to the pre-collision obstacle color after each
#     hit.
#   * Re-hits can make a beam revisit the same (r,c,d) with different object
#     colors. Cycle detection must therefore include both beam color and current
#     obstacle-color state.
#   * Recoloring an obstacle to teal would make a later hit impossible to decode
#     from the legend. The generator avoids that failure mode by forcing the only
#     repeated obstacle to be hit first by an earlier already-recolored beam, so
#     its later color is still a legend color rather than teal.
#   * Recoloring could also change future turn geometry if the replacement color
#     had a different legend direction. The generator binds the source and target
#     obstacle of the 2-hit chain to the same turn direction, so recolor changes
#     visible color state without silently changing path topology.
# - Placement legality and cleanliness:
#   Obstacles remain disjoint solid rectangles, separated from the teal square and
#   from one another, with spans chosen by required/excluded intervals exactly as
#   in the base generator. Samples with conflicting multi-beam overwrites or mixed
#   obstacle footprints are rejected.
# - Visual observability and inferability:
#   Each accepted sample must contain (1) a non-teal obstacle repaint event and
#   (2) a later collision whose pre-hit obstacle color differs from that
#   obstacle's input color. That guarantees the fused statefulness is not merely
#   latent. The generator also rejects cases where the combo output matches either
#   the original fixed-teal rule or the color-inheritance-only rule, so the fusion
#   is visible in the final pixels.
# - New solver assumptions that the generator must satisfy:
#   The cropped main grid contains exactly one solid teal square; every obstacle is
#   a monochrome 4-connected component; any sz-wide collision footprint covers at
#   most one obstacle component; and repeated hits occur only in the single
#   generator-controlled ordered chain described above.

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

# Teal square
TEAL_SZ2_PROB = 0.75
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
MAX_RETRIES = 800


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """
    Legend is a top region with a background color (typically 5) and 4 vertical bars.
    Each bar's horizontal position inside its cell encodes turn direction:
      - bar on left  -> 'L' (turn counterclockwise)
      - bar on right -> 'R' (turn clockwise)
    """
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


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    """Return (r0, c0, size) for a solid square teal component."""
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        size = hh
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != size * size:
            continue
        candidates.append((int(r0), int(c0), int(size)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _collect_obstacles(
    base: np.ndarray, bg: int
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[int]]:
    """Return obstacle id grid plus per-obstacle rows, cols, and input colors."""
    id_grid = np.full(base.shape, -1, dtype=int)
    rows_list: list[np.ndarray] = []
    cols_list: list[np.ndarray] = []
    colors: list[int] = []
    oid = 0
    obstacle_colors = [
        int(c) for c in np.unique(base) if int(c) not in {int(bg), int(TEAL)}
    ]
    for color in obstacle_colors:
        for cells in components_mask(base == int(color), connectivity=4):
            assert cells.size, "obstacle components must be non-empty"
            rows = cells[:, 0].astype(int)
            cols = cells[:, 1].astype(int)
            id_grid[rows, cols] = oid
            rows_list.append(rows)
            cols_list.append(cols)
            colors.append(int(color))
            oid += 1
    return id_grid, rows_list, cols_list, colors


def _simulate_paths(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    *,
    inherit_color: bool,
    recolor_obstacle: bool,
    enforce_clean: bool,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]], list[dict[tuple[int, int], int]], list[dict[str, int]]]:
    """
    Simulate the four square beams on the cropped main grid.

    `inherit_color` controls whether the outgoing segment adopts the obstacle's
    pre-hit color. `recolor_obstacle` controls whether the whole obstacle adopts
    the incoming beam color.
    """
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    id_grid, obstacle_rows, obstacle_cols, input_colors = _collect_obstacles(base, bg)
    current_colors = list(map(int, input_colors))

    out = base.copy()
    H, W = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []
    beam_paints: list[dict[tuple[int, int], int]] = []
    events: list[dict[str, int]] = []
    global_paints: dict[tuple[int, int], int] = {}

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < H and c + sz - 1 < W

    def obstacle_at(r: int, c: int) -> tuple[int, int] | None:
        ids = set(map(int, np.unique(id_grid[r : r + sz, c : c + sz])))
        ids.discard(-1)
        if not ids:
            return None
        assert len(ids) == 1, "beam footprint must not hit multiple obstacles at once"
        oid = int(next(iter(ids)))
        return oid, int(current_colors[oid])

    def paint_block(rr: int, cc: int, color: int, beam_cells: dict[tuple[int, int], int]) -> None:
        blk_base = base[rr : rr + sz, cc : cc + sz]
        bg_rows, bg_cols = np.where(blk_base == bg)
        for roff, coff in zip(bg_rows, bg_cols):
            cell = (int(rr + roff), int(cc + coff))
            beam_cells[cell] = int(color)
            out[cell] = int(color)

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        paint_color = TEAL
        beam_hits: list[tuple[int, int, int, int]] = []
        beam_cells: dict[tuple[int, int], int] = {}
        seen: set[tuple[int, int, int, int, tuple[int, ...]]] = set()

        while True:
            state = (r, c, d, int(paint_color), tuple(current_colors))
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            hit = obstacle_at(nr, nc)
            if hit is None:
                r, c = nr, nc
                paint_block(r, c, int(paint_color), beam_cells)
                continue

            oid, obstacle_color = hit
            assert obstacle_color in turns, "encountered obstacle color must be encoded by legend"
            incoming_color = int(paint_color)
            beam_hits.append((d, int(obstacle_color), nr, nc))
            events.append(
                {
                    "beam": int(d0),
                    "dir_before": int(d),
                    "row": int(nr),
                    "col": int(nc),
                    "obstacle_id": int(oid),
                    "input_color": int(input_colors[oid]),
                    "obstacle_before": int(obstacle_color),
                    "incoming_color": int(incoming_color),
                }
            )

            if recolor_obstacle:
                current_colors[oid] = int(incoming_color)
                out[obstacle_rows[oid], obstacle_cols[oid]] = int(incoming_color)

            d = _rot(d, turns[obstacle_color])
            if inherit_color:
                paint_color = int(obstacle_color)
            assert len(beam_hits) <= 2, "generator only supports up to two hits per beam"

        if enforce_clean:
            for cell, color in beam_cells.items():
                prev = global_paints.get(cell)
                if prev is not None:
                    assert prev == color, "overlapping beam trails must agree on color"
                global_paints[cell] = color
        else:
            global_paints.update(beam_cells)
        hits.append(beam_hits)
        beam_paints.append(beam_cells)

    return out, hits, beam_paints, events


def _conflict_stats(beam_paints: list[dict[tuple[int, int], int]]) -> tuple[int, int]:
    """Return (#cells painted by multiple beams, #cells with multiple colors)."""
    owners: dict[tuple[int, int], list[int]] = {}
    for paints in beam_paints:
        for cell, color in paints.items():
            owners.setdefault(cell, []).append(int(color))
    overlap_cells = sum(1 for colors in owners.values() if len(colors) > 1)
    conflict_cells = sum(1 for colors in owners.values() if len(set(colors)) > 1)
    return overlap_cells, conflict_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit four square beams that swap colors with obstacles whenever they bounce.

    Input structure:
    - Top band: a 6-row legend on color-5 background with four colored vertical bars; left placement means counterclockwise turn and right placement means clockwise turn for that color.
    - Below the legend: one main grid containing a single solid teal (8) square, background cells, and several solid colored obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - Launch one sz-wide beam from the teal square in each cardinal direction.
    - As a beam moves, it paints only background cells using its current color; every beam starts teal.
    - When the next sz-wide step would hit an obstacle of color C, first read C to decide the turn, then swap colors: the whole obstacle is repainted to the beam's incoming color, and the beam's new current color becomes C.
    - Continue from the turned direction until the next step would leave the grid.

    Key insight: The legend still encodes only turning, but collision color is now stateful. To predict later segments you must track both the moving beam color and which earlier beam last repainted each obstacle.
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
    _find_teal_square(base, teal=TEAL)

    out, _hits, _beam_paints, _events = _simulate_paths(
        base,
        legend_turn,
        bg,
        inherit_color=True,
        recolor_obstacle=True,
        enforce_clean=False,
    )
    return out


def _span_choices(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> list[tuple[int, int]]:
    """Enumerate spans covering all required intervals and avoiding all excluded intervals."""
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
    return [(start, end) for _, start, end in choices]


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a (start, end) span covering all required intervals and avoiding all excluded intervals."""
    choices = _span_choices(axis_len, required, excluded, min_len, max_len)
    if not choices:
        return None
    lengths = sorted({end - start + 1 for start, end in choices})
    if random.random() < SPAN_COMPACT_PROB:
        cutoff = lengths[min(len(lengths) - 1, max(0, len(lengths) // 2))]
        pool = [(start, end) for start, end in choices if end - start + 1 <= cutoff]
    else:
        pool = choices
    return random.choice(pool)


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    """Build the 6x24 legend band from 4 colors and their turn assignments."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


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


def _select_ordered_chain(present: set[int]) -> tuple[int, int] | None:
    """Pick one adjacent source->target pair with source beam processed earlier."""
    candidates: list[tuple[int, int]] = []
    for side in present:
        for turn, nxt in (("L", (side - 1) % 4), ("R", (side + 1) % 4)):
            if nxt not in present or side >= nxt:
                continue
            # Later, the target beam will read the source color and therefore
            # follow the source turn. Keep that redirected leg aimed at the
            # unique missing side to avoid extra hits and large overlaps.
            if _rot(nxt, turn) in present:
                continue
            candidates.append((side, nxt))
    return random.choice(candidates) if candidates else None


def _assign_turns(
    present: set[int], source: int, target: int
) -> dict[int, str]:
    """
    Assign turns with exactly one planned chain.

    The source turns into the target. The target uses the opposite turn so the
    source beam's second post-hit leg and the target beam's later history-driven
    leg diverge instead of painting the same long corridor.
    """
    source_turn = "L" if target == (source - 1) % 4 else "R"
    target_turn = "R" if source_turn == "L" else "L"
    side_turn: dict[int, str] = {source: source_turn, target: target_turn}
    for side in present:
        if side in side_turn:
            continue
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
    """Build required/excluded intervals for each obstacle's span selection."""
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
    obstacle_count = 3
    present = set(random.sample([_N, _E, _S, _W], obstacle_count))

    chain = _select_ordered_chain(present)
    if chain is None:
        return None
    source, target = chain
    chains = {source: target}
    target_hits = obstacle_count + 1

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

    side_turn = _assign_turns(present, source, target)
    required, excluded = _compute_span_constraints(
        present, side_turn, chains, sr, sc, sz, lane
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

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    obstacle_colors = random.sample(legend_colors, obstacle_count)
    side_color = dict(zip(sorted(present), obstacle_colors))
    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])

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
    unused_turn = random.choice(["L", "R"])
    for color in legend_colors:
        turns.setdefault(color, unused_turn)

    try:
        combo, hits, beam_paints, events = _simulate_paths(
            base,
            turns,
            bg,
            inherit_color=True,
            recolor_obstacle=True,
            enforce_clean=False,
        )
        d4_only, _d4_hits, _d4_paints, _d4_events = _simulate_paths(
            base,
            turns,
            bg,
            inherit_color=True,
            recolor_obstacle=False,
            enforce_clean=False,
        )
        original, _o_hits, _o_paints, _o_events = _simulate_paths(
            base,
            turns,
            bg,
            inherit_color=False,
            recolor_obstacle=False,
            enforce_clean=False,
        )
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    overlap_cells, conflict_cells = _conflict_stats(beam_paints)
    history_hits = sum(
        1 for ev in events if int(ev["obstacle_before"]) != int(ev["input_color"])
    )
    non_teal_recolors = sum(1 for ev in events if int(ev["incoming_color"]) != TEAL)
    double_hit_obstacles = {
        int(ev["obstacle_id"])
        for ev in events
        if sum(1 for x in events if int(x["obstacle_id"]) == int(ev["obstacle_id"])) >= 2
    }

    if sum(hit_counts) != target_hits or any(count > 2 for count in hit_counts):
        return None
    if hit_counts[source] != 2:
        return None
    if not any(count == 2 for count in hit_counts):
        return None
    if set(side_color.values()) - {int(ev["input_color"]) for ev in events}:
        return None
    if history_hits != 1:
        return None
    if non_teal_recolors < 1:
        return None
    if len(double_hit_obstacles) != 1:
        return None
    if conflict_cells != 0 or overlap_cells > sz:
        return None
    if np.array_equal(combo, d4_only):
        return None
    if np.array_equal(combo, original):
        return None

    legend = _make_legend(legend_colors, turns)
    inp = np.vstack([legend, base])
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, combo):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate e87109e9 combo input")
