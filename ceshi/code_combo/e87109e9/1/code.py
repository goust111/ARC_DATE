# Combo variant of e87109e9 (index 1)
# combo_spec: D2/3+D4/0
# Name: 交替转弯加颜色继承

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for the combo variant
#
# a) Concrete behavior / trace
#    - The input still has the original two-zone layout: a 6x24 legend strip on
#      top and a 19-24 x 24 main board below.
#    - The main board still contains one solid teal square plus 3-4 large
#      edge-touching obstacle rectangles.
#    - The output still deletes the legend and traces four sz-wide beams from
#      the square, but now each beam carries two pieces of state:
#      collision parity and current paint color.
#    - Example: the north beam starts teal, hits the north obstacle first, turns
#      clockwise because this is collision #1, and every later step on that
#      segment is painted in the north obstacle's color. If it then hits the
#      east obstacle, that second collision turns counterclockwise and the next
#      segment switches again to the east obstacle's color.
#    - Compared with the original task, both the turn rule and the visible trail
#      now depend on collision history rather than a fixed color->turn legend.
#
# b) Complexity calibration
#    - If every beam had at most one hit, the turn rule would look like "always
#      turn clockwise once" and the color rule would reduce to a single recolor.
#      That is too weak.
#    - Generation therefore forces at least one two-hit beam. This makes the
#      alternation visible (first hit CW, second hit CCW) and also yields a
#      teal -> color A -> color B inheritance chain on the output.
#    - A chained target obstacle is hit once as another beam's second collision
#      and once as its own beam's first collision, so no fixed color-specific
#      turn explanation can fit the examples.
#
# c) Hidden-assumption audit
#    - The original solver assumed the legend encoded color->turn. That is now
#      false; the legend remains only as removable decoration and structure.
#    - The original solver also checked the next footprint against the mutable
#      output grid. That breaks once trails can recolor to non-teal obstacle
#      colors, because old trail cells could masquerade as fresh blockers.
#      Obstacle lookup must therefore read from the immutable cropped input.
#    - Cycle detection can no longer use only (r, c, d). Under alternating
#      turns, the same pose with different collision parity has a different
#      future; under color inheritance, the same pose and parity with a
#      different current paint color also produces a different output.
#
# d) Placement legality
#    - Obstacles stay axis-aligned, edge-touching, mutually disjoint, and at
#      least one cell away from the source square.
#    - For a forced two-hit chain src -> cw(src), the target side must cover the
#      redirected lane from src, while non-chain clockwise-adjacent sides must
#      exclude that lane so accidental second hits do not appear.
#    - After the second hit, the beam turns back outward, so the source side
#      excludes the target lane to avoid an unintended third hit.
#
# e) Visual observability
#    - Decorative legend bars still exist, but their left/right positions are
#      randomized independently of the real behavior.
#    - Each generated sample contains at least one obstacle color that appears
#      in the output as an inherited trail color, and at least one two-hit beam
#      so a solver can see both alternation and recoloring from examples alone.
#    - The same obstacle color is observed in both first-hit and second-hit
#      contexts, forcing the conclusion that turn direction depends on hit order
#      rather than obstacle identity.
#
# f) Visual cleanliness
#    - The generator rejects layouts where different beams would paint the same
#      cell in different colors. This prevents order-dependent artifacts.
#    - Large edge-touching obstacle bars and generous square margins preserve
#      the original clean composition, while allowing only small benign overlaps
#      when beams agree on color.
#
# g) Color-rule audit
#    - Colors are not used for numeric ordering or parity.
#    - Obstacle colors only serve two visible roles: obstacle identity in the
#      input and inherited trail color after a collision in the output.

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
MAX_RETRIES = 600


def _legend_colors(grid: np.ndarray, legend_h: int) -> list[int]:
    """Read the four legend colors for structure validation; bar direction is decorative."""
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    colors: list[int] = []
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        interior = legend[1 : legend_h - 1, a + 1 : b]
        cell_colors = set(map(int, np.unique(interior)))
        cell_colors.discard(bg)
        cell_colors.discard(0)
        assert len(cell_colors) == 1, "each legend cell must contain exactly one colored bar"
        colors.append(int(next(iter(cell_colors))))
    assert len(colors) == 4, "legend must encode four colors"
    return colors


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    """Return (r0, c0, size) for the unique solid teal square."""
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


def _simulate_paths(
    base: np.ndarray,
    bg: int,
    *,
    turn_mode: str,
    inherit_color: bool,
    legend_turn: dict[int, str] | None = None,
    enforce_clean: bool,
    max_hits_per_beam: int | None = None,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, int]]], list[dict[tuple[int, int], int]]]:
    """
    Simulate the four square beams on the cropped main grid.

    `turn_mode` is either "alternate" or "legend".
    `inherit_color` toggles the D4 behavior.
    `enforce_clean=True` rejects cross-beam color conflicts.
    """
    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int, int]]] = []
    beam_paints: list[dict[tuple[int, int], int]] = []
    global_paints: dict[tuple[int, int], int] = {}

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def obstacle_at(r: int, c: int) -> int | None:
        blk = base[r : r + sz, c : c + sz]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        hit_parity = 0
        paint_color = TEAL
        beam_hits: list[tuple[int, int, int, int, int]] = []
        beam_cells: dict[tuple[int, int], int] = {}
        seen: set[tuple[int, int, int, int, int]] = set()

        def paint_block(rr: int, cc: int) -> None:
            blk_base = base[rr : rr + sz, cc : cc + sz]
            bg_rows, bg_cols = np.where(blk_base == bg)
            for roff, coff in zip(bg_rows, bg_cols):
                cell = (int(rr + roff), int(cc + coff))
                beam_cells[cell] = int(paint_color if inherit_color else TEAL)
                out[cell] = beam_cells[cell]

        while True:
            state = (
                r,
                c,
                d,
                hit_parity if turn_mode == "alternate" else 0,
                paint_color if inherit_color else TEAL,
            )
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            beam_hits.append((hit_parity + 1, d, obs, nr, nc))
            if turn_mode == "alternate":
                d = _rot_cw(d) if hit_parity == 0 else _rot_ccw(d)
                hit_parity ^= 1
            else:
                assert legend_turn is not None and obs in legend_turn, "legend turn must exist"
                d = _rot(d, legend_turn[obs])
            if inherit_color:
                paint_color = int(obs)
            if max_hits_per_beam is not None:
                assert len(beam_hits) <= max_hits_per_beam, "too many hits for generated sample"

        if enforce_clean:
            for cell, color in beam_cells.items():
                prev = global_paints.get(cell)
                if prev is not None:
                    assert prev == color, "overlapping beams must agree on color"
                global_paints[cell] = color
        else:
            global_paints.update(beam_cells)
        hits.append(beam_hits)
        beam_paints.append(beam_cells)

    return out, hits, beam_paints


def _conflict_stats(beam_paints: list[dict[tuple[int, int], int]]) -> tuple[int, int]:
    """Return (#cells painted by multiple beams, #cells painted in multiple colors)."""
    owners: dict[tuple[int, int], list[int]] = {}
    for paints in beam_paints:
        for cell, color in paints.items():
            owners.setdefault(cell, []).append(int(color))
    overlap_cells = sum(1 for colors in owners.values() if len(colors) > 1)
    conflict_cells = sum(1 for colors in owners.values() if len(set(colors)) > 1)
    return overlap_cells, conflict_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace four square beams that alternate turn direction and inherit obstacle colors.

    Input structure:
    - The top band is still a grey legend strip with four colored bars, but here it is only a removable decorative region rather than an operational lookup table.
    - The rows below form one main board with a single solid teal square, a uniform background, and several colored edge-touching obstacle rectangles.

    Rule:
    - Remove the entire legend band from the output.
    - Launch one sz-wide beam from the teal square in each cardinal direction.
    - Each beam paints only background cells: the initial segment is teal, then after every collision the following segment changes to the collided obstacle's color.
    - Turning ignores legend colors: the 1st, 3rd, 5th, ... collisions of a beam turn clockwise, while the 2nd, 4th, 6th, ... collisions turn counterclockwise.
    - Obstacles themselves are never overwritten, and a beam stops once its next sz-wide step would leave the main board.

    Key insight: Both visible trail color and future direction now depend on per-beam collision history, so the solver must track two evolving states at once: hit parity and inherited paint color.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"
    _legend_colors(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    out, _hits, _beam_paints = _simulate_paths(
        base,
        bg,
        turn_mode="alternate",
        inherit_color=True,
        enforce_clean=False,
        max_hits_per_beam=None,
    )
    return out


def _random_legend_dirs(colors: list[int]) -> dict[int, str]:
    """Assign decorative legend bars without encoding the real rule."""
    while True:
        dirs = {color: random.choice(["L", "R"]) for color in colors}
        if len(set(dirs.values())) == 2:
            return dirs


def _make_legend(legend_colors: list[int], legend_dirs: dict[int, str]) -> np.ndarray:
    """Build the unchanged-looking 6x24 legend band."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if legend_dirs[color] == "L" else start + 3
        legend[1:5, bar_col] = color
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
    """Compute obstacle thicknesses for the two ends of one axis."""
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
    """Pick non-overlapping clockwise second-hit chains src -> cw(src)."""
    candidates = [side for side in present if _rot_cw(side) in present]
    random.shuffle(candidates)
    chains: dict[int, int] = {}
    used: set[int] = set()
    for side in candidates:
        if len(chains) == second_hits:
            break
        tgt = _rot_cw(side)
        if side in used or tgt in used:
            continue
        chains[side] = tgt
        used.update((side, tgt))
    return chains if len(chains) == second_hits else None


def _compute_span_constraints(
    present: set[int],
    chains: dict[int, int],
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Build required/excluded intervals for the alternating-turn geometry."""
    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}

    for side in present:
        base_coord = sc if side in (_N, _S) else sr
        required[side].append((base_coord, base_coord + sz - 1))

    def lane_interval(side: int) -> tuple[int, int]:
        return lane[side], lane[side] + sz - 1

    for src, tgt in chains.items():
        required[tgt].append(lane_interval(src))
        excluded[src].append(lane_interval(tgt))

    for side in present:
        tgt = _rot_cw(side)
        if tgt in present and chains.get(side) != tgt:
            excluded[tgt].append(lane_interval(side))

    return required, excluded


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3
    present = set(random.sample([_N, _E, _S, _W], obstacle_count))
    second_hits = random.choice([1, 2]) if obstacle_count == 4 else 1

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

    required, excluded = _compute_span_constraints(present, chains, sr, sc, sz, lane)

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

    legend_dirs = _random_legend_dirs(legend_colors)

    try:
        combo_out, combo_hits, combo_paints = _simulate_paths(
            base,
            bg,
            turn_mode="alternate",
            inherit_color=True,
            enforce_clean=True,
            max_hits_per_beam=2,
        )
        alt_only_out, _alt_hits, _alt_paints = _simulate_paths(
            base,
            bg,
            turn_mode="alternate",
            inherit_color=False,
            enforce_clean=False,
            max_hits_per_beam=2,
        )
        color_only_out, _color_hits, _color_paints = _simulate_paths(
            base,
            bg,
            turn_mode="legend",
            inherit_color=True,
            legend_turn=legend_dirs,
            enforce_clean=False,
            max_hits_per_beam=2,
        )
        original_out, _orig_hits, _orig_paints = _simulate_paths(
            base,
            bg,
            turn_mode="legend",
            inherit_color=False,
            legend_turn=legend_dirs,
            enforce_clean=False,
            max_hits_per_beam=2,
        )
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in combo_hits]
    total_hits = sum(hit_counts)
    hit_colors = {color for beam in combo_hits for _idx, _d, color, _r, _c in beam}
    visible_colors = {color for paints in combo_paints for color in paints.values() if color != TEAL}
    non_teal_cells = sum(1 for paints in combo_paints for color in paints.values() if color != TEAL)
    overlap_cells, conflict_cells = _conflict_stats(combo_paints)
    first_hit_colors = {color for beam in combo_hits for idx, _d, color, _r, _c in beam if idx == 1}
    second_hit_colors = {color for beam in combo_hits for idx, _d, color, _r, _c in beam if idx == 2}
    dual_role_colors = first_hit_colors & second_hit_colors

    if total_hits != obstacle_count + second_hits:
        return None
    if any(count > 2 for count in hit_counts):
        return None
    if sum(count == 2 for count in hit_counts) != second_hits:
        return None
    if set(side_color.values()) - hit_colors:
        return None
    if len(visible_colors) < 2 or non_teal_cells < sz:
        return None
    if not dual_role_colors:
        return None
    if conflict_cells != 0 or overlap_cells > 2 * sz * sz:
        return None
    if np.array_equal(combo_out, alt_only_out):
        return None
    if np.array_equal(combo_out, color_only_out):
        return None
    if np.array_equal(combo_out, original_out):
        return None

    legend = _make_legend(legend_colors, legend_dirs)
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
