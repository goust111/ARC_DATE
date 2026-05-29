# Variant of e87109e9
# Dimension: D18 — 源方块颜色的语义 (src)
# Change: 源方块固定为青色(8)，颜色仅作为标识不编码额外信息，射线也涂青色... → 多色源方块（内部图案）

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))


# Design analysis for the D18 variant
# a) Concretization + behavior trace
#    - Legend stays exactly the same: a 6x24 top band with four colored bars, where
#      each bar's left/right placement still means CCW/CW for one obstacle color.
#    - The source is no longer a solid square. It is always one isolated 2x2 block
#      whose four cells are all different. Their positions encode beam colors:
#        NW -> upward beam, NE -> rightward beam, SE -> downward beam, SW -> leftward beam.
#    - The beam geometry stays close to the original task: each beam still moves as a
#      2x2 sliding footprint that advances one cell at a time and turns only when the
#      next footprint hits an obstacle color listed in the legend.
#    - Concrete trace:
#        Original: the square is all 8, so the upward and rightward beams both paint 8.
#        Variant: if the 2x2 source is [[2, 7], [9, 4]], then up paints 2, right paints 7,
#        down paints 4, left paints 9. Turning behavior is unchanged; only the carried
#        paint color is different.
# b) Complexity calibration
#    - The visible change must be obvious in 2-3 examples, so the source uses four
#      distinct colors rather than a subtle recoloring rule.
#    - To keep the task solvable, obstacle placement, turn logic, obstacle count, and
#      total number of bounces remain near the original distribution.
#    - Source colors, obstacle colors, and background are made pairwise disjoint inside
#      each sample. This prevents accidental ambiguity where a beam color could be
#      mistaken for an obstacle or the background.
# c) Hidden assumptions audited against the old solver
#    - Old `_find_teal_square` assumed one monochrome component of color 8. That is no
#      longer valid; the new solver must instead find one 2x2 block of four singleton colors.
#    - Old `paint_block` always wrote color 8 onto background. That would erase the new
#      per-direction semantics, so the beam color is now carried in beam state.
#    - Old `obstacle_color_at_next` ignored only `bg` and teal. In this variant, the
#      moving footprint overlaps source-colored cells near the launch point, so all four
#      source colors must be treated as passable beam colors, not as obstacles.
# d) Placement legality
#    - The generator still places 3-4 edge-touching obstacle rectangles with at least
#      one-cell clearance from the source footprint.
#    - The source remains far from the border, so all four beams have a visible first step.
#    - A dedicated oracle rejects any sample where different beams would overlap, because
#      multiple beam colors in the same corridor would make the output visually messy.
# e) Visual observability
#    - Position, not color id, determines which beam gets which color. The 2x2 pattern
#      is therefore a visible lookup table that a human can align with the four emitted beams.
#    - Colors vary between instances; the rule cannot be reduced to a fixed palette hack.
# f) Visual cleanliness
#    - Only one multicolor object exists in the main grid, obstacles stay monochrome, and
#      the oracle enforces disjoint beam trails. This keeps the output easy to read.
# g) Color inference
#    - The mapping is structural and local: each beam color is read directly from one corner
#      of the source block. No color sorting, arithmetic, or hidden ordering is used.
# New solver assumptions that the generator guarantees
#    - Exactly one 2x2 block has four distinct colors that each occur once in the main grid.
#    - Obstacle colors are disjoint from the four source/beam colors.
#    - Different beams never share painted cells.


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)
_SOURCE_SZ = 2
_ALL_MAIN_COLORS = [0, 1, 2, 3, 4, 6, 7, 8, 9]
_LEGEND_COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]


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


def _find_source_block(g: np.ndarray) -> tuple[int, int, dict[int, int], set[int]]:
    """Return top-left coord plus direction->beam-color map for the unique 2x2 source."""
    vals, cnts = np.unique(g, return_counts=True)
    counts = {int(v): int(c) for v, c in zip(vals, cnts, strict=False)}

    candidates: list[tuple[int, int, dict[int, int], set[int]]] = []
    H, W = g.shape
    for r in range(H - 1):
        for c in range(W - 1):
            blk = g[r : r + 2, c : c + 2]
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project four independently colored beams from a 2x2 source, turning by legend rules.

    Input structure:
    - Top band: a grey legend with four colored bars; each bar's left/right placement encodes CCW/CW for one obstacle color.
    - Below: a main grid containing one isolated 2x2 multicolor source block and several monochrome obstacle rectangles.

    Rule:
    - Remove the legend from the output.
    - Read the source corners clockwise as a direction-to-color key: NW launches the upward beam, NE the rightward beam, SE the downward beam, SW the leftward beam.
    - Each beam keeps the original 2x2 sliding footprint and paints only background cells in its own fixed color; when the next footprint hits an obstacle color, turn according to the legend and continue.
    - Beams stop when the next 2x2 footprint would leave the main grid.

    Key insight:
    - The legend still controls turning, but the source block now supplies the four beam colors through its corner positions, so color semantics come from geometry inside the source rather than a single global beam color.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_turn = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    sr, sc, beam_colors, beam_palette = _find_source_block(base)
    assert not (beam_palette & set(legend_turn)), "source colors and obstacle colors must be disjoint"

    out = base.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + _SOURCE_SZ - 1 < out.shape[0] and c + _SOURCE_SZ - 1 < out.shape[1]

    def paint_block(r: int, c: int, color: int) -> None:
        blk = out[r : r + _SOURCE_SZ, c : c + _SOURCE_SZ]
        blk[blk == bg] = int(color)
        out[r : r + _SOURCE_SZ, c : c + _SOURCE_SZ] = blk

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = out[nr : nr + _SOURCE_SZ, nc : nc + _SOURCE_SZ]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols -= beam_palette
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in range(4):
        r, c, d = sr, sc, d0
        beam_color = beam_colors[d0]
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c, beam_color)
                continue

            assert oc in legend_turn, "obstacle color must be encoded by legend"
            turn = legend_turn[oc]
            d = _rot_ccw(d) if turn == "L" else _rot_cw(d)

    return out


# --------------- generator ---------------

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
MAX_RETRIES = 300


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
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


def _simulate_paths(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]], list[set[tuple[int, int]]]]:
    """Oracle for the variant: beams are colored by the 2x2 source corners."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, beam_colors, beam_palette = _find_source_block(base)
    assert not (beam_palette & set(turns))

    out = base.copy()
    H, W = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []
    per_beam_cells: list[set[tuple[int, int]]] = []
    owner = np.full((H, W), -1, dtype=int)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + _SOURCE_SZ - 1 < H and c + _SOURCE_SZ - 1 < W

    def paint(r: int, c: int, color: int, beam_id: int) -> set[tuple[int, int]]:
        blk = out[r : r + _SOURCE_SZ, c : c + _SOURCE_SZ]
        own = owner[r : r + _SOURCE_SZ, c : c + _SOURCE_SZ]
        assert not np.any((own >= 0) & (own != beam_id)), "different beams must stay disjoint"
        mask = blk == bg
        blk[mask] = int(color)
        own[mask] = int(beam_id)
        painted = {
            (int(r + rr), int(c + cc))
            for rr, cc in zip(*np.where(mask), strict=False)
        }
        return painted

    def obstacle_at(r: int, c: int) -> int | None:
        colors = set(map(int, np.unique(out[r : r + _SOURCE_SZ, c : c + _SOURCE_SZ])))
        colors.discard(bg)
        colors -= beam_palette
        if not colors:
            return None
        assert len(colors) == 1
        return int(next(iter(colors)))

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        beam_hits: list[tuple[int, int, int, int]] = []
        beam_cells: set[tuple[int, int]] = set()
        seen: set[tuple[int, int, int]] = set()
        beam_color = beam_colors[d0]
        while True:
            state = (r, c, d)
            assert state not in seen
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                beam_cells |= paint(r, c, beam_color, d0)
                continue

            assert obs in turns
            beam_hits.append((d, obs, nr, nc))
            d = _rot(d, turns[obs])
            assert len(beam_hits) <= 2

        hits.append(beam_hits)
        per_beam_cells.append(beam_cells)

    for i in range(len(per_beam_cells)):
        for j in range(i + 1, len(per_beam_cells)):
            assert not (per_beam_cells[i] & per_beam_cells[j]), "beam trails must be disjoint"

    return out, hits, per_beam_cells


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
    present: set[int], side_turn: dict[int, str], chains: dict[int, int],
    sr: int, sc: int, sz: int, lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}

    for side in present:
        base_coord = sc if side in (_N, _S) else sr
        required[side].append((base_coord, base_coord + sz - 1))

    def lane_interval(s: int) -> tuple[int, int]:
        return lane[s], lane[s] + sz - 1

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

    remaining_bg = [c for c in _ALL_MAIN_COLORS if c not in source_colors and c not in legend_colors]
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
    thickness = {s: t for s, t in zip((_N, _S, _W, _E), (*ns, *ew), strict=False) if t is not None}

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

    lane = {}
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
        present, side_turn, chains, sr, sc, sz, lane,
    )

    spans: dict[int, tuple[int, int]] = {}
    for side in present:
        axis_len = w if side in (_N, _S) else h
        span = _choose_span(
            axis_len, required[side], excluded[side],
            SPAN_MIN_MULT * sz, min(SPAN_MAX_MULT * sz, axis_len),
        )
        if span is None:
            return None
        spans[side] = span

    beam_colors, source_block, legend_colors, bg = _sample_color_roles()
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
        base[r0 : r1 + 1, c0 : c1 + 1] = side_color[side]

    vals, counts = np.unique(base, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    turns = {side_color[s]: side_turn[s] for s in present}
    for color in legend_colors:
        turns.setdefault(int(color), random.choice(["L", "R"]))

    try:
        expected, hits, per_beam_cells = _simulate_paths(base, turns, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _ in beam}
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None
    if len({int(source_block[0, 0]), int(source_block[0, 1]), int(source_block[1, 0]), int(source_block[1, 1])}) != 4:
        return None
    if any(not cells for cells in per_beam_cells):
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
    raise RuntimeError("failed to generate e87109e9 D18 input")
