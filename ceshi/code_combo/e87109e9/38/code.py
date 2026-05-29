# Combo variant of e87109e9 (index 38)
# combo_spec: D4/3+D10/1
# Name: 轨迹最小外接矩形加仅转弯点涂色

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, components_mask

# ---------------------------------------------------------------------------
# Design analysis for the fused variant D4/3 + D10/1
#
# a) Concrete behavior and trace walk-through:
#    The input layout stays close to the original task: a 6x24 legend on top and
#    one main playfield below. The legend still maps obstacle colors to left/right
#    turns. The beam motion also stays the same: a sz-by-sz teal square emits one
#    beam in each cardinal direction, moving one cell at a time and turning when
#    the next footprint would hit a monochrome obstacle.
#
#    The fused change is in what gets painted and what portion survives:
#    - Original: every clear step paints a full teal footprint, so long solid
#      corridors are visible; the output is the whole legend-free board.
#    - D4/3 here: straight runs are invisible. When a beam turns at anchor
#      position P, only three local footprints are painted on a separate blank
#      canvas: the footprint one step before P along the incoming direction, the
#      footprint at P itself, and the first clear footprint after the turn.
#    - D10/1 here: after all beams finish, crop that marker-only canvas to the
#      bbox of its teal cells.
#
#    Small anchor example: if a north-going beam has anchors ... A -> B -> C and
#    from C the next north step would hit an obstacle, then the beam turns east.
#    The original task painted A, B, C, and the entire eastward continuation.
#    The fused task paints only B, C, and the first eastward anchor; everything
#    between this turn and the next one remains background. Because the output is
#    taken from a blank canvas, obstacles and the source square do not survive.
#
# b) Complexity calibration:
#    If the generator allowed too few turns or allowed the marker bbox to touch
#    all borders, the variant would visually collapse toward a trivial crop. The
#    generator therefore keeps the original 4-5 total obstacle hits, requires the
#    marker crop to trim both axes, requires multiple marker components, and
#    rejects cases where marker coverage is too close to the full continuous-path
#    coverage.
#
# c) Implicit-assumption audit against the original solver:
#    - Original `paint_block` assumed every clear step is visible; this is no
#      longer true. We must separate "motion state" from "visible state".
#    - Original collision checks could read from the mutable output because all
#      trails were teal and explicitly ignored. Here the output no longer mirrors
#      the solved board at all, so obstacle detection must read from the immutable
#      input playfield.
#    - Original `return out` assumed the main playfield itself is the output.
#      That breaks under the fusion, because the output is now a marker canvas
#      cropped to the marker bbox, not a crop of the obstacle field.
#    - Cropping by all teal cells would accidentally include the source in a
#      marker-free region if we reused the old solved board. Rendering to a blank
#      canvas avoids that conflict and matches the fusion description.
#
# d) Placement legality:
#    The generator preserves the original geometric legality: one solid teal
#    square, 3-4 disjoint edge-touching obstacle rectangles, no overlap with the
#    source, no mixed-color obstacle footprints, and at most two turns per beam.
#    Additional fusion-specific checks ensure that each turn has a valid visible
#    predecessor/successor marker and that the cropped marker field is non-empty.
#
# e) Visual observability:
#    Across examples, the legend still explains turning, but the output now shows
#    only short teal elbows plus blank gaps. That is a strong visible contrast
#    from the original continuous corridors. Requiring both row and column trim
#    makes the non-fixed crop size directly observable from a few examples.
#
# f) Visual cleanliness:
#    The output uses only background plus teal markers. Removing obstacles and the
#    source from the output avoids multi-layer clutter near the turn locations.
#    Generator checks keep several distinct marker islands and reserve blank area
#    inside the crop so the result reads as sparse corner hints rather than a
#    dense filled region.
#
# g) Inference of the changed rule:
#    No hidden ordering on numeric color ids is used. The only color semantics
#    remain the visible legend turns. The new behavior is structural: "where
#    turns happen" determines the only painted cells, and those cells alone define
#    the crop. Because the generator samples varying obstacle layouts, a solver
#    must infer invisible straight motion from the visible elbow markers.
# ---------------------------------------------------------------------------

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


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
        color = int(next(iter(colors)))
        xs = np.where(interior == color)[1]
        x = int(xs[0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
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


def _teal_bbox(g: np.ndarray, teal: int = 8) -> tuple[int, int, int, int]:
    coords = np.argwhere(g == teal)
    assert coords.size, "marker grid must contain teal cells"
    return bbox_coords(coords)


def _crop_to_teal_bbox(g: np.ndarray, teal: int = 8) -> np.ndarray:
    r0, c0, r1, c1 = _teal_bbox(g, teal=teal)
    return g[r0 : r1 + 1, c0 : c1 + 1]


def _simulate_marker_canvas(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, int]]], int]:
    """
    Simulate invisible beams and paint only local turn markers on a blank canvas.

    Returns:
      - full marker canvas (same size as `base`)
      - per-beam hit records: (incoming_dir, obstacle_color, turn_r, turn_c, outgoing_dir)
      - count of turns whose before/turn/after trio was fully valid and visible
    """
    sr, sc, sz = _find_teal_square(base, teal=8)
    canvas = np.full(base.shape, bg, dtype=int)
    h, w = base.shape
    complete_turn_markers = 0
    hits: list[list[tuple[int, int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def obstacle_at(r: int, c: int) -> int | None:
        blk = base[r : r + sz, c : c + sz]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(8)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    def paint_block(r: int, c: int) -> None:
        canvas[r : r + sz, c : c + sz] = 8

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        beam_hits: list[tuple[int, int, int, int, int]] = []
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            obs = obstacle_at(nr, nc)
            if obs is None:
                r, c = nr, nc
                continue

            assert obs in turns, "obstacle color must be encoded by legend"
            nd = _rot(d, turns[obs])
            beam_hits.append((d, obs, r, c, nd))

            prev_r, prev_c = r - dr, c - dc
            next_dr, next_dc = _DIRS[nd]
            next_r, next_c = r + next_dr, c + next_dc
            prev_ok = in_bounds(prev_r, prev_c)
            next_ok = in_bounds(next_r, next_c) and obstacle_at(next_r, next_c) is None

            if prev_ok:
                paint_block(prev_r, prev_c)
            paint_block(r, c)
            if next_ok:
                paint_block(next_r, next_c)
            if prev_ok and next_ok:
                complete_turn_markers += 1

            d = nd
            assert len(beam_hits) <= 2, "generator only supports up to two hits per beam"
        hits.append(beam_hits)

    return canvas, hits, complete_turn_markers


def _simulate_full_path_canvas(
    base: np.ndarray, turns: dict[int, str], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Reference simulation for the original continuous teal corridors on a blank canvas."""
    sr, sc, sz = _find_teal_square(base, teal=8)
    canvas = np.full(base.shape, bg, dtype=int)
    h, w = base.shape
    hits: list[list[tuple[int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def obstacle_at(r: int, c: int) -> int | None:
        blk = base[r : r + sz, c : c + sz]
        colors = set(map(int, np.unique(blk)))
        colors.discard(bg)
        colors.discard(8)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    def paint_block(r: int, c: int) -> None:
        canvas[r : r + sz, c : c + sz] = 8

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        beam_hits: list[tuple[int, int, int, int]] = []
        while True:
            state = (r, c, d)
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

            assert obs in turns, "obstacle color must be encoded by legend"
            beam_hits.append((d, obs, nr, nc))
            d = _rot(d, turns[obs])
            assert len(beam_hits) <= 2, "generator only supports up to two hits per beam"
        hits.append(beam_hits)

    return canvas, hits


def _marker_metrics(
    marker_canvas: np.ndarray,
    full_path_canvas: np.ndarray,
    hits: list[list[tuple[int, int, int, int, int]]],
    complete_turn_markers: int,
    teal: int = 8,
) -> dict[str, float]:
    """Summarize fusion-specific observability metrics for generator filtering and analysis."""
    r0, c0, r1, c1 = _teal_bbox(marker_canvas, teal=teal)
    h, w = marker_canvas.shape
    crop = marker_canvas[r0 : r1 + 1, c0 : c1 + 1]
    marker_cells = int(np.sum(marker_canvas == teal))
    full_path_cells = int(np.sum(full_path_canvas == teal))
    total_hits = int(sum(len(beam) for beam in hits))
    comp_count = len(list(components_mask(marker_canvas == teal, connectivity=4)))
    blank_in_crop = int(crop.size - np.sum(crop == teal))
    return {
        "top_trim": float(r0),
        "left_trim": float(c0),
        "bottom_trim": float(h - 1 - r1),
        "right_trim": float(w - 1 - c1),
        "crop_h": float(crop.shape[0]),
        "crop_w": float(crop.shape[1]),
        "marker_cells": float(marker_cells),
        "full_path_cells": float(full_path_cells),
        "marker_ratio": float(marker_cells / max(1, full_path_cells)),
        "total_hits": float(total_hits),
        "components": float(comp_count),
        "blank_in_crop": float(blank_in_crop),
        "complete_turn_markers": float(complete_turn_markers),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the original turning beams, but only reveal short teal markers around turns.

    Input structure:
    - Top band: a 6-row legend on color-5 background with four colored vertical bars; left/right placement still encodes counterclockwise or clockwise turning for that color.
    - Below the legend: one main playfield containing a single solid teal square source, one uniform background color, and 3-4 solid obstacle rectangles whose colors all appear in the legend.

    Rule:
    - Ignore the legend region when tracing motion, but keep using it as the color-to-turn map.
    - Launch one sz-by-sz beam from the teal square in each cardinal direction. Straight motion is still simulated exactly as in the original task, but it is invisible.
    - Whenever the next step would hit an obstacle, turn in place according to that obstacle's legend entry. Paint teal only at the turn location, at the previous anchor immediately before the turn, and at the first clear anchor immediately after the turn.
    - Draw those markers onto a blank background canvas, then crop the result to the smallest rectangle containing all teal marker cells.

    Key insight: The legend still explains how beams bend, but the output no longer shows the corridors themselves. Only elbow neighborhoods survive, and those sparse markers alone determine the final crop.
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
    _find_teal_square(base, teal=8)

    marker_canvas, _hits, _complete_turn_markers = _simulate_marker_canvas(base, legend_turn, bg)
    return _crop_to_teal_bbox(marker_canvas, teal=8)


# --------------- generator (o5x, fused D4/3 + D10/1) ---------------

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

MIN_TOTAL_TRIM = 6
MIN_BLANK_IN_CROP = 10
MIN_MARKER_COMPONENTS = 2
MAX_MARKER_RATIO = 0.55

MAX_RETRIES = 900


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals while avoiding all excluded intervals."""
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


def _select_chains(
    present: set[int], second_hits: int
) -> dict[int, int] | None:
    """Pick non-overlapping (source -> adjacent target) pairs for multi-bounce beams."""
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
    """Build required/excluded intervals for each obstacle's span selection."""
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


def _is_visible_fused_case(metrics: dict[str, float]) -> bool:
    """Reject cases whose marker crop is too small, too dense, or too close to the original trace."""
    trimmed_rows = metrics["top_trim"] > 0 or metrics["bottom_trim"] > 0
    trimmed_cols = metrics["left_trim"] > 0 or metrics["right_trim"] > 0
    total_trim = metrics["top_trim"] + metrics["left_trim"] + metrics["bottom_trim"] + metrics["right_trim"]
    return (
        trimmed_rows
        and trimmed_cols
        and total_trim >= MIN_TOTAL_TRIM
        and metrics["components"] >= MIN_MARKER_COMPONENTS
        and metrics["blank_in_crop"] >= MIN_BLANK_IN_CROP
        and metrics["marker_ratio"] <= MAX_MARKER_RATIO
        and metrics["complete_turn_markers"] == metrics["total_hits"]
    )


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

    vals, counts = np.unique(base, return_counts=True)
    if int(vals[counts.argmax()]) != bg:
        return None

    turns = {side_color[s]: side_turn[s] for s in present}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        marker_canvas, hits, complete_turn_markers = _simulate_marker_canvas(base, turns, bg)
        full_path_canvas, full_path_hits = _simulate_full_path_canvas(base, turns, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _, _ in beam}
    if full_path_hits and sum(len(beam) for beam in full_path_hits) != sum(hit_counts):
        return None
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None

    metrics = _marker_metrics(marker_canvas, full_path_canvas, hits, complete_turn_markers, teal=TEAL)
    if not _is_visible_fused_case(metrics):
        return None

    expected = _crop_to_teal_bbox(marker_canvas, teal=TEAL)
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
