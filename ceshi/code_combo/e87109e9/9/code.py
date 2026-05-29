# Combo variant of e87109e9 (index 9)
# combo_spec: D1/0+D6/2
# Name: 双向射线加终止颜色

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, components_mask

# Design analysis for the combo variant D1/0 + D6/2
# 1. Concrete fused object change and behavior trace:
#    The top legend keeps the original 6x24 structure, but one cell uses the D6
#    double-bar encoding to mark a terminal obstacle color. The source block no
#    longer stays a solid teal square: it is a 3x3 teal square with exactly one
#    orange marker pixel (7) on the midpoint of one side. Marker on left/right
#    means only the horizontal pair of beams exists; marker on top/bottom means
#    only the vertical pair exists. A concrete horizontal example:
#      - input source marker at left-middle -> active starts are west and east
#      - west beam hits the stop-colored west obstacle and halts immediately
#      - east beam hits the east obstacle, turns to the north or south obstacle,
#        then turns again into the stop obstacle and halts there
#    In the original task, north and south beams would also launch, and every hit
#    color would bend the beam. In this combo, those orthogonal start beams do not
#    exist at all, and the stop color ends the path instead of rotating it.
# 2. Complexity calibration:
#    Reducing from four starts to two could make the puzzle too weak if the missing
#    directions never mattered, so the generator forces a non-degenerate pattern:
#    one active beam is a direct stop, the other follows a three-hit chain
#    turn -> turn -> stop. This keeps both D1 and D6 visually necessary in every
#    sample. The source size is fixed to 3x3 so the single marker can sit at an
#    unambiguous side midpoint; that is simpler than inventing a brittle 2x2 code.
# 3. Old implicit assumptions that break in main():
#    The original source parser assumed exactly one solid teal square. That is now
#    false because the source contains one non-teal marker cell. The original beam
#    collision test also discarded only bg and teal, relying on the source being
#    monochrome; with a sliding sz-by-sz footprint, the first move overlaps most of
#    the source block, so the marker must be ignored during collision detection or
#    every beam would falsely "hit" the source marker. Finally, the original loop
#    iterated over all four directions and assumed every obstacle color turns; the
#    combo must instead derive exactly two start directions from the marker and
#    branch between turn colors versus one terminal color.
# 4. Placement legality and generator guarantees:
#    Obstacles remain edge-touching solid rectangles with thickness 2-4 and no
#    overlap with each other or the source block. The generator picks three sides:
#      - stop_side: active side hit directly and again as the terminal third hit
#      - turn_side: opposite active side that sends the beam toward a middle side
#      - mid_side: orthogonal side that turns the beam into stop_side
#    Required span intervals guarantee that the three intended hits occur, while
#    occupied-mask checks reject any corner overlap between rectangles.
# 5. Visual observability:
#    Humans can see both control variables directly. The source block visibly has a
#    single off-color notch on one side, so examples can support "left/right notch
#    -> horizontal beams" and "top/bottom notch -> vertical beams". Separately, one
#    legend cell has the double-bar stop encoding, and in the output the beam that
#    meets that obstacle ends flush against it rather than bending around it.
# 6. Visual cleanliness:
#    The only new main-grid element is one marker pixel embedded in the source. The
#    rest of the scene stays sparse: three large edge rectangles and at most two teal
#    trails. Because the marker color is reserved and never reused as bg/obstacle,
#    the source remains easy to locate.
# 7. Why the fused behavior is inferable from 2-4 examples:
#    The legend still contains only four cells and one special stop color, so color
#    roles stay low-cardinality. The axis marker role is structural rather than
#    color-dependent: its side position, not its numeric value, determines which two
#    beams exist. Across examples, both horizontal and vertical markers can appear,
#    and the corresponding missing pair of beams changes the output in a way that a
#    four-beam explanation cannot match.
# 8. Quantitative non-degeneracy target used by the generator:
#    Every accepted sample must satisfy two checks:
#      - the actual combo output differs from the same grid simulated with four
#        starting beams plus the same stop color, proving D1 is active
#      - both active beams end on the stop color (one immediate, one after turns),
#        proving D6 is active
# 9. New solver assumptions audited against the generator:
#    main() assumes exactly one 3x3 source component made from {8,7}, exactly one
#    side-midpoint marker, exactly one double-bar legend cell, and obstacle
#    footprints that never mix multiple obstacle colors. generate_input() constructs
#    exactly that object vocabulary and re-validates each sample by running main().

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)

# Color palettes
LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 9]
LEGEND_BG = 5
MARKER = 7
TEAL = 8

# Canvas dimensions
GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

# Source block
SOURCE_SZ = 3
SQUARE_EDGE_GAP = 5

# Obstacles
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

# Span selection
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

# Retry budget
MAX_RETRIES = 600


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _parse_legend(grid: np.ndarray, legend_h: int) -> tuple[dict[int, str], int]:
    """
    Single left/right bars encode turn direction. One double-sided bar cell marks
    the terminal obstacle color instead of a turn.
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, counts = np.unique(legend, return_counts=True)
    bg = int(vals[counts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    turn: dict[int, str] = {}
    stop_color: int | None = None
    seen_colors: set[int] = set()
    cell_count = 0
    for a, b in zip(sep_cols, sep_cols[1:], strict=False):
        if b - a <= 2:
            continue
        cell_count += 1
        interior = legend[1 : legend_h - 1, a + 1 : b]
        colors = set(map(int, np.unique(interior)))
        colors.discard(bg)
        colors.discard(0)
        assert len(colors) == 1, "each legend cell must encode exactly one color"
        col = int(next(iter(colors)))
        assert col not in seen_colors, "legend colors must be distinct"
        seen_colors.add(col)

        xs = sorted(set(map(int, np.where(interior == col)[1])))
        if len(xs) == 1:
            turn[col] = "L" if xs[0] <= (interior.shape[1] - 1) // 2 else "R"
        else:
            assert stop_color is None, "legend must encode exactly one stop color"
            assert xs == [0, interior.shape[1] - 1], "stop marker must occupy both sides"
            stop_color = col

    assert cell_count == 4, "legend must have four cells"
    assert stop_color is not None, "legend must encode one stop color"
    assert len(turn) == 3, "legend must encode three turn colors"
    return turn, int(stop_color)


def _find_source_block(g: np.ndarray) -> tuple[int, int, int, str]:
    """Return (r0, c0, size, axis) for the 3x3 teal source with one side marker."""
    candidates: list[tuple[int, int, int, str]] = []
    mask = (g == TEAL) | (g == MARKER)
    for cells in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww or hh != SOURCE_SZ:
            continue
        block = g[r0 : r1 + 1, c0 : c1 + 1]
        colors = set(map(int, np.unique(block)))
        if not colors <= {TEAL, MARKER}:
            continue
        if int(np.sum((block == TEAL) | (block == MARKER))) != hh * ww:
            continue
        if int(np.sum(block == MARKER)) != 1:
            continue
        mid = hh // 2
        mr, mc = map(int, np.argwhere(block == MARKER)[0])
        if (mr, mc) in ((mid, 0), (mid, hh - 1)):
            axis = "H"
        elif (mr, mc) in ((0, mid), (hh - 1, mid)):
            axis = "V"
        else:
            continue
        candidates.append((int(r0), int(c0), int(hh), axis))
    assert len(candidates) == 1, "main grid must contain exactly one marked 3x3 source block"
    return candidates[0]


def _active_dirs(axis: str) -> tuple[int, int]:
    return (_W, _E) if axis == "H" else (_N, _S)


def _obstacle_color_at_next(out: np.ndarray, nr: int, nc: int, sz: int, bg: int) -> int | None:
    blk = out[nr : nr + sz, nc : nc + sz]
    cols = set(map(int, np.unique(blk)))
    cols.discard(bg)
    cols.discard(TEAL)
    cols.discard(MARKER)
    if not cols:
        return None
    assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
    return int(next(iter(cols)))


def _simulate_paths(
    base: np.ndarray,
    turns: dict[int, str],
    stop_color: int,
    start_dirs: tuple[int, ...],
    bg: int,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, bool]]]]:
    """Simulate the active beams, recording obstacle hits and terminal hits."""
    sr, sc, sz, _axis = _find_source_block(base)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int, bool]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL

    for d0 in start_dirs:
        r, c, d = int(sr), int(sc), int(d0)
        beam_hits: list[tuple[int, int, int, int, bool]] = []
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            obs = _obstacle_color_at_next(out, nr, nc, sz, bg)
            if obs is None:
                r, c = nr, nc
                paint(r, c)
                continue

            is_stop = obs == stop_color
            beam_hits.append((d, obs, nr, nc, is_stop))
            if is_stop:
                break

            assert obs in turns, "obstacle color must be encoded by the turning legend"
            d = _rot(d, turns[obs])
        hits.append(beam_hits)

    return out, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit only two teal beams from a marked source square; one legend color now ends
    a beam instead of turning it.

    Input structure:
    - Top band: grey (5) legend with four 4-column cells. Three cells use a single
      left/right bar to encode CCW/CW turning, while one cell uses matching outer
      bars to denote the terminal obstacle color.
    - Below: main grid with one background color, three edge-touching obstacle
      rectangles, and a 3x3 teal source block containing one orange marker pixel on
      the midpoint of one side.

    Rule:
    - Remove the legend from the output.
    - Read the source marker: left/right marker means launch only west+east beams;
      top/bottom marker means launch only north+south beams.
    - Each beam is a sliding 3x3 copy that paints teal onto background cells only.
    - On meeting a normal obstacle color, rotate according to that color's legend
      cell and continue.
    - On meeting the legend's terminal color, stop immediately without turning.

    Key insight: The puzzle now asks for two separate classifications before tracing:
    first infer which axis is active from the source marker, then split obstacle
    colors into "turn" versus one "stop" color using the legend's single-bar versus
    double-bar encodings.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = LEGEND_BG
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    legend_turn, stop_color = _parse_legend(g, legend_h)

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
    sr, sc, sz, axis = _find_source_block(base)
    start_dirs = _active_dirs(axis)

    out = base.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = TEAL
        out[r : r + sz, c : c + sz] = blk

    for d0 in start_dirs:
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

            oc = _obstacle_color_at_next(out, nr, nc, sz, bg)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            if oc == stop_color:
                break

            assert oc in legend_turn, "obstacle color must be encoded by the turning legend"
            d = _rot(d, legend_turn[oc])

    return out


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals while avoiding all excluded ones."""
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


def _make_legend(legend_colors: list[int], turns: dict[int, str], stop_color: int) -> np.ndarray:
    """Build the 6x24 legend; the stop color is drawn as two outer bars."""
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        if color == stop_color:
            legend[1:5, start] = color
            legend[1:5, start + 3] = color
        else:
            bar_col = start if turns[color] == "L" else start + 3
            legend[1:5, bar_col] = color
    return legend


def _rand_thickness(lo: int, hi: int) -> int | None:
    return random.randint(lo, hi) if hi >= lo else None


def _thickness_pair(
    axis: int, sz: int, a_present: bool, b_present: bool
) -> tuple[int | None, int | None] | None:
    """Compute obstacle thicknesses for the two ends of an axis."""
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


def _choose_chain(axis: str) -> tuple[tuple[int, int], int, int, int, dict[int, str]]:
    """Choose start directions plus stop/turn/mid obstacle sides and turn directions."""
    start_dirs = _active_dirs(axis)
    orthogonal = (_N, _S) if axis == "H" else (_W, _E)
    stop_side = random.choice(start_dirs)
    turn_side = start_dirs[0] if stop_side == start_dirs[1] else start_dirs[1]
    mid_side = random.choice(orthogonal)
    side_turn = {
        turn_side: "L" if _rot(turn_side, "L") == mid_side else "R",
        mid_side: "L" if _rot(mid_side, "L") == stop_side else "R",
    }
    return start_dirs, stop_side, turn_side, mid_side, side_turn


def _required_spans(
    present: set[int],
    stop_side: int,
    turn_side: int,
    mid_side: int,
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    """Required intervals guarantee stop, turn, second turn, then terminal hit."""
    required: dict[int, list[tuple[int, int]]] = {s: [] for s in present}
    excluded: dict[int, list[tuple[int, int]]] = {s: [] for s in present}

    def base_interval(side: int) -> tuple[int, int]:
        coord = sc if side in (_N, _S) else sr
        return coord, coord + sz - 1

    def lane_interval(side: int) -> tuple[int, int]:
        return lane[side], lane[side] + sz - 1

    required[stop_side].append(base_interval(stop_side))
    required[turn_side].append(base_interval(turn_side))
    required[mid_side].append(lane_interval(turn_side))
    required[stop_side].append(lane_interval(mid_side))
    return required, excluded


def _make_source_block(axis: str) -> tuple[np.ndarray, tuple[int, int]]:
    """Create the 3x3 source block and return the marker's local coordinate."""
    block = np.full((SOURCE_SZ, SOURCE_SZ), TEAL, dtype=int)
    mid = SOURCE_SZ // 2
    if axis == "H":
        marker = random.choice(((mid, 0), (mid, SOURCE_SZ - 1)))
    else:
        marker = random.choice(((0, mid), (SOURCE_SZ - 1, mid)))
    block[marker] = MARKER
    return block, marker


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = SOURCE_SZ
    axis = random.choice(["H", "V"])
    start_dirs, stop_side, turn_side, mid_side, side_turn = _choose_chain(axis)
    present = {stop_side, turn_side, mid_side}

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

    required, excluded = _required_spans(present, stop_side, turn_side, mid_side, sr, sc, sz, lane)
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
    obstacle_colors = random.sample(legend_colors, len(present))
    side_order = [stop_side, turn_side, mid_side]
    side_color = dict(zip(side_order, obstacle_colors))
    stop_color = side_color[stop_side]
    turns = {
        side_color[turn_side]: side_turn[turn_side],
        side_color[mid_side]: side_turn[mid_side],
    }
    for color in legend_colors:
        if color != stop_color:
            turns.setdefault(color, random.choice(["L", "R"]))

    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    if not bg_choices:
        return None
    bg = random.choice(bg_choices)

    base = np.full((h, w), bg, dtype=int)
    source_block, _marker = _make_source_block(axis)
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

    try:
        expected, hits = _simulate_paths(base, turns, stop_color, start_dirs, bg)
        four_dir_expected, _four_hits = _simulate_paths(base, turns, stop_color, (_N, _E, _S, _W), bg)
    except AssertionError:
        return None

    if np.array_equal(expected, four_dir_expected):
        return None

    hit_counts = [len(beam) for beam in hits]
    all_hits = [hit for beam in hits for hit in beam]
    stop_hits = sum(1 for *_, is_stop in all_hits if is_stop)
    terminal_beams = sum(1 for beam in hits if beam and beam[-1][-1])
    hit_colors = {color for _d, color, _r, _c, _stop in all_hits}
    if len(hits) != 2:
        return None
    if sorted(hit_counts) != [1, 3]:
        return None
    if stop_hits != 2 or terminal_beams != 2:
        return None
    if hit_colors != set(side_color.values()):
        return None

    legend = _make_legend(legend_colors, turns, stop_color)
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
