# Combo variant of e87109e9 (index 39)
# combo_spec: D1/0+D18/0
# Name: 双向射线加源色编码方向

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original: the fixed cyan 2x2/3x3 source always launches 4 beams.
#    New rule: the source color varies in {6,7,8,9}; even source colors launch only
#    left/right beams, odd source colors launch only up/down beams. Example:
#    a 6-colored square hits W and E obstacles only, then may turn into N/S via the
#    legend. Under the old 4-direction rule, extra vertical beams would also be
#    painted, so the output is measurably different.
# b) Complexity calibration:
#    There are only two axis classes, which keeps the new inference compact, while
#    still preserving 3-4 obstacle rectangles and the original legend semantics.
#    Source colors are limited to four visible colors so repeated training examples
#    can support the even/odd grouping without adding many free variables.
# c) Assumption audit:
#    The original solver assumed source color == 8 and iterated over 4 start dirs.
#    That is no longer valid. The new solver must:
#    - locate the unique solid square whose color is not a legend/obstacle color,
#    - paint trails using that source color,
#    - derive the active axis from source_color % 2,
#    - ignore inactive directions completely.
#    The generator guarantees those assumptions by reserving legend colors {1,2,3,4},
#    choosing source colors from {6,7,8,9}, and keeping the source distinct from bg.
# d) Placement legality:
#    Path construction is still output-first/path-first. Active-side obstacles are
#    always present; optional perpendicular obstacles are added only when a beam is
#    chosen to make a second hit. Span constraints force required hits and explicitly
#    exclude third-hit lanes on the opposite active edge.
# e) Visual observability:
#    The new behavior is visible in two ways: only one axis is painted from the source,
#    and the source color itself changes across samples. We also reject any sample
#    where the old 4-direction simulation matches the fused output, preventing silent
#    regression to the original task.
# f) Visual cleanliness:
#    Obstacles stay as single-color edge-touching rectangles, the legend keeps its
#    original layout, and the source color never overlaps legend colors, so role
#    separation remains clear.
# g) Color-rule inferability:
#    The rule depends on source color class only, not on hidden ordering of many
#    colors. Restricting the source palette to four values keeps the mapping learnable
#    from a few examples while still implementing the requested parity-based behavior.


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
    """Read obstacle-color -> turn-direction mapping from the top legend band."""
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
        x = int(np.where(interior == color)[1][0])
        turn[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four turn colors"
    return turn


def _find_source_square(
    grid: np.ndarray, obstacle_colors: set[int], bg: int
) -> tuple[int, int, int, int]:
    """Return (r0, c0, size, color) for the unique solid non-legend square source."""
    candidates: list[tuple[int, int, int, int]] = []
    forbidden = set(obstacle_colors) | {bg, 0, 5}
    for color in map(int, np.unique(grid)):
        if color in forbidden:
            continue
        for cells in components_mask(grid == color, connectivity=4):
            r0, c0, r1, c1 = bbox_coords(cells)
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            if hh != ww:
                continue
            if int(np.sum(grid[r0 : r1 + 1, c0 : c1 + 1] == color)) != hh * ww:
                continue
            candidates.append((int(r0), int(c0), int(hh), int(color)))
    assert len(candidates) == 1, "main grid must contain exactly one source square"
    return candidates[0]


def _rot(d: int, turn: str) -> int:
    return (d - 1) % 4 if turn == "L" else (d + 1) % 4


def _opposite(d: int) -> int:
    return (d + 2) % 4


def _turn_to(src: int, dst: int) -> str:
    if dst == (src - 1) % 4:
        return "L"
    if dst == (src + 1) % 4:
        return "R"
    raise AssertionError("turn_to expects an adjacent side")


def _active_dirs_from_color(color: int) -> tuple[int, int]:
    return (_W, _E) if color % 2 == 0 else (_N, _S)


def _simulate_paths(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    src_color: int,
    dirs: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    """Simulate the fused beam rule and return the painted grid plus beam hit logs."""
    legend_like_colors = set(turns)
    sr, sc, sz, found_color = _find_source_square(base, legend_like_colors, bg)
    assert found_color == src_color

    out = base.copy()
    H, W = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []
    start_dirs = dirs if dirs is not None else _active_dirs_from_color(src_color)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < H and c + sz - 1 < W

    def paint(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = src_color
        out[r : r + sz, c : c + sz] = blk

    def obstacle_at(r: int, c: int) -> int | None:
        colors = set(map(int, np.unique(out[r : r + sz, c : c + sz])))
        colors.discard(bg)
        colors.discard(src_color)
        if not colors:
            return None
        assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(colors)))

    for d0 in start_dirs:
        r, c, d = sr, sc, d0
        beam_hits: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
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
                paint(r, c)
                continue

            assert obs in turns, "obstacle color must be encoded by legend"
            beam_hits.append((d, obs, nr, nc))
            d = _rot(d, turns[obs])
            assert len(beam_hits) <= 2
        hits.append(beam_hits)

    return out, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit only two square beams, with the active axis encoded by the source color.

    Input structure:
    - A fixed legend band sits on top: each colored vertical bar still says whether its obstacle color turns left or right.
    - The main field below contains a background, several edge-touching obstacle rectangles, and one solid source square whose color varies across examples.

    Rule:
    - Remove the legend from the output.
    - Read the source square color: even colors launch beams horizontally (left and right), odd colors launch beams vertically (up and down).
    - Each beam is a sliding copy of the whole square, painting only background cells in the source color.
    - When a beam's next footprint touches an obstacle color from the legend, rotate 90 degrees according to that color's left/right code and continue until the beam exits the grid.

    Key insight:
    The legend still explains how obstacles rotate beams, but it no longer tells you which initial directions exist. The source square's own color decides whether the puzzle uses the horizontal pair or the vertical pair.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend must be a top band"

    legend_turn = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, counts = np.unique(base, return_counts=True)
    bg = int(vals[counts.argmax()])
    sr, sc, sz, src_color = _find_source_square(base, set(legend_turn), bg)

    out = base.copy()
    H, W = out.shape
    active_dirs = _active_dirs_from_color(src_color)

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < H and c + sz - 1 < W

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = src_color
        out[r : r + sz, c : c + sz] = blk

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = out[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(src_color)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    for d0 in active_dirs:
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

            oc = obstacle_color_at_next(nr, nc)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            assert oc in legend_turn, "obstacle color must appear in the legend"
            d = _rot(d, legend_turn[oc])

    return out


# --------------- generator ---------------

LEGEND_COLOR_POOL = [1, 2, 3, 4]
SOURCE_COLOR_POOL = [6, 7, 8, 9]
BG_COLOR_POOL = [6, 7, 8, 9]
LEGEND_BG = 5

GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

SOURCE_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5

OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

MAX_RETRIES = 300


def _choose_span(
    axis_len: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
    min_len: int,
    max_len: int,
) -> tuple[int, int] | None:
    """Pick a span covering all required intervals while avoiding excluded lanes."""
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
        cutoff = lengths[min(len(lengths) - 1, len(lengths) // 2)]
        pool = [(start, end) for length, start, end in choices if length <= cutoff]
    else:
        pool = [(start, end) for _, start, end in choices]
    return random.choice(pool)


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors, strict=False):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
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


def _select_layout(src_color: int, second_hits: int) -> tuple[set[int], dict[int, int], dict[int, str]]:
    active = list(_active_dirs_from_color(src_color))
    perpendicular = [_N, _S] if set(active) == {_W, _E} else [_W, _E]

    if second_hits == 1:
        source = random.choice(active)
        target = random.choice(perpendicular)
        chains = {source: target}
        present = set(active) | {target}
    else:
        targets = perpendicular[:]
        random.shuffle(targets)
        chains = {active[0]: targets[0], active[1]: targets[1]}
        present = set(active) | set(perpendicular)

    side_turn: dict[int, str] = {}
    for side in active:
        if side in chains:
            side_turn[side] = _turn_to(side, chains[side])
        else:
            absent = [t for t in perpendicular if t not in present]
            assert len(absent) == 1
            side_turn[side] = _turn_to(side, absent[0])

    for target, source in {t: s for s, t in chains.items()}.items():
        side_turn[target] = _turn_to(target, _opposite(source))

    return present, chains, side_turn


def _compute_constraints(
    src_color: int,
    present: set[int],
    chains: dict[int, int],
    sr: int,
    sc: int,
    sz: int,
    lane: dict[int, int],
) -> tuple[dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]:
    required: dict[int, list[tuple[int, int]]] = {side: [] for side in present}
    excluded: dict[int, list[tuple[int, int]]] = {side: [] for side in present}
    active = set(_active_dirs_from_color(src_color))

    for side in active:
        if side not in present:
            continue
        base_coord = sc if side in (_N, _S) else sr
        required[side].append((base_coord, base_coord + sz - 1))

    for source, target in chains.items():
        target_lane = (lane[source], lane[source] + sz - 1)
        required[target].append(target_lane)
        exit_side = _opposite(source)
        if exit_side in excluded:
            excluded[exit_side].append((lane[target], lane[target] + sz - 1))

    return required, excluded


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < SOURCE_SZ2_PROB else 3
    src_color = random.choice(SOURCE_COLOR_POOL)
    second_hits = random.choice([1, 2])
    present, chains, side_turn = _select_layout(src_color, second_hits)

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

    lane: dict[int, int] = {}
    if _N in present:
        lane[_N] = thickness[_N] + 1
    if _S in present:
        lane[_S] = h - thickness[_S] - sz
    if _W in present:
        lane[_W] = thickness[_W] + 1
    if _E in present:
        lane[_E] = w - thickness[_E] - sz

    required, excluded = _compute_constraints(src_color, present, chains, sr, sc, sz, lane)

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
    side_color = dict(zip(sorted(present), obstacle_colors, strict=False))
    bg = random.choice([c for c in BG_COLOR_POOL if c != src_color])

    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = src_color
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

    turns = {side_color[side]: side_turn[side] for side in present}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    try:
        expected, hits = _simulate_paths(base, turns, bg, src_color)
        legacy_out, _ = _simulate_paths(base, turns, bg, src_color, dirs=(_N, _E, _S, _W))
    except AssertionError:
        return None

    if sum(len(beam) for beam in hits) != len(present):
        return None
    if any(len(beam) == 0 for beam in hits):
        return None
    if any(len(beam) > 2 for beam in hits):
        return None
    if {color for beam in hits for _, color, _, _ in beam} != set(side_color.values()):
        return None
    if np.array_equal(expected, legacy_out):
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
    raise RuntimeError("failed to generate fused e87109e9 input")
