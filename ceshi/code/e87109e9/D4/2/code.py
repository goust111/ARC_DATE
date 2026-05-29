# Variant of e87109e9
# Dimension: D4 — 轨迹涂色模式 (trajectory)
# Change: 沿路径连续涂青色(8)，从源到障碍物或网格边缘形成连续实线轨迹... → 按图例条目顺序循环涂色

import random

import numpy as np

from arc_utils import bbox_coords, components_mask

# Design analysis for the D4 variant
# spec_variant:
#   a_specific_behavior:
#     input_pixels:
#       legend remains a 6x24 top band with four colored vertical bars.
#       Left/right bar placement still means turn direction, while the visible
#       left-to-right legend order is now the color cycle used by beam trails.
#     output_pixels:
#       the legend is removed; the teal source square stays intact; background
#       cells visited by beams are recolored with legend colors instead of teal.
#       All beams start with legend entry 1. After a beam hits obstacle color X,
#       its next straight run switches to the legend color immediately after X.
#     trace_example:
#       original: source -> obstacle -> turned continuation all stayed teal.
#       variant: source -> first obstacle paints color A, then the continuation
#       after obstacle A paints B, and after obstacle B paints C.
#   b_complexity_calibration:
#     keep the original obstacle geometry and one-source/four-beam setup. To make
#     the cycle visible, generator instances require at least one beam with two
#     turns so the first three legend colors all appear on the output trails.
#   c_implicit_assumptions_review:
#     original_solver_assumption_1:
#       reading collisions from the mutable output was harmless when all trails
#       stayed teal and were explicitly ignored. Here trail colors overlap the
#       legend palette, so collision checks must read from the immutable base grid.
#     original_solver_assumption_2:
#       once trail colors vary, shared suffixes become possible. A strict
#       no-overlap rule makes the original four-beam geometry nearly unsamplable,
#       so the variant uses first-arrival painting: beams only color background
#       cells and never overwrite earlier trail cells.
#     original_solver_assumption_3:
#       parsing the legend into only color->turn loses the visible order. The
#       variant needs both the order and the per-color turn direction.
#   d_placement_legality:
#     the generator keeps edge-touching rectangular obstacles, non-overlap with
#     the source, and span constraints for intended hit chains. It additionally
#     assigns colors so the chosen two-turn chain follows legend colors A then B,
#     making A/B/C visible after simulation.
#   e_visual_observability:
#     the same legend colors reappear on trails in the visible legend order.
#     Legend order varies per sample, so the rule cannot be solved from numeric
#     color ids. The first three legend entries are always exercised in output.
#   f_visual_cleanliness:
#     shared suffixes stay visually stable because only the first arrival paints;
#     later beams follow the same geometry without recoloring earlier trail cells.
#   g_color_inference:
#     all color changes are anchored to visible legend entries only; no hidden
#     sorting or invisible color-id arithmetic is used.

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> tuple[list[int], dict[int, str]]:
    """
    Read the legend as an ordered list of colors plus a per-color turn map.

    Separator columns are solid legend background. Each cell contains one colored
    bar on black; the bar's left/right placement means L/R turning.
    """
    legend = grid[:legend_h, :]
    assert legend.size, "legend must be non-empty"
    vals, cnts = np.unique(legend, return_counts=True)
    bg = int(vals[cnts.argmax()])

    sep_cols = [c for c in range(legend.shape[1]) if np.all(legend[:, c] == bg)]
    assert len(sep_cols) >= 2, "legend separator columns must exist"

    order: list[int] = []
    turns: dict[int, str] = {}
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
        turns[color] = "L" if x <= (interior.shape[1] - 1) // 2 else "R"
        order.append(color)
    assert len(order) == 4, "legend must encode four ordered colors"
    return order, turns


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
    """Return (r0,c0,size) for the unique solid teal square."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot four square-width beams whose trail colors advance through the legend.

    Input structure:
    - The top band is a grey legend with four colored vertical bars. Bar side still encodes the turn direction for that legend color.
    - The main grid below contains one solid teal source square plus several solid obstacle rectangles colored from the legend.

    Rule:
    - Remove the legend band in the output.
    - Launch one beam in each cardinal direction from the teal square, moving as a sz-by-sz sliding footprint.
    - All beams start by painting with the first legend color. Whenever a beam advances through background, it paints using its current active color, but only onto still-background cells.
    - When the next footprint hits an obstacle of color C, keep position, rotate according to C's legend bar, and switch the active color to the legend entry immediately after C in the visible left-to-right cycle. Stop at grid edges.

    Key insight: The legend now carries two linked but distinct signals: each color still encodes turn direction, and the left-to-right legend order controls how beam colors advance after every obstacle encounter.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    cycle_colors, legend_turn = _parse_legend(g, legend_h)
    next_color = {
        cycle_colors[i]: cycle_colors[(i + 1) % len(cycle_colors)]
        for i in range(len(cycle_colors))
    }

    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after legend crop"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    teal = 8
    sr, sc, sz = _find_teal_square(base, teal=teal)
    out = base.copy()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < out.shape[0] and c + sz - 1 < out.shape[1]

    def obstacle_color_at_next(nr: int, nc: int) -> int | None:
        blk = base[nr : nr + sz, nc : nc + sz]
        cols = set(map(int, np.unique(blk)))
        cols.discard(bg)
        cols.discard(teal)
        if not cols:
            return None
        assert len(cols) == 1, "beam footprint must not hit mixed obstacle colors"
        return int(next(iter(cols)))

    def paint_block(r: int, c: int, color: int) -> None:
        base_blk = base[r : r + sz, c : c + sz]
        out_blk = out[r : r + sz, c : c + sz]
        mask = base_blk == bg
        out_blk[mask & (out_blk == bg)] = color
        out[r : r + sz, c : c + sz] = out_blk

    for d0 in range(4):
        r, c, d = sr, sc, d0
        active_color = cycle_colors[0]
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
                paint_block(r, c, active_color)
                continue

            assert oc in legend_turn, "obstacle color must be encoded by legend"
            d = _rot(d, legend_turn[oc])
            active_color = next_color[oc]

    return out


# --------------- generator ---------------

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

OBS_COUNT_4_PROB = 0.45
OBS_THICKNESS_RANGE = (2, 4)
OBS_SINGLE_MARGIN = 6
OBS_BOTH_GAP = 2

SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.4

MAX_RETRIES = 800


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


def _simulate_paths(
    base: np.ndarray, turns: dict[int, str], cycle_colors: list[int], bg: int
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]], set[int]]:
    """Simulate the variant on a fixed base grid and collect visible path colors."""
    vals, counts = np.unique(base, return_counts=True)
    assert int(vals[counts.argmax()]) == bg

    sr, sc, sz = _find_teal_square(base, teal=TEAL)
    out = base.copy()
    H, W = out.shape
    hits: list[list[tuple[int, int, int, int]]] = []
    next_color = {
        cycle_colors[i]: cycle_colors[(i + 1) % len(cycle_colors)]
        for i in range(len(cycle_colors))
    }
    used_path_colors: set[int] = set()

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < H and c + sz - 1 < W

    def obstacle_at(r: int, c: int) -> int | None:
        colors = set(map(int, np.unique(base[r : r + sz, c : c + sz])))
        colors.discard(bg)
        colors.discard(TEAL)
        if not colors:
            return None
        assert len(colors) == 1
        return int(next(iter(colors)))

    def paint(r: int, c: int, color: int) -> None:
        base_blk = base[r : r + sz, c : c + sz]
        out_blk = out[r : r + sz, c : c + sz]
        mask = base_blk == bg
        out_blk[mask & (out_blk == bg)] = color
        out[r : r + sz, c : c + sz] = out_blk

    for d0 in range(4):
        r, c, d = int(sr), int(sc), d0
        active_color = int(cycle_colors[0])
        beam_hits: list[tuple[int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
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
                paint(r, c, active_color)
                used_path_colors.add(active_color)
                continue

            assert obs in turns
            beam_hits.append((d, obs, nr, nc))
            d = _rot(d, turns[obs])
            active_color = int(next_color[obs])
            assert len(beam_hits) <= 2

        hits.append(beam_hits)

    return out, hits, used_path_colors


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    """Build the 6x24 legend band from ordered colors and turn assignments."""
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
    """Pick non-overlapping source->adjacent-target pairs for beams with two hits."""
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
    """Assign L/R turns per side, preferring outward exits for non-chain sources."""
    side_turn: dict[int, str] = {}
    for side in present:
        if side in chains:
            nxt = chains[side]
            side_turn[side] = "L" if nxt == (side - 1) % 4 else "R"
            continue
        outward = [t for t in ("L", "R") if _rot(side, t) not in present]
        if outward:
            side_turn[side] = random.choice(outward)
        else:
            side_turn[side] = random.choice(["L", "R"])
    return side_turn


def _compute_span_constraints(
    present: set[int], side_turn: dict[int, str], chains: dict[int, int],
    sr: int, sc: int, sz: int, lane: dict[int, int],
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


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    obstacle_count = 4 if random.random() < OBS_COUNT_4_PROB else 3
    present = set(random.sample([_N, _E, _S, _W], obstacle_count))

    second_hits = 1
    target_hits = obstacle_count + second_hits

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
        present, side_turn, chains, sr, sc, sz, lane,
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
    chain_src, chain_tgt = next(iter(chains.items()))
    side_color: dict[int, int] = {
        chain_src: legend_colors[0],
        chain_tgt: legend_colors[1],
    }
    remaining_sides = [side for side in sorted(present) if side not in side_color]
    remaining_colors = legend_colors[2:].copy()
    random.shuffle(remaining_colors)
    for side, color in zip(remaining_sides, remaining_colors, strict=False):
        side_color[side] = color
    assert len(set(side_color.values())) == len(side_color)
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
        expected, hits, used_path_colors = _simulate_paths(base, turns, legend_colors, bg)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    hit_colors = {color for beam in hits for _, color, _, _ in beam}
    if sum(hit_counts) != target_hits or any(c > 2 for c in hit_counts):
        return None
    if set(side_color.values()) - hit_colors:
        return None
    if not any(len(beam) == 2 for beam in hits):
        return None
    if not set(legend_colors[:3]).issubset(used_path_colors):
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
    raise RuntimeError("failed to generate e87109e9 input")
