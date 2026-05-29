# Variant of e87109e9
# Dimension: D17 — 障碍物功能分化 (obstacle)
# Change: 所有障碍物功能相同——都触发转弯，只是转弯方向不同（由图例决定）... → 传送门障碍物

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the D17 portal variant
# ------------------------------------------
# a) Concrete behavior:
#    - The top 6x24 legend stays visually the same: each used color still has a left/right bar.
#    - In the main grid, obstacle colors no longer appear once each. Instead, every used obstacle
#      color appears exactly twice, as a pair of congruent edge bars on opposite sides:
#        * one horizontal pair on north/south edges
#        * one vertical pair on west/east edges
#    - A teal sz x sz beam that would step into a portal bar does not paint that bar. It jumps to
#      the paired bar, reappears on that partner's interior side with the same alignment, paints the
#      reappearance square, then turns using the legend and keeps going.
#    - Small trace example:
#        old rule: north beam -> hit top red bar -> rotate left/right at that same top bar
#        new rule: north beam -> hit top red bar -> jump to bottom red bar -> turn there -> continue
#      The visible difference is a broken path segment plus an emergence point beside the matching
#      red bar elsewhere.
# b) Complexity calibration:
#    - Two portal colors are used per instance, each repeated twice. This gives repeated evidence for
#      pairing without creating too many colors or too many shapes.
#    - The legend still matters, because teleport alone is not enough: exit direction depends on the
#      left/right bar encoding.
#    - At least one beam in every sample performs a second portal interaction, so the variant is not
#      visually reducible to "straight teleport once and stop".
# c) Old assumptions that break under D17:
#    - Old `obstacle_color_at_next(...)` only needed the obstacle color, because every obstacle of that
#      color behaved identically in-place. That is no longer enough; the solver must identify which
#      component of the pair was hit.
#    - Old hit handling updated only direction (`d = turn(...)`) and left position unchanged. Under the
#      portal rule that would silently implement the original task. The new logic must change position
#      first (teleport) and only then rotate.
#    - Old generator assigned each side its own color. That prevents portal pairing entirely. The new
#      generator must deliberately duplicate colors in matched pairs.
# d) Placement legality:
#    - Portal pairs are opposite-edge bars with identical orientation and shape, so alignment transfer
#      is unambiguous.
#    - North/south spans are kept away from west/east obstacle columns, and west/east spans are kept
#      away from north/south obstacle rows. This prevents corner overlap.
#    - The teal square is placed strictly inside the portal lanes so every beam reaches its first portal
#      through background cells before any teleport occurs.
# e) Visual observability:
#    - Humans can see the same color occurring twice and can see the beam disappear into one copy and
#      reappear next to the other copy in the output.
#    - The legend still gives a repeated binary clue (left vs right) for what the beam does after
#      emergence, so the task remains inferable from a few examples.
# f) Visual cleanliness:
#    - Only four obstacle rectangles are present, all edge-anchored, with clean corner gaps.
#    - The beam paints only background cells, never overwriting portals, so obstacles remain legible.
# g) Color-rule inferability:
#    - No hidden ordering by color id is used. Pairing is purely by visible same-color repetition.
#    - Only two portal colors are active per sample, keeping the color logic small and observable.
# h) New solver assumptions and generator guarantees:
#    - `main()` assumes every used obstacle color has exactly two solid rectangular components. The
#      generator enforces exactly one north/south pair and one west/east pair.
#    - `main()` assumes each pair is congruent so relative alignment can be transferred. The generator
#      renders each pair from one shared span and one shared thickness.
#    - `main()` assumes portal paths do not cycle. The generator chooses required/excluded spans and
#      then validates hit counts with the shared oracle before accepting a sample.


_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_N, _E, _S, _W = range(4)


def _parse_legend(grid: np.ndarray, legend_h: int) -> dict[int, str]:
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
        turn[color] = "L" if int(xs[0]) <= (interior.shape[1] - 1) // 2 else "R"
    assert len(turn) == 4, "legend must encode four colors"
    return turn


def _find_teal_square(g: np.ndarray, teal: int = 8) -> tuple[int, int, int]:
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


def _interval(start: int, size: int) -> tuple[int, int]:
    return start, start + size - 1


def _rectangular_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for cells in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert int(np.sum(mask[r0 : r1 + 1, c0 : c1 + 1])) == area, "portal components must be solid rectangles"
        rects.append((int(r0), int(c0), int(r1), int(c1)))
    rects.sort()
    return rects


def _collect_portals(base: np.ndarray, bg: int, teal: int) -> dict[int, list[tuple[int, int, int, int]]]:
    portals: dict[int, list[tuple[int, int, int, int]]] = {}
    for color in sorted(map(int, np.unique(base))):
        if color in (bg, teal):
            continue
        rects = _rectangular_components(base == color)
        assert len(rects) == 2, "each used obstacle color must appear exactly twice"
        (a0, b0, a1, b1), (c0, d0, c1, d1) = rects
        assert (a1 - a0, b1 - b0) == (c1 - c0, d1 - d0), "paired portals must be congruent"
        portals[color] = rects
    assert len(portals) >= 1, "at least one portal pair must exist"
    return portals


def _overlaps_bbox(r: int, c: int, sz: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return not (r + sz - 1 < r0 or r > r1 or c + sz - 1 < c0 or c > c1)


def _obstacle_color_at(base: np.ndarray, r: int, c: int, sz: int, bg: int, teal: int) -> int | None:
    blk = base[r : r + sz, c : c + sz]
    colors = set(map(int, np.unique(blk)))
    colors.discard(bg)
    colors.discard(teal)
    if not colors:
        return None
    assert len(colors) == 1, "beam footprint must not cross mixed obstacle colors"
    return int(next(iter(colors)))


def _paired_exit_anchor(
    hit_bbox: tuple[int, int, int, int],
    other_bbox: tuple[int, int, int, int],
    nr: int,
    nc: int,
    sz: int,
    d: int,
) -> tuple[int, int]:
    hr0, hc0, hr1, hc1 = hit_bbox
    or0, oc0, or1, oc1 = other_bbox
    if d in (_N, _S):
        assert (hr1 - hr0) <= (hc1 - hc0), "north/south movement must hit a horizontal portal"
        assert (hc1 - hc0) == (oc1 - oc0), "paired portal spans must match"
        offset = nc - hc0
        ec = oc0 + offset
        er = or0 - sz if d == _N else or1 + 1
        return er, ec
    assert (hc1 - hc0) <= (hr1 - hr0), "east/west movement must hit a vertical portal"
    assert (hr1 - hr0) == (or1 - or0), "paired portal spans must match"
    offset = nr - hr0
    er = or0 + offset
    ec = oc0 - sz if d == _W else oc1 + 1
    return er, ec


def _trace_portal_beams(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    teal: int = 8,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, int, int]]]]:
    sr, sc, sz = _find_teal_square(base, teal=teal)
    portals = _collect_portals(base, bg, teal)
    out = base.copy()
    h, w = out.shape
    hits: list[list[tuple[int, int, int, int, int, int]]] = []

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal
        out[r : r + sz, c : c + sz] = blk

    for d0 in range(4):
        r, c, d = sr, sc, d0
        beam_hits: list[tuple[int, int, int, int, int, int]] = []
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "beam path must not cycle"
            seen.add(state)

            dr, dc = _DIRS[d]
            nr, nc = r + dr, c + dc
            if not in_bounds(nr, nc):
                break

            oc = _obstacle_color_at(base, nr, nc, sz, bg, teal)
            if oc is None:
                r, c = nr, nc
                paint_block(r, c)
                continue

            assert oc in turns, "portal color must be listed in the legend"
            rects = portals[oc]
            hit_ids = [i for i, bbox in enumerate(rects) if _overlaps_bbox(nr, nc, sz, bbox)]
            assert len(hit_ids) == 1, "beam must enter exactly one portal component"
            hit_idx = hit_ids[0]
            exit_idx = 1 - hit_idx
            er, ec = _paired_exit_anchor(rects[hit_idx], rects[exit_idx], nr, nc, sz, d)
            assert in_bounds(er, ec), "portal exit must stay inside the main grid"
            assert _obstacle_color_at(base, er, ec, sz, bg, teal) is None, "portal exit must be on a non-obstacle lane"

            r, c = er, ec
            paint_block(r, c)
            beam_hits.append((d, oc, nr, nc, er, ec))
            d = _rot(d, turns[oc])
            assert len(beam_hits) <= 2, "generator only creates up to two portal hits per beam"
        hits.append(beam_hits)

    return out, hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch four teal beams from the central square, but colored blockers now work as paired portals.

    Input structure:
    - The top band is still a 6x24 legend on grey background, with one colored vertical bar per cell.
    - The lower region contains a single solid teal square plus colored obstacle bars; each used obstacle color
      appears exactly twice, forming a same-color portal pair.

    Rule:
    - Remove the legend from the output.
    - From the teal square, slide a sz x sz teal beam in each cardinal direction, painting only background cells.
    - If the next step would overlap a colored portal, do not turn there. Instead, jump to the partner portal of the
      same color, reappear on that partner's interior side with matching alignment, paint the reappearance square,
      then rotate according to the legend (left bar = counterclockwise, right bar = clockwise).
    - Continue until the beam leaves the grid. Portal blocks themselves remain unchanged.

    Key insight:
    - The duplicated obstacle colors are the visible pairing signal: each hit is a spatial jump to the other copy of
      that color, and the legend only controls the post-teleport turn direction.
    """
    g = np.asarray(grid, dtype=int)

    legend_bg = 5
    rows_with_legend = np.where(np.any(g == legend_bg, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    turns = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after cropping the legend"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    out, _hits = _trace_portal_beams(base, turns, bg, teal=8)
    return out


# --------------- generator ---------------

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

# Teal square and portal sizes
TEAL_SZ2_PROB = 0.75
SQUARE_EDGE_GAP = 5
OBS_THICKNESS_RANGE = (2, 4)
SPAN_MIN_MULT = 3
SPAN_MAX_MULT = 14
SPAN_COMPACT_PROB = 0.45

# Retry budget
MAX_RETRIES = 400


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
    lo_bound: int,
    hi_bound: int,
) -> tuple[int, int] | None:
    req0 = min(a for a, _ in required)
    req1 = max(b for _, b in required)
    lo = max(min_len, req1 - req0 + 1)
    hi = min(max_len, hi_bound - lo_bound + 1, axis_len)
    choices: list[tuple[int, int, int]] = []
    for length in range(lo, hi + 1):
        start0 = max(lo_bound, req1 - length + 1)
        start1 = min(req0, hi_bound - length + 1)
        for start in range(start0, start1 + 1):
            end = start + length - 1
            if end > hi_bound:
                continue
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


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    t_ns = random.randint(*OBS_THICKNESS_RANGE)
    t_we = random.randint(*OBS_THICKNESS_RANGE)

    min_r = max(SQUARE_EDGE_GAP, t_ns + 3)
    max_r = min(h - sz - SQUARE_EDGE_GAP, h - t_ns - sz - 3)
    min_c = max(SQUARE_EDGE_GAP, t_we + 3)
    max_c = min(w - sz - SQUARE_EDGE_GAP, w - t_we - sz - 3)
    if min_r > max_r or min_c > max_c:
        return None
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    top_inner = t_ns
    bottom_inner = h - t_ns - sz
    left_inner = t_we
    right_inner = w - t_we - sz

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    ns_color, we_color = random.sample(legend_colors, 2)
    turns = {color: random.choice(["L", "R"]) for color in legend_colors}

    second_hit_count = random.choice([1, 2, 2])
    second_hit_dirs = set(random.sample([_N, _E, _S, _W], second_hit_count))

    src_cols = [_interval(sc, sz)]
    src_rows = [_interval(sr, sz)]
    left_cols = _interval(left_inner, sz)
    right_cols = _interval(right_inner, sz)
    top_rows = _interval(top_inner, sz)
    bottom_rows = _interval(bottom_inner, sz)

    required_ns = list(src_cols)
    excluded_ns: list[tuple[int, int]] = []
    required_we = list(src_rows)
    excluded_we: list[tuple[int, int]] = []

    if _E in second_hit_dirs:
        required_ns.append(left_cols)
        excluded_we.append(bottom_rows if turns[we_color] == "L" else top_rows)
    else:
        excluded_ns.append(left_cols)
    if _W in second_hit_dirs:
        required_ns.append(right_cols)
        excluded_we.append(top_rows if turns[we_color] == "L" else bottom_rows)
    else:
        excluded_ns.append(right_cols)
    if _N in second_hit_dirs:
        required_we.append(bottom_rows)
        excluded_ns.append(right_cols if turns[ns_color] == "L" else left_cols)
    else:
        excluded_we.append(bottom_rows)
    if _S in second_hit_dirs:
        required_we.append(top_rows)
        excluded_ns.append(left_cols if turns[ns_color] == "L" else right_cols)
    else:
        excluded_we.append(top_rows)

    ns_span = _choose_span(
        w,
        required_ns,
        excluded_ns,
        SPAN_MIN_MULT * sz,
        min(SPAN_MAX_MULT * sz, w),
        t_we,
        w - t_we - 1,
    )
    if ns_span is None:
        return None
    we_span = _choose_span(
        h,
        required_we,
        excluded_we,
        SPAN_MIN_MULT * sz,
        min(SPAN_MAX_MULT * sz, h),
        t_ns,
        h - t_ns - 1,
    )
    if we_span is None:
        return None

    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    bg = random.choice(bg_choices)
    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    base[0:t_ns, ns_span[0] : ns_span[1] + 1] = ns_color
    base[h - t_ns : h, ns_span[0] : ns_span[1] + 1] = ns_color
    base[we_span[0] : we_span[1] + 1, 0:t_we] = we_color
    base[we_span[0] : we_span[1] + 1, w - t_we : w] = we_color

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        expected, hits = _trace_portal_beams(base, turns, bg, teal=TEAL)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    actual_second = {d for d, beam in enumerate(hits) if len(beam) == 2}
    if any(c < 1 or c > 2 for c in hit_counts):
        return None
    if actual_second != second_hit_dirs:
        return None
    if sum(hit_counts) < 5:
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
    raise RuntimeError("failed to generate e87109e9 portal variant input")
