# Combo variant of e87109e9 (index 24)
# combo_spec: D7/0+D17/2
# Name: 传送门加统一方块

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the fused "uniform square portals" variant
# ---------------------------------------------------------------
# a) Concrete behavior and trace:
#    - The legend remains a 6x24 grey strip with four colored bars. Bar side still encodes turn
#      direction for that color.
#    - The main grid now contains exactly four obstacle blocks, and every obstacle is the same
#      square size (4x4). Two opposite squares share one color; the other two share another color.
#    - Example with sz=2:
#        old task: north beam reaches a red blocker above the source and turns there in place
#        fused task: north beam reaches the upper red square, jumps to the lower red square's
#        interior side, paints the re-entry footprint there, then turns by the red legend rule
#        and continues horizontally. If that new horizontal lane matches a side portal square,
#        it can portal a second time.
#      The visual difference is that the path is no longer spatially continuous around the first
#      obstacle: the beam disappears at one square and reappears beside the other same-color square.
#    - D7 and D17 are genuinely fused here: making every obstacle the same square removes shape cues,
#      so the only way to pair portals is by repeated color.
#
# b) Complexity calibration:
#    - Only two active portal colors are used per sample, each appearing twice. This gives repeated
#      evidence for color-based pairing without flooding the board with many mappings.
#    - All four obstacles are identical 4x4 squares. The task is therefore not about recognizing
#      special geometry; it is about combining same-color pairing with legend-controlled turning.
#    - At least one of the north/south beams performs a second portal interaction in every sample,
#      so the variant is visibly different from "teleport once and stop".
#
# c) Implicit-assumption audit against the original solver:
#    - Original `obstacle_color_at_next(...)` only needed a color and then turned in place. That is
#      no longer sufficient: the solver must detect which component of a same-color pair was hit and
#      relocate the beam before rotating.
#    - Original obstacles could be arbitrary rectangles. That breaks D7/0. The fused solver now
#      asserts that every used obstacle component is a solid square and that all used squares have
#      the same side length.
#    - Original generator gave each obstacle a distinct color. That would destroy portal pairing.
#      The fused generator deliberately duplicates the two active colors across opposite squares.
#    - Portal exit alignment assumes the hit beam lane fits inside the partner square in the
#      transverse dimension. The generator enforces that by placing top/bottom squares so they fully
#      cover the source columns, and side squares so they fully cover the relevant beam rows.
#
# d) Placement legality:
#    - The generator builds four non-overlapping 4x4 squares: one top, one bottom, one left, one
#      right. Each keeps at least one-cell separation from the teal source square.
#    - The teal source is placed far enough from borders that all four portal squares fit and remain
#      inside the main grid.
#    - The chosen side-portal row band is searched explicitly so intended second-hit lanes are
#      contained while forbidden lanes are disjoint, preventing accidental extra portal hits.
#
# e) Visual observability:
#    - A human can see two repeated colors and four identical squares. Because shape and size are
#      uniform, repeated color becomes the only plausible portal-pair signal.
#    - The output shows beams vanishing at one square and resuming near the matching square. The
#      legend still supplies the repeated left/right clue for post-teleport turning.
#
# f) Visual cleanliness:
#    - The layout is a sparse cross: source near the middle, four isolated portal squares around it.
#      There is no corner clutter, no mixed-color overlaps, and teal never overwrites obstacle cells.
#    - Keeping obstacle size fixed across a sample makes the board easier to parse than the original
#      mix of unrelated rectangles.
#
# g) Change inferability:
#    - No hidden ordering by numeric color id is used. Same-color repetition is the only pairing key.
#    - The legend retains exactly the same observable coding scheme as the base task, so the new
#      burden on the solver is specifically "pair same-color squares and jump", not a new color map.
#
# h) New solver assumptions and generator guarantees:
#    - `main()` assumes every used obstacle color has exactly two solid congruent square components.
#      `generate_input()` always constructs one top/bottom pair and one left/right pair.
#    - `main()` assumes portal paths do not cycle. The generator validates each sample with the same
#      portal oracle and rejects any layout that produces extra or cyclic hits.
#    - `main()` assumes the portal rule changes behavior relative to plain in-place turning. The
#      generator compares against a non-portal trace and rejects degenerate layouts with zero output
#      difference.


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

# Source and portal geometry
TEAL_SZ2_PROB = 0.75
PORTAL_SZ = 4
VERT_GAP_CHOICES = [1, 2, 3]
HORIZ_GAP_CHOICES = [2, 3, 4]
MAX_RETRIES = 500


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


def _find_teal_square(g: np.ndarray, teal: int = TEAL) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for cells in components_mask(g == teal, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        hh = r1 - r0 + 1
        ww = c1 - c0 + 1
        if hh != ww:
            continue
        if int(np.sum(g[r0 : r1 + 1, c0 : c1 + 1] == teal)) != hh * ww:
            continue
        candidates.append((int(r0), int(c0), int(hh)))
    assert len(candidates) == 1, "main grid must contain exactly one solid teal square"
    return candidates[0]


def _rot_ccw(d: int) -> int:
    return (d - 1) % 4


def _rot_cw(d: int) -> int:
    return (d + 1) % 4


def _rot(d: int, turn: str) -> int:
    return _rot_ccw(d) if turn == "L" else _rot_cw(d)


def _rectangular_components(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for cells in components_mask(mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(cells)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert int(np.sum(mask[r0 : r1 + 1, c0 : c1 + 1])) == area, "components must be solid rectangles"
        rects.append((int(r0), int(c0), int(r1), int(c1)))
    rects.sort()
    return rects


def _collect_portals(
    base: np.ndarray, bg: int, teal: int
) -> tuple[dict[int, list[tuple[int, int, int, int]]], int]:
    portals: dict[int, list[tuple[int, int, int, int]]] = {}
    portal_size: int | None = None
    for color in sorted(map(int, np.unique(base))):
        if color in (bg, teal):
            continue
        rects = _rectangular_components(base == color)
        assert len(rects) == 2, "each used portal color must appear exactly twice"
        checked: list[tuple[int, int, int, int]] = []
        for r0, c0, r1, c1 in rects:
            hh = r1 - r0 + 1
            ww = c1 - c0 + 1
            assert hh == ww, "portal components must be squares"
            if portal_size is None:
                portal_size = hh
            assert hh == portal_size, "all portal squares must share one size"
            checked.append((r0, c0, r1, c1))
        portals[color] = checked
    assert len(portals) >= 1, "at least one portal pair must exist"
    assert portal_size is not None
    return portals, portal_size


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
    assert len(colors) == 1, "beam footprint must not hit mixed obstacle colors"
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
    hit_h = hr1 - hr0 + 1
    hit_w = hc1 - hc0 + 1
    other_h = or1 - or0 + 1
    other_w = oc1 - oc0 + 1
    assert (hit_h, hit_w) == (other_h, other_w), "portal pairs must be congruent"
    if d in (_N, _S):
        offset = nc - hc0
        assert 0 <= offset <= hit_w - sz, "portal hit must preserve a valid horizontal offset"
        ec = oc0 + offset
        er = or0 - sz if d == _N else or1 + 1
        return er, ec
    offset = nr - hr0
    assert 0 <= offset <= hit_h - sz, "portal hit must preserve a valid vertical offset"
    er = or0 + offset
    ec = oc0 - sz if d == _W else oc1 + 1
    return er, ec


def _trace_portal_beams(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    teal: int = TEAL,
) -> tuple[np.ndarray, list[list[tuple[int, int, int, int, int, int]]]]:
    sr, sc, sz = _find_teal_square(base, teal=teal)
    portals, _portal_size = _collect_portals(base, bg, teal)
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


def _trace_plain_turn_beams(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    teal: int = TEAL,
) -> np.ndarray:
    sr, sc, sz = _find_teal_square(base, teal=teal)
    out = base.copy()
    h, w = out.shape

    def in_bounds(r: int, c: int) -> bool:
        return 0 <= r and 0 <= c and r + sz - 1 < h and c + sz - 1 < w

    def paint_block(r: int, c: int) -> None:
        blk = out[r : r + sz, c : c + sz]
        blk[blk == bg] = teal
        out[r : r + sz, c : c + sz] = blk

    for d0 in range(4):
        r, c, d = sr, sc, d0
        seen: set[tuple[int, int, int]] = set()
        while True:
            state = (r, c, d)
            assert state not in seen, "plain-turn beam path must not cycle"
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

            assert oc in turns, "obstacle color must be listed in the legend"
            d = _rot(d, turns[oc])

    return out


def _behavior_metrics_from_base(base: np.ndarray, turns: dict[int, str], bg: int) -> dict[str, int]:
    portal_out, hits = _trace_portal_beams(base, turns, bg, teal=TEAL)
    second_hits = sum(max(0, len(beam) - 1) for beam in hits)
    total_hits = sum(len(beam) for beam in hits)
    teleport_distance = sum(abs(er - nr) + abs(ec - nc) for beam in hits for _, _, nr, nc, er, ec in beam)
    try:
        plain_out = _trace_plain_turn_beams(base, turns, bg, teal=TEAL)
        plain_diff = int(np.count_nonzero(portal_out != plain_out))
    except AssertionError:
        plain_diff = int(base.size)
    return {
        "plain_diff": int(plain_diff),
        "second_hits": int(second_hits),
        "total_hits": int(total_hits),
        "teleport_distance": int(teleport_distance),
    }


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _choose_lane_start(
    axis_len: int,
    lane_size: int,
    required: list[tuple[int, int]],
    excluded: list[tuple[int, int]],
) -> int | None:
    starts: list[int] = []
    for start in range(0, axis_len - lane_size + 1):
        end = start + lane_size - 1
        if not all(start <= a and b <= end for a, b in required):
            continue
        if not all(b < start or a > end for a, b in excluded):
            continue
        starts.append(start)
    return random.choice(starts) if starts else None


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    p = PORTAL_SZ

    gap_t = random.choice(VERT_GAP_CHOICES)
    gap_b = random.choice(VERT_GAP_CHOICES)
    gap_l = random.choice(HORIZ_GAP_CHOICES)
    gap_r = random.choice(HORIZ_GAP_CHOICES)

    min_r = p + gap_t
    max_r = h - p - sz - gap_b
    min_c = p + gap_l
    max_c = w - p - sz - gap_r
    if min_r > max_r or min_c > max_c:
        return None
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    src_rows = (sr, sr + sz - 1)
    north_rows = (sr + gap_b, sr + gap_b + sz - 1)
    south_rows = (sr - gap_t, sr - gap_t + sz - 1)

    lane_choices: list[tuple[frozenset[int], int]] = []
    for subset in (frozenset({_N}), frozenset({_S}), frozenset({_N, _S})):
        required = [src_rows]
        excluded: list[tuple[int, int]] = []
        if _N in subset:
            required.append(north_rows)
        else:
            excluded.append(north_rows)
        if _S in subset:
            required.append(south_rows)
        else:
            excluded.append(south_rows)
        lane = _choose_lane_start(h, p, required, excluded)
        if lane is not None:
            lane_choices.append((subset, lane))
    if not lane_choices:
        return None

    lane_choices.sort(key=lambda item: (len(item[0]), random.random()), reverse=True)
    if len(lane_choices[0][0]) == 2 and random.random() < 0.65:
        second_dirs, we_r = lane_choices[0]
    else:
        second_dirs, we_r = random.choice(lane_choices)

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    ns_color, we_color = random.sample(legend_colors, 2)
    turns = {color: random.choice(["L", "R"]) for color in legend_colors}

    ns_c_lo = max(0, sc - (p - sz))
    ns_c_hi = min(sc, w - p)
    if ns_c_lo > ns_c_hi:
        return None
    ns_c = random.randint(ns_c_lo, ns_c_hi)

    top_r = sr - gap_t - p
    bot_r = sr + sz + gap_b
    left_c = sc - gap_l - p
    right_c = sc + sz + gap_r

    rects = {
        "top": (top_r, ns_c, top_r + p - 1, ns_c + p - 1),
        "bottom": (bot_r, ns_c, bot_r + p - 1, ns_c + p - 1),
        "left": (we_r, left_c, we_r + p - 1, left_c + p - 1),
        "right": (we_r, right_c, we_r + p - 1, right_c + p - 1),
        "src": (sr, sc, sr + sz - 1, sc + sz - 1),
    }

    for name, rect in rects.items():
        r0, c0, r1, c1 = rect
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            return None
        if name != "src" and _rects_overlap(rect, rects["src"]):
            return None
    for a, b in (("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right"), ("left", "right"), ("top", "bottom")):
        if _rects_overlap(rects[a], rects[b]):
            return None

    bg_choices = [c for c in BG_COLOR_POOL if c not in legend_colors]
    bg = random.choice(bg_choices)
    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL

    base[top_r : top_r + p, ns_c : ns_c + p] = ns_color
    base[bot_r : bot_r + p, ns_c : ns_c + p] = ns_color
    base[we_r : we_r + p, left_c : left_c + p] = we_color
    base[we_r : we_r + p, right_c : right_c + p] = we_color

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        expected, hits = _trace_portal_beams(base, turns, bg, teal=TEAL)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    if any(count < 1 or count > 2 for count in hit_counts):
        return None
    if hit_counts[_E] != 1 or hit_counts[_W] != 1:
        return None

    actual_second_dirs = {d for d, beam in enumerate(hits) if len(beam) == 2}
    if actual_second_dirs != set(second_dirs):
        return None

    metrics = _behavior_metrics_from_base(base, turns, bg)
    if metrics["second_hits"] < 1:
        return None
    if metrics["total_hits"] < 5:
        return None
    if metrics["plain_diff"] <= 0:
        return None
    if metrics["teleport_distance"] <= 0:
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Send four teal square beams through a field of same-sized portal squares whose colors still encode turns.

    Input structure:
    - The top band is a grey legend with four colored vertical bars; left-bar means counterclockwise and right-bar means clockwise.
    - The lower region contains one solid teal square plus four obstacle squares of one shared size.
    - Used obstacle colors appear exactly twice, so each repeated color forms a portal pair.

    Rule:
    - Remove the legend from the output.
    - Launch one sz x sz teal beam in each cardinal direction from the source square, painting only background cells.
    - When the next beam step enters a colored square, identify the other square of the same color, jump to that partner's interior side with the same transverse offset, paint the re-entry footprint there, then rotate according to the legend color for that portal.
    - Keep tracing until the beam leaves the grid. Portal squares themselves do not get repainted.

    Key insight:
    - Because every obstacle has the same square shape and size, geometry no longer distinguishes them; the repeated color is the portal-pair signal, while the legend still determines only the post-teleport turn.
    """
    g = np.asarray(grid, dtype=int)

    rows_with_legend = np.where(np.any(g == LEGEND_BG, axis=1))[0]
    assert rows_with_legend.size, "legend rows must be present"
    legend_h = int(rows_with_legend.max() + 1)
    assert np.array_equal(rows_with_legend, np.arange(legend_h)), "legend color must be confined to the top band"

    turns = _parse_legend(g, legend_h)
    base = g[legend_h:, :].copy()
    assert base.size, "main grid must remain after cropping the legend"

    vals, cnts = np.unique(base, return_counts=True)
    bg = int(vals[cnts.argmax()])
    out, _hits = _trace_portal_beams(base, turns, bg, teal=TEAL)
    return out
