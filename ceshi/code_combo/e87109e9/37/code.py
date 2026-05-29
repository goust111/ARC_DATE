# Combo variant of e87109e9 (index 37)
# combo_spec: D8/2+D17/2
# Name: 对称障碍物加传送门

import random

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis for the D8/2 + D17/2 combo
# ------------------------------------------
# a) Concrete behavior:
#    - The legend keeps the original 6x24 format, but the main grid is no longer "several unrelated
#      one-off blockers". Instead there are exactly four blockers arranged as a symmetric box around
#      the teal square:
#        * one horizontal pair above/below the square
#        * one vertical pair left/right of the square
#      Same-color opposite blockers are the portal pair.
#    - Trace example:
#        old task: north beam approaches the upper bar -> turns there in place
#        combo:    north beam approaches the upper bar -> jumps to the lower bar's inner side ->
#                  paints the reappearance square -> turns there using the legend
#      The teleport breaks the local continuity of the old path, while the symmetric placement makes
#      the pairing visible without hidden ordering.
# b) Complexity calibration:
#    - Only two obstacle colors are active in each instance, each shown twice. This is enough repeated
#      evidence for portal pairing while keeping the color logic human-trackable.
#    - The two active colors are forced to use opposite legend turns. Geometry is symmetric, but the
#      output path family is not reducible to pure mirror symmetry because the horizontal and vertical
#      portal pairs redirect differently.
#    - Exactly one beam axis gets a second portal hit (north/south or east/west, chosen per sample).
#      That keeps the variant visibly richer than "teleport once", but avoids tangled 3-hit loops.
# c) Old assumptions that break:
#    - Original hit handling assumed a color appears once and turning happens at the hit location.
#      Under this combo, a hit must identify which member of a same-color pair was entered, relocate
#      to the partner, and only then rotate.
#    - Original generation treated obstacle colors as distinct singleton semantics. Here a color's
#      meaning is defined jointly by its two symmetric copies.
#    - Original obstacle distribution did not matter beyond blocking a beam. Here symmetry is part of
#      the visual clue, so the generator must preserve congruent opposite pairs around the source.
# d) Placement legality:
#    - The four rectangles are built from one symmetric template around the teal square, so portals
#      are congruent and non-overlapping by construction.
#    - One span is "full" and reaches the orthogonal exit lanes, enabling exactly one second-hit axis;
#      the other span is "short" and deliberately misses those lanes, preventing unwanted third hits.
#    - The solver assumes portal exits land on background-only lanes. The generator enforces those
#      clear lanes through the full/short span choice and validates with the shared oracle.
# e) Visual observability:
#    - In the input, humans can directly see that every used obstacle color occurs twice in a symmetric
#      opposite-side layout, so same-color pairing is a visible hypothesis.
#    - In the output, beams disappear into one copy and resume beside the opposite copy, making the
#      teleport effect inferable from 2-3 examples.
# f) Visual cleanliness:
#    - The obstacles form a clean box-like arrangement with background margins around it. Portals never
#      overlap each other or the teal square, and beams paint only background cells.
# g) Color-rule inferability:
#    - No ordering by color id is used. Same-color repetition is the only pairing signal.
#    - Two active colors plus two unused legend colors keeps the legend format familiar while making
#      the opposite-turn relation between the active pairs easy to notice across samples.
# h) New solver assumptions and generator guarantees:
#    - `main()` assumes every used obstacle color forms exactly two congruent solid rectangles; the
#      generator constructs exactly that.
#    - `main()` assumes a beam footprint never overlaps mixed obstacle colors and that portal traces do
#      not cycle; the generator validates both with oracle simulation before accepting a sample.
#    - The combo-specific symmetry itself is produced by the generator rather than required by the
#      solver, so the solver stays focused on the visible portal rule.


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
    assert portals, "at least one portal pair must exist"
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


def _trace_plain_turn_beams(
    base: np.ndarray,
    turns: dict[int, str],
    bg: int,
    teal: int = 8,
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
            assert oc in turns, "obstacle color must be in the legend"
            d = _rot(d, turns[oc])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shoot four teal beams from the source square, but the symmetric obstacle pairs now act as portals.

    Input structure:
    - The top band is the same 6x24 grey legend with four colored vertical bars; each used color still encodes
      a left/right turn direction by bar placement inside its cell.
    - The main grid contains one solid teal square and four colored obstacle rectangles arranged symmetrically
      around it. Each used obstacle color appears exactly twice as an opposite-side pair.

    Rule:
    - Remove the legend from the output.
    - Launch a sz x sz teal beam in each cardinal direction, painting only background cells.
    - If the next step would overlap a colored obstacle, treat that obstacle as a portal: jump to the other
      rectangle of the same color, reappear on its inner side with the same alignment, paint the reappearance
      square, then rotate using the legend (left bar = counterclockwise, right bar = clockwise).
    - Continue until the beam would leave the grid. Portal rectangles themselves are never painted over.

    Key insight:
    - The crucial cue is the symmetric duplication of each active obstacle color. Same-color copies are paired
      portals, while the legend only determines how the beam turns after it emerges from the partner copy.
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

LEGEND_COLOR_POOL = [1, 2, 3, 4, 6]
BG_COLOR_POOL = [1, 2, 3, 4, 6, 7, 9]
LEGEND_BG = 5
TEAL = 8

GRID_H_RANGE = (19, 24)
GRID_W = 24
LEGEND_H = 6
LEGEND_CELL_STARTS = [1, 7, 13, 19]

TEAL_SZ2_PROB = 0.75
OBS_THICKNESS_RANGE = (2, 4)
PORTAL_GAP_RANGE = (3, 5)
OUTER_MARGIN = 1
MAX_RETRIES = 400


def _make_legend(legend_colors: list[int], turns: dict[int, str]) -> np.ndarray:
    legend = np.full((LEGEND_H, GRID_W), LEGEND_BG, dtype=int)
    for start, color in zip(LEGEND_CELL_STARTS, legend_colors):
        legend[1:5, start : start + 4] = 0
        bar_col = start if turns[color] == "L" else start + 3
        legend[1:5, bar_col] = color
    return legend


def _portal_rectangles(
    sr: int,
    sc: int,
    sz: int,
    t_ns: int,
    t_we: int,
    gap_ns: int,
    gap_we: int,
    h_ext: int,
    v_ext: int,
) -> dict[int, tuple[int, int, int, int]]:
    return {
        _N: (sr - gap_ns - t_ns, sc - h_ext, sr - gap_ns - 1, sc + sz - 1 + h_ext),
        _S: (sr + sz + gap_ns, sc - h_ext, sr + sz + gap_ns + t_ns - 1, sc + sz - 1 + h_ext),
        _W: (sr - v_ext, sc - gap_we - t_we, sr + sz - 1 + v_ext, sc - gap_we - 1),
        _E: (sr - v_ext, sc + sz + gap_we, sr + sz - 1 + v_ext, sc + sz + gap_we + t_we - 1),
    }


def _validate_rectangles(rects: dict[int, tuple[int, int, int, int]], h: int, w: int) -> bool:
    occupied = np.zeros((h, w), dtype=bool)
    for r0, c0, r1, c1 in rects.values():
        if not (0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w):
            return False
        region = occupied[r0 : r1 + 1, c0 : c1 + 1]
        if region.any():
            return False
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    h, w = random.randint(*GRID_H_RANGE), GRID_W
    sz = 2 if random.random() < TEAL_SZ2_PROB else 3
    t_ns = random.randint(*OBS_THICKNESS_RANGE)
    t_we = random.randint(*OBS_THICKNESS_RANGE)
    gap_ns = random.randint(*PORTAL_GAP_RANGE)
    gap_we = random.randint(*PORTAL_GAP_RANGE)

    min_r = OUTER_MARGIN + t_ns + gap_ns
    max_r = h - OUTER_MARGIN - t_ns - gap_ns - sz
    min_c = OUTER_MARGIN + t_we + gap_we
    max_c = w - OUTER_MARGIN - t_we - gap_we - sz
    if min_r > max_r or min_c > max_c:
        return None
    sr = random.randint(min_r, max_r)
    sc = random.randint(min_c, max_c)

    min_ext = 2 if sz == 2 else 1
    if gap_ns - 1 < min_ext or gap_we - 1 < min_ext:
        return None
    chain_axis = random.choice(["vertical", "horizontal"])
    if chain_axis == "vertical":
        h_ext = random.randint(min_ext, gap_we - 1)
        v_ext = gap_ns
    else:
        h_ext = gap_we
        v_ext = random.randint(min_ext, gap_ns - 1)

    rects = _portal_rectangles(sr, sc, sz, t_ns, t_we, gap_ns, gap_we, h_ext, v_ext)
    if not _validate_rectangles(rects, h, w):
        return None

    legend_colors = random.sample(LEGEND_COLOR_POOL, 4)
    ns_color, we_color = random.sample(legend_colors, 2)
    ns_turn = random.choice(["L", "R"])
    we_turn = "R" if ns_turn == "L" else "L"
    turns = {ns_color: ns_turn, we_color: we_turn}
    for color in legend_colors:
        turns.setdefault(color, random.choice(["L", "R"]))

    bg = random.choice([c for c in BG_COLOR_POOL if c not in legend_colors])
    base = np.full((h, w), bg, dtype=int)
    base[sr : sr + sz, sc : sc + sz] = TEAL
    for side, color in ((_N, ns_color), (_S, ns_color), (_W, we_color), (_E, we_color)):
        r0, c0, r1, c1 = rects[side]
        base[r0 : r1 + 1, c0 : c1 + 1] = color

    vals, cnts = np.unique(base, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        expected, hits = _trace_portal_beams(base, turns, bg, teal=TEAL)
        plain = _trace_plain_turn_beams(base, turns, bg, teal=TEAL)
    except AssertionError:
        return None

    hit_counts = [len(beam) for beam in hits]
    if chain_axis == "vertical":
        if hit_counts != [2, 1, 2, 1]:
            return None
    else:
        if hit_counts != [1, 2, 1, 2]:
            return None
    if np.array_equal(expected, plain):
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
    raise RuntimeError("failed to generate e87109e9 combo variant input")
