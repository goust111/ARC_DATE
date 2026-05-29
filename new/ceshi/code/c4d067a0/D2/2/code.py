# Variant of c4d067a0
# Dimension: D2 — 源对象：图例的结构与位置 (src)
# Change: 左上角稀疏像素模式（非背景色像素以偶数间隔排列在奇数行/列上），每个像素代表一个待放大的色块... → 多个图例精灵

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the D2 variant
# ----------------------------------
# a) Concrete behavior:
#    The original task has one sparse legend sprite and one block region. This
#    variant always has two separate sparse legend sprites in the left strip.
#    Each sprite is still a step-2 lattice of singleton pixels, but each sprite
#    now maps to its own translated block region. Input keeps two anchor blocks
#    per legend, so the solver must detect two legend components, infer two
#    independent bases, then paint both regions.
#    Small trace:
#      - Original: all legend pixels vote for one base -> fill one region.
#      - Variant: legend A votes for base A, legend B votes for base B -> fill
#        two different regions on the same block lattice.
#      - Difference: the "find one global offset" stage becomes "split legends,
#        then find one offset per legend".
# b) Complexity calibration:
#    The variant uses exactly two legends, not an unbounded number. That is
#    enough to make the multi-source behavior visible, while still leaving many
#    repeated cues: each legend keeps the same sparse-pixel language, each
#    region keeps the same block size and lattice step, and each legend has two
#    surviving anchor blocks.
# c) Hidden-assumption audit against the original solver:
#      - "all singles in the top-left belong to one legend": no longer valid.
#        Fix: normalize all singleton dots to the step-2 lattice, then split
#        them into coarse connected components.
#      - "all blocks share one offset": no longer valid.
#        Fix: keep one global block step, but infer a separate base per legend.
#      - "global voting over all legend pixels returns one unique base":
#        invalid in the variant. Fix: vote within each legend component only.
#      - "gcd of all block-position diffs gives the step": still valid only if
#        both regions stay on the same step lattice residue. The generator
#        guarantees that by placing both regions at positions
#        BASE_* + k * step.
# d) Placement legality:
#    Legends are stacked in the left strip with at least two coarse blank rows
#    between them, so they remain distinct after coarse component detection.
#    Block regions are placed on a shared coarse lattice in the main area; the
#    generator searches for two non-overlapping, bbox-separated region bases and
#    validates the result with main(input) == output.
# e) Visual observability:
#    In examples, a human sees two separate mini-sprites on the left and two
#    partially filled block areas on the right. The change is therefore visible
#    without reading color ids or hidden ordering.
# f) Visual cleanliness:
#    The left strip is reserved for legend sprites, and the right side contains
#    blocks only. Region bboxes are forced apart so the completed areas do not
#    interleave into a messy composite.
# g) Color inference:
#    Each legend gets its own disjoint palette. The two anchor colors of a
#    legend appear exactly once in that legend and are never reused in the
#    other legend. This gives unambiguous support for the per-legend mapping
#    without requiring invisible color-based sorting.


def _split_objects(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for pts in components_mask(g != bg, connectivity=4):
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), col))
    return singles, blocks


def _infer_legend_components(
    singles: list[tuple[int, int, int]]
) -> tuple[list[list[tuple[int, int, int]]], int]:
    rs = sorted({r for r, _, _ in singles})
    cs = sorted({c for _, c, _ in singles})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    stepL = reduce(gcd, drs + dcs)
    assert stepL > 0, "legend lattice step must be positive"

    rmin = min(r for r, _, _ in singles)
    cmin = min(c for _, c, _ in singles)
    coarse = [((r - rmin) // stepL, (c - cmin) // stepL, v) for r, c, v in singles]
    max_r = max(rr for rr, _, _ in coarse)
    max_c = max(cc for _, cc, _ in coarse)
    mask = np.zeros((max_r + 1, max_c + 1), dtype=bool)
    color_at: dict[tuple[int, int], int] = {}
    for rr, cc, v in coarse:
        mask[rr, cc] = True
        color_at[(rr, cc)] = int(v)

    legends: list[list[tuple[int, int, int]]] = []
    for comp in components_mask(mask, connectivity=4):
        rr0, cc0, rr1, cc1 = bbox_coords(comp)
        pts: list[tuple[int, int, int]] = []
        for rr, cc in comp:
            pts.append((int(rr - rr0), int(cc - cc0), color_at[(int(rr), int(cc))]))
        pts.sort()
        legends.append(pts)

    legends.sort(key=lambda pts: (min(r for r, _, _ in pts), min(c for _, c, _ in pts), len(pts)))
    return legends, stepL


def _infer_block_step(blocks: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks2 = [b for b in blocks if b[2] == block_size]
    assert blocks2, "must contain at least one block of the dominant block size"

    tops = sorted({r0 for r0, _, _, _ in blocks2})
    lefts = sorted({c0 for _, c0, _, _ in blocks2})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    assert step_candidates, "block step needs at least two aligned blocks"
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"
    return block_size, step


def _infer_base(
    legend_pts: list[tuple[int, int, int]],
    by_color: dict[int, list[tuple[int, int]]],
    step: int,
) -> tuple[int, int]:
    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        for br0, bc0 in by_color.get(col, []):
            base = (int(br0 - rr * step), int(bc0 - cc * step))
            bases[base] = bases.get(base, 0) + 1
    assert bases, "existing blocks must determine each legend offset"
    max_votes = max(bases.values())
    best = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best) == 1, "each legend must map to one unique block offset"
    return best[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete multiple block regions by reading multiple sparse legend sprites.

    Input structure:
    - bg = most common color.
    - Left side: two separate legend sprites made of singleton pixels on the
      same step-2 lattice, each sprite forming its own tiny staircase-like map.
    - Main area: uniformly colored square blocks (NxN) on one shared block
      lattice; each legend keeps only two anchor blocks in the input.

    Rule:
    - Treat each disconnected legend sprite as an independent minimap.
    - Use the existing anchor blocks of matching colors to infer that sprite's
      block-region offset on the shared lattice.
    - For every pixel in every legend sprite, paint the corresponding NxN block
      at that sprite's inferred offset.

    Key insight: The task is no longer "one legend -> one region"; the solver
    must first separate the singleton pixels into multiple legend sprites, then
    solve the same scale-and-translate relation independently for each sprite.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    singles, blocks = _split_objects(g, bg)
    assert singles and blocks, "grid must contain legend pixels and main blocks"

    legends, _stepL = _infer_legend_components(singles)
    assert len(legends) >= 2, "variant requires multiple legend sprites"

    block_size, step = _infer_block_step(blocks)
    blocks = [b for b in blocks if b[2] == block_size]

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r0, c0, _s, col in blocks:
        by_color.setdefault(int(col), []).append((int(r0), int(c0)))

    for legend_pts in legends:
        base_r, base_c = _infer_base(legend_pts, by_color, step)
        for rr, cc, col in legend_pts:
            tr0 = base_r + rr * step
            tc0 = base_c + cc * step
            assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
                "legend block projection must fit in canvas"
            )
            out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


# --------------- generator (variant) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10
LEGEND_COUNT = 2

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (2, 3)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 4, 6
LEGEND_GAP_COARSE = 2
LEGEND_COL_SHIFT_CHOICES = (0, 1)
PALETTE_SIZE_CHOICES = (3, 4)

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 4)
GAP_RANGE = range(1, 3)
STEP_MIN, STEP_MAX = 3, 4
BASE_R_MIN = 1
BASE_C_MIN = 8
REGION_BBOX_GAP = 1

# === CFG: generation ===
MAX_RETRIES = 400


def _enumerate_staircase_shapes():
    shapes = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix, remaining, r):
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            widths = tuple(prefix)
                            if any(w >= 2 for w in widths):
                                shapes.append(widths)
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)

                rec([], count, 0)
    return shapes


_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE if STEP_MIN <= n + g <= STEP_MAX
]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return pts, pairs


def _assign_legend_colors(
    pts: list[tuple[int, int]],
    anchor_pair: tuple[int, int],
    palette: list[int],
) -> dict[tuple[int, int], int] | None:
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    remaining_pts = [p for p in pts if p not in {first, second}]
    if len(palette) < 3 or len(remaining_pts) < len(palette) - 2:
        return None

    a, b = random.sample(palette, 2)
    rest_cols = [col for col in palette if col not in {a, b}]
    random.shuffle(rest_cols)
    random.shuffle(remaining_pts)

    colors = {first: a, second: b}
    for pos, col in zip(remaining_pts, rest_cols):
        colors[pos] = col
    for pos in remaining_pts[len(rest_cols) :]:
        colors[pos] = random.choice(rest_cols)
    return colors


def _bbox_from_pts(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    return min(rs), min(cs), max(rs), max(cs)


def _shift_pts(pts: list[tuple[int, int]], base: tuple[int, int]) -> set[tuple[int, int]]:
    br, bc = base
    return {(br + r, bc + c) for r, c in pts}


def _bbox_gap_ok(
    base1: tuple[int, int],
    bbox1: tuple[int, int, int, int],
    base2: tuple[int, int],
    bbox2: tuple[int, int, int, int],
) -> bool:
    a_r0, a_c0, a_r1, a_c1 = bbox1
    b_r0, b_c0, b_r1, b_c1 = bbox2
    a_r0 += base1[0]
    a_r1 += base1[0]
    a_c0 += base1[1]
    a_c1 += base1[1]
    b_r0 += base2[0]
    b_r1 += base2[0]
    b_c0 += base2[1]
    b_c1 += base2[1]
    row_gap = max(b_r0 - a_r1 - 1, a_r0 - b_r1 - 1, 0)
    col_gap = max(b_c0 - a_c1 - 1, a_c0 - b_c1 - 1, 0)
    return row_gap >= REGION_BBOX_GAP or col_gap >= REGION_BBOX_GAP


def _find_region_bases(
    size: int,
    block_size: int,
    step: int,
    shape_pts_list: list[list[tuple[int, int]]],
) -> list[tuple[int, int]] | None:
    max_top = size - block_size
    max_br = (max_top - BASE_R_MIN) // step
    max_bc = (max_top - BASE_C_MIN) // step
    if max_br < 0 or max_bc < 0:
        return None

    bboxes = [_bbox_from_pts(pts) for pts in shape_pts_list]
    candidate_lists: list[list[tuple[int, int]]] = []
    for pts, bbox in zip(shape_pts_list, bboxes):
        _r0, _c0, r1, c1 = bbox
        cands = []
        for br in range(max_br - r1 + 1):
            for bc in range(max_bc - c1 + 1):
                cands.append((br, bc))
        random.shuffle(cands)
        candidate_lists.append(cands)

    valid_pairs = []
    for base1 in candidate_lists[0]:
        cells1 = _shift_pts(shape_pts_list[0], base1)
        for base2 in candidate_lists[1]:
            cells2 = _shift_pts(shape_pts_list[1], base2)
            if cells1 & cells2:
                continue
            if not _bbox_gap_ok(base1, bboxes[0], base2, bboxes[1]):
                continue
            valid_pairs.append((base1, base2))

    if not valid_pairs:
        return None
    base_pair = random.choice(valid_pairs)
    out = []
    for br, bc in base_pair:
        out.append((BASE_R_MIN + br * step, BASE_C_MIN + bc * step))
    return out


def _place_legends(
    legend_pts_list: list[list[tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    placed: list[list[tuple[int, int]]] = []
    row_offset = 0
    for pts in legend_pts_list:
        col_offset = random.choice(LEGEND_COL_SHIFT_CHOICES)
        shifted = [(row_offset + r, col_offset + c) for r, c in pts]
        placed.append(shifted)
        max_r = max(r for r, _ in shifted)
        row_offset = max_r + 1 + LEGEND_GAP_COARSE
    return placed


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    bg = random.randrange(NUM_COLORS)

    legend_shapes = []
    anchor_pairs = []
    for _ in range(LEGEND_COUNT):
        pts, pairs = _pick_staircase()
        legend_shapes.append(pts)
        anchor_pairs.append(random.choice(pairs))

    placed_legends = _place_legends(legend_shapes)
    max_legend_c = max(c for pts in placed_legends for _, c in pts)
    if LEGEND_ORIGIN[1] + max_legend_c * LEGEND_STEP >= BASE_C_MIN - 1:
        return None
    max_legend_r = max(r for pts in placed_legends for r, _ in pts)
    if LEGEND_ORIGIN[0] + max_legend_r * LEGEND_STEP >= size:
        return None

    available = [c for c in range(NUM_COLORS) if c != bg]
    palette_sizes = []
    for pts in legend_shapes:
        palette_sizes.append(random.randint(3, min(max(PALETTE_SIZE_CHOICES), len(pts))))
    if sum(palette_sizes) > len(available):
        return None
    random.shuffle(available)

    legend_colors_list: list[dict[tuple[int, int], int]] = []
    at = 0
    for pts, anchor_pair, k in zip(legend_shapes, anchor_pairs, palette_sizes):
        palette = available[at : at + k]
        at += k
        colors = _assign_legend_colors(pts, anchor_pair, palette)
        if colors is None:
            return None
        legend_colors_list.append(colors)

    region_bases = _find_region_bases(size, block_size, step, legend_shapes)
    if region_bases is None:
        return None

    out = np.full((size, size), bg, dtype=int)

    for placed_pts, colors in zip(placed_legends, legend_colors_list):
        for rr, cc in placed_pts:
            local_r = rr - min(r for r, _ in placed_pts)
            local_c = cc - min(c for _, c in placed_pts)
            out[
                LEGEND_ORIGIN[0] + rr * LEGEND_STEP,
                LEGEND_ORIGIN[1] + cc * LEGEND_STEP,
            ] = colors[(local_r, local_c)]

    for pts, colors, (base_r, base_c) in zip(legend_shapes, legend_colors_list, region_bases):
        for rr, cc in pts:
            col = colors[(rr, cc)]
            br = base_r + rr * step
            bc = base_c + cc * step
            out[br : br + block_size, bc : bc + block_size] = col

    inp = out.copy()
    for pts, colors, anchor_pair, (base_r, base_c) in zip(
        legend_shapes, legend_colors_list, anchor_pairs, region_bases
    ):
        keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
        for rr, cc in pts:
            if (rr, cc) in keep:
                continue
            br = base_r + rr * step
            bc = base_c + cc * step
            inp[br : br + block_size, bc : bc + block_size] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 D2 variant input")
