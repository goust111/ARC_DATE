# Combo variant of c4d067a0 (index 5)
# combo_spec: D2/2+D5/0
# Name: 多图例精灵+图例消失

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    The original task has one sparse legend in the corner, one block region,
#    and the legend survives into the output. This combo uses exactly two
#    separate sparse legends in the left strip. Each legend still describes a
#    staircase-like set of lattice positions, but each one controls its own
#    translated block region on the right. After the missing blocks are filled,
#    both legends are erased back to background.
#    Small trace:
#      - Original: legend L votes for one base B -> paint one region -> keep L.
#      - Combo: legends L1 and L2 are split first; L1 votes for B1 and L2 votes
#        for B2 -> paint two regions -> erase all legend singleton cells.
#      - Difference: the solver must both branch the mapping by legend component
#        and invert the source trajectory from "keep" to "remove".
# b) Complexity calibration:
#    The variant fixes the legend count at two rather than an open-ended number.
#    That keeps the multi-source rule obvious in a few examples while avoiding
#    combinatorial ambiguity. Both legends share the same sprite language and
#    the same block step, so the only new freedoms are "which legend maps to
#    which region" and "the legends disappear afterward".
# c) Hidden-assumption audit against the original solver:
#      - "all singleton pixels belong to one legend" is false now.
#        Fix: normalize singleton pixels to the step-2 lattice and split them
#        into coarse connected components.
#      - "one global base offset explains every legend pixel" is false now.
#        Fix: infer one base per legend component.
#      - "copy input, add missing blocks, done" is false now because the source
#        is no longer preserved.
#        Fix: explicitly erase all singleton legend pixels in the output.
#      - "global voting over all legend pixels yields one best base" is no
#        longer safe; voting must be restricted within each legend.
# d) Placement legality:
#    Legends are stacked with blank coarse rows between them so coarse-component
#    splitting is stable. Block regions are placed on one shared block lattice,
#    constrained to be non-overlapping and bbox-separated. The generator works
#    output-first and only accepts cases where main(input) reproduces output.
# e) Visual observability:
#    A human can see two mini-sprites in the input, two partially populated
#    block regions on the right, and then both left-side sprites vanish in the
#    output while the regions are completed. The fusion therefore shows both
#    dimensions directly in the pixels.
# f) Visual cleanliness:
#    The left strip is reserved for legends and the right side for blocks. The
#    output is visually cleaner than the input because the left-side markers are
#    removed, so there is no overlap between "instruction" and "result" areas.
# g) Change inferability:
#    Each legend receives a palette disjoint from the other legend, and the two
#    surviving anchor colors of a legend are unique inside that legend. This
#    makes the per-legend mapping recoverable from repeated structure rather
#    than from invisible color ordering or arbitrary tie-breaking.


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
    diffs = drs + dcs
    assert diffs, "legend pixels need at least one lattice difference"
    stepL = reduce(gcd, diffs)
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
        rr0, cc0, _rr1, _cc1 = bbox_coords(comp)
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
    kept = [b for b in blocks if b[2] == block_size]
    assert kept, "must contain at least one block of the dominant block size"

    tops = sorted({r0 for r0, _, _, _ in kept})
    lefts = sorted({c0 for _, c0, _, _ in kept})
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
    Complete multiple block regions from multiple temporary legend sprites, then remove the legends.

    Input structure:
    - bg = most common color.
    - Left side: two disconnected legend sprites made of singleton colored
      pixels on the same sparse step-2 lattice; each sprite is its own tiny
      staircase-like minimap.
    - Main area: uniformly colored square blocks (NxN) placed on one shared
      block lattice; each legend contributes only two surviving anchor blocks.

    Rule:
    - Split the singleton pixels into separate legend sprites and treat each
      sprite as an independent minimap.
    - For each legend, use its anchor-block colors to infer that legend's block
      region offset on the shared lattice, then paint every missing block from
      that legend.
    - After all regions are completed, erase the legend singleton pixels back
      to the background color instead of keeping them.

    Key insight: The single-pixel sprites are temporary instructions, not part
    of the final picture. Solving now requires both multi-source alignment
    (one offset per legend) and a final cleanup step that removes the legends.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
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

    out = g.copy()
    for r, c, _col in singles:
        out[r, c] = bg

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


# Solver-side implicit assumptions introduced by the combo
# --------------------------------------------------------
# 1. Every singleton non-background cell belongs to a legend sprite.
# 2. All legend sprites share one lattice step, so a global gcd recovers step 2.
# 3. All block regions share one block step, though their base offsets differ.
# 4. Each legend has enough surviving color evidence to vote for a unique base.
# The generator below enforces these assumptions explicitly.


# --------------- generator (combo variant) ---------------

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


def _enumerate_staircase_shapes() -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix: list[int], remaining: int, r: int) -> None:
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols and any(w >= 2 for w in prefix):
                            shapes.append(tuple(prefix))
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


def _pick_staircase() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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
    for bbox in bboxes:
        _r0, _c0, r1, c1 = bbox
        cands = [(br, bc) for br in range(max_br - r1 + 1) for bc in range(max_bc - c1 + 1)]
        random.shuffle(cands)
        candidate_lists.append(cands)

    valid_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
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

    bases = random.choice(valid_pairs)
    return [(BASE_R_MIN + br * step, BASE_C_MIN + bc * step) for br, bc in bases]


def _place_legends(
    legend_pts_list: list[list[tuple[int, int]]]
) -> list[list[tuple[int, int]]]:
    placed: list[list[tuple[int, int]]] = []
    row_offset = 0
    for pts in legend_pts_list:
        col_offset = random.choice(LEGEND_COL_SHIFT_CHOICES)
        shifted = [(row_offset + r, col_offset + c) for r, c in pts]
        placed.append(shifted)
        row_offset = max(r for r, _ in shifted) + 1 + LEGEND_GAP_COARSE
    return placed


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    bg = random.randrange(NUM_COLORS)

    legend_shapes: list[list[tuple[int, int]]] = []
    anchor_pairs: list[tuple[int, int]] = []
    for _ in range(LEGEND_COUNT):
        pts, pairs = _pick_staircase()
        legend_shapes.append(pts)
        anchor_pairs.append(random.choice(pairs))

    placed_legends = _place_legends(legend_shapes)
    max_legend_c = max(c for pts in placed_legends for _, c in pts)
    max_legend_r = max(r for pts in placed_legends for r, _ in pts)
    if LEGEND_ORIGIN[1] + max_legend_c * LEGEND_STEP >= BASE_C_MIN - 1:
        return None
    if LEGEND_ORIGIN[0] + max_legend_r * LEGEND_STEP >= size:
        return None

    available = [c for c in range(NUM_COLORS) if c != bg]
    palette_sizes = [
        random.randint(3, min(max(PALETTE_SIZE_CHOICES), len(pts))) for pts in legend_shapes
    ]
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
    for pts, colors, (base_r, base_c) in zip(legend_shapes, legend_colors_list, region_bases):
        for rr, cc in pts:
            col = colors[(rr, cc)]
            br = base_r + rr * step
            bc = base_c + cc * step
            out[br : br + block_size, bc : bc + block_size] = col

    inp = out.copy()
    for local_pts, placed_pts, colors in zip(legend_shapes, placed_legends, legend_colors_list):
        for (rr, cc), (pr, pc) in zip(local_pts, placed_pts):
            inp[LEGEND_ORIGIN[0] + pr * LEGEND_STEP, LEGEND_ORIGIN[1] + pc * LEGEND_STEP] = colors[(rr, cc)]

    for pts, anchor_pair, (base_r, base_c) in zip(legend_shapes, anchor_pairs, region_bases):
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


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
