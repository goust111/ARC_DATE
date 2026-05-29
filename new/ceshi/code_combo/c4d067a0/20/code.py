# Combo variant of c4d067a0 (index 20)
# combo_spec: D2/3+D9/1
# Name: 边缘像素行图例+不均匀间距

import itertools
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg

# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   grid:
#     size: 23-27 square, one uniform background color
#   legend:
#     form: exactly one top-edge row of singleton pixels
#     placement:
#       - legend lies on row 0
#       - first pixel starts at col 1
#       - center-to-center gaps are 2 or 3 cells, and both gap sizes appear
#     meaning:
#       - the visible empty run between consecutive legend pixels is the true 1D lattice step
#       - if legend columns are [1,3,6,8], the inferred axis offsets are [0,1,3,4]
#         because the blank runs are [1,2,1]
#   expanded_object:
#     shape: upper-right staircase with cells (r,c) where 0 <= r <= c < n
#     colors: column c uses the c-th legend color
#     coordinates:
#       - row/col axis positions both reuse the same non-uniform offset list from the 1D legend
#       - this is the fusion point: one irregular edge line controls a 2D irregular staircase
#   blocks:
#     block_size: 2-3
#     basic_step: 3-4
#     actual top-left difference between neighboring staircase levels:
#       - offset_gap 1 -> one basic_step
#       - offset_gap 2 -> two basic_steps
#     input_count: exactly 2 kept blocks at staircase cells (0,0) and (0,1)
#
# a) Concrete behavior trace
#    Original task:
#      - every legend singleton already had its own (row,col) lattice coordinate.
#      - scaling only enlarged that ready-made 2D lattice into square blocks.
#    New fused task:
#      - the legend is only a top border row, for example colors [A,B,C,D] at columns [1,3,6,8].
#      - the visible gap pattern is irregular: blank runs [1,2,1], so axis offsets become [0,1,3,4].
#      - the 1D row first expands into staircase cells:
#          col 0 -> (0,0) color A
#          col 1 -> (0,1),(1,1) color B
#          col 2 -> (0,2),(1,2),(2,2) color C
#          col 3 -> (0,3),(1,3),(2,3),(3,3) color D
#      - those staircase row/col indices are then mapped through the non-uniform offsets [0,1,3,4],
#        so the enlarged blocks no longer sit on a uniform grid.
#    Difference:
#      - D2 changes "what the legend describes" from a 2D sprite to a 1D sequence.
#      - D9 changes "where the described cells land" from equal pitch to unequal pitch.
#      - combined, the solver must reconstruct hidden 2D cells and hidden irregular coordinates.
#
# b) Complexity calibration
#    - Legend length stays at 4-5, so the 1D order is easy to read.
#    - Gap values use only two sizes (1 or 2 blank cells), which is visually obvious but not noisy.
#    - The staircase still has 10 or 15 blocks, enough repetition for a human to notice that
#      wider legend gaps cause wider block spacing along both axes.
#
# c) Old assumptions that break
#    - Original legend normalization used a single gcd lattice step; that would erase whether the
#      visible legend gap was "small" or "large". This variant must preserve each individual gap.
#    - Original block placement assumed one global block pitch. Here only a base pitch exists; actual
#      row/col deltas are multiples of that pitch according to the legend's uneven spacing.
#    - Original voting directly matched legend pixels to blocks. Here a legend pixel first expands to
#      a staircase column, so solver matching must use expanded 2D points plus irregular offsets.
#
# d) Placement legality
#    - Legend pixels live on row 0 only, so blocks start several rows below and never overlap the legend.
#    - Unit-gap sums are capped so the largest irregular staircase still fits inside a 23-27 canvas.
#    - Block_size and basic_step are chosen so even the tightest offset gap leaves a positive separator.
#
# e) Visual observability
#    - The top row visibly has uneven spacing; this is directly observable without reading color ids.
#    - The solved output visibly has matching uneven spacing between block columns and block rows.
#    - Because only the first two blocks are present in the input, the solver must use both the color
#      order and the irregular spacing cue from the legend to reconstruct the whole staircase.
#
# f) Visual cleanliness
#    - Only three layers are present: background, edge singletons, and solid square blocks.
#    - The legend stays isolated on the border; the staircase occupies one coherent interior region.
#    - No extra markers or overlapping helper shapes are added.
#
# g) Change inferability
#    - Colors are copied directly from the legend; there is no hidden remapping.
#    - The meaningful variable is spacing, not color value: repeated block columns show that all cells
#      in a column share one legend color, while the non-uniform offsets show where columns/rows land.
#    - Generator validation rejects samples where the two anchors would allow multiple interpretations.


def _extract_objects(g: np.ndarray, bg: int):
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


def _read_edge_legend(
    singles: list[tuple[int, int, int]], h: int, w: int
) -> tuple[list[int], list[int], list[int]]:
    win_r = min(h, 6)
    win_c = min(w, 16)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "grid must contain a top-edge legend in the upper-left window"

    rows = sorted({r for r, _, _ in legend})
    assert rows == [0], "legend must be a single top-edge row"
    cols = sorted(c for _r, c, _v in legend)
    assert len(cols) >= 2, "legend must contain at least two pixels"

    color_by_col = {int(c): int(v) for _r, c, v in legend}
    assert len(color_by_col) == len(cols), "legend columns must be unique"
    for i in range(len(cols) - 1):
        assert cols[i + 1] > cols[i], "legend columns must increase strictly"

    unit_gaps = [cols[i + 1] - cols[i] - 1 for i in range(len(cols) - 1)]
    assert unit_gaps and min(unit_gaps) > 0, "legend pixels must be separated by blanks"
    offsets = [0]
    for gap in unit_gaps:
        offsets.append(offsets[-1] + int(gap))

    seq = [color_by_col[c] for c in cols]
    return cols, offsets, seq


def _expand_staircase(seq: list[int]) -> list[tuple[int, int, int]]:
    pts: list[tuple[int, int, int]] = []
    for cc, col in enumerate(seq):
        for rr in range(cc + 1):
            pts.append((int(rr), int(cc), int(col)))
    return pts


def _infer_candidate(
    b1: tuple[int, int, int, int],
    b2: tuple[int, int, int, int],
    p1: tuple[int, int, int],
    p2: tuple[int, int, int],
    offsets: list[int],
):
    br1, bc1, _s1, _col1 = b1
    br2, bc2, _s2, _col2 = b2
    rr1, cc1, _pc1 = p1
    rr2, cc2, _pc2 = p2
    dr_cells = offsets[rr2] - offsets[rr1]
    dc_cells = offsets[cc2] - offsets[cc1]
    dr = br2 - br1
    dc = bc2 - bc1

    scales: list[int] = []
    if dr_cells == 0:
        if dr != 0:
            return None
    else:
        if dr <= 0 or dr % dr_cells != 0:
            return None
        scales.append(dr // dr_cells)
    if dc_cells == 0:
        if dc != 0:
            return None
    else:
        if dc <= 0 or dc % dc_cells != 0:
            return None
        scales.append(dc // dc_cells)
    if not scales:
        return None
    scale = scales[0]
    if any(s != scale for s in scales) or scale <= 0:
        return None

    base_r = br1 - offsets[rr1] * scale
    base_c = bc1 - offsets[cc1] * scale
    return int(scale), int(base_r), int(base_c)


def _validate_mapping(
    blocks: list[tuple[int, int, int, int]],
    legend_pts: list[tuple[int, int, int]],
    offsets: list[int],
    block_size: int,
    scale: int,
    base_r: int,
    base_c: int,
):
    allowed = {
        (offsets[rr], offsets[cc]): int(col) for rr, cc, col in legend_pts
    }
    for r0, c0, size, col in blocks:
        if size != block_size:
            return False
        dr = r0 - base_r
        dc = c0 - base_c
        if dr < 0 or dc < 0 or dr % scale != 0 or dc % scale != 0:
            return False
        key = (dr // scale, dc // scale)
        if key not in allowed or allowed[key] != int(col):
            return False
    return True


def _infer_scale_and_base(
    blocks: list[tuple[int, int, int, int]],
    legend_pts: list[tuple[int, int, int]],
    offsets: list[int],
    block_size: int,
) -> tuple[int, int, int]:
    unique: dict[tuple[int, int, int], None] = {}
    for b1, b2 in itertools.permutations(blocks, 2):
        for p1, p2 in itertools.permutations(legend_pts, 2):
            if b1[3] != p1[2] or b2[3] != p2[2]:
                continue
            cand = _infer_candidate(b1, b2, p1, p2, offsets)
            if cand is None:
                continue
            scale, base_r, base_c = cand
            if _validate_mapping(blocks, legend_pts, offsets, block_size, scale, base_r, base_c):
                unique[(scale, base_r, base_c)] = None
    assert unique, "existing blocks must determine a valid irregular projection"
    assert len(unique) == 1, "irregular legend projection must be unique"
    return next(iter(unique))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete an enlarged staircase by combining a 1D edge legend with uneven spacing.

    Input structure:
    - The background is the most frequent color.
    - Along the top edge, a single row of isolated pixels forms the legend; the blank runs between neighboring pixels are not all equal.
    - In the main area, only two same-size solid square blocks of the larger pattern are already present.

    Rule:
    - Read the legend left-to-right as a 1D color sequence, and also keep each individual blank run as a spacing unit.
    - Expand that sequence into an upper-right staircase where legend entry c colors every cell in staircase column c.
    - Use the legend's non-uniform spacing units on both staircase axes, infer the interior scale and base offset from the existing blocks, then place every missing block.

    Key insight:
    - The border row is not merely a color list: its irregular gaps define the hidden coordinate system of the enlarged 2D staircase, so solving requires recovering both the staircase structure and its uneven axis spacing.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    singles, blocks = _extract_objects(g, bg)
    assert singles and blocks, "grid must contain legend pixels and enlarged blocks"

    _cols, offsets, seq = _read_edge_legend(singles, h, w)
    legend_pts = _expand_staircase(seq)

    sizes = [s for _r0, _c0, s, _col in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert len(blocks) >= 2, "at least two anchor blocks are required"

    scale, base_r, base_c = _infer_scale_and_base(blocks, legend_pts, offsets, block_size)

    for rr, cc, col in legend_pts:
        tr0 = base_r + offsets[rr] * scale
        tc0 = base_c + offsets[cc] * scale
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, \
            "legend block projection must fit in canvas"
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return out


# --------------- generator (o5x) ---------------

# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: edge legend ===
LEGEND_ROW = 0
LEGEND_START_COL = 1
LEGEND_LENGTHS = (4, 5)
LEGEND_GAP_VALUES = (1, 2)  # blank cells between neighboring legend pixels
LEGEND_MAX_TOTAL_OFFSET = 5
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 3

# === CFG: block area ===
BLOCK_SIZE_RANGE = (2, 3)
BASIC_STEP_RANGE = (3, 4)
BASE_R_MIN = 3
BASE_C_MIN = 2

# === CFG: generation ===
MAX_RETRIES = 500


def _random_color_counts(total: int, k: int):
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        i = random.choice(open_idxs)
        counts[i] += 1
        remaining -= 1
    random.shuffle(counts)
    return counts


def _build_sequence(length: int, palette: list[int]):
    counts = dict(zip(palette, _random_color_counts(length, len(palette))))
    seq: list[int] = []

    def search():
        i = len(seq)
        if i == length:
            return all(v == 0 for v in counts.values())
        options = [col for col in palette if counts[col] > 0]
        if i == 1:
            options = [col for col in options if col != seq[0]]
        random.shuffle(options)
        for col in options:
            if i >= 2 and (seq[i - 1], col) == (seq[0], seq[1]):
                continue
            seq.append(col)
            counts[col] -= 1
            if search():
                return True
            counts[col] += 1
            seq.pop()
        return False

    if not search():
        return None
    return seq


def _build_gap_pattern(length: int):
    gap_count = length - 1
    choices = []
    for bits in itertools.product(LEGEND_GAP_VALUES, repeat=gap_count):
        if len(set(bits)) < 2:
            continue
        if sum(bits) > LEGEND_MAX_TOTAL_OFFSET:
            continue
        choices.append(tuple(int(x) for x in bits))
    return random.choice(choices) if choices else None


def _legend_columns_from_gaps(gaps: tuple[int, ...]) -> list[int]:
    cols = [LEGEND_START_COL]
    for gap in gaps:
        cols.append(cols[-1] + int(gap) + 1)
    return cols


def _offsets_from_gaps(gaps: tuple[int, ...]) -> list[int]:
    offsets = [0]
    for gap in gaps:
        offsets.append(offsets[-1] + int(gap))
    return offsets


def _staircase_points(seq: list[int]):
    pts: dict[tuple[int, int], int] = {}
    for cc, col in enumerate(seq):
        for rr in range(cc + 1):
            pts[(rr, cc)] = int(col)
    keep = {(0, 0), (0, 1)}
    return pts, keep


def _can_fit(size: int, base_min: int, max_offset: int, step: int, block_size: int):
    return base_min + max_offset * step + block_size <= size


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    length = random.choice(LEGEND_LENGTHS)
    gaps = _build_gap_pattern(length)
    if gaps is None:
        return None
    cols = _legend_columns_from_gaps(gaps)
    offsets = _offsets_from_gaps(gaps)
    max_offset = offsets[-1]

    block_size = random.choice(BLOCK_SIZE_RANGE)
    step_options = [step for step in BASIC_STEP_RANGE if step > block_size]
    random.shuffle(step_options)
    step = None
    for cand in step_options:
        if _can_fit(size, BASE_R_MIN, max_offset, cand, block_size) and \
                _can_fit(size, BASE_C_MIN, max_offset, cand, block_size):
            step = cand
            break
    if step is None:
        return None

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (length + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, length))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k)
    seq = _build_sequence(length, palette)
    if seq is None:
        return None

    pts, keep = _staircase_points(seq)
    r_hi = size - block_size - max_offset * step
    c_hi = size - block_size - max_offset * step
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    out = np.full((size, size), bg, dtype=int)
    for c, col in zip(cols, seq):
        out[LEGEND_ROW, c] = col

    for (rr, cc), col in pts.items():
        br = base_r + offsets[rr] * step
        bc = base_c + offsets[cc] * step
        out[br : br + block_size, bc : bc + block_size] = col

    inp = out.copy()
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        br = base_r + offsets[rr] * step
        bc = base_c + offsets[cc] * step
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
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
