# Combo variant of b99e7126 (index 6)
# combo_spec: D3/3+D14/0
# Name: 多次叠加 + 环绕式边界
#
# Design analysis:
# - concrete_behavior_and_trace:
#   - Original task: infer one 3x3 enlarged placement of the minority cell's inner mask inside the 7x7
#     cell lattice, then fill only that one placement.
#   - Combo task: infer one toroidal base placement `(top,left)` from the shown minority cells, but the
#     final output uses four related placements at offsets `(0,0),(0,2),(2,0),(2,2)` from that base.
#     These are the four corners of a 5x5 toroidal window in cell coordinates. Because each placement is
#     3x3, neighboring placements overlap by one row and/or one column. If `top=4,left=5`, then the four
#     anchors are `(4,5),(4,0),(6,5),(6,0)` modulo 7; the lower/right anchors wrap, and the overlap bands
#     run through wrapped edge cells. The output is the OR-union of all active hits across those four
#     wrapped placements.
#   - Difference from original: uniqueness no longer refers to one in-bounds rectangle. It refers to one
#     toroidal 2x2 corner class, and the final repaint is built from the union of four wrapped blocks.
# - complexity_calibration:
#   - Using exactly four anchors keeps the multi-placement effect obvious without saturating the board.
#   - Offset step 2 is deliberate: it causes visible overlap/interference, unlike step 4 which would make
#     four disconnected copies and weaken the interaction between D3 and D14.
#   - The base anchor is restricted to the sparse toroidal lattice `{0,2,4} x {0,2,4}`. That leaves 9
#     legal four-anchor classes: enough variety for examples, but small enough that 3-4 given cells can
#     reliably identify the class after generator filtering.
# - implicit_assumptions_review:
#   - Divider inference and cell-pattern extraction stay valid because the grid format is unchanged.
#   - Original code assumed candidate blocks were contiguous non-wrapping slices; the combo must search all
#     7x7 toroidal anchors and use modulo addressing for every touched cell.
#   - Original code assumed one enlarged block owned every shown minority cell. Here the input still samples
#     givens from one concrete instance, but the solver must map that instance to a base class that expands
#     into four placements.
#   - Original code could write block contents directly because one placement had no inter-block conflicts.
#     That fails here: inactive pixels from one placement must not erase active pixels from another. The new
#     solver computes the global active union first, then repaints cells once.
# - placement_legality:
#   - Each single 3x3 placement on a 7x7 torus has 9 distinct target cells. The four placements may overlap
#     only because their 3x3 footprints intersect by one row/column; no invalid out-of-range coordinates
#     occur because every access is reduced modulo 7.
#   - Generator always chooses a legal base whose four-anchor class wraps on the row axis, column axis, or
#     both, so the D14 behavior is guaranteed to appear in the output.
#   - Inputs are built output-first and are accepted only if `main(input)` reproduces the intended output.
# - visual_observability_and_cleanliness:
#   - Across examples, the output consistently looks like a 5x5 toroidal region whose four corners contain
#     the same enlarged motif; overlap bands create stronger minority traces near the middle seams.
#   - Only majority/minority cell interiors are used, so even when four placements interfere, the final
#     board still contains only two cell textures plus divider lines. The picture is denser than the base
#     task but remains legible.
#   - No rule depends on color IDs or hidden ordering. The only color logic remains "dominant color inside
#     the minority cell defines the active mask", which is directly visible.
# - generator_checks_for_new_solver_assumptions:
#   - `main()` now assumes there is exactly one toroidal base class such that the shown minority cells can
#     all come from one of its four concrete anchors.
#   - `main()` also assumes the minority cell's dominant color is the active mask color and that the input
#     contains exactly two recurring cell interiors. Generator preserves both conditions.
#   - Because overlaps exist, `main()` assumes the target output is determined by active-cell union rather
#     than placement order. Generator constructs outputs with that exact union rule.

from functools import reduce
from math import gcd

import numpy as np
import random
from itertools import combinations
from arc_utils import paste


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """
    Infer a regular cell grid separated by full rows/cols of the same line color.
    Returns (step, row_divs, col_divs) where:
      - row_divs/col_divs are divider indices
      - step is the spacing between dividers (cell size including divider).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = reduce(gcd, drs + dcs)
    assert step > 1, "divider step must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be (k*step + 1)"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"
    return int(step), row_divs, col_divs


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(mask)]


def _class_anchor_tops_lefts(base_top: int, base_left: int, n: int) -> list[tuple[int, int]]:
    return [((base_top + dr) % n, (base_left + dc) % n) for dr, dc in CLASS_OFFSETS]


def _placed_cells(mask: np.ndarray, top: int, left: int, n: int) -> set[tuple[int, int]]:
    return {((top + r) % n, (left + c) % n) for r, c in _mask_coords(mask)}


def _class_union(mask: np.ndarray, base_top: int, base_left: int, n: int) -> np.ndarray:
    union = np.zeros((n, n), dtype=bool)
    for top, left in _class_anchor_tops_lefts(base_top, base_left, n):
        for r, c in _mask_coords(mask):
            union[(top + r) % n, (left + c) % n] = True
    return union


def _compatible_base_classes(mask: np.ndarray, existing: set[tuple[int, int]], n: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for base_top in BASE_CHOICES:
        for base_left in BASE_CHOICES:
            anchors = _class_anchor_tops_lefts(base_top, base_left, n)
            if any(existing <= _placed_cells(mask, top, left, n) for top, left in anchors):
                candidates.append((base_top, base_left))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a minority-cell blueprint, then project it through a four-anchor toroidal overlay.

    Input structure:
    - A regular cell lattice is separated by full divider rows and columns of one color.
    - Most cells share one interior pattern; a small set use one distinct minority pattern.
    - The shown minority cells come from one concrete 3x3 enlarged instance, but the completed pattern is
      generated by a larger four-anchor toroidal class built from that instance.

    Rule:
    - Extract the minority cell's interior and treat its dominant color as a 3x3 active mask.
    - Find the unique toroidal base anchor whose four related placements at offsets `(0,0),(0,2),(2,0),(2,2)`
      contain a concrete instance consistent with all visible minority cells.
    - Place the same enlarged 3x3 motif at all four anchors with modulo-7 wraparound.
    - Combine the four placements by active-cell union: any cell hit by at least one active mask pixel becomes
      minority; all other cells keep the majority pattern.

    Key insight:
    - The odd cell still encodes a self-similar stencil, but now that stencil is copied to the four corners
      of a wrapped 5x5 toroidal frame, so border crossing and multi-placement overlap must both be understood.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns (minority + majority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color

    existing: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(n):
            if cell_pat[i][j] == minority_pat:
                existing.add((i, j))

    candidates = _compatible_base_classes(mask, existing, n)
    assert len(candidates) == 1, "must have a unique four-anchor toroidal class aligning the given minority cells"
    base_top, base_left = candidates[0]
    union = _class_union(mask, base_top, base_left, n)

    out = g.copy()
    for i in range(n):
        for j in range(n):
            target_pat = minority_arr if union[i, j] else majority_arr
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
CLASS_OFFSETS = ((0, 0), (0, 2), (2, 0), (2, 2))
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 600

BASE_CHOICES = (0, 2, 4)
NON_WRAP_BASES = (0, 2)
WRAP_BASES = (4,)
WRAP_BRANCHES = ("row", "col", "both")

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks():
    """Enumerate all valid LR-symmetric 3x3 masks, keyed by active count."""
    weights = [len(g) for g in _LR_GROUPS]
    by_count = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[i] for i in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for i in combo:
                    for r, c in _LR_GROUPS[i]:
                        mask[r, c] = False
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a, color_b):
    """Generate a random LR-symmetric 3x3 pattern using two colors (guaranteed non-uniform)."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask():
    """Pick a random valid minority mask (uniform over active count, then uniform within count)."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _cell_origin(i, j):
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator, majority, minority, active_cells):
    """Construct a 29x29 grid: fill all cells with majority, then overwrite active_cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _choose_wrapping_base():
    branch = random.choice(WRAP_BRANCHES)
    if branch == "row":
        return random.choice(WRAP_BASES), random.choice(NON_WRAP_BASES)
    if branch == "col":
        return random.choice(NON_WRAP_BASES), random.choice(WRAP_BASES)
    return random.choice(WRAP_BASES), random.choice(WRAP_BASES)


def _class_anchor_active_cells(mask: np.ndarray, base_top: int, base_left: int) -> list[list[tuple[int, int]]]:
    anchors: list[list[tuple[int, int]]] = []
    for top, left in _class_anchor_tops_lefts(base_top, base_left, CELL_GRID_N):
        anchors.append(sorted(_placed_cells(mask, top, left, CELL_GRID_N)))
    return anchors


def _union_active_cells(mask: np.ndarray, base_top: int, base_left: int) -> list[tuple[int, int]]:
    union = _class_union(mask, base_top, base_left, CELL_GRID_N)
    return [tuple(map(int, coord)) for coord in np.argwhere(union)]


def _choose_given_subset(mask: np.ndarray, base_top: int, base_left: int):
    """Choose global minority cells from one concrete anchor that identify the base class uniquely."""
    anchor_cells = _class_anchor_active_cells(mask, base_top, base_left)
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    target = (base_top, base_left)
    for size in sizes:
        options = []
        for cells in anchor_cells:
            for subset in combinations(cells, size):
                candidates = _compatible_base_classes(mask, set(subset), CELL_GRID_N)
                if candidates == [target]:
                    options.append(tuple(sorted(subset)))
        if options:
            return list(random.choice(options))
    return None


def _gen_attempt():
    """Single generation attempt: build output, choose input subset, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    base_top, base_left = _choose_wrapping_base()
    active_cells = _union_active_cells(mask, base_top, base_left)
    out = _build_grid(separator, majority, minority, active_cells)

    given_cells = _choose_given_subset(mask, base_top, base_left)
    if given_cells is None:
        return None
    inp = _build_grid(separator, majority, minority, given_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
