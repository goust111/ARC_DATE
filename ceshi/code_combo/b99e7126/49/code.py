# Combo variant of b99e7126 (index 49)
# combo_spec: D9/1+D14/0
# Name: 输出扩展 + 环绕边界

from functools import reduce
from itertools import combinations
from math import gcd
import random

import numpy as np

from arc_utils import paste

# Design analysis for the D9/1 + D14/0 combo
# -------------------------------------------
# concrete_behavior:
#   - Start from the original 7x7 cell lattice with 3x3 interiors and 1-cell dividers.
#     A minority cell still encodes a 3x3 Boolean mask through its dominant color.
#   - The fused placement is solved on the 7x7 lattice as a torus. Example:
#       choose top=6, left=5. The local 3x3 blueprint occupies raw cell rows 6,7,8
#       and raw cols 5,6,7. Inside the original 7x7 window these land at wrapped
#       coordinates rows (6,0,1) and cols (5,6,0).
#   - D14 contributes the modulo-7 projection above: the input may already show odd
#     cells split across opposite borders, and the solver must search over all 7x7
#     toroidal offsets instead of only in-bounds rectangles.
#   - D9 then changes only the output canvas: after applying the wrapped placement to
#     the original 7x7 period, the output grows to include any raw rows/cols that ran
#     past the bottom or right edge. In the same top=6,left=5 example, the output has
#     9 cell rows and 8 cell cols, so the extra bottom/right bands explicitly show the
#     overflow continuation that caused the wrap.
# complexity_calibration:
#   - The puzzle is still recognizably b99e7126: same two recurring cell interiors,
#     same self-similar mask, same 3x3 cell-level enlargement.
#   - To make both dimensions observable, generation always chooses an offset that
#     wraps on at least one axis (top or left in {5,6}); this guarantees output
#     expansion as well.
#   - The mask family still has 6-7 active pixels and touches all four mask edges, so
#     any wrapped axis also produces visible minority cells in the added band rather
#     than a larger but visually empty border.
# implicit_assumptions_in_original_main:
#   - `_infer_cell_grid_params` only assumes a regular divider lattice; unchanged.
#   - Cell-pattern extraction/counting still assumes only majority vs minority cell
#     interiors appear in the input. The generator preserves that.
#   - The original candidate loops `range(0, n-mh+1)` assumed the enlarged mask must
#     fit as one in-bounds rectangle. The combo breaks that, so candidate search must
#     range over every toroidal offset `0..n-1`.
#   - The original boolean block slice also assumed contiguous in-bounds occupancy.
#     That is replaced by an explicit wrapped-coordinate set.
#   - The original output write assumed the output shape equals the input shape and
#     each target cell exists exactly once. Under the combo we need two render views:
#     the wrapped base-period cells, plus any raw overflow cells in the expanded band.
# placement_legality:
#   - With n=7 and mask size 3, the raw 3x3 placement can exceed the base period by at
#     most 2 rows and 2 cols, so the expanded canvas is at most 9x9 cells.
#   - Raw overflow is only allowed on the bottom/right because the anchor top-left is
#     always chosen inside the base period and local mask coordinates are nonnegative.
#   - The generator enforces unique toroidal placement from the visible minority subset
#     and verifies every produced input by comparing a reconstructed output to `main()`.
# observability_and_tidiness:
#   - Human-visible novelty comes from two linked cues: border-crossing minority cells
#     already visible in the 29x29 input, and a larger output that duplicates the
#     overflow continuation in aligned extra rows/cols.
#   - Only four colors are used and every cell interior remains a clean 3x3 motif, so
#     the expanded band looks like a coherent continuation of the same lattice.
# inferability_of_change:
#   - The added rows/cols are not arbitrary decoration: they contain the same cell
#     vocabulary as the base period, and their minority locations line up with the
#     wrapped minority cells on the opposite edges.
#   - Because the same top/left choice controls both the split input evidence and the
#     enlarged output shape, the two dimensions reinforce each other rather than acting
#     as independent tricks.
# new_solver_assumptions_checked_against_generator:
#   - The dominant color inside the minority pattern is still the active mask color
#     because active pixel count is restricted to 6 or 7.
#   - At least one wrap axis is present, so output expansion is exercised in every
#     generated sample.
#   - The visible minority subset is chosen to straddle every wrapped axis, so the
#     input itself provides direct border-crossing evidence instead of hiding wrap
#     until the output stage.
#   - Whenever an axis wraps, the extra band contains at least one active minority cell
#     because the mask is required to touch every edge of its 3x3 frame.
#   - The solver assumes the expanded band should show only the first raw overflow
#     beyond the base period; generation matches that same convention exactly.


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


def _placed_cells(mask: np.ndarray, top: int, left: int, n: int) -> set[tuple[int, int]]:
    return {
        ((top + r) % n, (left + c) % n)
        for r, c in _mask_coords(mask)
    }


def _overflow_counts(top: int, left: int, mh: int, mw: int, n: int) -> tuple[int, int]:
    return max(0, top + mh - n), max(0, left + mw - n)


def _cell_origin(i: int, j: int, step: int) -> tuple[int, int]:
    return i * step + 1, j * step + 1


def _render_periodic_grid(
    separator: int,
    majority: np.ndarray,
    cell_rows: int,
    cell_cols: int,
    step: int,
) -> np.ndarray:
    grid = np.full((cell_rows * step + 1, cell_cols * step + 1), int(separator), dtype=int)
    for i in range(cell_rows):
        for j in range(cell_cols):
            paste(grid, majority, *_cell_origin(i, j, step))
    return grid


def _build_combo_output(
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
    n: int,
    step: int,
) -> np.ndarray:
    mh, mw = mask.shape
    extra_rows, extra_cols = _overflow_counts(top, left, mh, mw, n)
    out = _render_periodic_grid(separator, majority, n + extra_rows, n + extra_cols, step)

    for i in range(mh):
        for j in range(mw):
            target_pat = minority if mask[i, j] else majority

            rr = (top + i) % n
            cc = (left + j) % n
            paste(out, target_pat, *_cell_origin(rr, cc, step))

            raw_r = top + i
            raw_c = left + j
            if raw_r >= n or raw_c >= n:
                paste(out, target_pat, *_cell_origin(raw_r, raw_c, step))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the odd cell's self-similar blueprint with toroidal placement and canvas growth.

    Input structure:
    - The board is a regular lattice of equal cells separated by one-color divider lines.
    - One cell interior pattern is the majority template, while a smaller set of cells uses one minority template.
    - Those visible minority cells may already be split across opposite edges because the large stencil is positioned on a wrapped cell lattice.

    Rule:
    - Read the minority cell's interior, using its dominant color as the active 3x3 mask.
    - Find the unique toroidal offset whose wrapped active coordinates contain all visible minority cells.
    - Produce the output by first applying that 3x3 stencil to the original cell period with wraparound, then extending the canvas enough to also show any raw overflow beyond the bottom or right border.

    Key insight:
    - The minority cell does double duty: it is both the motif to stamp and the map of where stamps belong, and in this combo the same placement must be understood simultaneously as a periodic wrap on the base lattice and as a larger unwrapped continuation in the output.
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
            r0, c0 = _cell_origin(i, j, step)
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

    candidates: list[tuple[int, int]] = []
    for top in range(n):
        for left in range(n):
            placed = _placed_cells(mask, top, left, n)
            if existing <= placed:
                candidates.append((top, left))
    assert len(candidates) == 1, "must have a unique toroidal mask placement aligning existing minority cells"
    top, left = candidates[0]

    return _build_combo_output(int(g[0, 0]), majority_arr, minority_arr, mask, top, left, n, step)


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 300

NON_WRAP_OFFSETS = tuple(range(CELL_GRID_N - MASK_SIZE + 1))
WRAP_OFFSETS = tuple(range(CELL_GRID_N - MASK_SIZE + 1, CELL_GRID_N))
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
    """Enumerate all valid LR-symmetric 3x3 masks, keyed by active pixel count."""
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


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a random LR-symmetric 3x3 pattern using two colors (guaranteed non-uniform)."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask() -> np.ndarray:
    """Pick a random valid minority mask (uniform over active count, then uniform within count)."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _wrapped_active_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [
        ((top + r) % CELL_GRID_N, (left + c) % CELL_GRID_N)
        for r, c in _mask_coords(mask)
    ]


def _placement_candidates(mask: np.ndarray, subset_cells: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    """Find all toroidal offsets where the visible minority cells are consistent with the mask."""
    existing = set(subset_cells)
    candidates = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            placed = _placed_cells(mask, top, left, CELL_GRID_N)
            if existing <= placed:
                candidates.append((top, left))
    return candidates


def _subset_shows_wrap(subset, top: int, left: int) -> bool:
    row_wrap = top in WRAP_OFFSETS
    col_wrap = left in WRAP_OFFSETS
    rows = {r for r, _ in subset}
    cols = {c for _, c in subset}

    if row_wrap and not (any(r >= top for r in rows) and any(r < top for r in rows)):
        return False
    if col_wrap and not (any(c >= left for c in cols) and any(c < left for c in cols)):
        return False
    return True


def _choose_given_subset(mask: np.ndarray, top: int, left: int):
    """Choose a subset of active cells that uniquely determines the toroidal placement."""
    placed = _wrapped_active_cells(mask, top, left)
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(placed, size)
            if _placement_candidates(mask, subset) == [(top, left)] and _subset_shows_wrap(subset, top, left)
        ]
        if valid:
            return random.choice(valid)
    return None


def _build_input_grid(separator: int, majority: np.ndarray, minority: np.ndarray, active_cells) -> np.ndarray:
    """Construct the fixed 29x29 input grid."""
    grid = _render_periodic_grid(separator, majority, CELL_GRID_N, CELL_GRID_N, CELL_STEP)
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j, CELL_STEP))
    return grid


def _choose_wrapping_offset() -> tuple[int, int]:
    branch = random.choice(WRAP_BRANCHES)
    if branch == "row":
        return random.choice(WRAP_OFFSETS), random.choice(NON_WRAP_OFFSETS)
    if branch == "col":
        return random.choice(NON_WRAP_OFFSETS), random.choice(WRAP_OFFSETS)
    return random.choice(WRAP_OFFSETS), random.choice(WRAP_OFFSETS)


def _gen_attempt():
    """Single generation attempt: build expected output, choose input subset, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top, left = _choose_wrapping_offset()
    active_cells = _wrapped_active_cells(mask, top, left)
    out = _build_combo_output(separator, majority, minority, mask, top, left, CELL_GRID_N, CELL_STEP)

    given_cells = _choose_given_subset(mask, top, left)
    if given_cells is None:
        return None
    inp = _build_input_grid(separator, majority, minority, given_cells)

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
