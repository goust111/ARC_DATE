# Variant of b99e7126
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: 输出与输入网格尺寸相同... → 输出紧凑裁剪到叠加影响区域

# Design analysis for the D9 variant
# ----------------------------------
# spec:
#   variant_goal:
#     Keep the original fractal-completion rule, but return only the enlarged
#     3x3 cell block instead of the entire 29x29 board.
#   concrete_behavior:
#     - Input stays unchanged: a 29x29 divider grid containing 7x7 cells.
#     - The odd cell still provides a 3x3 binary mask through its dominant color.
#     - Completing that mask at cell scale still affects exactly one 3x3 cell block.
#     - The output is the minimal self-contained mini-grid covering that block:
#       rows top*step .. (top+3)*step and cols left*step .. (left+3)*step.
#       With step=4, every output is a 13x13 cropped subgrid with its outer
#       divider frame preserved.
#   trace_example:
#     - Original logic:
#       1. Parse all 49 cell interiors.
#       2. Infer minority/majority patterns.
#       3. Use the minority interior as a 3x3 placement stencil.
#       4. Fill the full 29x29 board and return the whole board.
#     - Variant logic:
#       1. Steps 1-3 are unchanged.
#       2. Complete the same 3x3 cell block.
#       3. Return only that block with its border dividers, dropping the
#          untouched outer 26 cells.
#     - Difference:
#       The changed stage is only the final materialization of the answer.
#       Recognition and completion remain the same; global context is removed.
#   complexity_guardrails:
#     - Keep MASK_SIZE=3 and the same majority/minority pattern system so the
#       task is still recognizably the same seed.
#     - Preserve LR-symmetric mask generation and unique-placement filtering,
#       avoiding extra degrees of freedom unrelated to D9.
#     - Crop on cell-block boundaries, not the raw overwritten interior pixels.
#       A stricter 11x11 crop would delete the outer divider frame and make the
#       miniature grid harder to read visually.
#   implicit_assumptions_review:
#     - `_infer_cell_grid_params` still assumes full divider rows/cols on the
#       input boundary. This remains true because only the output shape changes.
#     - Cell-pattern counting still assumes a dominant majority pattern and a
#       rarer minority pattern. Generator continues to create exactly that.
#     - Candidate search still assumes the shown minority cells uniquely pin down
#       one placement. Generator keeps rejecting ambiguous subsets.
#     - The old `out = g.copy()` + global overwrite was harmless because the
#       output size matched the input. In this variant that would be wrong if
#       returned directly, because untouched context must be removed. The solver
#       now renders only the local completed block after finding the placement.
#   placement_legality:
#     - The 3x3 mask still fits in the 7x7 cell board via top/left in 0..4.
#     - Cropping to the completed block cannot go out of bounds because the full
#       block already fits inside the original board by construction.
#   visual_observability:
#     - Across samples, the input remains a full board while the output is a
#       compact 13x13 mini-board. That size jump is immediately visible.
#     - Because the outer divider frame is kept, humans can read the output as
#       “the completed local block” instead of a confusing partial crop.
#   cleanliness:
#     - The output contains exactly one coherent object: the completed 3x3 cell
#       patch. No extra board area survives, so there is less visual clutter.
#   color_reasoning:
#     - No sorting or branching depends on color ids themselves.
#     - Colors only play their original structural roles: separator, shared,
#       majority-exclusive, minority-exclusive.
#   new_solver_assumptions_to_match_in_generator:
#     - The generator must continue to expose at least one minority cell and one
#       majority cell in the input.
#     - The shown minority subset must still uniquely determine a placement,
#       even though the final crop discards global coordinates.

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

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


def _cell_pattern_data(g: np.ndarray):
    """Extract per-cell interior patterns plus their multiplicities."""
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            counts[pat] = counts.get(pat, 0) + 1
            row.append(pat)
        cell_pat.append(row)
    return step, n, inner, patterns, cell_pat, counts


def _solve_mask_placement(g: np.ndarray):
    """Recover majority/minority cell patterns and the unique block placement."""
    step, n, _inner, patterns, cell_pat, counts = _cell_pattern_data(g)
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns (minority + majority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    candidates: list[tuple[int, int]] = []
    for top in range(0, n - mh + 1):
        for left in range(0, n - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique mask placement aligning existing minority cells"
    return step, majority_arr, minority_arr, mask, candidates[0]


def _render_completed_block(
    line_color: int,
    step: int,
    majority_arr: np.ndarray,
    minority_arr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Render the completed local cell block, including its outer divider frame."""
    mh, mw = mask.shape
    inner = step - 1
    out = np.full((mh * step + 1, mw * step + 1), int(line_color), dtype=int)
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if mask[i, j] else majority_arr
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


def _crop_cell_block(g: np.ndarray, top: int, left: int, cells_h: int, cells_w: int, step: int) -> np.ndarray:
    """Crop a cell-aligned block while preserving its outer divider frame."""
    r0 = top * step
    c0 = left * step
    r1 = (top + cells_h) * step
    c1 = (left + cells_w) * step
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the fractal cell-block and return only the affected mini-grid.

    Input structure:
    - A full separator grid partitions the board into equal square cells.
    - Almost every cell shares one interior pattern; a few shown cells use one odd pattern.

    Rule:
    - Read the odd cell's interior as a 3x3 binary mask, using its dominant color as "active".
    - Find the unique 3x3 placement in the 7x7 cell lattice that is consistent with the shown odd cells.
    - Fill the entire placed 3x3 block with odd-pattern cells on active mask entries and majority cells elsewhere.
    - Return only that completed block, cropped to its outer divider frame.

    Key insight:
    The odd cell is both the texture to stamp and the blueprint for where that texture belongs at the cell scale; once the local block is reconstructed, the untouched outer board is irrelevant and should be discarded.
    """
    g = np.asarray(grid, dtype=int)
    step, majority_arr, minority_arr, mask, _placement = _solve_mask_placement(g)
    return _render_completed_block(int(g[0, 0]), step, majority_arr, minority_arr, mask)


# --------------- generator (o5x) ---------------

# --- CFG: tunable generation parameters ---
CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 300

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks():
    """Enumerate valid LR-symmetric 3x3 masks, keyed by active pixel count."""
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
                for idx in combo:
                    for r, c in _LR_GROUPS[idx]:
                        mask[r, c] = False
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a, color_b):
    """Generate a non-uniform LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask():
    """Pick a valid minority mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _placement_candidates(mask, subset, top, left):
    """Find all offsets where the given shown active cells fit the mask."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    candidates = []
    for t in range(PLACEMENT_RANGE):
        for l in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[t : t + MASK_SIZE, l : l + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[t : t + MASK_SIZE, l : l + MASK_SIZE] & ~mask):
                continue
            candidates.append((t, l))
    return candidates


def _choose_given_subset(mask, top, left):
    """Choose a shown subset that uniquely determines the placement."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i, j):
    """Pixel origin of cell (i, j) interior inside the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator, majority, minority, active_cells):
    """Construct the full 29x29 board from cell patterns."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt():
    """Build an input whose solver output is the tight local crop."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    full_out = _build_grid(separator, majority, minority, active_cells)
    cropped_out = _crop_cell_block(full_out, top, left, MASK_SIZE, MASK_SIZE, CELL_STEP)

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, cropped_out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 D9 input")
