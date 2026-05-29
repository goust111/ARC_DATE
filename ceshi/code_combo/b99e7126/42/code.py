# Combo variant of b99e7126 (index 42)
# combo_spec: D8/2+D15/0
# Name: 嵌套层级单元格 + 双层分形
#
# Design analysis for the fused variant
# -------------------------------------
# Concrete trace:
# - Original task: one 3x3 pixel pattern inside a minority cell is expanded once onto a
#   single cell-grid level.
# - Fused task here: the same 3x3 source pattern is used twice. First it expands from
#   pixel scale to a 3x3 block of small cells. That completed 3x3 block acts as one
#   "large cell". Then the same mask expands again from large-cell scale to the outer
#   grid of large cells. The visible output is therefore a mask-of-a-mask:
#   source pixels -> small cells inside a large cell -> large-cell placements in the grid.
# - Example sketch (mask = plus-like 3x3):
#   1) A minority small cell contains the 3x3 source pixels.
#   2) Inside an active large cell, those pixels are reinterpreted as a 3x3 arrangement
#      of small cells, producing a first-level fractal block.
#   3) Across the whole canvas, only some large-cell blocks are present in the input.
#      The solver completes the missing large-cell blocks by applying the same mask again.
#
# Complexity calibration:
# - The mask still has only 6-7 active pixels, so the structure is rich enough to be
#   visually distinctive but not noisy.
# - Colors stay at 4 roles, matching the original task's discipline.
# - The input shows 3-4 complete active large cells, enough to pin down a unique outer
#   placement without flooding the grid.
# - The large-cell size is fixed to 3x3 small cells, keeping the nested rule legible in
#   2-4 ARC examples.
#
# Implicit assumptions audited against the original solver:
# - Original assumption "minority cells directly mark the outer mask" no longer holds.
#   Minority small cells now come in 3x3 grouped macro-blocks, so the solver must first
#   compress the fine grid into large cells before searching for the outer placement.
# - Original assumption "the finest divider grid is the task grid" also changes. The full
#   separator rows/cols now expose only the small-cell lattice; the large-cell lattice is
#   implicit and must be reconstructed by grouping every source-size chunk of small cells.
# - The minority small-cell pattern is still the source blueprint, but it is no longer
#   enough to stop after one placement test. A second, coarser placement is required.
#
# Placement legality:
# - The full grid is a regular 15x15 small-cell lattice.
# - Large cells are aligned 3x3 groups of small cells, giving a 5x5 outer large-cell grid.
# - The outer 3x3 mask is placed only where it fits in the 5x5 large-cell grid, so no
#   overlap/out-of-bounds issues arise.
#
# Visual observability / cleanliness:
# - Large-cell structure is visible because minority small cells always appear as repeated
#   3x3 grouped fractal blocks, never as isolated noise.
# - Only two cell interiors appear at the finest level (majority vs source), so the image
#   remains readable despite the extra hierarchy.
# - The transformation is structural rather than color-order based; no hidden dependence on
#   numeric color ids is used.

from functools import reduce
from itertools import combinations
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
_SOLVER_DIR = _ROOT / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import paste


SMALL_INNER = 3
SMALL_STEP = SMALL_INNER + 1
MASK_SIZE = 3
OUTER_MACRO_N = 5
FINE_GRID_N = OUTER_MACRO_N * MASK_SIZE
GRID_SIZE = FINE_GRID_N * SMALL_STEP + 1
PLACEMENT_RANGE = OUTER_MACRO_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_MACRO_SIZES = (3, 4)
COLOR_POOL = 10
MAX_RETRIES = 400

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _infer_small_grid_params(g: np.ndarray) -> tuple[int, int]:
    """Infer the finest regular cell grid from full separator rows/cols."""
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need repeated separator rows/cols"
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = reduce(gcd, drs + dcs)
    assert step > 1, "separator step must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "grid must start on a separator"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "grid must end on a separator"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid size must fit the inferred step"
    n = (h - 1) // step
    return int(step), int(n)


def _extract_small_cells(g: np.ndarray, step: int, n: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Extract all smallest cell interiors from the finest lattice."""
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
    return patterns, cell_pat


def _count_patterns(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def _mask_from_pattern(arr: np.ndarray) -> tuple[np.ndarray, int]:
    vals, cnts = np.unique(arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    return arr == active_color, active_color


def _bool_block_from_pattern(
    cell_pat: list[list[tuple[int, ...]]], pat_key: tuple[int, ...], r0: int, c0: int, size: int
) -> np.ndarray:
    block = np.zeros((size, size), dtype=bool)
    for i in range(size):
        for j in range(size):
            block[i, j] = cell_pat[r0 + i][c0 + j] == pat_key
    return block


def _compress_to_macro_activity(
    cell_pat: list[list[tuple[int, ...]]], source_key: tuple[int, ...], mask: np.ndarray
) -> np.ndarray:
    """Group the fine grid into 3x3 macro-cells and mark which macro-cells are active."""
    fine_n = len(cell_pat)
    k = mask.shape[0]
    assert fine_n % k == 0, "fine lattice must be divisible by the source pattern size"
    macro_n = fine_n // k
    activity = np.zeros((macro_n, macro_n), dtype=bool)
    for bi in range(macro_n):
        for bj in range(macro_n):
            block = _bool_block_from_pattern(cell_pat, source_key, bi * k, bj * k, k)
            if np.array_equal(block, mask):
                activity[bi, bj] = True
            else:
                assert not block.any(), "each macro-cell must be either empty or a full first-level fractal"
    return activity


def _find_unique_outer_placement(existing: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Locate the unique outer placement whose active macro-cells fit the observed subset."""
    n = existing.shape[0]
    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(n - mh + 1):
        for left in range(n - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "outer placement must be unique"
    return candidates[0]


def _render_from_macro_activity(
    separator: int,
    majority_small: np.ndarray,
    source_small: np.ndarray,
    mask: np.ndarray,
    macro_activity: np.ndarray,
) -> np.ndarray:
    """Render the full pixel grid from macro activity and the shared 3x3 source mask."""
    fine_n = macro_activity.shape[0] * mask.shape[0]
    grid = np.full((fine_n * SMALL_STEP + 1, fine_n * SMALL_STEP + 1), separator, dtype=int)
    k = mask.shape[0]
    for macro_r in range(macro_activity.shape[0]):
        for macro_c in range(macro_activity.shape[1]):
            for inner_r in range(k):
                for inner_c in range(k):
                    fine_r = macro_r * k + inner_r
                    fine_c = macro_c * k + inner_c
                    pat = source_small if macro_activity[macro_r, macro_c] and mask[inner_r, inner_c] else majority_small
                    paste(grid, pat, fine_r * SMALL_STEP + 1, fine_c * SMALL_STEP + 1)
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a two-level fractal whose source lives inside the smallest repeated cells.

    Input structure:
    - The visible divider lattice is the finest grid: a regular array of small cells.
    - Most small cells share one interior pattern; a minority small-cell pattern appears only
      inside a few complete 3x3 macro-blocks.
    - Interpreting every 3x3 pack of small cells as one large cell reveals a second, coarser
      grid of macro-cells.

    Rule:
    - Read the minority small-cell interior as the source 3x3 pixel pattern and derive its
      active mask from the dominant color inside that source.
    - Use that mask once to recognize which 3x3 small-cell packs are complete active macro-cells.
    - Use the same mask again at macro-cell scale to place the missing active macro-cells in the
      unique outer position consistent with the observed subset.

    Key insight:
    - The source pattern is applied twice: first to build each active macro-cell from small cells,
      then to arrange those macro-cells across the outer grid. The task only makes sense if the
      solver reconstructs the hidden large-cell hierarchy instead of treating the fine grid as the
      final scale.
    """
    g = np.asarray(grid, dtype=int)
    separator = int(g[0, 0])
    step, fine_n = _infer_small_grid_params(g)
    assert step == SMALL_STEP, "unexpected small-cell step"

    patterns, cell_pat = _extract_small_cells(g, step, fine_n)
    counts = _count_patterns(cell_pat)
    assert len(counts) >= 2, "need at least a majority and a source small-cell pattern"

    source_key = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_key = max(counts.items(), key=lambda kv: kv[1])[0]
    source_small = patterns[source_key]
    majority_small = patterns[majority_key]

    mask, _ = _mask_from_pattern(source_small)
    assert mask.shape[0] == mask.shape[1] == MASK_SIZE, "source mask must be 3x3"

    macro_activity = _compress_to_macro_activity(cell_pat, source_key, mask)
    top, left = _find_unique_outer_placement(macro_activity, mask)

    full_macro_activity = np.zeros_like(macro_activity)
    full_macro_activity[top : top + MASK_SIZE, left : left + MASK_SIZE] = mask
    return _render_from_macro_activity(separator, majority_small, source_small, mask, full_macro_activity)


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate LR-symmetric 3x3 masks with active pixels spanning all four sides."""
    weights = [len(group) for group in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
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


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform LR-symmetric 3x3 pattern from two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_source_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _macro_placement_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find all outer placements compatible with the observed subset of active macro-cells."""
    existing = np.zeros((OUTER_MACRO_N, OUTER_MACRO_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    candidates: list[tuple[int, int]] = []
    for t in range(PLACEMENT_RANGE):
        for l in range(PLACEMENT_RANGE):
            block = np.zeros((OUTER_MACRO_N, OUTER_MACRO_N), dtype=bool)
            block[t : t + MASK_SIZE, l : l + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[t : t + MASK_SIZE, l : l + MASK_SIZE] & ~mask):
                continue
            candidates.append((t, l))
    return candidates


def _choose_given_macro_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    """Choose a subset of active macro-cells that fixes the outer placement uniquely."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_MACRO_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _macro_placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _build_grid(separator: int, majority_small: np.ndarray, source_small: np.ndarray, mask: np.ndarray, active_macros: list[tuple[int, int]]) -> np.ndarray:
    macro_activity = np.zeros((OUTER_MACRO_N, OUTER_MACRO_N), dtype=bool)
    for r, c in active_macros:
        macro_activity[r, c] = True
    return _render_from_macro_activity(separator, majority_small, source_small, mask, macro_activity)


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, src_exclusive = random.sample(range(COLOR_POOL), 4)
    majority_small = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_source_mask()
    source_small = np.where(mask, src_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_macros = [(top + r, left + c) for r, c in np.argwhere(mask)]
    out = _build_grid(separator, majority_small, source_small, mask, active_macros)

    subset = _choose_given_macro_subset(mask, top, left)
    if subset is None:
        return None
    given_macros = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority_small, source_small, mask, given_macros)

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
    raise RuntimeError("failed to generate a valid nested two-level fractal input")
