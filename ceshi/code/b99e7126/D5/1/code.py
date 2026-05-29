# Variant of b99e7126
# Dimension: D5 — 变换：叠加时的覆盖规则 (transform)
# Change: 放大图案中非背景色的单元格覆写原始网格中对应位置的单元格内容
#         → 颜色混合叠加：叠加区域使用新颜色

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# ---------------------------------------------------------------------------
# Design analysis for the D5 variant
#
# variant_spec:
#   core_change:
#     original_behavior:
#       - The minority cell's 3x3 interior was scaled to the 7x7 cell grid.
#       - Every active mask position in the scaled block was filled by directly
#         copying the minority cell pattern.
#     new_behavior:
#       - The minority cell still provides the 3x3 activity mask.
#       - But the completed active cells are not raw minority cells anymore.
#       - Instead, each completed active cell is a "mixed" cell: start from the
#         majority pattern, then recolor the minority-mask pixels to a third color.
#       - This realizes a visible color-combination rule:
#         majority-pixel + minority-active-overlay -> mix color.
#   concrete_pixel_behavior:
#     majority_cell:
#       - 3x3 LR-symmetric pattern using {shared, maj_exclusive}.
#     minority_cell:
#       - 3x3 pattern using {shared, min_exclusive}.
#       - min_exclusive occupies 6 or 7 pixels and defines the fractal mask.
#     mixed_cell:
#       - Computed from the majority cell, not copied from the minority cell.
#       - On mask=True pixels, the majority colors are replaced by mix_color.
#       - On mask=False pixels, the original majority pixels remain unchanged.
#     example_trace:
#       - Suppose a majority cell pixel at some location is maj_exclusive.
#       - In the minority blueprint, that location is active (min_exclusive).
#       - Original task output: the whole target cell becomes the minority cell,
#         so that pixel turns into min_exclusive.
#       - New variant output: the target cell becomes the mixed cell, so that
#         pixel turns into mix_color instead. This is the visible D5 change.
#   why_input_needs_adaptation:
#     - A solver only sees one input grid, so it cannot invent an unseen mix
#       color from nowhere.
#     - Therefore the input includes both:
#       - a few raw minority cells, used as blueprint / mask clues
#       - a few already-mixed cells, used as examples of the color-combination rule
#     - The remaining active cells are still majority cells in the input and must
#       be completed as mixed cells in the output.
#   complexity_calibration:
#     - Keep the original 7x7 cell lattice, 3x3 cell interiors, and 3x3 fractal mask.
#     - Use exactly 5 colors:
#       separator, shared, maj_exclusive, min_exclusive, mix_color.
#     - Only one nontrivial color-combination rule appears per instance:
#       overlaying the minority-active pixels onto a majority cell yields mix_color.
#     - This is enough to make the change visible without introducing a large
#       unconstrained mapping table that would be hard to infer from 2-4 examples.
#   main_implicit_assumption_audit:
#     - Inferring the divider grid is unchanged; mixed cells never touch divider lines.
#     - Counting cell patterns can no longer assume exactly two patterns.
#       The new solver expects three patterns:
#       majority (most common), minority clue (rarest), mixed preview (middle).
#     - The original `existing` mask used only minority cells. That would fail here
#       because some active positions are already mixed, not minority. The new solver
#       marks both minority and mixed cells as evidence for active positions.
#     - The original writeback used `minority_arr` on active cells. That would erase
#       the D5 behavior. The new writeback uses `mixed_arr`.
#   generator_legality:
#     - All cells still live on the fixed 7x7 lattice; no overlap or out-of-bounds
#       placement is possible because every write is a whole-cell paste at legal
#       cell origins.
#     - The generator only accepts subsets of active positions that give a unique
#       mask placement when minority and mixed evidence cells are combined.
#     - It also enforces at least one minority clue and at least two mixed previews,
#       so the new color behavior is both solvable and observable.
#   visual_observability:
#     - Humans can compare three cell types directly in the input:
#       majority cells, minority clue cells, and mixed preview cells.
#     - In the output, every active position matches the mixed preview cells rather
#       than the minority clue cells, so the D5 difference is visible immediately.
#   cleanliness:
#     - The grid remains sparse and regular; only one 3x3 block of cells is involved.
#     - Mixed cells preserve the majority cell's inactive pixels, so the result
#       remains visually coherent rather than noisy.
#   new_solver_assumptions_to_guarantee_in_generator:
#     - Exactly three cell patterns appear in the input.
#     - minority_count < mixed_count < majority_count.
#     - mixed cells equal the majority cell outside the minority mask.
#     - minority cells use a dominant active color occupying 6 or 7 pixels.
#     - The union of minority and mixed evidence positions determines a unique
#       placement of the 3x3 mask inside the 7x7 cell grid.
# ---------------------------------------------------------------------------


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


def _extract_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Collect every cell interior pattern and its array form."""
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
    return patterns, cell_pat


def _pattern_counts(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for p in row:
            counts[p] = counts.get(p, 0) + 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal cell pattern by turning active positions into mixed-color overlays.

    Input structure:
    - A 7x7 cell lattice is separated by one-color divider lines.
    - Most cells show one majority interior pattern.
    - A few cells show the raw minority blueprint pattern, and a few others already
      show the mixed result of overlaying that blueprint onto the majority pattern.

    Rule:
    - Read the minority cell's 3x3 interior as a miniature boolean mask, where its
      dominant color marks active pixels.
    - Find the unique placement of that 3x3 mask on the 7x7 cell lattice using both
      minority cells and already-mixed cells as evidence of active positions.
    - Replace every active cell in that placed block with the mixed cell pattern,
      leaving inactive cells as majority cells.

    Key insight:
    - The minority cell is still the fractal blueprint, but the finished large-scale
      image is not made of raw minority cells. Active positions are completed with
      a third, already-demonstrated mixed pattern that encodes the overlay result.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _extract_patterns(g, step)
    counts = _pattern_counts(cell_pat)
    assert len(counts) == 3, "variant expects exactly three cell patterns"

    ranked = sorted(counts.items(), key=lambda kv: kv[1])
    minority_pat = ranked[0][0]
    mixed_pat = ranked[1][0]
    majority_pat = ranked[2][0]
    assert counts[minority_pat] < counts[mixed_pat] < counts[majority_pat], "pattern frequencies must be ordered"

    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]
    mixed_arr = patterns[mixed_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape
    assert mh == inner and mw == inner, "minority mask must match cell interior"
    assert mask.sum() >= 6, "active color should dominate the minority blueprint"

    # The mixed preview must preserve the majority pixels outside the active mask.
    assert np.array_equal(mixed_arr[~mask], majority_arr[~mask]), "mixed cell must match majority outside mask"
    assert np.all(mixed_arr[mask] != majority_arr[mask]), "mixed cell must change active pixels"
    assert np.all(mixed_arr[mask] != minority_arr[mask]), "mixed color must differ from raw minority color"

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] in (minority_pat, mixed_pat)

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
    assert len(candidates) == 1, "must have a unique mask placement aligning all active evidence"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = mixed_arr if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

# --- CFG: tunable generation parameters ---
CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_COUNT_PAIRS = ((1, 2), (1, 3), (2, 3))  # (minority clues, mixed previews)
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


def _precompute_masks():
    """Enumerate all valid LR-symmetric 3x3 masks, keyed by active pixel count."""
    weights = [len(g) for g in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
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
    """Generate a random LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, mixed: np.ndarray,
                minority_cells: list[tuple[int, int]], mixed_cells: list[tuple[int, int]]) -> np.ndarray:
    """Construct a full grid from majority cells plus minority/mixed exceptions."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(i, j))
    for i, j in mixed_cells:
        paste(grid, mixed, *_cell_origin(i, j))
    return grid


def _placement_candidates(mask: np.ndarray, evidence_subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find all offsets where the observed active-evidence subset fits the mask."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in evidence_subset:
        existing[top + r, left + c] = True

    candidates: list[tuple[int, int]] = []
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


def _choose_evidence_partition(mask: np.ndarray, top: int, left: int):
    """
    Choose active positions shown in the input.

    Returns:
      (minority_subset, mixed_subset) in mask-local coordinates, or None.
    """
    active = [tuple(coord) for coord in np.argwhere(mask)]
    pair_options = list(GIVEN_COUNT_PAIRS)
    random.shuffle(pair_options)
    for clue_count, preview_count in pair_options:
        total = clue_count + preview_count
        if total >= len(active):
            continue
        valid_total_subsets = [
            subset
            for subset in combinations(active, total)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        random.shuffle(valid_total_subsets)
        for subset in valid_total_subsets:
            local = list(subset)
            minority_choices = [
                tuple(sorted(choice))
                for choice in combinations(local, clue_count)
            ]
            random.shuffle(minority_choices)
            for minority_subset in minority_choices:
                mixed_subset = tuple(pos for pos in local if pos not in minority_subset)
                return tuple(minority_subset), mixed_subset
    return None


def _gen_attempt():
    """
    Single generation attempt.

    Output-first construction:
    - Output contains the full active block as mixed cells.
    - Input keeps a strict subset of active positions visible, split into raw
      minority clues and mixed previews.
    """
    separator, shared, maj_exclusive, min_exclusive, mix_color = random.sample(range(COLOR_POOL), 5)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)
    mixed = majority.copy()
    mixed[mask] = mix_color

    # Keep the three cell patterns distinct and the mixed result visibly tied
    # to the majority outside the active mask.
    if np.array_equal(majority, minority) or np.array_equal(majority, mixed) or np.array_equal(minority, mixed):
        return None
    if not np.array_equal(mixed[~mask], majority[~mask]):
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    out = _build_grid(separator, majority, minority, mixed, [], active_cells)

    partition = _choose_evidence_partition(mask, top, left)
    if partition is None:
        return None
    minority_subset, mixed_subset = partition
    minority_cells = [(top + r, left + c) for r, c in minority_subset]
    mixed_cells = [(top + r, left + c) for r, c in mixed_subset]
    inp = _build_grid(separator, majority, minority, mixed, minority_cells, mixed_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 D5 input")
