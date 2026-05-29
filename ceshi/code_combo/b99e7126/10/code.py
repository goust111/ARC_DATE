# Combo variant of b99e7126 (index 10)
# combo_spec: D5/1+D11/0
# Name: 颜色混合叠加 + 颜色重映射

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# ---------------------------------------------------------------------------
# Design analysis for the fused D5/1 + D11/0 variant
#
# concretization_and_trace:
# - The original task had only two meaningful cell interiors: a majority template
#   and a minority template. The minority cell's dominant color defined a 3x3 mask,
#   and every active large-scale cell in the output became a literal copy of the
#   minority template.
# - In this combo, the minority cell is only the first clue. Its dominant color
#   still gives the 3x3 mask, but that active color is not used directly when the
#   pattern is enlarged. First it is recolored into the majority-only foreground
#   color, producing a separate recolored preview cell. Then that recolored overlay
#   is mixed with the majority cell: because the majority carries the shared color
#   under every active mask pixel, the single visible mix rule is
#   shared + majority-only -> mix-color.
# - Example trace:
#   minority pixel = min-exclusive on a mask=True site
#   -> recolor step turns it into maj-exclusive
#   -> mix step lays that over a majority pixel known to be shared
#   -> final pixel becomes mix-color.
#   The output active cells therefore match neither the raw minority clue nor the
#   recolored preview exactly; they show the mixed result.
#
# complexity_calibration:
# - Keep the original 7x7 cell lattice, 3x3 cell interiors, and 3x3 LR-symmetric
#   minority mask with 6-7 active pixels.
# - Use exactly 5 colors:
#   separator, shared, majority-only, minority-only, and mix-color.
# - Expose exactly four cell patterns in the input:
#   majority, minority clue, recolored preview, mixed preview.
#   This is enough to make the two-step rule observable without exploding the
#   number of color cases a human must infer from 2-4 examples.
#
# hidden_assumption_audit:
# - The original solver assumed exactly two patterns and used the minority pattern
#   itself as the final writeback template. Both assumptions break here.
# - The D5-only solver assumption "all non-majority active evidence can be treated
#   as minority-or-mixed" is also insufficient, because the recolored preview is a
#   third active-state pattern that must be identified separately.
# - To keep the mixing rule small and visible, the generator forces the majority
#   cell to contain the shared color under every mask=True position. That prevents
#   multiple hidden mix cases such as
#   shared + overlay and maj-exclusive + overlay appearing in the same sample.
#
# placement_legality:
# - Cells are still written only at legal 3x3 interior origins inside the fixed
#   7x7 lattice; no object can overlap divider lines or leave the board.
# - The placed 3x3 mask still uses offsets 0..4, so the active block always fits.
# - Input evidence is chosen from true active positions only, and the union of all
#   visible non-majority cells must determine a unique placement.
#
# visual_observability_and_cleanliness:
# - A human can compare the four cell types directly:
#   majority shows the host palette, minority shows the raw blueprint, recolored
#   shows the D11 mapping, and mixed shows the D5 result of combining recolored
#   overlay with the host.
# - The mixed cell preserves the majority cell outside the mask, so the output
#   remains visually coherent rather than noisy.
# - Because minority, recolored, and mixed all share the same 3x3 geometry while
#   differing in color roles, the two-step color chain is repeatedly supported by
#   structure rather than by hidden numeric color identities.
#
# new_solver_assumptions_and_generator_guarantees:
# - The input contains exactly four distinct cell patterns.
# - Majority is the most frequent pattern.
# - Minority and majority each use exactly two colors and share exactly one color.
# - Recolored preview equals the minority geometry with
#   minority-only -> majority-only.
# - Mixed preview equals the majority template outside the mask and uses one new
#   uniform mix-color on the mask.
# - At least one minority clue, one recolored preview, and one mixed preview are
#   present in every input.
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
    """Collect cell-interior patterns and their per-cell keys."""
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
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(np.asarray(arr, dtype=int).ravel().tolist())


def _derive_recolored_overlay(minority_arr: np.ndarray, majority_arr: np.ndarray) -> tuple[np.ndarray, int, int, int, np.ndarray]:
    """
    Infer the D11 recoloring step from the minority and majority templates.

    Returns:
      mask, shared_color, majority_only_color, minority_only_color, recolored_arr
    """
    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    assert int(mask.sum()) in ACTIVE_PIXEL_COUNTS, "minority blueprint must have 6-7 active pixels"

    minority_colors = set(map(int, np.unique(minority_arr)))
    majority_colors = set(map(int, np.unique(majority_arr)))
    shared_colors = minority_colors & majority_colors
    assert len(shared_colors) == 1, "majority and minority must share exactly one color"
    shared_color = next(iter(shared_colors))

    minority_only = minority_colors - {shared_color}
    majority_only = majority_colors - {shared_color}
    assert len(minority_only) == 1 and len(majority_only) == 1, "each pattern must have one exclusive color"
    minority_only_color = next(iter(minority_only))
    majority_only_color = next(iter(majority_only))
    assert active_color == minority_only_color, "dominant minority color must be the minority-only color"
    assert np.all(majority_arr[mask] == shared_color), "active overlay must always land on shared host pixels"

    recolored_arr = np.where(mask, majority_only_color, shared_color).astype(int)
    return mask, shared_color, majority_only_color, minority_only_color, recolored_arr


def _infer_combo_roles(
    patterns: dict[tuple[int, ...], np.ndarray],
    counts: dict[tuple[int, ...], int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...], np.ndarray, np.ndarray]:
    """Identify the majority, minority, recolored, and mixed cell templates."""
    assert len(counts) == 4, "combo variant expects exactly four cell patterns"
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]
    others = [pat for pat in counts if pat != majority_pat]

    for minority_pat in others:
        minority_arr = patterns[minority_pat]
        try:
            mask, shared_color, majority_only_color, minority_only_color, recolored_arr = _derive_recolored_overlay(
                minority_arr, majority_arr
            )
        except AssertionError:
            continue

        recolored_pat = _pattern_key(recolored_arr)
        if recolored_pat not in patterns or recolored_pat in (majority_pat, minority_pat):
            continue

        remaining = [pat for pat in others if pat not in (minority_pat, recolored_pat)]
        if len(remaining) != 1:
            continue

        mixed_pat = remaining[0]
        mixed_arr = patterns[mixed_pat]
        if not np.array_equal(mixed_arr[~mask], majority_arr[~mask]):
            continue
        mix_colors = np.unique(mixed_arr[mask])
        if len(mix_colors) != 1:
            continue
        mix_color = int(mix_colors[0])
        if mix_color in (shared_color, majority_only_color, minority_only_color):
            continue

        expected_mixed = majority_arr.copy()
        expected_mixed[mask] = mix_color
        if not np.array_equal(expected_mixed, mixed_arr):
            continue
        if np.array_equal(recolored_arr, mixed_arr):
            continue
        return majority_pat, minority_pat, recolored_pat, mixed_pat, mask, mixed_arr

    raise AssertionError("failed to infer combo pattern roles")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal cell overlay by recoloring the odd-cell blueprint and then mixing it into the host pattern.

    Input structure:
    - The board is a 7x7 lattice of 3x3 cell interiors separated by full divider rows and columns.
    - Most cells share one majority template.
    - The visible exceptions come in three active-state flavors: a raw minority blueprint, a recolored preview of that blueprint, and a mixed preview showing how the recolored overlay combines with the majority cell.

    Rule:
    - Read the minority cell's dominant color as the active pixels of a 3x3 mask.
    - Infer the recoloring rule by translating the minority-only color into the majority-only foreground color, yielding the recolored preview template.
    - Use every non-majority cell as evidence for the unique placement of that 3x3 mask on the 7x7 lattice, then fill every active large-scale cell with the mixed template while leaving inactive cells as majority.

    Key insight:
    - The odd cell is still a miniature placement blueprint, but solving now requires a two-step color chain: first remap the blueprint's active color into the host palette, then recognize that the finished large-scale cells are the mixed result of overlaying that recolored stamp onto the majority template.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _extract_patterns(g, step)
    counts = _pattern_counts(cell_pat)
    majority_pat, minority_pat, recolored_pat, mixed_pat, mask, mixed_arr = _infer_combo_roles(patterns, counts)
    majority_arr = patterns[majority_pat]
    mh, mw = mask.shape
    assert mh == inner and mw == inner, "minority mask must match cell interior size"

    active_evidence = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            active_evidence[i, j] = cell_pat[i][j] in (minority_pat, recolored_pat, mixed_pat)

    candidates: list[tuple[int, int]] = []
    for top in range(0, n - mh + 1):
        for left in range(0, n - mw + 1):
            block = np.zeros_like(active_evidence)
            block[top : top + mh, left : left + mw] = True
            if np.any(active_evidence & ~block):
                continue
            seen = active_evidence[top : top + mh, left : left + mw]
            if np.any(seen & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique placement for the active blueprint"
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

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
VISIBLE_COUNT_TRIPLES = (
    (1, 1, 2),  # minority, recolored, mixed
    (1, 1, 3),
    (1, 2, 2),
)
COLOR_POOL = 10
MAX_RETRIES = 500

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks():
    """Enumerate all valid LR-symmetric 3x3 masks keyed by active-count."""
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


def _random_minority_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _random_majority_pattern(mask: np.ndarray, shared: int, maj_exclusive: int) -> np.ndarray:
    """
    Build an LR-symmetric majority pattern that uses the shared color on all
    mask=True pixels, so the mix rule has exactly one visible host case.
    """
    arr = np.full((MASK_SIZE, MASK_SIZE), shared, dtype=int)
    inactive_groups = [
        idx
        for idx, group in enumerate(_LR_GROUPS)
        if all(not mask[r, c] for r, c in group)
    ]
    assert inactive_groups, "mask must leave at least one inactive group"
    chosen_count = random.randint(1, len(inactive_groups))
    chosen = set(random.sample(inactive_groups, chosen_count))
    for idx in chosen:
        for r, c in _LR_GROUPS[idx]:
            arr[r, c] = maj_exclusive
    return arr


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority_cells: list[tuple[int, int]],
    minority: np.ndarray,
    recolored_cells: list[tuple[int, int]],
    recolored: np.ndarray,
    mixed_cells: list[tuple[int, int]],
    mixed: np.ndarray,
) -> np.ndarray:
    """Construct a full grid from the majority template plus visible active-state exceptions."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(i, j))
    for i, j in recolored_cells:
        paste(grid, recolored, *_cell_origin(i, j))
    for i, j in mixed_cells:
        paste(grid, mixed, *_cell_origin(i, j))
    return grid


def _placement_candidates(mask: np.ndarray, evidence_subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find all offsets where the observed active-evidence subset fits the 3x3 mask."""
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


def _choose_visible_partition(mask: np.ndarray, top: int, left: int):
    """
    Pick which active cells are shown as minority, recolored, and mixed previews.

    Returns:
      (minority_subset, recolored_subset, mixed_subset) in mask-local coordinates.
    """
    active = [tuple(coord) for coord in np.argwhere(mask)]
    options = list(VISIBLE_COUNT_TRIPLES)
    random.shuffle(options)
    for minority_count, recolored_count, mixed_count in options:
        total = minority_count + recolored_count + mixed_count
        if total >= len(active):
            continue
        total_subsets = [
            subset
            for subset in combinations(active, total)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        random.shuffle(total_subsets)
        for subset in total_subsets:
            subset = list(subset)
            minority_choices = list(combinations(subset, minority_count))
            random.shuffle(minority_choices)
            for minority_subset in minority_choices:
                remaining_after_minority = [pos for pos in subset if pos not in minority_subset]
                recolored_choices = list(combinations(remaining_after_minority, recolored_count))
                random.shuffle(recolored_choices)
                for recolored_subset in recolored_choices:
                    mixed_subset = tuple(pos for pos in remaining_after_minority if pos not in recolored_subset)
                    return tuple(minority_subset), tuple(recolored_subset), mixed_subset
    return None


def _gen_attempt():
    """
    Output-first generation for the combo task.

    Output: every active large-scale cell is a mixed cell.
    Input: a strict subset of active positions is shown, split across
    minority clues, recolored previews, and mixed previews.
    """
    separator, shared, maj_exclusive, min_exclusive, mix_color = random.sample(range(COLOR_POOL), 5)
    mask = _random_minority_mask()
    majority = _random_majority_pattern(mask, shared, maj_exclusive)
    minority = np.where(mask, min_exclusive, shared).astype(int)
    recolored = np.where(mask, maj_exclusive, shared).astype(int)
    mixed = majority.copy()
    mixed[mask] = mix_color

    if len({_pattern_key(majority), _pattern_key(minority), _pattern_key(recolored), _pattern_key(mixed)}) != 4:
        return None
    if not np.array_equal(majority[mask], np.full(int(mask.sum()), shared, dtype=int)):
        return None
    if not np.array_equal(mixed[~mask], majority[~mask]):
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    out = _build_grid(separator, majority, [], minority, [], recolored, active_cells, mixed)

    partition = _choose_visible_partition(mask, top, left)
    if partition is None:
        return None
    minority_subset, recolored_subset, mixed_subset = partition
    minority_cells = [(top + r, left + c) for r, c in minority_subset]
    recolored_cells = [(top + r, left + c) for r, c in recolored_subset]
    mixed_cells = [(top + r, left + c) for r, c in mixed_subset]
    inp = _build_grid(
        separator,
        majority,
        minority_cells,
        minority,
        recolored_cells,
        recolored,
        mixed_cells,
        mixed,
    )

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
