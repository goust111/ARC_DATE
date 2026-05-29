# Combo variant of b99e7126 (index 53)
# combo_spec: D6/0+D11/1
# Name: 颜色继承 + 棋盘铺贴

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the combo variant
# a) Concrete behavior:
#    The canvas is still a 29x29 divider grid containing a 7x7 lattice of 3x3 cell interiors.
#    The hidden "original world" is a checkerboard: parity-even cells use base tile A and parity-odd
#    cells use base tile B. A third tile C is the anomaly stencil, and a fourth tile D is a blank
#    occluder. The input shows the checkerboard outside one 3x3 macro-block; inside that block,
#    only 3-4 active cells are replaced by C and every other position is hidden by D. The output
#    keeps D on mask-inactive positions but restores the checkerboard A/B tile on every mask-active
#    position. Small trace: if the located block sits at (top,left) and mask[1,2] is active, the
#    old D6-only logic would write C into cell (top+1,left+2). The fused logic instead writes A or
#    B according to the absolute parity of that cell, so neighboring active slots can inherit
#    different colors in checkerboard order.
# b) Complexity calibration:
#    Pure checkerboard inheritance would be invisible if the whole 3x3 block already showed A/B
#    everywhere, because the output would collapse to "remove the anomaly". To expose D11 clearly,
#    the generator hides the whole target block behind blank tile D, leaving only anomaly anchors.
#    This creates exactly four recurring cell types with strong repetition: A, B, D, and rare C.
# c) Hidden-assumption audit of the original solver:
#    - The original single `majority_pat` assumption is broken twice: the base is now A/B rather
#      than one tile, and the hidden area uses blank D rather than the background.
#    - The original "active pixels stamp the anomaly tile" assumption is exactly what D11 removes;
#      active positions must inherit their local checkerboard base tile instead.
#    - Placement search from anomaly anchors still works because it only depends on the boolean mask.
#    - New inference steps are required: identify the unique blank tile, infer the A/B parity
#      assignment, then use parity to restore active cells.
# d) Placement legality:
#    The macro-block remains 3x3 within a 7x7 cell lattice, so offsets stay in 0..4. The generator
#    never overlaps objects because each cell occupies one fixed 3x3 interior and only one of
#    {A, B, C, D} is written into a cell. Chosen anchor subsets must still determine the offset
#    uniquely under the same mask-consistency test as the original task.
# e) Visual observability:
#    A and B are distinct asymmetric motifs built from the same two colors, so the checkerboard is
#    readable by eye. D is a solid shared-color blank, making the restored active cells inside the
#    target block visibly alternate between inherited A/B motifs rather than all copying C.
# f) Visual cleanliness:
#    Each cell still contains exactly one 3x3 motif. The fused effect is produced by swapping which
#    motif a cell receives, not by stacking marks. Divider lines remain untouched.
# g) Color-change inferability:
#    The palette stays at 4 colors total: separator, shared, base-exclusive, anomaly-exclusive.
#    The inherited color difference is explained structurally by checkerboard parity, not by hidden
#    numeric color ids. Blank D uses only the shared color, so it reads as an intentional mask.
# h) Fusion-behavior metric:
#    A generated case is non-degenerate only if the solved 3x3 block contains both inherited base
#    tiles A and B among its active cells. With 6-7 active pixels this is easy to guarantee and it
#    proves the combo is not acting like the old single-pattern stamp rule.
# i) New solver assumptions and generator guarantees:
#    - The solver assumes exactly one uniform cell pattern exists; the generator makes only D uniform.
#    - The solver assumes the remaining non-blank, non-anomaly cells fit one checkerboard assignment;
#      the generator fills every visible base cell accordingly.
#    - The solver assumes the rare anomaly count is smaller than the blank count; generator uses
#      3-4 anchors and 5-6 blank cells inside the hidden block to enforce that ordering.


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


def _extract_cell_patterns(
    g: np.ndarray, step: int
) -> tuple[int, int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
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
            pat = tuple(int(v) for v in pat_arr.ravel())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return n, inner, patterns, cell_pat


def _checkerboard_score(
    cell_pat: list[list[tuple[int, ...]]],
    ignored: set[tuple[int, ...]],
    even_pat: tuple[int, ...],
    odd_pat: tuple[int, ...],
) -> int:
    mismatches = 0
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat in ignored:
                continue
            expected = even_pat if (i + j) % 2 == 0 else odd_pat
            mismatches += int(pat != expected)
    return mismatches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate a hidden macro-stencil and reveal only its checkerboard-colored active cells.

    Input structure:
    - The grid is partitioned into equal cells by one-color divider lines.
    - Outside one 3x3 target block, visible cells alternate between two base motifs in checkerboard order.
    - Inside that block, most cells are replaced by one blank motif, and only a few cells show the same rare anomaly motif.

    Rule:
    - Read the anomaly motif as a binary 3x3 mask by taking its most frequent color as the active color.
    - Find the unique placement of that mask whose active positions contain all visible anomaly anchors.
    - Keep blank cells on mask-inactive positions, but on every mask-active position restore the checkerboard base
      motif that belongs to that absolute cell parity.

    Key insight:
    - The rare anomaly tile is only a locator and stencil. The colors of the completed enlarged pattern come from
      the alternating A/B checkerboard already present elsewhere, so the restored active region inherits location-dependent color.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n, inner, patterns, cell_pat = _extract_cell_patterns(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) == 4, "combo variant expects exactly 4 cell patterns: A, B, blank, anomaly"

    uniform_pats = [pat for pat in counts if np.all(patterns[pat] == patterns[pat][0, 0])]
    assert len(uniform_pats) == 1, "expected exactly one uniform blank pattern"
    blank_pat = uniform_pats[0]

    anomaly_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    assert anomaly_pat != blank_pat, "anomaly and blank must be different patterns"

    base_pats = [pat for pat in counts if pat not in {blank_pat, anomaly_pat}]
    assert len(base_pats) == 2, "need exactly two checkerboard base patterns"

    assignments = []
    ignored = {blank_pat, anomaly_pat}
    for even_pat, odd_pat in ((base_pats[0], base_pats[1]), (base_pats[1], base_pats[0])):
        score = _checkerboard_score(cell_pat, ignored, even_pat, odd_pat)
        assignments.append((score, even_pat, odd_pat))
    assignments.sort(key=lambda item: item[0])
    assert assignments[0][0] == 0, "visible base cells must fit one checkerboard assignment exactly"
    assert assignments[0][0] < assignments[1][0], "checkerboard parity assignment must be unique"
    _, even_pat, odd_pat = assignments[0]

    anomaly_arr = patterns[anomaly_pat]
    vals, cnts = np.unique(anomaly_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = anomaly_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == anomaly_pat

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
    assert len(candidates) == 1, "anomaly mask placement must be uniquely determined"
    top, left = candidates[0]

    out = g.copy()
    blank_arr = patterns[blank_pat]
    for i in range(mh):
        for j in range(mw):
            abs_i = top + i
            abs_j = left + j
            if mask[i, j]:
                base_pat = even_pat if (abs_i + abs_j) % 2 == 0 else odd_pat
                target_pat = patterns[base_pat]
            else:
                target_pat = blank_arr
            r0 = abs_i * step + 1
            c0 = abs_j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (combo) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
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

_BASE_SEEDS = [
    np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=bool),
]


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate valid anomaly masks with visible edge coverage."""
    weights = [len(group) for group in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[idx] for idx in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for idx in combo:
                    for r, c in _LR_GROUPS[idx]:
                        mask[r, c] = False
                if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
                    continue
                masks.append(mask)
        by_count[active_count] = masks
    return by_count


def _precompute_base_masks() -> list[np.ndarray]:
    """Build a catalog of asymmetric 3x3 motifs for the checkerboard base."""
    seen: set[tuple[int, ...]] = set()
    catalog: list[np.ndarray] = []
    for seed in _BASE_SEEDS:
        variants = [
            seed,
            np.rot90(seed, 1),
            np.rot90(seed, 2),
            np.rot90(seed, 3),
            np.fliplr(seed),
            np.flipud(seed),
        ]
        for variant in variants:
            key = tuple(int(v) for v in variant.ravel())
            if key not in seen:
                seen.add(key)
                catalog.append(variant.copy())
    return catalog


_MASKS_BY_COUNT = _precompute_masks()
_BASE_MASKS = _precompute_base_masks()


def _random_minority_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _random_checkerboard_pair(shared: int, base_exclusive: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(50):
        mask_a, mask_b = random.sample(_BASE_MASKS, 2)
        if np.count_nonzero(mask_a != mask_b) >= 3:
            arr_a = np.where(mask_a, base_exclusive, shared).astype(int)
            arr_b = np.where(mask_b, base_exclusive, shared).astype(int)
            return arr_a, arr_b
    raise RuntimeError("failed to choose distinct checkerboard motifs")


def _placement_candidates(
    mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int
) -> list[tuple[int, int]]:
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


def _subset_spans_both_parities(subset: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    parities = {(top + r + left + c) % 2 for r, c in subset}
    return len(parities) == 2


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    active = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _subset_spans_both_parities(subset, top, left)
            and _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _blank_pattern(shared: int) -> np.ndarray:
    return np.full((CELL_INNER, CELL_INNER), shared, dtype=int)


def _minority_pattern(shared: int, anomaly_color: int, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, anomaly_color, shared).astype(int)


def _base_tile(even_pat: np.ndarray, odd_pat: np.ndarray, i: int, j: int) -> np.ndarray:
    return even_pat if (i + j) % 2 == 0 else odd_pat


def _build_output_grid(
    separator: int,
    even_pat: np.ndarray,
    odd_pat: np.ndarray,
    blank: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            if top <= i < top + MASK_SIZE and left <= j < left + MASK_SIZE:
                rel_i = i - top
                rel_j = j - left
                pat = _base_tile(even_pat, odd_pat, i, j) if mask[rel_i, rel_j] else blank
            else:
                pat = _base_tile(even_pat, odd_pat, i, j)
            paste(grid, pat, *_cell_origin(i, j))
    return grid


def _build_input_grid(
    separator: int,
    even_pat: np.ndarray,
    odd_pat: np.ndarray,
    blank: np.ndarray,
    anomaly_pat: np.ndarray,
    top: int,
    left: int,
    given_cells: list[tuple[int, int]],
) -> np.ndarray:
    given_set = set(given_cells)
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            if top <= i < top + MASK_SIZE and left <= j < left + MASK_SIZE:
                pat = anomaly_pat if (i, j) in given_set else blank
            else:
                pat = _base_tile(even_pat, odd_pat, i, j)
            paste(grid, pat, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    separator, base_exclusive, shared, anomaly_color = random.sample(range(COLOR_POOL), 4)
    even_pat, odd_pat = _random_checkerboard_pair(shared, base_exclusive)
    blank = _blank_pattern(shared)
    mask = _random_minority_mask()
    anomaly_pat = _minority_pattern(shared, anomaly_color, mask)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None

    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_input_grid(separator, even_pat, odd_pat, blank, anomaly_pat, top, left, given_cells)
    out = _build_output_grid(separator, even_pat, odd_pat, blank, mask, top, left)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    active_base_keys = {
        tuple(int(v) for v in _base_tile(even_pat, odd_pat, top + r, left + c).ravel())
        for r, c in np.argwhere(mask)
    }
    if len(active_base_keys) != 2:
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
