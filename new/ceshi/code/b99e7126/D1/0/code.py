# Variant of b99e7126
# Dimension: D1 — 匹配条件：异常单元格的识别方式 (match_cond)
# Change: 通过比较所有单元格的内部颜色图案，找到与多数单元格图案不同的那个唯一异常单元格... → 频率最低的图案类型作为源

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the D1 variant
# ----------------------------------
# The source is now the least frequent repeated motif family rather than the global
# least frequent exact motif. To make that distinction observable, each generated
# board includes:
# - one dominant background motif
# - one source motif family shown 3-4 times
# - two distractor motif families shown more often than the source
# - one singleton curiosity cell that must be ignored
# This forces the solver to rank repeated motif frequencies instead of simply
# grabbing the absolute minimum count, which would collapse back to the base task.


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


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Extract every cell interior as both a flat key and a 2D array."""
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
    return patterns, cell_pat


def _count_patterns(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    """Count exact cell-interior motifs across the cell grid."""
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a cell-level fractal using the least frequent repeated motif as the source.

    Input structure:
    - The board is partitioned into equal 3x3 cell interiors separated by single-color divider lines.
    - Several motif families repeat: one dominant background, one rarer source family, and larger distractor
      families; singleton oddities may also appear.

    Rule:
    - Count exact cell-interior motifs and ignore singleton motifs.
    - Among the repeated motifs, choose the uniquely least frequent one as the source blueprint.
    - Read that source cell's dominant color as a boolean 3x3 mask, locate the unique placement where the
      shown source cells fit, then fill the entire 3x3 block with source cells on active mask pixels and
      background cells on inactive mask pixels.

    Key insight: The right motif is no longer "the one odd cell" but the smallest repeated family, so the
    puzzle becomes a visible frequency-ranking task before the usual fractal completion.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _extract_cell_patterns(g, step)
    counts = _count_patterns(cell_pat)
    repeated = {pat: cnt for pat, cnt in counts.items() if cnt >= 2}
    assert len(repeated) >= 4, "need at least 4 repeated motif families"

    ordered = sorted(repeated.items(), key=lambda kv: kv[1])
    assert ordered[0][1] < ordered[1][1], "source motif frequency must be uniquely minimal among repeats"
    assert ordered[-1][1] > ordered[-2][1], "background motif frequency must be uniquely maximal"

    source_pat = ordered[0][0]
    background_pat = ordered[-1][0]
    source_arr = patterns[source_pat]
    background_arr = patterns[background_pat]

    vals, cnts = np.unique(source_arr, return_counts=True)
    order = np.argsort(cnts)[::-1]
    assert len(order) >= 2 and cnts[order[0]] > cnts[order[1]], "source motif needs a unique active color"
    active_color = int(vals[order[0]])
    mask = source_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == source_pat

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
    assert len(candidates) == 1, "source motif must admit a unique placement"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = source_arr if mask[i, j] else background_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator ---------------

CELL_GRID_N = 7
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
SOURCE_GIVEN_SIZES = (3, 4)
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


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate valid LR-symmetric 3x3 masks keyed by active-pixel count."""
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
                for i in combo:
                    for r, c in _LR_GROUPS[i]:
                        mask[r, c] = False
                if mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform LR-symmetric 3x3 pattern using exactly two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_source_mask() -> np.ndarray:
    """Pick one valid source mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _placement_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find every placement where the shown source cells could fit the source mask."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
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


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    """Choose a strict subset of active cells that uniquely determines the placement."""
    active = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
    sizes = list(SOURCE_GIVEN_SIZES)
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


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, background: np.ndarray, overrides: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Fill all cells with the background motif, then overwrite selected cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, background, *_cell_origin(i, j))
    for (i, j), pat in overrides.items():
        paste(grid, pat, *_cell_origin(i, j))
    return grid


def _positions_outside_block(top: int, left: int) -> list[tuple[int, int]]:
    """All cell coordinates outside the 3x3 completion block."""
    return [
        (i, j)
        for i in range(CELL_GRID_N)
        for j in range(CELL_GRID_N)
        if not (top <= i < top + MASK_SIZE and left <= j < left + MASK_SIZE)
    ]


def _count_signature(grid: np.ndarray) -> list[int]:
    step, _, _ = _infer_cell_grid_params(grid)
    _, cell_pat = _extract_cell_patterns(grid, step)
    return sorted(_count_patterns(cell_pat).values())


def _gen_attempt() -> np.ndarray | None:
    """Build one valid input for the least-frequent-repeated-motif variant."""
    separator, shared, maj_color, src_color, d1_color, d2_color, s_color = random.sample(range(COLOR_POOL), 7)

    background = _random_lr_pattern(shared, maj_color)
    mask = _random_source_mask()
    source = np.where(mask, src_color, shared).astype(int)
    distractor1 = _random_lr_pattern(shared, d1_color)
    distractor2 = _random_lr_pattern(shared, d2_color)
    singleton = _random_lr_pattern(shared, s_color)

    pats = [background, source, distractor1, distractor2, singleton]
    if any(np.array_equal(pats[i], pats[j]) for i in range(len(pats)) for j in range(i + 1, len(pats))):
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]

    source_count = len(given_cells)
    d1_count = source_count + random.choice((2, 3))
    d2_count = d1_count + random.choice((2, 3))

    free_cells = _positions_outside_block(top, left)
    need = d1_count + d2_count + 1
    if len(free_cells) < need:
        return None
    chosen = random.sample(free_cells, need)
    d1_cells = chosen[:d1_count]
    d2_cells = chosen[d1_count:d1_count + d2_count]
    singleton_cell = chosen[-1]

    input_overrides = {(i, j): source for i, j in given_cells}
    input_overrides.update({(i, j): distractor1 for i, j in d1_cells})
    input_overrides.update({(i, j): distractor2 for i, j in d2_cells})
    input_overrides[singleton_cell] = singleton

    output_overrides = {(i, j): source for i, j in active_cells}
    output_overrides.update({(i, j): distractor1 for i, j in d1_cells})
    output_overrides.update({(i, j): distractor2 for i, j in d2_cells})
    output_overrides[singleton_cell] = singleton

    inp = _build_grid(separator, background, input_overrides)
    out = _build_grid(separator, background, output_overrides)

    sig = _count_signature(inp)
    assert len(sig) == 5
    assert sig[0] == 1
    assert sig[1] == source_count and sig[1] < sig[2] < sig[3] < sig[4]

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
    raise RuntimeError("failed to generate a valid b99e7126 D1 input")
