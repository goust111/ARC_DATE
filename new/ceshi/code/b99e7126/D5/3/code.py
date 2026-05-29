# Variant of b99e7126
# Dimension: D5 — 变换：叠加时的覆盖规则 (transform)
# Change: 放大图案中非背景色的单元格覆写原始网格中对应位置的单元格内容... → 颜色交换叠加：叠加区域内的颜色进行对换

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis for the D5 color-swap variant
# a) Concrete behavior and trace:
#    The 29x29 grid is still a 7x7 array of 3x3 cells separated by divider lines. The rare cell still
#    carries a 3x3 binary mask, and a strict subset of that mask is visible as minority cells in the input.
#    The changed step is only the large-scale overlay:
#    - Original: every active mask position is overwritten with the minority template.
#    - Variant: every active mask position keeps its own 3x3 geometry but swaps its two visible colors.
#    Example at one target cell:
#      original majority cell [[a,b,a],[a,b,a],[b,a,b]] -> variant output [[b,a,b],[b,a,b],[a,b,a]]
#      original minority cell [[x,y,x],[x,x,x],[x,y,x]] -> variant output [[y,x,y],[y,y,y],[y,x,y]]
#    So the mask still selects which cells are affected, but the transform is "invert the local 2-color
#    cell" instead of "replace by the minority cell".
# b) Complexity calibration:
#    I keep the same 7x7 / 3x3 / 3-to-4 clue-cell setup, because the placement puzzle already has enough
#    structure. To make the new rule visible, both cell templates are forced to be non-uniform 2-color
#    patterns, so swapping colors always changes the cell. Four distinct colors are still used, which
#    prevents swapped-majority cells from collapsing into the old minority template.
# c) Hidden-assumption audit against the original main():
#    - Grid inference, cell extraction, minority/majority counting, and mask placement all only depend on
#      the input, which still contains exactly two cell patterns, so those lines remain valid.
#    - The old final loop assumed every cell in the mask bbox should be rewritten as either minority or
#      majority. That assumption is specific to the overwrite rule and becomes wrong here.
#    - The new final loop therefore only touches mask=True cells, and each touched cell must contain
#      exactly two colors so a local swap is well-defined. The generator guarantees that every cell is
#      either the 2-color majority template or the 2-color minority template.
# d) Placement legality:
#    The 3x3 scaled mask still lives wholly inside the 7x7 cell grid with top/left in 0..4. The chosen
#    visible minority subset is still required to determine a unique placement, so the solver's candidate
#    search remains safe and unambiguous.
# e) Visual observability:
#    Humans can still use the special cell as the miniature blueprint, but now the marked cells in the
#    output are recognizable because their internal colors are inverted rather than copied from one shared
#    template. Newly affected majority cells repeat the same swapped-majority pattern, making the color
#    swap rule visible across multiple locations.
# f) Visual cleanliness:
#    Only one layer of action is present: some cells stay unchanged, some cells are inverted in place.
#    Divider lines remain untouched, and no extra objects or colors are introduced.
# g) Color-change inferability:
#    The rule never depends on color ids. Each affected cell swaps only its own two visible colors, and
#    the repeated cell templates provide enough examples for a solver or human to infer that the change is
#    a local 2-color inversion rather than a global recoloring table.
# Post-edit assumption audit:
#    - _swap_two_colors() assumes exactly two colors inside a cell; generator enforces this.
#    - minority_arr's most frequent color still defines the active mask; generator keeps 6-7 active pixels.
#    - The unique placement assertion still depends on the clue subset being placement-determining;
#      generator preserves that filter exactly as before.


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


def _swap_two_colors(arr: np.ndarray) -> np.ndarray:
    """Return a copy of arr with its two present colors swapped."""
    vals = [int(v) for v in np.unique(arr)]
    assert len(vals) == 2, "color-swap cells must contain exactly two colors"
    a, b = vals
    out = arr.copy()
    mask_a = out == a
    mask_b = out == b
    out[mask_a] = b
    out[mask_b] = a
    return out


def _apply_color_swap_overlay(grid: np.ndarray, mask: np.ndarray, top: int, left: int, step: int) -> np.ndarray:
    """Swap the two colors inside each active cell selected by the scaled mask placement."""
    inner = step - 1
    out = grid.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if not bool(mask[i, j]):
                continue
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = _swap_two_colors(out[r0 : r0 + inner, c0 : c0 + inner])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the rare cell's inner pattern to the cell grid, then invert colors inside the hit cells.

    Input structure:
    - A 29x29 board is split into 7x7 cells by a single divider color.
    - Most 3x3 cell interiors share one pattern; a few clue cells share a rarer 3x3 pattern.

    Rule:
    - Read the rare cell's 3x3 interior as a binary mask, using its more frequent color as the active one.
    - Place that mask uniquely on the 7x7 cell lattice so the visible rare clue cells land on active sites.
    - For every active site in the placed mask, keep the current cell but swap its two colors in place.

    Key insight:
    - The rare cell is still a miniature map of the large-scale target region, but the completion step is
      not template copying: the marked cells are locally color-inverted.
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
    minority_arr = patterns[minority_pat]

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
    top, left = candidates[0]

    return _apply_color_swap_overlay(g, mask, top, left, step)


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
    """Generate a random LR-symmetric 3x3 2-color pattern that changes under color swap."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask() -> np.ndarray:
    """Pick a random valid minority mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _placement_candidates(mask: np.ndarray, subset, top: int, left: int) -> list[tuple[int, int]]:
    """Find all offsets where the chosen visible minority subset fits the mask."""
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


def _choose_given_subset(mask: np.ndarray, top: int, left: int):
    """Choose 3-4 active sites whose minority clues uniquely determine the mask placement."""
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


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of a cell interior inside the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, active_cells) -> np.ndarray:
    """Construct a 29x29 grid with majority everywhere and minority at selected cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt():
    """Single attempt: build an input, derive the expected swapped output, and verify the solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None

    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)
    out = _apply_color_swap_overlay(inp, mask, top, left, CELL_STEP)

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
    raise RuntimeError("failed to generate a valid b99e7126 D5 color-swap input")
