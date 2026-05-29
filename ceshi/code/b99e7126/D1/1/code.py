# Variant of b99e7126
# Dimension: D1 — 匹配条件：异常单元格的识别方式 (match_cond)
# Change: 通过比较所有单元格的内部颜色图案，找到与多数单元格图案不同的那个唯一异常单元格... → 颜色匹配：异常单元格的特定颜色标记

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis
# a) Concrete behavior and trace
#    - The outer 29x29 layout is unchanged: a 7x7 cell grid, 3x3 interiors, 1-pixel dividers.
#    - Majority cells still repeat one 3x3 pattern.
#    - The large-scale 3x3 active footprint is still encoded by a 3x3 binary mask and still
#      determines which cell locations must become the minority pattern in the output.
#    - The D1 change is only how the blueprint/source cell is found:
#      old: the source was the least frequent cell pattern itself.
#      new: one active cell is a marked template, using a dedicated marker color on every active
#      pixel of the 3x3 mask; other shown active cells already use the ordinary minority color.
#    - Small trace: suppose the active mask is
#          1 1 0
#          1 1 1
#          0 1 1
#      Old input: 3-4 shown cells all use the same minority pattern, so the solver locates the
#      template by pattern-frequency contrast.
#      New input: one shown cell uses marker-color + shared-color with exactly that mask, while
#      2-3 other shown cells use minority-color + shared-color. The solver must first find the
#      unique color that appears in exactly one cell, take that cell as the template, then use
#      the marked pixels as the large-scale footprint and fill all footprint positions with the
#      ordinary minority pattern.
# b) Complexity calibration
#    - The task still has repeated support: the ordinary minority pattern appears in multiple
#      cells, the majority pattern dominates, and the marker appears exactly once.
#    - The marker is not a single arbitrary pixel; it colors all active pixels of the source
#      cell, so the footprint is visually obvious and not a hidden metadata trick.
#    - Only five colors are used, keeping the mapping small and legible:
#      separator, majority-exclusive, shared, minority-exclusive, marker.
# c) Old assumptions that no longer hold
#    - Counting cell patterns to identify the source no longer works because input now has three
#      pattern types: majority, ordinary minority, and marked minority. The rarest pattern is
#      merely the marked template, not the pattern to propagate.
#    - Existing active cells can no longer be recovered by equality to a single minority pattern;
#      the source cell must also count as active even though its colors differ from the ordinary
#      minority cells.
#    - The mask is no longer derived from the most frequent color inside the rarest pattern.
#      It is derived specifically from the marker color inside the marked template cell.
# d) Placement legality
#    - The active footprint still fits in a 3x3 block inside the 7x7 cell grid.
#    - The marked source cell is chosen from the shown active subset, so it lies on a valid
#      active location and can safely be overwritten by the ordinary minority pattern in output.
#    - Output-first generation is preserved and every candidate input is verified with main().
# e) Visual observability
#    - Across examples, the only color that appears in exactly one cell and disappears in output
#      is the marker color. That makes its role human-detectable from 2-3 examples.
#    - The marked template and ordinary minority cells share the same 3x3 shape, so it is clear
#      that the special color marks the template rather than introducing a new geometry.
# f) Visual cleanliness
#    - There are still only cell-sized stamps and divider lines; no extra overlays are added.
#    - The source cell is part of the same active cluster logic, so the scene remains a clean
#      tiled-grid completion puzzle instead of mixing a separate legend object into the board.
# g) Color-change inferability
#    - The marker color has one structural meaning only: "this cell is the template".
#    - The minority-exclusive color has one structural meaning only: "ordinary filled active cell".
#    - Because the source cell is recolored to the ordinary minority pattern in output, the
#      marker-to-minority replacement is supported by the other already-shown minority cells.
# h) Anti-degeneration metric to inspect after implementation
#    - On generated samples, the original solver should usually fail or disagree because it still
#      expects a single minority pattern class. We will measure old-solver disagreement rate.
# i) New solver assumptions to enforce in the generator
#    - Exactly one interior color appears in exactly one cell: the marker color.
#    - At least one ordinary minority cell exists in addition to the marked source cell.
#    - Among non-majority, non-marker cells there is exactly one repeated ordinary minority
#      pattern. The generator constructs exactly that regime and re-validates it through main().


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


def _extract_cells(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    inner = step - 1
    n = (g.shape[0] - 1) // step
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the uniquely color-marked template cell to complete a fractal cell overlay.

    Input structure:
    - Grid divided into NxN cells by separator lines of one color.
    - Most cells share one majority interior pattern.
    - A few active cells show a second pattern, and exactly one of those active cells is
      distinguished by a dedicated marker color inside its 3x3 interior.

    Rule:
    - Find the unique marker color and the cell containing it; that marked cell is the template.
    - Read the template's marked pixels as a 3x3 boolean mask at cell-grid scale.
    - Use the already-shown active cells to place that mask uniquely on the 7x7 cell grid.
    - Replace every mask-active cell with the ordinary minority pattern and keep mask-inactive
      cells as the majority pattern.

    Key insight: The special color does not mean a new output color; it singles out which cell is
    the blueprint. Its marked pixels encode the large-scale footprint, while the ordinary minority
    cells reveal the pattern/coloring that must be propagated.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    patterns, cell_pat = _extract_cells(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        for j in range(n):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1
    assert len(counts) >= 3, "need majority, ordinary minority, and marked template patterns"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    color_cells: dict[int, list[tuple[int, int]]] = {}
    for i in range(n):
        for j in range(n):
            arr = patterns[cell_pat[i][j]]
            for color in np.unique(arr):
                color_cells.setdefault(int(color), []).append((i, j))

    marker_colors = [color for color, cells in color_cells.items() if len(cells) == 1]
    assert len(marker_colors) == 1, "must have exactly one marker color"
    marker_color = marker_colors[0]
    source_i, source_j = color_cells[marker_color][0]

    source_pat = cell_pat[source_i][source_j]
    source_arr = patterns[source_pat]
    mask = source_arr == marker_color
    mh, mw = mask.shape
    assert bool(mask.any()), "marked template must contain marker pixels"

    ordinary_candidates = {
        pat: cnt
        for pat, cnt in counts.items()
        if pat != majority_pat and pat != source_pat
    }
    assert ordinary_candidates, "need an ordinary minority pattern besides the marked template"
    ordinary_pat = max(ordinary_candidates.items(), key=lambda kv: kv[1])[0]
    ordinary_arr = patterns[ordinary_pat]

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            pat = cell_pat[i][j]
            if pat == ordinary_pat or (i == source_i and j == source_j):
                existing[i, j] = True

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
    assert len(candidates) == 1, "must have a unique mask placement"
    top, left = candidates[0]

    inner = step - 1
    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = ordinary_arr if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
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
    """Enumerate valid LR-symmetric 3x3 masks, keyed by active count."""
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
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _placement_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find all mask placements consistent with the shown active subset."""
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


def _choose_given_subset(mask: np.ndarray, top: int, left: int):
    """Choose a shown active subset that uniquely determines the footprint placement."""
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
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, overrides: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    """Construct a 29x29 grid with majority as default cell interior."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for (i, j), pat in overrides.items():
        paste(grid, pat, *_cell_origin(i, j))
    return grid


def _gen_attempt():
    separator, maj_exclusive, shared, ordinary_active, marker = random.sample(range(COLOR_POOL), 5)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    ordinary = np.where(mask, ordinary_active, shared).astype(int)
    source = np.where(mask, marker, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [tuple((top + r, left + c)) for r, c in np.argwhere(mask)]

    out = _build_grid(
        separator,
        majority,
        {cell: ordinary for cell in active_cells},
    )

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None

    subset_cells = [tuple((top + r, left + c)) for r, c in subset]
    source_cell = random.choice(subset_cells)
    ordinary_cells = [cell for cell in subset_cells if cell != source_cell]
    assert ordinary_cells, "shown subset must include at least one ordinary minority cell"

    overrides = {cell: ordinary for cell in ordinary_cells}
    overrides[source_cell] = source
    inp = _build_grid(separator, majority, overrides)

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
    raise RuntimeError("failed to generate a valid b99e7126 D1 input")
