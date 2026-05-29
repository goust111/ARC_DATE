# Combo variant of b99e7126 (index 16)
# combo_spec: D4/2+D8/1
# Name: 非均匀缩放 + 非正方形单元格

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the fused variant:
# a) Concrete behavior: the original task used 3x3 square cells, so the minority
#    cell's 3x3 interior became a 3x3 block of cells. Here the cell is rectangular:
#    for example, a 2x3 minority interior expands to a 2-rows-by-3-cols block of
#    cells. At pixel resolution that means scaling by 2 vertically and 3 horizontally.
# b) Complexity calibration: only two aspect ratios are used, (2x3) and (3x2). This
#    keeps the rule visible and repeated while avoiding a combinatorial explosion.
# c) Broken assumptions: the original solver assumed one shared step, one square
#    interior size, and one square mask size. All grid parsing, mask extraction, and
#    placement search below now use independent row/col sizes.
# d) Placement legality: output-first construction guarantees every minority cell is
#    inside the grid, and subset selection explicitly checks for a unique placement.
# e/f/g) Visual clarity and inferability: the separator color never appears inside
#    cells, the minority mask covers all four borders of its rectangle, and both
#    majority/minority patterns are mirror-structured. The rectangle aspect itself is
#    the clue that row and column scaling must be inferred separately.


GEOMETRY_SPECS = (
    {"grid_rows": 7, "grid_cols": 7, "inner_h": 2, "inner_w": 3},
    {"grid_rows": 7, "grid_cols": 7, "inner_h": 3, "inner_w": 2},
)
GIVEN_CELL_COUNTS = (3, 4)
COLOR_POOL = 10
MAX_RETRIES = 400


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    """
    Infer a rectangular cell grid separated by full rows/cols of one line color.
    Returns (row_step, col_step, row_divs, col_divs).
    """
    # Generator guarantee: the separator color is reserved for divider lines, so a full
    # line of that color is a reliable structural marker rather than an interior motif.
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"

    row_divs.sort()
    col_divs.sort()
    row_diffs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    col_diffs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    row_step = reduce(gcd, row_diffs)
    col_step = reduce(gcd, col_diffs)

    assert row_step > 1 and col_step > 1, "divider steps must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"
    assert (h - 1) % row_step == 0 and (w - 1) % col_step == 0, "grid dimensions must match divider steps"

    expected_rows = list(range(0, h, row_step))
    expected_cols = list(range(0, w, col_step))
    assert row_divs == expected_rows, "row dividers must be regularly spaced"
    assert col_divs == expected_cols, "col dividers must be regularly spaced"
    return int(row_step), int(col_step), row_divs, col_divs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal overlay by enlarging a minority cell's rectangular interior pattern.

    Input structure:
    - The grid is partitioned by separator lines into a regular array of rectangular cells.
    - Most cells share one interior pattern; a few cells show a distinct minority pattern.
    - The minority cell's interior is itself a non-square pixel design.

    Rule:
    - Read the minority cell's interior as a boolean mask, using its more frequent color as active.
    - Because the cell interior is MxN, its enlargement is non-uniform: rows scale by M cells and
      columns scale by N cells, producing an MxN block of cell positions.
    - Find the unique placement where the already-present minority cells fit the active pixels of
      that rectangular mask, then fill the remaining active positions with the minority pattern.

    Key insight:
    - The odd cell is a rectangular blueprint for the larger arrangement. Its aspect ratio tells
      you both the cell dimensions and the different vertical/horizontal scaling factors.
    """
    g = np.asarray(grid, dtype=int)
    row_step, col_step, _, _ = _infer_cell_grid_params(g)
    grid_rows = (g.shape[0] - 1) // row_step
    grid_cols = (g.shape[1] - 1) // col_step
    inner_h = row_step - 1
    inner_w = col_step - 1

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_patterns: list[list[tuple[int, ...]]] = []
    for i in range(grid_rows):
        row: list[tuple[int, ...]] = []
        for j in range(grid_cols):
            r0 = i * row_step + 1
            c0 = j * col_step + 1
            pat_arr = g[r0 : r0 + inner_h, c0 : c0 + inner_w].copy()
            pat_key = tuple(pat_arr.ravel().tolist())
            if pat_key not in patterns:
                patterns[pat_key] = pat_arr
            row.append(pat_key)
        cell_patterns.append(row)

    counts: dict[tuple[int, ...], int] = {}
    for i in range(grid_rows):
        for j in range(grid_cols):
            pat_key = cell_patterns[i][j]
            counts[pat_key] = counts.get(pat_key, 0) + 1
    # Generator guarantee: only majority and minority cell interiors are used.
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]

    # Generator guarantee: the active color is a strict majority inside the minority cell,
    # so the mask can be recovered without consulting raw color ids.
    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    mh, mw = mask.shape
    assert (mh, mw) == (inner_h, inner_w), "minority mask must match the rectangular cell interior"

    existing = np.zeros((grid_rows, grid_cols), dtype=bool)
    for i in range(grid_rows):
        for j in range(grid_cols):
            existing[i, j] = cell_patterns[i][j] == minority_pat

    candidates: list[tuple[int, int]] = []
    for top in range(grid_rows - mh + 1):
        for left in range(grid_cols - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique rectangular mask placement"
    top, left = candidates[0]

    out = g.copy()
    majority_arr = patterns[majority_pat]
    for i in range(mh):
        for j in range(mw):
            target = minority_arr if mask[i, j] else majority_arr
            r0 = (top + i) * row_step + 1
            c0 = (left + j) * col_step + 1
            out[r0 : r0 + inner_h, c0 : c0 + inner_w] = target
    return out


def _symmetry_groups(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    """Use LR symmetry for wide cells and TB symmetry for tall cells."""
    groups: list[tuple[tuple[int, int], ...]] = []
    if w > h:
        half = w // 2
        for r in range(h):
            for c in range(half):
                groups.append(((r, c), (r, w - 1 - c)))
            if w % 2 == 1:
                groups.append(((r, half),))
    else:
        half = h // 2
        for r in range(half):
            for c in range(w):
                groups.append(((r, c), (h - 1 - r, c)))
        if h % 2 == 1:
            mid = half
            for c in range(w):
                groups.append(((mid, c),))
    return groups


def _active_count_choices(h: int, w: int) -> tuple[int, ...]:
    total = h * w
    lo = total // 2 + 1
    hi = min(total - 1, lo + 1)
    return tuple(range(lo, hi + 1))


def _precompute_masks(h: int, w: int) -> dict[int, list[np.ndarray]]:
    groups = _symmetry_groups(h, w)
    weights = [len(group) for group in groups]
    by_count: dict[int, list[np.ndarray]] = {}
    total = h * w
    for active_count in _active_count_choices(h, w):
        inactive_count = total - active_count
        masks: list[np.ndarray] = []
        for k in range(len(groups) + 1):
            for combo in combinations(range(len(groups)), k):
                if sum(weights[i] for i in combo) != inactive_count:
                    continue
                mask = np.ones((h, w), dtype=bool)
                for idx in combo:
                    for r, c in groups[idx]:
                        mask[r, c] = False
                if not np.any(~mask):
                    continue
                if not mask[0, :].any():
                    continue
                if not mask[-1, :].any():
                    continue
                if not mask[:, 0].any():
                    continue
                if not mask[:, -1].any():
                    continue
                masks.append(mask)
        assert masks, f"no valid masks for shape {(h, w)}"
        by_count[active_count] = masks
    return by_count


_MASKS_BY_SHAPE = {
    (spec["inner_h"], spec["inner_w"]): _precompute_masks(spec["inner_h"], spec["inner_w"])
    for spec in GEOMETRY_SPECS
}


def _random_structured_pattern(h: int, w: int, color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform symmetric pattern using two colors."""
    groups = _symmetry_groups(h, w)
    arr = np.empty((h, w), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in groups]
    if all(choice == choices[0] for choice in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(groups, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask(h: int, w: int) -> np.ndarray:
    by_count = _MASKS_BY_SHAPE[(h, w)]
    active_count = random.choice(sorted(by_count))
    return random.choice(by_count[active_count])


def _placement_candidates(
    mask: np.ndarray,
    subset: tuple[tuple[int, int], ...],
    grid_rows: int,
    grid_cols: int,
    top: int,
    left: int,
) -> list[tuple[int, int]]:
    existing = np.zeros((grid_rows, grid_cols), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for t in range(grid_rows - mh + 1):
        for l in range(grid_cols - mw + 1):
            block = np.zeros((grid_rows, grid_cols), dtype=bool)
            block[t : t + mh, l : l + mw] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[t : t + mh, l : l + mw] & ~mask):
                continue
            candidates.append((t, l))
    return candidates


def _choose_given_subset(mask: np.ndarray, grid_rows: int, grid_cols: int, top: int, left: int):
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = [size for size in GIVEN_CELL_COUNTS if 1 < size < len(active)]
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(mask, subset, grid_rows, grid_cols, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i: int, j: int, row_step: int, col_step: int) -> tuple[int, int]:
    return i * row_step + 1, j * col_step + 1


def _build_grid(
    grid_rows: int,
    grid_cols: int,
    row_step: int,
    col_step: int,
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    active_cells: list[tuple[int, int]],
) -> np.ndarray:
    height = grid_rows * row_step + 1
    width = grid_cols * col_step + 1
    grid = np.full((height, width), separator, dtype=int)
    for i in range(grid_rows):
        for j in range(grid_cols):
            paste(grid, majority, *_cell_origin(i, j, row_step, col_step))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j, row_step, col_step))
    return grid


def _gen_attempt():
    spec = random.choice(GEOMETRY_SPECS)
    grid_rows = spec["grid_rows"]
    grid_cols = spec["grid_cols"]
    inner_h = spec["inner_h"]
    inner_w = spec["inner_w"]
    row_step = inner_h + 1
    col_step = inner_w + 1

    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_structured_pattern(inner_h, inner_w, shared, maj_exclusive)
    mask = _random_minority_mask(inner_h, inner_w)
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(grid_rows - inner_h + 1)
    left = random.randrange(grid_cols - inner_w + 1)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    out = _build_grid(grid_rows, grid_cols, row_step, col_step, separator, majority, minority, active_cells)

    subset = _choose_given_subset(mask, grid_rows, grid_cols, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(grid_rows, grid_cols, row_step, col_step, separator, majority, minority, given_cells)

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
    raise RuntimeError("failed to generate a valid fused b99e7126 input")
