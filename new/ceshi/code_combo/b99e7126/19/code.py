# Combo variant of b99e7126 (index 19)
# combo_spec: D3/1+D9/1
# Name: 放大图案铺满网格 + 输出扩展

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis for the fused variant.
# a) Concrete behavior:
#    Original rule: a 3x3 minority mask was translated inside the cell-grid and only
#    filled a local 3x3 block. Fused rule: if the input has n x n cells and the minority
#    cell interior is m x m, use scale = ceil(n / m). Each source pixel becomes a scale x
#    scale block of cells, aligned to the grid's top-left, so the full blueprint has
#    (m*scale) x (m*scale) cells. Example: n=5, m=3 -> scale=2 -> full blueprint is 6x6.
#    The input shows only the top-left 5x5 crop with a sparse subset of minority cells;
#    the output restores the full 6x6 cell-grid.
# b) Complexity calibration:
#    Only two cell interiors remain in play (majority/minority), and scale is restricted
#    to small integers (2 or 3). This keeps the visual rule strong: the output clearly
#    becomes block-scaled and often larger, without introducing arbitrary translations.
# c) Assumption audit:
#    The original solver assumed a unique (top,left) placement of an m x m block inside
#    the n x n grid. That assumption is invalid here. We replace it with deterministic
#    top-left full-cover scaling. The old in-bounds requirement is also removed because
#    the full scaled blueprint may extend beyond the input crop.
# d) Placement legality:
#    The generator first builds the complete scaled blueprint, then crops its top-left
#    n x n cells for the input, and finally keeps only a strict subset of visible active
#    cells. This guarantees no overlap, no out-of-bounds writes, and a valid expanded
#    output canvas.
# e) Visual observability:
#    Masks span all four edges and every row/column, so the missing border extension is
#    visible. The expanded output is not a subtle pixel edit; it adds whole cell rows and
#    columns and completes large repeated blocks.
# f) Visual cleanliness:
#    Cell interiors use only two colors each, with left-right symmetry, so the grid stays
#    readable even after scaling. Expansion happens as a clean continuation of the same
#    block layout rather than as a second unrelated overlay.
# g) Color inference:
#    No numeric color ordering is used. The active mask is inferred from the more frequent
#    color inside the minority cell, and the generator guarantees that this majority color
#    is unique. Majority/minority cells share one color and differ by one exclusive color.
# 9) New solver assumptions guaranteed by the generator:
#    - exactly two cell interior patterns appear;
#    - the input minority cells are a subset of the visible crop of the top-left aligned
#      scaled blueprint;
#    - at least one visible active cell is omitted and at least one active cell lies
#      outside the input crop, so both completion and expansion are exercised.


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """
    Infer a regular cell grid separated by full rows/cols of the same line color.
    Returns (step, row_divs, col_divs).
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
    inner = step - 1
    n_rows = (g.shape[0] - 1) // step
    n_cols = (g.shape[1] - 1) // step
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(n_rows):
        row: list[tuple[int, ...]] = []
        for j in range(n_cols):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def _render_cell_grid(separator: int, majority: np.ndarray, minority: np.ndarray, active: np.ndarray) -> np.ndarray:
    inner = majority.shape[0]
    step = inner + 1
    h_cells, w_cells = active.shape
    out = np.full((h_cells * step + 1, w_cells * step + 1), int(separator), dtype=int)
    for i in range(h_cells):
        for j in range(w_cells):
            paste(out, minority if active[i, j] else majority, i * step + 1, j * step + 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the minority cell as a global blueprint, scale it to cover the whole grid, and
    extend the output canvas if the full scaled blueprint is larger than the input crop.

    Input structure:
    - A divider-lined cell grid whose cells all share one of two interior patterns:
      a common majority pattern and a rarer minority pattern.
    - The minority cell's interior is itself an m x m pixel design; only a few of the
      grid-scale active locations are already shown as minority cells in the input.

    Rule:
    - Extract the minority cell's interior mask using its more frequent color as "active".
    - Let n be the input cell-grid width/height and use scale = ceil(n / m).
    - Expand each mask pixel to a scale x scale block of cells, aligned to the top-left,
      producing an m*scale by m*scale cell blueprint that fully covers the input crop.
    - Render the complete blueprint with minority cells on active blocks and majority
      cells elsewhere; if the full blueprint is larger than the input, extend the output.

    Key insight:
    The minority cell no longer tells you where to place a small patch. It is the coarse
    template for the entire final board, and the input is only a cropped, partially filled
    view of that larger block-scaled template.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n_rows = (g.shape[0] - 1) // step
    n_cols = (g.shape[1] - 1) // step

    patterns, cell_pat = _extract_cell_patterns(g, step)
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n_rows):
        for j in range(n_cols):
            p = cell_pat[i][j]
            counts[p] = counts.get(p, 0) + 1
    assert len(counts) == 2, "expected exactly 2 distinct cell patterns"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[int(np.argmax(cnts))])
    mask = minority_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n_rows, n_cols), dtype=bool)
    for i in range(n_rows):
        for j in range(n_cols):
            existing[i, j] = cell_pat[i][j] == minority_pat

    scale_r = (n_rows + mh - 1) // mh
    scale_c = (n_cols + mw - 1) // mw
    full_active = np.kron(mask.astype(bool), np.ones((scale_r, scale_c), dtype=bool))
    visible_active = full_active[:n_rows, :n_cols]
    assert not np.any(existing & ~visible_active), "input minority cells must match visible scaled blueprint"

    return _render_cell_grid(int(g[0, 0]), majority_arr, minority_arr, full_active)


# --------------- generator ---------------

INNER_CHOICES = (3, 4)
GRID_CHOICES_BY_INNER = {
    3: (5, 7),
    4: (6, 7),
}
ACTIVE_COUNTS_BY_SIZE = {
    3: (6, 7),
    4: (10, 12),
}
GIVEN_CELL_SIZES = (3, 4, 5)
COLOR_POOL = 10
MAX_RETRIES = 400


def _lr_groups(size: int) -> list[tuple[tuple[int, int], ...]]:
    groups: list[tuple[tuple[int, int], ...]] = []
    for r in range(size):
        for c in range((size + 1) // 2):
            mate = size - 1 - c
            if c == mate:
                groups.append(((r, c),))
            else:
                groups.append(((r, c), (r, mate)))
    return groups


_LR_GROUPS_BY_SIZE = {size: _lr_groups(size) for size in INNER_CHOICES}


def _mask_ok(mask: np.ndarray) -> bool:
    size = mask.shape[0]
    if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
        return False
    for idx in range(size):
        if not mask[idx, :].any():
            return False
        if not mask[:, idx].any():
            return False
    if len({tuple(row.tolist()) for row in mask}) == 1:
        return False
    if len({tuple(mask[:, c].tolist()) for c in range(size)}) == 1:
        return False
    return True


def _precompute_masks() -> dict[int, dict[int, list[np.ndarray]]]:
    by_size: dict[int, dict[int, list[np.ndarray]]] = {}
    for size in INNER_CHOICES:
        groups = _LR_GROUPS_BY_SIZE[size]
        weights = [len(group) for group in groups]
        size_masks: dict[int, list[np.ndarray]] = {}
        total = size * size
        for active_count in ACTIVE_COUNTS_BY_SIZE[size]:
            inactive_count = total - active_count
            masks: list[np.ndarray] = []
            for k in range(1, len(groups) + 1):
                for combo in combinations(range(len(groups)), k):
                    if sum(weights[idx] for idx in combo) != inactive_count:
                        continue
                    mask = np.ones((size, size), dtype=bool)
                    for idx in combo:
                        for r, c in groups[idx]:
                            mask[r, c] = False
                    if _mask_ok(mask):
                        masks.append(mask)
            assert masks, f"no valid masks for size={size}, active_count={active_count}"
            size_masks[active_count] = masks
        by_size[size] = size_masks
    return by_size


_MASKS_BY_SIZE = _precompute_masks()


def _random_lr_pattern(size: int, color_a: int, color_b: int) -> np.ndarray:
    arr = np.empty((size, size), dtype=int)
    groups = _LR_GROUPS_BY_SIZE[size]
    choices = [random.choice((color_a, color_b)) for _ in groups]
    if all(color == choices[0] for color in choices):
        idx = random.randrange(len(choices))
        choices[idx] = color_b if choices[0] == color_a else color_a
    for group, color in zip(groups, choices):
        for r, c in group:
            arr[r, c] = int(color)
    return arr


def _random_minority_mask(size: int) -> np.ndarray:
    active_count = random.choice(ACTIVE_COUNTS_BY_SIZE[size])
    return random.choice(_MASKS_BY_SIZE[size][active_count])


def _scale_for_cover(n_cells: int, mask_size: int) -> int:
    return (n_cells + mask_size - 1) // mask_size


def _choose_given_subset(visible_active: list[tuple[int, int]], scale: int) -> list[tuple[int, int]] | None:
    if len(visible_active) < 4:
        return None
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        if size >= len(visible_active):
            continue
        for _ in range(80):
            subset = random.sample(visible_active, size)
            blocks = {(r // scale, c // scale) for r, c in subset}
            if len(blocks) < min(3, size):
                continue
            return subset
    return None


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    inner = random.choice(INNER_CHOICES)
    n_cells = random.choice(GRID_CHOICES_BY_INNER[inner])
    scale = _scale_for_cover(n_cells, inner)
    out_cells = inner * scale
    assert out_cells > n_cells

    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(inner, shared, maj_exclusive)
    mask = _random_minority_mask(inner)
    minority = np.where(mask, min_exclusive, shared).astype(int)

    full_active = np.kron(mask.astype(bool), np.ones((scale, scale), dtype=bool))
    visible_active_mask = full_active[:n_cells, :n_cells]
    visible_active = [tuple(coord) for coord in np.argwhere(visible_active_mask)]
    hidden_active = int(full_active.sum()) - int(visible_active_mask.sum())
    if hidden_active <= 0:
        return None

    given_cells = _choose_given_subset(visible_active, scale)
    if given_cells is None:
        return None

    given_map = np.zeros((n_cells, n_cells), dtype=bool)
    for r, c in given_cells:
        given_map[r, c] = True

    if int(visible_active_mask.sum()) <= int(given_map.sum()):
        return None

    inp = _render_cell_grid(separator, majority, minority, given_map)
    out = _render_cell_grid(separator, majority, minority, full_active)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        pair = _gen_attempt()
        if pair is not None:
            inp, _out = pair
            return inp
    raise RuntimeError("failed to generate a valid fused b99e7126 input")
