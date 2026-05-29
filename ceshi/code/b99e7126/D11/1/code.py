# Variant of b99e7126
# Dimension: D11 — 颜色映射：放大图案中的颜色处理 (transform)
# Change: 放大图案直接使用异常单元格内部的原始颜色，前景色区域用对应颜色填充整个单元格大小... → 颜色继承：放大后的每个单元格保留原位置的颜色信息

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis for the D11 variant
# a) Concrete behavior:
#    - The odd 3x3 cell still provides a 3x3 boolean mask, and the visible odd cells still
#      uniquely pin down where that mask sits in the 7x7 cell lattice.
#    - The changed step is the expansion phase. Original behavior stamps the same odd-cell
#      pattern into every active enlarged cell. Here, each active enlarged cell instead reveals
#      the template that belongs to that absolute grid position.
#    - Small trace: if the located block is at (top,left) and mask[1,2] is active, the original
#      solver writes the minority pattern into cell (top+1,left+2). The variant writes that
#      cell's position-coded template, inferred from repeated cells elsewhere, so color content
#      depends on location rather than on the odd cell's own palette.
# b) Complexity calibration:
#    - Pure color inheritance would be invisible if every normal cell were identical. To expose
#      the change without adding unrelated mechanics, the generator uses a repeated 3x3 template
#      field outside the target block: each cell contains one accent pixel whose location matches
#      the cell's absolute row/col modulo 3.
#    - The target block input is blank except for 3 minority anchors, so output changes occur on
#      6-7 cells instead of only on the anchors.
# c) Hidden-assumption audit of the original logic:
#    - Original `majority_pat = max(counts...)` assumes one dominant repeated cell pattern.
#      That is false now because normal cells come from 9 repeated templates plus a blank filler.
#    - Original `target_pat = minority_arr if mask[i,j] else majority` assumes every active slot
#      receives identical content. That is exactly the property D11 removes.
#    - The placement search remains valid because it depends only on the minority anchors and the
#      binary mask, not on how colors are written after placement.
# d) Placement legality:
#    - The block is still 3x3 inside a 7x7 cell grid, so `top,left` stay in 0..4.
#    - The generator only chooses anchor subsets that make the placement unique under the same
#      consistency test as the original task.
# e) Visual observability:
#    - Surrounding cells visibly repeat a higher-order 3x3 pattern, so the restored active cells
#      in the output are clearly position-dependent instead of all copying the odd cell.
#    - The odd cell still acts as the only binary stencil source, keeping the rule inferable.
# f) Visual cleanliness:
#    - Only one overlay mechanism exists: blank cells inside the target block are selectively
#      replaced by neat one-accent templates. There is no multi-layer clutter.
# g) Color-change inferability:
#    - The variant uses only 4 colors total: separator, shared blank color, repeated accent
#      color, and minority-exclusive color. The color change is tied to repeated positional
#      templates, not to hidden numeric color ids.
# h) New solver assumptions and generator guarantees:
#    - The solver assumes each residue class modulo 3 has at least one example outside the target
#      block. A contiguous 3x3 block removes exactly one cell from each class; in a 7x7 grid each
#      class appears at least 4 times, so at least 3 remain.
#    - The solver assumes non-minority cells inside the target block are blank; the generator
#      enforces that directly.


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


def _most_common(items: list[tuple[int, ...]]) -> tuple[int, ...]:
    counts: dict[tuple[int, ...], int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    assert counts, "expected at least one pattern"
    return max(counts.items(), key=lambda kv: kv[1])[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a located 3x3 template window by using the odd cell as a stencil instead of a paint stamp.

    Input structure:
    - The grid is partitioned into equal cells by divider lines.
    - Outside one 3x3 target block, cells repeat a location-coded template field; inside that block,
      most cells are blank and only a few cells share the same minority pattern.

    Rule:
    - Read the minority cell's dominant-color layout as a binary 3x3 mask.
    - Find the unique placement of that mask that matches the visible minority anchor cells.
    - For every active position in the placed block, restore the template belonging to that absolute
      grid location; inactive positions stay blank.

    Key insight:
    - The odd cell contributes only the enlarged stencil. The colors of the completed pattern come
      from the repeated position-dependent templates already present elsewhere in the grid.
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
    assert len(counts) >= 3, "need minority, blank, and position templates"

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

    block_non_minority = [
        cell_pat[top + i][left + j]
        for i in range(mh)
        for j in range(mw)
        if cell_pat[top + i][left + j] != minority_pat
    ]
    blank_pat = _most_common(block_non_minority)

    residue_templates: dict[tuple[int, int], tuple[int, ...]] = {}
    for ri in range(mh):
        for cj in range(mw):
            samples: list[tuple[int, ...]] = []
            for i in range(n):
                for j in range(n):
                    if top <= i < top + mh and left <= j < left + mw:
                        continue
                    if i % mh != ri or j % mw != cj:
                        continue
                    p = cell_pat[i][j]
                    if p == minority_pat:
                        continue
                    samples.append(p)
            residue_templates[(ri, cj)] = _most_common(samples)

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            if mask[i, j]:
                target_pat = patterns[residue_templates[((top + i) % mh, (left + j) % mw)]]
            else:
                target_pat = patterns[blank_pat]
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (D11 variant) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3]
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
    by_count = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks = []
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


def _random_minority_mask():
    """Pick a random valid minority mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _placement_candidates(mask, subset, top, left):
    """Find all offsets where the given subset of active cells is consistent with mask."""
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
    """Choose a subset of active mask positions that uniquely determines the placement."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    for size in GIVEN_CELL_SIZES:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i, j):
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _blank_pattern(shared):
    return np.full((CELL_INNER, CELL_INNER), shared, dtype=int)


def _position_template(shared, accent, i, j):
    arr = _blank_pattern(shared)
    arr[i % CELL_INNER, j % CELL_INNER] = accent
    return arr


def _minority_pattern(shared, minority_color, mask):
    return np.where(mask, minority_color, shared).astype(int)


def _build_output_grid(separator, shared, accent, mask, top, left):
    """Outside the target block use position templates; inside use blank/inherited-by-mask."""
    blank = _blank_pattern(shared)
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            if top <= i < top + MASK_SIZE and left <= j < left + MASK_SIZE:
                rel_i = i - top
                rel_j = j - left
                pat = _position_template(shared, accent, i, j) if mask[rel_i, rel_j] else blank
            else:
                pat = _position_template(shared, accent, i, j)
            paste(grid, pat, *_cell_origin(i, j))
    return grid


def _build_input_grid(separator, shared, accent, minority, mask, top, left, given_cells):
    """Hide the target block behind blank cells, leaving only a few minority anchors visible."""
    blank = _blank_pattern(shared)
    given_set = set(given_cells)
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            if top <= i < top + MASK_SIZE and left <= j < left + MASK_SIZE:
                if (i, j) in given_set:
                    pat = minority
                else:
                    pat = blank
            else:
                pat = _position_template(shared, accent, i, j)
            paste(grid, pat, *_cell_origin(i, j))
    return grid


def _gen_attempt():
    """Single generation attempt: build output, derive input, and verify with the solver."""
    separator, shared, accent, minority_color = random.sample(range(COLOR_POOL), 4)
    mask = _random_minority_mask()
    minority = _minority_pattern(shared, minority_color, mask)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [tuple(coord) for coord in np.argwhere(mask)]

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None

    out = _build_output_grid(separator, shared, accent, mask, top, left)
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_input_grid(separator, shared, accent, minority, mask, top, left, given_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 D11 input")
