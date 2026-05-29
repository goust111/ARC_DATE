# Variant of b99e7126
# Dimension: D11 — 颜色映射：放大图案中的颜色处理 (transform)
# Change: 放大图案直接使用异常单元格内部的原始颜色，前景色区域用对应颜色填充整个单元格大小... → 颜色重映射：放大时使用不同于源的颜色

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the D11 recolor variant
# -------------------------------------------
# concretization_and_trace:
# - The 29x29 canvas still contains a 7x7 grid of 3x3 cell interiors separated by
#   1-pixel divider lines. Most cells show one repeated majority template. A few
#   visible cells show a minority template whose 3x3 pixel layout encodes the 3x3
#   placement mask at the cell-grid level.
# - Original behavior at the last step:
#   minority interior [[m,m,m],[m,s,m],[m,s,m]] would be scaled into more cells
#   using the exact same colors {m,s}.
# - Variant behavior at the last step:
#   the same mask is located in the same way, but the scaled cells use a recolored
#   template [[a,a,a],[a,s,a],[a,s,a]], where `s` is the shared color between the
#   two cell types and `a` is the majority-only foreground color. The minority-only
#   color never appears in the completed output region.
# - This keeps the fractal placement logic unchanged and moves the D11 change into
#   the transform stage only.
#
# complexity_calibration:
# - Keep the task at 4 visible colors total: divider, shared, majority-only, and
#   minority-only. No hidden arithmetic on color ids is used.
# - The recolor rule has exactly one mapping case per sample:
#   minority-only -> majority-only, shared -> shared.
# - The generator rejects cases where the recolored template would match the
#   majority template or differ in too few pixels, because then the D11 change
#   would become visually weak.
#
# hidden_assumption_audit:
# - Grid parameter inference still assumes full divider rows/cols of one color.
#   The variant does not alter that.
# - The original solver assumed the minority pattern itself is the output stamp.
#   That no longer holds, so the final writeback must use a derived template.
# - The original "active color = most frequent color in minority cell" assumption
#   still holds, so mask extraction remains valid. The generator preserves active
#   counts 6-7 to make the minority-only color dominant.
# - The original code never needed to understand color roles beyond the mask.
#   The variant now requires identifying the single shared color between the
#   majority and minority cells, plus the unique majority-only color. The
#   generator guarantees exactly one shared color.
#
# placement_legality:
# - Objects do not move differently from the original task. The 3x3 mask is still
#   placed wholly inside the 7x7 cell lattice using offsets 0..4.
# - Input still shows only a strict subset of mask-active cells, chosen so the
#   placement is uniquely recoverable.
#
# visual_observability_and_cleanliness:
# - The recolored output remains a clean single overlay on the same cell lattice;
#   there are no extra overlapping object classes.
# - The key visible cue is that cells participating in the completed fractal all
#   share the minority geometry, but their dominant color switches to the other
#   foreground color already demonstrated by the majority cells.
# - Because the recolored template uses the majority palette but a different pixel
#   arrangement, the completed region is distinct from both the input minority and
#   the ordinary majority cells.
#
# new_solver_assumptions_and_generator_guarantees:
# - `main()` assumes the majority and minority cell interiors each use exactly two
#   colors and share exactly one color. `_gen_attempt()` constructs them that way.
# - `main()` assumes the recolored template differs from the majority template so
#   writing active positions is observable. `_gen_attempt()` enforces a minimum
#   pixel difference.


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


def _derive_recolored_pattern(minority_arr: np.ndarray, majority_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, recolored_pattern) for the D11 transform."""
    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color

    minority_colors = set(map(int, np.unique(minority_arr)))
    majority_colors = set(map(int, np.unique(majority_arr)))
    shared_colors = minority_colors & majority_colors
    assert len(shared_colors) == 1, "majority/minority must share exactly one color"
    shared_color = next(iter(shared_colors))

    minority_only = minority_colors - {shared_color}
    majority_only = majority_colors - {shared_color}
    assert len(minority_only) == 1 and len(majority_only) == 1, "each template must have one exclusive color"
    assert active_color == next(iter(minority_only)), "minority active color must be the minority-only color"

    mapped_active = next(iter(majority_only))
    recolored = np.where(mask, mapped_active, shared_color).astype(int)
    return mask, recolored


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a cell-grid fractal by enlarging the odd cell's pattern with recolored active cells.

    Input structure:
    - The board is partitioned into equal 3x3 cell interiors separated by full divider rows and columns.
    - Most cells share one repeated interior template; a smaller set of cells shows a distinct minority template.

    Rule:
    - Read the minority cell as a 3x3 mask, using its dominant color as the active pixels.
    - Find the unique placement of that 3x3 mask inside the larger cell grid so the visible minority cells fit it.
    - Fill every active cell position with the minority geometry recolored into the majority palette:
      keep the shared color, but replace the minority-only color with the majority-only foreground color.

    Key insight: The odd cell still acts as a miniature blueprint for the larger layout, but the enlarged copy is not literal—the active color is translated through the relationship between the majority and minority cell palettes.
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
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]
    mask, recolored_arr = _derive_recolored_pattern(minority_arr, majority_arr)
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

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = recolored_arr if mask[i, j] else majority_arr
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
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 300
MIN_RECOLOR_DIFF = 3

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
    weights = [len(group) for group in _LR_GROUPS]
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
    """Pick a random valid minority mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _placement_candidates(mask: np.ndarray, subset, top: int, left: int) -> list[tuple[int, int]]:
    """Find all offsets where the visible subset is consistent with the mask."""
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
    """Choose a subset of active mask positions that uniquely determines the placement."""
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
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, active_pattern: np.ndarray, active_cells) -> np.ndarray:
    """Construct a 29x29 grid: fill all cells with majority, then overwrite active_cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, active_pattern, *_cell_origin(i, j))
    return grid


def _gen_attempt():
    """Single generation attempt: build output, choose input subset, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    mask = _random_minority_mask()
    majority = _random_lr_pattern(shared, maj_exclusive)
    minority = np.where(mask, min_exclusive, shared).astype(int)
    recolored = np.where(mask, maj_exclusive, shared).astype(int)

    if np.array_equal(majority, recolored):
        return None
    if int(np.sum(majority != recolored)) < MIN_RECOLOR_DIFF:
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    out = _build_grid(separator, majority, recolored, active_cells)

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 input")
