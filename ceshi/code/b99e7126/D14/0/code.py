# Variant of b99e7126
# Dimension: D14 — 网格边界处理 (meta)
# Change: 放大图案可能超出网格边界的部分被忽略（仅在网格范围内叠加）... → 环绕式叠加：超出边界的部分从对侧继续

from functools import reduce
from math import gcd

import numpy as np
import random
from itertools import combinations
from arc_utils import paste

# Design analysis for the D14 variant
# ----------------------------------
# concrete_behavior:
#   - The original task enlarges the minority cell's 3x3 interior into a 3x3 block
#     of whole cells placed inside the 7x7 cell lattice.
#   - In this variant the enlarged 3x3 block lives on a torus: if the chosen top-left
#     offset is (6, 5), then local cell (0, 0) lands at global (6, 5), local (1, 0)
#     lands at (0, 5), and local (2, 2) lands at (1, 0). The pattern is visibly split
#     across opposite borders but is still one contiguous stencil under wraparound.
#   - Original behavior at that stage relied on a plain slice `top:top+3, left:left+3`,
#     so anything beyond row/col 6 could not be represented. The new behavior replaces
#     that rectangle with modulo-7 addressing on both axes.
# complexity_calibration:
#   - The cell interiors, colors, and minority-mask semantics stay unchanged so the
#     puzzle remains recognizable as b99e7126.
#   - To make D14 observable, generation always chooses a placement that wraps on the
#     row axis, column axis, or both. This avoids accidental regression to the original
#     in-bounds-only regime.
#   - The mask still has 6-7 active pixels and spans all four edges of its 3x3 frame,
#     so a wrapped placement produces clear border-crossing evidence instead of a tiny,
#     ambiguous fragment.
# implicit_assumptions_in_original_main:
#   - `_infer_cell_grid_params` only assumes a regular divider grid; that remains valid.
#   - Pattern extraction/counting only assumes exactly two recurring cell interiors in
#     the input; generation still guarantees majority vs minority only.
#   - The old candidate loop `for top in range(0, n-mh+1)` assumes the enlarged mask is
#     a contiguous in-bounds rectangle. That is false under D14 and must become a search
#     over all `0..n-1` toroidal offsets.
#   - The old `block[top:top+mh, left:left+mw] = True` also assumes non-wrapping slices.
#     It is replaced by an explicit set of wrapped coordinates.
#   - The old output write `out[r0:r0+inner, c0:c0+inner] = ...` was harmless only because
#     the candidate itself never wrapped. Under D14 we must compute each target cell by
#     modulo and write one cell interior at a time.
# placement_legality:
#   - With `n=7` and mask size `3`, every wrapped `(top+r) % 7, (left+c) % 7` position is
#     distinct, so the enlarged stencil never self-overlaps.
#   - The generator only emits inputs whose existing minority subset identifies exactly one
#     toroidal offset under the new solver.
# observability_and_tidiness:
#   - Minority and majority interiors remain two-color 3x3 motifs, so the only new visual
#     event is boundary crossing, which is easy to compare across examples.
#   - Using at most 4 colors preserves readability; no hidden ordering or color-id trick is
#     introduced.
# new_solver_assumptions_checked_against_generator:
#   - The minority interior's dominant color is still the active mask color because active
#     pixels count is fixed to 6 or 7.
#   - There is exactly one valid toroidal placement candidate for the input subset.
#   - Every cell touched by the wrapped 3x3 stencil is either meant to be minority (mask=1)
#     or majority (mask=0); generation builds outputs from that exact rule and verifies them
#     with `main()` before returning an input.


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


def _mask_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, coord)) for coord in np.argwhere(mask)]


def _placed_cells(mask: np.ndarray, top: int, left: int, n: int) -> set[tuple[int, int]]:
    return {
        ((top + r) % n, (left + c) % n)
        for r, c in _mask_coords(mask)
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the odd cell's motif to the cell-grid level and complete it with wraparound.

    Input structure:
    - The board is partitioned into a 7x7-style lattice of equal cells by single-color divider lines.
    - Almost every cell shares one interior pattern, while a small subset uses one distinct minority pattern.
    - Those visible minority cells may already be split across opposite borders because the large-scale placement is toroidal.

    Rule:
    - Read the minority cell's 3x3 interior and treat its dominant color as the active mask.
    - Search for the unique placement of that 3x3 mask on the cell lattice, but with row/column indices wrapping modulo the lattice size.
    - Repaint every cell reached by the wrapped stencil: active mask positions get the minority pattern, inactive positions get the majority pattern.

    Key insight:
    - The odd cell still encodes a self-similar blueprint, but the enlarged copy sits on a periodic grid, so any part that leaves one edge re-enters from the opposite side.
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

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = (minority_arr == active_color)
    mh, mw = mask.shape

    existing: set[tuple[int, int]] = set()
    for i in range(n):
        for j in range(n):
            if cell_pat[i][j] == minority_pat:
                existing.add((i, j))

    candidates: list[tuple[int, int]] = []
    for top in range(n):
        for left in range(n):
            placed = _placed_cells(mask, top, left, n)
            if existing <= placed:
                candidates.append((top, left))
    assert len(candidates) == 1, "must have a unique toroidal mask placement aligning existing minority cells"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if mask[i, j] else patterns[majority_pat]
            rr = (top + i) % n
            cc = (left + j) % n
            r0 = rr * step + 1
            c0 = cc * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 300

NON_WRAP_OFFSETS = tuple(range(CELL_GRID_N - MASK_SIZE + 1))
WRAP_OFFSETS = tuple(range(CELL_GRID_N - MASK_SIZE + 1, CELL_GRID_N))
WRAP_BRANCHES = ("row", "col", "both")

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


def _random_lr_pattern(color_a, color_b):
    """Generate a random LR-symmetric 3x3 pattern using two colors (guaranteed non-uniform)."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask():
    """Pick a random valid minority mask (uniform over active count, then uniform within count)."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _wrapped_active_cells(mask, top, left):
    return [
        ((top + r) % CELL_GRID_N, (left + c) % CELL_GRID_N)
        for r, c in _mask_coords(mask)
    ]


def _placement_candidates(mask, subset_cells):
    """Find all toroidal offsets where the given visible minority cells are consistent with mask."""
    existing = set(subset_cells)
    candidates = []
    for t in range(PLACEMENT_RANGE):
        for l in range(PLACEMENT_RANGE):
            placed = _placed_cells(mask, t, l, CELL_GRID_N)
            if existing <= placed:
                candidates.append((t, l))
    return candidates


def _choose_given_subset(mask, top, left):
    """Choose a subset of active mask positions that uniquely determines the toroidal placement."""
    placed = _wrapped_active_cells(mask, top, left)
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(placed, size)
            if _placement_candidates(mask, subset) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i, j):
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator, majority, minority, active_cells):
    """Construct a 29x29 grid: fill all cells with majority, then overwrite active_cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _choose_wrapping_offset():
    branch = random.choice(WRAP_BRANCHES)
    if branch == "row":
        return random.choice(WRAP_OFFSETS), random.choice(NON_WRAP_OFFSETS)
    if branch == "col":
        return random.choice(NON_WRAP_OFFSETS), random.choice(WRAP_OFFSETS)
    return random.choice(WRAP_OFFSETS), random.choice(WRAP_OFFSETS)


def _gen_attempt():
    """Single generation attempt: build output, choose input subset, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top, left = _choose_wrapping_offset()
    active_cells = _wrapped_active_cells(mask, top, left)
    out = _build_grid(separator, majority, minority, active_cells)

    given_cells = _choose_given_subset(mask, top, left)
    if given_cells is None:
        return None
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
    raise RuntimeError("failed to generate a valid b99e7126 D14 input")
