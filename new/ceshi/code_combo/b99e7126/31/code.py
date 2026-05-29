# Combo variant of b99e7126 (index 31)
# combo_spec: D11/0+D14/0
# Name: 颜色重映射 + 环绕边界

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the D11/0 + D14/0 combo
# -------------------------------------------
# concretization_and_trace:
# - The canvas is still a 29x29 divider grid containing a 7x7 lattice of 3x3 cell
#   interiors. Most cells repeat one majority template. A few visible cells show a
#   minority template whose dominant color defines a 3x3 binary mask.
# - Original last-stage behavior:
#   choose an in-bounds top-left offset in 0..4, place the 3x3 mask as a plain
#   rectangle of whole cells, and stamp active positions with the minority template.
# - Fused behavior:
#   choose a toroidal offset in 0..6 that wraps on at least one axis; then enlarge
#   the same 3x3 mask onto whole cells with modulo-7 addressing. Active positions do
#   not receive the literal minority template. Instead, the minority-only color is
#   remapped to the majority-only color while the shared color stays fixed.
# - Example trace:
#   if minority cell is [[m,m,m],[m,s,m],[m,s,m]], majority uses {a,s}, and the
#   toroidal offset is (6,5), then local mask coordinates land at rows {6,0,1} and
#   cols {5,6,0}. The output cells at those wrapped coordinates use
#   [[a,a,a],[a,s,a],[a,s,a]], visibly crossing the border and changing palette.
#
# complexity_calibration:
# - Keep the original 4-color budget: divider, shared, majority-only, minority-only.
#   This gives one clean recolor relation per sample: minority-only -> majority-only.
# - Force wrapping on row, col, or both so D14 is always visible; otherwise the
#   sample could collapse back to the original bounded placement regime.
# - Reject outputs where the recolored active template equals the ordinary majority
#   template or differs by too few pixels, because then the D11 effect becomes weak.
# - Preserve the original 6-7 active pixels and edge-spanning mask geometry so the
#   wrapped placement leaves enough border evidence for humans to notice.
#
# hidden_assumption_audit:
# - `_infer_cell_grid_params` only assumes full divider rows/cols and regular spacing.
#   The combo does not break that.
# - Cell counting still assumes the input shows only two recurring cell interiors:
#   the majority pattern and a minority hint pattern. Generator construction keeps it
#   that way; the recolored active template appears only in outputs.
# - Original candidate search assumed a contiguous in-bounds 3x3 block. The combo
#   breaks that: `main()` must search all 7x7 toroidal offsets and compare against
#   wrapped active coordinates rather than rectangular slices.
# - Original writeback assumed active cells copy the minority template literally.
#   The combo breaks that too: `main()` must derive the shared color and the two
#   exclusive colors, then write a recolored template at wrapped active positions.
# - New solver assumption: majority and minority templates each use two colors and
#   share exactly one color, with the minority-only color being the dominant one.
#   `_gen_attempt()` constructs exactly that.
#
# placement_legality:
# - With lattice size 7 and mask size 3, wrapped positions `((top+r)%7,(left+c)%7)`
#   are all distinct, so the toroidal stencil never self-overlaps.
# - Input minority cells are chosen from active wrapped positions only, and the
#   chosen subset must identify exactly one toroidal offset under the fused solver.
# - Output-first generation writes the full wrapped-and-recolored target first, then
#   downgrades some active cells back to majority for the input, preserving legality.
#
# visual_observability_and_cleanliness:
# - The only object classes remain divider lines plus two recurring cell templates in
#   the input, so the board stays tidy instead of becoming layered noise.
# - The combo is human-visible in two independent ways on the same region:
#   active output cells switch to the majority palette while also reappearing across
#   the opposite border. Because both cues act on the same stencil, they reinforce
#   rather than compete with each other.
# - No rule depends on numeric color ids or hidden ordering; the mapping is inferred
#   from the shared color between majority/minority templates and the border-crossing
#   shape of the wrapped stencil.
#
# new_solver_assumptions_and_generator_guarantees:
# - `main()` assumes a unique toroidal placement candidate exists for the visible
#   minority subset. `_choose_given_subset()` filters for that exactly.
# - `main()` assumes recolored active cells are visually distinct from majority cells.
#   `_gen_attempt()` enforces a minimum pixel difference.
# - The generator also forces at least one wrapped axis, so the fused task cannot
#   silently degrade into the non-wrapping recolor-only case.


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


def _derive_recolored_pattern(minority_arr: np.ndarray, majority_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask, recolored_pattern) for the fused transform."""
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
    Complete a wrapped cell-grid fractal by enlarging the odd cell with a recolored stamp.

    Input structure:
    - The board is partitioned into equal 3x3 cell interiors by single-color divider rows and columns.
    - Most cells repeat one majority interior template; a smaller visible subset shows one minority template.
    - Those minority hint cells can already be split across opposite borders because the large stencil wraps.

    Rule:
    - Read the minority template as a 3x3 mask using its dominant color as the active pixels.
    - Recover the unique toroidal placement of that mask on the cell lattice, allowing row and column indices to wrap modulo the lattice size.
    - Build the completed stencil on that wrapped placement: inactive positions stay majority, while active positions use the minority geometry recolored into the majority palette by preserving the shared color and replacing the minority-only color with the majority-only color.

    Key insight:
    - The odd cell still encodes a self-similar blueprint, but the enlarged copy is both periodic and recolored: it re-enters from the opposite edge and its active color is translated through the palette relationship between the two cell types.
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
    assert len(candidates) == 1, "must have a unique wrapped mask placement aligning existing minority cells"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            rr = (top + i) % n
            cc = (left + j) % n
            r0 = rr * step + 1
            c0 = cc * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = recolored_arr if mask[i, j] else majority_arr
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
MIN_RECOLOR_DIFF = 3

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


def _wrapped_active_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    return [
        ((top + r) % CELL_GRID_N, (left + c) % CELL_GRID_N)
        for r, c in _mask_coords(mask)
    ]


def _placement_candidates(mask: np.ndarray, subset_cells) -> list[tuple[int, int]]:
    """Find all toroidal offsets where the given visible minority cells are consistent with mask."""
    existing = set(subset_cells)
    candidates = []
    for t in range(PLACEMENT_RANGE):
        for l in range(PLACEMENT_RANGE):
            placed = _placed_cells(mask, t, l, CELL_GRID_N)
            if existing <= placed:
                candidates.append((t, l))
    return candidates


def _choose_given_subset(mask: np.ndarray, top: int, left: int):
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


def _choose_wrapping_offset() -> tuple[int, int]:
    branch = random.choice(WRAP_BRANCHES)
    if branch == "row":
        return random.choice(WRAP_OFFSETS), random.choice(NON_WRAP_OFFSETS)
    if branch == "col":
        return random.choice(NON_WRAP_OFFSETS), random.choice(WRAP_OFFSETS)
    return random.choice(WRAP_OFFSETS), random.choice(WRAP_OFFSETS)


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

    top, left = _choose_wrapping_offset()
    active_cells = _wrapped_active_cells(mask, top, left)
    out = _build_grid(separator, majority, recolored, active_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
