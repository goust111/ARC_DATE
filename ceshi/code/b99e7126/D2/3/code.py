# Variant of b99e7126
# Dimension: D2 — 源对象：异常单元格的内部图案结构 (src)
# Change: 唯一异常的3x3单元格的内部颜色像素图案，包含两种颜色（前景色和背景色）... → 源图案需要旋转/翻转后才能使用

# Design analysis for the D2 variant
# a) Concrete behavior and trace:
#    The grid is still a 7x7 array of 3x3 cell interiors separated by 1-pixel dividers.
#    Most cells show one majority 3x3 pattern; a few shown cells use one minority 3x3 pattern.
#    In the original task, the minority cell's foreground/background layout is used directly as
#    the 3x3 blueprint for where minority cells belong at grid scale. Here that same blueprint is
#    first transformed according to the placement of the 3x3 block: lower-half placement flips the
#    blueprint vertically, right-half placement flips it horizontally, and both together yield a
#    180-degree rotation. Example: if the source minority pattern has foreground mask
#    [[1,1,1],[1,0,0],[1,0,0]], then a block placed in the bottom-right region uses
#    [[0,0,1],[0,0,1],[1,1,1]] as the grid-scale blueprint. The minority cell pattern stamped into
#    active cells is unchanged; only the blueprint used for placement is transformed.
# b) Complexity calibration:
#    The transform family is limited to four dihedral cases generated compositionally from two
#    binary position cues (upper/lower and left/right), so the rule is visible without becoming
#    arbitrary. The minority masks are filtered to be asymmetric under both flips so every branch
#    is visually different. Given 3-4 minority cells remain enough for unique placement.
# c) Implicit assumptions audit:
#    - Original code assumes the raw minority pattern can be used as the mask. That fails here, so
#      candidate checking must transform the mask separately for each candidate (top,left).
#    - Original fill loop blindly reuses the raw mask. That would paint the wrong cells in this
#      variant, so the same location-driven transform must also be applied during output filling.
#    - The "active color = most frequent minority color" assumption still holds, so generation
#      keeps minority foreground counts at 5 or 6 and background counts at 4 or 3.
#    - Majority/minority pattern counting still works because only two cell patterns appear.
# d) Placement legality:
#    The transformed blueprint always remains 3x3, so the original 7x7-within-29x29 geometry and
#    bounds logic remain valid. Generator placement uses anchors in {0,1,3,4} for both axes to
#    avoid the ambiguous center anchor while still covering all four transform branches.
# e) Visual observability:
#    The rule depends only on visible spatial position, never on color ids. The minority blueprint
#    masks are explicitly asymmetric, so horizontal/vertical/180-degree changes are obvious across
#    training examples. Because horizontal and vertical decisions compose independently, seeing a
#    few examples is enough to extrapolate the missing branch.
# f) Visual cleanliness:
#    The grid still contains only divider lines plus two cell motifs. No extra annotations or
#    overlays are introduced, so the scene remains as tidy as the original task.
# g) Color inference:
#    Colors keep the original four-role setup. The transform changes geometry only; it does not
#    introduce any hidden color remapping, so humans can focus on structure rather than palette.
# New solver assumptions checked against generator:
#    - A unique minority pattern exists and appears 3-4 times in input.
#    - The minority pattern uses exactly two colors and its more frequent color is the active one.
#    - The transformed blueprint has a unique legal placement consistent with the shown subset.
#    - The generator verifies every sample by running main(inp) and comparing to the constructed
#      output, so these assumptions are enforced rather than merely hoped for.

from functools import reduce
from math import gcd

from itertools import combinations
import random

import numpy as np

from arc_utils import paste


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


def _transform_for_location(mask: np.ndarray, top: int, left: int, n_cells: int) -> np.ndarray:
    """Apply the location-driven blueprint transform for a candidate 3x3 block anchor."""
    out = mask.copy()
    split = (n_cells - mask.shape[0]) // 2
    if top > split:
        out = np.flipud(out)
    if left > split:
        out = np.fliplr(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a cell-grid fractal by transforming the source blueprint before scaling it up.

    Input structure:
    - The board is partitioned into equal 3x3 cell interiors by 1-pixel divider lines.
    - One cell pattern is the majority motif; a smaller set of cells show one minority motif.
    - The minority motif itself is a 2-color 3x3 blueprint, but its orientation is not yet the
      one used at the larger cell-grid scale.

    Rule:
    - Read the minority motif's more frequent color as the active pixels of a 3x3 blueprint.
    - Infer where that 3x3 block sits in the 7x7 cell grid, then transform the blueprint from the
      block position: lower placements flip vertically, right placements flip horizontally.
    - Use the transformed blueprint to decide which cell positions should be minority cells, and
      fill the missing active positions while leaving inactive positions as majority cells.

    Key insight:
    The abnormal cell still encodes the global arrangement, but only after a location-dependent
    flip/rotation; position tells you how to orient the blueprint before projecting it upward.
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
    base_mask = minority_arr == active_color
    mh, mw = base_mask.shape

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
            transformed = _transform_for_location(base_mask, top, left, n)
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~transformed):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique transformed mask placement"
    top, left = candidates[0]
    transformed = _transform_for_location(base_mask, top, left, n)

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if transformed[i, j] else patterns[majority_pat]
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
ACTIVE_PIXEL_COUNTS = (5, 6)
GIVEN_CELL_SIZES = [3, 4]
PLACEMENT_CHOICES = (
    (0, 3), (0, 4), (1, 3), (1, 4),
    (3, 0), (3, 1), (4, 0), (4, 1),
    (3, 3), (3, 4), (4, 3), (4, 4),
)
COLOR_POOL = 10
MAX_RETRIES = 400


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate asymmetric 3x3 masks whose four region-driven variants are all distinct."""
    by_count: dict[int, list[np.ndarray]] = {count: [] for count in ACTIVE_PIXEL_COUNTS}
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(MASK_SIZE * MASK_SIZE)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        active_count = int(mask.sum())
        if active_count not in by_count:
            continue
        if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        variants = {
            _transform_for_location(mask, 0, 0, CELL_GRID_N).tobytes(),
            _transform_for_location(mask, 0, 4, CELL_GRID_N).tobytes(),
            _transform_for_location(mask, 4, 0, CELL_GRID_N).tobytes(),
            _transform_for_location(mask, 4, 4, CELL_GRID_N).tobytes(),
        }
        if len(variants) != 4:
            continue
        by_count[active_count].append(mask)
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_two_color_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform 3x3 pattern using exactly two colors."""
    arr = np.array([[random.choice((color_a, color_b)) for _ in range(MASK_SIZE)] for _ in range(MASK_SIZE)], dtype=int)
    if np.all(arr == arr[0, 0]):
        rr = random.randrange(MASK_SIZE)
        cc = random.randrange(MASK_SIZE)
        arr[rr, cc] = color_b if arr[rr, cc] == color_a else color_a
    return arr


def _random_minority_mask() -> np.ndarray:
    """Pick a random asymmetric minority blueprint mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _placement_candidates(mask: np.ndarray, subset_abs: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find all offsets whose transformed mask is consistent with the shown minority-cell subset."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset_abs:
        existing[r, c] = True

    candidates: list[tuple[int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            transformed = _transform_for_location(mask, top, left, CELL_GRID_N)
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[top : top + MASK_SIZE, left : left + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[top : top + MASK_SIZE, left : left + MASK_SIZE] & ~transformed):
                continue
            candidates.append((top, left))
    return candidates


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]] | None:
    """Choose a shown subset of active cells that pins down one transformed placement uniquely."""
    transformed = _transform_for_location(mask, top, left, CELL_GRID_N)
    active_abs = [(top + r, left + c) for r, c in np.argwhere(transformed)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            list(subset)
            for subset in combinations(active_abs, size)
            if _placement_candidates(mask, subset) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, active_cells: list[tuple[int, int]]) -> np.ndarray:
    """Construct a 29x29 grid: fill all cells with majority, then overwrite active_cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, choose input subset, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_two_color_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top, left = random.choice(PLACEMENT_CHOICES)
    transformed = _transform_for_location(mask, top, left, CELL_GRID_N)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(transformed)]
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


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 D2 variant input")
