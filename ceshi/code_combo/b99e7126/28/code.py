# Combo variant of b99e7126 (index 28)
# combo_spec: D2/3+D5/3+D13/1
# Name: 颜色交换 + 旋转叠加 + 源图案旋转

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The canvas remains a 29x29 divider grid holding a 7x7 array of 3x3 cell interiors. Input still
#    contains exactly two visible cell motifs: a majority 2-color pattern and a rarer minority 2-color
#    pattern shown in only 3-4 clue cells. In the original task, the minority motif's frequent color is
#    read directly as a 3x3 placement mask, and active sites are overwritten by the minority template.
#    Here the chain is deliberately longer:
#    1. Read the minority motif as a 3x3 binary source mask.
#    2. Apply a source-orientation transform from the target block location:
#       top half / left half -> no flip; top-right -> LR flip; bottom-left -> UD flip;
#       bottom-right -> both flips (= 180-degree turn).
#    3. Apply a second transform to the enlarged mask before overlay: the quadrant chooses an extra
#       quarter-turn count TL/TR/BL/BR = 90/180/270/0 degrees.
#    4. The resulting 3x3 cell-scale mask marks which cells are affected, but affected cells are not
#       overwritten. Instead, each affected 3x3 cell swaps its own two colors in place.
#    Example sketch with source mask
#       [[1,1,1],[1,0,0],[1,0,0]]
#    placed in the bottom-left branch:
#       source transform -> [[1,0,0],[1,0,0],[1,1,1]]   (UD flip)
#       overlay rotation -> [[0,0,1],[0,0,1],[1,1,1]]   (270-degree turn)
#    Those final 1-cells indicate where local color swaps happen. The clue minority cells are only a
#    subset of those sites in the input; in the output they also get swapped, so the result contains
#    swapped-minority and swapped-majority cells rather than copied minority cells.
# b) Complexity calibration:
#    The transform families are kept deliberately small and repeated: source orientation has 4 branches,
#    overlay rotation has 4 branches, both keyed only by visible coarse position. That is enough to make
#    the geometry visibly different from the original and from either single variant, while still being
#    learnable from a few examples. Generator masks are filtered so both the source-stage variants and the
#    final-stage variants are all distinct; otherwise one branch could collapse and hide a control knob.
# c) Implicit assumptions audit:
#    - Original main() assumed the raw minority motif was already the final placement mask. This breaks
#      twice here: candidate checking must derive the final mask through source transform + overlay
#      rotation, and the same derived mask must be reused for the output operation.
#    - Original fill logic assumed active sites should become the minority template. That is invalid under
#      D5/3; the output operation now preserves local geometry and only swaps the two colors of each
#      affected cell.
#    - _swap_two_colors() introduces a new assumption that every touched cell has exactly two colors.
#      Generator enforces that by constructing both majority and minority motifs as strict 2-color 3x3
#      patterns.
#    - The minority motif's more frequent color is still interpreted as the active color. To keep that
#      reliable after adding asymmetry constraints, generator uses 5-6 active pixels, not balanced counts.
#    - Unique placement now depends on the composed final mask, not the raw source mask. The clue subset
#      search therefore tests uniqueness against the same composition used by the solver.
# d) Placement legality:
#    The transformed source mask and the rotated overlay mask remain 3x3, so the original 7x7-within-29x29
#    bounds still work. Placements are restricted to anchors with row/col in {0,1,3,4}, avoiding the
#    central anchor where the "upper/lower" and "left/right" cues would become visually weaker. The final
#    active cells always lie inside the chosen 3x3 block, and color swap never touches divider lines.
# e) Visual observability:
#    A human can inspect the minority clue motif, infer that its orientation is not used raw, and compare
#    input/output to see two simultaneous effects: the affected large-scale shape is reoriented, and the
#    hit cells invert their internal colors instead of being replaced by one template. Because every sample
#    repeats the same local swap on multiple cells, the color rule is visible rather than hidden in one-off
#    recoloring noise.
# f) Visual cleanliness:
#    The scene still has only divider lines plus repeated cell motifs. No third object type is added; the
#    only changes are which cells are selected and whether those cells are locally swapped. That keeps the
#    output structured and avoids unreadable overlap.
# g) Change inferability:
#    The rule never relies on numeric color ids. Spatial branch selection is driven only by the visible
#    quadrant of the 3x3 block. Color changes are local two-color inversions repeated across several cells,
#    so the observer can infer "selected cells swap their two colors" without needing a hidden global
#    palette mapping.
# Post-edit solver assumptions guaranteed by generator:
#    - Input contains exactly two cell patterns, with the minority motif shown in 3-4 cells.
#    - The minority motif has 5 or 6 active pixels and exactly two colors.
#    - Source-stage quadrant transforms are all distinct for the sampled mask.
#    - Final overlay-stage quadrant transforms are all distinct for the sampled mask.
#    - The visible clue subset determines exactly one legal (top, left) anchor under the composed rule.

from functools import reduce
from itertools import combinations
from math import gcd
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


def _branch_flags(top: int, left: int, n_cells: int, mask_size: int) -> tuple[int, int]:
    """Return coarse row/col branch flags for a 3x3 block anchor inside the 7x7 cell grid."""
    split = (n_cells - mask_size) // 2
    return int(top > split), int(left > split)


def _transform_source_mask(mask: np.ndarray, top: int, left: int, n_cells: int) -> np.ndarray:
    """Apply the D2 stage: derive the usable source blueprint from block location."""
    row_flag, col_flag = _branch_flags(top, left, n_cells, mask.shape[0])
    out = mask.copy()
    if row_flag:
        out = np.flipud(out)
    if col_flag:
        out = np.fliplr(out)
    return out


def _overlay_rotation_k(top: int, left: int, n_cells: int, mask_size: int) -> int:
    """Apply the D13 stage: quadrant chooses an extra CCW quarter-turn count."""
    row_flag, col_flag = _branch_flags(top, left, n_cells, mask_size)
    quadrant = row_flag * 2 + col_flag
    return int((quadrant + 1) % 4)


def _final_overlay_mask(mask: np.ndarray, top: int, left: int, n_cells: int) -> np.ndarray:
    """Compose source reorientation and overlay rotation into the final cell-scale mask."""
    source_mask = _transform_source_mask(mask, top, left, n_cells)
    return np.rot90(source_mask, _overlay_rotation_k(top, left, n_cells, mask.shape[0]))


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
    """Swap local two-color cell interiors at every active site of the placed mask."""
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
    Read the rare 3x3 cell as a blueprint, reorient it twice, and color-swap the selected cells.

    Input structure:
    - The board is a 7x7 lattice of 3x3 cell interiors separated by 1-pixel divider lines.
    - One 2-color cell pattern is the majority motif; a smaller set of clue cells share one rarer
      2-color minority motif.
    - The minority motif is the source blueprint, but neither its raw orientation nor direct copying
      rule is valid anymore.

    Rule:
    - Extract the minority motif and mark its more frequent color as the active pixels of a 3x3 mask.
    - For each candidate 3x3 block anchor in the cell grid, first flip that source mask according to
      whether the block sits in the lower half and/or right half, then rotate the enlarged mask by a
      quadrant-dependent quarter-turn count.
    - Find the unique anchor whose final mask contains all visible minority clue cells.
    - Apply the final mask as an overlay: every active cell keeps its shape but swaps its two colors
      locally; inactive cells stay unchanged.

    Key insight:
    The odd cell is still a miniature map of the large-scale target, but it must be reoriented twice
    before use, and the large-scale action is local color inversion rather than template replacement.
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
    assert len(vals) == 2, "minority blueprint must use exactly two colors"
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
            final_mask = _final_overlay_mask(base_mask, top, left, n)
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~final_mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique composed-mask placement"
    top, left = candidates[0]

    return _apply_color_swap_overlay(g, _final_overlay_mask(base_mask, top, left, n), top, left, step)


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (5, 6)
GIVEN_CELL_SIZES = [3, 4]
PLACEMENT_CHOICES = tuple((r, c) for r in (0, 1, 3, 4) for c in (0, 1, 3, 4))
COLOR_POOL = 10
MAX_RETRIES = 500


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate asymmetric masks whose source and final quadrant variants are all distinct."""
    by_count: dict[int, list[np.ndarray]] = {count: [] for count in ACTIVE_PIXEL_COUNTS}
    reps = ((0, 0), (0, 4), (4, 0), (4, 4))
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(MASK_SIZE * MASK_SIZE)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        active_count = int(mask.sum())
        if active_count not in by_count:
            continue
        if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        source_variants = {
            _transform_source_mask(mask, top, left, CELL_GRID_N).tobytes()
            for top, left in reps
        }
        if len(source_variants) != 4:
            continue
        final_variants = {
            _final_overlay_mask(mask, top, left, CELL_GRID_N).tobytes()
            for top, left in reps
        }
        if len(final_variants) != 4:
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
    if len(np.unique(arr)) != 2:
        rr = random.randrange(MASK_SIZE)
        cc = random.randrange(MASK_SIZE)
        arr[rr, cc] = color_b if arr[rr, cc] == color_a else color_a
    return arr


def _random_minority_mask() -> np.ndarray:
    """Pick a random minority blueprint mask with visible source/final asymmetry."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _placement_candidates(mask: np.ndarray, subset_abs: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find all anchors whose composed final mask is consistent with the shown clue cells."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset_abs:
        existing[r, c] = True

    candidates: list[tuple[int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            final_mask = _final_overlay_mask(mask, top, left, CELL_GRID_N)
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[top : top + MASK_SIZE, left : left + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[top : top + MASK_SIZE, left : left + MASK_SIZE] & ~final_mask):
                continue
            candidates.append((top, left))
    return candidates


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]] | None:
    """Choose 3-4 clue cells that pin down the composed placement uniquely."""
    final_mask = _final_overlay_mask(mask, top, left, CELL_GRID_N)
    active_abs = [(top + r, left + c) for r, c in np.argwhere(final_mask)]
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


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, minority_cells: list[tuple[int, int]]) -> np.ndarray:
    """Construct a 29x29 grid with majority everywhere and minority at the chosen clue cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build input/output under the composed transform and verify."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_two_color_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top, left = random.choice(PLACEMENT_CHOICES)
    given_cells = _choose_given_subset(mask, top, left)
    if given_cells is None:
        return None

    inp = _build_grid(separator, majority, minority, given_cells)
    out = _apply_color_swap_overlay(inp, _final_overlay_mask(mask, top, left, CELL_GRID_N), top, left, CELL_STEP)

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
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
