# Combo variant of b99e7126 (index 5)
# combo_spec: D2/3+D13/1
# Name: 源图案需旋转 + 放大图案旋转后叠加

# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    The scene keeps the original 7x7 lattice of 3x3 cell interiors separated by 1-pixel dividers.
#    Most cells share one majority motif; a few visible cells share one minority motif. In the
#    base task, the minority motif's own 3x3 foreground mask is used directly as the larger-scale
#    3x3 blueprint. Here the process has two explicit stages. First, the source blueprint is
#    rotated before use: anchors in the upper half keep it as-is, anchors in the lower half rotate
#    it by 180 degrees. Second, after that 3x3 blueprint is enlarged to cell scale, anchors in the
#    right half rotate the whole enlarged 3x3 block by 90 degrees clockwise before it is overlaid.
#    Example: if the source mask is
#      [[1,1,1],
#       [1,0,0],
#       [1,1,0]]
#    and the inferred anchor is bottom-right, the source step gives
#      [[0,1,1],
#       [0,0,1],
#       [1,1,1]]
#    and the enlarged-block rotation then gives
#      [[1,0,0],
#       [1,0,1],
#       [1,1,1]]
#    as the actual cell-grid placement mask. The original task would have used the raw mask
#    unchanged, so every non-top-left quadrant now produces a visibly different completion.
# b) Complexity calibration:
#    Each stage uses one binary spatial cue: vertical half controls the source rotation, horizontal
#    half controls the enlarged-block rotation. Their composition yields the four net quarter-turn
#    outcomes {0,90,180,270}, which is rich enough to be visible but still simple enough to infer
#    from a few examples. The generator filters masks so all four rotations are distinct; this
#    avoids accidental symmetry that would hide one of the control variables.
# c) Implicit assumptions audit:
#    - The original solver assumed the raw minority mask could be checked against every candidate.
#      That is no longer valid: candidate validation must apply both location-dependent rotations.
#    - The original fill loop assumed the same raw mask determined which cells become minority in
#      the output. That also fails here; output painting must use the twice-transformed mask.
#    - The active color is still the more frequent minority color, so the minority motif must keep
#      5 or 6 active pixels; the generator enforces this.
#    - The enlarged block remains 3x3 and square, so rotating it after enlargement is equivalent
#      to rotating the 3x3 boolean placement mask. That equivalence is what lets the solver stay
#      compact while still modeling the two-stage rule explicitly.
# d) Placement legality:
#    The transformed block is still a 3x3 block of cells inside a 7x7 cell grid, so the original
#    bounds logic remains valid. The generator samples anchors from the four outer quadrants
#    ({0,1} or {3,4} per axis) to expose every branch clearly, and it keeps only subsets whose
#    shown minority cells determine one unique anchor among all 25 possible anchors.
# e) Visual observability:
#    The control signal is the visible anchor position of the 3x3 block, never an invisible color
#    ordering. Because the same asymmetric source motif can land in four different orientations, a
#    human can compare examples and notice that moving the block downward adds a half-turn while
#    moving it rightward adds an extra quarter-turn.
# f) Visual cleanliness:
#    Nothing new is drawn on top of the grid besides the completed minority cells. The board still
#    contains divider lines plus two cell motifs, so the output remains as legible as the base
#    task even though the placement logic is richer.
# g) Change inferability:
#    Colors keep the original four-role scheme: separator, majority-only, shared, minority-only.
#    No color remapping is introduced; all novelty is geometric. This keeps the fusion learnable,
#    because the solver must infer orientation from shape and location instead of from arbitrary
#    palette conventions.
# New solver assumptions checked against generator:
#    - Exactly two cell motifs appear in the input: one majority and one minority.
#    - The minority motif uses exactly two colors and its active color is the more frequent one.
#    - The chosen minority mask has four distinct rotational states, so the quadrant cues matter.
#    - The shown minority subset uniquely identifies one anchor after both rotation stages.
#    - The generator verifies every accepted sample by running main(inp) and matching the
#      constructed output, so these assumptions are enforced constructively.

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


def _rot_cw(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    """Rotate an array clockwise by 0/90/180/270 degrees."""
    return np.rot90(arr, -int(quarter_turns) % 4)


def _source_rotation(top: int, mask_size: int, n_cells: int) -> int:
    """Lower-half anchors rotate the source blueprint 180 degrees; upper-half anchors keep it."""
    split = (n_cells - mask_size) // 2
    return 2 if top > split else 0


def _overlay_rotation(left: int, mask_size: int, n_cells: int) -> int:
    """Right-half anchors rotate the enlarged 3x3 block by 90 degrees clockwise; left-half anchors keep it."""
    split = (n_cells - mask_size) // 2
    return 1 if left > split else 0


def _source_mask_for_location(base_mask: np.ndarray, top: int, left: int, n_cells: int) -> np.ndarray:
    """Stage 1: rotate the source blueprint before enlargement."""
    return _rot_cw(base_mask, _source_rotation(top, base_mask.shape[0], n_cells))


def _final_mask_for_location(base_mask: np.ndarray, top: int, left: int, n_cells: int) -> np.ndarray:
    """Stage 2: rotate the enlarged block, represented equivalently as a second rotation of the 3x3 mask."""
    src_mask = _source_mask_for_location(base_mask, top, left, n_cells)
    return _rot_cw(src_mask, _overlay_rotation(left, base_mask.shape[1], n_cells))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a cell-grid fractal after rotating the source motif and then rotating the enlarged block.

    Input structure:
    - The board is partitioned into equal 3x3 cell interiors by 1-pixel divider lines.
    - One interior motif is the dominant cell pattern; a smaller set of cells show one minority motif.
    - The minority motif is still the compact 3x3 blueprint, but its visible orientation is not yet
      the one used directly at the larger cell-grid level.

    Rule:
    - Read the minority motif's more frequent color as the active pixels of the source blueprint.
    - For each candidate 3x3 anchor in the 7x7 cell grid, first rotate that source blueprint by
      180 degrees when the anchor is in the lower half, then rotate the enlarged 3x3 block by
      90 degrees clockwise when the anchor is in the right half.
    - Pick the unique anchor whose twice-transformed blueprint contains all shown minority cells,
      then fill every active cell of that final blueprint with the minority motif while leaving the
      inactive cells as the majority motif.

    Key insight:
    The abnormal cell still encodes the global layout, but now the layout is recovered through a
    two-step orientation rule: vertical position changes the source motif first, and horizontal
    position rotates the enlarged placement afterward.
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
            final_mask = _final_mask_for_location(base_mask, top, left, n)
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~final_mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique twice-rotated mask placement"
    top, left = candidates[0]
    final_mask = _final_mask_for_location(base_mask, top, left, n)

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if final_mask[i, j] else patterns[majority_pat]
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
    (0, 0), (0, 1), (1, 0), (1, 1),
    (0, 3), (0, 4), (1, 3), (1, 4),
    (3, 0), (3, 1), (4, 0), (4, 1),
    (3, 3), (3, 4), (4, 3), (4, 4),
)
COLOR_POOL = 10
MAX_RETRIES = 500


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate asymmetric 3x3 masks whose four rotations are all distinct and edge-spanning."""
    by_count: dict[int, list[np.ndarray]] = {count: [] for count in ACTIVE_PIXEL_COUNTS}
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(MASK_SIZE * MASK_SIZE)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        active_count = int(mask.sum())
        if active_count not in by_count:
            continue
        if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        variants = {_rot_cw(mask, k).tobytes() for k in range(4)}
        if len(variants) != 4:
            continue
        by_count[active_count].append(mask)
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_two_color_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform 3x3 pattern using exactly two colors."""
    arr = np.array([[random.choice((color_a, color_b)) for _ in range(MASK_SIZE)] for _ in range(MASK_SIZE)], dtype=int)
    if len(set(int(v) for v in arr.ravel())) == 1:
        rr = random.randrange(MASK_SIZE)
        cc = random.randrange(MASK_SIZE)
        arr[rr, cc] = color_b if arr[rr, cc] == color_a else color_a
    return arr


def _random_minority_mask() -> np.ndarray:
    """Pick a random minority blueprint whose rotations stay visually distinct."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _placement_candidates(mask: np.ndarray, subset_abs: tuple[tuple[int, int], ...] | list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Find all anchors whose twice-rotated placement is consistent with the shown minority subset."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset_abs:
        existing[r, c] = True

    candidates: list[tuple[int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            final_mask = _final_mask_for_location(mask, top, left, CELL_GRID_N)
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[top : top + MASK_SIZE, left : left + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[top : top + MASK_SIZE, left : left + MASK_SIZE] & ~final_mask):
                continue
            candidates.append((top, left))
    return candidates


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]] | None:
    """Choose a shown subset of active cells that pins down one twice-rotated placement uniquely."""
    final_mask = _final_mask_for_location(mask, top, left, CELL_GRID_N)
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
    final_mask = _final_mask_for_location(mask, top, left, CELL_GRID_N)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(final_mask)]
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
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
