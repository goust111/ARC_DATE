# Combo variant of b99e7126 (index 12)
# combo_spec: D3/0+D9/0
# Name: 输出紧凑裁剪 + 放大图案居中放置

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Base task: the odd 3x3 cell interior is read as a binary mask and then
#      enlarged to a 3x3 block of whole cells somewhere on the 7x7 board.
#    - D3 changes only the destination logic: the enlarged block is no longer
#      solved from the shown odd-cell locations. It is always centered on the
#      whole 7x7 lattice, i.e. rows/cols 2..4 in cell coordinates.
#    - D9 then removes all untouched context and returns only the enlarged
#      centered block with its border dividers.
#    - Small trace with cell coordinates:
#      1. Suppose the odd cell interior encodes a mask whose active positions are
#         {(0,0),(0,2),(1,0),(1,1),(1,2),(2,1)}.
#      2. Original logic would search the whole 7x7 board for a unique offset
#         explaining the shown odd cells.
#      3. Combo logic ignores that offset search, fixes the block at top=2,
#         left=2, completes the 3x3 centered block there, and finally crops to
#         rows 8..20 and cols 8..20 in pixel space.
#      4. The output is therefore a 13x13 mini-board containing only the
#         completed centered block.
#    - Difference:
#      D3 changes "where to place"; D9 changes "how much of the board to keep".
#      Because D9 would visually hide absolute placement, the generator makes the
#      shown odd-cell subset intentionally ambiguous under the old placement rule
#      so the centering decision remains a real part of the task.
#
# b) Complexity calibration:
#    - Keep the original 7x7 divider grid, 3x3 cell interiors, 4-color palette,
#      LR-symmetric majority/minority textures, and 6-7 active mask pixels.
#    - Keep the output crop cell-aligned and framed by divider lines, yielding a
#      readable 13x13 mini-grid instead of a raw 11x11 interior crop.
#    - Keep 3-4 shown odd cells in the input so the seed still feels like a
#      partial completion task rather than a single-token copy task.
#    - Reject samples unless those 3-4 shown cells admit multiple old-style
#      placements. That preserves visible structure while ensuring the new global
#      centering rule is doing genuine work.
#
# c) Implicit-assumption audit:
#    - `_infer_cell_grid_params` still assumes full divider rows/cols and a
#      boundary frame. The input grid is unchanged, so that remains valid.
#    - Cell-pattern counting still assumes one dominant majority pattern and one
#      rarer minority pattern. The generator continues to build exactly that.
#    - The original solver's key assumption was stronger: the shown minority
#      cells uniquely determine one translation of the 3x3 mask. That is now
#      false by design. The combo solver must not search for a unique placement.
#    - The old `out = g.copy()` approach would also be wrong now because the
#      final answer is no longer the full board; it must be a rendered centered
#      completion followed by a tight crop.
#    - New assumption introduced here: the cell board is large enough that the
#      centered mask fits. The generator fixes a 7x7 board and a 3x3 mask, so
#      top=left=2 is always legal.
#
# d) Placement legality:
#    - The centered 3x3 block occupies cell rows/cols 2..4, safely inside the
#      7x7 board.
#    - The output crop is the exact cell-aligned bbox of that centered block:
#      pixel rows/cols `top*step .. (top+3)*step`, inclusive.
#    - The input shown subset is sampled only from active cells of the centered
#      true mask, so every shown odd cell still belongs to the final structure.
#
# e) Visual observability:
#    - Across examples, the input always contains a few odd cells scattered only
#      inside the future 3x3 center zone, but those cells no longer uniquely say
#      where the enlargement goes. The output is always a compact completed block.
#    - Humans can still see the fractal reuse because the odd cell interior
#      becomes the large 3x3 arrangement of odd vs majority tiles.
#    - The ambiguity of the shown subset is repeated across samples, so the task
#      reads as "complete the centered motif" rather than "infer one offset from
#      sparse markers".
#
# f) Visual cleanliness:
#    - Only one coherent object survives in the output: the completed local
#      block, with its outer divider frame intact.
#    - No unrelated board context remains, so there is little clutter despite
#      the input still using the full 29x29 board.
#
# g) Change inferability:
#    - No rule depends on numeric color ids or scan order. Colors keep the same
#      structural roles as in the seed task.
#    - The anti-degeneration control variable is measurable: the old placement
#      search must find more than one valid offset for the shown subset. The
#      generator enforces that, so D3 is active even though D9 later removes the
#      global board frame from the answer.

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

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


def _cell_pattern_data(g: np.ndarray):
    """Extract per-cell interior patterns plus their multiplicities."""
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    counts: dict[tuple[int, ...], int] = {}
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            counts[pat] = counts.get(pat, 0) + 1
            row.append(pat)
        cell_pat.append(row)
    return step, n, inner, patterns, cell_pat, counts


def _recover_task_objects(g: np.ndarray):
    """Recover the majority/minority cell textures and the minority-derived mask."""
    step, n, inner, patterns, cell_pat, counts = _cell_pattern_data(g)
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns (minority + majority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat

    return step, n, inner, majority_arr, minority_arr, mask, existing


def _centered_top_left(n: int, mh: int, mw: int) -> tuple[int, int]:
    """Center an mh×mw block on an n×n cell lattice."""
    assert mh <= n and mw <= n, "centered mask must fit in the cell board"
    return (n - mh) // 2, (n - mw) // 2


def _old_style_candidates(mask: np.ndarray, existing: np.ndarray) -> list[tuple[int, int]]:
    """Placements accepted by the original translation-based rule."""
    n = int(existing.shape[0])
    mh, mw = mask.shape
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
    return candidates


def _render_full_board(
    line_color: int,
    step: int,
    n: int,
    majority_arr: np.ndarray,
    minority_arr: np.ndarray,
    mask: np.ndarray,
    top: int,
    left: int,
) -> np.ndarray:
    """Render the completed board after placing the enlarged mask at a fixed offset."""
    inner = step - 1
    out = np.full((n * step + 1, n * step + 1), int(line_color), dtype=int)
    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = majority_arr

    mh, mw = mask.shape
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


def _crop_cell_block(g: np.ndarray, top: int, left: int, cells_h: int, cells_w: int, step: int) -> np.ndarray:
    """Crop a cell-aligned block while preserving its outer divider frame."""
    r0 = top * step
    c0 = left * step
    r1 = (top + cells_h) * step
    c1 = (left + cells_w) * step
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the enlarged minority blueprint at the board center, then keep only that local block.

    Input structure:
    - A 29x29 separator grid partitions the board into a 7x7 lattice of 3x3 cell interiors.
    - One interior texture is the majority tile; a rarer odd texture appears in only a few shown cells.

    Rule:
    - Read the odd tile's dominant-color pixels as a 3x3 active/inactive mask.
    - Ignore any translation implied by the shown odd-cell positions and instead place the enlarged 3x3 mask at the center of the 7x7 cell board.
    - Fill active entries of that centered block with the odd tile texture and inactive entries with the majority texture.
    - Return only the centered completed block, tightly cropped on cell boundaries with divider lines intact.

    Key insight:
    The odd tile still serves as both texture and blueprint, but the blueprint no longer tells you where to place the enlargement; location is fixed globally at the board center, and the output keeps only that centered construction.
    """
    g = np.asarray(grid, dtype=int)
    step, n, _inner, majority_arr, minority_arr, mask, _existing = _recover_task_objects(g)
    mh, mw = mask.shape
    top, left = _centered_top_left(n, mh, mw)
    centered = _render_full_board(int(g[0, 0]), step, n, majority_arr, minority_arr, mask, top, left)
    return _crop_cell_block(centered, top, left, mh, mw, step)


# --------------- generator (o5x) ---------------

# --- CFG: tunable generation parameters ---
CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
CENTER_TOP = (CELL_GRID_N - MASK_SIZE) // 2
CENTER_LEFT = (CELL_GRID_N - MASK_SIZE) // 2
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
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
    """Enumerate valid LR-symmetric 3x3 masks, keyed by active pixel count."""
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


def _random_lr_pattern(color_a, color_b):
    """Generate a non-uniform LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask():
    """Pick a valid minority mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _cell_origin(i, j):
    """Pixel origin of cell (i, j) interior inside the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator, majority, minority, active_cells):
    """Construct the full 29x29 board from cell patterns."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _existing_from_subset(subset, top, left):
    """Lift local mask coordinates into a 7x7 board-sized boolean map."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True
    return existing


def _choose_ambiguous_subset(mask, top, left):
    """Choose a shown subset that keeps the old placement rule ambiguous."""
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = []
        for subset in combinations(active, size):
            existing = _existing_from_subset(subset, top, left)
            candidates = _old_style_candidates(mask, existing)
            if len(candidates) > 1 and (top, left) in candidates:
                valid.append((subset, len(candidates)))
        if valid:
            subset, candidate_count = random.choice(valid)
            return subset, candidate_count
    return None


def _gen_attempt():
    """Build an input whose answer is the centered crop, not an old-style unique placement."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = CENTER_TOP
    left = CENTER_LEFT
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    full_out = _build_grid(separator, majority, minority, active_cells)
    cropped_out = _crop_cell_block(full_out, top, left, MASK_SIZE, MASK_SIZE, CELL_STEP)

    choice = _choose_ambiguous_subset(mask, top, left)
    if choice is None:
        return None
    subset, _candidate_count = choice
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, cropped_out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 D3/0+D9/0 input")
