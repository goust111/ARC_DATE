# Combo variant of b99e7126 (index 36)
# combo_spec: D9/2+D15/1
# Name: 仅输出放大图案 + 缩小嵌入

"""
Design analysis for the D9/2 + D15/1 combo variant.

spec:
  inherited_input:
    grid_size: 29x29
    cell_grid: 7x7 logical cells
    cell_interior: 3x3
    dividers: separator rows/cols at 0,4,8,...,28
    visible_patterns:
      majority_tile: one repeated 3x3 pattern
      minority_tile: one rarer 3x3 pattern whose dominant color defines a 3x3 mask
    minority_subset:
      only 3-4 active positions of a hidden 3x3 placement are shown in the input
  combo_output:
    shape: 3x3
    meaning:
      first reconstruct the completed 7x7 logical minority/majority layout exactly as in the source task
      then shrink that completed 7x7 logical board into a 3x3 occupancy summary
      finally render that 3x3 summary with the minority tile's two colors
    removed_from_output:
      no 29x29 frame
      no divider lines
      no surrounding cells outside the modified anomaly-cell interior
  concrete_behavior_trace:
    example:
      if the hidden 3x3 mask is placed at logical offset (top=1,left=3), the original task would complete
      several minority cells inside that 3x3 logical block and write them back into the 29x29 board
      this combo instead completes the same hidden 7x7 logical minority map, partitions rows/cols as
      [0:2], [2:5], [5:7], OR-pools each macro-block, and writes the resulting 3x3 coarse map as the
      new anomaly-tile interior; then D9/2 crops away everything except that 3x3 tile
    difference:
      original: local 3x3 anomaly blueprint grows outward into a larger board patch
      combo: the recovered larger board pattern is folded back inward into a 3x3 miniature
  complexity_calibration:
    keep the original 6-7 active-pixel minority masks and 3-4 visible minority cells
    keep the unique-placement subset selection so the full logical board is still inferable
    use contiguous 7-to-3 pooling bands so the "shrink the whole board" step is structural and visible
    reject generated cases where the shrunken 3x3 summary exactly equals the original minority tile,
    because that would make the combo look like a trivial tile copy
  implicit_assumption_review:
    divider inference:
      unchanged; input geometry still uses full separator rows/cols
    pattern counting:
      still relies on one rare tile and one common tile
    active-color detection:
      still valid because generation keeps the minority-exclusive color as the 6-or-7-times majority
      inside the minority tile
    placement search:
      still required; without locating the hidden 3x3 logical block we cannot reconstruct the completed
      7x7 minority occupancy that is later shrunk
    writeback assumption:
      broken on purpose; the original solver edited a copy of the 29x29 board, but this combo outputs
      only the modified anomaly-cell interior, so the final stage must build a fresh 3x3 array
    new pooling assumption:
      the logical board is shrunk by contiguous macro-regions instead of arbitrary pixel sampling;
      generator and solver share the same deterministic partition rule
  placement_legality:
    the hidden 3x3 block still fits entirely inside the 7x7 logical board at offsets 0..4
    visible minority cells are always chosen from active mask positions only
    the completed logical board never overlaps out of bounds because it is exactly one placed 3x3 mask
    the 7-to-3 pooling bands cover the logical board without overlap or gaps
  visual_observability:
    across samples, inputs still show a large separator-lined board with a few anomalous cells
    outputs are now tiny 3x3 mosaics, so the D9/2 crop is immediately visible
    the output pattern varies with both the recovered placement and the recovered mask, so the D15/1
    inversion is not reducible to copying the minority tile
    the shrinking uses only spatial occupancy of minority cells, not hidden color ordering
  cleanliness:
    the output contains one clean 3x3 object with no divider clutter or overlapping layers
    only the minority tile's own two colors are used in the output, keeping the miniature readable
  inferability_of_change:
    a human can trace "few rare cells -> infer hidden 3x3 logical placement -> imagine full 7x7 pattern ->
    see that the output is a compressed map of that pattern"
    this avoids arbitrary recoloring rules and keeps the combo's two changes jointly legible in 2-4 examples
"""

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


def _extract_patterns(
    g: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Return:
      majority_tile, minority_tile, minority_mask, existing_minority_cells,
      active_color, inactive_color, logical_grid_size
    """
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

    vals, cnts = np.unique(minority_arr, return_counts=True)
    assert len(vals) == 2, "minority tile must use exactly two colors"
    active_color = int(vals[cnts.argmax()])
    inactive_color = int(vals[cnts.argmin()])
    mask = minority_arr == active_color

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat
    return majority_arr, minority_arr, mask, existing, active_color, inactive_color, n


def _locate_mask(existing: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
    """Find the unique logical 3x3 placement consistent with the visible minority subset."""
    n = existing.shape[0]
    mh, mw = mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(n - mh + 1):
        for left in range(n - mw + 1):
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "must have a unique mask placement aligning existing minority cells"
    return candidates[0]


def _complete_logical_board(n: int, mask: np.ndarray, top: int, left: int) -> np.ndarray:
    """Return the full logical minority occupancy after completing the hidden 3x3 mask."""
    board = np.zeros((n, n), dtype=bool)
    mh, mw = mask.shape
    board[top : top + mh, left : left + mw] = mask
    return board


def _partition_edges(length: int, parts: int) -> list[int]:
    """Split a length into contiguous near-even bands using symmetric rounded edges."""
    edges = [int(np.floor(i * length / parts + 0.5)) for i in range(parts + 1)]
    assert edges[0] == 0 and edges[-1] == length
    assert all(edges[i] < edges[i + 1] for i in range(parts)), "partition bands must be non-empty"
    return edges


def _shrink_logical_board(board: np.ndarray, out_size: int) -> np.ndarray:
    """
    Compress the completed logical board to out_size x out_size using OR pooling over contiguous bands.
    """
    n = board.shape[0]
    row_edges = _partition_edges(n, out_size)
    col_edges = _partition_edges(n, out_size)
    out = np.zeros((out_size, out_size), dtype=bool)
    for i in range(out_size):
        for j in range(out_size):
            block = board[row_edges[i] : row_edges[i + 1], col_edges[j] : col_edges[j + 1]]
            out[i, j] = bool(np.any(block))
    return out


def _render_micro_grid(mask: np.ndarray, active_color: int, inactive_color: int) -> np.ndarray:
    """Render a boolean 3x3 summary with the minority tile's two colors."""
    return np.where(mask, active_color, inactive_color).astype(int)


def _solve_components(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Shared inference chain used by both solver and generator verification."""
    g = np.asarray(grid, dtype=int)
    _majority, _minority, mask, existing, active_color, inactive_color, n = _extract_patterns(g)
    top, left = _locate_mask(existing, mask)
    full_board = _complete_logical_board(n, mask, top, left)
    micro_mask = _shrink_logical_board(full_board, mask.shape[0])
    return mask, micro_mask, full_board, active_color, inactive_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden global minority layout, compress it back into anomaly-cell scale, and return only that tile.

    Input structure:
    - The input is a 29x29 board of 3x3 cell interiors separated by 1-pixel divider lines.
    - One rare 3x3 tile appears in only a few cells; the common 3x3 tile fills the rest.
    - The rare tile still encodes a hidden 3x3 logical mask exactly as in the source task.

    Rule:
    - Identify the majority and minority tiles, then read the minority tile's dominant color as the active mask color.
    - Use the visible minority-cell subset to locate the unique 3x3 logical placement and reconstruct the completed 7x7
      minority/majority cell map.
    - Shrink that completed 7x7 logical map into a 3x3 occupancy summary by OR-pooling contiguous row/column bands, and
      render the summary with the minority tile's active/inactive colors.
    - Return only this 3x3 embedded miniature, not the full board.

    Key insight:
    - The odd tile is no longer expanded into the board for the final answer; instead it is used as a seed to recover the
      full board pattern, and that recovered global structure is folded back into a tiny self-contained anomaly-cell icon.
    """
    _mask, micro_mask, _full_board, active_color, inactive_color = _solve_components(grid)
    return _render_micro_grid(micro_mask, active_color, inactive_color)


# --------------- generator (o5x) ---------------

# --- CFG: tunable generation parameters ---
CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 400

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


def _placement_candidates(mask, subset, top, left):
    """Find all (t, l) offsets where the given subset of active cells is consistent with mask."""
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


def _expected_output(mask: np.ndarray, top: int, left: int, active_color: int, inactive_color: int) -> np.ndarray:
    """Return the cropped 3x3 miniature produced by the combo rule."""
    full_board = _complete_logical_board(CELL_GRID_N, mask, top, left)
    micro_mask = _shrink_logical_board(full_board, MASK_SIZE)
    return _render_micro_grid(micro_mask, active_color, inactive_color)


def _gen_attempt():
    """Single generation attempt: build the hidden logical board, expose a subset, verify the combo solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None

    out = _expected_output(mask, top, left, min_exclusive, shared)
    if np.array_equal(out, minority):
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
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
