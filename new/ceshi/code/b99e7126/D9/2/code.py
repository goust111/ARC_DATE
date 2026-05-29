# Variant of b99e7126
# Dimension: D9 — 输出网格形状 (grid_shape)
# Change: 输出与输入网格尺寸相同... → 仅输出放大后的图案本身（无原始网格背景）

"""
Design analysis for the D9 variant.

spec:
  inherited_input:
    grid_size: 29x29
    cell_grid: 7x7 cells
    cell_interior: 3x3
    dividers: 1-pixel separator rows/cols at indices 0,4,8,...,28
    cell_types:
      majority: one repeated 3x3 interior pattern
      minority: one rarer 3x3 interior pattern
    minority_subset:
      the input still shows only 3-4 minority cells from a hidden 3x3 placement of the minority mask
  variant_output:
    shape: 9x9
    meaning:
      take the minority cell's 3x3 interior as a binary blueprint
      each blueprint pixel expands to one 3x3 tile with no separator lines
      active pixels become the minority tile, inactive pixels become the majority tile
    removed_from_output:
      no 29x29 frame
      no separator-color background
      no unrelated surrounding majority cells outside the 3x3 enlarged block
  concrete_behavior_trace:
    original:
      from the minority tile, derive a 3x3 active mask and place that mask into the 7x7 cell lattice
      then overwrite the matching 3x3 cell block inside the original 29x29 board
    variant:
      derive the same 3x3 active mask and the same minority/majority tiles
      but instead of writing back into the 29x29 board, assemble only the 3x3 block itself
      each logical cell contributes its bare 3x3 interior, so the result is 9x9
    difference:
      the inference target is the isolated fractal patch, not the completed source board
  complexity_calibration:
    keep the original 6-or-7-active-pixel masks and 3-or-4 given minority cells
    keep LR-symmetric but non-uniform patterns so the minority tile visibly differs from the majority tile
    keep the unique-placement subset selection in generation so input examples still visually support the
    "minority tile is a blueprint" reading, even though the final output is cropped
  implicit_assumption_review:
    divider inference:
      still valid because the input format is unchanged
    pattern counting:
      still assumes exactly one minority pattern and one majority pattern dominate the cells
    active-color detection:
      still relies on the minority-exclusive color being the majority inside the minority tile
      generation preserves this with 6-7 active pixels out of 9
    placement search:
      no longer needed for output coordinates, but retained as a consistency check that the shown minority
      subset really comes from one hidden 3x3 placement
    writeback logic:
      the original code copied onto g; that depended on keeping the original board shape
      in this variant that assumption is false, so the final stage must assemble a fresh 9x9 canvas
  placement_legality:
    the hidden 3x3 mask still fits inside the 7x7 cell board at offsets 0..4
    given minority cells are chosen from active mask positions only
    generator rejects any subset that does not imply a unique placement under the original constraints
  visual_observability:
    across examples, the input remains a large tiled board with a few anomalous cells
    the output is a compact 9x9 mosaic composed of repeated 3x3 tiles, which is visibly different from the
    original full-board completion behavior
    the shape change is directly observable without relying on color ids or hidden metadata
  cleanliness:
    output contains only one 3x3-by-3x3 fractal patch, so there is no overlap between separators and content
    this keeps the transformed object compact and easy to parse
"""

from functools import reduce
from math import gcd

import numpy as np
import random
from itertools import combinations
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


def _extract_patterns(g: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (majority_tile, minority_tile, minority_mask) from the tiled input board."""
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
    active_color = int(vals[cnts.argmax()])
    mask = minority_arr == active_color
    return majority_arr, minority_arr, mask


def _assert_unique_placement(g: np.ndarray, minority_arr: np.ndarray, mask: np.ndarray) -> None:
    """
    Keep the original placement-consistency check.

    The cropped output no longer uses the absolute offset, but the input is still generated from a
    uniquely placeable subset of minority cells, and this assertion protects that contract.
    """
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    minority_pat = tuple(minority_arr.ravel().tolist())

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat = tuple(g[r0 : r0 + inner, c0 : c0 + inner].ravel().tolist())
            existing[i, j] = pat == minority_pat

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
    assert len(candidates) == 1, "must have a unique mask placement aligning existing minority cells"


def _build_scaled_patch(majority_arr: np.ndarray, minority_arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Assemble the 9x9 cropped fractal block with no divider lines."""
    tile_h, tile_w = minority_arr.shape
    mh, mw = mask.shape
    out = np.zeros((mh * tile_h, mw * tile_w), dtype=int)
    for i in range(mh):
        for j in range(mw):
            tile = minority_arr if mask[i, j] else majority_arr
            paste(out, tile, i * tile_h, j * tile_w)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Isolate the enlarged fractal patch encoded by the minority cell and return only that patch.

    Input structure:
    - The input is still a separator-lined board of repeated 3x3 cell interiors.
    - One rare cell pattern appears in a few positions; the common cell pattern fills the rest.

    Rule:
    - Read the rare cell's 3x3 interior as a binary blueprint, with its dominant color marking active pixels.
    - Expand that 3x3 blueprint by replacing each active pixel with the full minority tile and each inactive
      pixel with the full majority tile.
    - Return just this enlarged 3x3-by-3x3 tile mosaic, without the original 29x29 board or divider lines.

    Key insight:
    - The unusual cell is a self-referential template: its internal on/off layout dictates a larger copy of the
      same minority-versus-majority tile relationship, and the variant asks for the isolated enlarged motif only.
    """
    g = np.asarray(grid, dtype=int)
    majority_arr, minority_arr, mask = _extract_patterns(g)
    _assert_unique_placement(g, minority_arr, mask)
    return _build_scaled_patch(majority_arr, minority_arr, mask)


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


def _expected_output(majority, minority, mask):
    """Return the cropped 9x9 fractal patch expected from the variant solver."""
    return _build_scaled_patch(majority, minority, mask)


def _gen_attempt():
    """Single generation attempt: build expected output, choose input subset, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None

    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)
    out = _expected_output(majority, minority, mask)

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
