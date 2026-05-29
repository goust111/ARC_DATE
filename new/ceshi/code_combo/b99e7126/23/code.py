# Combo variant of b99e7126 (index 23)
# combo_spec: D9/0+D10/1
# Name: 先叠加后提取 + 输出紧凑裁剪

# Design analysis for the combo variant
# -------------------------------------
# spec:
#   variant_goal:
#     Keep the original fractal-completion mechanism, but treat the completed
#     3x3 cell block as an intermediate object rather than the final answer.
#     The final output is the tight cell-aligned crop of the cells that were
#     newly written by that completion.
#   concrete_behavior:
#     - Input remains a 29x29 board: a 7x7 lattice of 3x3 cell interiors
#       separated by 1-pixel divider lines.
#     - The rare cell pattern still encodes a 3x3 binary mask through its
#       dominant color, and the shown rare cells still pin down one placement
#       of that mask in the 7x7 cell lattice.
#     - The solver must first reconstruct the full 3x3 completed overlay block.
#     - Let "shown active cells" be the rare cells already present in the input.
#       Let "missing active cells" be the active mask positions that the overlay
#       adds on top of majority cells.
#     - Output only the minimal cell-aligned rectangle covering those missing
#       active cells, copied from the completed overlay result with divider frame.
#   trace_example:
#     - Suppose the completed local mask is:
#         0 1 0
#         1 1 1
#         1 1 1
#       and the input already shows local active cells {(0,1), (1,0), (2,1)}.
#     - Original task:
#       fill every active position and return the whole 29x29 board.
#     - D9 alone:
#       fill every active position and return the whole 13x13 local 3x3 block.
#     - This combo:
#       fill every active position first, then identify the newly added active
#       cells {(1,1), (1,2), (2,0), (2,2)} and crop the completed block to
#       their bounding rectangle. Here that would be a 9x13 local mini-grid,
#       not the full board and not necessarily the full 3x3 block.
#   complexity_guardrails:
#     - Keep the same mask vocabulary, texture construction, and unique
#       placement logic so the seed remains recognizable.
#     - Crop only on cell boundaries. Raw-pixel cropping would remove divider
#       structure and make the result harder to parse as a mini-grid.
#     - Reject generator samples whose missing-cell bbox is the full 3x3 area,
#       because those would visually collapse back to the D9-only behavior.
#   implicit_assumptions_review:
#     - `_infer_cell_grid_params` still assumes full divider rows/cols on the
#       input boundary. That remains true because only the output changes.
#     - Pattern counting still assumes one majority texture and one rarer
#       minority texture. Generator keeps exactly those two cell interiors.
#     - Active-color detection assumes the minority-exclusive color is the most
#       frequent one in the minority cell. Generator keeps 6-7 active pixels.
#     - Candidate search still assumes the shown minority cells uniquely pin
#       down one placement. Generator filters subsets for that property.
#     - The new post-processing step assumes at least one active cell is missing.
#       Generator always uses a strict subset of active cells.
#     - The new fusion-specific observability assumption is that the missing
#       cells' bbox is strictly smaller than the full 3x3 block. Generator
#       enforces that so the combo does not degenerate into D9 alone.
#   placement_legality:
#     - The 3x3 mask is still placed via top/left in 0..4, so the completed
#       block fits inside the 7x7 cell board.
#     - The final crop is a sub-rectangle of that already-legal block, so it
#       cannot overlap anything else or go out of bounds.
#   visual_observability:
#     - Inputs still show a sparse, incomplete rare-cell configuration on the
#       full board, so the solver must infer the omitted active cells.
#     - Outputs now vary in size according to the bbox of the newly added cells
#       (for example 5x9, 9x9, 9x13, 13x9) instead of always being 29x29 or
#       always being the same 13x13 local crop.
#     - Keeping divider frames makes the extracted region read as a coherent
#       mini-grid rather than an arbitrary texture snippet.
#   cleanliness:
#     - The output contains only the portion of the completion that changed.
#       Untouched outer board context is removed, and already-shown minority
#       cells outside the crop do not clutter the final image.
#   color_reasoning:
#     - No branch depends on numeric color ids.
#     - Colors keep only structural roles: separator, shared,
#       majority-exclusive, minority-exclusive.
#   new_solver_assumptions_to_match_in_generator:
#     - Input must show both majority and minority cells.
#     - Input must omit at least one active cell.
#     - The omitted cells must have a bbox smaller than 3x3.

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


def _solve_mask_placement(g: np.ndarray):
    """Recover cell patterns, local occupancy, and the unique mask placement."""
    step, n, _inner, patterns, cell_pat, counts = _cell_pattern_data(g)
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns (minority + majority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

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
    existing_block = existing[top : top + mh, left : left + mw].copy()
    return step, majority_arr, minority_arr, mask, existing_block


def _render_completed_block(
    line_color: int,
    step: int,
    majority_arr: np.ndarray,
    minority_arr: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Render the fully completed local mask block with its outer divider frame."""
    mh, mw = mask.shape
    inner = step - 1
    out = np.full((mh * step + 1, mw * step + 1), int(line_color), dtype=int)
    for i in range(mh):
        for j in range(mw):
            target_pat = minority_arr if mask[i, j] else majority_arr
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Bounding box of True cells in a local mask."""
    coords = np.argwhere(mask)
    assert coords.size > 0, "mask bbox requires at least one True cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_cell_block(g: np.ndarray, top: int, left: int, cells_h: int, cells_w: int, step: int) -> np.ndarray:
    """Crop a cell-aligned block while preserving its outer divider frame."""
    r0 = top * step
    c0 = left * step
    r1 = (top + cells_h) * step
    c1 = (left + cells_w) * step
    return g[r0 : r1 + 1, c0 : c1 + 1].copy()


def _crop_changed_region(block: np.ndarray, changed_mask: np.ndarray, step: int) -> np.ndarray:
    """Crop the completed local block to the bbox of newly written active cells."""
    r0, c0, r1, c1 = _mask_bbox(changed_mask)
    return _crop_cell_block(block, r0, c0, r1 - r0 + 1, c1 - c0 + 1, step)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the fractal overlay, then return only the newly added local region.

    Input structure:
    - A full divider lattice partitions the board into equal square cells.
    - Most cells share one interior texture, while a smaller set shows one odd texture.
    - Those odd cells form only a partial glimpse of a larger 3x3 self-similar placement.

    Rule:
    - Decode the odd cell's dominant-color pattern as a 3x3 active/inactive mask.
    - Find the unique 3x3 placement of that mask that matches the shown odd cells.
    - Reconstruct the entire local 3x3 overlay block, filling active entries with the odd texture and inactive entries with the majority texture.
    - Compare the completed block to the input evidence, keep only the cells that had to be newly filled, and return the tight cell-aligned crop of that changed region.

    Key insight:
    The odd cell still acts as both texture and blueprint, but the final answer is no longer the whole completion. The important object is the part of the completion that was missing from the input, so the solver must overlay first and crop second.
    """
    g = np.asarray(grid, dtype=int)
    step, majority_arr, minority_arr, mask, existing_block = _solve_mask_placement(g)
    completed_block = _render_completed_block(int(g[0, 0]), step, majority_arr, minority_arr, mask)
    changed_mask = mask & ~existing_block
    assert np.any(changed_mask), "input must omit at least one active cell"
    return _crop_changed_region(completed_block, changed_mask, step)


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


def _placement_candidates(mask, subset, top, left):
    """Find all offsets where the shown active cells are consistent with mask."""
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


def _changed_bbox_shape(mask: np.ndarray, subset) -> tuple[int, int]:
    """Shape of the bbox covering the active cells omitted from the input subset."""
    shown = _subset_mask(mask.shape, subset)
    changed = mask & ~shown
    r0, c0, r1, c1 = _mask_bbox(changed)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _choose_given_subset(mask, top, left):
    """
    Choose a shown subset that uniquely determines the placement and leaves a
    non-full changed-region bbox, so the combo stays visibly distinct from D9.
    """
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    valid_choices = []
    for size in sizes:
        for subset in combinations(active, size):
            if _placement_candidates(mask, subset, top, left) != [(top, left)]:
                continue
            if _changed_bbox_shape(mask, subset) == (MASK_SIZE, MASK_SIZE):
                continue
            valid_choices.append(subset)
        if valid_choices:
            break
    if not valid_choices:
        return None
    return random.choice(valid_choices)


def _cell_origin(i, j):
    """Pixel origin of cell (i, j) interior inside the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _subset_mask(shape: tuple[int, int], subset) -> np.ndarray:
    """Boolean mask for the local active cells already shown in the input."""
    out = np.zeros(shape, dtype=bool)
    for r, c in subset:
        out[r, c] = True
    return out


def _build_grid(separator, majority, minority, active_cells):
    """Construct the full 29x29 board from cell patterns."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt():
    """Build an input whose output is the tight crop of newly added active cells."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    completed_block = _render_completed_block(separator, CELL_STEP, majority, minority, mask)

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)
    shown = _subset_mask(mask.shape, subset)
    expected = _crop_changed_region(completed_block, mask & ~shown, CELL_STEP)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
