# Variant of b99e7126
# Dimension: D16 — 叠加后的视觉效果处理 (transform)
# Change: 简单覆盖——放大图案的前景区域直接覆写原网格对应位置... → XOR式叠加：叠加区域的颜色取异或

# Design analysis
# a) Concrete behavior and trace:
#    The original task uses the minority cell's 3x3 interior as a cell-level blueprint. After locating
#    that blueprint inside the 7x7 cell lattice, every active blueprint pixel simply becomes a minority
#    cell in the output. In this variant, the same blueprint is still inferred and placed in the same
#    way, but the placed blueprint is treated as a second foreground layer that toggles the current
#    cell state. A mask-active cell flips: minority -> majority, majority -> minority. A mask-inactive
#    cell stays unchanged. Example: if the hidden 3x3 mask has six active positions and the input
#    already shows three of them as minority cells, then the original task would end with all six
#    active positions as minority. The XOR variant instead turns those shown three back to majority
#    and turns the other three active positions into minority, producing the symmetric difference.
# b) Complexity calibration:
#    Keeping the original 7x7 grid, 3x3 cell interiors, and 3x3 mask preserves the original inference
#    burden. The output change is visible because both "disappear" and "appear" events happen inside
#    the same placed block. Active counts stay at 6-7 while the input shows only 3-4 minority cells,
#    so every sample has both kinds of toggles and does not collapse back to simple completion.
# c) Review of original assumptions:
#    - Grid-divider inference is unchanged because the divider geometry is identical.
#    - Cell-pattern counting still assumes exactly one dominant pattern and one minority pattern in the
#      input; the generator keeps that property.
#    - Using the most frequent color inside the minority cell as the active color still holds because
#      the generator continues to encode the blueprint with a 6-7 pixel foreground.
#    - The original final loop assumed mask-active means unconditional overwrite with minority. That is
#      the key assumption that breaks for D16. The new loop must instead inspect the current cell state
#      and flip it only where mask is active.
# d) Placement legality:
#    The hidden 3x3 blueprint still fits inside the 7x7 cell grid via offsets 0..4. Input minority
#    cells are always chosen as a subset of active blueprint cells, so the inferred placement remains
#    valid and unique. Toggling never crosses cell borders because each operation rewrites one complete
#    3x3 cell interior at a legal cell origin.
# e) Visual observability:
#    Across examples, the solver-visible clue remains the same: the minority cell interior is the
#    enlarged block's blueprint. The new rule is visible because the output does not merely add missing
#    minority cells; it swaps which active positions are lit. Human solvers can compare input/output
#    and see that shown minority cells vanish while complementary positions appear inside the same 3x3
#    placement, which is the boolean-overlay effect.
# f) Visual cleanliness:
#    There are still only separator lines plus two cell patterns. No extra overlay colors or partial
#    cell corruption are introduced, so the block remains easy to read.
# g) Color inference:
#    No invisible color-id arithmetic is used. Colors only mark the existing majority/minority cell
#    patterns, and the XOR behavior is expressed structurally by presence/absence of the minority cell
#    pattern at active blueprint positions.
# h) Generator-side guarantee of the new behavior:
#    Because the input shows a strict subset of the active blueprint positions, the XOR result always
#    differs from both the input and the original completion output: some current minority cells are
#    toggled off, and some currently-majority cells are toggled on.
# i) New solver assumptions and generator support:
#    The new main() assumes every visible minority cell belongs to the hidden placed mask and that
#    only mask-active cells are toggled. The generator enforces this by choosing input minority cells
#    only from active mask positions and by verifying `main(input)` against the constructed XOR output.

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover a hidden cell-grid blueprint and apply it as a toggle mask over the current arrangement.

    Input structure:
    - The board is partitioned by single-color divider lines into a 7x7 lattice of 3x3 cell interiors.
    - Nearly every cell repeats one interior pattern, while a smaller set shows a second pattern.

    Rule:
    - Read the minority cell's 3x3 interior as a binary blueprint, using its more frequent color as the active pixels.
    - Find the unique placement of that 3x3 blueprint on the 7x7 cell lattice that contains all currently minority cells.
    - Apply the placed blueprint with XOR-style toggling: every blueprint-active cell flips between the majority and minority patterns, while blueprint-inactive cells are left alone.

    Key insight:
    - The special cell is still a miniature map of a larger 3x3 region, but the larger region is not completed by painting it in; it interferes with the current grid, so active positions switch state instead of being overwritten unconditionally.
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

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            if not mask[i, j]:
                continue
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            target_pat = majority_arr if existing[top + i, left + j] else minority_arr
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


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
                for idx in combo:
                    for r, c in _LR_GROUPS[idx]:
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
    if all(value == choices[0] for value in choices):
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
    """Find all offsets where the given subset of active cells is consistent with mask."""
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


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, active_cells) -> np.ndarray:
    """Construct a full grid with majority cells everywhere except the chosen minority cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _xor_active_cells(active_cells, given_cells):
    """Cell-level XOR between the full blueprint support and the currently visible minority cells."""
    full = set(active_cells)
    visible = set(given_cells)
    return sorted(full ^ visible)


def _gen_attempt():
    """Single generation attempt: build XOR output, derive input subset, and verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + int(coord[0]), left + int(coord[1])) for coord in np.argwhere(mask)]

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    xor_cells = _xor_active_cells(active_cells, given_cells)

    inp = _build_grid(separator, majority, minority, given_cells)
    out = _build_grid(separator, majority, minority, xor_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 D16 input")
