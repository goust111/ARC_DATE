# Combo variant of b99e7126 (index 47)
# combo_spec: D1/2+D14/2
# Name: 严格边界约束 + 位置规则选择

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste

# Design analysis for combo D1/2 + D14/2
# ---------------------------------------
# concretization_and_trace:
#   - Keep the original 29x29 board made of a 7x7 lattice of 3x3 cell interiors.
#   - Replace "find the minority cell by frequency" with a pure position rule:
#     inspect every non-majority special cell, discard the ones on the outer border
#     because a centered 3x3 macro-expansion would stick out there, then choose the
#     remaining special cell with smallest (row, col).
#   - The enlarged pattern is no longer searched over all 0..4 offsets. It is anchored
#     directly on the chosen source cell: the source must be the center of the 3x3 macro
#     block, so local source pixel (1,1) maps back to the source cell itself.
#   - Concrete example: if specials are at (0,4), (2,3), (5,5), the old logic would try
#     to identify one by rarity. The fused rule instead rejects (0,4) because a centered
#     3x3 macro block would need row -1, then picks (2,3) because it is the upper-left-most
#     remaining valid source. Its internal 3x3 mask is copied to the 3x3 neighborhood
#     centered on (2,3).
# complexity_calibration:
#   - Only one position rule is used: "upper-left-most special cell whose full 3x3 macro
#     neighborhood fits". This is visible in few-shot ARC examples and avoids overfitting
#     to many rule families.
#   - There are 3-4 special cells per sample, with at least one invalid border decoy and
#     at least one later valid decoy. That prevents collapse into "the only special cell"
#     while keeping the scene readable.
#   - The source mask still uses 6-7 active pixels and spans all four edges of its 3x3
#     micro-grid, so the fractal enlargement remains visually close to the original task.
# implicit_assumption_audit:
#   - Original `minority_pat = min(counts...)` is invalid here: all special cells may be
#     unique, and the chosen source is determined only by position after boundary filtering.
#   - Original placement search over all top/left offsets is also invalid here: once the
#     source is chosen, the macro block is forced to be centered on that source.
#   - The generator therefore guarantees:
#       1. a unique majority pattern,
#       2. at least one border special cell lexicographically earlier than the true source,
#       3. no other special cell lies inside the chosen source's 3x3 macro neighborhood.
# placement_legality:
#   - A valid source must satisfy 1 <= row,col <= 5 in the 7x7 cell lattice, which is the
#     strict in-bounds version of D14.
#   - Output is built from a full-majority board plus disjoint special cells, then the
#     centered 3x3 macro overlay writes only inside bounds, so there is no clipping,
#     overlap ambiguity, or wraparound.
# observability_and_tidiness:
#   - Border decoys visibly show why "just take the upper-left-most special cell" is wrong.
#   - All non-majority cells use the same two-color palette, so they look like one semantic
#     family of candidates; the deciding factor is location, not hidden color ids.
#   - The chosen source's mask expands into a compact 3x3 neighborhood, keeping the output
#     organized and easy to compare with the input.
# new_solver_assumptions_checked_by_generator:
#   - The source pattern's dominant color is unique and includes the center pixel.
#   - The majority pattern is the unique most frequent cell interior.
#   - The lexicographically first valid special cell is never also the lexicographically
#     first special cell overall; a border-filtered choice is always required.


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
MASK_SIZE = 3
COLOR_POOL = 10
MAX_RETRIES = 400
SPECIAL_COUNT_OPTIONS = (3, 4)
ACTIVE_PIXEL_COUNTS = (6, 7)

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


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


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _is_valid_source_pos(pos: tuple[int, int], n: int) -> bool:
    r, c = pos
    return 0 < r < n - 1 and 0 < c < n - 1


def _macro_cells_around(pos: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pos
    return [(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)]


def _active_macro_cells(pos: tuple[int, int], mask: np.ndarray) -> list[tuple[int, int]]:
    r0, c0 = pos
    return [
        (r0 + r - 1, c0 + c - 1)
        for r, c in np.argwhere(mask)
    ]


def _extract_cells(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
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
            pat = tuple(int(v) for v in pat_arr.ravel())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def _dominant_mask(arr: np.ndarray) -> tuple[int, np.ndarray]:
    vals, cnts = np.unique(arr, return_counts=True)
    order = np.argsort(-cnts)
    assert len(order) >= 2 and cnts[order[0]] > cnts[order[1]], "dominant color must be unique"
    color = int(vals[order[0]])
    mask = arr == color
    assert mask[1, 1], "source center pixel must be active"
    return color, mask


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _precompute_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    coords = [(r, c) for r in range(MASK_SIZE) for c in range(MASK_SIZE)]
    center = (1, 1)
    for active_count in ACTIVE_PIXEL_COUNTS:
        for combo in combinations(coords, active_count):
            if center not in combo:
                continue
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            if not mask[0, :].any():
                continue
            if not mask[2, :].any():
                continue
            if not mask[:, 0].any():
                continue
            if not mask[:, 2].any():
                continue
            masks.append(mask)
    return masks


_SPECIAL_MASKS = _precompute_masks()


def _mask_to_pattern(mask: np.ndarray, active_color: int, passive_color: int) -> np.ndarray:
    return np.where(mask, active_color, passive_color).astype(int)


def _all_cells() -> list[tuple[int, int]]:
    return [(r, c) for r in range(CELL_GRID_N) for c in range(CELL_GRID_N)]


_ALL_CELLS = _all_cells()
_SAFE_CELLS = [pos for pos in _ALL_CELLS if _is_valid_source_pos(pos, CELL_GRID_N)]
_BORDER_CELLS = [pos for pos in _ALL_CELLS if pos not in _SAFE_CELLS]


def _choose_special_positions(source_pos: tuple[int, int], special_count: int) -> list[tuple[int, int]] | None:
    neighborhood = set(_macro_cells_around(source_pos))
    earlier_invalid = [pos for pos in _BORDER_CELLS if pos < source_pos and pos not in neighborhood]
    later_valid = [pos for pos in _SAFE_CELLS if pos > source_pos and pos not in neighborhood]
    spare_border = [pos for pos in _BORDER_CELLS if pos not in neighborhood and pos not in earlier_invalid]
    if not earlier_invalid or not later_valid:
        return None

    chosen = [random.choice(earlier_invalid), source_pos, random.choice(later_valid)]
    used = set(chosen)

    if special_count == 4:
        extras = [pos for pos in later_valid + spare_border if pos not in used]
        if not extras:
            return None
        chosen.append(random.choice(extras))

    return chosen


def _build_grid(separator: int, majority: np.ndarray, special_cells: dict[tuple[int, int], np.ndarray]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for pos, pat in special_cells.items():
        paste(grid, pat, *_cell_origin(*pos))
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Select the boundary-safe source cell by position and expand its motif as a centered fractal block.

    Input structure:
    - The board is a regular lattice of equal 3x3 cell interiors separated by one-color divider lines.
    - One interior pattern is the global majority background for almost all cells.
    - Several other cells are special candidates with distinct non-majority motifs; some of them may sit on the outer border.

    Rule:
    - Ignore frequency among the special cells. First discard any special cell on the outer border, because a 3x3 macro-copy centered on it would leave the grid.
    - Among the remaining special cells, choose the upper-left-most one as the source.
    - Read that source cell's dominant-color mask and stamp the whole source cell pattern onto the centered 3x3 neighborhood: active mask pixels become source cells, inactive pixels become majority cells.

    Key insight:
    The chosen source is defined jointly by geometry and boundary legality: the correct cell is not "the rare one", but the earliest special cell whose full enlarged neighborhood fits entirely inside the 7x7 cell lattice.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _extract_cells(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) >= 2, "need a majority pattern plus special candidates"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    candidates = [
        (i, j)
        for i in range(n)
        for j in range(n)
        if cell_pat[i][j] != majority_pat
    ]
    valid = [pos for pos in candidates if _is_valid_source_pos(pos, n)]
    assert valid, "need at least one boundary-safe special cell"
    source_pos = min(valid)
    sr, sc = source_pos

    source_pat = cell_pat[sr][sc]
    source_arr = patterns[source_pat]
    _active_color, mask = _dominant_mask(source_arr)

    out = g.copy()
    for r in range(MASK_SIZE):
        for c in range(MASK_SIZE):
            rr = sr + r - 1
            cc = sc + c - 1
            assert 0 <= rr < n and 0 <= cc < n, "source-centered macro block must stay in bounds"
            target = source_arr if mask[r, c] else majority_arr
            r0 = rr * step + 1
            c0 = cc * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    separator, maj_exclusive, shared, special_color = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)

    source_mask = random.choice(_SPECIAL_MASKS)
    source_pattern = _mask_to_pattern(source_mask, special_color, shared)

    other_masks = random.sample(
        [mask for mask in _SPECIAL_MASKS if not np.array_equal(mask, source_mask)],
        k=3,
    )
    decoy_patterns = [_mask_to_pattern(mask, special_color, shared) for mask in other_masks]

    source_pos_choices = [
        pos
        for pos in _SAFE_CELLS
        if pos[0] <= 3
    ]
    source_pos = random.choice(source_pos_choices)

    special_count = random.choice(SPECIAL_COUNT_OPTIONS)
    positions = _choose_special_positions(source_pos, special_count)
    if positions is None:
        return None

    source_neighborhood = set(_macro_cells_around(source_pos))
    if any(pos != source_pos and pos in source_neighborhood for pos in positions):
        return None

    specials: dict[tuple[int, int], np.ndarray] = {source_pos: source_pattern}
    decoy_positions = [pos for pos in positions if pos != source_pos]
    random.shuffle(decoy_patterns)
    for pos, pat in zip(decoy_positions, decoy_patterns):
        specials[pos] = pat

    inp = _build_grid(separator, majority, specials)

    out_specials = dict(specials)
    for pos in _active_macro_cells(source_pos, source_mask):
        out_specials[pos] = source_pattern
    out = _build_grid(separator, majority, out_specials)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    first_special = min(specials)
    valid_specials = sorted(pos for pos in specials if _is_valid_source_pos(pos, CELL_GRID_N))
    if first_special == source_pos:
        return None
    if not valid_specials or valid_specials[0] != source_pos:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        built = _gen_attempt()
        if built is not None:
            inp, _out = built
            return inp
    raise RuntimeError("failed to generate a valid combo input for b99e7126 D1/2+D14/2")
