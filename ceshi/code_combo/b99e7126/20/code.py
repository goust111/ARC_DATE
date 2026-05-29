# Combo variant of b99e7126 (index 20)
# combo_spec: D1/0+D7/2
# Name: 频率排序 + 异常是多数

from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    Base task: one minority motif sits inside a sea of identical majority cells; the solver finds
#    the odd motif, reads its 3x3 interior as a mask, and fills the whole 3x3 cell block with
#    minority on active pixels and majority on inactive pixels.
#    Combo task: this relation is inverted. Suppose the source motif appears 4 times in the input
#    while all other 45 cells are mutually different singletons. The 4 repeated cells are only a
#    subset of the source mask's active positions; the output extends that repeated motif to all
#    6-7 active positions. The inactive positions are not repainted to a shared background anymore;
#    they keep their original singleton fillers. This preserves the fractal idea while making the
#    search strategy flip from "find the lone oddball" to "find the only exact repeat".
# b) Complexity calibration:
#    If we used a repeated background again, the inversion would collapse and the puzzle would look
#    too close to the original. If we chose the absolute least frequent motif over all cells, any
#    singleton distractor would incorrectly win. To keep the D1 frequency idea meaningful, the
#    solver ranks only repeated motif families; with D7 inversion there is exactly one such family.
#    The board still contains 40+ distinct motifs, so the repeated source is visually salient.
# c) Hidden assumptions that changed:
#    - Original `min(counts)` assumed the source was globally rare; now singleton anomalies are
#      more frequent as motif types, so the solver must ignore count-1 families.
#    - Original `majority_pat` was needed to repaint inactive mask cells; now inactive cells are
#      left unchanged because there is no repeated background family.
#    - The source still needs a unique dominant color so its interior can be read as a boolean mask.
#    - The shown source cells must fit a unique placement of the 3x3 mask within the 7x7 cell grid.
# d) Placement legality:
#    The generator picks a 3x3 mask placement inside the 7x7 board, chooses a subset of active
#    positions that yields exactly one valid placement, and assigns distinct singleton fillers to
#    every non-source cell. There is no overlap because each cell owns one 3x3 interior slot.
# e) Visual observability:
#    Across examples, humans see one exact motif repeated a few times and many one-off distractors.
#    The source cannot be recovered from hidden metadata such as color id ordering; only visible
#    repetition matters. The output always adds more copies of that same repeated motif.
# f) Visual cleanliness:
#    Singleton fillers are still LR-symmetric 3x3 motifs built from a small color palette, so the
#    board looks structured rather than noisy. The only repeated exact pattern is the source motif.
# g) Fusion-specific metric:
#    In every valid input, the source count is 3-4, while the absolute minimum motif count is 1 and
#    belongs to many singleton distractors. Therefore any solver that still relies on global minimum
#    frequency would fail; the generator explicitly enforces this non-degenerate regime.


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


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Extract every cell interior as both a flat key and a 2D array."""
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


def _count_patterns(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    """Count exact cell-interior motifs across the cell grid."""
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the only repeated cell motif across the larger fractal mask.

    Input structure:
    - The board is a 7x7 grid of 3x3 cell interiors separated by single-color divider lines.
    - Almost every cell motif is unique; exactly one exact motif repeats several times.
    - That repeated motif's dominant color encodes a 3x3 active/inactive mask, and only a subset
      of the active positions are already occupied by copies of the repeated motif.

    Rule:
    - Count exact cell motifs and keep only the repeated motif families.
    - Choose the uniquely least frequent repeated family; in this inverted setting it is also the
      only repeated "normal" motif hidden inside a majority of singleton anomalies.
    - Read the repeated motif's dominant color as the active mask, find the unique 3x3 placement
      consistent with the shown copies, and stamp that same repeated motif onto every active cell
      in the placement.
    - Leave every non-active cell exactly as it already is.

    Key insight: The source is no longer a lone outlier against a uniform background. Instead, the
    meaningful clue is the only motif that repeats at all, and its own pixels still blueprint where
    more copies of itself must appear on the cell grid.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _extract_cell_patterns(g, step)
    counts = _count_patterns(cell_pat)

    repeated = {pat: cnt for pat, cnt in counts.items() if cnt >= 2}
    assert len(repeated) == 1, "expected exactly one repeated motif family"
    assert any(cnt == 1 for cnt in counts.values()), "expected singleton anomalies around the source"

    source_pat, source_count = next(iter(repeated.items()))
    assert source_count in (3, 4, 5, 6, 7), "unexpected source multiplicity"
    source_arr = patterns[source_pat]

    vals, cnts = np.unique(source_arr, return_counts=True)
    order = np.argsort(cnts)[::-1]
    assert len(order) >= 2 and cnts[order[0]] > cnts[order[1]], "source motif needs a unique dominant color"
    active_color = int(vals[order[0]])
    mask = source_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == source_pat

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
    assert len(candidates) == 1, "source motif must admit a unique placement"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            if not mask[i, j]:
                continue
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = source_arr
    return out


# --------------- generator ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
SOURCE_GIVEN_SIZES = (3, 4)
COLOR_POOL = 10
MAX_RETRIES = 500

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_source_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate valid LR-symmetric source masks keyed by active-pixel count."""
    weights = [len(group) for group in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        inactive_count = MASK_SIZE * MASK_SIZE - active_count
        masks: list[np.ndarray] = []
        for k in range(1, len(_LR_GROUPS) + 1):
            for combo in combinations(range(len(_LR_GROUPS)), k):
                if sum(weights[i] for i in combo) != inactive_count:
                    continue
                mask = np.ones((MASK_SIZE, MASK_SIZE), dtype=bool)
                for i in combo:
                    for r, c in _LR_GROUPS[i]:
                        mask[r, c] = False
                if mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


def _precompute_filler_masks() -> list[np.ndarray]:
    """All non-uniform LR-symmetric boolean templates for singleton filler motifs."""
    masks: list[np.ndarray] = []
    for bits in range(1, (1 << len(_LR_GROUPS)) - 1):
        mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
        for i, group in enumerate(_LR_GROUPS):
            if (bits >> i) & 1:
                for r, c in group:
                    mask[r, c] = True
        masks.append(mask)
    return masks


_SOURCE_MASKS_BY_COUNT = _precompute_source_masks()
_FILLER_MASKS = _precompute_filler_masks()


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(int(v) for v in arr.ravel())


def _random_source_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_SOURCE_MASKS_BY_COUNT[active_count]).copy()


def _build_filler_library(shared: int, accent_colors: list[int]) -> dict[tuple[int, ...], np.ndarray]:
    """Create many distinct singleton motifs from a small palette."""
    library: dict[tuple[int, ...], np.ndarray] = {}
    for accent in accent_colors:
        for mask in _FILLER_MASKS:
            arr = np.where(mask, accent, shared).astype(int)
            library[_pattern_key(arr)] = arr
    return library


def _placement_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find every placement where the shown source cells could fit the source mask."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True

    candidates: list[tuple[int, int]] = []
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


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    """Choose a strict subset of active cells that uniquely determines the placement."""
    active = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
    sizes = list(SOURCE_GIVEN_SIZES)
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
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, fillers: dict[tuple[int, int], np.ndarray], source: np.ndarray, source_cells: set[tuple[int, int]]) -> np.ndarray:
    """Fill each cell with its singleton filler, then overwrite chosen source cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, fillers[(i, j)], *_cell_origin(i, j))
    for i, j in source_cells:
        paste(grid, source, *_cell_origin(i, j))
    return grid


def _frequency_profile(grid: np.ndarray) -> tuple[list[int], int]:
    """Return sorted motif multiplicities and the number of repeated motif families."""
    step, _, _ = _infer_cell_grid_params(grid)
    _, cell_pat = _extract_cell_patterns(grid, step)
    counts = _count_patterns(cell_pat)
    multiplicities = sorted(int(v) for v in counts.values())
    repeated_families = sum(1 for v in counts.values() if v >= 2)
    return multiplicities, repeated_families


def _gen_attempt() -> np.ndarray | None:
    """Build one valid input for the inverted frequency-ranked variant."""
    separator, shared, src_color, filler_a, filler_b = random.sample(range(COLOR_POOL), 5)

    source_mask = _random_source_mask()
    source = np.where(source_mask, src_color, shared).astype(int)
    source_key = _pattern_key(source)

    filler_library = _build_filler_library(shared, [src_color, filler_a, filler_b])
    filler_library.pop(source_key, None)
    if len(filler_library) < CELL_GRID_N * CELL_GRID_N:
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = {(top + int(r), left + int(c)) for r, c in np.argwhere(source_mask)}

    subset = _choose_given_subset(source_mask, top, left)
    if subset is None:
        return None
    given_cells = {(top + r, left + c) for r, c in subset}

    filler_items = list(filler_library.items())
    random.shuffle(filler_items)
    chosen_fillers = filler_items[: CELL_GRID_N * CELL_GRID_N]
    positions = [(i, j) for i in range(CELL_GRID_N) for j in range(CELL_GRID_N)]
    fillers = {pos: arr.copy() for pos, (_, arr) in zip(positions, chosen_fillers)}

    inp = _build_grid(separator, fillers, source, given_cells)
    out = _build_grid(separator, fillers, source, active_cells)

    in_mults, in_repeats = _frequency_profile(inp)
    out_mults, out_repeats = _frequency_profile(out)
    assert in_repeats == 1 and out_repeats == 1
    assert in_mults[0] == 1 and out_mults[0] == 1
    assert in_mults[-1] == len(given_cells)
    assert out_mults[-1] == len(active_cells)

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
