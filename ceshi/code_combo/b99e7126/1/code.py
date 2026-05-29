# Combo variant of b99e7126 (index 1)
# combo_spec: D1/0+D6/0
# Name: 频率排序 + 棋盘式双图案铺贴

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    The 29x29 canvas still holds a 7x7 lattice of 3x3 cell interiors separated by
#    1-pixel dividers. The base is no longer one repeated tile: parity-even cells use
#    tile A and parity-odd cells use tile B, so the untouched board is a visible
#    checkerboard. A third tile C is the fractal source. Example trace:
#    - Original task: all inactive macro-cells inside the completed 3x3 block became
#      the same majority tile.
#    - This combo: after locating the source block, active macro-cells become C but
#      inactive macro-cells must alternate back to A/B by absolute cell parity.
#    - Source selection also changes: instead of "the odd one out", the solver counts
#      exact cell motifs and chooses the uniquely least frequent motif family. With
#      counts like A=22, B=24, C=4, the source is identified by frequency and the
#      background is reconstructed by checkerboard parity.
# b) Complexity calibration:
#    Keeping exactly three visible motif families is deliberate. Two high-frequency
#    motifs are needed to make the checkerboard obvious, while one low-frequency motif
#    is enough to force motif counting without cluttering the board with many extra
#    exceptions. Source cells appear 3-4 times in the input so they are clearly rarer
#    than both checkerboard motifs but still repeated often enough to define a family.
# c) Hidden-assumption audit against the base solver:
#    - "majority_pat = max(counts)" is invalid because non-source cells split into A
#      and B.
#    - "target_pat = source if mask else majority" is invalid because inactive macro
#      positions can require two different tiles depending on parity.
#    - The old rarity logic assumed one minority motif against one dominant motif. The
#      combo instead assumes exactly one uniquely least frequent motif and exactly two
#      remaining motifs that fit one checkerboard assignment.
#    - Placement search from the shown source subset still works unchanged because it
#      only depends on where source cells already exist.
# d) Placement legality:
#    The generator keeps the 3x3 source mask fully inside the 7x7 cell lattice by
#    choosing offsets in 0..4. Only full cell interiors are overwritten, so no overlap
#    is possible. Input source cells are a strict subset of the output source cells,
#    and every subset is filtered so the macro placement is unique.
# e) Visual observability:
#    A and B are distinct asymmetric 3x3 motifs using the same two colors, so humans
#    can see the checkerboard without relying on color ids. C introduces one extra
#    color and remains the least frequent motif. The mask is constrained so inactive
#    macro positions hit both checkerboard parities; otherwise the completed region
#    could accidentally look compatible with a single background tile and D6 would be
#    visually muted.
# f) Visual cleanliness:
#    Each cell interior contains exactly one motif among A, B, or C. There are no
#    extra markers, outlines, or overlaid counters. Divider lines remain unchanged, so
#    the board still reads as one clean cell lattice.
# g) Inferability of the change:
#    The color budget stays at four colors: separator, shared base/source background,
#    base-exclusive, and source-exclusive. A/B differ by arrangement rather than by
#    introducing more colors, while C's frequent source-exclusive color visibly defines
#    the internal 3x3 mask. Across examples, a solver can consistently infer:
#    least-frequent motif -> source blueprint, parity pattern -> inactive fill rule.


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


def _extract_cell_patterns(
    g: np.ndarray, step: int
) -> tuple[int, int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Extract each cell interior both as a flat hashable key and as a 2D array."""
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
    return n, inner, patterns, cell_pat


def _count_patterns(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    """Count exact cell-interior motifs over the cell lattice."""
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def _checkerboard_score(
    cell_pat: list[list[tuple[int, ...]]],
    source_pat: tuple[int, ...],
    even_pat: tuple[int, ...],
    odd_pat: tuple[int, ...],
) -> int:
    """Count non-source cells that disagree with a proposed checkerboard assignment."""
    mismatches = 0
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat == source_pat:
                continue
            expected = even_pat if (i + j) % 2 == 0 else odd_pat
            mismatches += int(pat != expected)
    return mismatches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal source motif chosen by frequency, over a checkerboard base.

    Input structure:
    - The board is split into equal 3x3 cell interiors by single-color divider lines.
    - Two high-frequency motifs alternate across the non-source cells in checkerboard order.
    - A third motif appears only a few times; it is the uniquely least frequent repeated motif.

    Rule:
    - Count exact cell motifs and select the uniquely least frequent motif as the source tile.
    - Interpret that source tile's most frequent color as a 3x3 boolean mask and find the unique
      placement whose active positions contain all shown source cells.
    - Fill the full 3x3 macro-block: active positions get the source tile, inactive positions get
      whichever checkerboard base tile matches that absolute cell parity.

    Key insight: The rare tile is still a self-referential blueprint, but now solving requires two
    coupled inferences: rank motif frequencies to find the source, then preserve the A/B checkerboard
    when expanding the source mask back to cell scale.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n, inner, patterns, cell_pat = _extract_cell_patterns(g, step)

    counts = _count_patterns(cell_pat)
    assert len(counts) == 3, "combo variant expects exactly 3 visible motif families"

    ordered = sorted(counts.items(), key=lambda kv: kv[1])
    assert ordered[0][1] < ordered[1][1], "source motif frequency must be uniquely minimal"
    source_pat = ordered[0][0]
    base_pats = [pat for pat, _ in ordered[1:]]
    assert len(base_pats) == 2, "need exactly two checkerboard base motifs"

    assignments = []
    for even_pat, odd_pat in ((base_pats[0], base_pats[1]), (base_pats[1], base_pats[0])):
        score = _checkerboard_score(cell_pat, source_pat, even_pat, odd_pat)
        assignments.append((score, even_pat, odd_pat))
    assignments.sort(key=lambda item: item[0])
    assert assignments[0][0] == 0, "non-source cells must match one checkerboard assignment"
    assert assignments[0][0] < assignments[1][0], "checkerboard parity assignment must be unique"
    _, even_pat, odd_pat = assignments[0]

    source_arr = patterns[source_pat]
    vals, cnts = np.unique(source_arr, return_counts=True)
    order = np.argsort(cnts)[::-1]
    assert len(order) >= 2 and cnts[order[0]] > cnts[order[1]], "source tile needs a unique active color"
    active_color = int(vals[order[0]])
    mask = source_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == source_pat

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
    assert len(candidates) == 1, "source placement must be uniquely determined"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            abs_i = top + i
            abs_j = left + j
            if mask[i, j]:
                target_pat = source_arr
            else:
                target_key = even_pat if (abs_i + abs_j) % 2 == 0 else odd_pat
                target_pat = patterns[target_key]
            r0 = abs_i * step + 1
            c0 = abs_j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = (3, 4)
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

_BASE_SEEDS = [
    np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=bool),
]


def _precompute_source_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate source masks that expose both active and inactive checkerboard parities."""
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
                for idx in combo:
                    for r, c in _LR_GROUPS[idx]:
                        mask[r, c] = False
                if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
                    continue
                active_parities = {(int(r) + int(c)) % 2 for r, c in np.argwhere(mask)}
                inactive_parities = {(int(r) + int(c)) % 2 for r, c in np.argwhere(~mask)}
                if len(active_parities) != 2 or len(inactive_parities) != 2:
                    continue
                masks.append(mask)
        by_count[active_count] = masks
    return by_count


def _precompute_base_masks() -> list[np.ndarray]:
    """Build a catalog of distinct asymmetric checkerboard motifs."""
    seen: set[tuple[int, ...]] = set()
    catalog: list[np.ndarray] = []
    for seed in _BASE_SEEDS:
        variants = [
            seed,
            np.rot90(seed, 1),
            np.rot90(seed, 2),
            np.rot90(seed, 3),
            np.fliplr(seed),
            np.flipud(seed),
        ]
        for variant in variants:
            key = tuple(int(v) for v in variant.ravel())
            if key not in seen:
                seen.add(key)
                catalog.append(variant.copy())
    return catalog


_SOURCE_MASKS_BY_COUNT = _precompute_source_masks()
_BASE_MASKS = _precompute_base_masks()


def _random_source_mask() -> np.ndarray:
    """Pick one valid 3x3 source mask."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_SOURCE_MASKS_BY_COUNT[active_count]).copy()


def _random_checkerboard_pair(shared: int, base_exclusive: int) -> tuple[np.ndarray, np.ndarray]:
    """Choose two clearly different base motifs using the same two colors."""
    for _ in range(50):
        mask_a, mask_b = random.sample(_BASE_MASKS, 2)
        if np.count_nonzero(mask_a != mask_b) < 3:
            continue
        arr_a = np.where(mask_a, base_exclusive, shared).astype(int)
        arr_b = np.where(mask_b, base_exclusive, shared).astype(int)
        return arr_a, arr_b
    raise RuntimeError("failed to sample a distinct checkerboard pair")


def _placement_candidates(
    mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int
) -> list[tuple[int, int]]:
    """Find every placement whose active positions can contain the shown source subset."""
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


def _subset_spans_both_parities(subset: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    """Keep visible source samples on both checkerboard parities."""
    parities = {(top + r + left + c) % 2 for r, c in subset}
    return len(parities) == 2


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    """Choose a strict active subset that still pins down the macro placement uniquely."""
    active = [tuple(int(v) for v in coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _subset_spans_both_parities(subset, top, left)
            and _placement_candidates(mask, subset, top, left) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(
    separator: int,
    even_pat: np.ndarray,
    odd_pat: np.ndarray,
    source_pat: np.ndarray,
    source_cells: list[tuple[int, int]],
) -> np.ndarray:
    """Render the checkerboard base, then overwrite selected cells with the source motif."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            base_pat = even_pat if (i + j) % 2 == 0 else odd_pat
            paste(grid, base_pat, *_cell_origin(i, j))
    for i, j in source_cells:
        paste(grid, source_pat, *_cell_origin(i, j))
    return grid


def _generate_case() -> dict[str, object] | None:
    """Construct one valid checkerboard-plus-frequency combo instance."""
    separator, shared, base_exclusive, source_exclusive = random.sample(range(COLOR_POOL), 4)
    even_pat, odd_pat = _random_checkerboard_pair(shared, base_exclusive)
    mask = _random_source_mask()
    source_pat = np.where(mask, source_exclusive, shared).astype(int)

    motifs = [even_pat, odd_pat, source_pat]
    if any(np.array_equal(motifs[i], motifs[j]) for i in range(len(motifs)) for j in range(i + 1, len(motifs))):
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + int(r), left + int(c)) for r, c in np.argwhere(mask)]

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]

    inp = _build_grid(separator, even_pat, odd_pat, source_pat, given_cells)
    out = _build_grid(separator, even_pat, odd_pat, source_pat, active_cells)

    step, _, _ = _infer_cell_grid_params(inp)
    _, _, _, cell_pat = _extract_cell_patterns(inp, step)
    counts = _count_patterns(cell_pat)
    freqs = sorted(counts.values())
    assert len(counts) == 3
    assert freqs[0] < freqs[1] <= freqs[2]

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    return {
        "input": inp,
        "output": out,
        "mask": mask,
        "top": top,
        "left": left,
        "given_cells": given_cells,
        "active_cells": active_cells,
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        case = _generate_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
