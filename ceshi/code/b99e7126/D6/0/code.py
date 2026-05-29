# Variant of b99e7126
# Dimension: D6 — 网格铺贴结构：单元格的重复模式 (tiling)
# Change: 整个网格由完全相同的NxN单元格无缝平铺，只有一个单元格图案不同... → 棋盘式双图案铺贴

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the D6 variant
# a) Concrete behavior:
#    The 29x29 canvas still contains a 7x7 grid of 3x3 cell interiors split by 1-pixel dividers.
#    The old task used one background cell pattern everywhere except the anomaly cells. Here the
#    background is a checkerboard: parity-even cells repeat pattern A, parity-odd cells repeat
#    pattern B. The anomaly cells use a third pattern C. Pattern C still contains a dense "active"
#    color whose 3x3 arrangement becomes the macro-level mask. The output completes all mask=True
#    positions with C; mask=False positions must preserve the local checkerboard base, not a single
#    global majority pattern. Example of the changed step: in the old logic, a missing inactive slot
#    inside the 3x3 macro-block was always filled with one repeated tile. In this variant, the same
#    inactive slot must be filled with A or B depending on the absolute cell parity, so two nearby
#    inactive slots can receive different tiles.
# b) Complexity calibration:
#    Keep the original geometry and fractal rule unchanged so D6 is the only conceptual change.
#    Use exactly three cell patterns in the input: A, B, and C. A/B repeat many times, C appears
#    only 3-4 times in the input, so the anomaly is still easy to isolate, while the alternating
#    checkerboard is visually obvious because A and B are distinct asymmetric 3x3 motifs.
# c) Hidden assumptions from the old solver:
#    - "majority_pat = max(counts)" assumed every non-anomaly cell is identical. This is invalid now.
#    - "target_pat = minority if mask else majority" assumed one uniform fallback tile. Also invalid.
#    - Placement search based only on the anomaly subset still works, but output synthesis must use
#      parity-aware background reconstruction. The new solver therefore infers C from rarity, then
#      infers the A/B parity assignment by checking which assignment yields zero mismatches on all
#      non-anomaly cells.
# d) Placement legality:
#    The 3x3 anomaly mask is still placed inside the 7x7 cell lattice at offsets 0..4. Generator
#    chooses a strict subset of anomaly cells whose positions uniquely determine that offset. No cell
#    overlaps occur because every cell occupies one fixed 3x3 interior between divider lines.
# e) Visual observability:
#    The checkerboard change must be visible from 2-4 examples, so A and B are different by shape,
#    not by hidden metadata. The anomaly mask is additionally constrained so its inactive pixels span
#    both parities; this guarantees the completed macro-block contains both background tile types,
#    making the D6 change visible inside the transformed region itself.
# f) Visual cleanliness:
#    Only one layer exists per cell interior: either A, B, or C. Dividers remain untouched. The
#    output therefore stays as legible as the source task instead of adding overlapping markers.
# g) Color inference:
#    Keep the color budget small: separator + shared + base-exclusive + anomaly-exclusive. A and B
#    share the same two colors and differ only by arrangement, while C introduces one distinct color.
#    This keeps the color story simple while still making the anomaly pattern easy to identify.


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


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[int, int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
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
    return n, inner, patterns, cell_pat


def _checkerboard_score(
    cell_pat: list[list[tuple[int, ...]]],
    anomaly_pat: tuple[int, ...],
    even_pat: tuple[int, ...],
    odd_pat: tuple[int, ...],
) -> int:
    mismatches = 0
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat == anomaly_pat:
                continue
            expected = even_pat if (i + j) % 2 == 0 else odd_pat
            mismatches += int(pat != expected)
    return mismatches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal anomaly overlay on top of a checkerboard-tiled cell lattice.

    Input structure:
    - Grid divided into NxN cells by separator lines of one color.
    - Non-anomaly cells alternate between two repeating 3x3 interior patterns in checkerboard order.
    - A few cells show a third interior pattern; these are visible samples of the anomaly tile.

    Rule:
    - Read the anomaly tile's own 3x3 interior as a binary mask, using its most frequent color as the active pixels.
    - Place that mask onto the cell grid so the already-shown anomaly cells land on active positions.
    - Fill every remaining active position with the anomaly tile, and fill every inactive position with the
      checkerboard base tile that matches that cell's parity.

    Key insight: The rare third tile is still a self-referential blueprint, but now it must be expanded over a
    two-pattern checkerboard background rather than over a single uniform majority tile.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n, inner, patterns, cell_pat = _extract_cell_patterns(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) == 3, "variant expects exactly 3 cell patterns: A, B, anomaly"

    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    anomaly_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    base_pats = [pat for pat, _ in ordered if pat != anomaly_pat]
    assert len(base_pats) == 2, "need exactly two checkerboard base patterns"

    assignments = []
    for even_pat, odd_pat in ((base_pats[0], base_pats[1]), (base_pats[1], base_pats[0])):
        score = _checkerboard_score(cell_pat, anomaly_pat, even_pat, odd_pat)
        assignments.append((score, even_pat, odd_pat))
    assignments.sort(key=lambda item: item[0])
    assert assignments[0][0] == 0, "non-anomaly cells must fit one checkerboard assignment exactly"
    assert assignments[0][0] < assignments[1][0], "checkerboard parity assignment must be unique"
    _, even_pat, odd_pat = assignments[0]

    anomaly_arr = patterns[anomaly_pat]
    vals, cnts = np.unique(anomaly_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = anomaly_arr == active_color
    mh, mw = mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == anomaly_pat

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
    assert len(candidates) == 1, "anomaly mask placement must be uniquely determined"
    top, left = candidates[0]

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            abs_i = top + i
            abs_j = left + j
            if mask[i, j]:
                target_pat = anomaly_arr
            else:
                base_pat = even_pat if (abs_i + abs_j) % 2 == 0 else odd_pat
                target_pat = patterns[base_pat]
            r0 = abs_i * step + 1
            c0 = abs_j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

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

_BASE_SEEDS = [
    np.array([[1, 1, 0], [0, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 0, 1], [0, 0, 1]], dtype=bool),
]


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate valid anomaly masks, forcing inactive cells onto both checkerboard parities."""
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
                if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
                    continue
                inactive_parities = {(r + c) % 2 for r, c in np.argwhere(~mask)}
                if len(inactive_parities) != 2:
                    continue
                masks.append(mask)
        by_count[active_count] = masks
    return by_count


def _precompute_base_masks() -> list[np.ndarray]:
    """Build a small catalog of asymmetric base motifs for the checkerboard tiling."""
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


_MASKS_BY_COUNT = _precompute_masks()
_BASE_MASKS = _precompute_base_masks()


def _random_minority_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _random_checkerboard_pair(shared: int, base_exclusive: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(50):
        mask_a, mask_b = random.sample(_BASE_MASKS, 2)
        if np.count_nonzero(mask_a != mask_b) >= 3:
            arr_a = np.where(mask_a, base_exclusive, shared).astype(int)
            arr_b = np.where(mask_b, base_exclusive, shared).astype(int)
            return arr_a, arr_b
    raise RuntimeError("failed to choose distinct checkerboard motifs")


def _placement_candidates(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
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


def _subset_spans_both_parities(subset: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    parities = {(top + r + left + c) % 2 for r, c in subset}
    return len(parities) == 2


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    active = [tuple(coord) for coord in np.argwhere(mask)]
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
    anomaly_pat: np.ndarray,
    active_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            base_pat = even_pat if (i + j) % 2 == 0 else odd_pat
            paste(grid, base_pat, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, anomaly_pat, *_cell_origin(i, j))
    return grid


def _generate_case():
    separator, base_exclusive, shared, anomaly_exclusive = random.sample(range(COLOR_POOL), 4)
    even_pat, odd_pat = _random_checkerboard_pair(shared, base_exclusive)
    mask = _random_minority_mask()
    anomaly_pat = np.where(mask, anomaly_exclusive, shared).astype(int)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    out = _build_grid(separator, even_pat, odd_pat, anomaly_pat, active_cells)

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, even_pat, odd_pat, anomaly_pat, given_cells)

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
        "active_cells": active_cells,
        "given_cells": given_cells,
    }


def generate_input():
    for _ in range(MAX_RETRIES):
        case = _generate_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid b99e7126 D6 input")
