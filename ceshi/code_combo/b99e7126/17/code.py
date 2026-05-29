# Combo variant of b99e7126 (index 17)
# combo_spec: D6/0+D12/1
# Name: 行间镜像对称 + 棋盘式双图案

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the fused D6/0 + D12/1 variant
# a) Concrete behavior and trace:
#    The 29x29 canvas still contains a 7x7 lattice of 3x3 cell interiors separated by 1-pixel lines.
#    The old task had one repeated background tile M. D6 replaced that with a checkerboard A/B.
#    This fused variant goes one step further: even rows use A/B/A/B..., while odd rows use
#    fliplr(A)/fliplr(B)/fliplr(A)/fliplr(B)... at the same column parities. So the background is a
#    2x2 periodic structure with four literal tiles: A, B, A', B', where A' = fliplr(A), B' = fliplr(B).
#    Example trace:
#    - Old task: a False mask pixel inside the macro block always restored one constant majority tile.
#    - D6 only: the same False pixel restored A or B from cell parity.
#    - Fused task: the False pixel restores A/B on even rows but A'/B' on odd rows, so restoring the
#      block requires both column alternation and row-dependent mirroring.
# b) Complexity calibration:
#    Keep the original geometry, mask size, and output-first generation so only the background logic changes.
#    Use exactly one anomaly tile plus four repeated background tiles. The background is still regular and
#    strongly repeated, but not so simple that a pure checkerboard solver can succeed.
#    Generator constrains the chosen block so its inactive cells cover both absolute row parities and both
#    column parities; this makes the fused rule visible inside the completed area itself.
# c) Hidden-assumption audit against the original solver:
#    - `majority_pat = max(...)` no longer works because the background is not one literal pattern.
#    - D6's `even_pat/odd_pat` parity checker also fails because odd rows are not copies of even rows.
#    - `existing = (cell == minority_pat)` still works only because the anomaly remains one literal tile
#      that is rarer than each background tile; generator guarantees 3-4 anomaly cells versus 10+ repeats
#      of every background tile.
#    - Output synthesis cannot write a single fallback tile or even a row-agnostic checkerboard tile; it
#      must restore the exact 2x2 background role at each inactive cell.
# d) Placement legality:
#    The anomaly mask is still 3x3 and is placed with top/left offsets in 0..4, so it always fits in the
#    7x7 cell lattice. Cells never overlap because each occupies one fixed 3x3 interior. Generator builds
#    the full output first, then removes some active anomaly cells to form the input, and finally verifies
#    `main(input) == output`.
# e) Visual observability:
#    A and B are chosen as asymmetric motifs whose horizontal mirrors are different from themselves and from
#    each other, so a human can see that odd rows are mirrored versions rather than arbitrary new noise.
#    The anomaly tile still uses a dense active color (6-7 of 9 pixels), so its internal mask remains easy
#    to read as the macro-level blueprint.
# f) Visual cleanliness:
#    The color budget stays at four colors: separator, shared background/anomaly color, background-exclusive,
#    and anomaly-exclusive. Each cell interior contains exactly one of five clean tiles (A, B, A', B', C),
#    with no overlays or auxiliary markers.
# g) Change inferability:
#    Color identities are not used for ordering or branching. The fused rule is supported by repeated spatial
#    evidence: the same two even-row motifs repeat across multiple columns, and their mirrored odd-row forms
#    repeat across multiple rows. This gives enough structure for humans to infer both the checkerboard and
#    the row-mirror relationship from a few examples.
# New solver-assumption review:
#    - Solver assumes the four background literals can be explained by two even-row anchors plus fliplr on
#      odd rows; generator explicitly samples only from that family.
#    - Solver assumes the four background literals are all distinct; generator rejects pairs where A, B, A',
#      or B' collide.
#    - Solver assumes anomaly cells are the unique rarest literal pattern and that their subset uniquely pins
#      down the 3x3 placement; generator enforces both and self-checks every sample.


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


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    return tuple(np.asarray(arr, dtype=int).ravel().tolist())


def _extract_cell_patterns(
    g: np.ndarray, step: int
) -> tuple[int, int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
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
            pat = _pattern_key(pat_arr)
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return n, inner, patterns, cell_pat


def _mirror_key(patterns: dict[tuple[int, ...], np.ndarray], pat: tuple[int, ...]) -> tuple[int, ...]:
    return _pattern_key(np.fliplr(patterns[pat]))


def _expected_background_pat(
    even_col0: tuple[int, ...],
    even_col1: tuple[int, ...],
    odd_col0: tuple[int, ...],
    odd_col1: tuple[int, ...],
    i: int,
    j: int,
) -> tuple[int, ...]:
    if i % 2 == 0:
        return even_col0 if j % 2 == 0 else even_col1
    return odd_col0 if j % 2 == 0 else odd_col1


def _background_mismatch_score(
    cell_pat: list[list[tuple[int, ...]]],
    anomaly_pat: tuple[int, ...],
    even_col0: tuple[int, ...],
    even_col1: tuple[int, ...],
    odd_col0: tuple[int, ...],
    odd_col1: tuple[int, ...],
) -> int:
    mismatches = 0
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat == anomaly_pat:
                continue
            expected = _expected_background_pat(even_col0, even_col1, odd_col0, odd_col1, i, j)
            mismatches += int(pat != expected)
    return mismatches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the fractal anomaly overlay on a checkerboard whose odd rows are mirrored copies.

    Input structure:
    - The grid is partitioned into equal 3x3 cell interiors by 1-pixel separator lines.
    - Even rows alternate between two base tiles, while odd rows alternate between the horizontal mirrors
      of those same two tiles.
    - A few cells instead show one rarer anomaly tile; these are the visible samples to be completed.

    Rule:
    - Infer the two even-row base tiles and their odd-row mirrored counterparts.
    - Read the anomaly tile's dominant-color pixels as a 3x3 mask and place that mask on the cell lattice
      so the already-shown anomaly cells land on active positions.
    - Fill every remaining active position with the anomaly tile and restore every inactive position with
      the exact background tile demanded by both checkerboard parity and row-mirror status.

    Key insight: The rare tile is still a self-similar blueprint, but the background beneath it is not a
    single pattern or a simple checkerboard; it is a checkerboard whose odd rows are horizontally mirrored,
    so solving requires reconstructing that fused 2D structure first.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n, inner, patterns, cell_pat = _extract_cell_patterns(g, step)

    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1

    assert len(counts) == 5, "variant expects 4 background literals plus 1 anomaly literal"
    anomaly_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    base_pats = [pat for pat in counts if pat != anomaly_pat]
    base_set = set(base_pats)
    assert len(base_pats) == 4, "need exactly four background literals"

    assignments: list[tuple[int, tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    for even_col0 in base_pats:
        odd_col0 = _mirror_key(patterns, even_col0)
        if odd_col0 not in base_set:
            continue
        for even_col1 in base_pats:
            if even_col1 == even_col0:
                continue
            odd_col1 = _mirror_key(patterns, even_col1)
            if odd_col1 not in base_set:
                continue
            if len({even_col0, even_col1, odd_col0, odd_col1}) != 4:
                continue
            score = _background_mismatch_score(
                cell_pat, anomaly_pat, even_col0, even_col1, odd_col0, odd_col1
            )
            assignments.append((score, even_col0, even_col1, odd_col0, odd_col1))

    assert assignments, "failed to find any mirror-checkerboard interpretation"
    assignments.sort(key=lambda item: item[0])
    assert assignments[0][0] == 0, "background must fit the fused mirror-checkerboard exactly"
    assert len(assignments) == 1 or assignments[0][0] < assignments[1][0], "background assignment must be unique"
    _, even_col0, even_col1, odd_col0, odd_col1 = assignments[0]

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
    for i in range(n):
        for j in range(n):
            inside = top <= i < top + mh and left <= j < left + mw
            if inside and mask[i - top, j - left]:
                target = anomaly_arr
            else:
                target_key = _expected_background_pat(even_col0, even_col1, odd_col0, odd_col1, i, j)
                target = patterns[target_key]
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target
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
MAX_RETRIES = 500

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
    np.array([[1, 1, 0], [1, 0, 0], [0, 1, 0]], dtype=bool),
]


def _precompute_masks() -> dict[int, list[np.ndarray]]:
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
                masks.append(mask)
        by_count[active_count] = masks
    return by_count


def _precompute_base_masks() -> list[np.ndarray]:
    seen: set[tuple[int, ...]] = set()
    catalog: list[np.ndarray] = []
    for seed in _BASE_SEEDS:
        variants = [
            seed,
            np.rot90(seed, 1),
            np.rot90(seed, 2),
            np.rot90(seed, 3),
            np.flipud(seed),
            np.fliplr(seed),
        ]
        for variant in variants:
            key = tuple(int(v) for v in variant.ravel())
            if key in seen:
                continue
            if np.array_equal(variant, np.fliplr(variant)):
                continue
            seen.add(key)
            catalog.append(variant.copy())
    return catalog


_MASKS_BY_COUNT = _precompute_masks()
_BASE_MASKS = _precompute_base_masks()


def _random_minority_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _pairwise_min_distance(arrs: list[np.ndarray]) -> int:
    best = 9
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            best = min(best, int(np.count_nonzero(arrs[i] != arrs[j])))
    return best


def _random_row_mirror_pair(shared: int, base_exclusive: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        mask_a, mask_b = random.sample(_BASE_MASKS, 2)
        arr_a = np.where(mask_a, base_exclusive, shared).astype(int)
        arr_b = np.where(mask_b, base_exclusive, shared).astype(int)
        arrs = [arr_a, arr_b, np.fliplr(arr_a), np.fliplr(arr_b)]
        keys = {_pattern_key(arr) for arr in arrs}
        if len(keys) != 4:
            continue
        if _pairwise_min_distance(arrs) < 2:
            continue
        return arr_a, arr_b
    raise RuntimeError("failed to choose distinct row-mirror checkerboard tiles")


def _placement_candidates(
    mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int
) -> list[tuple[int, int]]:
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


def _inactive_shows_fusion(mask: np.ndarray, top: int, left: int) -> bool:
    inactive = [(top + r, left + c) for r, c in np.argwhere(~mask)]
    row_parities = {r % 2 for r, _ in inactive}
    col_parities = {c % 2 for _, c in inactive}
    return len(row_parities) == 2 and len(col_parities) == 2


def _subset_diversity_ok(subset: tuple[tuple[int, int], ...], top: int, left: int) -> bool:
    rows = {(top + r) % 2 for r, _ in subset}
    cols = {(left + c) % 2 for _, c in subset}
    return len(rows) == 2 and len(cols) == 2


def _choose_given_subset(mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    active = [tuple(coord) for coord in np.argwhere(mask)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        diversified = []
        fallback = []
        for subset in combinations(active, size):
            if _placement_candidates(mask, subset, top, left) != [(top, left)]:
                continue
            if _subset_diversity_ok(subset, top, left):
                diversified.append(subset)
            else:
                fallback.append(subset)
        if diversified:
            return random.choice(diversified)
        if fallback:
            return random.choice(fallback)
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _expected_background(even_col0: np.ndarray, even_col1: np.ndarray, i: int, j: int) -> np.ndarray:
    if i % 2 == 0:
        return even_col0 if j % 2 == 0 else even_col1
    return np.fliplr(even_col0) if j % 2 == 0 else np.fliplr(even_col1)


def _build_grid(
    separator: int,
    even_col0: np.ndarray,
    even_col1: np.ndarray,
    anomaly_pat: np.ndarray,
    active_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, _expected_background(even_col0, even_col1, i, j), *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, anomaly_pat, *_cell_origin(i, j))
    return grid


def _fusion_sensitive_inactive_count(mask: np.ndarray, top: int) -> int:
    return sum(1 for r, _ in np.argwhere(~mask) if (top + int(r)) % 2 == 1)


def _generate_case():
    separator, base_exclusive, shared, anomaly_exclusive = random.sample(range(COLOR_POOL), 4)
    even_col0, even_col1 = _random_row_mirror_pair(shared, base_exclusive)
    mask = _random_minority_mask()
    anomaly_pat = np.where(mask, anomaly_exclusive, shared).astype(int)

    placements = [(t, l) for t in range(PLACEMENT_RANGE) for l in range(PLACEMENT_RANGE) if _inactive_shows_fusion(mask, t, l)]
    if not placements:
        return None
    top, left = random.choice(placements)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    out = _build_grid(separator, even_col0, even_col1, anomaly_pat, active_cells)

    subset = _choose_given_subset(mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, even_col0, even_col1, anomaly_pat, given_cells)

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
        "fusion_sensitive_inactive_count": _fusion_sensitive_inactive_count(mask, top),
    }


def generate_input():
    for _ in range(MAX_RETRIES):
        case = _generate_case()
        if case is not None:
            return case["input"]
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
