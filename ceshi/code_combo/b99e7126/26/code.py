# Combo variant of b99e7126 (index 26)
# combo_spec: D3/2+D7/1
# Name: 锚点标记 + 多个相同异常

from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[5] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import paste


# Design analysis
# ---------------
# a) Concreteization and behavior trace
#    - The board is still a 7x7 lattice of 3x3 cell interiors with 1-pixel divider lines.
#    - One majority cell pattern fills almost every cell.
#    - A repeated minority cell pattern now appears several times inside one source 3x3 coarse block.
#      Those repeated minority cells are identical; their relative cell positions, not their internal
#      pixels, form the coarse blueprint to be copied.
#    - A separate marker cell looks like the majority pattern except for one special center pixel.
#      That marker does not contribute to the blueprint; it only chooses the destination anchor.
#    - Small trace:
#      original seed logic: read one minority cell's internal dominant-color mask -> infer where the
#      enlarged 3x3 block belongs from observed minority cells -> complete that same block.
#      fused logic: first collect all identical minority cells and recover their 3x3 coarse mask from
#      their bounding box -> then ignore their absolute position and use the marker as the destination
#      center -> stamp one new copy of that coarse mask there.
#      difference: D7 provides the mask content at cell-grid level, while D3 provides the placement.
#
# b) Complexity calibration
#    - Only one new special object family is added beyond the seed task: a single marker cell that is
#      visually "majority plus one center dot".
#    - The repeated minority cells are constrained to 4-6 copies inside one 3x3 coarse frame with an
#      active center and support on all four edges. That makes the source shape visible without
#      exploding the number of possibilities.
#    - The minority cell's own 3x3 interior is still a clean two-color pattern, but the generator
#      forces its internal dominant-color mask to differ from the source-cell distribution. This
#      prevents the combo from collapsing back to the D3-only rule.
#
# c) Implicit-assumption audit of the original solver
#    - `minority_pat = min(counts...)` no longer works because the least frequent pattern is the
#      marker, not the repeated anomaly.
#    - `existing[i, j] = cell_pat[i][j] == minority_pat` used to mean "these observed cells are also
#      the destination clues". In the combo they instead describe a source blueprint that must be
#      copied elsewhere.
#    - Deriving the mask from `minority_arr` is no longer valid. The combo solver must derive the
#      coarse mask from the repeated minority cells' coordinates.
#    - The old placement search over `(top, left)` based on anomaly positions is obsolete because the
#      marker determines destination directly.
#    - New solver assumptions:
#      * exactly one marker pattern differs from the majority by one pixel and appears once;
#      * exactly one minority pattern appears multiple times;
#      * the minority-cell bounding box is 3x3 at cell-grid scale and its center is occupied.
#      The generator explicitly enforces all three.
#
# d) Placement legality
#    - The source minority cluster always fits in a 3x3 coarse block, and the marker always sits at
#      the center of another valid 3x3 coarse block.
#    - Source and destination 3x3 coarse blocks are generated disjointly, so the copied block never
#      collides with the source blueprint and the marker stays easy to read in the input.
#
# e) Visual observability
#    - A human can see two distinct roles in the input: one cluster of several identical odd cells,
#      and one single majority-like cell with a center dot.
#    - The output keeps the original cluster and grows exactly one second copy at the marker-defined
#      location. No hidden use of color ids or ordering is required.
#
# f) Visual cleanliness
#    - All minority cells share the same appearance, so the source cluster reads as one object class.
#    - The marker is only one pixel away from the majority template, which keeps the board tidy while
#      still being clearly separate from the full minority cells.
#    - Disjoint source/destination blocks avoid stacked overwrites that would make the causal roles
#      ambiguous.
#
# g) Change inferability
#    - Only five colors are used: divider, majority-exclusive, shared, minority-exclusive, and one
#      marker-exclusive color.
#    - The repeated minority cells structurally support the D7 part, and the displaced marker block
#      structurally supports the D3 part. The generator also checks that using the minority cell's
#      own interior mask at the marker would produce a different output, proving the source-cluster
#      distribution is genuinely necessary.


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
    g: np.ndarray,
) -> tuple[int, int, int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]], dict[tuple[int, ...], int]]:
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
            pat = tuple(int(x) for x in pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            counts[pat] = counts.get(pat, 0) + 1
            row.append(pat)
        cell_pat.append(row)
    return step, n, inner, patterns, cell_pat, counts


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _find_pattern_cells(cell_pat: list[list[tuple[int, ...]]], target: tuple[int, ...]) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat == target:
                hits.append((int(i), int(j)))
    return hits


def _dominant_mask(arr: np.ndarray) -> tuple[np.ndarray, int]:
    vals, cnts = np.unique(arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    return arr == active_color, active_color


def _source_mask_from_cells(cells: list[tuple[int, int]]) -> tuple[int, int, np.ndarray]:
    assert cells, "need at least one minority cell"
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    top = min(rows)
    left = min(cols)
    bottom = max(rows)
    right = max(cols)
    mh = bottom - top + 1
    mw = right - left + 1
    assert mh == mw == MASK_SIZE, "minority-cell cluster must occupy one 3x3 coarse frame"

    mask = np.zeros((mh, mw), dtype=bool)
    for r, c in cells:
        mask[r - top, c - left] = True
    anchor_r = mh // 2
    anchor_c = mw // 2
    assert mask[anchor_r, anchor_c], "source blueprint must include its center cell"
    return int(top), int(left), mask


def _render_block(base: np.ndarray, majority: np.ndarray, minority: np.ndarray, mask: np.ndarray, top: int, left: int) -> np.ndarray:
    out = base.copy()
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            paste(out, minority if mask[i, j] else majority, *_cell_origin(top + i, left + j))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy one repeated anomaly cluster to a marker-chosen anchor on the cell lattice.

    Input structure:
    - The board is a regular NxN array of 3x3 cell interiors separated by one-color divider lines.
    - Most cells share one majority interior pattern.
    - Several cells share one identical minority interior pattern and together occupy part of a 3x3
      coarse block.
    - One extra marker cell looks like the majority pattern except for a single special center pixel.

    Rule:
    - Treat the repeated minority cells' coarse positions as the true 3x3 blueprint: their bounding
      box defines the local frame and the occupied cell coordinates define the active mask.
    - Keep the minority cell's own interior as the stamp to place at active positions, and keep the
      majority interior at inactive positions.
    - Ignore the source cluster's absolute location when choosing the destination. Instead, use the
      marker cell as the center anchor of a new 3x3 coarse block.
    - Overwrite that marker-anchored 3x3 block once according to the recovered coarse mask.

    Key insight:
    - The repeated anomaly cells and the marker have different jobs. The anomaly cells say which
      coarse positions are active, while the marker alone says where that coarse pattern must be
      copied.
    """
    g = np.asarray(grid, dtype=int)
    step, n, inner, patterns, cell_pat, counts = _extract_cell_patterns(g)
    assert len(counts) >= 3, "need majority, minority cluster, and marker"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    marker_candidates: list[tuple[int, ...]] = []
    minority_candidates: list[tuple[int, ...]] = []
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        arr = patterns[pat]
        dist = _hamming_distance(arr, majority_arr)
        if cnt == 1 and dist == 1:
            marker_candidates.append(pat)
        else:
            minority_candidates.append(pat)

    assert len(marker_candidates) == 1, "must have exactly one marker pattern"
    assert len(minority_candidates) == 1, "must have exactly one repeated minority pattern"

    marker_pat = marker_candidates[0]
    minority_pat = minority_candidates[0]
    assert counts[minority_pat] >= 2, "minority pattern must repeat"
    minority_arr = patterns[minority_pat]

    marker_cells = _find_pattern_cells(cell_pat, marker_pat)
    minority_cells = _find_pattern_cells(cell_pat, minority_pat)
    assert len(marker_cells) == 1, "marker must occur exactly once"

    _src_top, _src_left, source_mask = _source_mask_from_cells(minority_cells)
    assert source_mask.shape == (inner, inner), "coarse blueprint must match the 3x3 cell interior size"

    marker_i, marker_j = marker_cells[0]
    anchor_r = source_mask.shape[0] // 2
    anchor_c = source_mask.shape[1] // 2
    top = marker_i - anchor_r
    left = marker_j - anchor_c
    assert 0 <= top <= n - source_mask.shape[0] and 0 <= left <= n - source_mask.shape[1], "marker anchor out of bounds"

    out = g.copy()
    for i in range(source_mask.shape[0]):
        for j in range(source_mask.shape[1]):
            target_pat = minority_arr if source_mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (combo D3/2 + D7/1) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ANCHOR_REL = (1, 1)

SOURCE_ACTIVE_COUNTS = (4, 5, 6)
STAMP_ACTIVE_COUNTS = (5, 6, 7)
COLOR_POOL = 10
MAX_RETRIES = 500
MIN_MINORITY_DISTANCE = 3

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]


def _precompute_masks(active_counts: tuple[int, ...]) -> dict[int, list[np.ndarray]]:
    """Enumerate LR-symmetric 3x3 masks with active center and all-edge support."""
    weights = [len(group) for group in _LR_GROUPS]
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in active_counts:
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
                if not mask[ANCHOR_REL]:
                    continue
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_SOURCE_MASKS_BY_COUNT = _precompute_masks(SOURCE_ACTIVE_COUNTS)
_STAMP_MASKS_BY_COUNT = _precompute_masks(STAMP_ACTIVE_COUNTS)

_SOURCE_MASKS = [mask for masks in _SOURCE_MASKS_BY_COUNT.values() for mask in masks]
_STAMP_MASKS = [mask for masks in _STAMP_MASKS_BY_COUNT.values() for mask in masks]

_DISJOINT_BLOCK_PAIRS = [
    ((top1, left1), (top2, left2))
    for top1 in range(PLACEMENT_RANGE)
    for left1 in range(PLACEMENT_RANGE)
    for top2 in range(PLACEMENT_RANGE)
    for left2 in range(PLACEMENT_RANGE)
    if max(top1, top2) > min(top1, top2) + MASK_SIZE - 1 or max(left1, left2) > min(left1, left2) + MASK_SIZE - 1
]


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _choose_distinct_masks() -> tuple[np.ndarray, np.ndarray]:
    source_mask = random.choice(_SOURCE_MASKS)
    choices = [mask for mask in _STAMP_MASKS if not np.array_equal(mask, source_mask)]
    assert choices, "need at least one stamp mask different from the source mask"
    stamp_mask = random.choice(choices)
    return source_mask, stamp_mask


def _build_grid(
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    source_cells: list[tuple[int, int]],
    marker_cell: tuple[int, int] | None = None,
    marker_pattern: np.ndarray | None = None,
    dst_cells: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in source_cells:
        paste(grid, minority, *_cell_origin(i, j))
    if dst_cells is not None:
        for i, j in dst_cells:
            paste(grid, minority, *_cell_origin(i, j))
    if marker_cell is not None and marker_pattern is not None:
        paste(grid, marker_pattern, *_cell_origin(*marker_cell))
    return grid


def _anchor_only_output(inp: np.ndarray) -> np.ndarray:
    """Counterfactual D3-only output: use the minority cell's own dominant-color mask at the marker."""
    g = np.asarray(inp, dtype=int)
    _step, _n, _inner, patterns, cell_pat, counts = _extract_cell_patterns(g)
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    marker_pat = None
    minority_pat = None
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        arr = patterns[pat]
        if cnt == 1 and _hamming_distance(arr, majority_arr) == 1:
            marker_pat = pat
        else:
            minority_pat = pat
    assert marker_pat is not None and minority_pat is not None

    minority_arr = patterns[minority_pat]
    mask, _ = _dominant_mask(minority_arr)
    marker_i, marker_j = _find_pattern_cells(cell_pat, marker_pat)[0]
    top = marker_i - 1
    left = marker_j - 1
    return _render_block(g, majority_arr, minority_arr, mask, top, left)


def _analyze_variant_input(grid: np.ndarray) -> dict[str, int]:
    """Quantify that both the source distribution and the marker matter."""
    g = np.asarray(grid, dtype=int)
    _step, _n, _inner, patterns, cell_pat, counts = _extract_cell_patterns(g)

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    marker_pat = None
    minority_pat = None
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        arr = patterns[pat]
        if cnt == 1 and _hamming_distance(arr, majority_arr) == 1:
            marker_pat = pat
        else:
            minority_pat = pat
    assert marker_pat is not None and minority_pat is not None

    minority_arr = patterns[minority_pat]
    minority_cells = _find_pattern_cells(cell_pat, minority_pat)
    src_top, src_left, source_mask = _source_mask_from_cells(minority_cells)
    internal_mask, _ = _dominant_mask(minority_arr)
    marker_i, marker_j = _find_pattern_cells(cell_pat, marker_pat)[0]
    dst_top = marker_i - 1
    dst_left = marker_j - 1

    true_out = main(g)
    anchor_only = _anchor_only_output(g)
    return {
        "minority_count": int(len(minority_cells)),
        "source_vs_internal_mask_diff": int(np.count_nonzero(source_mask != internal_mask)),
        "source_dest_block_manhattan": int(abs(src_top - dst_top) + abs(src_left - dst_left)),
        "anchor_only_output_diff": int(np.count_nonzero(true_out != anchor_only)),
    }


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, min_exclusive, marker_exclusive = random.sample(range(COLOR_POOL), 5)
    majority = _random_lr_pattern(shared, maj_exclusive)
    source_mask, stamp_mask = _choose_distinct_masks()
    minority = np.where(stamp_mask, min_exclusive, shared).astype(int)
    if _hamming_distance(minority, majority) < MIN_MINORITY_DISTANCE:
        return None

    marker_pattern = majority.copy()
    marker_pattern[ANCHOR_REL] = marker_exclusive
    if np.array_equal(marker_pattern, minority):
        return None

    (src_top, src_left), (dst_top, dst_left) = random.choice(_DISJOINT_BLOCK_PAIRS)
    source_cells = [(src_top + int(r), src_left + int(c)) for r, c in np.argwhere(source_mask)]
    dst_cells = [(dst_top + int(r), dst_left + int(c)) for r, c in np.argwhere(source_mask)]
    marker_cell = (dst_top + ANCHOR_REL[0], dst_left + ANCHOR_REL[1])

    out = _build_grid(separator, majority, minority, source_cells, dst_cells=dst_cells)
    inp = _build_grid(separator, majority, minority, source_cells, marker_cell=marker_cell, marker_pattern=marker_pattern)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    metrics = _analyze_variant_input(inp)
    if metrics["source_vs_internal_mask_diff"] <= 0:
        return None
    if metrics["source_dest_block_manhattan"] <= 0:
        return None
    if metrics["anchor_only_output_diff"] <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo D3/2+D7/1 input")
