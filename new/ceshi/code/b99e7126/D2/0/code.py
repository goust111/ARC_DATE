# Variant of b99e7126
# Dimension: D2 — 源对象：异常单元格的内部图案结构 (src)
# Change: 唯一异常的3x3单元格的内部颜色像素图案，包含两种颜色（前景色和背景色）... → 多个异常单元格共同构成源图案

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis
# spec:
#   concreteization:
#     - The canvas is still a 7x7 grid of 3x3 cell interiors separated by 1-pixel dividers.
#     - One majority cell pattern fills almost every cell.
#     - Input now contains 2 or 3 exceptional cells, each using the same shared color plus the
#       same anomaly-only color. Each exceptional cell is a strict fragment of the final 3x3
#       source mask, so no single cell reveals the whole blueprint.
#     - The output still paints one translated 3x3 block on the cell grid, but every active
#       position in that block receives the fully combined anomaly pattern.
#   behavior_trace:
#     - Original: pick the unique minority cell, read its active pixels directly, then scale them
#       to cell level.
#     - Variant: gather all non-majority cells, treat their anomaly-colored pixels as partial
#       masks in the same 3x3 coordinate frame, OR them together, then use that union as the
#       scaled blueprint.
#     - Example: if one fragment shows the top row and another shows the middle-left plus bottom
#       row, the solver must combine both before searching for the translated placement.
#   complexity_calibration:
#     - Fragment count is fixed to 2 or 3, not arbitrary.
#     - Final active count stays at 6 or 7, matching the original visual density.
#     - Each fragment contributes 2-4 active pixels so every fragment is visible and the union is
#       strictly richer than any single fragment.
#   implicit_assumption_audit:
#     - `min(counts)` was tied to a single minority pattern; with multiple singleton fragments that
#       becomes invalid, so the solver now takes the majority pattern plus every non-majority cell.
#     - "most frequent color inside the anomaly cell" used to reveal the active color; sparse
#       fragments break that because active pixels can be fewer than inactive ones. The new solver
#       instead finds the color present in anomaly cells but absent from the majority cell.
#     - Equality checks against one `minority_pat` used to locate existing anomaly cells; now any
#       non-majority cell marks an existing active location.
#     - The output writer used the observed minority pattern directly; now it must synthesize a new
#       full pattern that may not appear anywhere in the input.
#   placement_legality:
#     - Fragment cells are placed only on true active positions of the final translated mask.
#     - The translated 3x3 mask always fits the 7x7 cell lattice, and the generator rejects any
#       sample whose shown fragment positions do not determine a unique placement.
#   visual_observability:
#     - The difference from the original is visible immediately: several distinct anomaly cells
#       appear, and the output repeats a richer pattern than any one input anomaly cell.
#     - The combination rule is a direct pixelwise union in the shared 3x3 frame, so examples do
#       not require hidden ordering, color ranking, or case splits.
#   visual_cleanliness:
#     - Only one region of the board is modified, as in the seed task.
#     - Combination happens inside the tiny cell interiors, so the global scene remains a clean
#       repeated lattice with one highlighted translated block.
#   color_inference:
#     - Colors still have structural roles only: divider, majority-exclusive, shared, and
#       anomaly-exclusive.
#     - No color-id ordering is used; the anomaly-exclusive color is inferred from set difference
#       against the majority cell palette.


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
    return patterns, cell_pat


def _compose_source_from_fragments(
    patterns: dict[tuple[int, ...], np.ndarray], cell_pat: list[list[tuple[int, ...]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for p in row:
            counts[p] = counts.get(p, 0) + 1
    assert len(counts) >= 3, "need a majority pattern plus multiple anomaly fragments"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]
    majority_colors = set(map(int, majority_arr.ravel().tolist()))

    fragment_arrays: list[np.ndarray] = []
    fragment_positions: list[tuple[int, int]] = []
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat != majority_pat:
                fragment_arrays.append(patterns[pat])
                fragment_positions.append((i, j))
    assert 2 <= len(fragment_arrays) <= 3, "variant expects 2 or 3 fragment cells"

    shared_color: int | None = None
    extra_colors = set()
    for arr in fragment_arrays:
        colors = set(map(int, arr.ravel().tolist()))
        shared = colors & majority_colors
        extra = colors - majority_colors
        assert len(shared) == 1 and len(extra) == 1, "each fragment must share one color and add one anomaly color"
        if shared_color is None:
            shared_color = next(iter(shared))
        else:
            assert shared == {shared_color}, "all fragments must share the same background color"
        extra_colors |= extra
    assert shared_color is not None and len(extra_colors) == 1, "fragments must agree on the anomaly color"
    active_color = next(iter(extra_colors))

    union_mask = np.zeros_like(majority_arr, dtype=bool)
    for arr in fragment_arrays:
        frag_mask = arr == active_color
        assert np.any(frag_mask), "every fragment must contribute active pixels"
        union_mask |= frag_mask
    assert np.any(union_mask), "combined source mask cannot be empty"
    assert all(not np.array_equal(arr == active_color, union_mask) for arr in fragment_arrays), "each fragment must be partial"

    full_pattern = np.where(union_mask, active_color, shared_color).astype(int)
    return majority_arr, full_pattern, union_mask, fragment_positions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine several fragment cells into one source pattern, then scale that union to the cell grid.

    Input structure:
    - The board is partitioned into equal cells by uniform divider lines.
    - Nearly every cell shows the same majority interior pattern.
    - Exactly 2 or 3 exceptional cells use a shared background color plus one anomaly-only color;
      each of those cells displays only part of the final 3x3 source pattern.

    Rule:
    - Collect every non-majority cell and overlay their anomaly-colored pixels inside the common
      3x3 coordinate frame to reconstruct the full source pattern.
    - Treat the reconstructed 3x3 mask as a cell-scale stencil: active pixels become the completed
      anomaly pattern and inactive pixels remain the majority pattern.
    - Translate that stencil so the shown exceptional cells sit on active positions, then fill in
      the rest of the translated active positions with the completed anomaly pattern.

    Key insight: No single anomaly cell is the blueprint anymore; the blueprint is the pixelwise
    union of all anomaly fragments, while the fragment locations still reveal where the scaled block
    belongs on the larger cell lattice.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _extract_cell_patterns(g, step)
    majority_arr, full_pattern, mask, fragment_positions = _compose_source_from_fragments(patterns, cell_pat)
    assert mask.shape == (inner, inner), "source pattern must match cell interior size"

    existing = np.zeros((n, n), dtype=bool)
    for i, j in fragment_positions:
        existing[i, j] = True

    candidates: list[tuple[int, int]] = []
    for top in range(0, n - inner + 1):
        for left in range(0, n - inner + 1):
            block = np.zeros_like(existing)
            block[top : top + inner, left : left + inner] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + inner, left : left + inner]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "fragment cells must determine a unique placement"
    top, left = candidates[0]

    out = g.copy()
    for i in range(inner):
        for j in range(inner):
            target_pat = full_pattern if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
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
FRAGMENT_COUNTS = (2, 3)
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


def _size_partitions(total: int, parts: int, lo: int = 2, hi: int = 4) -> list[tuple[int, ...]]:
    if parts == 1:
        return [(total,)] if lo <= total <= hi else []
    out: list[tuple[int, ...]] = []
    for first in range(lo, hi + 1):
        for rest in _size_partitions(total - first, parts - 1, lo, hi):
            out.append((first,) + rest)
    return out


_MASKS_BY_COUNT = _precompute_masks()
_FRAGMENT_SIZE_OPTIONS = {
    (active_count, fragment_count): _size_partitions(active_count, fragment_count)
    for active_count in ACTIVE_PIXEL_COUNTS
    for fragment_count in FRAGMENT_COUNTS
}


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a random LR-symmetric 3x3 pattern using two colors."""
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_minority_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _split_mask_into_fragments(mask: np.ndarray, fragment_count: int) -> list[np.ndarray] | None:
    active = [tuple(coord) for coord in np.argwhere(mask)]
    options = _FRAGMENT_SIZE_OPTIONS[(len(active), fragment_count)]
    if not options:
        return None
    sizes = random.choice(options)
    random.shuffle(active)
    fragments: list[np.ndarray] = []
    start = 0
    for size in sizes:
        frag = np.zeros_like(mask, dtype=bool)
        for r, c in active[start : start + size]:
            frag[r, c] = True
        fragments.append(frag)
        start += size
    random.shuffle(fragments)
    return fragments


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


def _choose_fragment_positions(mask: np.ndarray, top: int, left: int, fragment_count: int) -> tuple[tuple[int, int], ...] | None:
    active = [tuple(coord) for coord in np.argwhere(mask)]
    valid = [
        subset
        for subset in combinations(active, fragment_count)
        if _placement_candidates(mask, subset, top, left) == [(top, left)]
    ]
    if not valid:
        return None
    return random.choice(valid)


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_output_grid(separator: int, majority: np.ndarray, full_pattern: np.ndarray, active_cells: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, full_pattern, *_cell_origin(i, j))
    return grid


def _build_input_grid(
    separator: int, majority: np.ndarray, fragment_cells: list[tuple[int, int, np.ndarray]]
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j, fragment in fragment_cells:
        paste(grid, fragment, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    full_pattern = np.where(mask, min_exclusive, shared).astype(int)

    fragment_count = random.choice(FRAGMENT_COUNTS)
    fragments = _split_mask_into_fragments(mask, fragment_count)
    if fragments is None:
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(mask)]
    out = _build_output_grid(separator, majority, full_pattern, active_cells)

    shown_positions = _choose_fragment_positions(mask, top, left, fragment_count)
    if shown_positions is None:
        return None

    fragment_cells = []
    for (r, c), frag_mask in zip(shown_positions, fragments):
        frag_pattern = np.where(frag_mask, min_exclusive, shared).astype(int)
        fragment_cells.append((top + r, left + c, frag_pattern))
    inp = _build_input_grid(separator, majority, fragment_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 D2 input")
