# Combo variant of b99e7126 (index 45)
# combo_spec: D2/0+D10/2
# Name: 多异常组合源图案 + 顺序处理

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis
# spec:
#   concreteization:
#     - The board stays a 7x7 lattice of 3x3 cell interiors separated by 1-pixel divider lines.
#     - One majority cell pattern still fills almost the whole board.
#     - There are now two visible anomaly roles:
#       1. Source fragments: 2 or 3 singleton anomaly cells. Their interiors use the shared color
#          plus one source-only color, and each cell shows only a disjoint fragment of the final
#          3x3 source mask.
#       2. Stamp markers: 2 or 3 identical anomaly cells. Their repeated marker glyph uses a
#          separate marker-only color and indicates where full 3x3 cell-level overlays start.
#     - Output keeps the fragment cells, but sequentially stamps the reconstructed full source
#       pattern at every marker anchor. These 3x3 stamped regions may overlap.
#   behavior_trace:
#     - Original: find one minority cell, read its whole 3x3 mask, locate one translated 3x3
#       block on the cell grid, and fill it once.
#     - Fused variant:
#       1. Collect all non-majority singleton fragment cells and OR their source-colored pixels to
#          reconstruct the full 3x3 source mask.
#       2. Detect the repeated marker cells and sort their positions from top-left to bottom-right.
#       3. For each marker in that order, stamp the reconstructed 3x3 block onto the cell grid,
#          writing source-pattern cells on mask-true positions and majority cells on mask-false
#          positions.
#       4. Because later stamps rewrite whole cells, overlapping blocks are order-sensitive.
#     - Example: if marker A is above-left of marker B and their 3x3 blocks overlap, a cell first
#       made anomalous by A can be restored to majority by B if B covers that cell at an inactive
#       source-mask coordinate.
#   complexity_calibration:
#     - Fragment count and marker count are both limited to 2 or 3, so the scene is richer than
#       the seed task but still readable from a few examples.
#     - The reconstructed source mask still has 6 or 7 active pixels, so the final stamped shapes
#       remain dense and visually comparable to the original task.
#     - Marker anchors are chosen so at least one overlap occurs and reversing the processing order
#       changes the final board; this prevents the sequential rule from degenerating into an
#       order-irrelevant multi-placement.
#   implicit_assumption_audit:
#     - `min(counts)` no longer identifies a single minority prototype, because fragment cells are
#       distinct singletons and marker cells form a repeated non-majority pattern. The solver must
#       separate "repeated marker pattern" from "singleton fragment patterns".
#     - Reading the active color from one anomaly cell by majority-within-cell frequency breaks for
#       sparse fragments; the solver instead infers the source-only color by palette difference
#       against the majority cell.
#     - The seed task had one placement search; here placement is no longer inferred from fragment
#       positions at all. Marker cell locations directly define multiple stamp anchors, and sorted
#       position provides the processing order.
#     - A one-shot output writer is insufficient because later overlays may intentionally overwrite
#       earlier ones. The solver must replay the ordered stamp sequence exactly.
#   placement_legality:
#     - Every marker is placed in the 0..4 x 0..4 top-left-anchor range so its 3x3 stamp stays
#       inside the 7x7 cell lattice.
#     - Fragment cells are sampled from cells not covered by any stamped 3x3 block, keeping the
#       source bank visually separate from the sequentially modified area.
#     - Generator rejects anchor sets with no overlap or with order-insensitive overlap, and also
#       rejects any sample whose fragment cells fail to stay unique.
#   visual_observability:
#     - A human can separate the two anomaly roles directly: fragments are a few distinct partial
#       blueprints, while markers are identical repeated glyphs in another color.
#     - The output repeatedly expands the same reconstructed pattern at each marker, so the link
#       between marker positions and stamped regions is exposed across examples.
#     - Processing order uses row-major geometry only; no hidden dependence on color ids, counts,
#       or off-screen metadata is required.
#   visual_cleanliness:
#     - Only one repeated marker glyph and one repeated majority pattern are used, avoiding a noisy
#       proliferation of unrelated symbols.
#     - Fragment cells are kept outside all stamped blocks, so the source-composition clue remains
#       intact in the output and does not visually collide with the sequential overlays.
#   color_inference:
#     - Five structural colors are used: divider, majority-exclusive, shared, source-exclusive,
#       and marker-exclusive.
#     - Their roles are discoverable from repetition and local palette structure rather than from
#       numeric color identity.


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
FRAGMENT_COUNTS = (2, 3)
MARKER_COUNTS = (2, 3)
COLOR_POOL = 10
MAX_RETRIES = 600

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]

_MARKER_MASK = np.array(
    [
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0],
    ],
    dtype=bool,
)


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
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def _row_major(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda rc: (rc[0], rc[1]))


def _summarize_scene(
    patterns: dict[tuple[int, ...], np.ndarray], cell_pat: list[list[tuple[int, ...]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]]]:
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]
    majority_colors = set(map(int, majority_arr.ravel().tolist()))

    non_majority = [(pat, cnt) for pat, cnt in counts.items() if pat != majority_pat]
    marker_candidates = [pat for pat, cnt in non_majority if cnt >= 2]
    assert len(marker_candidates) == 1, "need one repeated marker pattern"
    marker_pat = marker_candidates[0]
    assert 2 <= counts[marker_pat] <= 3, "marker count must be 2 or 3"

    fragment_arrays: list[np.ndarray] = []
    shared_color: int | None = None
    active_color: int | None = None
    for pat, cnt in non_majority:
        if pat == marker_pat:
            continue
        assert cnt == 1, "fragment patterns must be singleton anomalies"
        arr = patterns[pat]
        colors = set(map(int, arr.ravel().tolist()))
        shared = colors & majority_colors
        extra = colors - majority_colors
        assert len(shared) == 1 and len(extra) == 1, "each fragment must expose one shared and one source-only color"
        this_shared = next(iter(shared))
        this_active = next(iter(extra))
        if shared_color is None:
            shared_color = this_shared
            active_color = this_active
        else:
            assert this_shared == shared_color and this_active == active_color, "all fragments must agree on the source palette"
        fragment_arrays.append(arr)
    assert 2 <= len(fragment_arrays) <= 3, "need 2 or 3 fragment cells"
    assert shared_color is not None and active_color is not None

    full_mask = np.zeros((CELL_INNER, CELL_INNER), dtype=bool)
    for arr in fragment_arrays:
        frag_mask = arr == active_color
        assert np.any(frag_mask), "every fragment must contribute source pixels"
        full_mask |= frag_mask
    assert np.count_nonzero(full_mask) in ACTIVE_PIXEL_COUNTS, "reconstructed source mask has unexpected density"
    assert all(not np.array_equal(arr == active_color, full_mask) for arr in fragment_arrays), "no fragment may already be complete"

    full_pattern = np.where(full_mask, active_color, shared_color).astype(int)
    marker_positions = _row_major(
        [(i, j) for i, row in enumerate(cell_pat) for j, pat in enumerate(row) if pat == marker_pat]
    )
    return majority_arr, full_pattern, full_mask, marker_positions


def _apply_stamps(
    out: np.ndarray,
    step: int,
    majority_arr: np.ndarray,
    full_pattern: np.ndarray,
    mask: np.ndarray,
    anchors: list[tuple[int, int]],
) -> np.ndarray:
    inner = step - 1
    assert mask.shape == (inner, inner)
    for top, left in anchors:
        assert 0 <= top <= CELL_GRID_N - inner and 0 <= left <= CELL_GRID_N - inner, "stamp anchor out of range"
        for i in range(inner):
            for j in range(inner):
                target_pat = full_pattern if mask[i, j] else majority_arr
                r0 = (top + i) * step + 1
                c0 = (left + j) * step + 1
                out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a full source pattern from several fragment cells, then replay multiple ordered overlays.

    Input structure:
    - The board is partitioned into equal 3x3-interior cells by uniform divider lines.
    - Most cells show one majority pattern.
    - A few singleton anomaly cells are partial source fragments, while 2 or 3 identical anomaly
      cells act as repeated stamp markers.

    Rule:
    - Merge all fragment cells by taking the union of their source-only pixels in the shared 3x3
      coordinate frame, producing one completed source pattern.
    - Sort the repeated marker cells from top-left to bottom-right and treat each marker position
      as the top-left anchor of a 3x3 block on the cell lattice.
    - Replay those anchors in order, writing the completed source pattern on mask-active cells and
      the majority pattern on mask-inactive cells for each stamped block.

    Key insight: The anomalies split into two jobs: unique fragment cells describe what to stamp,
    and repeated marker cells describe where to stamp it and in which visible geometric order.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    patterns, cell_pat = _extract_cell_patterns(g, step)
    majority_arr, full_pattern, mask, marker_positions = _summarize_scene(patterns, cell_pat)
    out = g.copy()
    return _apply_stamps(out, step, majority_arr, full_pattern, mask, marker_positions)


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
    arr = np.empty((MASK_SIZE, MASK_SIZE), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _random_source_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count])


def _split_mask_into_fragments(mask: np.ndarray, fragment_count: int) -> list[np.ndarray] | None:
    active = [tuple(coord) for coord in np.argwhere(mask)]
    options = _FRAGMENT_SIZE_OPTIONS[(len(active), fragment_count)]
    if not options:
        return None
    for _ in range(30):
        sizes = random.choice(options)
        shuffled = active[:]
        random.shuffle(shuffled)
        fragments: list[np.ndarray] = []
        start = 0
        for size in sizes:
            frag = np.zeros_like(mask, dtype=bool)
            for r, c in shuffled[start : start + size]:
                frag[r, c] = True
            fragments.append(frag)
            start += size
        if len({tuple(f.ravel().tolist()) for f in fragments}) == len(fragments):
            random.shuffle(fragments)
            return fragments
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _covered_cells(anchors: list[tuple[int, int]]) -> set[tuple[int, int]]:
    covered: set[tuple[int, int]] = set()
    for top, left in anchors:
        for i in range(MASK_SIZE):
            for j in range(MASK_SIZE):
                covered.add((top + i, left + j))
    return covered


def _anchors_overlap(anchors: list[tuple[int, int]]) -> bool:
    seen: set[tuple[int, int]] = set()
    for top, left in anchors:
        block = {(top + i, left + j) for i in range(MASK_SIZE) for j in range(MASK_SIZE)}
        if seen & block:
            return True
        seen |= block
    return False


def _apply_label_stamps(mask: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    labels = np.full((CELL_GRID_N, CELL_GRID_N), -1, dtype=int)
    for top, left in anchors:
        for i in range(MASK_SIZE):
            for j in range(MASK_SIZE):
                labels[top + i, left + j] = 1 if mask[i, j] else 0
    return labels


def _order_sensitive(mask: np.ndarray, anchors: list[tuple[int, int]]) -> bool:
    forward = _apply_label_stamps(mask, anchors)
    backward = _apply_label_stamps(mask, list(reversed(anchors)))
    return not np.array_equal(forward, backward)


def _choose_marker_anchors(mask: np.ndarray, fragment_count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(PLACEMENT_RANGE) for c in range(PLACEMENT_RANGE)]
    count_options = list(MARKER_COUNTS)
    random.shuffle(count_options)
    for marker_count in count_options:
        valid: list[list[tuple[int, int]]] = []
        for combo in combinations(cells, marker_count):
            anchors = _row_major(list(combo))
            if not _anchors_overlap(anchors):
                continue
            if not _order_sensitive(mask, anchors):
                continue
            if CELL_GRID_N * CELL_GRID_N - len(_covered_cells(anchors)) < fragment_count:
                continue
            valid.append(anchors)
        if valid:
            return random.choice(valid)
    return None


def _pick_fragment_positions(covered: set[tuple[int, int]], fragment_count: int) -> list[tuple[int, int]] | None:
    safe = [(i, j) for i in range(CELL_GRID_N) for j in range(CELL_GRID_N) if (i, j) not in covered]
    if len(safe) < fragment_count:
        return None
    return random.sample(safe, fragment_count)


def _build_base_grid(separator: int, majority: np.ndarray) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    return grid


def _marker_pattern(majority_exclusive: int, marker_exclusive: int) -> np.ndarray:
    return np.where(_MARKER_MASK, marker_exclusive, majority_exclusive).astype(int)


def _build_input_grid(
    separator: int,
    majority: np.ndarray,
    marker_pat: np.ndarray,
    anchors: list[tuple[int, int]],
    fragment_cells: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    grid = _build_base_grid(separator, majority)
    for i, j, fragment in fragment_cells:
        paste(grid, fragment, *_cell_origin(i, j))
    for i, j in anchors:
        paste(grid, marker_pat, *_cell_origin(i, j))
    return grid


def _build_output_grid(
    inp: np.ndarray,
    majority: np.ndarray,
    full_pattern: np.ndarray,
    mask: np.ndarray,
    anchors: list[tuple[int, int]],
) -> np.ndarray:
    return _apply_stamps(inp.copy(), CELL_STEP, majority, full_pattern, mask, anchors)


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, src_exclusive, marker_exclusive = random.sample(range(COLOR_POOL), 5)
    majority = _random_lr_pattern(shared, maj_exclusive)
    source_mask = _random_source_mask()
    full_pattern = np.where(source_mask, src_exclusive, shared).astype(int)

    fragment_count = random.choice(FRAGMENT_COUNTS)
    fragments = _split_mask_into_fragments(source_mask, fragment_count)
    if fragments is None:
        return None

    anchors = _choose_marker_anchors(source_mask, fragment_count)
    if anchors is None:
        return None
    covered = _covered_cells(anchors)
    fragment_positions = _pick_fragment_positions(covered, fragment_count)
    if fragment_positions is None:
        return None

    marker_pat = _marker_pattern(maj_exclusive, marker_exclusive)
    fragment_cells: list[tuple[int, int, np.ndarray]] = []
    for (i, j), frag_mask in zip(fragment_positions, fragments):
        fragment_pat = np.where(frag_mask, src_exclusive, shared).astype(int)
        fragment_cells.append((i, j, fragment_pat))

    inp = _build_input_grid(separator, majority, marker_pat, anchors, fragment_cells)
    out = _build_output_grid(inp, majority, full_pattern, source_mask, anchors)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(_apply_label_stamps(source_mask, anchors), _apply_label_stamps(source_mask, list(reversed(anchors)))):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
