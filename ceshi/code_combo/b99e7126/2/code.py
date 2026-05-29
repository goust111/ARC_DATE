# Combo variant of b99e7126 (index 2)
# combo_spec: D1/1+D3/2
# Name: 颜色标记识别 + 锚点由标记单元格确定

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis
# a) Concrete behavior and trace
#    - The board remains the original 29x29 canvas: a 7x7 array of cells, each with a 3x3
#      interior, separated by full divider rows/cols.
#    - The large-scale 3x3 footprint is still the core object. A boolean mask says which of the
#      3x3 cell positions must become the minority pattern in the output.
#    - Fusion change 1 (D1/1): the source/template cell is no longer identified by being the
#      rarest repeated pattern. Instead it is the unique cell containing a dedicated source-marker
#      color. Inside that cell, the ordinary active color fills most active pixels and exactly one
#      active pixel is replaced by the source-marker color.
#    - Fusion change 2 (D3/2): the footprint placement is no longer inferred from the source
#      cell's absolute position. A separate anchor cell, visually "majority + one center dot",
#      fixes the center of the enlarged 3x3 block.
#    - Small trace:
#          true mask =
#            1 1 0
#            1 1 1
#            0 1 1
#      Suppose the ordinary active color is orange and the source marker is red. The source cell
#      shows the same 3x3 geometry, except one orange active pixel is red. A separate anchor cell
#      looks like the majority pattern except for a unique teal center pixel. The solver must:
#      1. detect the red-marked source cell as "what pattern";
#      2. treat both orange and red source pixels as active, because red is just a label on one
#         active location;
#      3. detect the teal-dot majority-like cell as "where to place";
#      4. stamp the ordinary orange/shared minority cell across the 3x3 block centered on teal.
#      Old logic would fail twice: using dominant-color-only source parsing would miss the red
#      active pixel, and using shown special cells to infer placement would stay ambiguous.
# b) Complexity calibration
#    - Only six colors are used: divider, majority-exclusive, shared, ordinary-active,
#      source-marker, anchor-marker.
#    - Input contains exactly two special cells beyond the majority field: one source template and
#      one anchor marker. This makes the role split legible rather than cluttered.
#    - The source marker changes only one active pixel, so humans can still see the ordinary
#      minority pattern inside the template. The anchor marker is a distinct one-pixel center dot,
#      so the two marker types are visually different.
# c) Old assumptions that no longer hold
#    - "Rarest cell pattern = source" is invalid because there are now two singleton special
#      patterns: source template and anchor marker.
#    - "Minority pattern already appears in several cells" is invalid: the input shows no ordinary
#      minority cells at all.
#    - "Active mask = dominant color inside the source/minority cell" is invalid because one true
#      active pixel is recolored to the source-marker color.
#    - "Placement is uniquely recoverable from shown active cells" is invalid because only one
#      active location is shown; the anchor must be used.
# d) Placement legality
#    - The enlarged footprint is still a 3x3 block, so anchor placement is legal iff its center
#      cell lies in rows/cols 1..5. The generator samples top/left in 0..4 to guarantee that.
#    - The mask center is forced active so the anchor cell is always overwritten in the output.
#    - The source cell is chosen from a non-center active footprint position, so it never overlaps
#      the anchor cell.
# e) Visual observability
#    - Across examples, one unique color always appears as a single odd pixel inside a richer
#      template cell and disappears in output; that supports "this marks the source".
#    - Another unique color always appears as one center pixel inside an otherwise majority-like
#      cell and also disappears in output; that supports "this marks the anchor".
#    - The output shows that the source cell's single odd pixel is not a hole: the corresponding
#      large-scale active position is filled, so humans can infer that the marker pixel still
#      counts as active.
# f) Visual cleanliness
#    - No extra legend objects are added outside the cell grid. Both markers are expressed as
#      ordinary cells with small local edits.
#    - Source and anchor semantics are separated by structure: one is a many-difference tricolor
#      template cell, the other is a one-pixel perturbation of the majority.
# g) Color-change inferability
#    - The source marker color has exactly one meaning: "this active source pixel is tagged but
#      still active". Its output behavior is visible because it turns into the ordinary active
#      color when the source cell is overwritten.
#    - The anchor marker color has exactly one meaning: "this majority-like cell is the center of
#      the enlarged footprint". It never affects the minority mask itself.
# h) Anti-degeneration metrics to inspect after implementation
#    - old_candidate_count: if we ignore the anchor and only use the shown source cell location as
#      an active hint, there must be multiple legal 3x3 placements.
#    - dominant_mask_diff: if we ignore the source marker and read only the dominant active color
#      from the source cell, the recovered 3x3 mask must miss exactly one active position.
#    - ordinary_input_cells: must be zero, so the source marker really is the only place where the
#      blueprint can be read.
# i) New solver assumptions enforced by the generator
#    - Exactly one anchor pattern differs from the majority by one center pixel and appears once.
#    - Exactly one source pattern appears once and contains exactly three colors: shared,
#      ordinary-active, source-marker.
#    - The source-marker color appears exactly once inside the source cell.
#    - The shared color is the only color common to the source and majority patterns.
#    - The center of the large-scale mask is active, and the source cell lies on another active
#      position of that same mask.


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
) -> tuple[int, int, dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]], dict[tuple[int, ...], int]]:
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
    return step, n, patterns, cell_pat, counts


def _hamming_distance(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def _find_pattern_cells(cell_pat: list[list[tuple[int, ...]]], target: tuple[int, ...]) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for i, row in enumerate(cell_pat):
        for j, pat in enumerate(row):
            if pat == target:
                hits.append((int(i), int(j)))
    return hits


def _source_components(source_arr: np.ndarray, majority_arr: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    shared_candidates = set(map(int, np.unique(source_arr))) & set(map(int, np.unique(majority_arr)))
    assert len(shared_candidates) == 1, "source and majority must share exactly one color"
    shared_color = next(iter(shared_candidates))

    vals, cnts = np.unique(source_arr, return_counts=True)
    color_counts = {int(v): int(c) for v, c in zip(vals, cnts)}
    non_shared = [color for color in color_counts if color != shared_color]
    assert len(non_shared) == 2, "source must contain shared + active + source-marker colors"

    marker_candidates = [color for color in non_shared if color_counts[color] == 1]
    assert len(marker_candidates) == 1, "source-marker color must appear exactly once in the source cell"
    source_marker = marker_candidates[0]
    ordinary_active = next(color for color in non_shared if color != source_marker)

    mask = source_arr != shared_color
    assert mask.any(), "source cell must contain active pixels"
    return mask, shared_color, ordinary_active, source_marker


def _ordinary_from_source(mask: np.ndarray, shared_color: int, ordinary_active: int) -> np.ndarray:
    return np.where(mask, ordinary_active, shared_color).astype(int)


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _old_style_candidates(mask: np.ndarray, shown_active_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Placements consistent with active hints if the anchor marker were ignored."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in shown_active_cells:
        existing[r, c] = True

    candidates: list[tuple[int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[top : top + MASK_SIZE, left : left + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + MASK_SIZE, left : left + MASK_SIZE]
            if np.any(ex_block & ~mask):
                continue
            candidates.append((int(top), int(left)))
    return candidates


def _build_grid(
    separator: int,
    majority: np.ndarray,
    overrides: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for (i, j), pat in overrides.items():
        paste(grid, pat, *_cell_origin(i, j))
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use one color-marked source cell for the blueprint and a separate marked cell for the anchor.

    Input structure:
    - The canvas is an NxN cell grid with 3x3 interiors separated by uniform divider lines.
    - Most cells share one majority pattern.
    - Exactly one source cell contains a dedicated source-marker color inside an otherwise
      minority-like pattern, and exactly one anchor cell looks like the majority except for one
      special center pixel.

    Rule:
    - Identify the anchor cell as the unique one-pixel deviation from the majority; it fixes the
      center of the enlarged 3x3 placement.
    - Identify the source cell as the other singleton special pattern, then read all of its
      non-shared pixels as active, including the source-marker pixel.
    - Convert that source template into the ordinary minority pattern by replacing the tagged
      source-marker pixel with the ordinary active color already present in the same cell.
    - Stamp the resulting minority/majority 3x3 block around the anchor-centered position.

    Key insight:
    - The two marked cells carry different information. The source marker answers "what counts as
      active in the template", while the anchor marker answers "where the enlarged template goes".
      Ignoring either one loses essential information.
    """
    g = np.asarray(grid, dtype=int)
    step, n, patterns, cell_pat, counts = _extract_cell_patterns(g)
    inner = step - 1
    assert len(counts) >= 3, "need majority, source template, and anchor patterns"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    anchor_candidates: list[tuple[int, ...]] = []
    source_candidates: list[tuple[int, ...]] = []
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        arr = patterns[pat]
        if cnt == 1 and _hamming_distance(arr, majority_arr) == 1:
            anchor_candidates.append(pat)
        else:
            source_candidates.append(pat)

    assert len(anchor_candidates) == 1, "must have exactly one anchor marker pattern"
    assert len(source_candidates) == 1, "must have exactly one source template pattern"

    anchor_pat = anchor_candidates[0]
    source_pat = source_candidates[0]
    source_arr = patterns[source_pat]

    mask, shared_color, ordinary_active, _source_marker = _source_components(source_arr, majority_arr)
    mh, mw = mask.shape
    anchor_r = mh // 2
    anchor_c = mw // 2
    assert mask[anchor_r, anchor_c], "anchor-aligned center must be active in the source mask"

    anchor_cells = _find_pattern_cells(cell_pat, anchor_pat)
    source_cells = _find_pattern_cells(cell_pat, source_pat)
    assert len(anchor_cells) == 1 and len(source_cells) == 1
    anchor_i, anchor_j = anchor_cells[0]
    source_i, source_j = source_cells[0]

    top = anchor_i - anchor_r
    left = anchor_j - anchor_c
    assert 0 <= top <= n - mh and 0 <= left <= n - mw, "anchor would place the block out of bounds"
    assert top <= source_i < top + mh and left <= source_j < left + mw, "source cell must lie inside the anchored block"
    assert mask[source_i - top, source_j - left], "source cell must occupy an active block position"

    ordinary_arr = _ordinary_from_source(mask, shared_color, ordinary_active)

    out = g.copy()
    for i in range(mh):
        for j in range(mw):
            target_pat = ordinary_arr if mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ANCHOR_REL = (1, 1)

ACTIVE_PIXEL_COUNTS = (6, 7)
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


def _precompute_masks():
    """Enumerate valid LR-symmetric 3x3 masks with active center and edge support."""
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
                if not mask[ANCHOR_REL]:
                    continue
                if mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any():
                    masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a random LR-symmetric 3x3 pattern using two colors (non-uniform)."""
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
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _make_source_pattern(mask: np.ndarray, ordinary_active: int, shared: int, source_marker: int) -> np.ndarray:
    ordinary = np.where(mask, ordinary_active, shared).astype(int)
    active_positions = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    source_marker_pos = random.choice(active_positions)
    source = ordinary.copy()
    source[source_marker_pos] = source_marker
    return source


def _analyze_variant_input(grid: np.ndarray) -> dict[str, int]:
    """
    Post-generation metrics:
    - old_candidate_count > 1 means the anchor is necessary.
    - dominant_mask_diff == 1 means the source marker is necessary.
    - ordinary_input_cells == 0 means the template is only present in marked form.
    """
    g = np.asarray(grid, dtype=int)
    _step, _n, patterns, cell_pat, counts = _extract_cell_patterns(g)
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    anchor_pat = None
    source_pat = None
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        arr = patterns[pat]
        if cnt == 1 and _hamming_distance(arr, majority_arr) == 1:
            anchor_pat = pat
        else:
            source_pat = pat
    assert anchor_pat is not None and source_pat is not None

    source_arr = patterns[source_pat]
    mask, shared_color, ordinary_active, _source_marker = _source_components(source_arr, majority_arr)
    ordinary_arr = _ordinary_from_source(mask, shared_color, ordinary_active)

    source_cell = _find_pattern_cells(cell_pat, source_pat)[0]
    old_candidates = _old_style_candidates(mask, [source_cell])

    dominant_mask = source_arr == ordinary_active
    dominant_output_diff = int(np.count_nonzero(mask != dominant_mask))

    ordinary_input_cells = len(_find_pattern_cells(cell_pat, tuple(ordinary_arr.ravel().tolist())))
    return {
        "old_candidate_count": int(len(old_candidates)),
        "dominant_mask_diff": dominant_output_diff,
        "ordinary_input_cells": int(ordinary_input_cells),
    }


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, ordinary_active, source_marker, anchor_marker = random.sample(range(COLOR_POOL), 6)
    majority = _random_lr_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    source = _make_source_pattern(mask, ordinary_active, shared, source_marker)
    ordinary = np.where(mask, ordinary_active, shared).astype(int)

    marker_pattern = majority.copy()
    marker_pattern[ANCHOR_REL] = anchor_marker

    if np.array_equal(source, majority) or np.array_equal(source, marker_pattern):
        return None

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_rel = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    source_rel_choices = [coord for coord in active_rel if coord != ANCHOR_REL]
    if not source_rel_choices:
        return None
    source_rel = random.choice(source_rel_choices)
    source_cell = (top + source_rel[0], left + source_rel[1])
    anchor_cell = (top + ANCHOR_REL[0], left + ANCHOR_REL[1])
    active_cells = [(top + r, left + c) for r, c in active_rel]

    out = _build_grid(
        separator,
        majority,
        {cell: ordinary for cell in active_cells},
    )
    inp = _build_grid(
        separator,
        majority,
        {
            source_cell: source,
            anchor_cell: marker_pattern,
        },
    )

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    metrics = _analyze_variant_input(inp)
    if metrics["old_candidate_count"] <= 1:
        return None
    if metrics["dominant_mask_diff"] != 1:
        return None
    if metrics["ordinary_input_cells"] != 0:
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo D1/1+D3/2 input")
