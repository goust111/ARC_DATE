# Combo variant of b99e7126 (index 15)
# combo_spec: D1/2+D13/2
# Name: 位置规则选择 + 镜像后对齐叠加

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    Example: if the source cell sits at the top midpoint (cell (0,3)), the old task would
#    look for the rare pattern and use that pattern's mask directly. Here several cells can
#    already share the minority pattern, so rarity does not identify the source. Instead the
#    top-mid cell is the source because it is in the special side-mid position. Its 3x3 mask
#    is flipped vertically, then expanded into the inward 3x3 cell block rows 0..2, cols 2..4,
#    aligned so the source remains the top-middle cell of that block. Input shows only a strict
#    subset of those active cells; output completes the mirrored block.
# b) Complexity calibration:
#    Only four source positions are allowed: top/bottom/left/right midpoints. That gives a small
#    repeated category set, and each category maps to exactly one mirror direction. The output
#    stays visually close to the original 3x3-on-7x7 construction while still being visibly
#    different because the completed shape depends on side-specific reflection.
# c) Old-assumption audit:
#    - Old code assumed the rare cell itself was the source. That is no longer valid because the
#      input may show 3-4 minority cells already.
#    - Old code searched every possible 3x3 placement. That is replaced by fixed anchoring from
#      the side-mid source position.
#    - Old code could accept a mirror-symmetric mask. This variant forbids that because mirrored
#      and unmirrored overlays must differ.
# d) Placement legality:
#    The source is always one of the four side-mid cells, and the 3x3 overlay block is anchored
#    inward from that source, so the full block always stays inside the 7x7 cell grid.
# e) Visual observability:
#    The border-mid source is conspicuous by geometry, not by color value. The completed cells
#    always lie in the adjacent inward 3x3 block, so humans can compare the source's interior
#    mask with the output and notice that the block is mirrored relative to the source.
# f) Visual cleanliness:
#    Only two cell interiors are used at grid scale: a majority pattern and a repeated minority
#    pattern. There are no extra markers, and the overlay affects just one compact 3x3 region.
# g) Inferable change structure:
#    No hidden numeric ordering is used. The position category (top/bottom vs left/right) is the
#    visible cue that decides whether the source mask flips up/down or left/right before stamping.
# i) New solver assumptions that the generator must guarantee:
#    - Exactly one of the four side-mid cells is a minority cell.
#    - The minority pattern uses two colors, with a strict frequency-majority active color.
#    - The mirrored mask keeps the side anchor active, otherwise the chosen source would vanish.
#    - The mirrored mask is not identical to the source mask, ensuring the combo does real work.


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = 29
MASK_SIZE = 3
MID = CELL_GRID_N // 2
COLOR_POOL = 10
MAX_RETRIES = 400
ACTIVE_PIXEL_COUNTS = (5, 6, 7)
GIVEN_CELL_SIZES = (3, 4)

SIDE_SPECS = {
    "top": {"source": (0, MID), "anchor": (0, 1)},
    "bottom": {"source": (CELL_GRID_N - 1, MID), "anchor": (MASK_SIZE - 1, 1)},
    "left": {"source": (MID, 0), "anchor": (1, 0)},
    "right": {"source": (MID, CELL_GRID_N - 1), "anchor": (1, MASK_SIZE - 1)},
}


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


def _apply_side_flip(mask: np.ndarray, side: str) -> np.ndarray:
    """Mirror the mask according to the source side."""
    if side in ("top", "bottom"):
        return np.flipud(mask)
    return np.fliplr(mask)


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i, j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Read every cell interior as a flattened pattern token plus its 3x3 array."""
    inner = step - 1
    n = (g.shape[0] - 1) // step
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


def _count_patterns(cell_pat: list[list[tuple[int, ...]]]) -> dict[tuple[int, ...], int]:
    """Count how many times each cell interior pattern appears."""
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    return counts


def _find_source_side(cell_pat: list[list[tuple[int, ...]]], minority_pat: tuple[int, ...]) -> tuple[str, tuple[int, int]]:
    """Locate the unique minority cell among the four side-mid candidate positions."""
    hits = []
    for side, spec in SIDE_SPECS.items():
        i, j = spec["source"]
        if cell_pat[i][j] == minority_pat:
            hits.append((side, (i, j)))
    assert len(hits) == 1, "exactly one side-mid source cell must use the minority pattern"
    return hits[0]


def _overlay_bounds(source_rc: tuple[int, int], side: str) -> tuple[int, int]:
    """Return the 3x3 cell-block origin anchored to the source cell."""
    sr, sc = source_rc
    ar, ac = SIDE_SPECS[side]["anchor"]
    top = sr - ar
    left = sc - ac
    return int(top), int(left)


def _active_mask_from_pattern(minority_arr: np.ndarray) -> np.ndarray:
    """Use the unique most-frequent color inside the source cell as the active mask."""
    vals, cnts = np.unique(minority_arr, return_counts=True)
    order = np.argsort(cnts)
    assert cnts[order[-1]] > cnts[order[-2]], "minority pattern must have a strict active-color majority"
    active_color = int(vals[order[-1]])
    return minority_arr == active_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a cell-grid fractal using a source chosen by position, then mirrored toward the center.

    Input structure:
    - The grid is partitioned into 7x7 cells with 3x3 interiors separated by one-color divider lines.
    - Every cell uses either a majority interior pattern or a repeated minority interior pattern.
    - Exactly one minority cell sits at a side-mid position of the cell grid; that border-mid cell is the source.

    Rule:
    - Read the source cell's 3x3 interior as a binary mask using its dominant color as the active color.
    - Mirror that mask toward the grid center: top/bottom sources flip vertically, left/right sources flip horizontally.
    - Expand the mirrored 3x3 mask to cell scale in the inward 3x3 block anchored at the source position.
    - Fill every active cell of that block with the minority pattern and every inactive cell with the majority pattern.

    Key insight: The source is identified by geometry rather than rarity, and that same geometry also decides
    which mirror axis to apply before the miniature pattern is stamped onto the larger cell grid.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    assert n == CELL_GRID_N and inner == CELL_INNER, "this variant expects the original 7x7-of-3x3 layout"

    patterns, cell_pat = _extract_cell_patterns(g, step)
    counts = _count_patterns(cell_pat)
    assert len(counts) == 2, "input must use exactly majority and minority cell patterns"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    side, source_rc = _find_source_side(cell_pat, minority_pat)
    source_mask = _active_mask_from_pattern(minority_arr)
    mirrored_mask = _apply_side_flip(source_mask, side)

    anchor = SIDE_SPECS[side]["anchor"]
    assert mirrored_mask[anchor], "mirrored mask must keep the source anchor active"
    top, left = _overlay_bounds(source_rc, side)
    assert 0 <= top <= n - MASK_SIZE and 0 <= left <= n - MASK_SIZE, "overlay block must stay inside the cell grid"

    out = g.copy()
    for i in range(MASK_SIZE):
        for j in range(MASK_SIZE):
            target_pat = minority_arr if mirrored_mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


def _random_majority_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform 3x3 pattern using exactly two colors."""
    arr = np.array([[random.choice((color_a, color_b)) for _ in range(MASK_SIZE)] for _ in range(MASK_SIZE)], dtype=int)
    if not (np.any(arr == color_a) and np.any(arr == color_b)):
        rr, cc = divmod(random.randrange(MASK_SIZE * MASK_SIZE), MASK_SIZE)
        arr[rr, cc] = color_b if arr[rr, cc] == color_a else color_a
    return arr


def _valid_source_masks_for_side(side: str) -> list[np.ndarray]:
    """Enumerate source masks that make the mirror effect visible for a given side."""
    anchor = SIDE_SPECS[side]["anchor"]
    masks = []
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(MASK_SIZE * MASK_SIZE)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        active_count = int(mask.sum())
        if active_count not in ACTIVE_PIXEL_COUNTS:
            continue
        if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        mirrored = _apply_side_flip(mask, side)
        if not mirrored[anchor]:
            continue
        if np.array_equal(mask, mirrored):
            continue
        masks.append(mask)
    assert masks, f"need at least one valid source mask for {side}"
    return masks


_MASKS_BY_SIDE = {side: _valid_source_masks_for_side(side) for side in SIDE_SPECS}


def _choose_given_cells(mirrored_mask: np.ndarray, anchor: tuple[int, int]) -> list[tuple[int, int]] | None:
    """Choose a strict visible subset of active cells, always including the source anchor."""
    active = [tuple(coord) for coord in np.argwhere(mirrored_mask)]
    assert anchor in active, "source anchor must be active in the mirrored mask"
    others = [coord for coord in active if coord != anchor]
    for size in random.sample(list(GIVEN_CELL_SIZES), k=len(GIVEN_CELL_SIZES)):
        if size >= len(active):
            continue
        valid = []
        for extra in combinations(others, size - 1):
            subset = [anchor, *extra]
            rows = {r for r, _ in subset}
            cols = {c for _, c in subset}
            if len(rows) < 2 or len(cols) < 2:
                continue
            valid.append(subset)
        if valid:
            return list(random.choice(valid))
    return None


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, active_cells: list[tuple[int, int]]) -> np.ndarray:
    """Construct a full grid from majority fill plus a set of minority cell locations."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Generate one input whose solved output follows the fused position+mirror rule."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_majority_pattern(shared, maj_exclusive)

    side = random.choice(list(SIDE_SPECS))
    source_mask = random.choice(_MASKS_BY_SIDE[side])
    mirrored_mask = _apply_side_flip(source_mask, side)
    minority = np.where(source_mask, min_exclusive, shared).astype(int)
    if np.array_equal(minority, majority):
        return None

    source_rc = SIDE_SPECS[side]["source"]
    anchor = SIDE_SPECS[side]["anchor"]
    top, left = _overlay_bounds(source_rc, side)

    full_active = [(top + r, left + c) for r, c in np.argwhere(mirrored_mask)]
    given_local = _choose_given_cells(mirrored_mask, anchor)
    if given_local is None:
        return None
    given_active = [(top + r, left + c) for r, c in given_local]

    out = _build_grid(separator, majority, minority, full_active)
    inp = _build_grid(separator, majority, minority, given_active)

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
    raise RuntimeError("failed to generate a valid combo variant input")
