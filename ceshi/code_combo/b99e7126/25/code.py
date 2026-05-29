# Combo variant of b99e7126 (index 25)
# combo_spec: D5/0+D13/1
# Name: 仅覆盖差异区域 + 旋转后对齐

from __future__ import annotations

from functools import reduce
from itertools import combinations, product
from math import gcd
import random

import numpy as np

from arc_utils import paste


"""
Design analysis for combo D5/0 + D13/1 on b99e7126.

a) Concrete behavior and trace
   - Original task:
     read the minority cell's 3x3 interior as a boolean mask, place that same mask at cell-grid
     scale, and fill every active cell with the minority pattern.
   - Fused task:
     the input minority cells still reveal the unrotated 3x3 placement block, but the final
     enlarged mask is rotated after that placement is identified. The quadrant of the 3x3 block
     inside the 7x7 cell grid decides the rotation. The solver then compares the rotated target
     block against the original grid and only rewrites cells whose whole 3x3 interiors differ.
   - Small trace:
     suppose the minority interior mask is
         1 1 1
         1 1 0
         1 0 1
     and the given minority cells show a subset of those seven active positions inside a block
     whose top-left cell anchor is in the lower-left quadrant of the 7x7 layout. Placement is
     recovered from the unrotated subset, then the full 3x3 mask is rotated 270 degrees clockwise
     relative to the base orientation. Some given minority cells remain minority because they still
     match active cells after rotation; others revert to the majority pattern because the rotated
     target wants those locations inactive. New minority cells are added wherever the rotated mask
     is active and the input still shows majority there.

b) Complexity calibration
   - The task still has only two cell interiors to distinguish: majority vs minority. That keeps
     object detection as simple as the source task.
   - The generator restricts placement to non-central outer quadrants of the 7x7 cell grid, so the
     quadrant->rotation control is always visually unambiguous.
   - Minority masks are filtered to be asymmetric under every non-zero quarter-turn and to differ
     from their rotations by multiple cells. This makes the rotation visible rather than cosmetic.
   - Given subsets remain 3-4 cells, so placement still requires reasoning from a strict subset.

c) Implicit assumption audit against the original solver
   - Original assumption: the observed minority cells are already a subset of the final enlarged
     mask. This no longer holds; they are only a subset of the unrotated base mask used to locate
     the block.
   - Original assumption: after locating the block, writing the full mask directly is sufficient.
     Now the solver must derive a quadrant-conditioned rotation first.
   - Original assumption: overwriting every cell in the 3x3 block is harmless and equivalent to the
     rule. Under D5/0 the intended operation is selective replacement based on whole-cell mismatch.
   - New assumption introduced here: the block center must lie strictly away from the grid's central
     row and column so that its quadrant is defined. The generator enforces this by never placing
     the 3x3 block on the middle offset.

d) Placement legality
   - The base 3x3 block still fits entirely inside the fixed 7x7 cell grid.
   - All cells remain aligned to the original divider lattice; no pixel-level overlaps occur.
   - Because only full cell interiors are replaced, divider rows/columns remain untouched.
   - The generator validates that the chosen minority subset yields exactly one legal base-mask
     placement, and that the rotated target remains inside the same 3x3 block.

e) Visual observability
   - A human can still read the minority cell interior as the local blueprint.
   - The changed rule is visible because the output is not just "complete the visible subset":
     at least one shown minority cell is deliberately removed after rotation, and at least one new
     minority cell is added elsewhere. This makes the rotation and selective-difference rewrite
     evident across examples.
   - The block quadrant is visible from the cluster location within the 7x7 lattice, so the control
     signal is structural rather than color-id based.

f) Visual cleanliness
   - Only one 3x3 cell block is ever affected at grid scale, just as in the source task, so the
     image does not become cluttered.
   - The generator requires both "kept" and "removed" minority clues, but only a small number of
     them, avoiding noisy widespread toggling.
   - Majority/minority interiors still use only two colors each, and divider lines stay constant.

g) Color and transform inference
   - No rule depends on numeric color ordering. The minority-exclusive color is identified by being
     the more frequent color inside the minority cell, exactly as in the source task.
   - Rotation is inferred from geometry only: the same minority interior can produce different
     large-scale outputs depending on the block quadrant.
   - D5/0 stays meaningful because the generator guarantees a non-empty symmetric difference between
     the input minority-cell set and the rotated target minority-cell set.

h) Variant behavior metric used after implementation
   - `_fusion_metrics(inp, out)` recovers the base placement, the quadrant-conditioned rotation, and
     counts how many minority cells were removed, kept, and added.
   - Accepted samples are required to have: a non-zero rotation, at least one removed given cell,
     at least one kept given cell, and at least one added minority cell. If any of these counts were
     zero, the fusion would visually collapse toward the source task.

i) New solver assumptions checked against the generator
   - Exactly two cell interiors dominate the grid: one majority pattern and one minority pattern.
   - The minority interior's active color is uniquely the most frequent interior color.
   - The input minority cells uniquely place the unrotated base mask in one outer quadrant.
   - That outer quadrant implies a non-zero rotation in generated samples.
   - Whole-cell comparison is enough for selective overwrite because every cell interior is always
     exactly either the majority pattern or the minority pattern.
"""


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """Infer the regular cell lattice from full divider rows/cols of one separator color."""
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


def _cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]]]:
    """Return unique cell interiors and the pattern id stored at each cell position."""
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
            pat = tuple(int(x) for x in pat_arr.ravel())
            if pat not in patterns:
                patterns[pat] = pat_arr
            row.append(pat)
        cell_pat.append(row)
    return patterns, cell_pat


def _majority_minority(cell_pat: list[list[tuple[int, ...]]]) -> tuple[tuple[int, ...], tuple[int, ...], dict[tuple[int, ...], int]]:
    """Identify the most common and least common cell interiors."""
    counts: dict[tuple[int, ...], int] = {}
    for row in cell_pat:
        for pat in row:
            counts[pat] = counts.get(pat, 0) + 1
    assert len(counts) >= 2, "must have at least 2 distinct cell patterns"
    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    return minority_pat, majority_pat, counts


def _active_mask(minority_arr: np.ndarray) -> np.ndarray:
    """Treat the minority interior's uniquely most frequent color as the active blueprint color."""
    vals, cnts = np.unique(minority_arr, return_counts=True)
    order = np.argsort(-cnts)
    vals = vals[order]
    cnts = cnts[order]
    assert cnts.size >= 2 and int(cnts[0]) > int(cnts[1]), "minority active color must be uniquely most frequent"
    return minority_arr == int(vals[0])


def _existing_minority(cell_pat: list[list[tuple[int, ...]]], minority_pat: tuple[int, ...]) -> np.ndarray:
    """Boolean cell-grid mask of where the input currently shows the minority interior."""
    n = len(cell_pat)
    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat
    return existing


def _placement_candidates(mask: np.ndarray, existing: np.ndarray) -> list[tuple[int, int]]:
    """All placements where `existing` is a subset of the given unrotated mask."""
    n = existing.shape[0]
    mh, mw = mask.shape
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
    return candidates


def _block_quadrant(top: int, left: int, n: int, mh: int, mw: int) -> str:
    """Quadrant of the mask block, using the block center relative to the cell-grid center."""
    center_r = int(top + mh // 2)
    center_c = int(left + mw // 2)
    mid = n // 2
    assert center_r != mid and center_c != mid, "block center must not lie on the central row or column"
    vertical = "T" if center_r < mid else "B"
    horizontal = "L" if center_c < mid else "R"
    return vertical + horizontal


def _rotation_k_ccw(quadrant: str) -> int:
    """
    Quarter-turn count for `np.rot90(..., k)`.

    The quadrant order is read clockwise around the large grid:
    TL -> 0, TR -> 90 CW, BR -> 180, BL -> 270 CW.
    """
    return {"TL": 0, "TR": 3, "BR": 2, "BL": 1}[quadrant]


def _target_pattern_grid(
    base_mask: np.ndarray,
    top: int,
    left: int,
    n: int,
) -> tuple[np.ndarray, int]:
    """Return the rotated target minority mask for the placed 3x3 block plus the rotation id."""
    quadrant = _block_quadrant(top, left, n, base_mask.shape[0], base_mask.shape[1])
    k = _rotation_k_ccw(quadrant)
    rot_mask = np.rot90(base_mask, k)
    return rot_mask, int(k)


def _fusion_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int | str]:
    """Recover a few counts proving that rotation and selective difference both occurred."""
    g_in = np.asarray(inp, dtype=int)
    step, _, _ = _infer_cell_grid_params(g_in)
    patterns, cell_pat = _cell_patterns(g_in, step)
    minority_pat, majority_pat, _counts = _majority_minority(cell_pat)
    minority_arr = patterns[minority_pat]
    base_mask = _active_mask(minority_arr)
    existing = _existing_minority(cell_pat, minority_pat)
    candidates = _placement_candidates(base_mask, existing)
    assert len(candidates) == 1, "metric expects a unique placement"
    top, left = candidates[0]
    n = len(cell_pat)
    quadrant = _block_quadrant(top, left, n, base_mask.shape[0], base_mask.shape[1])
    rot_mask = np.rot90(base_mask, _rotation_k_ccw(quadrant))

    given = {(int(top + r), int(left + c)) for r, c in np.argwhere(existing[top : top + base_mask.shape[0], left : left + base_mask.shape[1]])}
    target = {(int(top + r), int(left + c)) for r, c in np.argwhere(rot_mask)}
    kept = len(given & target)
    removed = len(given - target)
    added = len(target - given)

    g_out = np.asarray(out, dtype=int)
    out_patterns, out_cell_pat = _cell_patterns(g_out, step)
    del out_patterns
    output_minority = sum(1 for i in range(n) for j in range(n) if out_cell_pat[i][j] == minority_pat)

    return {
        "quadrant": quadrant,
        "rotation_k_ccw": int(_rotation_k_ccw(quadrant)),
        "given": int(len(given)),
        "kept": int(kept),
        "removed": int(removed),
        "added": int(added),
        "output_minority": int(output_minority),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Locate the minority-cell blueprint, rotate its enlarged version by block quadrant, and rewrite only mismatching cells.

    Input structure:
    - A 29x29 grid is partitioned into a 7x7 lattice of 3x3 cell interiors separated by 1-pixel divider lines.
    - Most cells share one interior pattern (majority). A few cells show a second interior pattern (minority).
    - The minority interior itself encodes a 3x3 boolean blueprint via its more frequent color, but the visible minority cells in the input mark only a subset of that unrotated blueprint.

    Rule:
    - Read the minority interior as the base 3x3 mask and find the unique cell-grid placement where the visible minority cells fit inside that unrotated mask.
    - Use the placed block's quadrant within the 7x7 grid to choose a quarter-turn rotation, then rotate the full 3x3 mask before projecting it back to cell scale.
    - Compare each cell in that placed 3x3 block against the rotated target: active positions should be minority cells, inactive positions should be majority cells. Replace only the cells whose current interior differs from that target.

    Key insight:
    - The minority interior still tells you where the enlarged 3x3 block belongs, but the final large-scale pattern is not the same orientation as the clue. First recover the block from the unrotated subset, then use its quadrant to rotate the blueprint and flip only the cells that disagree with that rotated target.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat = _cell_patterns(g, step)
    minority_pat, majority_pat, _counts = _majority_minority(cell_pat)
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]
    base_mask = _active_mask(minority_arr)
    existing = _existing_minority(cell_pat, minority_pat)

    candidates = _placement_candidates(base_mask, existing)
    assert len(candidates) == 1, "must have a unique base-mask placement"
    top, left = candidates[0]

    rot_mask, _k = _target_pattern_grid(base_mask, top, left, n)

    out = g.copy()
    for i in range(rot_mask.shape[0]):
        for j in range(rot_mask.shape[1]):
            want_minority = bool(rot_mask[i, j])
            target_pat = minority_pat if want_minority else majority_pat
            if cell_pat[top + i][left + j] == target_pat:
                continue
            patch = minority_arr if want_minority else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = patch
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
MASK_SIZE = 3
PLACEMENT_RANGE = CELL_GRID_N - MASK_SIZE + 1
ACTIVE_PIXEL_COUNTS = (6, 7)
GIVEN_CELL_SIZES = (3, 4)
COLOR_POOL = 10
MAX_RETRIES = 500

_OUTER_TOPS = {
    "TR": [0, 1],
    "BR": [3, 4],
    "BL": [3, 4],
}
_OUTER_LEFTS = {
    "TR": [3, 4],
    "BR": [3, 4],
    "BL": [0, 1],
}
_QUADRANT_CYCLE = ("TR", "BR", "BL")
_GEN_CALL_COUNT = 0


def _precompute_masks() -> list[np.ndarray]:
    """Enumerate asymmetric 3x3 masks that visibly change under every non-zero rotation."""
    masks: list[np.ndarray] = []
    coords = list(product(range(MASK_SIZE), repeat=2))
    for active_count in ACTIVE_PIXEL_COUNTS:
        for combo in combinations(coords, active_count):
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
            for r, c in combo:
                mask[r, c] = True
            if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
                continue
            diffs = [int(np.sum(mask != np.rot90(mask, k))) for k in (1, 2, 3)]
            if min(diffs) < 4:
                continue
            masks.append(mask)
    assert masks, "expected at least one valid asymmetric mask"
    return masks


_MASKS = _precompute_masks()


def _random_two_color_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Random non-uniform 3x3 interior using exactly the two supplied colors."""
    while True:
        arr = np.random.choice([color_a, color_b], size=(MASK_SIZE, MASK_SIZE)).astype(int)
        present = {int(x) for x in np.unique(arr)}
        if present == {int(color_a), int(color_b)}:
            return arr


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Top-left pixel origin of cell `(i,j)`'s 3x3 interior."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(separator: int, majority: np.ndarray, minority: np.ndarray, minority_cells: list[tuple[int, int]]) -> np.ndarray:
    """Fill all cells with the majority pattern, then paste minority interiors at the chosen cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), int(separator), dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in minority_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _placement_candidates_from_subset(mask: np.ndarray, subset: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    """Find all unrotated placements consistent with a chosen subset of active cells."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True
    return _placement_candidates(mask, existing)


def _choose_given_subset(base_mask: np.ndarray, rot_mask: np.ndarray, top: int, left: int) -> tuple[tuple[int, int], ...] | None:
    """
    Pick a subset of base-mask active cells that
    - uniquely fixes the unrotated placement,
    - keeps at least one cell after rotation,
    - loses at least one cell after rotation.
    """
    active = [tuple(int(x) for x in coord) for coord in np.argwhere(base_mask)]
    rot_active = {tuple(int(x) for x in coord) for coord in np.argwhere(rot_mask)}
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    for size in sizes:
        valid = []
        for subset in combinations(active, size):
            subset_set = set(subset)
            if len(subset_set & rot_active) < 1:
                continue
            if len(subset_set - rot_active) < 1:
                continue
            if len(rot_active - subset_set) < 1:
                continue
            if _placement_candidates_from_subset(base_mask, subset, top, left) != [(top, left)]:
                continue
            valid.append(subset)
        if valid:
            return random.choice(valid)
    return None


def _pick_quadrant_and_anchor() -> tuple[str, int, int]:
    """Cycle through non-identity quadrants so rotation is always visually active in generated samples."""
    global _GEN_CALL_COUNT
    quadrant = _QUADRANT_CYCLE[_GEN_CALL_COUNT % len(_QUADRANT_CYCLE)]
    _GEN_CALL_COUNT += 1
    top = random.choice(_OUTER_TOPS[quadrant])
    left = random.choice(_OUTER_LEFTS[quadrant])
    return quadrant, int(top), int(left)


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int | str]] | None:
    """Build one input/output pair and verify it with the solver plus fusion metrics."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_two_color_pattern(shared, maj_exclusive)
    base_mask = random.choice(_MASKS)
    minority = np.where(base_mask, min_exclusive, shared).astype(int)

    quadrant, top, left = _pick_quadrant_and_anchor()
    k = _rotation_k_ccw(quadrant)
    rot_mask = np.rot90(base_mask, k)

    target_cells = [(int(top + r), int(left + c)) for r, c in np.argwhere(rot_mask)]
    out = _build_grid(separator, majority, minority, target_cells)

    subset = _choose_given_subset(base_mask, rot_mask, top, left)
    if subset is None:
        return None
    given_cells = [(int(top + r), int(left + c)) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)

    try:
        solved = main(inp)
        metrics = _fusion_metrics(inp, solved)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if int(metrics["rotation_k_ccw"]) == 0:
        return None
    if int(metrics["removed"]) < 1 or int(metrics["kept"]) < 1 or int(metrics["added"]) < 1:
        return None
    return inp, out, metrics


def generate_input() -> np.ndarray:
    """Output-first generator: accept only inputs that realize the fused behavior."""
    for _ in range(MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            inp, _out, _metrics = sample
            return inp
    raise RuntimeError("failed to generate a valid combo input for b99e7126 D5/0+D13/1")
