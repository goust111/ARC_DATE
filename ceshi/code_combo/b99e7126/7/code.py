# Combo variant of b99e7126 (index 7)
# combo_spec: D4/0+D15/0
# Name: 递归分形缩放 + 双层分形嵌套

from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np
import random

from arc_utils import paste


# Design analysis
# ===============
# spec:
#   concretization:
#     - Preserve the original ingredients: a regular divider grid, one majority 3x3 cell pattern,
#       one minority 3x3 cell pattern, and the minority pattern's dominant-color bitmap as the rule source.
#     - The fusion changes the transform, not the object vocabulary. Let the minority cell encode a
#       3x3 boolean mask M. The original task paints one 3x3 block of cells from M.
#     - Here the first stage is still that 3x3 macro layout, but every active macro cell is itself
#       replaced by another copy of M. The final cell-level blueprint is therefore kron(M, M), a 9x9
#       two-level self-similar mask.
#     - A 9x9 fractal window no longer fits inside the original 7x7 cell grid, so the cell lattice is
#       enlarged to 13x13. With 3x3 interiors and 1-pixel dividers this yields a 53x53 pixel canvas.
#   behavior_trace:
#     - Example: suppose the minority cell has 5 active pixels in a "cross with one corner missing" mask.
#       In the original task, those 5 actives directly mark 5 minority cells inside one 3x3 block.
#     - In the combo task, those same 5 actives first choose 5 macro positions in a 3x3 arrangement.
#       Inside each chosen macro position, the same 3x3 mask appears again, so the output has 5*5 = 25
#       minority cells spread over a 9x9 window.
#     - The input does not reveal all 25 or 36 final actives. Instead it shows 2-3 full second-level
#       subcopies plus a few anchor cells. This makes the nesting visible while still leaving completion work.
#   complexity_calibration:
#     - Keep the base mask size fixed at 3x3 and restrict its active count to 5 or 6. This keeps the
#       9x9 result dense enough to look fractal but not so dense that majority cells disappear.
#     - Require the base mask to touch all four edges and reject masks with obvious flip/rotation symmetry.
#       That lowers placement ambiguity and makes the repeated motif visually distinctive.
#     - Limit the visible evidence to 2-3 complete second-level copies, optionally with up to 2 extra anchors.
#       This gives the solver repeated support for the recursive rule without leaking the entire output.
#   implicit_assumption_audit:
#     - Original `mh, mw = mask.shape` assumed the final paint window had the same size as the minority
#       cell interior. The combo breaks that: the real window is 9x9, so placement and painting must use
#       `fractal_mask.shape`.
#     - Original candidate search assumed every shown minority cell lived inside one flat 3x3 placement.
#       Now shown minority cells may occupy several separated 3x3 subwindows, so matching must be done
#       against the full recursive mask, not the base mask.
#     - Original filling logic directly mapped each base-mask pixel to one output cell. Now there are two
#       stages: first-level actives decide which macro slots exist, and second-level actives decide the
#       final cell coordinates inside those slots. Encoding this as `kron(M, M)` keeps the reasoning exact.
#     - The "dominant color inside the minority pattern is active" heuristic still works, but generation
#       must guarantee the dominant color is unique; using 5 or 6 active pixels in a 3x3 cell ensures that.
#   placement_legality:
#     - The recursive window is 9x9 cells and the board is 13x13 cells, so legal top-left offsets remain
#       0..4 in each axis, matching the original placement freedom without boundary overflow.
#     - Every painted element is still a whole 3x3 cell interior. There are no partial overlaps and no
#       extra overlay layer beyond swapping majority cells for minority cells in selected positions.
#   observability:
#     - A human can inspect one minority cell to read the 3x3 motif, then notice the same motif already
#       reappearing as complete 3x3 subcopies at the cell-grid scale. That repeated evidence is what makes
#       "recursive self-copy" inferable from a few examples rather than looking like a random larger blob.
#   visual_cleanliness:
#     - The scene still has only divider lines plus two cell textures, so the image remains clean.
#       The recursive pattern lives inside one bounded 9x9 window, keeping the composition organized.
#   inferability_of_change:
#     - No numeric property of colors is used. Colors only distinguish separator / majority-exclusive /
#       shared / minority-exclusive roles. The nontrivial rule is carried by repeated spatial structure.
#   solver_generator_contract:
#     - Solver assumes exactly two cell patterns are present and the shown minority cells form a strict
#       subset of one uniquely placeable two-level fractal mask.
#     - Generator enforces those assumptions by constructing output first, revealing only a subset with
#       complete second-level copies, checking uniqueness, and validating `main(inp) == out`.


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


def _recursive_fractal_mask(mask: np.ndarray) -> np.ndarray:
    """Build the explicit two-level recursive mask encoded by the combo."""
    base = mask.astype(int)
    return np.kron(base, base).astype(bool)


def _extract_cell_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], list[list[tuple[int, ...]]], dict[tuple[int, ...], int]]:
    """Read every cell interior as a pattern key and count occurrences."""
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    counts: dict[tuple[int, ...], int] = {}
    n = (g.shape[0] - 1) // step
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
            counts[pat] = counts.get(pat, 0) + 1
        cell_pat.append(row)
    return patterns, cell_pat, counts


def _existing_minority_mask(cell_pat: list[list[tuple[int, ...]]], minority_pat: tuple[int, ...]) -> np.ndarray:
    """Convert the observed minority cells into a boolean cell-grid mask."""
    n = len(cell_pat)
    existing = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            existing[i, j] = cell_pat[i][j] == minority_pat
    return existing


def _candidate_offsets(existing: np.ndarray, target_mask: np.ndarray) -> list[tuple[int, int]]:
    """All placements where every shown minority cell agrees with the target mask."""
    n = existing.shape[0]
    th, tw = target_mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(n - th + 1):
        for left in range(n - tw + 1):
            block = np.zeros_like(existing)
            block[top : top + th, left : left + tw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + th, left : left + tw]
            if np.any(ex_block & ~target_mask):
                continue
            candidates.append((int(top), int(left)))
    return candidates


def _relative_active_positions(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(coord) for coord in np.argwhere(mask)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fused two-level fractal whose cell-scale layout recursively repeats the minority motif.

    Input structure:
    - The picture is partitioned into equal 3x3 cell interiors by one-color separator lines.
    - Exactly two cell patterns appear: a common majority texture and a rarer minority texture.
    - Inside the minority texture, one color dominates; that dominant-color bitmap is the base 3x3 motif.

    Rule:
    - Read the minority cell's dominant-color bitmap as a 3x3 base mask.
    - Use that mask twice: first as a coarse 3x3 macro arrangement, then again inside every active macro slot.
    - This yields a 9x9 recursive blueprint. Find the unique placement where the already shown minority cells
      are a subset of that blueprint.
    - Paint every active position in the placed 9x9 window with the minority cell pattern and every inactive
      position there with the majority pattern, leaving the rest of the board unchanged.

    Key insight: The odd cell is not only a legend for one enlargement step. Its internal 3x3 shape is copied
    at two scales, so some minority cells in the input are already miniature repeats of the same motif and the
    missing cells are recovered by recognizing this nested self-similarity.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    patterns, cell_pat, counts = _extract_cell_patterns(g, step)
    assert len(counts) == 2, "must have exactly 2 distinct cell patterns (majority + minority)"

    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    base_mask = minority_arr == active_color
    fractal_mask = _recursive_fractal_mask(base_mask)
    fh, fw = fractal_mask.shape
    assert fh <= n and fw <= n, "recursive mask must fit in the cell grid"

    existing = _existing_minority_mask(cell_pat, minority_pat)
    candidates = _candidate_offsets(existing, fractal_mask)
    assert len(candidates) == 1, "must have a unique recursive mask placement"
    top, left = candidates[0]

    out = g.copy()
    for i in range(fh):
        for j in range(fw):
            target_pat = minority_arr if fractal_mask[i, j] else majority_arr
            r0 = (top + i) * step + 1
            c0 = (left + j) * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (recursive_double_fractal) ---------------

CELL_GRID_N = 13
CELL_INNER = 3
CELL_STEP = CELL_INNER + 1
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
MASK_SIZE = CELL_INNER
FRACTAL_LEVELS = 2
FRACTAL_SIZE = MASK_SIZE ** FRACTAL_LEVELS
PLACEMENT_RANGE = CELL_GRID_N - FRACTAL_SIZE + 1
ACTIVE_PIXEL_COUNTS = (5, 6)
GIVEN_MACRO_COPY_COUNTS = (2, 3)
COLOR_POOL = 10
MAX_RETRIES = 600


def _is_visibly_asymmetric(mask: np.ndarray) -> bool:
    """Reject masks whose recursive image would be too symmetric or placement-ambiguous."""
    return not any(
        np.array_equal(mask, transformed)
        for transformed in (
            np.fliplr(mask),
            np.flipud(mask),
            np.rot90(mask, 1),
            np.rot90(mask, 2),
            np.rot90(mask, 3),
        )
    )


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate 3x3 masks that stay legible after the second recursive embedding."""
    by_count: dict[int, list[np.ndarray]] = {}
    for active_count in ACTIVE_PIXEL_COUNTS:
        masks: list[np.ndarray] = []
        for combo in combinations(range(MASK_SIZE * MASK_SIZE), active_count):
            mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=bool)
            mask.flat[list(combo)] = True
            if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
                continue
            if not _is_visibly_asymmetric(mask):
                continue
            masks.append(mask)
        by_count[active_count] = masks
    return by_count


_MASKS_BY_COUNT = _precompute_masks()


def _random_two_color_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform 3x3 pattern that uses both colors."""
    arr = np.array(
        [random.choice((color_a, color_b)) for _ in range(MASK_SIZE * MASK_SIZE)],
        dtype=int,
    ).reshape(MASK_SIZE, MASK_SIZE)
    if np.all(arr == arr[0, 0]):
        other = color_b if arr[0, 0] == color_a else color_a
        arr[random.randrange(MASK_SIZE), random.randrange(MASK_SIZE)] = other
    if not np.any(arr == color_a):
        arr[0, 0] = color_a
    if not np.any(arr == color_b):
        arr[-1, -1] = color_b
    return arr


def _random_base_mask() -> np.ndarray:
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _full_second_level_copy_positions(
    base_mask: np.ndarray, macro_subset: tuple[tuple[int, int], ...]
) -> list[tuple[int, int]]:
    """Reveal complete 3x3 second-level copies for selected first-level active macro slots."""
    micro_active = _relative_active_positions(base_mask)
    subset: set[tuple[int, int]] = set()
    for macro_r, macro_c in macro_subset:
        for micro_r, micro_c in micro_active:
            subset.add((macro_r * MASK_SIZE + micro_r, macro_c * MASK_SIZE + micro_c))
    return sorted(subset)


def _placement_candidates_for_subset(
    fractal_mask: np.ndarray, subset: list[tuple[int, int]], top: int, left: int
) -> list[tuple[int, int]]:
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset:
        existing[top + r, left + c] = True
    return _candidate_offsets(existing, fractal_mask)


def _choose_given_subset(
    base_mask: np.ndarray, fractal_mask: np.ndarray, top: int, left: int
) -> list[tuple[int, int]] | None:
    """
    Reveal 2-3 complete second-level copies so the double nesting is visible, then add a small
    number of anchors only if needed to make the 9x9 placement unique.
    """
    macro_active = _relative_active_positions(base_mask)
    fractal_active = _relative_active_positions(fractal_mask)

    sizes = list(GIVEN_MACRO_COPY_COUNTS)
    random.shuffle(sizes)
    for size in sizes:
        if size >= len(macro_active):
            continue
        macro_combos = list(combinations(macro_active, size))
        random.shuffle(macro_combos)
        for macro_subset in macro_combos:
            base_subset = set(_full_second_level_copy_positions(base_mask, macro_subset))
            remaining = [pos for pos in fractal_active if pos not in base_subset]

            extra_options: list[tuple[tuple[int, int], ...]] = [tuple()]
            extra_options.extend((pos,) for pos in remaining)
            if len(remaining) >= 2:
                pair_pool = list(combinations(remaining, 2))
                random.shuffle(pair_pool)
                extra_options.extend(pair_pool[:20])

            for extras in extra_options:
                subset = sorted(base_subset | set(extras))
                if _placement_candidates_for_subset(fractal_mask, subset, top, left) == [(top, left)]:
                    return subset
    return None


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(
    separator: int, majority: np.ndarray, minority: np.ndarray, active_cells: list[tuple[int, int]]
) -> np.ndarray:
    """Construct the full pixel grid from the majority texture plus selected minority cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_two_color_pattern(shared, maj_exclusive)
    base_mask = _random_base_mask()
    minority = np.where(base_mask, min_exclusive, shared).astype(int)
    fractal_mask = _recursive_fractal_mask(base_mask)

    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    active_cells = [(top + r, left + c) for r, c in _relative_active_positions(fractal_mask)]
    out = _build_grid(separator, majority, minority, active_cells)

    subset = _choose_given_subset(base_mask, fractal_mask, top, left)
    if subset is None:
        return None
    given_cells = [(top + r, left + c) for r, c in subset]
    inp = _build_grid(separator, majority, minority, given_cells)

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
    raise RuntimeError("failed to generate a valid recursive double-fractal b99e7126 input")
