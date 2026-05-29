# Combo variant of b99e7126 (index 51)
# combo_spec: D2/3+D12/0
# Name: 源图案需旋转 + 相邻单元格旋转对称

from functools import reduce
from itertools import combinations
from math import gcd
import random

import numpy as np

from arc_utils import paste

# Design analysis for the fused variant
# a) Concrete behavior and trace:
#    The 29x29 board is still a 7x7 lattice of 3x3 cell interiors separated by 1-pixel lines.
#    In the original task, every normal cell is literally the same pattern M, the abnormal cells
#    all show one fixed pattern A, and A's dominant-color 3x3 mask is copied directly upward to
#    decide which cell positions should also be A. Here both changed stages interact:
#    - Background rule: normal cells follow a quarter-turn orbit, so cell (r,c) should equal
#      rot_cw(M0, phase + r + c). Moving right or down rotates the majority motif by 90 degrees.
#    - Source rule: A still supplies the 3x3 blueprint, but the blueprint is not read verbatim.
#      Before scaling it to cell level, rotate that binary mask by the same quarter-turn index as
#      the block's top-left background cell: rot_cw(mask(A), phase + top + left).
#    Example trace: if the inferred global phase is 1 and the 3x3 anomaly block starts at
#    (top,left)=(2,4), then the block anchor sits on turn (1+2+4)%4=3. The raw source mask from A
#    is rotated 270 degrees clockwise, and those True entries mark which cells in that 3x3 window
#    should contain A. False entries must be restored to each location's expected rotated majority
#    motif, not to one fixed background tile.
# b) Complexity calibration:
#    The fusion keeps only two coupled degrees of freedom: one global background phase in {0,1,2,3}
#    and one source-mask turn derived from that same phase plus block anchor. This is richer than
#    either single variant but still learnable because the same quarter-turn vocabulary repeats
#    across the whole board. Generator templates force all four majority rotations and all four
#    source-mask rotations to remain visually distinct, preventing collapse back to the original
#    task or to a weaker two-state variant.
# c) Implicit assumptions audit:
#    - Original `majority_pat = max(counts, ...)` assumed all normal cells are identical. Broken:
#      majority cells now split into four literal patterns, so solver must count rotation orbits.
#    - Original `existing = cell == minority_pat` assumed anomaly cells are the only literal
#      minority pattern and that this alone defines the object. Still partly true, but only after
#      reconstructing the rotating background; anomalies are cells that violate expected rotation.
#    - Original `mask = minority_arr == dominant_color` assumed the raw mask is used directly.
#      Broken: each candidate placement needs a phase-dependent quarter-turn before mask matching.
#    - Original fill logic wrote one constant majority tile into inactive positions. Broken: every
#      inactive cell must receive its own expected rotated background state.
#    - New coupled assumption: the source turn must be computed from the same `(phase + r + c) mod
#      4` law that governs the majority lattice. Generator enforces this exactly.
# d) Placement legality:
#    The anomaly blueprint is still 3x3, so legal anchors remain top,left in 0..4. Output-first
#    generation first builds the full rotating background, then overwrites the transformed active
#    positions with anomaly pattern A. Input keeps only a 3-4 cell subset of those active cells,
#    guaranteeing no overlap beyond the intended 3x3 window and no boundary violations.
# e) Visual observability:
#    A human can first see the majority motif rotating as one walks across rows/columns. The shown
#    anomaly cells all ignore that rotation and instead repeat the same literal pattern A. Across
#    examples, the final 3x3 anomaly block changes orientation in lockstep with the local rotation
#    phase of the background, so the fused dependency is visible without referencing hidden values.
# f) Visual cleanliness:
#    The scene still contains only divider lines, one rotating 2-color background orbit, and one
#    fixed 2-color anomaly motif. Using a small hand-picked library of asymmetric majority shapes
#    keeps the board legible; using asymmetric source masks keeps rotations easy to spot.
# g) Change inferability:
#    No color-id arithmetic is used. Colors keep the original four-role structure: separator,
#    majority-exclusive, shared, and anomaly-exclusive. The anomaly-exclusive color remains the
#    dominant one inside A, so the source mask is recoverable by frequency and then its required
#    rotation is inferred structurally from the visible background phase.
# New solver assumptions checked against generator:
#    - Majority background lies on one unique four-state quarter-turn orbit.
#    - The anomaly motif is fixed across all shown anomaly cells and uses exactly two colors.
#    - The anomaly source mask has 5-6 active pixels and four distinct quarter-turn forms.
#    - The chosen shown subset uniquely determines the 3x3 block once the background phase is
#      known, and generator self-verifies every sample with `main(inp) == out`.


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


def _rot_cw(arr: np.ndarray, k: int) -> np.ndarray:
    """Rotate a 2D pattern clockwise by k quarter-turns."""
    return np.rot90(arr, -int(k) % 4)


def _pattern_key(arr: np.ndarray) -> tuple[int, ...]:
    """Flatten a pattern into a hashable tuple."""
    return tuple(np.asarray(arr, dtype=int).ravel().tolist())


def _canonical_rotation_key(arr: np.ndarray) -> tuple[int, ...]:
    """Canonical key for a rotation orbit: min key among all 4 clockwise rotations."""
    return min(_pattern_key(_rot_cw(arr, k)) for k in range(4))


def _expected_majority(majority_anchor: np.ndarray, phase: int, i: int, j: int) -> np.ndarray:
    """Expected majority pattern at cell (i,j) under the inferred global phase."""
    return _rot_cw(majority_anchor, phase + i + j)


def _source_turn(phase: int, top: int, left: int) -> int:
    """Quarter-turn applied to the anomaly blueprint at a block anchor."""
    return (phase + top + left) % 4


def _transform_source_mask(mask: np.ndarray, phase: int, top: int, left: int) -> np.ndarray:
    """Rotate the anomaly blueprint according to the block anchor's background phase."""
    return _rot_cw(mask, _source_turn(phase, top, left))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a fractal anomaly overlay that is synchronized with a rotating background lattice.

    Input structure:
    - The board is divided into equal 3x3 cell interiors by single-color separator lines.
    - Normal cells are not identical copies: neighboring cells are quarter-turn rotations of one
      asymmetric majority motif.
    - A few cells repeat one fixed anomaly motif whose dominant-color pixels still encode a 3x3
      source blueprint, but that blueprint is not yet in its final orientation.

    Rule:
    - Infer the global quarter-turn phase of the majority background across the full 7x7 cell
      lattice.
    - Identify the anomaly motif as the repeated cell that breaks this rotation schedule.
    - Convert the anomaly motif into a binary source mask using its more frequent color, rotate
      that mask by the quarter-turn index of the anomaly block's top-left location, then find the
      unique 3x3 block placement consistent with the shown anomaly cells.
    - Fill every active position of that rotated source mask with the anomaly motif and restore all
      other cells to their expected rotated majority states.

    Key insight:
    The same local phase that tells you how a normal cell should rotate also tells you how to
    orient the anomaly's internal blueprint before projecting it up to cell scale.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1

    cell_arr: list[list[np.ndarray]] = []
    orbit_counts: dict[tuple[int, ...], int] = {}
    orbit_examples: dict[tuple[int, ...], np.ndarray] = {}
    for i in range(n):
        row: list[np.ndarray] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            row.append(arr)
            orbit = _canonical_rotation_key(arr)
            orbit_counts[orbit] = orbit_counts.get(orbit, 0) + 1
            if orbit not in orbit_examples:
                orbit_examples[orbit] = arr
        cell_arr.append(row)

    assert len(orbit_counts) >= 2, "need majority orbit plus anomaly pattern"
    majority_orbit = max(orbit_counts.items(), key=lambda kv: kv[1])[0]
    majority_anchor = orbit_examples[majority_orbit]

    phase_scores: list[tuple[int, int]] = []
    for phase in range(4):
        matches = 0
        for i in range(n):
            for j in range(n):
                expected = _expected_majority(majority_anchor, phase, i, j)
                if np.array_equal(cell_arr[i][j], expected):
                    matches += 1
        phase_scores.append((matches, phase))

    best_matches = max(score for score, _ in phase_scores)
    best_phases = [phase for score, phase in phase_scores if score == best_matches]
    assert len(best_phases) == 1, "rotation phase must be uniquely determined"
    phase = best_phases[0]

    expected_cells: list[list[np.ndarray]] = []
    mismatch_pos: list[tuple[int, int]] = []
    mismatch_patterns: dict[tuple[int, ...], np.ndarray] = {}
    for i in range(n):
        row: list[np.ndarray] = []
        for j in range(n):
            expected = _expected_majority(majority_anchor, phase, i, j)
            row.append(expected)
            if not np.array_equal(cell_arr[i][j], expected):
                mismatch_pos.append((i, j))
                mismatch_patterns[_pattern_key(cell_arr[i][j])] = cell_arr[i][j]
        expected_cells.append(row)

    assert mismatch_pos, "need at least one anomaly cell"
    assert len(mismatch_patterns) == 1, "all anomaly cells must share the same pattern"
    minority_arr = next(iter(mismatch_patterns.values()))

    vals, cnts = np.unique(minority_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    base_mask = minority_arr == active_color
    mh, mw = base_mask.shape

    existing = np.zeros((n, n), dtype=bool)
    for i, j in mismatch_pos:
        existing[i, j] = True

    candidates: list[tuple[int, int]] = []
    for top in range(0, n - mh + 1):
        for left in range(0, n - mw + 1):
            transformed = _transform_source_mask(base_mask, phase, top, left)
            block = np.zeros_like(existing)
            block[top : top + mh, left : left + mw] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + mh, left : left + mw]
            if np.any(ex_block & ~transformed):
                continue
            candidates.append((int(top), int(left)))

    assert len(candidates) == 1, "transformed mask placement must be uniquely determined"
    top, left = candidates[0]
    transformed = _transform_source_mask(base_mask, phase, top, left)

    out = g.copy()
    for i in range(n):
        for j in range(n):
            inside = top <= i < top + mh and left <= j < left + mw
            if inside and transformed[i - top, j - left]:
                target = minority_arr
            else:
                target = expected_cells[i][j]
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
ACTIVE_PIXEL_COUNTS = (5, 6)
GIVEN_CELL_SIZES = [3, 4]
COLOR_POOL = 10
MAX_RETRIES = 500

_MAJORITY_MASKS = [
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [0, 0, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=bool),
]


def _precompute_masks() -> dict[int, list[np.ndarray]]:
    """Enumerate asymmetric source masks whose four quarter-turns are all distinct."""
    by_count: dict[int, list[np.ndarray]] = {count: [] for count in ACTIVE_PIXEL_COUNTS}
    for bits in range(1 << (MASK_SIZE * MASK_SIZE)):
        mask = np.array([(bits >> k) & 1 for k in range(MASK_SIZE * MASK_SIZE)], dtype=bool).reshape(MASK_SIZE, MASK_SIZE)
        active_count = int(mask.sum())
        if active_count not in by_count:
            continue
        if not (mask[0, :].any() and mask[2, :].any() and mask[:, 0].any() and mask[:, 2].any()):
            continue
        variants = {_pattern_key(_rot_cw(mask.astype(int), k)) for k in range(4)}
        if len(variants) != 4:
            continue
        by_count[active_count].append(mask)
    return by_count


def _validate_majority_masks() -> None:
    """Fail fast if a background template collapses under quarter-turn rotation."""
    for mask in _MAJORITY_MASKS:
        variants = {_pattern_key(_rot_cw(mask.astype(int), k)) for k in range(4)}
        assert len(variants) == 4, "majority templates must expose all four rotations"


_MASKS_BY_COUNT = _precompute_masks()
_validate_majority_masks()


def _random_majority_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Pick a visually clean asymmetric template and color it with two colors."""
    mask = random.choice(_MAJORITY_MASKS)
    return np.where(mask, color_b, color_a).astype(int)


def _random_minority_mask() -> np.ndarray:
    """Pick a random asymmetric source mask with dominant active color."""
    active_count = random.choice(ACTIVE_PIXEL_COUNTS)
    return random.choice(_MASKS_BY_COUNT[active_count]).copy()


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    """Pixel origin of cell (i,j) interior within the full grid."""
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_grid(
    separator: int,
    majority_base: np.ndarray,
    phase: int,
    minority: np.ndarray,
    active_cells: list[tuple[int, int]],
) -> np.ndarray:
    """Construct the full grid from rotating background, then overwrite active anomaly cells."""
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, _expected_majority(majority_base, phase, i, j), *_cell_origin(i, j))
    for i, j in active_cells:
        paste(grid, minority, *_cell_origin(i, j))
    return grid


def _placement_candidates(
    mask: np.ndarray,
    subset_abs: tuple[tuple[int, int], ...] | list[tuple[int, int]],
    phase: int,
) -> list[tuple[int, int]]:
    """Find all anchors whose rotated source mask is consistent with the shown anomaly subset."""
    existing = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    for r, c in subset_abs:
        existing[r, c] = True

    candidates: list[tuple[int, int]] = []
    for top in range(PLACEMENT_RANGE):
        for left in range(PLACEMENT_RANGE):
            transformed = _transform_source_mask(mask, phase, top, left)
            block = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
            block[top : top + MASK_SIZE, left : left + MASK_SIZE] = True
            if np.any(existing & ~block):
                continue
            if np.any(existing[top : top + MASK_SIZE, left : left + MASK_SIZE] & ~transformed):
                continue
            candidates.append((top, left))
    return candidates


def _choose_given_subset(mask: np.ndarray, phase: int, top: int, left: int) -> list[tuple[int, int]] | None:
    """Choose a shown active subset that pins down the anchor and exposes multiple local phases."""
    transformed = _transform_source_mask(mask, phase, top, left)
    active_abs = [(top + r, left + c) for r, c in np.argwhere(transformed)]
    sizes = list(GIVEN_CELL_SIZES)
    random.shuffle(sizes)
    diversified: list[list[tuple[int, int]]] = []
    fallback: list[list[tuple[int, int]]] = []
    for size in sizes:
        valid = [
            list(subset)
            for subset in combinations(active_abs, size)
            if _placement_candidates(mask, subset, phase) == [(top, left)]
        ]
        if not valid:
            continue
        for subset in valid:
            local_turns = {(phase + r + c) % 4 for r, c in subset}
            if len(local_turns) >= 2:
                diversified.append(subset)
            else:
                fallback.append(subset)
        if diversified:
            return random.choice(diversified)
        if fallback:
            return random.choice(fallback)
    return None


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, choose input subset, verify with solver."""
    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_majority_pattern(shared, maj_exclusive)
    mask = _random_minority_mask()
    minority = np.where(mask, min_exclusive, shared).astype(int)

    phase = random.randrange(4)
    top = random.randrange(PLACEMENT_RANGE)
    left = random.randrange(PLACEMENT_RANGE)
    transformed = _transform_source_mask(mask, phase, top, left)
    active_cells = [(top + r, left + c) for r, c in np.argwhere(transformed)]
    out = _build_grid(separator, majority, phase, minority, active_cells)

    given_cells = _choose_given_subset(mask, phase, top, left)
    if given_cells is None:
        return None
    inp = _build_grid(separator, majority, phase, minority, given_cells)

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
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
