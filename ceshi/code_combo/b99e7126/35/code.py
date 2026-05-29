# Combo variant of b99e7126 (index 35)
# combo_spec: D2/2+D4/2
# Name: 非正方形源单元格 + 非均匀缩放

# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original task: a 3x3 minority cell becomes a 3x3 arrangement of cells, one source pixel -> one target cell.
#    Fused task: a rectangular minority cell such as 2x3 or 3x2 is still the blueprint, but each source pixel
#    now expands to a rectangular block of cells with the same height/width as the source cell interior.
#    Example: a 2x3 source mask with active pixels [[1, 0, 1], [1, 1, 0]] becomes a 4x9 cell-level mask,
#    because each source pixel expands to a 2x3 block of cells. The input shows only a few complete active blocks;
#    the output fills every remaining active block.
# b) Complexity calibration:
#    The source shape is restricted to two non-square families, 2x3 and 3x2. This keeps the rule legible while
#    making the deformation obvious: 2x3 -> 4x9 cells and 3x2 -> 9x4 cells are visibly different from the
#    original 1x1-per-pixel lift. Source masks use 4 or 5 active pixels, so the minority color is visibly dominant.
# c) Implicit assumptions removed:
#    The original solver assumed one scalar cell step, square interiors, and equal row/col scaling. This version
#    infers divider spacing separately for rows and cols, extracts rectangular cell interiors, and builds the
#    expanded placement mask with independent row/col block sizes.
# d) Placement legality:
#    The generator chooses a top-left placement only after the expanded mask size is known, so the full stretched
#    region stays inside the cell lattice. Active blocks never overlap incorrectly because they are formed by
#    disjoint rectangles in the fixed cell grid.
# e) Visual observability:
#    Showing complete stretched blocks in the input makes the anisotropic scaling human-visible. The solver still
#    reasons only from cell patterns, but the examples reveal both the rectangular source cell and the repeated
#    block size.
# f) Visual cleanliness:
#    Only two cell interiors are used across the grid: majority and minority. Divider lines stay unchanged, and
#    the stretched minority region is a single coherent rectangle of repeated blocks, avoiding clutter.
# g) Inferability:
#    Colors are limited to four roles exactly as in the seed task. No ordering or choice depends on numeric color
#    ids; the active color is identified structurally as the majority color inside the minority rectangle.

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import paste


def _infer_rect_cell_grid_params(g: np.ndarray) -> tuple[int, int, list[int], list[int]]:
    """Infer separate row/col divider spacing for a rectangular tiled cell grid."""
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"

    row_divs.sort()
    col_divs.sort()
    row_gaps = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1)]
    col_gaps = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1)]
    row_step = row_gaps[0]
    col_step = col_gaps[0]
    for gap in row_gaps[1:]:
        row_step = gcd(row_step, gap)
    for gap in col_gaps[1:]:
        col_step = gcd(col_step, gap)

    assert row_step > 1 and col_step > 1, "divider steps must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must start at 0"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must hit the boundary"
    assert (h - 1) % row_step == 0 and (w - 1) % col_step == 0, "grid size must match divider step"
    return int(row_step), int(col_step), row_divs, col_divs


def _extract_patterns(
    g: np.ndarray, row_step: int, col_step: int
) -> tuple[list[list[tuple[int, ...]]], dict[tuple[int, ...], np.ndarray]]:
    """Return cell-pattern ids and a lookup of flattened pattern ids to arrays."""
    cell_rows = (g.shape[0] - 1) // row_step
    cell_cols = (g.shape[1] - 1) // col_step
    inner_h = row_step - 1
    inner_w = col_step - 1

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    for i in range(cell_rows):
        row: list[tuple[int, ...]] = []
        for j in range(cell_cols):
            r0 = i * row_step + 1
            c0 = j * col_step + 1
            arr = g[r0 : r0 + inner_h, c0 : c0 + inner_w].copy()
            key = tuple(int(v) for v in arr.ravel())
            if key not in patterns:
                patterns[key] = arr
            row.append(key)
        cell_pat.append(row)
    return cell_pat, patterns


def _pattern_roles(cell_pat: list[list[tuple[int, ...]]]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Identify minority and majority cell patterns by frequency."""
    counts = Counter(p for row in cell_pat for p in row)
    assert len(counts) == 2, "expected exactly 2 cell patterns"
    minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]
    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    return minority_pat, majority_pat


def _expanded_mask_from_minority(minority_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Derive the rectangular source mask and its non-uniformly scaled cell-level expansion."""
    values, counts = np.unique(minority_arr, return_counts=True)
    active_color = int(values[counts.argmax()])
    source_mask = minority_arr == active_color
    src_h, src_w = source_mask.shape
    expanded_mask = np.kron(source_mask, np.ones((src_h, src_w), dtype=bool))
    return source_mask, expanded_mask, src_h, src_w


def _find_unique_placement(existing: np.ndarray, expanded_mask: np.ndarray) -> tuple[int, int]:
    """Find the unique top-left cell coordinate where expanded_mask can explain the existing minority cells."""
    cell_rows, cell_cols = existing.shape
    big_h, big_w = expanded_mask.shape
    candidates: list[tuple[int, int]] = []
    for top in range(cell_rows - big_h + 1):
        for left in range(cell_cols - big_w + 1):
            block = np.zeros_like(existing)
            block[top : top + big_h, left : left + big_w] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[top : top + big_h, left : left + big_w]
            if np.any(ex_block & ~expanded_mask):
                continue
            candidates.append((int(top), int(left)))
    assert len(candidates) == 1, "minority blocks must determine a unique placement"
    return candidates[0]


def _instance_metrics(grid: np.ndarray) -> dict[str, int | tuple[int, int]]:
    """Internal helper for validation scripts: summarize the fused geometry of one instance."""
    g = np.asarray(grid, dtype=int)
    row_step, col_step, _, _ = _infer_rect_cell_grid_params(g)
    cell_pat, patterns = _extract_patterns(g, row_step, col_step)
    minority_pat, _ = _pattern_roles(cell_pat)
    source_mask, expanded_mask, scale_r, scale_c = _expanded_mask_from_minority(patterns[minority_pat])
    return {
        "src_shape": tuple(int(v) for v in source_mask.shape),
        "scale": (int(scale_r), int(scale_c)),
        "expanded_shape": tuple(int(v) for v in expanded_mask.shape),
        "source_active": int(source_mask.sum()),
        "expanded_active": int(expanded_mask.sum()),
        "active_multiplier": int(expanded_mask.sum() // source_mask.sum()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Lift a rectangular minority cell pattern to the cell lattice using anisotropic block scaling.

    Input structure:
    - The board is partitioned by full separator rows and columns, but row stride and column stride can differ.
    - Each cell interior is rectangular rather than square.
    - Almost every cell shares one majority interior, while a smaller set uses one minority interior.

    Rule:
    - Read the minority cell's interior as a binary source mask, using its dominant color as the active value.
    - Let the source rectangle have height M and width N; expand each source pixel to an MxN block of cells.
    - Place this stretched cell-level mask so every already-present minority cell lies on an active block.
    - Fill every active block cell with the minority pattern and every inactive block cell with the majority pattern.

    Key insight:
    - The minority rectangle is both the source image and the scale template: its height controls vertical block
      replication and its width controls horizontal block replication, so solving requires inferring two different
      periodicities and two different scale factors at once.
    """
    g = np.asarray(grid, dtype=int)
    row_step, col_step, _, _ = _infer_rect_cell_grid_params(g)
    cell_pat, patterns = _extract_patterns(g, row_step, col_step)
    minority_pat, majority_pat = _pattern_roles(cell_pat)

    minority_arr = patterns[minority_pat]
    majority_arr = patterns[majority_pat]
    _, expanded_mask, _, _ = _expanded_mask_from_minority(minority_arr)

    cell_rows = len(cell_pat)
    cell_cols = len(cell_pat[0])
    existing = np.zeros((cell_rows, cell_cols), dtype=bool)
    for i in range(cell_rows):
        for j in range(cell_cols):
            existing[i, j] = cell_pat[i][j] == minority_pat

    top, left = _find_unique_placement(existing, expanded_mask)

    out = g.copy()
    big_h, big_w = expanded_mask.shape
    inner_h = row_step - 1
    inner_w = col_step - 1
    for i in range(big_h):
        for j in range(big_w):
            target = minority_arr if expanded_mask[i, j] else majority_arr
            r0 = (top + i) * row_step + 1
            c0 = (left + j) * col_step + 1
            out[r0 : r0 + inner_h, c0 : c0 + inner_w] = target
    return out


# --------------- generator ---------------

CELL_GRID_ROWS = 11
CELL_GRID_COLS = 11
SOURCE_SHAPES = [(2, 3), (3, 2)]
ACTIVE_COUNTS = (4, 5)
GIVEN_SOURCE_BLOCKS = (2, 3)
COLOR_POOL = 10
MAX_RETRIES = 400


def _enumerate_masks(shape: tuple[int, int]) -> list[np.ndarray]:
    """Enumerate valid rectangular source masks for one shape."""
    h, w = shape
    total = h * w
    masks: list[np.ndarray] = []
    for bits in range(1 << total):
        arr = np.array([(bits >> k) & 1 for k in range(total)], dtype=bool).reshape(h, w)
        active = int(arr.sum())
        if active not in ACTIVE_COUNTS:
            continue
        if not np.all(arr.any(axis=0)) or not np.all(arr.any(axis=1)):
            continue
        masks.append(arr)
    return masks


MASKS_BY_SHAPE = {shape: _enumerate_masks(shape) for shape in SOURCE_SHAPES}


def _random_pattern(h: int, w: int, color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform h x w two-color pattern."""
    total = h * w
    k = random.randint(1, total - 1)
    choices = random.sample(range(total), k)
    arr = np.full((h, w), color_a, dtype=int)
    for idx in choices:
        arr[idx // w, idx % w] = color_b
    return arr


def _cell_origin(i: int, j: int, row_step: int, col_step: int) -> tuple[int, int]:
    return i * row_step + 1, j * col_step + 1


def _occupancy_from_source_subset(
    top: int,
    left: int,
    subset: tuple[tuple[int, int], ...],
    scale_r: int,
    scale_c: int,
) -> np.ndarray:
    """Convert a subset of active source pixels into a cell-level minority occupancy grid."""
    existing = np.zeros((CELL_GRID_ROWS, CELL_GRID_COLS), dtype=bool)
    for sr, sc in subset:
        r0 = top + sr * scale_r
        c0 = left + sc * scale_c
        existing[r0 : r0 + scale_r, c0 : c0 + scale_c] = True
    return existing


def _placement_candidates(
    expanded_mask: np.ndarray,
    subset: tuple[tuple[int, int], ...],
    top: int,
    left: int,
    scale_r: int,
    scale_c: int,
) -> list[tuple[int, int]]:
    """Find all placements where the chosen complete source-block subset fits."""
    existing = _occupancy_from_source_subset(top, left, subset, scale_r, scale_c)
    big_h, big_w = expanded_mask.shape
    candidates: list[tuple[int, int]] = []
    for t in range(CELL_GRID_ROWS - big_h + 1):
        for l in range(CELL_GRID_COLS - big_w + 1):
            block = np.zeros((CELL_GRID_ROWS, CELL_GRID_COLS), dtype=bool)
            block[t : t + big_h, l : l + big_w] = True
            if np.any(existing & ~block):
                continue
            ex_block = existing[t : t + big_h, l : l + big_w]
            if np.any(ex_block & ~expanded_mask):
                continue
            candidates.append((t, l))
    return candidates


def _choose_given_subset(
    source_mask: np.ndarray, expanded_mask: np.ndarray, top: int, left: int, scale_r: int, scale_c: int
) -> tuple[tuple[int, int], ...] | None:
    """Choose active source pixels whose full stretched blocks uniquely identify the placement."""
    active = [tuple(int(v) for v in coord) for coord in np.argwhere(source_mask)]
    sizes = list(GIVEN_SOURCE_BLOCKS)
    random.shuffle(sizes)
    for size in sizes:
        valid = [
            subset
            for subset in combinations(active, size)
            if _placement_candidates(expanded_mask, subset, top, left, scale_r, scale_c) == [(top, left)]
        ]
        if valid:
            return random.choice(valid)
    return None


def _build_grid(
    row_step: int,
    col_step: int,
    separator: int,
    majority: np.ndarray,
    minority: np.ndarray,
    active_cells: np.ndarray,
) -> np.ndarray:
    """Fill the full grid with majority cells, then overwrite active cell positions with minority cells."""
    grid = np.full(
        (CELL_GRID_ROWS * row_step + 1, CELL_GRID_COLS * col_step + 1), separator, dtype=int
    )
    for i in range(CELL_GRID_ROWS):
        for j in range(CELL_GRID_COLS):
            paste(grid, majority, *_cell_origin(i, j, row_step, col_step))
    for i, j in np.argwhere(active_cells):
        paste(grid, minority, *_cell_origin(int(i), int(j), row_step, col_step))
    return grid


def _gen_attempt() -> np.ndarray | None:
    """Generate one rectangular fused instance and verify it against the solver."""
    src_h, src_w = random.choice(SOURCE_SHAPES)
    row_step = src_h + 1
    col_step = src_w + 1
    scale_r = src_h
    scale_c = src_w

    separator, maj_exclusive, shared, min_exclusive = random.sample(range(COLOR_POOL), 4)
    majority = _random_pattern(src_h, src_w, shared, maj_exclusive)
    source_mask = random.choice(MASKS_BY_SHAPE[(src_h, src_w)])
    minority = np.where(source_mask, min_exclusive, shared).astype(int)
    if np.array_equal(majority, minority):
        return None

    expanded_mask = np.kron(source_mask, np.ones((scale_r, scale_c), dtype=bool))
    big_h, big_w = expanded_mask.shape
    top = random.randrange(CELL_GRID_ROWS - big_h + 1)
    left = random.randrange(CELL_GRID_COLS - big_w + 1)

    active_out = np.zeros((CELL_GRID_ROWS, CELL_GRID_COLS), dtype=bool)
    active_out[top : top + big_h, left : left + big_w] = expanded_mask
    out = _build_grid(row_step, col_step, separator, majority, minority, active_out)

    subset = _choose_given_subset(source_mask, expanded_mask, top, left, scale_r, scale_c)
    if subset is None:
        return None
    active_in = _occupancy_from_source_subset(top, left, subset, scale_r, scale_c)
    inp = _build_grid(row_step, col_step, separator, majority, minority, active_in)

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
    raise RuntimeError("failed to generate a valid rectangular fused b99e7126 input")
