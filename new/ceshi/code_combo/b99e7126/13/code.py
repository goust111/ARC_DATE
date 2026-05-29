# Combo variant of b99e7126 (index 13)
# combo_spec: D7/0+D10/2
# Name: 多异常顺序处理 + 多异常独立图案

# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - The canvas keeps the original 7x7 coarse grid of cells, with 3x3 interiors and
#      1-pixel divider lines.
#    - Original task: one minority pattern appears in several cells; that one pattern's
#      own 3x3 pixels describe a single coarse 3x3 block that must be completed.
#    - Combo task: the input instead shows 2-3 singleton anomaly cells. Each anomaly cell
#      has its own 3x3 interior pattern, and these patterns are all different.
#    - Each anomaly cell is treated as the center of its own 3x3 coarse block. Its more
#      frequent interior color marks active positions, so the seed cell's picture is
#      enlarged to cell-grid scale around that seed.
#    - Those enlargements are not merged simultaneously. They are applied in row-major
#      order of the seed cells: first top-left, then later seeds farther down/right.
#      Because later writes replace earlier ones cell-by-cell, overlapping blocks can
#      visibly erase or overwrite earlier expansions.
#    - Small trace:
#      original: detect one odd pattern -> infer one global placement -> fill that one
#      coarse block.
#      combo: detect all singleton odd patterns -> sort them by position -> expand seed A
#      -> expand seed B over the result -> possibly expand seed C over both.
#      The important difference is that anomaly identity is local per seed, and write
#      order is part of the rule rather than an implementation detail.
# b) Complexity calibration
#    - Using exactly 2-3 seeds is enough to make multi-object reasoning visible without
#      making the grid unreadable.
#    - All seeds share the same visual grammar: a 3x3 two-color motif whose majority color
#      is the active stencil. That keeps the task coherent across examples.
#    - Seed masks are restricted to 5-7 active pixels, active center, connected support,
#      and full edge span. This prevents tiny or ambiguous motifs.
#    - Every generated instance has at least one order-sensitive overlap, so the D10
#      sequential-processing change is not cosmetic.
# c) Hidden assumptions from the original solver that must change
#    - `minority_pat = min(...)` assumed one globally rare pattern. That breaks because
#      several different singleton patterns are now all rare.
#    - Matching all anomaly cells to one shared mask no longer makes sense; each seed has
#      its own mask and its own local anchor.
#    - The original task's placement search disappears. The seed's cell position itself is
#      the anchor of its enlarged block.
#    - The D7 single-variant solver assumed expanded blocks should be disjoint. That
#      assumption must be removed here because D10 explicitly makes later overlays able to
#      cover earlier ones.
#    - The final write loop now depends on sort order. If seed order is not fixed, the
#      output is not well-defined.
# d) Placement legality
#    - Seed centers are chosen only from coarse coordinates 1..5 so each 3x3 enlarged
#      block fits fully inside the 7x7 cell grid.
#    - Center combinations are pre-filtered so at least one overlap exists, but no coarse
#      cell is covered by more than two blocks. This keeps overlays legible.
#    - Seeds are distinct cells with distinct masks, so the input visibly contains 2-3
#      different anomalies.
# e) Visual observability
#    - A human can spot the majority pattern and then the 2-3 singleton odd cells.
#    - Each odd cell clearly carries its own expansion blueprint because its active color is
#      the dominant color inside that 3x3 motif.
#    - Order is recoverable from geometry alone: process seeds from top-left to bottom-right.
#      No hidden dependence on numeric color ids is used.
#    - Overlap is visible in the output because later seed motifs replace part of earlier
#      enlarged regions.
# f) Visual cleanliness
#    - The background of majority cells stays uniform, so each expansion remains readable.
#    - Overlap depth is capped at two layers to avoid chaotic triple-stacked regions.
#    - Using one shared anomaly color family keeps the image compact instead of exploding
#      the palette.
# g) Change inferability
#    - Colors stay at four roles: divider, majority-exclusive, shared, anomaly-exclusive.
#    - Different seeds are distinguished by shape, not by arbitrary recoloring.
#    - The generator rejects cases where reversing the seed order produces the same coarse
#      result, so the sequential overlay rule is genuinely active in the sample set.
# 9) New solver assumptions and generator guarantees
#    - Solver assumes the majority cell pattern is the unique most frequent pattern.
#      Generator fills all non-seed cells with that majority.
#    - Solver assumes every non-majority pattern appears exactly once in the input.
#      Generator uses 2-3 singleton seeds and distinct masks.
#    - Solver assumes each seed's dominant color defines its active stencil and that the
#      stencil center is active. Generator constructs every seed from such a mask.
#    - Solver assumes row-major processing order. Generator guarantees that changing that
#      order would alter the result, so the order is meaningful rather than redundant.

from functools import reduce
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


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _extract_patterns(g: np.ndarray, step: int) -> tuple[dict[tuple[int, ...], np.ndarray], dict[tuple[int, ...], list[tuple[int, int]]]]:
    inner = step - 1
    n = (g.shape[0] - 1) // step
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    positions: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            key = tuple(arr.ravel().tolist())
            if key not in patterns:
                patterns[key] = arr
            positions.setdefault(key, []).append((i, j))
    return patterns, positions


def _active_mask(arr: np.ndarray) -> np.ndarray:
    vals, cnts = np.unique(arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    return arr == active_color


def _apply_seed_block(grid: np.ndarray, majority: np.ndarray, seed_arr: np.ndarray, center: tuple[int, int]) -> None:
    radius = seed_arr.shape[0] // 2
    mask = _active_mask(seed_arr)
    ci, cj = center
    for di in range(seed_arr.shape[0]):
        for dj in range(seed_arr.shape[1]):
            target = seed_arr if mask[di, dj] else majority
            paste(grid, target, *_cell_origin(ci + di - radius, cj + dj - radius))


def _simulate_label_overlay(seed_specs: list[tuple[tuple[int, int], np.ndarray]], order: list[int]) -> np.ndarray:
    labels = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=int)
    radius = CELL_INNER // 2
    for idx in order:
        center, mask = seed_specs[idx]
        ci, cj = center
        for di in range(CELL_INNER):
            for dj in range(CELL_INNER):
                labels[ci + di - radius, cj + dj - radius] = idx + 1 if mask[di, dj] else 0
    return labels


def _order_sensitive_cells(seed_specs: list[tuple[tuple[int, int], np.ndarray]]) -> int:
    forward = _simulate_label_overlay(seed_specs, list(range(len(seed_specs))))
    reverse = _simulate_label_overlay(seed_specs, list(reversed(range(len(seed_specs)))))
    return int(np.count_nonzero(forward != reverse))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand every singleton anomaly cell into its own local fractal block, processing seeds in order.

    Input structure:
    - The canvas is partitioned into equal cells by full divider rows and columns of one color.
    - One 3x3 cell-interior pattern is the majority pattern and fills almost the whole grid.
    - There are 2-3 singleton anomaly cells, and each anomaly has a different 3x3 interior motif.

    Rule:
    - For each singleton anomaly, take its dominant interior color as the active color and read the
      whole 3x3 cell interior as an active/inactive stencil.
    - Use that stencil as a 3x3 coarse block centered on the anomaly's own cell position.
    - Active coarse cells receive the anomaly pattern; inactive coarse cells are reset to the
      majority pattern.
    - Process all anomaly seeds from top-left to bottom-right, so later enlarged blocks overwrite
      earlier ones wherever they overlap.

    Key insight:
    Each odd cell is now an independent self-similar seed, and the final image depends not only on
    which seeds exist but also on the visible geometric order in which their enlarged blocks are laid down.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    assert inner == CELL_INNER, "generator and solver expect 3x3 cell interiors"

    patterns, positions = _extract_patterns(g, step)
    counts = {pat: len(locs) for pat, locs in positions.items()}
    assert len(counts) >= 3, "need a majority pattern plus 2-3 singleton anomalies"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    seed_specs: list[tuple[tuple[int, int], np.ndarray]] = []
    for pat, cnt in counts.items():
        if pat == majority_pat:
            continue
        assert cnt == 1, "every anomaly pattern must appear exactly once"
        center = positions[pat][0]
        arr = patterns[pat]
        mask = _active_mask(arr)
        assert mask[CELL_INNER // 2, CELL_INNER // 2], "seed center must be active"
        ci, cj = center
        radius = CELL_INNER // 2
        assert radius <= ci < n - radius and radius <= cj < n - radius, "expanded seed block must fit in bounds"
        seed_specs.append((center, arr))

    assert 2 <= len(seed_specs) <= 3, "variant expects 2-3 singleton anomaly seeds"
    seed_specs.sort(key=lambda item: item[0])

    out = g.copy()
    for center, seed_arr in seed_specs:
        _apply_seed_block(out, majority_arr, seed_arr, center)
    return out


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
SEED_COUNTS = (2, 3)
ACTIVE_PIXEL_COUNTS = (5, 6, 7)
COLOR_POOL = 10
MAX_RETRIES = 400
MIN_ORDER_SENSITIVE_CELLS = 2

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]

_VALID_CENTERS = [(r, c) for r in range(1, CELL_GRID_N - 1) for c in range(1, CELL_GRID_N - 1)]


def _precompute_seed_masks() -> list[np.ndarray]:
    """Enumerate readable 3x3 masks with active center, connected shape, and edge span."""
    masks: list[np.ndarray] = []
    for bits in range(1 << (CELL_INNER * CELL_INNER)):
        mask = np.array([(bits >> k) & 1 for k in range(CELL_INNER * CELL_INNER)], dtype=bool).reshape(CELL_INNER, CELL_INNER)
        active = int(mask.sum())
        if active not in ACTIVE_PIXEL_COUNTS:
            continue
        if not mask[1, 1]:
            continue
        if not mask[0, :].any() or not mask[2, :].any() or not mask[:, 0].any() or not mask[:, 2].any():
            continue

        coords = [tuple(x) for x in np.argwhere(mask)]
        seen = {coords[0]}
        frontier = [coords[0]]
        while frontier:
            r, c = frontier.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < CELL_INNER and 0 <= nc < CELL_INNER and mask[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    frontier.append((nr, nc))
        if len(seen) != active:
            continue
        masks.append(mask)
    return masks


def _block_mask(center: tuple[int, int]) -> np.ndarray:
    radius = CELL_INNER // 2
    ci, cj = center
    mask = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=bool)
    mask[ci - radius : ci + radius + 1, cj - radius : cj + radius + 1] = True
    return mask


def _precompute_center_sets() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    """Choose center tuples with some overlap but without triple-covered coarse cells."""
    blocks = {center: _block_mask(center) for center in _VALID_CENTERS}
    by_count: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for count in SEED_COUNTS:
        valid: list[tuple[tuple[int, int], ...]] = []
        for combo in combinations(_VALID_CENTERS, count):
            coverage = np.zeros((CELL_GRID_N, CELL_GRID_N), dtype=int)
            for center in combo:
                coverage += blocks[center].astype(int)
            if coverage.max() > 2:
                continue
            if coverage.max() < 2:
                continue
            valid.append(combo)
        by_count[count] = valid
    return by_count


_SEED_MASKS = _precompute_seed_masks()
_CENTER_SETS = _precompute_center_sets()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    """Generate a non-uniform LR-symmetric 3x3 majority pattern with two colors."""
    arr = np.empty((CELL_INNER, CELL_INNER), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _build_base_grid(separator: int, majority: np.ndarray) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    return grid


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, seed_color = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)

    seed_count = random.choice(SEED_COUNTS)
    centers = random.choice(_CENTER_SETS[seed_count])
    masks = random.sample(_SEED_MASKS, seed_count)
    seed_patterns = [np.where(mask, seed_color, shared).astype(int) for mask in masks]
    seed_specs = list(zip(centers, masks))

    if _order_sensitive_cells(seed_specs) < MIN_ORDER_SENSITIVE_CELLS:
        return None

    out = _build_base_grid(separator, majority)
    for center, seed_arr in sorted(zip(centers, seed_patterns), key=lambda item: item[0]):
        _apply_seed_block(out, majority, seed_arr, center)

    inp = _build_base_grid(separator, majority)
    for center, seed_arr in zip(centers, seed_patterns):
        paste(inp, seed_arr, *_cell_origin(*center))

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
