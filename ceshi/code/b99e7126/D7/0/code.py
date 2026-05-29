# Variant of b99e7126
# Dimension: D7 — 异常检测：异常单元格的数量与分布 (src)
# Change: 恰好有1个异常单元格，其余全部相同... → 多个异常单元格：每个包含不同图案

# Design analysis for the D7 variant
# a) Concrete behavior and trace
#    - The canvas stays a 7x7 grid of 3x3 cell interiors separated by 1-pixel dividers.
#    - One majority cell pattern still fills almost every cell.
#    - The input now contains 2-3 singleton anomaly cells. Each anomaly has its own 3x3
#      interior pattern, and each of those patterns is different from the others.
#    - Each anomaly cell acts as the center of its own missing 3x3 coarse block. Its
#      interior 3x3 pixels are read as a boolean stencil: the more frequent color is
#      "active", the less frequent color is "inactive". In the output, that stencil is
#      scaled to cell-grid size around the seed cell itself.
#    - Small trace:
#      original logic: one minority pattern appears in several cells -> solver searches
#      for a single global placement of that pattern's 3x3 mask -> fills the remaining
#      active cells of that one block.
#      variant logic: three different singleton patterns appear once each -> solver takes
#      each singleton's own location as the anchor -> expands three independent 3x3 blocks
#      centered on those three seeds -> overlays the results.
#      difference: placement is now local per seed instead of global for one repeated
#      minority type, because D7 changes the number/distribution of anomalies.
# b) Complexity calibration
#    - Keeping exactly 2-3 seeds prevents regression to the original single-anomaly task.
#    - Each seed uses a distinct mask but the same general visual grammar (3x3, active
#      majority inside the seed), so humans can see the repeated transformation rather than
#      memorizing unrelated cases.
#    - Masks are restricted to 5-7 active pixels, center active, connected, and spanning
#      all four edges. That keeps them readable and avoids tiny degenerate differences.
# c) Hidden assumptions from the original solver that must change
#    - `minority_pat = min(counts.items(), key=lambda kv: kv[1])[0]` assumed a single
#      globally least-frequent pattern. This fails once several distinct patterns each
#      appear once.
#    - `existing[i, j] = cell_pat[i][j] == minority_pat` assumed all interesting cells
#      share one pattern. This is false in the variant.
#    - The unique `(top, left)` search assumed multiple observed cells jointly identify one
#      global placement. With only singleton anomalies, that search becomes ambiguous.
#    - The final write loop assumed every active output cell receives the same minority
#      pattern. The variant instead needs per-seed patterns.
# d) Placement legality
#    - Every seed is placed on coarse coordinates 1..5 so its centered 3x3 block fits
#      entirely inside the 7x7 cell grid.
#    - Different seeds are generated with pairwise non-overlapping 3x3 coarse blocks. This
#      prevents conflicting writes and keeps the output visually clean.
# e) Visual observability
#    - The rule is visible from 2-4 examples because each input shows several different odd
#      cells, and each output grows each odd cell into a local 3x3 arrangement using the
#      same motif. No ordering depends on hidden color ids.
# f) Visual cleanliness
#    - Disjoint 3x3 blocks avoid clutter from stacked overlays.
#    - The majority pattern is consistent everywhere else, so each expansion reads as a
#      clear local operation.
# g) Color inferability
#    - Total colors remain small: divider, majority-exclusive, shared, and one anomaly
#      active color. Different anomalies are distinguished by shape, not by arbitrary
#      per-instance recoloring.
# 9) New solver assumptions and generator guarantees
#    - Solver assumes the majority pattern is the unique most frequent cell pattern.
#      Generator fills all non-seed cells with that majority.
#    - Solver assumes every non-majority pattern appears exactly once and has its center
#      pixel active. Generator enforces both.
#    - Solver assumes every seed-centered 3x3 coarse block lies in bounds and does not
#      overlap another such block. Generator only samples centers satisfying those rules.

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Enlarge every singleton anomaly cell's inner motif into its own local fractal block.

    Input structure:
    - The grid is partitioned into equal cells by full divider rows/columns of one color.
    - One cell interior pattern is the overwhelming majority pattern.
    - There are 2-3 singleton anomaly cells, each with a different 3x3 interior pattern.

    Rule:
    - For each singleton pattern, treat its most frequent interior color as the active color.
    - Use that 3x3 active/inactive stencil at cell-grid scale, centered on the singleton's
      own location.
    - Active coarse positions receive that singleton pattern; inactive coarse positions keep
      the majority pattern.
    - Apply this independently to every singleton anomaly cell and overlay the disjoint
      results onto the original grid.

    Key insight:
    The odd cells are no longer multiple sightings of one global exception. Each odd cell is
    now its own self-encoded blueprint, so the task is to detect every singleton seed and
    expand each one locally.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    assert inner % 2 == 1, "seed-centered expansion needs an odd interior size"
    radius = inner // 2

    patterns: dict[tuple[int, ...], np.ndarray] = {}
    cell_pat: list[list[tuple[int, ...]]] = []
    positions: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for i in range(n):
        row: list[tuple[int, ...]] = []
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            pat_arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(pat_arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = pat_arr
            positions.setdefault(pat, []).append((i, j))
            row.append(pat)
        cell_pat.append(row)

    counts = {pat: len(pos_list) for pat, pos_list in positions.items()}
    assert len(counts) >= 3, "need a majority pattern plus multiple singleton anomalies"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    seed_pats = [pat for pat, cnt in counts.items() if pat != majority_pat]
    assert 2 <= len(seed_pats) <= 3, "variant expects 2-3 distinct anomaly seeds"
    assert all(counts[pat] == 1 for pat in seed_pats), "every anomaly pattern must be a singleton"

    seed_specs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    occupied = np.zeros((n, n), dtype=bool)
    for pat in seed_pats:
        (ci, cj) = positions[pat][0]
        assert radius <= ci < n - radius and radius <= cj < n - radius, "seed block must fit in bounds"
        arr = patterns[pat]
        vals, cnts = np.unique(arr, return_counts=True)
        active_color = int(vals[cnts.argmax()])
        mask = arr == active_color
        assert mask[radius, radius], "seed cell must correspond to an active center pixel"

        block = np.zeros((n, n), dtype=bool)
        block[ci - radius : ci + radius + 1, cj - radius : cj + radius + 1] = True
        assert not np.any(occupied & block), "expanded seed blocks must not overlap"
        occupied |= block
        seed_specs.append((ci, cj, arr, mask))

    out = g.copy()
    for ci, cj, seed_arr, mask in seed_specs:
        for di in range(inner):
            for dj in range(inner):
                target_pat = seed_arr if mask[di, dj] else majority_arr
                r0 = (ci + di - radius) * step + 1
                c0 = (cj + dj - radius) * step + 1
                out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
SEED_COUNTS = (2, 3)
ACTIVE_PIXEL_COUNTS = (5, 6, 7)
COLOR_POOL = 10
MAX_RETRIES = 300

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
    """Enumerate readable 3x3 seed masks with an active center and full 3x3 span."""
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


_SEED_MASKS = _precompute_seed_masks()


def _precompute_center_sets() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    """All seed-center choices whose 3x3 expanded blocks are pairwise disjoint."""
    by_count: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for count in SEED_COUNTS:
        valid = []
        for combo in combinations(_VALID_CENTERS, count):
            ok = True
            for (r1, c1), (r2, c2) in combinations(combo, 2):
                if abs(r1 - r2) < CELL_INNER and abs(c1 - c2) < CELL_INNER:
                    ok = False
                    break
            if ok:
                valid.append(combo)
        by_count[count] = valid
    return by_count


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


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_base_grid(separator: int, majority: np.ndarray) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    return grid


def _apply_seed_block(grid: np.ndarray, majority: np.ndarray, seed_arr: np.ndarray, center: tuple[int, int]) -> None:
    ci, cj = center
    vals, cnts = np.unique(seed_arr, return_counts=True)
    active_color = int(vals[cnts.argmax()])
    mask = seed_arr == active_color
    radius = CELL_INNER // 2
    for di in range(CELL_INNER):
        for dj in range(CELL_INNER):
            target = seed_arr if mask[di, dj] else majority
            paste(grid, target, *_cell_origin(ci + di - radius, cj + dj - radius))


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, seed_color = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)

    seed_count = random.choice(SEED_COUNTS)
    centers = random.choice(_CENTER_SETS[seed_count])
    masks = random.sample(_SEED_MASKS, seed_count)
    seed_patterns = [np.where(mask, seed_color, shared).astype(int) for mask in masks]

    out = _build_base_grid(separator, majority)
    for center, seed_arr in zip(centers, seed_patterns):
        _apply_seed_block(out, majority, seed_arr, center)

    inp = _build_base_grid(separator, majority)
    for (ci, cj), seed_arr in zip(centers, seed_patterns):
        paste(inp, seed_arr, *_cell_origin(ci, cj))

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
    raise RuntimeError("failed to generate a valid b99e7126 D7 input")
