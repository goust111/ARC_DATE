# Combo variant of b99e7126 (index 40)
# combo_spec: D3/3+D7/0+D10/2
# Name: 多次叠加 + 多异常独立图案 + 顺序处理
#
# Design analysis
# a) Concrete behavior and trace
#    - The canvas is still a 29x29 pixel grid made from a 7x7 grid of coarse cells.
#      Each coarse cell has a 3x3 interior and is separated by one divider row/column.
#    - One majority cell pattern fills almost the entire input.
#    - The input now shows exactly two singleton anomaly cells, and the two anomaly
#      patterns are different. Each anomaly pattern is its own 3x3 seed blueprint.
#    - For one seed at coarse center (1,2), suppose its active mask is:
#          1 1 0
#          1 1 1
#          0 1 1
#      The seed is interpreted as a centered 3x3 coarse block, then that same block is
#      repeated vertically at every center with the same row parity and the same column:
#      centers (1,2), (3,2), (5,2). The active coarse cells from those three instances are
#      unioned, giving one repeated vertical band.
#    - A second seed at (3,4) with a different mask creates another repeated band. The
#      solver applies the top-left seed first, then the lower/right seed second. If both
#      bands claim the same coarse cell, the second seed's cell pattern becomes the final
#      winner at that cell.
#    - Compared with the original task:
#      original: one minority pattern, one inferred enlarged placement, one fill step.
#      fused: two different seed patterns, each enlarged into many placements, then applied
#      sequentially with visible overwrite conflicts.
# b) Complexity calibration
#    - Exactly two seeds are used instead of 2-3. That is still genuinely "multiple
#      anomalies", but it keeps the overlap logic readable and avoids output clutter.
#    - Each seed repeats only along one vertical parity class, so humans can see the
#      repeated-placement rule without facing an explosion of candidate locations.
#    - Seed columns differ by 1-2 cells, which guarantees spatial interaction while keeping
#      the output as two understandable overlapping bands rather than a full-grid tangle.
# c) Hidden assumptions that break from the original solver
#    - The original `minority_pat = min(...)` assumes one global minority pattern. That is
#      false here: there are two distinct singleton anomaly patterns.
#    - The original global placement search assumes several visible minority cells all come
#      from one enlarged instance. That is false here: each singleton seed has its own local
#      anchor and its own repeated class.
#    - The original final paint loop assumes one minority texture fills all active cells.
#      That is false here: different seed bands write different cell patterns, and row-major
#      processing order resolves conflicts.
#    - The original "inactive pixels write majority" behavior inside a repeated family would
#      let a later seed erase earlier active results for reasons invisible to the viewer.
#      This fused version instead treats each repeated family as the union of its active
#      coarse cells, and only those claimed cells are painted. Order still matters wherever
#      two different families both claim the same coarse cell.
# d) Placement legality
#    - Seed centers are chosen from coarse rows 1..5 and cols 1..5, so every centered 3x3
#      block stays in bounds.
#    - Both seeds share the same row parity and have column distance 1 or 2. Their repeated
#      bands therefore overlap, but no repeated placement leaves the 7x7 coarse grid.
#    - The generator is output-first and rejects any sample whose forward-order and
#      reverse-order outputs are identical, so overwrite order is guaranteed to matter.
# e) Visual observability
#    - In every sample, the input clearly shows two different odd cells.
#    - The output clearly shows that each odd cell expands into a repeated vertical band.
#    - The final overlap winner is explained by a visible spatial rule: process seeds from
#      top-left to bottom-right. No color id ordering is used.
# f) Cleanliness
#    - The output contains two repeated bands instead of arbitrary full-grid repetition.
#      This is dense enough to show D3 and D10, but still easy to parse.
#    - Only four colors are used: divider, majority-exclusive, shared, and one anomaly
#      color shared by both seeds. Distinct anomaly identities come from shape, not from a
#      large color vocabulary.
# g) New solver assumptions and generator guarantees
#    - Solver assumes the majority pattern is the unique most frequent interior pattern.
#      Generator fills every non-seed input cell with that majority pattern.
#    - Solver assumes every anomaly pattern is a singleton and its center pixel is active.
#      Generator enforces both.
#    - Solver assumes repeated families are ordered by seed position `(row, col)`.
#      Generator always places the two seeds at distinct positions and uses the same order
#      when constructing the output.

from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "solvers" / "arc_utils.py").exists())
_SOLVER_DIR = _ROOT / "solvers"
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


CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
REPEAT_STEP = 2
SEED_COUNT = 2
ACTIVE_PIXEL_COUNTS = (5, 6, 7)
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

_VALID_CENTERS = [(r, c) for r in range(1, CELL_GRID_N - 1) for c in range(1, CELL_GRID_N - 1)]


def _extract_cells(grid: np.ndarray) -> tuple[dict[tuple[int, ...], np.ndarray], dict[tuple[int, ...], list[tuple[int, int]]], int, int]:
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    positions: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            pat = tuple(arr.ravel().tolist())
            if pat not in patterns:
                patterns[pat] = arr
            positions.setdefault(pat, []).append((i, j))
    return patterns, positions, n, inner


def _active_mask(arr: np.ndarray) -> np.ndarray:
    vals, cnts = np.unique(arr, return_counts=True)
    return arr == int(vals[cnts.argmax()])


def _repeat_centers(center_row: int, n: int, radius: int) -> list[int]:
    rows = [r for r in range(radius, n - radius) if r % REPEAT_STEP == center_row % REPEAT_STEP]
    assert rows, "repeat class must contain at least one legal center"
    return rows


def _repeated_active_cells(mask: np.ndarray, center: tuple[int, int], n: int) -> list[tuple[int, int]]:
    radius = mask.shape[0] // 2
    ci, cj = center
    cells: set[tuple[int, int]] = set()
    for rr in _repeat_centers(ci, n, radius):
        for di, dj in np.argwhere(mask):
            cells.add((rr + int(di) - radius, cj + int(dj) - radius))
    return sorted(cells)


def _extract_seed_specs(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int, np.ndarray, list[tuple[int, int]]]], int, int]:
    g = np.asarray(grid, dtype=int)
    patterns, positions, n, inner = _extract_cells(g)
    assert inner % 2 == 1, "seed-centered expansion needs an odd interior size"
    radius = inner // 2

    counts = {pat: len(pos_list) for pat, pos_list in positions.items()}
    assert len(counts) >= 3, "need a majority pattern plus multiple singleton anomalies"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]

    seed_pats = [pat for pat in counts if pat != majority_pat]
    assert 2 <= len(seed_pats) <= 3, "variant expects 2-3 anomaly seeds"
    assert all(counts[pat] == 1 for pat in seed_pats), "every anomaly pattern must be a singleton"

    specs: list[tuple[int, int, np.ndarray, list[tuple[int, int]]]] = []
    for pat in seed_pats:
        (ci, cj) = positions[pat][0]
        assert radius <= ci < n - radius and radius <= cj < n - radius, "seed-centered block must fit in bounds"
        arr = patterns[pat]
        mask = _active_mask(arr)
        assert mask[radius, radius], "seed center must be active"
        specs.append((ci, cj, arr, _repeated_active_cells(mask, (ci, cj), n)))
    specs.sort(key=lambda spec: (spec[0], spec[1]))
    return majority_arr, specs, n, inner


def _render_with_order(base: np.ndarray, specs: list[tuple[int, int, np.ndarray, list[tuple[int, int]]]], step: int, inner: int) -> np.ndarray:
    out = base.copy()
    for _ci, _cj, arr, active_cells in specs:
        for i, j in active_cells:
            r0 = i * step + 1
            c0 = j * step + 1
            out[r0 : r0 + inner, c0 : c0 + inner] = arr
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each singleton anomaly into a repeated band, then overlay those bands in scan order.

    Input structure:
    - The grid is partitioned into equal 3x3-interior cells by one-color divider lines.
    - One cell interior pattern is the clear majority.
    - There are multiple singleton anomaly cells, and each anomaly cell has a different 3x3 interior pattern.

    Rule:
    - For each singleton anomaly, read its most frequent interior color as the active color of a 3x3 mask.
    - Treat the anomaly cell as the center of that 3x3 coarse block, then repeat the same centered block at every
      legal row with the same row parity and the same column.
    - Within one anomaly family, union all active coarse cells across those repeated placements.
    - Process the anomaly families from top-left to bottom-right; each family paints its own pattern onto its active
      coarse cells, so later families overwrite earlier ones wherever both claim the same coarse cell.

    Key insight:
    Each odd cell is both a local seed and a reusable stencil. The puzzle is not only to enlarge every seed, but also
    to notice that the enlargements repeat in bands and that overlap is resolved by a visible processing order.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    majority_arr, specs, n, inner = _extract_seed_specs(g)
    assert specs, "need at least one anomaly seed"

    # The base already contains the majority everywhere except the input singleton seeds.
    # Resetting to majority avoids leaving the input-only seed cells in places not claimed by
    # the repeated families after ordering.
    base = g.copy()
    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            base[r0 : r0 + inner, c0 : c0 + inner] = majority_arr

    return _render_with_order(base, specs, step, inner)


def _precompute_seed_masks() -> list[np.ndarray]:
    """Readable 3x3 masks with active center, connected foreground, and full 3x3 span."""
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

_CENTER_PAIRS = [
    pair
    for pair in combinations(_VALID_CENTERS, SEED_COUNT)
    if pair[0][0] % REPEAT_STEP == pair[1][0] % REPEAT_STEP
    if pair[0][1] < pair[1][1]
    if 1 <= pair[1][1] - pair[0][1] <= 2
]


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


def _seed_pattern(mask: np.ndarray, shared: int, seed_color: int) -> np.ndarray:
    return np.where(mask, seed_color, shared).astype(int)


def _specs_from_centers(centers: tuple[tuple[int, int], ...], patterns: list[np.ndarray]) -> list[tuple[int, int, np.ndarray, list[tuple[int, int]]]]:
    specs: list[tuple[int, int, np.ndarray, list[tuple[int, int]]]] = []
    for center, arr in zip(centers, patterns):
        specs.append((center[0], center[1], arr, _repeated_active_cells(_active_mask(arr), center, CELL_GRID_N)))
    specs.sort(key=lambda spec: (spec[0], spec[1]))
    return specs


def _coarse_claims(specs: list[tuple[int, int, np.ndarray, list[tuple[int, int]]]]) -> dict[tuple[int, int], list[int]]:
    claims: dict[tuple[int, int], list[int]] = {}
    for idx, spec in enumerate(specs):
        for cell in spec[3]:
            claims.setdefault(cell, []).append(idx)
    return claims


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, seed_color = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)
    centers = random.choice(_CENTER_PAIRS)
    masks = random.sample(_SEED_MASKS, SEED_COUNT)
    patterns = [_seed_pattern(mask, shared, seed_color) for mask in masks]
    specs = _specs_from_centers(centers, patterns)

    claims = _coarse_claims(specs)
    contested_cells = [cell for cell, owners in claims.items() if len(owners) >= 2]
    if not contested_cells:
        return None

    out = _render_with_order(_build_base_grid(separator, majority), specs, CELL_STEP, CELL_INNER)
    reverse = _render_with_order(_build_base_grid(separator, majority), list(reversed(specs)), CELL_STEP, CELL_INNER)
    if np.array_equal(out, reverse):
        return None

    inp = _build_base_grid(separator, majority)
    for ci, cj, arr, _active_cells in specs:
        paste(inp, arr, *_cell_origin(ci, cj))

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
