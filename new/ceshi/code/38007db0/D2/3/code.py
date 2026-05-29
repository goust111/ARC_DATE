# Variant of 38007db0
# Dimension: D2 — 源对象：行内唯一精灵单元格 (src)
# Change: 每行中模式只出现一次的方形单元格（内含独特精灵变体）... → 对称性缺失检测

from __future__ import annotations

from collections import Counter
import math
import numpy as np

from arc_utils import paste

"""
Design analysis for the D2 variant
==================================

a) Concrete behavior and execution trace
---------------------------------------
- The lattice, colors, cell sizing, and output format stay the same as the seed task.
- The changed part is the row-level selection rule:
  every non-target cell now contains a left-right mirror-symmetric sprite, while the
  extracted cell is the only one in that row whose sprite breaks that symmetry.
- To make the change visible, rows no longer rely on exact pattern repetition.
  The symmetric cells in a row are distinct symmetric variants from the same base family.
  Example row with 4 cells:
    old task row: [A, A, X, A] where X is the only byte-unique cell
    new task row: [S1, S2, X, S3] where S1/S2/S3 are all different but mirror-symmetric
                   and X is the only non-symmetric cell.
- Therefore the old "count exact pattern frequency, pick the singleton" logic no longer
  works: it would see several byte-unique cells. The correct behavior is to score each cell
  by left-right mirror mismatch and extract the only cell with a non-zero score.

b) Complexity calibration
-------------------------
- The symmetry rule is strong and local: one fixed symmetry type (left-right reflection)
  is reused for every sample in a task, so the rule is inferable from 2-4 examples.
- To avoid over-simplification, the symmetric distractors are not all copies. This prevents
  degeneration back to the original exact-singleton rule.
- To avoid excessive freedom, the palette is still exactly 3 colors and the sprite family
  still comes from one base symmetric inner pattern with small orbit toggles.

c) Old main() assumptions that break here
-----------------------------------------
- `sigs = [cell.tobytes() ...]` assumed the target was defined by exact cell identity.
  That is no longer valid because multiple cells can now be byte-unique while still being
  legitimate symmetric distractors.
- `cnt[s] == 1` assumed "unique pattern" and "desired pattern" are equivalent. In this
  variant they are intentionally different; only symmetry breaking identifies the source.
- Gridline discovery, lattice color recovery, background estimation, and output assembly do
  not depend on exact repetition, so those parts can stay unchanged.

d) Placement legality
---------------------
- The lattice builder is unchanged, so cells never overlap and always sit inside a regular
  1-pixel grid.
- Symmetric variants are created only by toggling cells inside the `(cell_size-2)^2` inner
  sprite area; the 1-pixel background padding remains intact.
- The odd cell is made by toggling only one side of a mirrored pair, guaranteeing that the
  cell stays in-bounds and becomes non-symmetric.

e) Visual observability
-----------------------
- Human solvers can compare each cell to its mirror image and see that exactly one sprite
  per row fails the reflection rule.
- Because symmetric distractors vary, the output cannot be explained by "pick the different
  picture"; the visible invariant is specifically mirror symmetry.

f) Visual cleanliness
---------------------
- Only three colors are used and there is still just one sprite per cell with background
  padding, so rows remain easy to parse.
- No extra marks or overlays are introduced.

g) Color-rule audit
-------------------
- No color remapping is used. Colors only label lattice, background, and sprite foreground.
  The rule is purely geometric, which is easier to infer from few examples.

h) Variant behavior metric used after implementation
----------------------------------------------------
- Degeneration check: compare this solver to the seed task's first-byte-singleton rule.
  Generated rows are built so there are multiple byte-unique cells, and the odd asymmetric
  cell is not in the first column. The old rule should therefore disagree with the new rule
  on nearly every sample.

i) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: exactly one cell per row has non-zero left-right mirror mismatch.
  Generator guarantee: all non-target cells are symmetric; the target is created by a
  one-sided toggle in a mirrored pair, so it always has positive mismatch.
- Assumption: the lattice is rectangular with uniform 1-pixel gridlines.
  Generator guarantee: `_build_lattice` constructs exactly that structure.
- Assumption: a single background color dominates non-lattice pixels.
  Generator guarantee: cells use one shared background color with a padded border.
"""


def _lr_mismatch_score(cell: np.ndarray) -> int:
    """Count left-right mirror disagreements; symmetric cells score 0."""
    return int(np.count_nonzero(cell != np.fliplr(cell)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the only left-right non-symmetric cell from each lattice row.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    row_lines = [r for r in range(H) if len(set(map(int, g[r, :]))) == 1]
    col_lines = [c for c in range(W) if len(set(map(int, g[:, c]))) == 1]
    assert row_lines and col_lines

    vals = [int(g[r, 0]) for r in row_lines] + [int(g[0, c]) for c in col_lines]
    L = int(Counter(vals).most_common(1)[0][0])
    row_lines = sorted([r for r in row_lines if int(g[r, 0]) == int(L)])
    col_lines = sorted([c for c in col_lines if int(g[0, c]) == int(L)])

    r_slices = [(int(row_lines[i]) + 1, int(row_lines[i + 1])) for i in range(len(row_lines) - 1)]
    c_slices = [(int(col_lines[j]) + 1, int(col_lines[j + 1])) for j in range(len(col_lines) - 1)]
    r_slices = [(a, b) for a, b in r_slices if int(b) > int(a)]
    c_slices = [(a, b) for a, b in c_slices if int(b) > int(a)]
    assert r_slices and c_slices

    cell_w = int(c_slices[0][1] - c_slices[0][0])

    bg_candidates = [int(x) for x in g.flatten() if int(x) != int(L)]
    assert bg_candidates
    bg = int(Counter(bg_candidates).most_common(1)[0][0])

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        scores = [_lr_mismatch_score(cell) for cell in cells]
        odd_idxs = [j for j, score in enumerate(scores) if int(score) > 0]
        assert len(odd_idxs) == 1
        chosen_cells.append(cells[int(odd_idxs[0])])

    outW = int(cell_w + 2)
    out = np.full((H, outW), int(bg), dtype=int)
    out[:, 0] = int(L)
    out[:, -1] = int(L)
    for r in row_lines:
        out[int(r), :] = int(L)

    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_w)] = chosen_cells[int(i)]

    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = inner.shape[0] + 2
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_lattice(cells: list[list[np.ndarray]], lattice: int, bg: int) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = cells[0][0].shape[0]
    grid = np.full((rows * (cell_size + 1) + 1, cols * (cell_size + 1) + 1), int(bg), dtype=int)
    for r in range(rows + 1):
        grid[r * (cell_size + 1), :] = int(lattice)
    for c in range(cols + 1):
        grid[:, c * (cell_size + 1)] = int(lattice)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, 1 + r * (cell_size + 1), 1 + c * (cell_size + 1))
    return grid


def _mirror_orbits(inner_size: int) -> tuple[list[tuple[tuple[int, int], ...]], list[tuple[tuple[int, int], tuple[int, int]]]]:
    all_orbits: list[tuple[tuple[int, int], ...]] = []
    pair_orbits: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(inner_size):
        for c in range((inner_size + 1) // 2):
            mc = inner_size - 1 - c
            if c == mc:
                all_orbits.append(((r, c),))
            else:
                orb = ((r, c), (r, mc))
                all_orbits.append(orb)
                pair_orbits.append(orb)
    return all_orbits, pair_orbits


def _apply_orbit_toggles(
    base_inner: np.ndarray,
    orbits: list[tuple[tuple[int, int], ...]],
    toggle_ids: tuple[int, ...],
    bg: int,
    sp: int,
) -> np.ndarray:
    inner = base_inner.copy()
    for orb_id in toggle_ids:
        for r, c in orbits[int(orb_id)]:
            inner[int(r), int(c)] = int(sp) if int(inner[int(r), int(c)]) == int(bg) else int(bg)
    return inner


def _rand_subset_ids(rng: np.random.Generator, count: int) -> tuple[int, ...]:
    max_k = min(3, count)
    k = int(rng.integers(0, max_k + 1))
    if k == 0:
        return ()
    ids = sorted(int(x) for x in rng.choice(count, size=k, replace=False))
    return tuple(ids)


def _gen_symmetric_base(
    rng: np.random.Generator,
    inner_size: int,
    bg: int,
    sp: int,
    orbits: list[tuple[tuple[int, int], ...]],
) -> np.ndarray | None:
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.9 * inner_area)
    for _ in range(200):
        inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        for orb in orbits:
            if rng.random() < 0.55:
                for r, c in orb:
                    inner[int(r), int(c)] = int(sp)
        fill = int(np.count_nonzero(inner == int(sp)))
        if min_fill <= fill <= max_fill and 0 < fill < inner_area:
            return inner
    return None


def _collect_symmetric_variants(
    rng: np.random.Generator,
    base_inner: np.ndarray,
    orbits: list[tuple[tuple[int, int], ...]],
    bg: int,
    sp: int,
    count: int,
) -> list[np.ndarray] | None:
    variants: list[np.ndarray] = []
    seen: set[bytes] = set()
    base_cell = _make_cell(base_inner, bg)
    variants.append(base_cell)
    seen.add(base_cell.tobytes())

    for _ in range(400):
        if len(variants) >= count:
            return variants[:count]
        toggle_ids = _rand_subset_ids(rng, len(orbits))
        inner = _apply_orbit_toggles(base_inner, orbits, toggle_ids, bg, sp)
        cell = _make_cell(inner, bg)
        sig = cell.tobytes()
        if sig in seen or _lr_mismatch_score(cell) != 0:
            continue
        seen.add(sig)
        variants.append(cell)
    return None


def _break_symmetry(
    rng: np.random.Generator,
    sym_cell: np.ndarray,
    pair_orbits: list[tuple[tuple[int, int], tuple[int, int]]],
    bg: int,
    sp: int,
    used: set[bytes],
) -> np.ndarray | None:
    inner = sym_cell[1:-1, 1:-1]
    order = list(rng.permutation(len(pair_orbits)))
    for orb_idx in order:
        orb = pair_orbits[int(orb_idx)]
        side = int(rng.integers(0, 2))
        r, c = orb[side]
        broken = inner.copy()
        broken[int(r), int(c)] = int(sp) if int(broken[int(r), int(c)]) == int(bg) else int(bg)
        cell = _make_cell(broken, bg)
        sig = cell.tobytes()
        if sig in used or _lr_mismatch_score(cell) == 0:
            continue
        return cell
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    lattice, bg, sp = map(int, rng.choice(10, size=3, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(4, 6))

    inner_size = cell_size - 2
    orbits, pair_orbits = _mirror_orbits(inner_size)
    if not pair_orbits:
        return None

    base_inner = _gen_symmetric_base(rng, inner_size, bg, sp, orbits)
    if base_inner is None:
        return None

    unique_cells: list[np.ndarray] = []
    used_unique: set[bytes] = set()
    in_cells: list[list[np.ndarray]] = []

    for _ in range(n_rows):
        sym_cells = _collect_symmetric_variants(rng, base_inner, orbits, bg, sp, n_cols - 1)
        if sym_cells is None:
            return None
        anchor = sym_cells[int(rng.integers(0, len(sym_cells)))]
        odd = _break_symmetry(rng, anchor, pair_orbits, bg, sp, used_unique)
        if odd is None:
            return None
        used_unique.add(odd.tobytes())
        unique_cells.append(odd.copy())

        row_cells = [cell.copy() for cell in sym_cells]
        odd_col = int(rng.integers(1, n_cols))
        row_cells.insert(odd_col, odd.copy())
        in_cells.append(row_cells)

    out = _build_lattice([[cell] for cell in unique_cells], lattice, bg)
    inp = _build_lattice(in_cells, lattice, bg)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
