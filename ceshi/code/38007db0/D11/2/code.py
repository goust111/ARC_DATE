# Variant of 38007db0
# Dimension: D11 — 格子线宽度 (meta)
# Change: 格子线宽度为1像素（显式格子线颜色） → 无显式格子线（依赖间距推断）

from __future__ import annotations

from collections import Counter
import itertools
import math

import numpy as np

from arc_utils import paste


# Design analysis for the D11 variant
# -----------------------------------
# a) Concrete behavior:
#    Original input/output had a dedicated lattice color L on every row/column boundary.
#    This variant removes L entirely. Cells are still arranged on a regular square lattice,
#    but the separators are now plain background-colored gaps. Each cell still contains
#    a 1px bg border plus an inner sprite. So the visible structure becomes:
#      [outer bg gap][cell(bg border + sprite)][bg gap][cell]...
#    A concrete row trace:
#      original: ... L | cell | L | cell | L ...
#      variant : ... bg gap | cell | bg gap | cell | bg gap ...
#    The odd-one-out rule is unchanged, but segmentation can no longer read a special
#    lattice color. Instead, solver must infer cell extents from repeated non-bg content
#    bands: each cell row contributes one consecutive band of non-bg rows/cols (the inner
#    sprite area), and expanding that band by 1 px on each side reconstructs the full cell.
#
# b) Complexity calibration:
#    If we only removed L without further constraints, some sprite rows/cols could be fully
#    bg, causing false extra separators and ambiguous cell slicing. To keep D11 visible yet
#    solvable, generator forces the shared base sprite to contain at least one sprite pixel
#    in every inner row and every inner column. This yields clean repeated content bands
#    while preserving the original per-row odd-one-out logic and sprite-flip variability.
#
# c) Old assumptions that break:
#    - "uniform rows/cols identify lattice lines" no longer holds: there are many uniform
#      bg rows/cols, not one dedicated lattice color.
#    - "most common uniform-row color is L" fails because the only uniform separators are bg.
#    - "gaps between consecutive uniform lines are cell interiors" fails because uniform bg
#      rows now include outer margins, cell borders, and inter-cell gaps.
#    The new solver therefore avoids any special-color lattice detection and instead finds
#    maximal consecutive non-bg row/col runs, then expands by the known 1px cell padding.
#
# d) Placement legality:
#    Generator uses a fixed 1px outer gap and 1px inter-cell gap, pasting square cells on a
#    bg canvas. Cells never overlap because pitch = cell_size + gap, and grid size is derived
#    from the number of rows/cols and that pitch.
#
# e) Visual observability:
#    Humans can still see repeated square "tiles" because the inner sprite blocks align on a
#    strict grid and the empty bg gutters recur at a constant spacing. Across examples, the
#    visible difference from the source task is clear: colored gridlines disappear completely.
#
# f) Visual cleanliness:
#    Only two colors remain (bg, sprite). There are no overlaid markers; the scene is just a
#    regular tiled arrangement of padded sprites, so rows/columns stay legible.
#
# g) Color inference:
#    No hidden color ordering is used. The rule depends only on repeated geometry and pattern
#    frequency within each row.
#
# New solver assumptions and how generator guarantees them:
#    - Global bg is the most common color: true because canvas, gaps, and cell padding are bg.
#    - Every logical cell row/col yields exactly one consecutive non-bg band: enforced by
#      requiring at least one sprite pixel in every inner row and inner column of the base sprite.
#    - Cell padding stays 1px: inherited from the original task and preserved here.
#    - Cell size is uniform and square: preserved from the original task and generator.


def _group_runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for x in indices[1:]:
        x = int(x)
        if x == prev + 1:
            prev = x
            continue
        runs.append((start, prev))
        start = prev = x
    runs.append((start, prev))
    return runs


def _infer_spaced_cells(g: np.ndarray) -> tuple[int, list[tuple[int, int]], list[tuple[int, int]], int]:
    H, W = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    content_rows = [r for r in range(H) if np.any(g[r, :] != bg)]
    content_cols = [c for c in range(W) if np.any(g[:, c] != bg)]
    assert content_rows and content_cols

    row_runs = _group_runs(content_rows)
    col_runs = _group_runs(content_cols)
    assert row_runs and col_runs

    inner_heights = {int(b - a + 1) for a, b in row_runs}
    inner_widths = {int(b - a + 1) for a, b in col_runs}
    assert len(inner_heights) == 1 and len(inner_widths) == 1

    r_slices = []
    for a, b in row_runs:
        assert a > 0 and b < H - 1
        assert np.all(g[a - 1, :] == bg) and np.all(g[b + 1, :] == bg)
        r_slices.append((int(a - 1), int(b + 2)))

    c_slices = []
    for a, b in col_runs:
        assert a > 0 and b < W - 1
        assert np.all(g[:, a - 1] == bg) and np.all(g[:, b + 1] == bg)
        c_slices.append((int(a - 1), int(b + 2)))

    cell_hs = {rb - ra for ra, rb in r_slices}
    cell_ws = {cb - ca for ca, cb in c_slices}
    assert len(cell_hs) == 1 and len(cell_ws) == 1
    cell_size = int(next(iter(cell_hs)))
    assert cell_size == int(next(iter(cell_ws)))

    return bg, r_slices, c_slices, cell_size


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the unique cell from each logical lattice row into a single-column layout.

    Variant-specific parsing:
    - There is no explicit lattice color.
    - Cells are separated only by background-colored spacing.
    - Cell boundaries are inferred from repeated non-background content bands.
    """
    g = np.array(grid, dtype=int)
    H, _W = g.shape

    bg, r_slices, c_slices, cell_size = _infer_spaced_cells(g)

    chosen_cells: list[np.ndarray] = []
    for ra, rb in r_slices:
        cells = [g[int(ra) : int(rb), int(ca) : int(cb)] for ca, cb in c_slices]
        sigs = [cell.tobytes() for cell in cells]
        cnt = Counter(sigs)
        uniq_idx = next((j for j, s in enumerate(sigs) if int(cnt[s]) == 1), None)
        assert uniq_idx is not None
        chosen_cells.append(cells[int(uniq_idx)])

    out = np.full((H, int(cell_size + 2)), int(bg), dtype=int)
    for i, (ra, rb) in enumerate(r_slices):
        out[int(ra) : int(rb), 1 : 1 + int(cell_size)] = chosen_cells[int(i)]
    return out


def _make_cell(inner: np.ndarray, bg: int) -> np.ndarray:
    s = inner.shape[0] + 2
    cell = np.full((s, s), int(bg), dtype=int)
    paste(cell, inner, 1, 1)
    return cell


def _build_spaced(cells: list[list[np.ndarray]], bg: int, gap: int = 1) -> np.ndarray:
    rows = len(cells)
    cols = len(cells[0])
    cell_size = cells[0][0].shape[0]
    H = rows * cell_size + (rows + 1) * gap
    W = cols * cell_size + (cols + 1) * gap
    grid = np.full((H, W), int(bg), dtype=int)
    for r, row in enumerate(cells):
        for c, cell in enumerate(row):
            paste(grid, cell, gap + r * (cell_size + gap), gap + c * (cell_size + gap))
    return grid


def _valid_base_inner(inner: np.ndarray, bg: int) -> bool:
    for r in range(inner.shape[0]):
        if not np.any(inner[r, :] != bg):
            return False
    for c in range(inner.shape[1]):
        if not np.any(inner[:, c] != bg):
            return False
    return True


def _make_base_inner(rng: np.random.Generator, inner_size: int, bg: int, sp: int) -> np.ndarray | None:
    inner_area = inner_size * inner_size
    min_fill = math.ceil(0.4 * inner_area)
    max_fill = math.floor(0.9 * inner_area)
    for _ in range(200):
        fill = int(rng.integers(min_fill, max_fill + 1))
        inner = np.full((inner_size, inner_size), int(bg), dtype=int)
        on_idx = rng.choice(inner_area, size=fill, replace=False)
        inner.flat[on_idx] = int(sp)
        if _valid_base_inner(inner, bg):
            return inner
    return None


def _make_unique_variants(
    base_inner: np.ndarray,
    n_rows: int,
    bg: int,
    sp: int,
    rng: np.random.Generator,
) -> list[np.ndarray] | None:
    inner_area = base_inner.size
    variants: list[np.ndarray] = []
    used: set[bytes] = set()
    candidates: list[np.ndarray] = []
    for flip_count in (1, 2, 3):
        for flip_set in itertools.combinations(range(inner_area), flip_count):
            variant = base_inner.copy()
            for idx in flip_set:
                variant.flat[int(idx)] = sp if int(variant.flat[int(idx)]) == int(bg) else bg
            key = variant.tobytes()
            if key != base_inner.tobytes() and key not in used:
                candidates.append(variant)
                used.add(key)
    if len(candidates) < n_rows:
        return None
    pick_idx = rng.choice(len(candidates), size=n_rows, replace=False)
    for idx in pick_idx:
        variants.append(candidates[int(idx)])
    return variants


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg, sp = map(int, rng.choice(10, size=2, replace=False))
    cell_size = int(rng.choice([5, 6]))
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 5))

    inner_size = cell_size - 2
    base_inner = _make_base_inner(rng, inner_size, bg, sp)
    if base_inner is None:
        return None

    unique_inners = _make_unique_variants(base_inner, n_rows, bg, sp, rng)
    if unique_inners is None:
        return None

    unique_cells = [_make_cell(inner, bg) for inner in unique_inners]
    out = _build_spaced([[cell] for cell in unique_cells], bg)

    base_cell = _make_cell(base_inner, bg)
    in_cells: list[list[np.ndarray]] = []
    for row in range(n_rows):
        odd_col = int(rng.integers(0, n_cols))
        row_cells = [base_cell.copy() for _ in range(n_cols)]
        row_cells[odd_col] = unique_cells[row].copy()
        in_cells.append(row_cells)
    inp = _build_spaced(in_cells, bg)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input after 300 attempts")
