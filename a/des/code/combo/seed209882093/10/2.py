from __future__ import annotations

import math
from collections import deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG = 0
PALETTE = np.arange(1, 10, dtype=int)
GEN_RETRIES = 400


def _components(grid: np.ndarray) -> list[dict[str, np.ndarray | int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, np.ndarray | int]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append({"color": color, "coords": np.array(coords, dtype=int)})

    return comps


def _fits_single_cell(coords: np.ndarray, pr: int, pc: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min() // pr) == int(rows.max() // pr)
        and int(cols.min() // pc) == int(cols.max() // pc)
    )


def _fold_with_period(grid: np.ndarray, pr: int, pc: int) -> np.ndarray | None:
    out = np.zeros((pr, pc), dtype=int)
    occupied = np.zeros((pr, pc), dtype=bool)

    for comp in _components(grid):
        coords = np.asarray(comp["coords"], dtype=int)
        if not _fits_single_cell(coords, pr, pc):
            return None
        color = int(comp["color"])
        rr = coords[:, 0] % pr
        cc = coords[:, 1] % pc
        for r, c in zip(rr, cc, strict=True):
            if occupied[r, c]:
                return None
            occupied[r, c] = True
            out[r, c] = color

    return out


def _infer_period(grid: np.ndarray) -> tuple[int, int]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g)
    assert comps, "expected at least one fragment"

    max_h = max(int(comp["coords"][:, 0].max() - comp["coords"][:, 0].min() + 1) for comp in comps)
    max_w = max(int(comp["coords"][:, 1].max() - comp["coords"][:, 1].min() + 1) for comp in comps)

    best: tuple[int, int] | None = None
    best_key: tuple[int, int, int, int] | None = None
    h, w = g.shape
    for pr in range(max_h, h + 1):
        for pc in range(max_w, w + 1):
            folded = _fold_with_period(g, pr, pc)
            if folded is None:
                continue
            filled = int(np.count_nonzero(folded))
            key = (pr * pc, pr + pc, -filled, max(pr, pc))
            if best_key is None or key < best_key:
                best = (pr, pc)
                best_key = key

    assert best is not None, "failed to infer a valid hidden period"
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """Fold scattered monochromatic fragments into their hidden periodic cell."""
    g = np.asarray(grid, dtype=int)
    pr, pc = _infer_period(g)
    out = _fold_with_period(g, pr, pc)
    assert out is not None
    return out


def _random_connected_shape(rng: np.random.Generator, area: int, max_h: int, max_w: int) -> np.ndarray:
    for _ in range(100):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for dr, dc in DIR4:
                    nr, nc = r + dr, c + dc
                    if (nr, nc) in cells:
                        continue
                    trial = cells | {(nr, nc)}
                    rows = [p[0] for p in trial]
                    cols = [p[1] for p in trial]
                    if max(rows) - min(rows) + 1 <= max_h and max(cols) - min(cols) + 1 <= max_w:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(frontier[int(rng.integers(len(frontier)))])
        if len(cells) != area:
            continue

        min_r = min(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        norm = sorted((r - min_r, c - min_c) for r, c in cells)
        return np.array(norm, dtype=int)

    raise RuntimeError("failed to sample a connected shape")


def _build_template(
    rng: np.random.Generator, pr: int, pc: int, n_fragments: int
) -> tuple[np.ndarray, list[dict[str, np.ndarray | int]]]:
    colors = rng.permutation(PALETTE)[:n_fragments]

    for _ in range(200):
        template = np.zeros((pr, pc), dtype=int)
        fragments: list[dict[str, np.ndarray | int]] = []
        ok = True

        for color in colors:
            placed = False
            for _ in range(200):
                area = int(rng.integers(2, 7))
                max_h = min(pr, int(rng.integers(2, min(pr, 4) + 1)))
                max_w = min(pc, int(rng.integers(2, min(pc, 4) + 1)))
                if max_h * max_w < area:
                    continue
                shape = _random_connected_shape(rng, area, max_h, max_w)
                h = int(shape[:, 0].max() + 1)
                w = int(shape[:, 1].max() + 1)
                r0 = int(rng.integers(0, pr - h + 1))
                c0 = int(rng.integers(0, pc - w + 1))
                coords = shape + np.array([r0, c0], dtype=int)
                if np.any(template[coords[:, 0], coords[:, 1]] != BG):
                    continue
                template[coords[:, 0], coords[:, 1]] = int(color)
                fragments.append({"color": int(color), "coords": coords})
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        mask = template != BG
        row_counts = np.count_nonzero(mask, axis=1)
        col_counts = np.count_nonzero(mask, axis=0)
        if int(mask.sum()) < 10:
            continue
        if int((row_counts > 0).sum()) < 3 or int((col_counts > 0).sum()) < 3:
            continue
        if not np.any(row_counts >= 2) or not np.any(col_counts >= 2):
            continue

        return template, fragments

    raise RuntimeError("failed to build a non-degenerate template")


def _sample_cell_positions(
    rng: np.random.Generator, n_fragments: int, n_rows: int, n_cols: int
) -> np.ndarray:
    for _ in range(200):
        picks = rng.choice(n_rows * n_cols, size=n_fragments, replace=False)
        cells = np.column_stack((picks // n_cols, picks % n_cols)).astype(int)
        used_rows = len(set(map(int, cells[:, 0])))
        used_cols = len(set(map(int, cells[:, 1])))
        if used_rows < 2 or used_cols < 2:
            continue
        if used_rows == n_fragments or used_cols == n_fragments:
            continue
        return cells
    raise RuntimeError("failed to sample diverse cell positions")


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    pr = int(rng.integers(5, 9))
    pc = int(rng.integers(5, 9))
    max_fragments = min(len(PALETTE), max(4, (pr * pc) // 5))
    n_fragments = int(rng.integers(4, min(8, max_fragments) + 1))

    try:
        template, fragments = _build_template(rng, pr, pc, n_fragments)
    except RuntimeError:
        return None

    n_rows = int(rng.integers(max(3, math.ceil(math.sqrt(n_fragments))), 7))
    n_cols = int(rng.integers(max(3, math.ceil(math.sqrt(n_fragments))), 7))
    while n_rows * n_cols < n_fragments:
        if n_rows <= n_cols:
            n_rows += 1
        else:
            n_cols += 1

    try:
        cells = _sample_cell_positions(rng, n_fragments, n_rows, n_cols)
    except RuntimeError:
        return None

    grid = np.zeros((n_rows * pr, n_cols * pc), dtype=int)
    for frag, (cell_r, cell_c) in zip(fragments, cells, strict=True):
        coords = np.asarray(frag["coords"], dtype=int).copy()
        coords[:, 0] += int(cell_r) * pr
        coords[:, 1] += int(cell_c) * pc
        color = int(frag["color"])
        grid[coords[:, 0], coords[:, 1]] = color

    try:
        inferred = _infer_period(grid)
    except AssertionError:
        return None
    if inferred != (pr, pc):
        return None
    if not np.array_equal(main(grid), template):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid hidden-period ARC input")
