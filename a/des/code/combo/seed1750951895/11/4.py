from __future__ import annotations

import random

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_ATTEMPTS = 400


def _find_dominoes(grid: np.ndarray) -> list[tuple[int, np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    dominoes: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(sorted(coords), dtype=int)
            assert arr.shape == (2, 2), "each foreground object must be a monochrome domino"
            dominoes.append((color, arr))

    assert dominoes, "expected at least one domino"
    return dominoes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each domino into a full row or full column depending on orientation.

    Horizontal dominoes paint their row; vertical dominoes paint their column.
    Columns are painted first, then rows overwrite them.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    out = np.full((h, w), BG, dtype=int)

    row_ops: list[tuple[int, int]] = []
    col_ops: list[tuple[int, int]] = []
    for color, coords in _find_dominoes(g):
        (r0, c0), (r1, c1) = coords
        if r0 == r1:
            assert abs(c0 - c1) == 1, "horizontal domino cells must be adjacent"
            row_ops.append((int(r0), int(color)))
        elif c0 == c1:
            assert abs(r0 - r1) == 1, "vertical domino cells must be adjacent"
            col_ops.append((int(c0), int(color)))
        else:
            raise AssertionError("domino must be horizontal or vertical")

    for c, color in col_ops:
        out[:, c] = int(color)
    for r, color in row_ops:
        out[r, :] = int(color)
    return out


def _chebyshev_clear(occupied: set[tuple[int, int]], cells: list[tuple[int, int]]) -> bool:
    for r, c in cells:
        for rr, cc in occupied:
            if max(abs(r - rr), abs(c - cc)) <= 1:
                return False
    return True


def _place_domino(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    color: int,
    orientation: str,
    fixed_index: int | None = None,
) -> tuple[int, int] | None:
    h, w = grid.shape
    candidates: list[list[tuple[int, int]]] = []

    if orientation == "h":
        rows = [fixed_index] if fixed_index is not None else list(range(h))
        for r in rows:
            for c in range(w - 1):
                cells = [(r, c), (r, c + 1)]
                if _chebyshev_clear(occupied, cells):
                    candidates.append(cells)
    else:
        cols = [fixed_index] if fixed_index is not None else list(range(w))
        for c in cols:
            for r in range(h - 1):
                cells = [(r, c), (r + 1, c)]
                if _chebyshev_clear(occupied, cells):
                    candidates.append(cells)

    if not candidates:
        return None

    cells = random.choice(candidates)
    for r, c in cells:
        grid[r, c] = color
        occupied.add((r, c))
    return cells[0]


def _scenario_counts(h: int, w: int) -> tuple[str, int, int]:
    scenario = random.choice(["h_only", "v_only", "mixed"])
    if scenario == "h_only":
        return scenario, random.randint(1, min(4, h)), 0
    if scenario == "v_only":
        return scenario, 0, random.randint(1, min(4, w))
    max_h = min(3, h)
    max_v = min(3, w)
    return scenario, random.randint(1, max_h), random.randint(1, max_v)


def _sample_distinct_indices(limit: int, count: int) -> list[int]:
    count = min(count, limit)
    choices = list(range(limit))
    random.shuffle(choices)
    return sorted(choices[:count])


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(5, 12)
    w = random.randint(5, 12)
    scenario, n_h, n_v = _scenario_counts(h, w)
    total = n_h + n_v
    if total == 0 or total > len(PALETTE):
        return None

    row_targets = _sample_distinct_indices(h, n_h)
    col_targets = _sample_distinct_indices(w, n_v)
    colors = random.sample(PALETTE, total)
    random.shuffle(colors)

    grid = np.full((h, w), BG, dtype=int)
    occupied: set[tuple[int, int]] = set()

    # Mixed cases often get an actual visible overwrite relation.
    if scenario == "mixed" and row_targets and col_targets and random.random() < 0.7:
        row_targets[0] = random.randrange(h)
        col_targets[0] = random.randrange(w)

    color_idx = 0
    for r in row_targets:
        placed = _place_domino(grid, occupied, colors[color_idx], "h", fixed_index=r)
        if placed is None:
            return None
        color_idx += 1

    for c in col_targets:
        placed = _place_domino(grid, occupied, colors[color_idx], "v", fixed_index=c)
        if placed is None:
            return None
        color_idx += 1

    # Require the generator to actually cover its intended scenario.
    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out != BG):
        return None

    # Structural sanity checks against the sampled targets.
    for r in row_targets:
        if len(set(map(int, out[r]))) != 1:
            return None
    for c in col_targets:
        if n_h == 0 and len(set(map(int, out[:, c]))) != 1:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if isinstance(out, np.ndarray) and out.shape == grid.shape:
            return grid
    raise RuntimeError("failed to generate a valid input")
