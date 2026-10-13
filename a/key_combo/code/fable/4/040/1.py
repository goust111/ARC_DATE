from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GRAY = 5


def _lattice_lines(g: np.ndarray) -> tuple[list[int], list[int]]:
    """Return full horizontal/vertical lattice lines in (row, column) order."""
    line_pixel = (g == GRAY) | (g == RED)
    rows = [int(r) for r in np.flatnonzero(np.all(line_pixel, axis=1))]
    cols = [int(c) for c in np.flatnonzero(np.all(line_pixel, axis=0))]
    h, w = g.shape
    assert rows and rows[0] == 0 and rows[-1] == h - 1
    assert cols and cols[0] == 0 and cols[-1] == w - 1
    assert all(b - a >= 3 for a, b in zip(rows, rows[1:]))
    assert all(b - a >= 3 for a, b in zip(cols, cols[1:]))
    return rows, cols


def _segment_is_red(g: np.ndarray, rows: list[int], cols: list[int], key: tuple[str, int, int]) -> bool:
    orientation, a, b = key
    if orientation == "h":
        pixels = g[rows[a], cols[b] + 1 : cols[b + 1]]
    else:
        pixels = g[rows[a] + 1 : rows[a + 1], cols[b]]
    assert pixels.size > 0
    assert np.all((pixels == GRAY) | (pixels == RED))
    return bool(np.all(pixels == RED))


def _trace_grid(
    g: np.ndarray,
    rows: list[int],
    cols: list[int],
    start: tuple[int, int],
    direction: tuple[int, int],
) -> set[tuple[int, int]]:
    """Trace in cell coordinates; a repeated state means the route is now periodic."""
    nr_cells, nc_cells = len(rows) - 1, len(cols) - 1
    i, j = start
    dr, dc = direction
    visited: set[tuple[int, int]] = set()
    states: set[tuple[int, int, int, int]] = set()

    while (i, j, dr, dc) not in states:
        states.add((i, j, dr, dc))
        visited.add((i, j))

        horizontal = ("h", i if dr < 0 else i + 1, j)
        vertical = ("v", i, j if dc < 0 else j + 1)
        if _segment_is_red(g, rows, cols, horizontal):
            dr = -dr
        if _segment_is_red(g, rows, cols, vertical):
            dc = -dc

        ni, nj = i + dr, j + dc
        if not (0 <= ni < nr_cells and 0 <= nj < nc_cells):
            break
        i, j = ni, nj

    return visited


def main(grid: np.ndarray) -> np.ndarray:
    """Fill every lattice cell entered by the invisible reflected diagonal ray with blue.

    Input structure:
    - A black grid is enclosed and partitioned by full gray horizontal and vertical lattice lines, with some complete open line segments recolored red.
    - Exactly one cell interior contains one blue pixel touching one horizontal and one vertical side of that cell, hence occupying one of its four visible corners.

    Rule:
    - The seed corner gives the initial outward diagonal direction in cell coordinates. At each cell's outgoing corner, a red horizontal segment reverses the row direction and a red vertical segment reverses the column direction; two red segments reverse both directions before the ray enters its next diagonal cell.
    - Fill the complete interior of every entered cell blue, including the seed cell, while preserving all gray and red lattice pixels. A route ends at the outer boundary; a periodic route is complete once its cell-and-direction state repeats.

    Key insight:
    - Reflections must be traced on the lattice of whole cells using the two line segments at each outgoing corner; the invisible pixel-scale ray is only rendered afterward as cell-level blue fills.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    assert set(map(int, np.unique(g))).issubset({BLACK, BLUE, RED, GRAY})
    rows, cols = _lattice_lines(g)

    line_mask = np.zeros(g.shape, dtype=bool)
    line_mask[rows, :] = True
    line_mask[:, cols] = True
    assert np.all((g[line_mask] == GRAY) | (g[line_mask] == RED))
    assert np.all((g[~line_mask] == BLACK) | (g[~line_mask] == BLUE))

    seed_coords = np.argwhere(g == BLUE)
    assert seed_coords.shape == (1, 2), "expected exactly one blue corner seed"
    sr, sc = map(int, seed_coords[0])
    i = int(np.searchsorted(rows, sr) - 1)
    j = int(np.searchsorted(cols, sc) - 1)
    assert 0 <= i < len(rows) - 1 and 0 <= j < len(cols) - 1

    row_directions = []
    if sr == rows[i] + 1:
        row_directions.append(-1)
    if sr == rows[i + 1] - 1:
        row_directions.append(1)
    col_directions = []
    if sc == cols[j] + 1:
        col_directions.append(-1)
    if sc == cols[j + 1] - 1:
        col_directions.append(1)
    assert len(row_directions) == len(col_directions) == 1, "seed must identify one cell corner uniquely"

    visited = _trace_grid(g, rows, cols, (i, j), (row_directions[0], col_directions[0]))
    out = g.copy()
    for cell_r, cell_c in visited:
        out[rows[cell_r] + 1 : rows[cell_r + 1], cols[cell_c] + 1 : cols[cell_c + 1]] = BLUE
    return out


# ------------------------------ generator ------------------------------


def _line_positions(interior_sizes: np.ndarray) -> list[int]:
    positions = [0]
    for size in interior_sizes:
        positions.append(positions[-1] + int(size) + 1)
    return positions


def _trace_flags(
    h_red: np.ndarray,
    v_red: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict:
    """Generator oracle over abstract segment flags, independent of pixel parsing."""
    nr_cells, nc_cells = h_red.shape[0] - 1, h_red.shape[1]
    i, j = start
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    events: list[tuple[bool, bool, tuple[str, int, int], tuple[str, int, int]]] = []
    states: set[tuple[int, int, int, int]] = set()

    while (i, j, dr, dc) not in states:
        states.add((i, j, dr, dc))
        cells.append((i, j))
        h_key = ("h", i if dr < 0 else i + 1, j)
        v_key = ("v", i, j if dc < 0 else j + 1)
        hb = bool(h_red[h_key[1], h_key[2]])
        vb = bool(v_red[v_key[1], v_key[2]])
        events.append((hb, vb, h_key, v_key))
        if hb:
            dr = -dr
        if vb:
            dc = -dc
        ni, nj = i + dr, j + dc
        if not (0 <= ni < nr_cells and 0 <= nj < nc_cells):
            return {"cells": cells, "events": events, "visited": set(cells), "loop": False}
        i, j = ni, nj

    return {"cells": cells, "events": events, "visited": set(cells), "loop": True}


def _all_red_keys(h_red: np.ndarray, v_red: np.ndarray) -> set[tuple[str, int, int]]:
    keys = {("h", int(a), int(b)) for a, b in np.argwhere(h_red)}
    keys |= {("v", int(a), int(b)) for a, b in np.argwhere(v_red)}
    return keys


def _without_key(h_red: np.ndarray, v_red: np.ndarray, key: tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray]:
    h2, v2 = h_red.copy(), v_red.copy()
    orientation, a, b = key
    if orientation == "h":
        h2[a, b] = False
    else:
        v2[a, b] = False
    return h2, v2


def _strong_candidate(
    h_red: np.ndarray,
    v_red: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> dict | None:
    trace = _trace_flags(h_red, v_red, start, direction)
    cells, events, visited = trace["cells"], trace["events"], trace["visited"]
    total_cells = h_red.shape[1] * v_red.shape[0]
    if trace["loop"] or not (7 <= len(visited) <= total_cells - 3):
        return None
    patterns = {(hb, vb) for hb, vb, _hk, _vk in events}
    if patterns != {(False, False), (True, False), (False, True), (True, True)}:
        return None

    # A double-red corner must visibly backtrack and then lead somewhere new;
    # this excludes treating the double reflection as a stop instruction.
    both_indices = [k for k, event in enumerate(events) if event[:2] == (True, True)]
    useful_both = False
    for k in both_indices:
        if k >= 1 and k + 2 < len(cells) and cells[k + 1] == cells[k - 1]:
            old = set(cells[: k + 1])
            if any(cell not in old for cell in cells[k + 2 :]):
                useful_both = True
                break
    if not useful_both:
        return None

    used_red: set[tuple[str, int, int]] = set()
    witnesses: set[tuple[str, int, int]] = set()
    for hb, vb, h_key, v_key in events:
        if hb:
            used_red.add(h_key)
        if vb:
            used_red.add(v_key)
        if (hb, vb) == (True, False):
            witnesses.add(h_key)
        elif (hb, vb) == (False, True):
            witnesses.add(v_key)
        elif (hb, vb) == (True, True):
            witnesses.update((h_key, v_key))
    if len(_all_red_keys(h_red, v_red) - used_red) < 2:
        return None

    # Every type of reflecting segment has a counterfactual witness whose
    # removal changes the set of filled cells, rather than merely the animation.
    for key in witnesses:
        h2, v2 = _without_key(h_red, v_red, key)
        if _trace_flags(h2, v2, start, direction)["visited"] == visited:
            return None

    # The seed's visible corner, not merely its containing cell, must matter.
    for other in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        if other != direction and _trace_flags(h_red, v_red, start, other)["visited"] == visited:
            return None
    return trace


def _render(
    row_sizes: np.ndarray,
    col_sizes: np.ndarray,
    h_red: np.ndarray,
    v_red: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> tuple[np.ndarray, list[int], list[int]]:
    rows, cols = _line_positions(row_sizes), _line_positions(col_sizes)
    grid = np.full((rows[-1] + 1, cols[-1] + 1), BLACK, dtype=int)
    grid[rows, :] = GRAY
    grid[:, cols] = GRAY
    for a, b in np.argwhere(h_red):
        grid[rows[int(a)], cols[int(b)] + 1 : cols[int(b) + 1]] = RED
    for a, b in np.argwhere(v_red):
        grid[rows[int(a)] + 1 : rows[int(a) + 1], cols[int(b)]] = RED

    i, j = start
    dr, dc = direction
    seed_r = rows[i] + 1 if dr < 0 else rows[i + 1] - 1
    seed_c = cols[j] + 1 if dc < 0 else cols[j + 1] - 1
    grid[seed_r, seed_c] = BLUE
    return grid, rows, cols


def _expected_output(grid: np.ndarray, rows: list[int], cols: list[int], visited: set[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for i, j in visited:
        out[rows[i] + 1 : rows[i + 1], cols[j] + 1 : cols[j + 1]] = BLUE
    return out


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    nr_cells = int(rng.integers(4, 7))
    nc_cells = int(rng.integers(4, 7))
    row_sizes = rng.integers(2, 5, size=nr_cells)
    col_sizes = rng.integers(2, 5, size=nc_cells)
    rows, cols = _line_positions(row_sizes), _line_positions(col_sizes)
    if rows[-1] + 1 > 30 or cols[-1] + 1 > 30:
        return None

    red_probability = float(rng.uniform(0.25, 0.42))
    h_red = rng.random((nr_cells + 1, nc_cells)) < red_probability
    v_red = rng.random((nr_cells, nc_cells + 1)) < red_probability

    candidates = [
        ((i, j), (dr, dc))
        for i in range(nr_cells)
        for j in range(nc_cells)
        for dr in (-1, 1)
        for dc in (-1, 1)
    ]
    rng.shuffle(candidates)
    for start, direction in candidates:
        trace = _strong_candidate(h_red, v_red, start, direction)
        if trace is None:
            continue
        grid, rows, cols = _render(row_sizes, col_sizes, h_red, v_red, start, direction)
        expected = _expected_output(grid, rows, cols, trace["visited"])
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
            continue

        # Semantic checks: all and only traced interiors are filled, while every
        # lattice pixel (including red decoys) is preserved exactly.
        line_mask = np.zeros(grid.shape, dtype=bool)
        line_mask[rows, :] = True
        line_mask[:, cols] = True
        if not np.array_equal(actual[line_mask], grid[line_mask]):
            continue
        valid = True
        for i in range(nr_cells):
            for j in range(nc_cells):
                tile = actual[rows[i] + 1 : rows[i + 1], cols[j] + 1 : cols[j + 1]]
                wanted = BLUE if (i, j) in trace["visited"] else BLACK
                if not np.all(tile == wanted):
                    valid = False
        if valid:
            return grid
    return None


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate lattice with all four reflection cases."""
    rng = np.random.default_rng()
    for _ in range(1200):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a strong invisible-lattice-ray input")
