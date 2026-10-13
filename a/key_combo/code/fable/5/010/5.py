from __future__ import annotations

"""ARC task 5: decode a relay chain embedded in a separator lattice.

This file is intentionally self-contained so that ``SourceFileLoader`` can load it
from any working directory.  Coordinates and inclusive/exclusive slicing follow
the common ``arc_utils`` convention: every coordinate is ``(row, column)``.
"""

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]
_DIRS8: tuple[Coord, ...] = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if (dr, dc) != (0, 0)
)


def _runs(mask: np.ndarray, value: bool) -> list[tuple[int, int]]:
    """Return inclusive runs whose boolean value equals ``value``."""
    wanted = np.asarray(mask, dtype=bool) == bool(value)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, present in enumerate(wanted.tolist()):
        if present and start is None:
            start = i
        elif not present and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(wanted) - 1))
    return runs


def _sign(value: int) -> int:
    return (int(value) > 0) - (int(value) < 0)


def _decode_lattice(grid: np.ndarray) -> dict:
    """Locate the separator lattice and decode each cell's marker geometry."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be two-dimensional"
    h, w = map(int, g.shape)
    assert 3 <= h <= 30 and 3 <= w <= 30, "grid must satisfy ARC size limits"

    uniform_rows: dict[int, list[int]] = {}
    uniform_cols: dict[int, list[int]] = {}
    for r in range(h):
        if bool(np.all(g[r, :] == g[r, 0])):
            uniform_rows.setdefault(int(g[r, 0]), []).append(r)
    for c in range(w):
        if bool(np.all(g[:, c] == g[0, c])):
            uniform_cols.setdefault(int(g[0, c]), []).append(c)

    candidates = set(uniform_rows) & set(uniform_cols)
    assert len(candidates) == 1, "expected one color forming both full separator rows and columns"
    separator = int(next(iter(candidates)))
    sep_row_mask = np.all(g == separator, axis=1)
    sep_col_mask = np.all(g == separator, axis=0)

    cell_row_runs = _runs(sep_row_mask, False)
    cell_col_runs = _runs(sep_col_mask, False)
    sep_row_runs = _runs(sep_row_mask, True)
    sep_col_runs = _runs(sep_col_mask, True)
    assert len(cell_row_runs) >= 2 and len(cell_col_runs) >= 2, "lattice needs multiple rows and columns"
    assert len(sep_row_runs) == len(cell_row_runs) - 1
    assert len(sep_col_runs) == len(cell_col_runs) - 1
    assert len({b - a + 1 for a, b in cell_row_runs}) == 1, "cell heights must be regular"
    assert len({b - a + 1 for a, b in cell_col_runs}) == 1, "cell widths must be regular"
    assert len({b - a + 1 for a, b in sep_row_runs}) == 1, "separator row widths must be regular"
    assert len({b - a + 1 for a, b in sep_col_runs}) == 1, "separator column widths must be regular"

    cell_h = cell_row_runs[0][1] - cell_row_runs[0][0] + 1
    cell_w = cell_col_runs[0][1] - cell_col_runs[0][0] + 1
    assert cell_h >= 3 and cell_w >= 3 and cell_h % 2 == 1 and cell_w % 2 == 1

    interior_mask = (~sep_row_mask)[:, None] & (~sep_col_mask)[None, :]
    interior_values = [int(v) for v in g[interior_mask].tolist()]
    background = int(Counter(interior_values).most_common(1)[0][0])
    assert background != separator, "background and separator colors must differ"

    cells: dict[Coord, dict] = {}
    for lr, (r0, r1) in enumerate(cell_row_runs):
        for lc, (c0, c1) in enumerate(cell_col_runs):
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            points = np.argwhere(patch != background)
            assert len(points) == 1, "each lattice cell must contain exactly one colored pixel"
            pr, pc = map(int, points[0])
            center_r, center_c = cell_h // 2, cell_w // 2
            direction = (_sign(pr - center_r), _sign(pc - center_c))
            cells[(lr, lc)] = {
                "color": int(patch[pr, pc]),
                "direction": direction,
                "pixel": (r0 + pr, c0 + pc),
            }

    return {
        "shape": (len(cell_row_runs), len(cell_col_runs)),
        "separator": separator,
        "background": background,
        "cell_runs": (cell_row_runs, cell_col_runs),
        "cells": cells,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace the position-coded relay through a separator lattice and extract its colors as one column.

    Input structure:
    - Full rows and columns of one separator color divide the grid into a regular lattice.
    - Every odd-sized cell interior has one non-background colored pixel in one of nine relative positions.
    - Marker colors may repeat and carry no identity or ordering meaning.

    Rule:
    - Read a marker's position relative to its own cell center as one of eight neighboring directions.
    - A centered marker has no outgoing arrow and is the terminal cell.
    - Find the unique cell receiving no arrow, follow arrows from it to the centered terminal, and output the visited marker colors from top to bottom in a single column.

    Key insight:
    - Geometry inside each local cell, not color, encodes the arrows; the invisible entrance is recovered globally as the sole unpointed cell before the chain can be traced and extracted.
    """
    decoded = _decode_lattice(grid)
    n_rows, n_cols = decoded["shape"]
    cells: dict[Coord, dict] = decoded["cells"]

    targets: dict[Coord, Coord] = {}
    incoming = {cell: 0 for cell in cells}
    terminal_count = 0
    for cell, info in cells.items():
        dr, dc = info["direction"]
        if (dr, dc) == (0, 0):
            terminal_count += 1
            continue
        target = (cell[0] + dr, cell[1] + dc)
        assert 0 <= target[0] < n_rows and 0 <= target[1] < n_cols, "arrow must target a neighboring cell"
        targets[cell] = target
        incoming[target] += 1

    assert terminal_count == 1, "expected exactly one centered terminal marker"
    starts = [cell for cell, degree in incoming.items() if degree == 0]
    assert len(starts) == 1, "expected exactly one cell that no arrow points to"

    colors: list[int] = []
    seen: set[Coord] = set()
    current = starts[0]
    while True:
        assert current not in seen, "start's relay must terminate rather than enter a cycle"
        seen.add(current)
        colors.append(int(cells[current]["color"]))
        direction = cells[current]["direction"]
        if direction == (0, 0):
            break
        current = targets[current]

    assert len(colors) >= 2, "relay chain must contain a start and a terminal"
    return np.asarray(colors, dtype=int).reshape(-1, 1)


# ------------------------------ generator ---------------------------------


def _hamiltonian_path(n_rows: int, n_cols: int) -> list[Coord] | None:
    """Generate a varied 8-neighbor Hamiltonian path using randomized Warnsdorff DFS."""
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols)]
    corners = [p for p in cells if p[0] in (0, n_rows - 1) and p[1] in (0, n_cols - 1)]
    edges = [
        p
        for p in cells
        if p not in corners and (p[0] in (0, n_rows - 1) or p[1] in (0, n_cols - 1))
    ]
    interiors = [p for p in cells if p not in corners and p not in edges]
    start_groups = [group for group in (corners, edges, interiors) if group]

    for _ in range(30):
        start = random.choice(random.choice(start_groups))
        path = [start]
        used = {start}

        def onward_degree(point: Coord) -> int:
            r, c = point
            return sum(
                0 <= r + dr < n_rows
                and 0 <= c + dc < n_cols
                and (r + dr, c + dc) not in used
                for dr, dc in _DIRS8
            )

        def search() -> bool:
            if len(path) == len(cells):
                return True
            r, c = path[-1]
            options = [
                (r + dr, c + dc)
                for dr, dc in _DIRS8
                if 0 <= r + dr < n_rows
                and 0 <= c + dc < n_cols
                and (r + dr, c + dc) not in used
            ]
            random.shuffle(options)
            options.sort(key=onward_degree)
            for nxt in options:
                path.append(nxt)
                used.add(nxt)
                if search():
                    return True
                used.remove(nxt)
                path.pop()
            return False

        if search():
            return path
    return None


def _render_case(
    n_rows: int,
    n_cols: int,
    cell_h: int,
    cell_w: int,
    separator_thickness: int,
    background: int,
    separator: int,
    colors: dict[Coord, int],
    next_cell: dict[Coord, Coord],
    terminal: Coord,
) -> np.ndarray:
    height = n_rows * cell_h + (n_rows - 1) * separator_thickness
    width = n_cols * cell_w + (n_cols - 1) * separator_thickness
    grid = np.full((height, width), background, dtype=int)

    for r in range(1, n_rows):
        r0 = r * cell_h + (r - 1) * separator_thickness
        grid[r0 : r0 + separator_thickness, :] = separator
    for c in range(1, n_cols):
        c0 = c * cell_w + (c - 1) * separator_thickness
        grid[:, c0 : c0 + separator_thickness] = separator

    center_r, center_c = cell_h // 2, cell_w // 2
    for cell, color in colors.items():
        if cell == terminal:
            dr, dc = 0, 0
        else:
            target = next_cell[cell]
            dr = _sign(target[0] - cell[0])
            dc = _sign(target[1] - cell[1])
        local_r = center_r if dr == 0 else center_r + dr * random.randint(1, center_r)
        local_c = center_c if dc == 0 else center_c + dc * random.randint(1, center_c)
        r0 = cell[0] * (cell_h + separator_thickness)
        c0 = cell[1] * (cell_w + separator_thickness)
        grid[r0 + local_r, c0 + local_c] = int(color)
    return grid


def _semantic_validation(grid: np.ndarray, spec: dict) -> bool:
    """Check the intended graph semantics, counterexamples, and oracle result."""
    try:
        decoded = _decode_lattice(grid)
        if decoded["shape"] != spec["shape"]:
            return False
        cells: dict[Coord, dict] = decoded["cells"]
        if set(cells) != set(spec["colors"]):
            return False

        # Geometry must encode exactly the graph that was constructed, in lattice coordinates.
        actual_next: dict[Coord, Coord] = {}
        incoming = {cell: 0 for cell in cells}
        centers: list[Coord] = []
        for cell, info in cells.items():
            if int(info["color"]) != int(spec["colors"][cell]):
                return False
            dr, dc = info["direction"]
            if (dr, dc) == (0, 0):
                centers.append(cell)
                continue
            target = (cell[0] + dr, cell[1] + dc)
            if target not in cells:
                return False
            actual_next[cell] = target
            incoming[target] += 1
        if actual_next != spec["next_cell"] or centers != [spec["terminal"]]:
            return False
        if [cell for cell, degree in incoming.items() if degree == 0] != [spec["path"][0]]:
            return False
        if any(degree != 1 for cell, degree in incoming.items() if cell != spec["path"][0]):
            return False

        # Independently trace the constructed graph and require all key dependencies.
        traced = [spec["path"][0]]
        while traced[-1] != spec["terminal"]:
            nxt = actual_next.get(traced[-1])
            if nxt is None or nxt in traced:
                return False
            traced.append(nxt)
        if traced != spec["path"]:
            return False
        off_chain = set(cells) - set(traced)
        if len(off_chain) < 2 or len(traced) > len(cells) - 2:
            return False

        directions = [
            (b[0] - a[0], b[1] - a[1])
            for a, b in zip(traced, traced[1:])
        ]
        if not any(dr != 0 and dc != 0 for dr, dc in directions):
            return False
        if not any((dr == 0) != (dc == 0) for dr, dc in directions):
            return False
        if len(set(directions)) < 3:
            return False

        chain_colors = [int(cells[cell]["color"]) for cell in traced]
        off_colors = [int(cells[cell]["color"]) for cell in off_chain]
        if len(set(chain_colors)) < 3 or len(set(chain_colors)) == len(chain_colors):
            return False
        if chain_colors[0] not in off_colors or chain_colors[-1] not in off_colors:
            return False

        # Reject a visible scan-order shortcut in addition to the graph checks.
        row_major = [int(cells[cell]["color"]) for cell in sorted(cells)]
        if chain_colors == row_major[: len(chain_colors)]:
            return False
        expected = np.asarray(chain_colors, dtype=int).reshape(-1, 1)
        produced = main(grid)
        return bool(np.array_equal(produced, expected))
    except (AssertionError, KeyError, IndexError, ValueError):
        return False


def _generation_attempt() -> np.ndarray | None:
    valid_layouts = [
        (nr, nc, ch, cw, thickness)
        for nr in range(3, 6)
        for nc in range(3, 6)
        for ch in (3, 5)
        for cw in (3, 5)
        for thickness in (1, 2)
        if nr * ch + (nr - 1) * thickness <= 30
        and nc * cw + (nc - 1) * thickness <= 30
    ]
    n_rows, n_cols, cell_h, cell_w, thickness = random.choice(valid_layouts)
    order = _hamiltonian_path(n_rows, n_cols)
    if order is None:
        return None

    total = len(order)
    possible_off_counts = [count for count in range(2, total - 4, 2)]
    if not possible_off_counts:
        return None
    off_count = random.choice(possible_off_counts)
    chain_length = total - off_count
    path = order[:chain_length]
    off_chain_order = order[chain_length:]

    next_cell: dict[Coord, Coord] = {
        source: target for source, target in zip(path, path[1:])
    }
    for i in range(0, len(off_chain_order), 2):
        a, b = off_chain_order[i], off_chain_order[i + 1]
        next_cell[a] = b
        next_cell[b] = a
    terminal = path[-1]

    background, separator = random.sample(range(10), 2)
    available = [color for color in range(10) if color not in (background, separator)]
    palette = random.sample(available, random.randint(3, min(6, len(available))))
    colors = {cell: random.choice(palette) for cell in order}

    # Force repeated, cross-chain endpoint colors so color cannot identify roles.
    first_color, terminal_color, third_color = random.sample(palette, 3)
    colors[path[0]] = first_color
    colors[path[1]] = terminal_color
    colors[path[2]] = third_color
    colors[path[3]] = first_color
    colors[terminal] = terminal_color
    colors[off_chain_order[0]] = first_color
    colors[off_chain_order[1]] = terminal_color

    grid = _render_case(
        n_rows,
        n_cols,
        cell_h,
        cell_w,
        thickness,
        background,
        separator,
        colors,
        next_cell,
        terminal,
    )
    spec = {
        "shape": (n_rows, n_cols),
        "colors": colors,
        "next_cell": next_cell,
        "path": path,
        "terminal": terminal,
    }
    return grid if _semantic_validation(grid, spec) else None


def generate_input() -> np.ndarray:
    """Generate a diverse, non-degenerate separator-lattice relay instance."""
    for _ in range(400):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid grid-relay instance")
