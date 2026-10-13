from __future__ import annotations

from collections import Counter
import random

import numpy as np


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            arr = np.array(coords, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (
                        int(arr[:, 0].min()),
                        int(arr[:, 1].min()),
                        int(arr[:, 0].max()),
                        int(arr[:, 1].max()),
                    ),
                }
            )
    return components


def _distance_to_horizontal(start_col: int, target_col: int) -> int:
    return min(abs(int(target_col) - int(start_col)), abs(int(target_col) - (int(start_col) + 1)))


def _distance_to_vertical(start_row: int, target_row: int) -> int:
    return min(abs(int(target_row) - int(start_row)), abs(int(target_row) - (int(start_row) + 1)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend every domino marker into its full row or column bar, deciding each crossing by the nearer generator.

    Input structure:
    - The background is the dominant color.
    - Every foreground object is a same-color 4-connected domino of exactly two cells.
    - A horizontal domino marks one row, and a vertical domino marks one column.

    Rule:
    - Paint every marked row across the full width using its horizontal domino color.
    - Paint every marked column across the full height using its vertical domino color.
    - At each row-column crossing, compare the crossing's along-bar distance to the nearest cell of the horizontal domino and to the nearest cell of the vertical domino; the smaller distance wins that cell.

    Key insight:
    - Crossings are resolved locally, not by a global layer order: whichever domino is physically closer to that specific intersection owns it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    row_markers: dict[int, tuple[int, int]] = {}
    col_markers: dict[int, tuple[int, int]] = {}
    components = _find_components(g, bg)
    assert components, "expected foreground domino markers"

    for comp in components:
        color = int(comp["color"])
        coords = np.asarray(comp["coords"], dtype=int)
        assert coords.shape == (2, 2), "each foreground object must be a two-cell domino"
        coords = coords[np.lexsort((coords[:, 1], coords[:, 0]))]
        (r0, c0), (r1, c1) = coords
        if r0 == r1 and c1 == c0 + 1:
            assert int(r0) not in row_markers, "each marked row must have exactly one horizontal domino"
            row_markers[int(r0)] = (color, int(c0))
        elif c0 == c1 and r1 == r0 + 1:
            assert int(c0) not in col_markers, "each marked column must have exactly one vertical domino"
            col_markers[int(c0)] = (color, int(r0))
        else:
            raise AssertionError("foreground object must be horizontal or vertical domino")

    assert row_markers and col_markers, "expected at least one marked row and one marked column"

    out = np.full_like(g, bg)
    for row, (color, _start_col) in row_markers.items():
        out[row, :] = int(color)
    for col, (color, _start_row) in col_markers.items():
        out[:, col] = int(color)

    for row, (row_color, row_start) in row_markers.items():
        for col, (col_color, col_start) in col_markers.items():
            row_distance = _distance_to_horizontal(row_start, col)
            col_distance = _distance_to_vertical(col_start, row)
            assert row_distance != col_distance, "crossing distances must be unequal"
            out[row, col] = int(row_color if row_distance < col_distance else col_color)

    return out


_GRID_H_RANGE = (10, 15)
_GRID_W_RANGE = (10, 15)
_H_COUNT_RANGE = (2, 3)
_V_COUNT_RANGE = (2, 3)
_PLACEMENT_ATTEMPTS = 120
_GEN_ATTEMPTS = 500


def _sample_spaced_indices(length: int, count: int, *, border: int = 1, padding: int = 1) -> list[int]:
    lo = border
    hi = length - border
    if hi - lo < count:
        lo = 0
        hi = length
    pool = list(range(lo, hi))
    if len(pool) < count:
        raise ValueError("not enough indices to sample")

    for _ in range(80):
        picked = sorted(random.sample(pool, count))
        if all(picked[i + 1] - picked[i] > padding for i in range(len(picked) - 1)):
            return picked
    return sorted(random.sample(pool, count))


def _bar_cells(orient: str, index: int, start: int) -> list[Coord]:
    if orient == "h":
        return [(int(index), int(start)), (int(index), int(start) + 1)]
    return [(int(start), int(index)), (int(start) + 1, int(index))]


def _can_place(cells: list[Coord], occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    cell_set = set(cells)
    for r, c in cells:
        if occupied[r, c]:
            return False
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and occupied[nr, nc] and (nr, nc) not in cell_set:
                return False
    return True


def _crossing_winners(
    rows: list[int],
    cols: list[int],
    h_starts: dict[int, int],
    v_starts: dict[int, int],
) -> np.ndarray:
    wins = np.zeros((len(rows), len(cols)), dtype=bool)
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            row_distance = _distance_to_horizontal(h_starts[row], col)
            col_distance = _distance_to_vertical(v_starts[col], row)
            assert row_distance != col_distance
            wins[i, j] = row_distance < col_distance
    return wins


def _interesting_winners(wins: np.ndarray) -> bool:
    if wins.size == 0 or bool(wins.all()) or not bool(wins.any()):
        return False
    row_mixed = any(bool(row.any()) and not bool(row.all()) for row in wins)
    col_mixed = any(bool(col.any()) and not bool(col.all()) for col in wins.T)
    return row_mixed and col_mixed


def _interleaved_bars(rows: list[int], cols: list[int], colors: list[int]) -> list[tuple[str, int, int]]:
    h_bars = [("h", row, colors[i]) for i, row in enumerate(rows)]
    v_bars = [("v", col, colors[len(rows) + i]) for i, col in enumerate(cols)]
    random.shuffle(h_bars)
    random.shuffle(v_bars)

    ordered: list[tuple[str, int, int]] = []
    while h_bars or v_bars:
        if h_bars:
            ordered.append(h_bars.pop())
        if v_bars:
            ordered.append(v_bars.pop())
    return ordered


def _place_dominoes(
    height: int,
    width: int,
    rows: list[int],
    cols: list[int],
    colors: list[int],
    bg: int,
) -> tuple[np.ndarray, dict[int, int], dict[int, int]] | None:
    bars = _interleaved_bars(rows, cols, colors)
    occupied = np.zeros((height, width), dtype=bool)
    grid = np.full((height, width), int(bg), dtype=int)
    h_starts: dict[int, int] = {}
    v_starts: dict[int, int] = {}

    def backtrack(i: int) -> bool:
        if i == len(bars):
            return _interesting_winners(_crossing_winners(rows, cols, h_starts, v_starts))

        orient, index, color = bars[i]
        options = list(range(width - 1)) if orient == "h" else list(range(height - 1))
        random.shuffle(options)

        for start in options:
            cells = _bar_cells(orient, index, start)
            if not _can_place(cells, occupied):
                continue

            if orient == "h":
                if any(
                    _distance_to_horizontal(start, col) == _distance_to_vertical(v_starts[col], index)
                    for col in v_starts
                ):
                    continue
            else:
                if any(
                    _distance_to_horizontal(h_starts[row], index) == _distance_to_vertical(start, row)
                    for row in h_starts
                ):
                    continue

            for r, c in cells:
                occupied[r, c] = True
                grid[r, c] = int(color)
            if orient == "h":
                h_starts[index] = start
            else:
                v_starts[index] = start

            if backtrack(i + 1):
                return True

            if orient == "h":
                del h_starts[index]
            else:
                del v_starts[index]
            for r, c in cells:
                occupied[r, c] = False
                grid[r, c] = int(bg)
        return False

    if backtrack(0):
        return grid, h_starts, v_starts
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GRID_H_RANGE)
    width = random.randint(*_GRID_W_RANGE)
    h_count = random.randint(*_H_COUNT_RANGE)
    v_count = random.randint(*_V_COUNT_RANGE)
    bg = random.randint(0, 9)

    try:
        rows = _sample_spaced_indices(height, h_count, border=1, padding=1)
        cols = _sample_spaced_indices(width, v_count, border=1, padding=1)
    except ValueError:
        return None

    palette = [color for color in range(10) if color != bg]
    colors = random.sample(palette, h_count + v_count)

    for _ in range(_PLACEMENT_ATTEMPTS):
        placed = _place_dominoes(height, width, rows, cols, colors, bg)
        if placed is None:
            continue
        grid, h_starts, v_starts = placed

        wins = _crossing_winners(rows, cols, h_starts, v_starts)
        if not _interesting_winners(wins):
            continue

        out = main(grid)
        row_over_col = np.full((height, width), bg, dtype=int)
        col_over_row = np.full((height, width), bg, dtype=int)
        for i, row in enumerate(rows):
            row_over_col[row, :] = colors[i]
            col_over_row[row, :] = colors[i]
        for j, col in enumerate(cols):
            row_over_col[:, col] = colors[h_count + j]
            col_over_row[:, col] = colors[h_count + j]
        for j, col in enumerate(cols):
            row_over_col[:, col] = colors[h_count + j]
        for i, row in enumerate(rows):
            row_over_col[row, :] = colors[i]

        if np.array_equal(out, grid):
            continue
        if np.array_equal(out, row_over_col) or np.array_equal(out, col_over_row):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
