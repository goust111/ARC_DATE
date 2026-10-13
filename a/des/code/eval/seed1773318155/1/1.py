from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 5
CELL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
ORIENTATIONS = ("horizontal", "vertical", "diag_down", "diag_up")
MAX_GENERATION_ATTEMPTS = 400


def _full_lines(grid: np.ndarray, color: int, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == color)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == color)]


def _find_separator(grid: np.ndarray) -> tuple[int, list[int], list[int]]:
    best: tuple[tuple[int, int, int], int, list[int], list[int]] | None = None
    for color in map(int, np.unique(grid)):
        rows = _full_lines(grid, color, axis=0)
        cols = _full_lines(grid, color, axis=1)
        if not rows or not cols:
            continue
        gaps = [b - a for a, b in zip(rows, rows[1:])] + [b - a for a, b in zip(cols, cols[1:])]
        regular = sum(gap == CELL + 1 for gap in gaps)
        score = (regular, len(rows) + len(cols), -color)
        if best is None or score > best[0]:
            best = (score, int(color), rows, cols)
    assert best is not None, "expected a unique separator color"
    return best[1], best[2], best[3]


def _spans(lines: list[int], limit: int) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(set(map(int, lines))) + [int(limit)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        if right - left > 1:
            spans.append((int(left + 1), int(right - 1)))
    return spans


def _grid_structure(grid: np.ndarray) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]]:
    sep, rows, cols = _find_separator(grid)
    row_spans = _spans(rows, grid.shape[0])
    col_spans = _spans(cols, grid.shape[1])
    assert row_spans and col_spans, "expected at least one cell"
    assert all(r1 - r0 + 1 == CELL for r0, r1 in row_spans), "row cell height must be 5"
    assert all(c1 - c0 + 1 == CELL for c0, c1 in col_spans), "col cell width must be 5"
    return BG, sep, row_spans, col_spans


def _pair_direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    assert dr != 0 or dc != 0, "paired seeds must be in different cells"
    if dr == 0:
        return "horizontal"
    if dc == 0:
        return "vertical"
    return "diag_down" if dr * dc > 0 else "diag_up"


def _stripe_mask(direction: str) -> np.ndarray:
    rr, cc = np.indices((CELL, CELL))
    if direction == "horizontal":
        return (rr % 2) == 0
    if direction == "vertical":
        return (cc % 2) == 0
    if direction == "diag_down":
        return np.isin(rr - cc, (-2, 0, 2))
    if direction == "diag_up":
        return np.isin(rr + cc, (2, 4, 6))
    raise AssertionError(f"unknown direction: {direction}")


def _extract_seed_cells(
    grid: np.ndarray, sep: int, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]]
) -> tuple[int, dict[int, list[tuple[int, int]]]]:
    counts = {
        int(color): int(np.sum(grid == color))
        for color in np.unique(grid)
        if int(color) != int(sep)
    }
    bg = max(counts.items(), key=lambda item: item[1])[0]

    by_color: dict[int, list[tuple[int, int]]] = {}
    for cell_r, (r0, r1) in enumerate(row_spans):
        for cell_c, (c0, c1) in enumerate(col_spans):
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            seeds = np.argwhere((patch != bg) & (patch != sep))
            assert len(seeds) <= 1, "each occupied cell must contain exactly one seed"
            if len(seeds) == 0:
                continue
            sr, sc = map(int, seeds[0])
            color = int(patch[sr, sc])
            by_color.setdefault(color, []).append((int(cell_r), int(cell_c)))

    seed_colors = {color for color, count in counts.items() if color != bg and count == 2}
    assert set(by_color) == seed_colors, "every seed color must appear exactly twice"
    assert by_color, "expected at least one colored pair"
    for cells in by_color.values():
        assert len(cells) == 2, "each seed color must occupy exactly two cells"
    return bg, by_color


def _paint_cell(
    out: np.ndarray,
    bg: int,
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    cell: tuple[int, int],
    color: int,
    direction: str,
) -> None:
    r0, r1 = row_spans[cell[0]]
    c0, c1 = col_spans[cell[1]]
    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    patch[:, :] = bg
    patch[_stripe_mask(direction)] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored seed pixels across a grey-separated cell grid and paint each
    occupied 5x5 cell with stripes aligned to that pair's direction.
    """
    g = np.asarray(grid, dtype=int)
    _default_bg, sep, row_spans, col_spans = _grid_structure(g)
    bg, by_color = _extract_seed_cells(g, sep, row_spans, col_spans)

    out = g.copy()
    for color, cells in by_color.items():
        direction = _pair_direction(cells[0], cells[1])
        for cell in cells:
            _paint_cell(out, bg, row_spans, col_spans, cell, color, direction)
    return out


def _empty_grid(n_rows: int, n_cols: int) -> np.ndarray:
    height = n_rows * CELL + (n_rows + 1)
    width = n_cols * CELL + (n_cols + 1)
    grid = np.full((height, width), SEP, dtype=int)
    for cell_r in range(n_rows):
        for cell_c in range(n_cols):
            r0 = 1 + cell_r * (CELL + 1)
            c0 = 1 + cell_c * (CELL + 1)
            grid[r0 : r0 + CELL, c0 : c0 + CELL] = BG
    return grid


def _candidate_pairs(free_cells: set[tuple[int, int]], direction: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    groups: dict[int, list[tuple[int, int]]] = {}
    for cell in free_cells:
        r, c = cell
        if direction == "horizontal":
            key = r
        elif direction == "vertical":
            key = c
        elif direction == "diag_down":
            key = r - c
        elif direction == "diag_up":
            key = r + c
        else:
            raise AssertionError(f"unknown direction: {direction}")
        groups.setdefault(int(key), []).append((int(r), int(c)))

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for cells in groups.values():
        if len(cells) < 2:
            continue
        random.shuffle(cells)
        for i in range(len(cells)):
            for j in range(i + 1, len(cells)):
                a = cells[i]
                b = cells[j]
                if _pair_direction(a, b) == direction:
                    pairs.append((a, b))
    random.shuffle(pairs)
    return pairs


def _place_pairs(n_rows: int, n_cols: int, directions: list[str]) -> list[tuple[str, tuple[int, int], tuple[int, int]]] | None:
    free_cells = {(r, c) for r in range(n_rows) for c in range(n_cols)}
    placed: list[tuple[str, tuple[int, int], tuple[int, int]]] = []

    def search(remaining: list[str]) -> bool:
        if not remaining:
            return True

        best_direction = None
        best_candidates = None
        for direction in sorted(set(remaining)):
            candidates = _candidate_pairs(free_cells, direction)
            if not candidates:
                return False
            if best_candidates is None or len(candidates) < len(best_candidates):
                best_direction = direction
                best_candidates = candidates

        assert best_direction is not None and best_candidates is not None
        next_remaining = list(remaining)
        next_remaining.remove(best_direction)

        for a, b in best_candidates[: min(120, len(best_candidates))]:
            free_cells.remove(a)
            free_cells.remove(b)
            placed.append((best_direction, a, b))
            if search(next_remaining):
                return True
            placed.pop()
            free_cells.add(a)
            free_cells.add(b)
        return False

    if search(list(directions)):
        return placed
    return None


def _seed_position(cell: tuple[int, int]) -> tuple[int, int]:
    cell_r, cell_c = cell
    r0 = 1 + cell_r * (CELL + 1)
    c0 = 1 + cell_c * (CELL + 1)
    return int(r0 + random.randrange(CELL)), int(c0 + random.randrange(CELL))


def _expected_output(
    inp: np.ndarray, pair_specs: list[tuple[int, str, tuple[int, int], tuple[int, int]]], n_rows: int, n_cols: int
) -> np.ndarray:
    out = inp.copy()
    row_spans = [(1 + r * (CELL + 1), 1 + r * (CELL + 1) + CELL - 1) for r in range(n_rows)]
    col_spans = [(1 + c * (CELL + 1), 1 + c * (CELL + 1) + CELL - 1) for c in range(n_cols)]
    for color, direction, cell_a, cell_b in pair_specs:
        _paint_cell(out, BG, row_spans, col_spans, cell_a, color, direction)
        _paint_cell(out, BG, row_spans, col_spans, cell_b, color, direction)
    return out


def _generate_case() -> np.ndarray | None:
    n_rows = random.randint(5, 8)
    n_cols = random.randint(5, 8)
    max_pairs = min(len(PALETTE), 6)
    n_pairs = random.randint(4, max_pairs)

    directions = list(ORIENTATIONS)
    directions.extend(random.choice(ORIENTATIONS) for _ in range(n_pairs - len(ORIENTATIONS)))
    random.shuffle(directions)

    placed = _place_pairs(n_rows, n_cols, directions)
    if placed is None:
        return None

    colors = random.sample(PALETTE, n_pairs)
    inp = _empty_grid(n_rows, n_cols)
    pair_specs: list[tuple[int, str, tuple[int, int], tuple[int, int]]] = []
    for color, (direction, cell_a, cell_b) in zip(colors, placed):
        ra, ca = _seed_position(cell_a)
        rb, cb = _seed_position(cell_b)
        inp[ra, ca] = int(color)
        inp[rb, cb] = int(color)
        pair_specs.append((int(color), direction, cell_a, cell_b))

    expected = _expected_output(inp, pair_specs, n_rows, n_cols)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
