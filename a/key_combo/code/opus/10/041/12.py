from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _dash_period(length: int) -> int:
    return 2 + (int(length) % 3)


def _border_marker_groups(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    groups: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            color = int(grid[r, c])
            if color in (BLACK, GREY):
                continue
            groups[color].append((r, c))
    return groups


def _detect_corridors(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    corridors: list[dict[str, int]] = []
    for color, coords in sorted(_border_marker_groups(g).items()):
        if len(coords) != 2:
            continue
        (r1, c1), (r2, c2) = sorted(coords)
        if r1 == r2 and {c1, c2} == {0, w - 1} and np.all(g[r1, 1 : w - 1] == BLACK):
            corridors.append(
                {
                    "color": int(color),
                    "axis": 0,  # horizontal
                    "anchor": int(r1),
                    "length": int(w - 2),
                }
            )
        elif c1 == c2 and {r1, r2} == {0, h - 1} and np.all(g[1 : h - 1, c1] == BLACK):
            corridors.append(
                {
                    "color": int(color),
                    "axis": 1,  # vertical
                    "anchor": int(c1),
                    "length": int(h - 2),
                }
            )
    return corridors


def _paint_corridor(
    out: np.ndarray,
    axis: int,
    anchor: int,
    color: int,
    reverse: bool = False,
    fixed_period: int | None = None,
) -> None:
    h, w = map(int, out.shape)
    if axis == 0:
        cells = [(anchor, c) for c in range(1, w - 1)]
    else:
        cells = [(r, anchor) for r in range(1, h - 1)]

    if reverse:
        cells = list(reversed(cells))

    period = int(fixed_period if fixed_period is not None else _dash_period(len(cells)))
    threshold = period // 2
    for idx, (r, c) in enumerate(cells):
        if idx % period < threshold:
            out[r, c] = int(color)


def _render_output(
    grid: np.ndarray,
    corridors: list[dict[str, int]] | None = None,
    reverse: bool = False,
    fixed_period: int | None = None,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for corridor in _detect_corridors(g) if corridors is None else corridors:
        _paint_corridor(
            out,
            axis=int(corridor["axis"]),
            anchor=int(corridor["anchor"]),
            color=int(corridor["color"]),
            reverse=reverse,
            fixed_period=fixed_period,
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each clear border-to-border corridor with a dash pattern whose period is determined by that corridor's length.

    Input structure:
    - The grid has a black background and may contain grey blocker cells in the interior.
    - Distinct non-grey marker colors appear on opposite outer edges, one pair per color.
    - Each valid pair lies on one shared row or one shared column, and every interior cell on that span is black.

    Rule:
    - Pair border markers by color, keeping only opposite-edge pairs on the same row or column with an unobstructed black span between them.
    - For each such corridor, let L be the number of interior cells between the markers.
    - Convert L mod 3 into a dash period T: 0 -> 2, 1 -> 3, 2 -> 4.
    - Starting from the left marker for horizontal corridors or the top marker for vertical corridors, index the interior cells 0, 1, 2, ... and repaint those with index mod T < floor(T / 2) to the marker color.
    - Preserve the marker pixels themselves, all grey blockers, and all background outside the painted dash cells.

    Key insight:
    - The corridor is found geometrically from same-color opposite-edge markers, but the visible dash rhythm is not fixed: it is encoded by the corridor length, so the solver must count before painting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    return _render_output(g)


def _sample_spaced(values: list[int], count: int, min_gap: int) -> list[int] | None:
    if len(values) < count:
        return None
    for _ in range(120):
        chosen = sorted(random.sample(values, count))
        if all(b - a > min_gap for a, b in zip(chosen, chosen[1:])):
            return chosen
    return None


def _free_cell_mask(shape: tuple[int, int], rows: list[int], cols: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    free = np.ones((h, w), dtype=bool)
    free[0, :] = False
    free[h - 1, :] = False
    free[:, 0] = False
    free[:, w - 1] = False
    for r in rows:
        free[r, 1 : w - 1] = False
    for c in cols:
        free[1 : h - 1, c] = False
    return free


def _place_blockers(grid: np.ndarray, rows: list[int], cols: list[int]) -> None:
    h, w = map(int, grid.shape)
    free = _free_cell_mask(grid.shape, rows, cols)
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1) if free[r, c]]
    random.shuffle(candidates)

    target = random.randint(max(4, len(candidates) // 18), max(8, len(candidates) // 10))
    target = min(target, len(candidates))
    placed = 0

    def can_place(cells: list[tuple[int, int]]) -> bool:
        return all(0 < r < h - 1 and 0 < c < w - 1 and free[r, c] and grid[r, c] == BLACK for r, c in cells)

    while candidates and placed < target:
        r, c = candidates.pop()
        direction = random.choice(((0, 1), (1, 0), (0, -1), (-1, 0)))
        span = random.choice((1, 1, 2, 2, 3))
        cells = [(r + k * direction[0], c + k * direction[1]) for k in range(span)]
        if not can_place(cells):
            cells = [(r, c)]
            if not can_place(cells):
                continue
        for rr, cc in cells:
            grid[rr, cc] = GREY
            free[rr, cc] = False
            placed += 1
            if placed >= target:
                break


def _intended_corridors(
    h: int, w: int, row_colors: list[int], col_colors: list[int]
) -> list[dict[str, int]]:
    corridors: list[dict[str, int]] = []
    for r, color in row_colors:
        corridors.append({"color": int(color), "axis": 0, "anchor": int(r), "length": int(w - 2)})
    for c, color in col_colors:
        corridors.append({"color": int(color), "axis": 1, "anchor": int(c), "length": int(h - 2)})
    return sorted(corridors, key=lambda x: (x["axis"], x["anchor"], x["color"]))


def _semantic_checks(grid: np.ndarray, intended: list[dict[str, int]]) -> bool:
    detected = _detect_corridors(grid)
    if intended != sorted(detected, key=lambda x: (x["axis"], x["anchor"], x["color"])):
        return False

    periods = {int(_dash_period(c["length"])) for c in intended}
    if len(periods) < 2:
        return False

    expected = _render_output(grid, corridors=intended)
    if np.array_equal(expected, grid):
        return False

    if np.array_equal(expected, _render_output(grid, corridors=intended, reverse=True)):
        return False

    for fixed_period in (2, 3, 4):
        if np.array_equal(expected, _render_output(grid, corridors=intended, fixed_period=fixed_period)):
            return False

    h_rows = [int(c["anchor"]) for c in intended if int(c["axis"]) == 0]
    v_cols = [int(c["anchor"]) for c in intended if int(c["axis"]) == 1]
    if not h_rows or not v_cols:
        return False

    for r in h_rows:
        for c in v_cols:
            if expected[r, c] != BLACK:
                return False

    if int(np.count_nonzero(grid == GREY)) == 0:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    for _ in range(200):
        h = random.randint(12, 20)
        w = random.randint(12, 20)
        if h == w or (h - 2) % 3 == (w - 2) % 3:
            continue

        h_period = _dash_period(w - 2)
        v_period = _dash_period(h - 2)

        row_candidates = [r for r in range(1, h - 1) if (r - 1) % v_period >= v_period // 2]
        col_candidates = [c for c in range(1, w - 1) if (c - 1) % h_period >= h_period // 2]
        if len(row_candidates) < 1 or len(col_candidates) < 1:
            continue

        row_count = random.choice((1, 1, 2))
        col_count = random.choice((1, 1, 2))
        if row_count + col_count > 4:
            if random.random() < 0.5:
                row_count -= 1
            else:
                col_count -= 1

        rows = _sample_spaced(row_candidates, row_count, min_gap=1)
        cols = _sample_spaced(col_candidates, col_count, min_gap=1)
        if rows is None or cols is None:
            continue

        pair_count = row_count + col_count
        colors = random.sample(MARKER_COLORS, pair_count)
        row_colors = list(zip(rows, colors[:row_count]))
        col_colors = list(zip(cols, colors[row_count:]))

        grid = np.full((h, w), BLACK, dtype=int)
        for r, color in row_colors:
            grid[r, 0] = int(color)
            grid[r, w - 1] = int(color)
        for c, color in col_colors:
            grid[0, c] = int(color)
            grid[h - 1, c] = int(color)

        _place_blockers(grid, rows, cols)
        intended = _intended_corridors(h, w, row_colors, col_colors)

        if not _semantic_checks(grid, intended):
            continue
        if not np.array_equal(main(grid), _render_output(grid, corridors=intended)):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 12")
