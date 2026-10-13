from __future__ import annotations

import itertools
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
MARKER_COLORS = (BLUE, RED, GREEN, YELLOW)


def _divider_indices(grid: np.ndarray, axis: int) -> list[int]:
    if axis == 0:
        return [r for r in range(grid.shape[0]) if np.all(grid[r, :] == MAGENTA)]
    return [c for c in range(grid.shape[1]) if np.all(grid[:, c] == MAGENTA)]


def _intervals(length: int, dividers: Iterable[int]) -> list[tuple[int, int]]:
    points = sorted(int(v) for v in dividers)
    spans: list[tuple[int, int]] = []
    prev = -1
    for stop in [*points, length]:
        if stop - prev > 1:
            spans.append((prev + 1, stop - 1))
        prev = stop
    assert spans, "expected at least one cell interval"
    return spans


def _cell_bounds(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    row_spans = _intervals(grid.shape[0], _divider_indices(grid, axis=0))
    col_spans = _intervals(grid.shape[1], _divider_indices(grid, axis=1))
    return row_spans, col_spans


def _parse_cell(cell: np.ndarray) -> dict[str, int]:
    markers = np.argwhere((cell != BLACK) & (cell != MAGENTA))
    assert markers.shape[0] == 2, "each cell must contain exactly two colored markers"

    (r0, c0), (r1, c1) = [tuple(map(int, coord)) for coord in markers]
    color0 = int(cell[r0, c0])
    color1 = int(cell[r1, c1])
    assert color0 in MARKER_COLORS and color1 in MARKER_COLORS and color0 != color1

    h, w = cell.shape
    if r0 == r1:
        assert {c0, c1} == {0, w - 1}, "horizontal markers must sit on opposite cell edges"
        if c0 < c1:
            left_c, right_c = c0, c1
            left_color, right_color = color0, color1
        else:
            left_c, right_c = c1, c0
            left_color, right_color = color1, color0
        return {
            "orientation": 0,
            "line": r0,
            "near": left_c,
            "far": right_c,
            "near_color": left_color,
            "far_color": right_color,
        }

    assert c0 == c1, "markers must share either a row or a column"
    assert {r0, r1} == {0, h - 1}, "vertical markers must sit on opposite cell edges"
    if r0 < r1:
        top_r, bottom_r = r0, r1
        top_color, bottom_color = color0, color1
    else:
        top_r, bottom_r = r1, r0
        top_color, bottom_color = color1, color0
    return {
        "orientation": 1,
        "line": c0,
        "near": top_r,
        "far": bottom_r,
        "near_color": top_color,
        "far_color": bottom_color,
    }


def _anchor(shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    return (h - 1) // 2, (w - 1) // 2


def _simulate_cell(cell: np.ndarray) -> dict[str, object]:
    info = _parse_cell(cell)
    h, w = cell.shape
    anchor_r, anchor_c = _anchor((h, w))
    paint = np.zeros_like(cell)

    if int(info["orientation"]) == 0:
        row = int(info["line"])
        left_c = int(info["near"])
        right_c = int(info["far"])
        left_color = int(info["near_color"])
        right_color = int(info["far_color"])
        while True:
            if left_c == right_c:
                paint[row, left_c] = GREY
                return {
                    "compressed": GREY,
                    "paint": paint,
                    "orientation": "h",
                    "branch": "meet",
                }
            if right_c - left_c == 1:
                paint[row, left_c] = left_color
                paint[row, right_c] = right_color
                d_left = abs(anchor_r - row) + abs(anchor_c - left_c)
                d_right = abs(anchor_r - row) + abs(anchor_c - right_c)
                assert d_left != d_right, "compressed color must be uniquely determined"
                return {
                    "compressed": left_color if d_left < d_right else right_color,
                    "paint": paint,
                    "orientation": "h",
                    "branch": "adjacent",
                }
            paint[row, left_c] = left_color
            paint[row, right_c] = right_color
            left_c += 1
            right_c -= 1

    col = int(info["line"])
    top_r = int(info["near"])
    bottom_r = int(info["far"])
    top_color = int(info["near_color"])
    bottom_color = int(info["far_color"])
    while True:
        if top_r == bottom_r:
            paint[top_r, col] = GREY
            return {
                "compressed": GREY,
                "paint": paint,
                "orientation": "v",
                "branch": "meet",
            }
        if bottom_r - top_r == 1:
            paint[top_r, col] = top_color
            paint[bottom_r, col] = bottom_color
            d_top = abs(anchor_r - top_r) + abs(anchor_c - col)
            d_bottom = abs(anchor_r - bottom_r) + abs(anchor_c - col)
            assert d_top != d_bottom, "compressed color must be uniquely determined"
            return {
                "compressed": top_color if d_top < d_bottom else bottom_color,
                "paint": paint,
                "orientation": "v",
                "branch": "adjacent",
            }
        paint[top_r, col] = top_color
        paint[bottom_r, col] = bottom_color
        top_r += 1
        bottom_r -= 1


def _oracle_cell(cell: np.ndarray) -> int:
    info = _parse_cell(cell)
    h, w = cell.shape
    anchor_r, anchor_c = _anchor((h, w))

    if int(info["orientation"]) == 0:
        row = int(info["line"])
        left_c = int(info["near"])
        right_c = int(info["far"])
        if (right_c - left_c) % 2 == 0:
            return GREY
        final_left = (left_c + right_c) // 2
        final_right = final_left + 1
        d_left = abs(anchor_r - row) + abs(anchor_c - final_left)
        d_right = abs(anchor_r - row) + abs(anchor_c - final_right)
        assert d_left != d_right
        return int(info["near_color"]) if d_left < d_right else int(info["far_color"])

    col = int(info["line"])
    top_r = int(info["near"])
    bottom_r = int(info["far"])
    if (bottom_r - top_r) % 2 == 0:
        return GREY
    final_top = (top_r + bottom_r) // 2
    final_bottom = final_top + 1
    d_top = abs(anchor_r - final_top) + abs(anchor_c - col)
    d_bottom = abs(anchor_r - final_bottom) + abs(anchor_c - col)
    assert d_top != d_bottom
    return int(info["near_color"]) if d_top < d_bottom else int(info["far_color"])


def _oracle_output(grid: np.ndarray) -> np.ndarray:
    row_spans, col_spans = _cell_bounds(grid)
    out = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            out[rr, cc] = _oracle_cell(grid[r0 : r1 + 1, c0 : c1 + 1])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each magenta-separated cell to the color produced by opposite-side markers sliding toward one another.

    Input structure:
    - The grid uses black background and full magenta rows/columns as dividers that split the canvas into rectangular cells.
    - Each cell contains exactly two distinct non-black, non-magenta marker pixels chosen from blue, red, green, and yellow.
    - The two markers lie on opposite edges of their cell and share one row or one column, defining a horizontal or vertical lane.

    Rule:
    - Detect the full magenta divider rows and columns to recover the rectangular cell layout.
    - Inside each cell, move the two edge markers simultaneously along their shared row or column, leaving trails in their own colors.
    - If they land on the same cell, that meeting cell becomes gray and the compressed output for that region is gray; if they stop as adjacent cells, compress the region to the color of the final marker closer to the cell's discrete center anchor.

    Key insight:
    - The divider geometry defines the local coordinate frame for every pair, and the parity of the lane length determines whether the inward slide yields a gray meeting point or one surviving side color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    row_spans, col_spans = _cell_bounds(g)
    out = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            sim = _simulate_cell(g[r0 : r1 + 1, c0 : c1 + 1])
            out[rr, cc] = int(sim["compressed"])
    return out


def _mixed_sizes(count: int, rng: np.random.Generator) -> list[int]:
    sizes = [int(rng.choice([4, 5])) for _ in range(count)]
    if all(v == sizes[0] for v in sizes):
        flip = int(rng.integers(0, count))
        sizes[flip] = 9 - sizes[flip]
    return sizes


def _pick_coverage_cells(
    row_sizes: list[int], col_sizes: list[int], rng: np.random.Generator
) -> dict[str, tuple[int, int]] | None:
    n = len(row_sizes)
    horiz_grey = [(r, c) for r in range(n) for c in range(n) if col_sizes[c] == 5]
    horiz_adj = [(r, c) for r in range(n) for c in range(n) if col_sizes[c] == 4]
    vert_grey = [(r, c) for r in range(n) for c in range(n) if row_sizes[r] == 5]
    vert_adj = [(r, c) for r in range(n) for c in range(n) if row_sizes[r] == 4]

    rng.shuffle(horiz_grey)
    rng.shuffle(horiz_adj)
    rng.shuffle(vert_grey)
    rng.shuffle(vert_adj)

    for a, b, c, d in itertools.product(horiz_grey, horiz_adj, vert_grey, vert_adj):
        picks = [a, b, c, d]
        if len(set(picks)) == 4:
            return {
                "h_grey": a,
                "h_adj": b,
                "v_grey": c,
                "v_adj": d,
            }
    return None


def _build_grid(row_sizes: list[int], col_sizes: list[int]) -> tuple[np.ndarray, list[int], list[int]]:
    n = len(row_sizes)
    h = sum(row_sizes) + (n - 1)
    w = sum(col_sizes) + (n - 1)
    grid = np.zeros((h, w), dtype=int)

    row_starts = []
    pos = 0
    for i, size in enumerate(row_sizes):
        row_starts.append(pos)
        pos += size
        if i < n - 1:
            grid[pos, :] = MAGENTA
            pos += 1

    col_starts = []
    pos = 0
    for i, size in enumerate(col_sizes):
        col_starts.append(pos)
        pos += size
        if i < n - 1:
            grid[:, pos] = MAGENTA
            pos += 1
    return grid, row_starts, col_starts


def _place_marker_pair(
    grid: np.ndarray,
    r0: int,
    c0: int,
    h: int,
    w: int,
    orientation: str,
    colors: tuple[int, int],
    rng: np.random.Generator,
) -> None:
    color_a, color_b = map(int, colors)
    if orientation == "h":
        row = int(rng.integers(0, h))
        grid[r0 + row, c0 + 0] = color_a
        grid[r0 + row, c0 + w - 1] = color_b
        return
    col = int(rng.integers(0, w))
    grid[r0 + 0, c0 + col] = color_a
    grid[r0 + h - 1, c0 + col] = color_b


def _validate_generated_case(grid: np.ndarray) -> None:
    row_spans, col_spans = _cell_bounds(grid)
    assert len(row_spans) == len(col_spans) and len(row_spans) in {3, 4}
    row_sizes = [r1 - r0 + 1 for r0, r1 in row_spans]
    col_sizes = [c1 - c0 + 1 for c0, c1 in col_spans]
    assert set(row_sizes) <= {4, 5} and set(col_sizes) <= {4, 5}
    assert 4 in row_sizes and 5 in row_sizes and 4 in col_sizes and 5 in col_sizes

    details: list[tuple[str, str, tuple[int, int], int]] = []
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            parsed = _parse_cell(cell)
            sim = _simulate_cell(cell)
            oracle = _oracle_cell(cell)
            assert int(sim["compressed"]) == oracle

            markers = sorted(int(v) for v in cell[(cell != BLACK) & (cell != MAGENTA)].tolist())
            details.append((str(sim["orientation"]), str(sim["branch"]), tuple(markers), int(sim["compressed"])))

            if int(parsed["orientation"]) == 0:
                assert cell.shape[1] in {4, 5}
            else:
                assert cell.shape[0] in {4, 5}

    out_main = main(grid)
    out_oracle = _oracle_output(grid)
    assert np.array_equal(out_main, out_oracle), "main must match the independent oracle"
    assert GREY in out_main and np.any(out_main != GREY), "need both meeting and non-meeting branches"
    assert "h" in {d[0] for d in details} and "v" in {d[0] for d in details}, "need both orientations"
    assert len(set(map(int, out_main.flatten()))) >= 3, "output should not collapse to one or two colors"
    assert any(d1[2] == d2[2] and d1[3] != d2[3] for d1, d2 in itertools.combinations(details, 2)), (
        "same color pairs should sometimes yield different outputs so geometry matters"
    )


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        n = int(rng.choice([3, 4]))
        row_sizes = _mixed_sizes(n, rng)
        col_sizes = _mixed_sizes(n, rng)
        coverage = _pick_coverage_cells(row_sizes, col_sizes, rng)
        if coverage is None:
            continue

        grid, row_starts, col_starts = _build_grid(row_sizes, col_sizes)

        palette = list(rng.permutation(MARKER_COLORS))
        pair_a = (int(palette[0]), int(palette[1]))
        pair_b = (int(palette[2]), int(palette[3]))

        cell_specs: dict[tuple[int, int], tuple[str, tuple[int, int]]] = {}
        cell_specs[coverage["h_grey"]] = ("h", pair_a)
        cell_specs[coverage["h_adj"]] = ("h", pair_a)
        cell_specs[coverage["v_grey"]] = ("v", pair_b)
        cell_specs[coverage["v_adj"]] = ("v", pair_b)

        all_pairs = [(a, b) for a in MARKER_COLORS for b in MARKER_COLORS if a != b]
        for r in range(n):
            for c in range(n):
                if (r, c) not in cell_specs:
                    orientation = str(rng.choice(["h", "v"]))
                    pair = all_pairs[int(rng.integers(0, len(all_pairs)))]
                    cell_specs[(r, c)] = (orientation, pair)

        for r in range(n):
            for c in range(n):
                r0 = row_starts[r]
                c0 = col_starts[c]
                h = row_sizes[r]
                w = col_sizes[c]
                orientation, pair = cell_specs[(r, c)]
                _place_marker_pair(grid, r0, c0, h, w, orientation, pair, rng)

        try:
            _validate_generated_case(grid)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
