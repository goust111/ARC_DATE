from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6

PLUS = frozenset({(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)})
BAR = frozenset({(0, 0), (0, 1), (0, 2)})
CORNERS = {
    frozenset({(0, 0), (0, 1), (1, 0)}),
    frozenset({(0, 0), (0, 1), (1, 1)}),
    frozenset({(0, 0), (1, 0), (1, 1)}),
    frozenset({(0, 1), (1, 0), (1, 1)}),
}
SHAPE_TO_COLOR = {"plus": RED, "bar": BLUE, "corner": GREEN}

GLYPH_MASKS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    "bar": np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=int),
    "corner": (
        np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=int),
        np.array([[0, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=int),
    ),
}

MAX_RETRIES = 400


def _full_grey_rows(g: np.ndarray) -> list[int]:
    return [r for r in range(g.shape[0]) if np.all(g[r, :] == GREY)]


def _full_grey_cols(g: np.ndarray) -> list[int]:
    return [c for c in range(g.shape[1]) if np.all(g[:, c] == GREY)]


def _intervals(length: int, separators: Iterable[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    prev = -1
    for cut in [*sorted(int(v) for v in separators), length]:
        if cut - prev > 1:
            spans.append((prev + 1, cut - 1))
        prev = cut
    return spans


def _region_grid(g: np.ndarray) -> list[list[tuple[int, int, int, int]]]:
    row_spans = _intervals(g.shape[0], _full_grey_rows(g))
    col_spans = _intervals(g.shape[1], _full_grey_cols(g))
    return [[(r0, c0, r1, c1) for c0, c1 in col_spans] for r0, r1 in row_spans]


def _normalize_shape(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _decode_glyph(subgrid: np.ndarray) -> int | None:
    coords = np.argwhere(subgrid == PINK)
    if coords.size == 0:
        return None
    shape = _normalize_shape(coords)
    if shape == PLUS:
        return RED
    if shape == BAR:
        return BLUE
    if shape in CORNERS:
        return GREEN
    raise AssertionError("unknown glyph shape")


def _center2(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r0 + r1, c0 + c1


def _solve_regions(
    regions: list[list[tuple[int, int, int, int]]],
    seed_colors: dict[tuple[int, int], int],
) -> dict[tuple[int, int], int]:
    seeded = [((rr, cc), _center2(regions[rr][cc]), color) for (rr, cc), color in seed_colors.items()]
    assert seeded, "expected at least one seeded region"

    fills: dict[tuple[int, int], int] = {}
    for rr, row in enumerate(regions):
        for cc, box in enumerate(row):
            if (rr, cc) in seed_colors:
                fills[(rr, cc)] = seed_colors[(rr, cc)]
                continue

            cr2, cc2 = _center2(box)
            distances = [
                (abs(cr2 - sr2) + abs(cc2 - sc2), color)
                for (_idx, (sr2, sc2), color) in seeded
            ]
            best = min(dist for dist, _color in distances)
            nearest_colors = {color for dist, color in distances if dist == best}
            fills[(rr, cc)] = next(iter(nearest_colors)) if len(nearest_colors) == 1 else YELLOW
    return fills


def _render_output(
    grid_shape: tuple[int, int],
    regions: list[list[tuple[int, int, int, int]]],
    seed_colors: dict[tuple[int, int], int],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    for r in _full_grey_rows(out):
        out[r, :] = GREY
    for c in _full_grey_cols(out):
        out[:, c] = GREY

    fills = _solve_regions(regions, seed_colors)
    for rr, row in enumerate(regions):
        for cc, (r0, c0, r1, c1) in enumerate(row):
            out[r0 : r1 + 1, c0 : c1 + 1] = fills[(rr, cc)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode magenta glyphs into region colors, flood their regions, then propagate those colors to blank regions by nearest seeded center.

    Input structure:
    - A black background is split into rectangular regions by full-row and full-column gray divider lines.
    - Some regions contain one small magenta glyph; other regions are empty black rectangles.
    - The glyph is a 4-connected plus, horizontal bar, or corner shape placed away from the gray dividers.

    Rule:
    - Partition the grid by the gray lines and inspect each region for a magenta glyph.
    - Decode glyph shape to color: plus -> red, horizontal bar -> blue, corner -> green, then fill that whole region with the decoded color and erase the glyph into the fill.
    - Fill every empty region with the color of the nearest seeded region center by Manhattan distance; if the nearest distance ties between different colors, fill that region yellow. Keep the gray dividers unchanged.

    Key insight:
    - The glyph shape determines seed colors locally, and those seeded regions then act as centers for a nearest-region partition over the remaining blank rectangles.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions = _region_grid(g)
    seed_colors: dict[tuple[int, int], int] = {}
    for rr, row in enumerate(regions):
        for cc, (r0, c0, r1, c1) in enumerate(row):
            color = _decode_glyph(g[r0 : r1 + 1, c0 : c1 + 1])
            if color is not None:
                seed_colors[(rr, cc)] = color

    out = np.where(g == GREY, GREY, BLACK)
    fills = _solve_regions(regions, seed_colors)
    for rr, row in enumerate(regions):
        for cc, (r0, c0, r1, c1) in enumerate(row):
            out[r0 : r1 + 1, c0 : c1 + 1] = fills[(rr, cc)]
    return out


def _build_partition(heights: list[int], widths: list[int]) -> tuple[np.ndarray, list[list[tuple[int, int, int, int]]]]:
    h = sum(heights) + len(heights) - 1
    w = sum(widths) + len(widths) - 1
    grid = np.zeros((h, w), dtype=int)

    r = 0
    row_spans: list[tuple[int, int]] = []
    for idx, height in enumerate(heights):
        row_spans.append((r, r + height - 1))
        r += height
        if idx != len(heights) - 1:
            grid[r, :] = GREY
            r += 1

    c = 0
    col_spans: list[tuple[int, int]] = []
    for idx, width in enumerate(widths):
        col_spans.append((c, c + width - 1))
        c += width
        if idx != len(widths) - 1:
            grid[:, c] = GREY
            c += 1

    regions = [[(r0, c0, r1, c1) for c0, c1 in col_spans] for r0, r1 in row_spans]
    return grid, regions


def _place_glyph(grid: np.ndarray, box: tuple[int, int, int, int], shape: str) -> None:
    r0, c0, r1, c1 = box
    assert r1 - r0 + 1 >= 5 and c1 - c0 + 1 >= 5
    top = (r0 + r1) // 2 - 1
    left = (c0 + c1) // 2 - 1
    pattern = GLYPH_MASKS[shape]
    if shape == "corner":
        pattern = random.choice(pattern)
    mask = np.asarray(pattern, dtype=bool)
    grid[top : top + 3, left : left + 3][mask] = PINK


def _candidate_extra_positions(
    rows: int,
    cols: int,
    required: set[tuple[int, int]],
    heights: list[int],
    widths: list[int],
) -> list[tuple[int, int]]:
    out = []
    for rr in range(rows):
        for cc in range(cols):
            if (rr, cc) in required:
                continue
            if heights[rr] >= 5 and widths[cc] >= 5:
                out.append((rr, cc))
    return out


def _choose_layout() -> tuple[list[int], list[int], list[tuple[int, int]], bool]:
    family = random.choices(
        ["horizontal_tie", "vertical_tie", "plain"],
        weights=[4, 4, 3],
        k=1,
    )[0]

    if family == "horizontal_tie":
        heights = [random.choice([5, 7]), random.choice([5, 7])]
        widths = [random.choice([5, 7]), random.choice([3, 5]), random.choice([5, 7])]
        tie_row = random.randrange(2)
        required = {(tie_row, 0), (tie_row, 2)}
        forbidden = required | {(tie_row, 1)}
        extras = _candidate_extra_positions(2, 3, forbidden, heights, widths)
        seed_positions = list(required)
        if extras and random.random() < 0.65:
            seed_positions.append(random.choice(extras))
        return heights, widths, seed_positions, True

    if family == "vertical_tie":
        heights = [random.choice([5, 7]), random.choice([3, 5]), random.choice([5, 7])]
        widths = [random.choice([5, 7]), random.choice([5, 7])]
        tie_col = random.randrange(2)
        required = {(0, tie_col), (2, tie_col)}
        forbidden = required | {(1, tie_col)}
        extras = _candidate_extra_positions(3, 2, forbidden, heights, widths)
        seed_positions = list(required)
        if extras and random.random() < 0.65:
            seed_positions.append(random.choice(extras))
        return heights, widths, seed_positions, True

    variant = random.choice(["2x2", "2x3", "3x2"])
    if variant == "2x2":
        heights = [random.choice([5, 7]), random.choice([5, 7])]
        widths = [random.choice([5, 7]), random.choice([5, 7])]
    elif variant == "2x3":
        heights = [random.choice([5, 7]), random.choice([5, 7])]
        widths = [random.choice([5, 7]), random.choice([3, 5]), random.choice([5, 7])]
    else:
        heights = [random.choice([5, 7]), random.choice([3, 5]), random.choice([5, 7])]
        widths = [random.choice([5, 7]), random.choice([5, 7])]

    rows, cols = len(heights), len(widths)
    candidates = [(rr, cc) for rr in range(rows) for cc in range(cols) if heights[rr] >= 5 and widths[cc] >= 5]
    seed_count = 3 if len(candidates) >= 3 and random.random() < 0.55 else 2
    return heights, widths, random.sample(candidates, seed_count), False


def _expected_from_metadata(
    inp: np.ndarray,
    regions: list[list[tuple[int, int, int, int]]],
    seed_shapes: dict[tuple[int, int], str],
) -> np.ndarray:
    out = np.where(inp == GREY, GREY, BLACK)
    seed_colors = {pos: SHAPE_TO_COLOR[shape] for pos, shape in seed_shapes.items()}
    fills = _solve_regions(regions, seed_colors)
    for rr, row in enumerate(regions):
        for cc, (r0, c0, r1, c1) in enumerate(row):
            out[r0 : r1 + 1, c0 : c1 + 1] = fills[(rr, cc)]
    return out


def _attempt() -> np.ndarray | None:
    heights, widths, seed_positions, expect_yellow = _choose_layout()
    grid, regions = _build_partition(heights, widths)

    shapes = random.sample(["plus", "bar", "corner"], len(seed_positions))
    seed_shapes = dict(zip(seed_positions, shapes))
    for pos, shape in seed_shapes.items():
        rr, cc = pos
        _place_glyph(grid, regions[rr][cc], shape)

    expected = _expected_from_metadata(grid, regions, seed_shapes)
    if expect_yellow and not np.any(expected == YELLOW):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-4 input")
