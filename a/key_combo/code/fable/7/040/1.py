from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 2000


def _decode_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    """Decode inclusive (r0, c0, r1, c1) rectangles from the two marker edges."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a non-empty 2D canvas"
    assert int(g[0, 0]) == BACKGROUND, "the shared edge corner must be background"
    assert np.all(g[1:, 1:] == BACKGROUND), "the canvas interior must be empty"

    top_colors = set(map(int, g[0, 1:])) - {BACKGROUND}
    left_colors = set(map(int, g[1:, 0])) - {BACKGROUND}
    assert top_colors and top_colors == left_colors, "both edges must encode the same colors"

    rectangles: list[dict[str, int]] = []
    for color in sorted(top_colors):
        cols = np.flatnonzero(g[0, :] == color)
        rows = np.flatnonzero(g[:, 0] == color)
        assert len(cols) == 2 and len(rows) == 2, "each color needs two markers on each edge"
        r0, r1 = map(int, rows)
        c0, c1 = map(int, cols)
        assert 0 < r0 < r1 and 0 < c0 < c1, "markers must bound an internal rectangle"
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        rectangles.append(
            {"color": color, "r0": r0, "c0": c0, "r1": r1, "c1": c1, "area": area}
        )

    areas = [rect["area"] for rect in rectangles]
    assert len(areas) == len(set(areas)), "rectangle areas must be pairwise distinct"
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """Restore edge-encoded solid rectangles and stack smaller areas on top.

    Input structure:
    - The background is black and the interior is completely empty.
    - Every foreground color occurs twice on the top row and twice on the left
      column; those markers give two column and two row coordinates.
    - The inclusive rectangles encoded by the colors have distinct areas.

    Rule:
    - Decode each color's inclusive row and column bounds, then start with a
      black canvas of the same size.
    - Paint all solid rectangles from largest area to smallest area, so every
      smaller rectangle covers larger rectangles where they overlap.
    - Do not copy any top-row or left-column coordinate markers to the output.

    Key insight:
    - The rectangles are absent from the input: both edge projections must be
      combined first, and all areas compared before overlap visibility is known.
    """
    g = np.asarray(grid, dtype=int)
    rectangles = _decode_rectangles(g)
    out = np.full(g.shape, BACKGROUND, dtype=int)
    for rect in sorted(rectangles, key=lambda item: item["area"], reverse=True):
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return out


def _intersection_area(a: dict[str, int], b: dict[str, int]) -> int:
    height = min(a["r1"], b["r1"]) - max(a["r0"], b["r0"]) + 1
    width = min(a["c1"], b["c1"]) - max(a["c0"], b["c0"]) + 1
    return max(0, height) * max(0, width)


def _render(shape: tuple[int, int], rectangles: list[dict[str, int]], order: list[int]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for index in order:
        rect = rectangles[index]
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return out


def _connected_overlap_graph(rectangles: list[dict[str, int]]) -> tuple[bool, int]:
    count = len(rectangles)
    adjacency = [set() for _ in rectangles]
    edges = 0
    for i in range(count):
        for j in range(i + 1, count):
            if _intersection_area(rectangles[i], rectangles[j]) > 0:
                adjacency[i].add(j)
                adjacency[j].add(i)
                edges += 1
    seen = {0}
    frontier = [0]
    while frontier:
        current = frontier.pop()
        for neighbor in adjacency[current] - seen:
            seen.add(neighbor)
            frontier.append(neighbor)
    return len(seen) == count, edges


def _semantically_strong(shape: tuple[int, int], rectangles: list[dict[str, int]]) -> bool:
    """Reject degenerate scenes that fail to demonstrate decoding or area stacking."""
    areas = [rect["area"] for rect in rectangles]
    if len(set(areas)) != len(areas):
        return False

    connected, edge_count = _connected_overlap_graph(rectangles)
    if not connected or edge_count < 2:
        return False

    coverage = np.zeros(shape, dtype=int)
    for rect in rectangles:
        coverage[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] += 1
    union = int(np.count_nonzero(coverage))
    overlap = int(np.count_nonzero(coverage >= 2))
    if union == 0 or overlap < 2 or overlap * 4 > union:
        return False

    area_order = sorted(range(len(rectangles)), key=lambda i: rectangles[i]["area"], reverse=True)
    correct = _render(shape, rectangles, area_order)
    reverse = _render(shape, rectangles, list(reversed(area_order)))
    color_order = _render(shape, rectangles, sorted(range(len(rectangles)), key=lambda i: rectangles[i]["color"]))
    if np.count_nonzero(correct != reverse) < 2 or np.array_equal(correct, color_order):
        return False

    # Every decoded rectangle remains visibly represented after all covering.
    if any(not np.any(correct == rect["color"]) for rect in rectangles):
        return False
    return True


def _sample_rectangles(height: int, width: int, count: int) -> list[dict[str, int]] | None:
    colors = random.sample(PALETTE, count)
    # Globally distinct edge coordinates ensure no marker pixel has two meanings.
    row_coords = random.sample(range(1, height), 2 * count)
    col_coords = random.sample(range(1, width), 2 * count)
    random.shuffle(row_coords)
    random.shuffle(col_coords)

    rectangles: list[dict[str, int]] = []
    for i, color in enumerate(colors):
        r0, r1 = sorted(row_coords[2 * i : 2 * i + 2])
        c0, c1 = sorted(col_coords[2 * i : 2 * i + 2])
        # Avoid visually ambiguous one-cell-thick rectangles.
        if r1 - r0 < 1 or c1 - c0 < 1:
            return None
        rectangles.append(
            {
                "color": color,
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "area": (r1 - r0 + 1) * (c1 - c0 + 1),
            }
        )
    return rectangles


def _build_input(shape: tuple[int, int], rectangles: list[dict[str, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for rect in rectangles:
        grid[0, [rect["c0"], rect["c1"]]] = rect["color"]
        grid[[rect["r0"], rect["r1"]], 0] = rect["color"]
    return grid


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate edge-marker rectangle scenes."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        count = random.randint(3, 5)
        height = random.randint(max(14, 2 * count + 3), 30)
        width = random.randint(max(14, 2 * count + 3), 30)
        rectangles = _sample_rectangles(height, width, count)
        if rectangles is None or not _semantically_strong((height, width), rectangles):
            continue

        grid = _build_input((height, width), rectangles)
        expected_order = sorted(range(count), key=lambda i: rectangles[i]["area"], reverse=True)
        expected = _render((height, width), rectangles, expected_order)
        try:
            decoded = _decode_rectangles(grid)
            actual = main(grid)
        except AssertionError:
            continue
        if len(decoded) != count or not np.array_equal(actual, expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid edge-marker rectangle scene")
