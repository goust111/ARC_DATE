from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 200
MIN_SIDE_A = 8
MIN_SIDE_B = 10
MAX_SIDE = 22
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _sorted_line_points(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int], int]:
    if len(points) != 3:
        raise AssertionError("each color must have exactly three marker pixels")

    rows = {r for r, _ in points}
    cols = {c for _, c in points}

    if len(rows) == 1:
        ordered = sorted(points, key=lambda p: p[1])
        d1 = ordered[1][1] - ordered[0][1]
        d2 = ordered[2][1] - ordered[1][1]
        if d1 <= 0 or d1 != d2:
            raise AssertionError("horizontal markers must be equally spaced")
        return ordered, (0, 1), d1

    if len(cols) == 1:
        ordered = sorted(points, key=lambda p: p[0])
        d1 = ordered[1][0] - ordered[0][0]
        d2 = ordered[2][0] - ordered[1][0]
        if d1 <= 0 or d1 != d2:
            raise AssertionError("vertical markers must be equally spaced")
        return ordered, (1, 0), d1

    diffs = {r - c for r, c in points}
    if len(diffs) == 1:
        ordered = sorted(points, key=lambda p: (p[0], p[1]))
        dr1 = ordered[1][0] - ordered[0][0]
        dc1 = ordered[1][1] - ordered[0][1]
        dr2 = ordered[2][0] - ordered[1][0]
        dc2 = ordered[2][1] - ordered[1][1]
        if dr1 <= 0 or dr1 != dr2 or dc1 != dc2 or dr1 != dc1:
            raise AssertionError("down-right diagonal markers must be equally spaced")
        return ordered, (1, 1), dr1

    sums = {r + c for r, c in points}
    if len(sums) == 1:
        ordered = sorted(points, key=lambda p: (p[0], -p[1]))
        dr1 = ordered[1][0] - ordered[0][0]
        dc1 = ordered[0][1] - ordered[1][1]
        dr2 = ordered[2][0] - ordered[1][0]
        dc2 = ordered[1][1] - ordered[2][1]
        if dr1 <= 0 or dr1 != dr2 or dc1 != dc2 or dr1 != dc1:
            raise AssertionError("down-left diagonal markers must be equally spaced")
        return ordered, (1, -1), dr1

    raise AssertionError("markers must lie on one horizontal, vertical, or 45-degree diagonal line")


def _extract_lines(grid: np.ndarray) -> list[dict[str, object]]:
    colors = [int(c) for c in np.unique(grid) if int(c) != BACKGROUND]
    if not (2 <= len(colors) <= 4):
        raise AssertionError("expected 2 to 4 marker colors")

    lines: list[dict[str, object]] = []
    for color in colors:
        coords = [tuple(map(int, p)) for p in np.argwhere(grid == color)]
        ordered, step, spacing = _sorted_line_points(coords)
        lines.append(
            {
                "color": color,
                "points": ordered,
                "step": step,
                "spacing": spacing,
            }
        )

    lines.sort(key=lambda item: tuple(item["points"][0]))  # stable, geometry-based order
    return lines


def _full_line_cells(shape: tuple[int, int], line: dict[str, object]) -> list[tuple[int, int]]:
    points = line["points"]
    dr, dc = line["step"]
    r, c = points[0]
    while _in_bounds(shape, r - dr, c - dc):
        r -= dr
        c -= dc

    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _build_walls(grid: np.ndarray, lines: list[dict[str, object]]) -> tuple[np.ndarray, int]:
    wall = np.zeros_like(grid)
    for line in lines:
        color = int(line["color"])
        for r, c in line["points"]:
            if wall[r, c] not in (BACKGROUND, color):
                raise AssertionError("marker colors must not overlap")
            wall[r, c] = color

    skipped = 0
    for line in lines:
        color = int(line["color"])
        for r, c in _full_line_cells(grid.shape, line):
            if wall[r, c] == BACKGROUND:
                wall[r, c] = color
            elif wall[r, c] != color:
                skipped += 1
    return wall, skipped


def _exterior_background(wall: np.ndarray) -> np.ndarray:
    h, w = wall.shape
    exterior = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if wall[r, c] == BACKGROUND and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if wall[r, c] == BACKGROUND and not exterior[r, c]:
                exterior[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(wall.shape, nr, nc) and wall[nr, nc] == BACKGROUND and not exterior[nr, nc]:
                exterior[nr, nc] = True
                q.append((nr, nc))
    return exterior


def _enclosed_regions(wall: np.ndarray) -> list[dict[str, object]]:
    h, w = wall.shape
    exterior = _exterior_background(wall)
    seen = exterior | (wall != BACKGROUND)
    regions: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            boundary: set[tuple[int, int]] = set()

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if not _in_bounds(wall.shape, nr, nc):
                        continue
                    if wall[nr, nc] == BACKGROUND:
                        if not seen[nr, nc]:
                            seen[nr, nc] = True
                            q.append((nr, nc))
                    else:
                        boundary.add((nr, nc))

            counts = Counter(int(wall[rr, cc]) for rr, cc in boundary)
            regions.append({"cells": cells, "boundary_counts": counts})
    return regions


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    if np.any(g < 0) or np.any(g > 9):
        raise AssertionError("ARC colors must stay in 0..9")

    lines = _extract_lines(g)
    wall, skipped = _build_walls(g, lines)
    regions = _enclosed_regions(wall)
    return {"lines": lines, "wall": wall, "skipped": skipped, "regions": regions}


def _majority_color(counts: Counter[int]) -> int:
    ranked = counts.most_common()
    if len(ranked) < 2:
        raise AssertionError("each enclosed region must touch at least two wall colors")
    if ranked[0][1] == ranked[1][1]:
        raise AssertionError("boundary wall colors must have a unique majority")
    return int(ranked[0][0])


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_grid(grid)
    except AssertionError:
        return False

    lines = analysis["lines"]
    marker_cells = {tuple(map(int, p)) for p in np.argwhere(grid != BACKGROUND)}
    line_cells = [_full_line_cells(grid.shape, line) for line in lines]

    intersections = 0
    for i in range(len(line_cells)):
        a = set(line_cells[i])
        for j in range(i + 1, len(line_cells)):
            inter = a & set(line_cells[j])
            if len(inter) > 1:
                return False
            if inter:
                intersections += 1
                if next(iter(inter)) not in marker_cells:
                    return False

    if intersections < 2:
        return False
    if int(analysis["skipped"]) < 1:
        return False

    regions = analysis["regions"]
    if len(regions) < 2:
        return False

    fill_colors = set()
    for region in regions:
        counts = region["boundary_counts"]
        if len(counts) < 2:
            return False
        ranked = counts.most_common()
        if ranked[0][1] == ranked[1][1]:
            return False
        fill_colors.add(int(ranked[0][0]))

    if len(fill_colors) < 2:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each three-marker line into a full wall network, then fill every enclosed background region with its boundary-majority wall color.

    Input structure:
    - A black background contains 2 to 4 color groups; each group has exactly 3 same-colored marker pixels.
    - Within each color group, the 3 markers are equally spaced on one horizontal, vertical, or 45-degree diagonal line.
    - Different groups use different colors, and the completed line walls can intersect to form closed and open background regions.

    Rule:
    - Treat each marker triple as a one-cell-wide wall direction and extend that wall through the markers and outward in both directions until the grid boundary.
    - If an extending wall reaches an already colored cell from another wall, leave that cell's existing color unchanged and keep extending past it.
    - After all walls are present, mark all background cells that can reach the border by 4-neighbor moves as exterior.
    - For every remaining 4-connected enclosed background region, count the colors of its 4-neighbor wall pixels and fill the region with the unique most frequent boundary color.

    Key insight:
    - The output color of a hole is not attached to any single marker group beforehand; it emerges from the geometry of the completed wall network and the color frequencies on that region's boundary.
    """
    analysis = _analyze_grid(np.asarray(grid, dtype=int))
    out = np.array(analysis["wall"], copy=True)

    for region in analysis["regions"]:
        fill = _majority_color(region["boundary_counts"])
        for r, c in region["cells"]:
            out[r, c] = fill
    return out


def _render_family_a(h: int, w: int) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    groups = [
        [(2, w - 7), (2, w - 6), (2, w - 5)],
        [(3, w - 3), (3, w - 2), (3, w - 1)],
        [(0, w - 5), (1, w - 4), (2, w - 3)],
        [(3, w - 4), (4, w - 3), (5, w - 2)],
    ]
    colors = random.sample(COLOR_POOL, len(groups))
    for color, cells in zip(colors, groups):
        for r, c in cells:
            grid[r, c] = color
    return grid


def _render_family_b(h: int, w: int) -> np.ndarray:
    top = random.randint(0, h - 8)
    left = random.randint(0, w - 8)
    grid = np.zeros((h, w), dtype=int)
    groups = [
        [(top + 2, left + 0), (top + 2, left + 1), (top + 2, left + 2)],
        [(top + 4, left + 0), (top + 4, left + 1), (top + 4, left + 2)],
        [(top + 5, left + 2), (top + 6, left + 2), (top + 7, left + 2)],
        [(top + 2, left + 7), (top + 3, left + 6), (top + 4, left + 5)],
    ]
    colors = random.sample(COLOR_POOL, len(groups))
    for color, cells in zip(colors, groups):
        for r, c in cells:
            grid[r, c] = color
    return grid


def _apply_transform(grid: np.ndarray, mode: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    if mode >= 4:
        out = np.fliplr(out)
    return np.rot90(out, mode % 4)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        family = random.choice(("a", "b"))
        if family == "a":
            h = random.randint(MIN_SIDE_A, MAX_SIDE)
            w = random.randint(MIN_SIDE_A, MAX_SIDE)
            grid = _render_family_a(h, w)
        else:
            h = random.randint(MIN_SIDE_B, MAX_SIDE)
            w = random.randint(MIN_SIDE_B, MAX_SIDE)
            grid = _render_family_b(h, w)

        grid = _apply_transform(grid, random.randrange(8))
        if _validate_instance(grid):
            return grid

    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
