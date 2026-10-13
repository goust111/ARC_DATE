from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
LEGEND_COLORS = (BLUE, RED, GREEN, YELLOW)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for row in range(height):
        for col in range(width):
            if not mask[row, col] or seen[row, col]:
                continue
            seen[row, col] = True
            queue = deque([(row, col)])
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue
                    if seen[nr, nc] or not mask[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _hole_components(mask: np.ndarray) -> list[np.ndarray]:
    empty = ~np.asarray(mask, dtype=bool)
    height, width = empty.shape
    seen = np.zeros_like(empty, dtype=bool)
    holes: list[np.ndarray] = []
    for row in range(height):
        for col in range(width):
            if not empty[row, col] or seen[row, col]:
                continue
            seen[row, col] = True
            queue = deque([(row, col)])
            coords: list[tuple[int, int]] = []
            touches_edge = False
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == height - 1 or cc == 0 or cc == width - 1:
                    touches_edge = True
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if nr < 0 or nr >= height or nc < 0 or nc >= width:
                        continue
                    if seen[nr, nc] or not empty[nr, nc]:
                        continue
                    seen[nr, nc] = True
                    queue.append((nr, nc))
            if not touches_edge:
                holes.append(np.asarray(coords, dtype=int))
    return holes


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    filled = np.asarray(mask, dtype=bool).copy()
    for hole in _hole_components(filled):
        filled[hole[:, 0], hole[:, 1]] = True
    return filled


def _paint(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    block = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    block[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape

    legend = [(row, int(g[row, 0])) for row in range(height) if int(g[row, 0]) != BLACK]

    out = np.full((height, width), BLACK, dtype=int)
    out[:, 0] = g[:, 0]

    for coords in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(coords)
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True

        holes = _hole_components(local)
        hole_count = len(holes)
        target_row, color = legend[hole_count - 1]
        filled = local.copy()
        for hole in holes:
            filled[hole[:, 0], hole[:, 1]] = True

        dest_top = int(target_row)
        _paint(out, dest_top, c0, filled, color)

    return out


def _sample_holes(base_h: int, base_w: int, hole_count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (row, col)
        for row in range(1, base_h - 1)
        for col in range(1, base_w - 1)
    ]
    random.shuffle(candidates)
    holes: list[tuple[int, int]] = []
    for row, col in candidates:
        if all(abs(row - rr) + abs(col - cc) > 1 for rr, cc in holes):
            holes.append((row, col))
            if len(holes) == hole_count:
                return holes
    return None


def _make_object_mask(hole_count: int, max_width: int) -> np.ndarray | None:
    for _ in range(120):
        base_h = random.randint(5, 7)
        base_w = random.randint(5, max_width)
        holes = _sample_holes(base_h, base_w, hole_count)
        if holes is None:
            continue

        occupied = {
            (row, col)
            for row in range(base_h)
            for col in range(base_w)
            if (row, col) not in holes
        }

        for side in random.sample(("top", "bottom"), random.randint(0, 2)):
            span = random.randint(1, base_w - 1)
            start = random.randint(0, base_w - span)
            edge_row = -1 if side == "top" else base_h
            for col in range(start, start + span):
                occupied.add((edge_row, col))

        min_row = min(row for row, _col in occupied)
        max_row = max(row for row, _col in occupied)
        min_col = min(col for _row, col in occupied)
        max_col = max(col for _row, col in occupied)
        mask = np.zeros((max_row - min_row + 1, max_col - min_col + 1), dtype=bool)
        for row, col in occupied:
            mask[row - min_row, col - min_col] = True

        if len(_components(mask)) != 1:
            continue
        holes_found = _hole_components(mask)
        if len(holes_found) != hole_count:
            continue
        if any(component.shape[0] != 1 for component in holes_found):
            continue
        return mask
    return None


def _sample_legend_rows(height: int, shape_heights: list[int]) -> list[int] | None:
    if not shape_heights:
        return []

    latest = [0] * len(shape_heights)
    latest[-1] = height - shape_heights[-1]
    for idx in range(len(shape_heights) - 2, -1, -1):
        latest[idx] = min(height - shape_heights[idx], latest[idx + 1] - 2)

    if latest[0] < 1:
        return None

    rows: list[int] = []
    low = 1
    for high in latest:
        if low > high:
            return None
        row = random.randint(low, high)
        rows.append(row)
        low = row + 2
    return rows


def _sample_initial_row(height: int, shape_h: int, final_row: int) -> int | None:
    options = [row for row in range(0, height - shape_h + 1) if row != final_row]
    if not options:
        return None

    above = [row for row in options if row < final_row]
    below = [row for row in options if row > final_row]
    if above and below:
        pool = above if random.random() < 0.5 else below
    else:
        pool = above or below
    return int(random.choice(pool))


def _build_expected(
    shape_by_holes: dict[int, np.ndarray],
    left_by_holes: dict[int, int],
    legend_rows: list[int],
    legend_colors: list[int],
    height: int,
    width: int,
) -> np.ndarray:
    out = np.full((height, width), BLACK, dtype=int)
    for row, color in zip(legend_rows, legend_colors, strict=True):
        out[row, 0] = int(color)

    for hole_count, mask in shape_by_holes.items():
        filled = _fill_holes(mask)
        _paint(
            out,
            legend_rows[hole_count - 1],
            left_by_holes[hole_count],
            filled,
            legend_colors[hole_count - 1],
        )
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    object_count = random.choice((3, 4))
    hole_counts = list(range(1, object_count + 1))
    max_width = 5 if object_count == 4 else 6

    shape_by_holes: dict[int, np.ndarray] = {}
    for hole_count in hole_counts:
        mask = _make_object_mask(hole_count, max_width=max_width)
        if mask is None:
            return None
        shape_by_holes[hole_count] = mask

    max_shape_h = max(mask.shape[0] for mask in shape_by_holes.values())
    min_height = max(18, max_shape_h + 2 * object_count + 2)
    if min_height > 28:
        return None
    height = random.randint(min_height, 28)

    shape_heights = [shape_by_holes[hole_count].shape[0] for hole_count in hole_counts]
    legend_rows = _sample_legend_rows(height, shape_heights)
    if legend_rows is None:
        return None

    legend_colors = random.sample(list(LEGEND_COLORS), object_count)

    order = random.sample(hole_counts, len(hole_counts))
    left_margin = random.randint(1, 2)
    gap_choices = [1] * (object_count - 1) if object_count == 4 else [
        random.randint(1, 2) for _ in range(object_count - 1)
    ]
    right_margin = random.randint(1, 2)

    total_width = 1 + left_margin + right_margin
    total_width += sum(shape_by_holes[hole_count].shape[1] for hole_count in order)
    total_width += sum(gap_choices)
    extra_width = random.randint(0, 2 if object_count == 3 else 1)
    width = total_width + extra_width
    if width > 30:
        return None

    left_by_holes: dict[int, int] = {}
    col = 1 + left_margin
    for idx, hole_count in enumerate(order):
        left_by_holes[hole_count] = col
        col += shape_by_holes[hole_count].shape[1]
        if idx < len(order) - 1:
            col += gap_choices[idx]

    expected = _build_expected(
        shape_by_holes=shape_by_holes,
        left_by_holes=left_by_holes,
        legend_rows=legend_rows,
        legend_colors=legend_colors,
        height=height,
        width=width,
    )

    grid = np.full((height, width), BLACK, dtype=int)
    for row, color in zip(legend_rows, legend_colors, strict=True):
        grid[row, 0] = int(color)

    for hole_count in hole_counts:
        mask = shape_by_holes[hole_count]
        final_row = legend_rows[hole_count - 1]
        initial_row = _sample_initial_row(height, mask.shape[0], final_row)
        if initial_row is None:
            return None
        _paint(grid, initial_row, left_by_holes[hole_count], mask, GREY)

    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            grid, _expected = result
            return grid
    raise RuntimeError("failed to generate a valid input")
