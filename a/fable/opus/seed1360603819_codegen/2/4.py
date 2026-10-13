from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREEN = 3
GRAY = 5
AZURE = 8

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    areas = [0]
    next_id = 0

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            next_id += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = next_id
            area = 0

            while q:
                rr, cc = q.popleft()
                area += 1
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = next_id
                        q.append((nr, nc))

            areas.append(area)

    return labels, areas


def _components(mask: np.ndarray) -> list[np.ndarray]:
    labels, areas = _label_components(mask)
    return [np.argwhere(labels == comp_id) for comp_id in range(1, len(areas))]


def _touching_area(
    coords: np.ndarray,
    bg_labels: np.ndarray,
    bg_areas: list[int],
    dr: int,
    dc: int,
) -> int:
    h, w = bg_labels.shape
    touched: set[int] = set()
    for rr, cc in coords:
        nr = int(rr) + dr
        nc = int(cc) + dc
        if 0 <= nr < h and 0 <= nc < w:
            comp_id = int(bg_labels[nr, nc])
            if comp_id != 0:
                touched.add(comp_id)
    return sum(bg_areas[comp_id] for comp_id in touched)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare the background component areas on the two sides of each grey wall segment,
    then stamp a one-cell shadow on the heavier side.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    bg = _dominant_color(g)
    out = g.copy()

    unique = set(map(int, np.unique(g)))
    if len(unique - {bg}) == 0:
        return out
    wall_color = GRAY if GRAY in unique else next(iter(unique - {bg}))

    bg_labels, bg_areas = _label_components(g == bg)
    walls = _components(g == wall_color)

    for coords in walls:
        rows = coords[:, 0]
        cols = coords[:, 1]
        horizontal = np.all(rows == rows[0])
        vertical = np.all(cols == cols[0])

        if horizontal and not vertical:
            wall_row = int(rows[0])
            up_area = _touching_area(coords, bg_labels, bg_areas, -1, 0)
            down_area = _touching_area(coords, bg_labels, bg_areas, 1, 0)
            if up_area > down_area and wall_row - 1 >= 0:
                out[wall_row - 1, cols] = GREEN
            elif down_area > up_area and wall_row + 1 < g.shape[0]:
                out[wall_row + 1, cols] = GREEN
        elif vertical and not horizontal:
            wall_col = int(cols[0])
            left_area = _touching_area(coords, bg_labels, bg_areas, 0, -1)
            right_area = _touching_area(coords, bg_labels, bg_areas, 0, 1)
            if left_area > right_area and wall_col - 1 >= 0:
                out[rows, wall_col - 1] = AZURE
            elif right_area > left_area and wall_col + 1 < g.shape[1]:
                out[rows, wall_col + 1] = AZURE

    return out


def _sample_relations(n_walls: int) -> list[int]:
    while True:
        relations = [random.choice((-1, 0, 1)) for _ in range(n_walls)]
        if any(rel != 0 for rel in relations):
            return relations


def _stripe_sizes(relations: list[int]) -> list[int]:
    levels = [0]
    for rel in relations:
        levels.append(levels[-1] + rel)
    min_level = min(levels)
    base = random.randint(2, 3)
    return [base + level - min_level for level in levels]


def _wall_positions(stripes: list[int]) -> list[int]:
    pos = stripes[0]
    walls: list[int] = []
    for next_stripe in stripes[1:]:
        walls.append(pos)
        pos += 1 + next_stripe
    return walls


def _render_case(mode: str, stripes: list[int], span: int) -> tuple[np.ndarray, np.ndarray]:
    walls = _wall_positions(stripes)

    if mode == "horizontal":
        h = sum(stripes) + len(walls)
        w = span
        inp = np.full((h, w), BLACK, dtype=int)
        for row in walls:
            inp[row, :] = GRAY

        out = inp.copy()
        for i, row in enumerate(walls):
            upper = stripes[i]
            lower = stripes[i + 1]
            if upper > lower:
                out[row - 1, :] = GREEN
            elif lower > upper:
                out[row + 1, :] = GREEN
        return inp, out

    h = span
    w = sum(stripes) + len(walls)
    inp = np.full((h, w), BLACK, dtype=int)
    for col in walls:
        inp[:, col] = GRAY

    out = inp.copy()
    for i, col in enumerate(walls):
        left = stripes[i]
        right = stripes[i + 1]
        if left > right:
            out[:, col - 1] = AZURE
        elif right > left:
            out[:, col + 1] = AZURE
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(("horizontal", "vertical"))
    n_walls = random.randint(2, 4)
    relations = _sample_relations(n_walls)
    stripes = _stripe_sizes(relations)
    span = random.randint(8, 22)

    inp, expected = _render_case(mode, stripes, span)
    if int(np.count_nonzero(inp == BLACK)) <= int(np.count_nonzero(inp == GRAY)):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
