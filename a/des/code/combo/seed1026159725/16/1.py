import random
from collections import Counter, deque

import numpy as np


GEN_RETRIES = 300
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 120


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_nonbg4(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rearrange each multi-color object into a compact rectangle at its bbox origin.

    For every 4-connected non-background object:
    - let C be the number of distinct colors and T its area;
    - build a C x (T/C) rectangle at the object's bbox top-left;
    - fill one row per color, ordered by descending count;
    - break count ties by first appearance in the object's bbox for determinism.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    out = g.copy()

    for comp in _components_nonbg4(g, bg):
        r0, c0, r1, c1 = _bbox(comp)
        counts = Counter(int(g[r, c]) for r, c in comp)
        colors = list(counts)
        n_colors = len(colors)
        total = len(comp)
        assert 2 <= n_colors <= 4, "expected multi-color objects with 2 to 4 colors"
        assert total % n_colors == 0, "object area must be divisible by the number of colors"
        row_len = total // n_colors

        first_pos: dict[int, tuple[int, int]] = {}
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                color = int(g[r, c])
                if color == bg or color in first_pos:
                    continue
                first_pos[color] = (r - r0, c - c0)

        order = sorted(colors, key=lambda color: (-counts[color], first_pos[color]))

        for r, c in comp:
            out[r, c] = bg
        for row_idx, color in enumerate(order):
            out[r0 + row_idx, c0 : c0 + row_len] = color

    return out


def _normalize(points: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return sorted((r - min_r, c - min_c) for r, c in points)


def _grow_connected_shape(rng: random.Random, n_cells: int, min_h: int, min_w: int) -> list[tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        box_h = rng.randint(min_h, min_h + 2)
        box_w = rng.randint(min_w, min_w + 2)
        if box_h * box_w <= n_cells:
            if rng.random() < 0.5:
                box_h += 1
            else:
                box_w += 1
        if box_h * box_w <= n_cells:
            continue

        start = (rng.randrange(box_h), rng.randrange(box_w))
        points = {start}
        frontier = set()
        for cell in _neighbors4(*start):
            if 0 <= cell[0] < box_h and 0 <= cell[1] < box_w:
                frontier.add(cell)

        while len(points) < n_cells and frontier:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            points.add(cell)
            nbrs = [nbr for nbr in _neighbors4(*cell) if 0 <= nbr[0] < box_h and 0 <= nbr[1] < box_w and nbr not in points]
            rng.shuffle(nbrs)
            frontier.update(nbrs)

        if len(points) != n_cells:
            continue

        norm = _normalize(points)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if h < min_h or w < min_w:
            continue
        if h * w == n_cells:
            continue
        return norm
    return None


def _colorize_shape(
    rng: random.Random, shape: list[tuple[int, int]], colors: list[int], repeat: int
) -> tuple[list[tuple[int, int, int]], tuple[int, ...]]:
    cells = shape[:]
    rng.shuffle(cells)
    cells.sort(key=lambda cell: (cell[0], cell[1], rng.random()))
    color_order = colors[:]
    rng.shuffle(color_order)

    colored: list[tuple[int, int, int]] = []
    idx = 0
    for color in color_order:
        for _ in range(repeat):
            r, c = cells[idx]
            colored.append((r, c, color))
            idx += 1
    return colored, tuple(color_order)


def _can_place(used: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    h, w = used.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + height + 1)
    c1 = min(w, left + width + 1)
    return not bool(used[r0:r1, c0:c1].any())


def _paint_object(grid: np.ndarray, used: np.ndarray, top: int, left: int, cells: list[tuple[int, int, int]]) -> None:
    max_r = 0
    max_c = 0
    for r, c, color in cells:
        grid[top + r, left + c] = color
        max_r = max(max_r, r)
        max_c = max(max_c, c)
    used[top : top + max_r + 1, left : left + max_c + 1] = True


def _plan_color_groups(rng: random.Random) -> list[int]:
    for _ in range(100):
        n_objects = rng.choice([2, 3])
        groups = [rng.randint(2, 4) for _ in range(n_objects)]
        if sum(groups) <= 9 and 4 in groups and 2 in groups:
            return groups
        if sum(groups) <= 9 and rng.random() < 0.35:
            return groups
    raise RuntimeError("failed to sample color groups")


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(14, 24)
    width = rng.randint(14, 24)
    bg = rng.randrange(10)
    grid = np.full((height, width), bg, dtype=int)
    used = np.zeros((height, width), dtype=bool)

    color_pool = [color for color in range(10) if color != bg]
    rng.shuffle(color_pool)
    group_sizes = _plan_color_groups(rng)

    cursor = 0
    objects: list[tuple[list[tuple[int, int, int]], int, int]] = []
    for n_colors in group_sizes:
        colors = color_pool[cursor : cursor + n_colors]
        cursor += n_colors
        row_len = rng.randint(2, 6)
        area = n_colors * row_len
        shape = _grow_connected_shape(rng, area, n_colors, row_len)
        if shape is None:
            return None
        colored, _ = _colorize_shape(rng, shape, colors, row_len)
        obj_h = max(r for r, _, _ in colored) + 1
        obj_w = max(c for _, c, _ in colored) + 1
        objects.append((colored, obj_h, obj_w))

    rng.shuffle(objects)
    for cells, obj_h, obj_w in objects:
        placed = False
        row_positions = list(range(0, height - obj_h + 1))
        col_positions = list(range(0, width - obj_w + 1))
        rng.shuffle(row_positions)
        rng.shuffle(col_positions)
        for top in row_positions:
            for left in col_positions:
                if _can_place(used, top, left, obj_h, obj_w):
                    _paint_object(grid, used, top, left, cells)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    if _background_color(grid) != bg:
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
