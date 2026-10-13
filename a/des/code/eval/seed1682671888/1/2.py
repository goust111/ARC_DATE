from __future__ import annotations

import random

import numpy as np


YELLOW = 4
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _component_center(coords: np.ndarray) -> tuple[int, int]:
    row_center = _round_half_up(float(coords[:, 0].mean()))
    col_center = _round_half_up(float(coords[:, 1].mean()))
    return row_center, col_center


def _path_cells(start: tuple[int, int], stop: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = start
    r1, c1 = stop
    cells: list[tuple[int, int]] = [(r0, c0)]

    c_step = 0 if c0 == c1 else (1 if c1 > c0 else -1)
    c = c0
    while c != c1:
        c += c_step
        cells.append((r0, c))

    r_step = 0 if r0 == r1 else (1 if r1 > r0 else -1)
    r = r0
    while r != r1:
        r += r_step
        cells.append((r, c1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored fragment pair with a horizontal-then-vertical path,
    marking the bend in yellow when the connection is not straight.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    out = g.copy()

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors

    for color in colors:
        components = _find_components(g == color)
        assert len(components) == 2, "each non-background color must form exactly two fragments"

        centers = [(_component_center(component), component) for component in components]
        centers.sort(key=lambda item: item[0])
        start, stop = centers[0][0], centers[1][0]
        path = _path_cells(start, stop)

        for r, c in path:
            if out[r, c] == bg:
                out[r, c] = color

        if start[0] != stop[0] and start[1] != stop[1]:
            corner = (start[0], stop[1])
            assert out[corner] in (bg, color), "corner must stay available for the yellow marker"
            out[corner] = YELLOW

    return out


GRID_MIN = 16
GRID_MAX = 24
PAIR_COUNT_MIN = 2
PAIR_COUNT_MAX = 4
PAIR_BRANCHES = ("horizontal", "vertical", "lshape")
PAIR_COLORS = [1, 2, 3, 5, 6, 7, 8, 9]
BG_COLOR = 0
GEN_RETRIES = 400
PAIR_RETRIES = 120
SHAPE_RETRIES = 40


def _cells_from_offsets(center: tuple[int, int], offsets: set[tuple[int, int]]) -> set[tuple[int, int]]:
    cr, cc = center
    return {(cr + dr, cc + dc) for dr, dc in offsets}


def _all_in_bounds(cells: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _touches_existing(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for dr, dc in DIR8 + [(0, 0)]:
            if (r + dr, c + dc) in occupied:
                return True
    return False


def _random_fragment_offsets(size: int) -> set[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < size:
        base_r, base_c = random.choice(tuple(cells))
        neighbors = [(base_r + dr, base_c + dc) for dr, dc in DIR4]
        random.shuffle(neighbors)
        placed = False
        for nr, nc in neighbors:
            if (nr, nc) in cells:
                continue
            if abs(nr) > 2 or abs(nc) > 2:
                continue
            cells.add((nr, nc))
            placed = True
            break
        if not placed:
            break

    if len(cells) != size:
        return _random_fragment_offsets(size)

    mean_r = sum(r for r, _ in cells) / len(cells)
    mean_c = sum(c for _, c in cells) / len(cells)
    shift_r = _round_half_up(mean_r)
    shift_c = _round_half_up(mean_c)
    return {(r - shift_r, c - shift_c) for r, c in cells}


def _branch_sequence(pair_count: int) -> list[str]:
    branches = list(PAIR_BRANCHES)
    random.shuffle(branches)
    if pair_count == 2:
        return branches[:2]
    if pair_count == 3:
        return branches
    return branches + [random.choice(PAIR_BRANCHES)]


def _choose_centers(shape: tuple[int, int], branch: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    h, w = shape
    if branch == "horizontal":
        row = random.randint(2, h - 3)
        c0 = random.randint(1, w - 8)
        c1 = random.randint(c0 + 4, min(w - 2, c0 + 10))
        return (row, c0), (row, c1)

    if branch == "vertical":
        col = random.randint(2, w - 3)
        r0 = random.randint(1, h - 8)
        r1 = random.randint(r0 + 4, min(h - 2, r0 + 10))
        return (r0, col), (r1, col)

    r0 = random.randint(1, h - 8)
    r1 = random.randint(r0 + 4, min(h - 2, r0 + 10))
    c0 = random.randint(1, w - 8)
    options = [c for c in range(1, w - 1) if abs(c - c0) >= 4]
    if not options:
        return None
    c1 = random.choice(options)
    return (r0, c0), (r1, c1)


def _choose_fragment_cells(
    center: tuple[int, int],
    path_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> set[tuple[int, int]] | None:
    for _ in range(SHAPE_RETRIES):
        size = random.randint(2, 5)
        offsets = _random_fragment_offsets(size)
        cells = _cells_from_offsets(center, offsets)
        if not _all_in_bounds(cells, shape):
            continue
        if _touches_existing(cells, occupied):
            continue
        if any(cell in path_cells and cell != center for cell in cells):
            continue
        return cells
    return None


def _build_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    branches = _branch_sequence(pair_count)
    colors = random.sample(PAIR_COLORS, pair_count)

    occupied_fragments: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    lshape_count = 0

    for color, branch in zip(colors, branches):
        placed = False
        for _ in range(PAIR_RETRIES):
            centers = _choose_centers((h, w), branch)
            if centers is None:
                continue
            start, stop = centers
            path = _path_cells(start, stop)
            path_set = set(path)

            if any(cell in reserved_paths or cell in occupied_fragments for cell in path_set):
                continue

            first_cells = _choose_fragment_cells(start, path_set, occupied_fragments, (h, w))
            if first_cells is None:
                continue

            second_cells = _choose_fragment_cells(stop, path_set, occupied_fragments | first_cells, (h, w))
            if second_cells is None:
                continue

            if first_cells & second_cells:
                continue
            if _touches_existing(first_cells, second_cells) or _touches_existing(second_cells, first_cells):
                continue

            for r, c in first_cells | second_cells:
                grid[r, c] = color
            occupied_fragments |= first_cells | second_cells
            reserved_paths |= path_set
            if branch == "lshape":
                lshape_count += 1
            placed = True
            break

        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if int(np.sum(out == YELLOW)) != lshape_count:
        return None

    for color in colors:
        if len(_find_components(grid == color)) != 2:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
