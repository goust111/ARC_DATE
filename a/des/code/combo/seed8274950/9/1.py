from __future__ import annotations

import random

import numpy as np


RED = 2
BLUE = 1
GREEN = 3

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400
PAIR_RETRIES = 120
GRID_MIN = 18
GRID_MAX = 26
PAIR_COUNT_MIN = 3
PAIR_COUNT_MAX = 5
BG_CHOICES = [0, 9]
INPUT_COLOR_POOL = [4, 5, 6, 7, 8, 9]
SIZE_CHOICES = [1, 2, 3]


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _round_index(value: float) -> int:
    return int(np.floor(value + 0.5))


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def _shared_axis_value(
    a0: int, a1: int, b0: int, b1: int, pref_a: float, pref_b: float
) -> int:
    lo = max(a0, b0)
    hi = min(a1, b1)
    preferred = _round_index((pref_a + pref_b) / 2.0)
    if lo <= hi:
        return _clamp(preferred, lo, hi)
    return preferred


def _segment_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segment must be horizontal or vertical")


def _dedupe_cells(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            ordered.append(cell)
    return ordered


def _classify_relation(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> str:
    ra, ca = _center(box_a)
    rb, cb = _center(box_b)
    dr = abs(ra - rb)
    dc = abs(ca - cb)

    if dr <= 0.5 and dc > 0.5:
        return "horizontal"
    if dc <= 0.5 and dr > 0.5:
        return "vertical"
    return "diagonal"


def _horizontal_path(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    ca = _center(box_a)[1]
    cb = _center(box_b)[1]
    left, right = (box_a, box_b) if ca <= cb else (box_b, box_a)
    la, _ = _center(left)
    ra, _ = _center(right)
    row = _shared_axis_value(left[0], left[2], right[0], right[2], la, ra)
    start_col = left[3] + 1
    end_col = right[1] - 1
    return _segment_cells(row, start_col, row, end_col)


def _vertical_path(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    ra = _center(box_a)[0]
    rb = _center(box_b)[0]
    top, bottom = (box_a, box_b) if ra <= rb else (box_b, box_a)
    _, ct = _center(top)
    _, cb = _center(bottom)
    col = _shared_axis_value(top[1], top[3], bottom[1], bottom[3], ct, cb)
    start_row = top[2] + 1
    end_row = bottom[0] - 1
    return _segment_cells(start_row, col, end_row, col)


def _diagonal_path(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    _, ca = _center(box_a)
    _, cb = _center(box_b)
    left, right = (box_a, box_b) if ca <= cb else (box_b, box_a)
    row_left = _clamp(_round_index(_center(left)[0]), left[0], left[2])
    row_right = _clamp(_round_index(_center(right)[0]), right[0], right[2])
    start_col = left[3] + 1
    elbow_col = right[1] - 1
    cells = _segment_cells(row_left, start_col, row_left, elbow_col)
    cells.extend(_segment_cells(row_left, elbow_col, row_right, elbow_col))
    return _dedupe_cells(cells)


def _path_cells(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], relation: str
) -> tuple[list[tuple[int, int]], int]:
    if relation == "horizontal":
        return _horizontal_path(box_a, box_b), RED
    if relation == "vertical":
        return _vertical_path(box_a, box_b), BLUE
    return _diagonal_path(box_a, box_b), GREEN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored object pair and color the connector by geometry.

    Horizontal pairs get a red straight line, vertical pairs get a blue
    straight line, and diagonal pairs get a green L-shaped path that goes
    horizontal first and then vertical.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3, "expected a 2D grid"
    bg = _background_color(g)

    pairs = []
    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        comps = _components_of_color(g, color)
        assert len(comps) == 2, "each foreground color must appear as exactly two objects"
        boxes = [_bbox(comp) for comp in comps]
        key = (
            min(boxes[0][0], boxes[1][0]),
            min(boxes[0][1], boxes[1][1]),
            max(boxes[0][2], boxes[1][2]),
            max(boxes[0][3], boxes[1][3]),
        )
        pairs.append((key, boxes[0], boxes[1]))

    out = g.copy()
    for _, box_a, box_b in sorted(pairs, key=lambda item: item[0]):
        relation = _classify_relation(box_a, box_b)
        cells, color = _path_cells(box_a, box_b, relation)
        for r, c in cells:
            if 0 <= r < g.shape[0] and 0 <= c < g.shape[1] and int(g[r, c]) == bg and int(out[r, c]) == bg:
                out[r, c] = color

    return out


def _rect_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _sample_horizontal_boxes(h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for _ in range(PAIR_RETRIES):
        if random.choice([False, True]):
            h1 = random.choice([1, 3])
            h2 = random.choice([1, 3])
            margin = max(h1 // 2, h2 // 2)
            if margin > h - 1 - margin:
                continue
            center_row = random.randint(margin, h - 1 - margin)
            r1 = center_row - h1 // 2
            r2 = center_row - h2 // 2
        else:
            if h < 2:
                continue
            h1 = h2 = 2
            top = random.randint(0, h - 2)
            r1 = r2 = top

        w1 = random.choice(SIZE_CHOICES)
        w2 = random.choice(SIZE_CHOICES)
        gap_max = min(8, w - w1 - w2)
        if gap_max < 2:
            continue
        gap = random.randint(2, gap_max)
        total = w1 + gap + w2
        c1 = random.randint(0, w - total)
        c2 = c1 + w1 + gap
        box_a = (r1, c1, r1 + h1 - 1, c1 + w1 - 1)
        box_b = (r2, c2, r2 + h2 - 1, c2 + w2 - 1)
        if _classify_relation(box_a, box_b) == "horizontal":
            return box_a, box_b
    return None


def _sample_vertical_boxes(h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for _ in range(PAIR_RETRIES):
        if random.choice([False, True]):
            w1 = random.choice([1, 3])
            w2 = random.choice([1, 3])
            margin = max(w1 // 2, w2 // 2)
            if margin > w - 1 - margin:
                continue
            center_col = random.randint(margin, w - 1 - margin)
            c1 = center_col - w1 // 2
            c2 = center_col - w2 // 2
        else:
            if w < 2:
                continue
            w1 = w2 = 2
            left = random.randint(0, w - 2)
            c1 = c2 = left

        h1 = random.choice(SIZE_CHOICES)
        h2 = random.choice(SIZE_CHOICES)
        gap_max = min(8, h - h1 - h2)
        if gap_max < 2:
            continue
        gap = random.randint(2, gap_max)
        total = h1 + gap + h2
        r1 = random.randint(0, h - total)
        r2 = r1 + h1 + gap
        box_a = (r1, c1, r1 + h1 - 1, c1 + w1 - 1)
        box_b = (r2, c2, r2 + h2 - 1, c2 + w2 - 1)
        if _classify_relation(box_a, box_b) == "vertical":
            return box_a, box_b
    return None


def _sample_diagonal_boxes(h: int, w: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    for _ in range(PAIR_RETRIES):
        h1 = random.choice(SIZE_CHOICES)
        w1 = random.choice(SIZE_CHOICES)
        h2 = random.choice(SIZE_CHOICES)
        w2 = random.choice(SIZE_CHOICES)

        gap_max = min(8, w - w1 - w2)
        if gap_max < 3:
            continue
        gap = random.randint(3, gap_max)
        total = w1 + gap + w2
        c1 = random.randint(0, w - total)
        c2 = c1 + w1 + gap

        r1 = random.randint(0, h - h1)
        r2 = random.randint(0, h - h2)
        box_a = (r1, c1, r1 + h1 - 1, c1 + w1 - 1)
        box_b = (r2, c2, r2 + h2 - 1, c2 + w2 - 1)

        center_a = _center(box_a)[0]
        center_b = _center(box_b)[0]
        if abs(center_a - center_b) < 2.0:
            continue
        if _classify_relation(box_a, box_b) == "diagonal":
            return box_a, box_b
    return None


def _sample_boxes_for_relation(
    relation: str, h: int, w: int
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]] | None:
    if relation == "horizontal":
        return _sample_horizontal_boxes(h, w)
    if relation == "vertical":
        return _sample_vertical_boxes(h, w)
    return _sample_diagonal_boxes(h, w)


def _relation_schedule(pair_count: int) -> list[str]:
    relations = ["horizontal", "vertical", "diagonal"]
    random.shuffle(relations)
    schedule = relations[:]
    while len(schedule) < pair_count:
        schedule.append(random.choice(relations))
    random.shuffle(schedule)
    return schedule


def _apply_random_symmetry(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    if random.choice([False, True]):
        out = np.flipud(out)
    if random.choice([False, True]):
        out = np.fliplr(out)
    if random.choice([False, True]):
        out = out.T
    return np.asarray(out, dtype=int).copy()


def _generate_case() -> np.ndarray | None:
    pair_count = random.randint(PAIR_COUNT_MIN, PAIR_COUNT_MAX)
    h = random.randint(max(GRID_MIN, pair_count * 4), GRID_MAX)
    w = random.randint(max(GRID_MIN, pair_count * 4), GRID_MAX)
    if h * w < pair_count * 48:
        return None

    bg = random.choice(BG_CHOICES)
    colors = [color for color in INPUT_COLOR_POOL if color != bg]
    if len(colors) < pair_count:
        return None
    pair_colors = random.sample(colors, pair_count)
    relations = _relation_schedule(pair_count)

    grid = np.full((h, w), bg, dtype=int)
    blocked: set[tuple[int, int]] = set()

    for relation, color in zip(relations, pair_colors, strict=False):
        placed = False
        for _ in range(PAIR_RETRIES):
            boxes = _sample_boxes_for_relation(relation, h, w)
            if boxes is None:
                continue
            box_a, box_b = boxes
            path_cells, _ = _path_cells(box_a, box_b, relation)
            if not path_cells:
                continue

            used = _rect_cells(box_a) | _rect_cells(box_b) | set(path_cells)
            if any(cell in blocked for cell in used):
                continue

            for r, c in _rect_cells(box_a) | _rect_cells(box_b):
                grid[r, c] = color

            blocked |= _expand_cells(used, h, w)
            placed = True
            break

        if not placed:
            return None

    return _apply_random_symmetry(grid)


def _generated_grid_is_valid(grid: np.ndarray) -> bool:
    bg = _background_color(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    if not (PAIR_COUNT_MIN <= len(colors) <= PAIR_COUNT_MAX):
        return False
    for color in colors:
        if len(_components_of_color(grid, color)) != 2:
            return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    return not np.array_equal(out, grid)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None and _generated_grid_is_valid(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
