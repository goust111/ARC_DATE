import random
from collections import Counter, defaultdict

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            objects.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(len(coords)),
                }
            )
    return objects


def _is_solid_rectangle(grid: np.ndarray, obj: dict) -> bool:
    r0, c0, r1, c1 = obj["bbox"]
    color = int(obj["color"])
    return obj["area"] == (r1 - r0 + 1) * (c1 - c0 + 1) and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color)


def _corners(obj: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = obj["bbox"]
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _best_corner_pair(a: dict, b: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    best = None
    best_pair = None
    for p1 in _corners(a):
        for p2 in _corners(b):
            dr = p2[0] - p1[0]
            dc = p2[1] - p1[1]
            if dr == 0 or dc == 0:
                continue
            score = (
                abs(abs(dr) - abs(dc)),
                max(abs(dr), abs(dc)),
                abs(dr) + abs(dc),
                p1,
                p2,
            )
            if best is None or score < best:
                best = score
                best_pair = (p1, p2)
    assert best_pair is not None, "expected a diagonal corner pair"
    return best_pair


def _draw_diagonal(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], color: int, bg: int) -> None:
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    step_r = 1 if dr > 0 else -1
    step_c = 1 if dc > 0 else -1
    steps = min(abs(dr), abs(dc))
    for k in range(steps + 1):
        r = start[0] + k * step_r
        c = start[1] + k * step_c
        if int(out[r, c]) == bg:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    objects = _find_components(g, bg)
    assert objects, "expected foreground rectangles"

    by_color: dict[int, list[dict]] = defaultdict(list)
    for obj in objects:
        assert _is_solid_rectangle(g, obj), "all foreground objects must be solid rectangles"
        by_color[int(obj["color"])].append(obj)

    out = g.copy()
    for color, comps in by_color.items():
        assert len(comps) == 2, "each foreground color must appear in exactly two rectangles"
        p1, p2 = _best_corner_pair(comps[0], comps[1])
        _draw_diagonal(out, p1, p2, int(color), bg)

    return out


COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIZE_RANGE = (14, 24)
PAIR_COUNT_CHOICES = [2, 3, 4]
RECT_SIDE_CHOICES = [1, 2, 3, 4]
DIAG_LEN_RANGE = (2, 8)
GEN_RETRY_LIMIT = 500
PAIR_PLACE_RETRIES = 200


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _inflate(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _bbox_from_corner(corner: tuple[int, int], row_sign: int, col_sign: int, rect_h: int, rect_w: int, first: bool) -> tuple[int, int, int, int]:
    r, c = corner
    if first:
        r0 = r - rect_h + 1 if row_sign == 1 else r
        c0 = c - rect_w + 1 if col_sign == 1 else c
    else:
        r0 = r if row_sign == 1 else r - rect_h + 1
        c0 = c if col_sign == 1 else c - rect_w + 1
    return r0, c0, rect_h, rect_w


def _pair_spec_to_cells(
    shape: tuple[int, int],
    start: tuple[int, int],
    row_sign: int,
    col_sign: int,
    dist: int,
    h1: int,
    w1: int,
    h2: int,
    w2: int,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    end = (start[0] + row_sign * dist, start[1] + col_sign * dist)
    r0a, c0a, _, _ = _bbox_from_corner(start, row_sign, col_sign, h1, w1, first=True)
    r0b, c0b, _, _ = _bbox_from_corner(end, row_sign, col_sign, h2, w2, first=False)
    h, w = shape
    if not (0 <= r0a and 0 <= c0a and r0a + h1 <= h and c0a + w1 <= w):
        return None
    if not (0 <= r0b and 0 <= c0b and r0b + h2 <= h and c0b + w2 <= w):
        return None

    rect1 = _rect_cells(r0a, c0a, h1, w1)
    rect2 = _rect_cells(r0b, c0b, h2, w2)
    if rect1 & rect2:
        return None

    line = {(start[0] + k * row_sign, start[1] + k * col_sign) for k in range(1, dist)}
    if rect1 & line or rect2 & line:
        return None
    return rect1, rect2, line


def _paint_rect(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _valid_input(grid: np.ndarray) -> bool:
    try:
        bg = _get_bg(grid)
        objects = _find_components(grid, bg)
        if not objects:
            return False
        color_counts = Counter(int(obj["color"]) for obj in objects)
        if any(v != 2 for v in color_counts.values()):
            return False
        if any(not _is_solid_rectangle(grid, obj) for obj in objects):
            return False
        for color in color_counts:
            comps = [obj for obj in objects if int(obj["color"]) == color]
            p1, p2 = _best_corner_pair(comps[0], comps[1])
            if abs(p1[0] - p2[0]) != abs(p1[1] - p2[1]):
                return False
        out = main(grid)
        return np.count_nonzero(out != grid) > 0
    except AssertionError:
        return False


def _generate_pair(
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    h, w = shape
    for _ in range(PAIR_PLACE_RETRIES):
        row_sign = random.choice([-1, 1])
        col_sign = random.choice([-1, 1])
        rect_h1 = random.choice(RECT_SIDE_CHOICES)
        rect_w1 = random.choice(RECT_SIDE_CHOICES)
        rect_h2 = random.choice(RECT_SIDE_CHOICES)
        rect_w2 = random.choice(RECT_SIDE_CHOICES)
        if rect_h1 == rect_w1 == rect_h2 == rect_w2 == 1:
            continue
        dist = random.randint(*DIAG_LEN_RANGE)
        start = (random.randrange(h), random.randrange(w))
        cells = _pair_spec_to_cells(shape, start, row_sign, col_sign, dist, rect_h1, rect_w1, rect_h2, rect_w2)
        if cells is None:
            continue
        rect1, rect2, line = cells
        reserve = _inflate(rect1 | rect2 | line, shape)
        if reserve & blocked:
            continue
        return rect1, rect2, reserve
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        h = random.randint(*SIZE_RANGE)
        w = random.randint(*SIZE_RANGE)
        grid = np.zeros((h, w), dtype=int)
        pair_count = random.choice(PAIR_COUNT_CHOICES)
        colors = random.sample(COLOR_POOL, pair_count)

        blocked: set[tuple[int, int]] = set()
        ok = True
        for color in colors:
            pair = _generate_pair((h, w), blocked)
            if pair is None:
                ok = False
                break
            rect1, rect2, reserve = pair
            _paint_rect(grid, rect1, color)
            _paint_rect(grid, rect2, color)
            blocked |= reserve

        if ok and _valid_input(grid):
            return grid

    raise RuntimeError("failed to generate a valid ARC instance")
