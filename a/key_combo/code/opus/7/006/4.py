import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    ORANGE = 7


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rectangles(grid: np.ndarray, bg: int | None = None) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    if bg is None:
        bg = _get_bg(g)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    rectangles: list[tuple[int, int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            r0, c0, r1, c1 = _bbox_from_coords(coords)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            if not np.all(patch == color):
                raise ValueError("foreground components must be solid rectangles")
            if len(coords) != patch.size:
                raise ValueError("foreground components must fill their bounding boxes")
            rectangles.append((r0, c0, r1, c1, color))

    rectangles.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
    return rectangles


def _gridline_background(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rows = np.arange(h)[:, None]
    cols = np.arange(w)[None, :]
    return np.where((rows % 2 == 0) | (cols % 2 == 0), Color.BLUE, Color.BLACK).astype(int)


def _paint_rectangle_layers(
    out: np.ndarray,
    rectangle: tuple[int, int, int, int, int],
) -> None:
    r0, c0, r1, c1, border_color = rectangle
    rows = np.arange(r0, r1 + 1)[:, None]
    cols = np.arange(c0, c1 + 1)[None, :]
    depth = np.minimum(
        np.minimum(rows - r0, r1 - rows),
        np.minimum(cols - c0, c1 - cols),
    )

    patch = out[r0 : r1 + 1, c0 : c1 + 1]
    patch[depth == 0] = int(border_color)
    patch[(depth > 0) & (depth % 2 == 1)] = Color.ORANGE
    patch[(depth > 0) & (depth % 2 == 0)] = Color.YELLOW


def _expected_color(
    r: int,
    c: int,
    rectangles: list[tuple[int, int, int, int, int]],
) -> int:
    for r0, c0, r1, c1, border_color in rectangles:
        if r0 <= r <= r1 and c0 <= c <= c1:
            depth = min(r - r0, r1 - r, c - c0, c1 - c)
            if depth == 0:
                return int(border_color)
            if depth % 2 == 1:
                return Color.ORANGE
            return Color.YELLOW
    return Color.BLUE if (r % 2 == 0 or c % 2 == 0) else Color.BLACK


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Draw the global gridline pattern, but let each solid rectangle override it with concentric layers.

    Input structure:
    - The input is an odd-sized square grid with black background and exactly two separated solid axis-aligned rectangles.
    - The rectangles are monochromatic, use red and green, stay away from the border, and their row ranges and column ranges do not overlap.

    Rule:
    - Start from a full-grid background where every pixel on an even-indexed row or column is blue and every odd-odd pixel is black.
    - For each pixel inside a rectangle, ignore the blue gridline rule and recolor it only by its depth from that rectangle's nearest edge.
    - Depth 0 keeps the rectangle's original color, odd positive depths become orange, and even positive depths become yellow.

    Key insight:
    - The puzzle is not sequential painting; each pixel is classified by whether it lies outside every rectangle or inside one rectangle, and the rectangle-local layer rule has higher priority than the global gridline pattern.
    """
    g = np.asarray(input_grid, dtype=int)
    rectangles = _find_rectangles(g, bg=Color.BLACK)
    out = _gridline_background(g.shape)
    for rectangle in rectangles:
        _paint_rectangle_layers(out, rectangle)
    return out


def _sample_ordered_intervals(total: int, len_a: int, len_b: int) -> tuple[int, int] | None:
    max_a = total - len_a - len_b - 3
    if max_a < 1:
        return None
    start_a = random.randint(1, max_a)
    min_b = start_a + len_a + 2
    max_b = total - len_b - 1
    if min_b > max_b:
        return None
    start_b = random.randint(min_b, max_b)
    return start_a, start_b


def _rect_has_grid_conflict(r0: int, c0: int, r1: int, c1: int) -> bool:
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = min(r - r0, r1 - r, c - c0, c1 - c)
            if depth > 0 and (r % 2 == 0 or c % 2 == 0):
                return True
    return False


def _sample_dimensions(size: int) -> tuple[int, int, int, int] | None:
    max_side = min(9, size - 4)
    want_small = random.random() < 0.45
    for _ in range(200):
        h1 = random.randint(3, max_side)
        w1 = random.randint(3, max_side)
        h2 = random.randint(3, max_side)
        w2 = random.randint(3, max_side)
        if h1 + h2 + 4 > size or w1 + w2 + 4 > size:
            continue
        if max(min(h1, w1), min(h2, w2)) < 5:
            continue
        if want_small and min(min(h1, w1), min(h2, w2)) > 4:
            continue
        return h1, w1, h2, w2
    return None


def _validate_semantics(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if h != w or h % 2 == 0 or h > 30 or w > 30:
        return False
    rectangles = _find_rectangles(grid, bg=Color.BLACK)
    if len(rectangles) != 2:
        return False
    colors = sorted(rect[-1] for rect in rectangles)
    if colors != [Color.RED, Color.GREEN]:
        return False

    (r0a, c0a, r1a, c1a, _), (r0b, c0b, r1b, c1b, _) = rectangles
    if min(r0a, c0a, r0b, c0b) <= 0:
        return False
    if max(r1a, c1a, r1b, c1b) >= h - 1:
        return False
    if min(r1a - r0a + 1, c1a - c0a + 1, r1b - r0b + 1, c1b - c0b + 1) < 3:
        return False

    row_gap = max(r0a, r0b) - min(r1a, r1b) - 1
    col_gap = max(c0a, c0b) - min(c1a, c1b) - 1
    if row_gap < 2 or col_gap < 2:
        return False
    if not (_rect_has_grid_conflict(r0a, c0a, r1a, c1a) and _rect_has_grid_conflict(r0b, c0b, r1b, c1b)):
        return False

    out = main(grid)
    if not isinstance(out, np.ndarray) or out.shape != grid.shape:
        return False
    if not np.any(out == Color.YELLOW) or not np.any(out == Color.ORANGE):
        return False

    saw_blue = False
    saw_black = False
    for r in range(h):
        for c in range(w):
            expected = _expected_color(r, c, rectangles)
            if int(out[r, c]) != expected:
                return False
            if all(not (r0 <= r <= r1 and c0 <= c <= c1) for r0, c0, r1, c1, _ in rectangles):
                if expected == Color.BLUE:
                    saw_blue = True
                if expected == Color.BLACK:
                    saw_black = True
    return saw_blue and saw_black


def _generate_attempt() -> np.ndarray | None:
    size = random.choice(list(range(13, 22, 2)))
    dims = _sample_dimensions(size)
    if dims is None:
        return None
    h1, w1, h2, w2 = dims

    row_starts = _sample_ordered_intervals(size, h1, h2)
    col_starts = _sample_ordered_intervals(size, w1, w2)
    if row_starts is None or col_starts is None:
        return None
    top_r, bottom_r = row_starts
    left_c, right_c = col_starts

    diagonal_flip = random.random() < 0.5
    if diagonal_flip:
        rects = [
            (top_r, right_c, top_r + h1 - 1, right_c + w1 - 1, Color.RED),
            (bottom_r, left_c, bottom_r + h2 - 1, left_c + w2 - 1, Color.GREEN),
        ]
    else:
        rects = [
            (top_r, left_c, top_r + h1 - 1, left_c + w1 - 1, Color.RED),
            (bottom_r, right_c, bottom_r + h2 - 1, right_c + w2 - 1, Color.GREEN),
        ]

    if not all(_rect_has_grid_conflict(r0, c0, r1, c1) for r0, c0, r1, c1, _ in rects):
        return None

    grid = np.full((size, size), Color.BLACK, dtype=int)
    for r0, c0, r1, c1, color in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)

    return grid if _validate_semantics(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC instance")
