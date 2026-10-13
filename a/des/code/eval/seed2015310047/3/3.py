import random
from typing import Dict, List, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]

BG = 0
DIVIDER = 5
RECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
VISIBLE_CORNERS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 500


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _same_color_component(zone: np.ndarray, start_r: int, start_c: int, visited: np.ndarray) -> List[Tuple[int, int]]:
    color = int(zone[start_r, start_c])
    stack = [(start_r, start_c)]
    visited[start_r, start_c] = True
    cells: List[Tuple[int, int]] = []
    h, w = zone.shape

    while stack:
        r, c = stack.pop()
        cells.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(zone[nr, nc]) == color:
                visited[nr, nc] = True
                stack.append((nr, nc))
    return cells


def _read_legend_order(zone: np.ndarray, bg: int) -> List[int]:
    if zone.size == 0:
        raise ValueError("empty legend zone")

    visited = np.zeros(zone.shape, dtype=bool)
    blocks: List[Tuple[int, int, int]] = []
    h, w = zone.shape

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(zone[r, c]) == bg:
                continue
            cells = _same_color_component(zone, r, c, visited)
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            color = int(zone[r, c])

            if len(cells) != 4 or r1 - r0 != 1 or c1 - c0 != 1:
                raise ValueError("legend components must be solid 2x2 blocks")
            if not np.all(zone[r0 : r1 + 1, c0 : c1 + 1] == color):
                raise ValueError("legend block must be solid")
            blocks.append((r0, c0, color))

    if not blocks:
        raise ValueError("no legend blocks found")

    blocks.sort()
    return [color for _, _, color in blocks]


def _find_divider_column(grid: np.ndarray, bg: int) -> int:
    h, w = grid.shape
    for c in range(1, w - 1):
        col = grid[:, c]
        if not np.all(col == col[0]):
            continue
        if int(col[0]) == bg:
            continue
        left = grid[:, :c]
        right = grid[:, c + 1 :]
        if left.size == 0 or right.size == 0 or not np.any(right != bg):
            continue
        try:
            _read_legend_order(left, bg)
        except ValueError:
            continue
        return c
    raise ValueError("divider column not found")


def _render_from_order(height: int, width: int, rects: Dict[int, BBox], top_to_bottom: List[int]) -> np.ndarray:
    canvas = np.full((height, width), BG, dtype=int)
    for color in reversed(top_to_bottom):
        r0, c0, r1, c1 = rects[color]
        canvas[r0 : r1 + 1, c0 : c1 + 1] = color
    return canvas


def _edge_pair_clear(covered: np.ndarray, bbox: BBox, corner: str) -> bool:
    r0, c0, r1, c1 = bbox
    top = not covered[r0, c0 : c1 + 1].any()
    bottom = not covered[r1, c0 : c1 + 1].any()
    left = not covered[r0 : r1 + 1, c0].any()
    right = not covered[r0 : r1 + 1, c1].any()
    if corner == "tl":
        return top and left
    if corner == "tr":
        return top and right
    if corner == "bl":
        return bottom and left
    return bottom and right


def _has_visible_corner(canvas: np.ndarray, bbox: BBox, color: int) -> bool:
    r0, c0, r1, c1 = bbox
    visible = canvas == color
    top = bool(np.all(visible[r0, c0 : c1 + 1]))
    bottom = bool(np.all(visible[r1, c0 : c1 + 1]))
    left = bool(np.all(visible[r0 : r1 + 1, c0]))
    right = bool(np.all(visible[r0 : r1 + 1, c1]))
    return (top and left) or (top and right) or (bottom and left) or (bottom and right)


def _bbox_matches(canvas: np.ndarray, bbox: BBox, color: int) -> bool:
    pts = np.argwhere(canvas == color)
    if len(pts) == 0:
        return False
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return (int(r0), int(c0), int(r1), int(c1)) == bbox


def _background_is_dominant(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts)) == BG and int(counts[BG]) > int(np.max(counts[1:]))


def _sample_legend_rows(rng: random.Random, height: int, count: int) -> List[int]:
    minimum = 3 * count + 1
    if height < minimum:
        raise ValueError("grid too short for legend")
    gaps = [1] * (count + 1)
    extra = height - minimum
    for _ in range(extra):
        gaps[rng.randrange(len(gaps))] += 1

    rows: List[int] = []
    cursor = gaps[0]
    for i in range(count):
        rows.append(cursor)
        cursor += 2 + gaps[i + 1]
    return rows


def _place_rectangles(
    rng: random.Random,
    height: int,
    width: int,
    wrong_top_order: List[int],
) -> Dict[int, BBox] | None:
    rects: Dict[int, BBox] = {}
    covered_by_higher = np.zeros((height, width), dtype=bool)
    has_overlap = False

    min_h = 3
    min_w = 3
    max_h = max(min_h, min(8, height - 1))
    max_w = max(min_w, min(8, width - 1))
    if max_h < min_h or max_w < min_w:
        return None

    for idx, color in enumerate(wrong_top_order):
        placed = False
        need_overlap = idx > 0 and (not has_overlap or rng.random() < 0.7)
        for _ in range(400):
            rect_h = rng.randint(min_h, max_h)
            rect_w = rng.randint(min_w, max_w)
            r0 = rng.randint(0, height - rect_h)
            c0 = rng.randint(0, width - rect_w)
            bbox = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            corner = rng.choice(VISIBLE_CORNERS)
            overlap = bool(covered_by_higher[r0 : r0 + rect_h, c0 : c0 + rect_w].any())

            if need_overlap and not overlap:
                continue
            if not _edge_pair_clear(covered_by_higher, bbox, corner):
                continue

            rects[color] = bbox
            covered_by_higher[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
            has_overlap = has_overlap or overlap
            placed = True
            break

        if not placed:
            return None

    if not has_overlap:
        return None
    return rects


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(3, 5)
    height = rng.randint(max(10, 3 * count + 1), 18)
    canvas_w = rng.randint(11, 18)
    legend_w = rng.randint(4, 6)

    colors = rng.sample(RECT_COLORS, count)
    correct_top_order = colors[:]
    rng.shuffle(correct_top_order)
    wrong_top_order = correct_top_order[:]
    while wrong_top_order == correct_top_order:
        rng.shuffle(wrong_top_order)

    rects = _place_rectangles(rng, height, canvas_w, wrong_top_order)
    if rects is None:
        return None

    wrong_canvas = _render_from_order(height, canvas_w, rects, wrong_top_order)
    correct_canvas = _render_from_order(height, canvas_w, rects, correct_top_order)

    if np.array_equal(wrong_canvas, correct_canvas):
        return None
    if int(np.count_nonzero(wrong_canvas == BG)) < height * canvas_w // 6:
        return None

    for color in colors:
        if not _has_visible_corner(wrong_canvas, rects[color], color):
            return None
        if not _bbox_matches(wrong_canvas, rects[color], color):
            return None
        if int(np.count_nonzero(correct_canvas == color)) == 0:
            return None

    grid = np.full((height, legend_w + 1 + canvas_w), BG, dtype=int)
    grid[:, legend_w] = DIVIDER
    grid[:, legend_w + 1 :] = wrong_canvas

    legend_rows = _sample_legend_rows(rng, height, count)
    legend_c = rng.randint(0, legend_w - 2)
    for row, color in zip(legend_rows, correct_top_order):
        grid[row : row + 2, legend_c : legend_c + 2] = color

    if not _background_is_dominant(grid):
        return None
    if _find_divider_column(grid, BG) != legend_w:
        return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapped solid rectangles from their visible fragments and
    repaint them in the top-to-bottom order given by the 2x2 legend blocks.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    divider_col = _find_divider_column(g, bg)

    legend_order = _read_legend_order(g[:, :divider_col], bg)
    canvas = g[:, divider_col + 1 :]
    height, width = canvas.shape

    rects: Dict[int, BBox] = {}
    for color in legend_order:
        pts = np.argwhere(canvas == color)
        if len(pts) == 0:
            raise ValueError(f"legend color {color} is absent from the canvas")
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        rects[color] = (int(r0), int(c0), int(r1), int(c1))

    return _render_from_order(height, width, rects, legend_order)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
