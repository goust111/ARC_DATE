import random
from typing import Iterable

import numpy as np


BG = 0
RED = 2
GREEN = 3

DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_NAMES = tuple(DIR_VECS.keys())

AREA_TO_DIMS = {
    4: [(2, 2)],
    6: [(2, 3), (3, 2)],
    8: [(2, 4), (4, 2)],
    9: [(3, 3)],
    10: [(2, 5), (5, 2)],
    12: [(3, 4), (4, 3)],
}

GEN_RETRIES = 500
PLACE_RETRIES = 300


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    pts = list(points)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(points: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected = set()
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    return set(points) == expected


def _midpoint_outside(bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mid_r = r0 + (r1 - r0) // 2
    mid_c = c0 + (c1 - c0) // 2
    dr, dc = direction
    if (dr, dc) == (-1, 0):
        return r0 - 1, mid_c
    if (dr, dc) == (1, 0):
        return r1 + 1, mid_c
    if (dr, dc) == (0, -1):
        return mid_r, c0 - 1
    return mid_r, c1 + 1


def _extract_rectangles(grid: np.ndarray) -> tuple[list[dict[str, object]], int]:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    rectangles: list[dict[str, object]] = []
    frame_colors = sorted(set(map(int, np.unique(g))) - {bg, RED, GREEN})
    for color in frame_colors:
        for points in _components(g == color):
            bbox = _bbox(points)
            if not _is_rectangular_frame(points, bbox):
                continue
            r0, c0, r1, c1 = bbox
            if r1 - r0 <= 1 or c1 - c0 <= 1:
                continue
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            red_pts = np.argwhere(inner == RED)
            green_pts = np.argwhere(inner == GREEN)
            if len(red_pts) != 1 or len(green_pts) != 1:
                continue
            tail_r, tail_c = map(int, red_pts[0] + np.array([r0 + 1, c0 + 1]))
            head_r, head_c = map(int, green_pts[0] + np.array([r0 + 1, c0 + 1]))
            dr = head_r - tail_r
            dc = head_c - tail_c
            if abs(dr) + abs(dc) != 1:
                continue
            rectangles.append(
                {
                    "color": int(color),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "direction": (int(dr), int(dc)),
                    "area": int((r1 - r0 - 1) * (c1 - c0 - 1)),
                }
            )
    return rectangles, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow a border-colored tentacle out of each hollow rectangle using its interior arrow and area.

    Input structure:
    - The background is black.
    - One or more one-pixel-thick rectangular frames sit on the canvas; each frame is a closed axis-aligned rectangle in a non-black border color.
    - Inside every frame there is exactly one two-pixel arrow marker: one red cell and one green cell, 4-neighbor adjacent, with direction given by red→green.

    Rule:
    - For each rectangular frame, compute its interior area N, counting every cell strictly inside the border, including the arrow cells.
    - Read the arrow direction from the red cell toward the green cell.
    - On the outside of the frame, find the midpoint of the border side pointed to by the arrow; if the midpoint falls between two cells, use the left or upper one.
    - Starting from the cell immediately outside that midpoint, draw a straight line of length N in the arrow direction using the frame's border color.
    - Keep the original frame and the interior arrow unchanged.

    Key insight:
    - The arrow contributes only direction, while the frame's interior area contributes only length; both clues must be combined to place the outward line correctly.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rectangles, _bg = _extract_rectangles(g)
    for rect in rectangles:
        bbox = rect["bbox"]
        color = int(rect["color"])
        direction = rect["direction"]
        length = int(rect["area"])
        start_r, start_c = _midpoint_outside(bbox, direction)
        dr, dc = direction
        for step in range(length):
            rr = start_r + dr * step
            cc = start_c + dc * step
            if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                out[rr, cc] = color
    return out


def _footprint(
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    direction: tuple[int, int],
    length: int,
) -> set[tuple[int, int]]:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    pts = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    sr, sc = _midpoint_outside((r0, c0, r1, c1), direction)
    dr, dc = direction
    for step in range(length):
        pts.add((sr + dr * step, sc + dc * step))
    return pts


def _padded(points: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    pad: set[tuple[int, int]] = set()
    for r, c in points:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    pad.add((rr, cc))
    return pad


def _arrow_cells(
    rng: random.Random,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
    direction: str,
) -> tuple[tuple[int, int], tuple[int, int]]:
    top = r0 + 1
    left = c0 + 1
    bottom = r0 + inner_h
    right = c0 + inner_w
    if direction == "up":
        tail_r = rng.randint(top + 1, bottom)
        tail_c = rng.randint(left, right)
        return (tail_r, tail_c), (tail_r - 1, tail_c)
    if direction == "down":
        tail_r = rng.randint(top, bottom - 1)
        tail_c = rng.randint(left, right)
        return (tail_r, tail_c), (tail_r + 1, tail_c)
    if direction == "left":
        tail_r = rng.randint(top, bottom)
        tail_c = rng.randint(left + 1, right)
        return (tail_r, tail_c), (tail_r, tail_c - 1)
    tail_r = rng.randint(top, bottom)
    tail_c = rng.randint(left, right - 1)
    return (tail_r, tail_c), (tail_r, tail_c + 1)


def _paint_frame(grid: np.ndarray, r0: int, c0: int, inner_h: int, inner_w: int, color: int) -> None:
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _valid_top_left_ranges(
    h: int,
    w: int,
    inner_h: int,
    inner_w: int,
    direction: str,
    length: int,
) -> tuple[range, range] | None:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r_lo = 0
    r_hi = h - outer_h
    c_lo = 0
    c_hi = w - outer_w
    if direction == "up":
        r_lo = max(r_lo, length)
    elif direction == "down":
        r_hi = min(r_hi, h - outer_h - length)
    elif direction == "left":
        c_lo = max(c_lo, length)
    else:
        c_hi = min(c_hi, w - outer_w - length)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return range(r_lo, r_hi + 1), range(c_lo, c_hi + 1)


def _place_object(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    border_color: int,
    area: int,
    inner_h: int,
    inner_w: int,
    direction: str,
) -> bool:
    h, w = grid.shape
    ranges = _valid_top_left_ranges(h, w, inner_h, inner_w, direction, area)
    if ranges is None:
        return False
    row_range, col_range = ranges
    if not row_range or not col_range:
        return False
    direction_vec = DIR_VECS[direction]
    for _ in range(PLACE_RETRIES):
        r0 = rng.choice(row_range)
        c0 = rng.choice(col_range)
        footprint = _footprint(r0, c0, inner_h, inner_w, direction_vec, area)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in footprint):
            continue
        padded = _padded(footprint, h, w)
        if any(reserved[r, c] for r, c in padded):
            continue
        _paint_frame(grid, r0, c0, inner_h, inner_w, border_color)
        tail, head = _arrow_cells(rng, r0, c0, inner_h, inner_w, direction)
        grid[tail] = RED
        grid[head] = GREEN
        for r, c in padded:
            reserved[r, c] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    border_color = rng.choice([1, 4, 5, 6, 7, 8, 9])
    n_rects = rng.randint(2, 4)
    areas = rng.sample(sorted(AREA_TO_DIMS), n_rects)
    directions = list(DIR_NAMES)
    rng.shuffle(directions)
    chosen_dirs = directions[:n_rects]

    specs = []
    for area, direction in zip(areas, chosen_dirs):
        inner_h, inner_w = rng.choice(AREA_TO_DIMS[area])
        footprint_size = (inner_h + 2) * (inner_w + 2) + area
        specs.append((footprint_size, area, inner_h, inner_w, direction))
    specs.sort(reverse=True)

    for _size, area, inner_h, inner_w, direction in specs:
        if not _place_object(rng, grid, reserved, border_color, area, inner_h, inner_w, direction):
            return None

    rectangles, bg = _extract_rectangles(grid)
    if bg != BG or len(rectangles) != n_rects:
        return None
    seen_areas = sorted(int(rect["area"]) for rect in rectangles)
    if seen_areas != sorted(areas):
        return None
    seen_dirs = sorted(tuple(rect["direction"]) for rect in rectangles)
    expected_dirs = sorted(DIR_VECS[name] for name in chosen_dirs)
    if seen_dirs != expected_dirs:
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
    raise RuntimeError("generate_input: failed to build a valid puzzle")
