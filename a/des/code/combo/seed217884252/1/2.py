from __future__ import annotations

from collections import deque
import random

import numpy as np


BG_COLOR = 0
OBJECT_COLOR = 5
LEGEND_SHAPE = (4, 2)
DIRECTION_ORDER = ("up", "right", "down", "left")
NON_OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

# Generator configuration.
GRID_HEIGHT_RANGE = (18, 26)
GRID_WIDTH_RANGE = (18, 26)
OBJECT_COUNT_CHOICES = (2, 3, 3, 4)
OBJECT_HEIGHT_RANGE = (5, 9)
OBJECT_WIDTH_RANGE = (5, 10)
MAX_HOLES_PER_OBJECT = 3
OBJECT_RETRIES = 120
CASE_RETRIES = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_grid_border(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _enclosed_holes(object_mask: np.ndarray) -> list[list[tuple[int, int]]]:
    holes: list[list[tuple[int, int]]] = []
    h, w = object_mask.shape
    for pts in _components(~object_mask):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        holes.append(pts)
    return holes


def _direction_index(object_mask: np.ndarray) -> int:
    side_has_gap = [
        bool(np.any(~object_mask[0, :])),
        bool(np.any(~object_mask[:, -1])),
        bool(np.any(~object_mask[-1, :])),
        bool(np.any(~object_mask[:, 0])),
    ]
    assert sum(side_has_gap) == 1, "each object must have a unique gap side"
    return int(side_has_gap.index(True))


def _find_legend(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[tuple[int, int, int, int]] = []
    for pts in _components(g != BG_COLOR):
        if not _touches_grid_border(pts, h, w):
            continue
        r0, c0, r1, c1 = _bbox(pts)
        if (r1 - r0 + 1, c1 - c0 + 1) != LEGEND_SHAPE:
            continue
        if len(pts) != LEGEND_SHAPE[0] * LEGEND_SHAPE[1]:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(sub == BG_COLOR):
            continue
        candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "expected one 4x2 legend on the grid edge"
    r0, c0, r1, c1 = candidates[0]
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    fill_colors = [int(g[r0 + row, c0 + 1]) for row in range(LEGEND_SHAPE[0])]
    return fill_colors, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each object's enclosed holes using the legend row selected by the object's gap direction.
    """
    g = np.asarray(grid, dtype=int)
    fill_colors, legend_mask = _find_legend(g)
    out = g.copy()

    object_mask = (g != BG_COLOR) & (~legend_mask)
    for pts in _components(object_mask):
        r0, c0, r1, c1 = _bbox(pts)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in pts:
            local_mask[r - r0, c - c0] = True

        fill_color = fill_colors[_direction_index(local_mask)]
        for hole in _enclosed_holes(local_mask):
            for rr, cc in hole:
                out[r0 + rr, c0 + cc] = fill_color

    return out


def _place_legend(
    grid: np.ndarray,
    occupied: np.ndarray,
    marker_colors: list[int],
    fill_colors: list[int],
) -> None:
    h, w = grid.shape
    side = random.choice(("left", "right", "top", "bottom"))
    if side == "left":
        r0 = random.randint(0, h - LEGEND_SHAPE[0])
        c0 = 0
    elif side == "right":
        r0 = random.randint(0, h - LEGEND_SHAPE[0])
        c0 = w - LEGEND_SHAPE[1]
    elif side == "top":
        r0 = 0
        c0 = random.randint(0, w - LEGEND_SHAPE[1])
    else:
        r0 = h - LEGEND_SHAPE[0]
        c0 = random.randint(0, w - LEGEND_SHAPE[1])

    for row in range(LEGEND_SHAPE[0]):
        grid[r0 + row, c0] = marker_colors[row]
        grid[r0 + row, c0 + 1] = fill_colors[row]

    occupied[max(0, r0 - 1) : min(h, r0 + LEGEND_SHAPE[0] + 1), max(0, c0 - 1) : min(w, c0 + LEGEND_SHAPE[1] + 1)] = True


def _max_hole_count(height: int, width: int) -> int:
    interior_area = (height - 2) * (width - 2)
    return max(1, min(MAX_HOLES_PER_OBJECT, interior_area // 8))


def _make_object_patch(direction: str, height: int, width: int, hole_count: int) -> np.ndarray | None:
    patch = np.full((height, width), OBJECT_COLOR, dtype=int)
    if direction == "up":
        patch[0, random.randint(1, width - 2)] = BG_COLOR
    elif direction == "right":
        patch[random.randint(1, height - 2), width - 1] = BG_COLOR
    elif direction == "down":
        patch[height - 1, random.randint(1, width - 2)] = BG_COLOR
    else:
        patch[random.randint(1, height - 2), 0] = BG_COLOR

    blocked = np.zeros((height, width), dtype=bool)
    max_hole_h = min(3, height - 4)
    max_hole_w = min(4, width - 4)
    if max_hole_h < 1 or max_hole_w < 1:
        return None

    for _ in range(hole_count):
        placed = False
        for _ in range(80):
            hole_h = random.randint(1, max_hole_h)
            hole_w = random.randint(1, max_hole_w)
            r0 = random.randint(1, height - hole_h - 1)
            c0 = random.randint(1, width - hole_w - 1)
            if blocked[r0 - 1 : r0 + hole_h + 1, c0 - 1 : c0 + hole_w + 1].any():
                continue
            patch[r0 : r0 + hole_h, c0 : c0 + hole_w] = BG_COLOR
            blocked[r0 - 1 : r0 + hole_h + 1, c0 - 1 : c0 + hole_w + 1] = True
            placed = True
            break
        if not placed:
            return None

    object_mask = patch == OBJECT_COLOR
    if len(_enclosed_holes(object_mask)) != hole_count:
        return None
    if _direction_index(object_mask) != DIRECTION_ORDER.index(direction):
        return None
    return patch


def _sample_directions(count: int) -> list[str]:
    if count <= 4 and random.random() < 0.6:
        return random.sample(list(DIRECTION_ORDER), count)

    directions = [random.choice(DIRECTION_ORDER) for _ in range(count)]
    if len(set(directions)) == 1:
        directions[-1] = DIRECTION_ORDER[(DIRECTION_ORDER.index(directions[-1]) + 1) % 4]
    return directions


def _valid_positions(occupied: np.ndarray, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = occupied.shape
    ph, pw = shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            if occupied[r0 - 1 : r0 + ph + 1, c0 - 1 : c0 + pw + 1].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_object(
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    fill_color: int,
    r0: int,
    c0: int,
) -> None:
    ph, pw = patch.shape
    object_mask = patch == OBJECT_COLOR
    input_grid[r0 : r0 + ph, c0 : c0 + pw][object_mask] = OBJECT_COLOR
    output_grid[r0 : r0 + ph, c0 : c0 + pw][object_mask] = OBJECT_COLOR

    for hole in _enclosed_holes(object_mask):
        for rr, cc in hole:
            output_grid[r0 + rr, c0 + cc] = fill_color

    h, w = occupied.shape
    occupied[max(0, r0 - 1) : min(h, r0 + ph + 1), max(0, c0 - 1) : min(w, c0 + pw + 1)] = True


def _generate_case() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    input_grid = np.full((height, width), BG_COLOR, dtype=int)
    output_grid = np.full((height, width), BG_COLOR, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    colors = list(NON_OBJECT_COLORS)
    random.shuffle(colors)
    marker_colors = colors[:4]
    fill_colors = colors[4:]
    _place_legend(input_grid, occupied, marker_colors, fill_colors)
    output_grid[:, :] = input_grid

    object_count = random.choice(OBJECT_COUNT_CHOICES)
    directions = _sample_directions(object_count)
    specs: list[tuple[np.ndarray, int]] = []
    for direction in directions:
        patch = None
        for _ in range(OBJECT_RETRIES):
            obj_h = random.randint(*OBJECT_HEIGHT_RANGE)
            obj_w = random.randint(*OBJECT_WIDTH_RANGE)
            hole_count = random.randint(1, _max_hole_count(obj_h, obj_w))
            patch = _make_object_patch(direction, obj_h, obj_w, hole_count)
            if patch is not None:
                break
        if patch is None:
            return None
        fill_color = fill_colors[DIRECTION_ORDER.index(direction)]
        specs.append((patch, fill_color))

    specs.sort(key=lambda item: int(np.count_nonzero(item[0] == OBJECT_COLOR)), reverse=True)
    for patch, fill_color in specs:
        positions = _valid_positions(occupied, patch.shape)
        if not positions:
            return None
        r0, c0 = random.choice(positions)
        _paint_object(input_grid, output_grid, occupied, patch, fill_color, r0, c0)

    if np.array_equal(input_grid, output_grid):
        return None
    try:
        if not np.array_equal(main(input_grid), output_grid):
            return None
    except Exception:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(CASE_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
