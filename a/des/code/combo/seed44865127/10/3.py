from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRECTIONS = ("up", "down", "left", "right")
BORDER_COLORS = {
    "up": 2,     # red
    "down": 1,   # blue
    "left": 3,   # green
    "right": 4,  # yellow
}
DIR_STEP = {
    "up": (1, 0),
    "down": (-1, 0),
    "left": (0, 1),
    "right": (0, -1),
}
NON_BORDER_COLORS = [0, 5, 6, 7, 8, 9]
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _infer_direction(coords: np.ndarray, center_r: int, center_c: int) -> str:
    mean_r = float(coords[:, 0].mean())
    mean_c = float(coords[:, 1].mean())
    delta_r = float(center_r) - mean_r
    delta_c = float(center_c) - mean_c
    if abs(delta_r) > abs(delta_c):
        return "up" if delta_r > 0 else "down"
    return "left" if delta_c > 0 else "right"


def _border_cells(coords: np.ndarray, direction: str, shape: tuple[int, int], bg_mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = shape
    dr, dc = DIR_STEP[direction]
    cells = {(int(r), int(c)) for r, c in coords}
    border: set[tuple[int, int]] = set()
    for r, c in cells:
        nr = r + dr
        nc = c + dc
        if not (0 <= nr < h and 0 <= nc < w):
            continue
        if (nr, nc) in cells:
            continue
        if bg_mask[nr, nc]:
            border.add((nr, nc))
    return border


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _get_bg(g)
    center_r, center_c = h // 2, w // 2

    object_mask = g != bg
    object_mask[center_r, center_c] = False
    components = _connected_components(object_mask)
    assert components

    out = g.copy()
    bg_mask = g == bg
    for coords in components:
        direction = _infer_direction(coords, center_r, center_c)
        border_color = BORDER_COLORS[direction]
        for r, c in _border_cells(coords, direction, g.shape, bg_mask):
            out[r, c] = border_color
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shape_has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    padded = np.pad(mask, 1, constant_values=False)
    seen = np.zeros_like(padded, dtype=bool)
    stack = [(0, 0)]
    seen[0, 0] = True
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h + 2 and 0 <= nc < w + 2 and not padded[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                stack.append((nr, nc))
    return bool((~padded & ~seen).any())


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(80):
        target_h = random.randint(2, max_h)
        target_w = random.randint(2, max_w)
        max_area = target_h * target_w
        min_area = max(3, (target_h + target_w) // 2)
        area = random.randint(min_area, max_area)

        seed = (random.randrange(target_h), random.randrange(target_w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < target_h and 0 <= nc < target_w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        cells = _normalize(cells)
        if random.random() < 0.5:
            max_c = max(c for _, c in cells)
            cells = {(r, max_c - c) for r, c in cells}
        if random.random() < 0.5:
            max_r = max(r for r, _ in cells)
            cells = {(max_r - r, c) for r, c in cells}

        real_h = max(r for r, _ in cells) + 1
        real_w = max(c for _, c in cells) + 1
        if real_h < 2 or real_w < 2:
            continue

        mask = np.zeros((real_h, real_w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _shape_has_hole(mask):
            continue
        return mask
    return None


def _direction_margin(coords: np.ndarray, center_r: int, center_c: int, direction: str) -> bool:
    mean_r = float(coords[:, 0].mean())
    mean_c = float(coords[:, 1].mean())
    delta_r = float(center_r) - mean_r
    delta_c = float(center_c) - mean_c
    if direction == "up":
        return delta_r >= abs(delta_c) + 1.0
    if direction == "down":
        return -delta_r >= abs(delta_c) + 1.0
    if direction == "left":
        return delta_c >= abs(delta_r) + 1.0
    return -delta_c >= abs(delta_r) + 1.0


def _shape_coords(mask: np.ndarray, top: int, left: int) -> np.ndarray:
    pts = np.argwhere(mask)
    pts = pts.astype(int)
    pts[:, 0] += int(top)
    pts[:, 1] += int(left)
    return pts


def _placement_ranges(direction: str, patch_h: int, patch_w: int, h: int, w: int, center_r: int, center_c: int) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if direction == "up":
        row_lo, row_hi = 1, center_r - patch_h - 1
        col_lo, col_hi = 1, w - patch_w - 2
    elif direction == "down":
        row_lo, row_hi = center_r + 2, h - patch_h - 2
        col_lo, col_hi = 1, w - patch_w - 2
    elif direction == "left":
        row_lo, row_hi = 1, h - patch_h - 2
        col_lo, col_hi = 1, center_c - patch_w - 1
    else:
        row_lo, row_hi = 1, h - patch_h - 2
        col_lo, col_hi = center_c + 2, w - patch_w - 2
    if row_lo > row_hi or col_lo > col_hi:
        return None
    return (row_lo, row_hi), (col_lo, col_hi)


def _choose_object_colors(directions: list[str], bg: int, marker: int) -> list[int] | None:
    available = [color for color in range(10) if color not in {bg, marker}]
    random.shuffle(available)
    used: set[int] = set()
    chosen: list[int] = []
    for direction in directions:
        options = [color for color in available if color not in used and color != BORDER_COLORS[direction]]
        if not options:
            return None
        color = random.choice(options)
        used.add(color)
        chosen.append(int(color))
    return chosen


def _touches_existing(coords: np.ndarray, occupied: set[tuple[int, int]]) -> bool:
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if (rr, cc) in occupied:
            return True
        for nr, nc in _neighbors4(rr, cc):
            if (nr, nc) in occupied:
                return True
    return False


def _place_objects(grid: np.ndarray, bg: int, object_specs: list[tuple[str, int]]) -> bool:
    h, w = grid.shape
    center_r, center_c = h // 2, w // 2
    object_cells: set[tuple[int, int]] = set()
    reserved_border: set[tuple[int, int]] = set()

    for direction, color in object_specs:
        axis_cap = 4
        ortho_cap = 5
        if direction in {"up", "down"}:
            max_h = axis_cap
            max_w = min(ortho_cap, w - 4)
        else:
            max_h = min(ortho_cap, h - 4)
            max_w = axis_cap
        mask = _random_shape(max_h, max_w)
        if mask is None:
            return False
        patch_h, patch_w = mask.shape
        ranges = _placement_ranges(direction, patch_h, patch_w, h, w, center_r, center_c)
        if ranges is None:
            return False
        (row_lo, row_hi), (col_lo, col_hi) = ranges

        placed = False
        for _ in range(200):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            coords = _shape_coords(mask, top, left)
            if _infer_direction(coords, center_r, center_c) != direction:
                continue
            if not _direction_margin(coords, center_r, center_c, direction):
                continue
            if _touches_existing(coords, object_cells | {(center_r, center_c)}):
                continue

            border = _border_cells(coords, direction, grid.shape, grid == bg)
            if not border:
                continue
            if any((r, c) == (center_r, center_c) for r, c in border):
                continue
            if any((r, c) in object_cells or (r, c) in reserved_border for r, c in border):
                continue
            if any((int(r), int(c)) in reserved_border for r, c in coords):
                continue

            for r, c in coords:
                grid[int(r), int(c)] = int(color)
                object_cells.add((int(r), int(c)))
            reserved_border.update(border)
            placed = True
            break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.choice([15, 17, 19, 21, 23])
    width = random.choice([15, 17, 19, 21, 23])
    bg, marker = random.sample(NON_BORDER_COLORS, 2)

    directions = list(DIRECTIONS)
    if random.random() < 0.5:
        directions.append(random.choice(DIRECTIONS))
    random.shuffle(directions)

    colors = _choose_object_colors(directions, bg, marker)
    if colors is None:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    center_r, center_c = height // 2, width // 2
    grid[center_r, center_c] = int(marker)

    specs = list(zip(directions, colors, strict=False))
    if not _place_objects(grid, bg, specs):
        return None

    object_mask = grid != bg
    object_mask[center_r, center_c] = False
    components = _connected_components(object_mask)
    if len(components) != len(specs):
        return None

    seen_dirs = Counter(_infer_direction(comp, center_r, center_c) for comp in components)
    for direction in DIRECTIONS:
        if seen_dirs[direction] < 1:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
