from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 8
YELLOW = 4
ORANGE = 7
RED = 2
INPUT_COLORS = (0, 1, 3, 5, 6, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
OBJECT_COUNT_RANGE = (2, 4)
PLACEMENT_MARGIN = 1
PLACEMENT_ATTEMPTS = 200
GEN_ATTEMPTS = 300


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    counts = np.bincount(grid.ravel().astype(int), minlength=10)
    return int(np.argmax(counts))


def _bg_neighbor_count(grid: np.ndarray, bg: int, r: int, c: int) -> int:
    count = 0
    h, w = grid.shape
    for dr, dc in DIRS4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == bg:
            count += 1
    return count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground pixel by how many 4-neighbors are background.

    Pixels with no background neighbors keep their original color.
    Pixels with 1, 2, or at least 3 background neighbors become
    yellow, orange, or red respectively.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    out = g.copy()
    h, w = g.shape

    for r in range(h):
        for c in range(w):
            if int(g[r, c]) == bg:
                continue
            exposed = _bg_neighbor_count(g, bg, r, c)
            if exposed == 1:
                out[r, c] = YELLOW
            elif exposed == 2:
                out[r, c] = ORANGE
            elif exposed >= 3:
                out[r, c] = RED

    return out


def _rect_coords(h: int, w: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(h) for c in range(w)}


def _coords_to_mask(coords: set[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    r1 = max(rows)
    c1 = max(cols)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out.copy()


def _component_sizes(mask: np.ndarray) -> list[int]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    sizes: list[int] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            size = 0
            while q:
                rr, cc = q.popleft()
                size += 1
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            sizes.append(size)
    return sizes


def _is_single_component(mask: np.ndarray) -> bool:
    return _component_sizes(mask) == [int(mask.sum())]


def _add_arm(
    coords: set[tuple[int, int]],
    side: str,
    anchor: int,
    length: int,
    body_h: int,
    body_w: int,
) -> None:
    if side == "top":
        for step in range(1, length + 1):
            coords.add((-step, anchor))
    elif side == "bottom":
        for step in range(1, length + 1):
            coords.add((body_h - 1 + step, anchor))
    elif side == "left":
        for step in range(1, length + 1):
            coords.add((anchor, -step))
    else:
        for step in range(1, length + 1):
            coords.add((anchor, body_w - 1 + step))


def _carve_notch(coords: set[tuple[int, int]], side: str, body_h: int, body_w: int) -> None:
    if side in ("top", "bottom"):
        span = random.randint(1, body_w - 2)
        start = random.randint(1, body_w - 1 - span)
        row = 0 if side == "top" else body_h - 1
        for c in range(start, start + span):
            coords.discard((row, c))
    else:
        span = random.randint(1, body_h - 2)
        start = random.randint(1, body_h - 1 - span)
        col = 0 if side == "left" else body_w - 1
        for r in range(start, start + span):
            coords.discard((r, col))


def _make_bulky_shape() -> np.ndarray:
    body_h = random.randint(4, 7)
    body_w = random.randint(4, 7)
    coords = _rect_coords(body_h, body_w)

    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    arm_count = random.randint(1, 2)
    arm_sides = sides[:arm_count]

    for side in arm_sides:
        length = random.randint(1, 3)
        if side in ("top", "bottom"):
            anchor = random.randint(1, body_w - 2)
        else:
            anchor = random.randint(1, body_h - 2)
        _add_arm(coords, side, anchor, length, body_h, body_w)

    remaining = [side for side in sides if side not in arm_sides]
    if remaining and random.random() < 0.9:
        _carve_notch(coords, random.choice(remaining), body_h, body_w)

    mask = _random_transform(_coords_to_mask(coords))
    assert mask.sum() >= 5
    assert _is_single_component(mask)
    return mask


def _make_branchy_shape() -> np.ndarray:
    body_h = random.randint(2, 4)
    body_w = random.randint(2, 4)
    coords = _rect_coords(body_h, body_w)

    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    arm_count = random.randint(2, 4)
    for side in sides[:arm_count]:
        length = random.randint(1, 4)
        if side in ("top", "bottom"):
            if body_w >= 3:
                anchor = random.randint(1, body_w - 2)
            else:
                anchor = random.randint(0, body_w - 1)
        else:
            if body_h >= 3:
                anchor = random.randint(1, body_h - 2)
            else:
                anchor = random.randint(0, body_h - 1)
        _add_arm(coords, side, anchor, length, body_h, body_w)

    if body_h >= 3 and body_w >= 3 and random.random() < 0.6:
        coords.discard(random.choice([(0, 0), (0, body_w - 1), (body_h - 1, 0), (body_h - 1, body_w - 1)]))

    mask = _random_transform(_coords_to_mask(coords))
    assert mask.sum() >= 5
    assert _is_single_component(mask)
    return mask


def _shape_catalog() -> list[np.ndarray]:
    shapes = [_make_bulky_shape(), _make_branchy_shape()]
    extra_count = random.randint(0, 2)
    for _ in range(extra_count):
        if random.random() < 0.5:
            shapes.append(_make_bulky_shape())
        else:
            shapes.append(_make_branchy_shape())
    return shapes


def _expanded_cells(cells: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _place_shapes(
    grid_h: int,
    grid_w: int,
    shapes: list[np.ndarray],
    colors: list[int],
) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked: set[tuple[int, int]] = set()
    order = sorted(range(len(shapes)), key=lambda i: int(shapes[i].sum()), reverse=True)

    for idx in order:
        mask = shapes[idx]
        color = colors[idx]
        h, w = mask.shape
        if h + 2 * PLACEMENT_MARGIN > grid_h or w + 2 * PLACEMENT_MARGIN > grid_w:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = random.randint(PLACEMENT_MARGIN, grid_h - h - PLACEMENT_MARGIN)
            left = random.randint(PLACEMENT_MARGIN, grid_w - w - PLACEMENT_MARGIN)
            cells = [
                (top + r, left + c)
                for r in range(h)
                for c in range(w)
                if bool(mask[r, c])
            ]
            if any(cell in blocked for cell in cells):
                continue

            for r, c in cells:
                grid[r, c] = color
            blocked |= _expanded_cells(cells, (grid_h, grid_w))
            placed = True
            break

        if not placed:
            return None

    return grid


def _non_bg_colors(grid: np.ndarray) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) != BG]


def _validate_input(grid: np.ndarray) -> bool:
    if grid.dtype.kind not in {"i", "u"}:
        return False

    colors = _non_bg_colors(grid)
    if not (OBJECT_COUNT_RANGE[0] <= len(colors) <= OBJECT_COUNT_RANGE[1]):
        return False

    exposures: set[int] = set()
    h, w = grid.shape
    for color in colors:
        mask = grid == color
        if not _is_single_component(mask):
            return False
        if int(mask.sum()) < 5:
            return False

    for r in range(h):
        for c in range(w):
            value = int(grid[r, c])
            if value == BG:
                continue
            for dr, dc in DIRS4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    neighbor = int(grid[rr, cc])
                    if neighbor != BG and neighbor != value:
                        return False
            exposures.add(min(3, _bg_neighbor_count(grid, BG, r, c)))

    if exposures != {0, 1, 2, 3}:
        return False

    out = main(grid)
    if not np.array_equal(out[grid == BG], grid[grid == BG]):
        return False
    if not (np.any(out == YELLOW) and np.any(out == ORANGE) and np.any(out == RED)):
        return False
    return True


def _attempt_grid() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    object_count = random.randint(*OBJECT_COUNT_RANGE)

    shapes = _shape_catalog()
    if len(shapes) < object_count:
        return None
    shapes = shapes[:object_count]
    colors = random.sample(INPUT_COLORS, object_count)

    grid = _place_shapes(grid_h, grid_w, shapes, colors)
    if grid is None:
        return None
    if not _validate_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
