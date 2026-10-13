from __future__ import annotations

import random

import numpy as np


BACKGROUND = 8
RECT_SHADOW = 5
MARKER_COLOR = 9
OBJECT_COLORS = (1, 2, 3, 4, 6, 7)

GRID_MIN = 12
GRID_MAX = 20
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _marker_and_direction(grid: np.ndarray) -> tuple[tuple[int, int], str]:
    h, w = grid.shape
    marker_cells = np.argwhere(grid == MARKER_COLOR)
    edge_cells = [
        (int(r), int(c))
        for r, c in marker_cells
        if int(r) in (0, h - 1) or int(c) in (0, w - 1)
    ]
    assert len(edge_cells) == 1, "expected exactly one edge marker"
    r, c = edge_cells[0]
    assert not ((r in (0, h - 1)) and (c in (0, w - 1))), "marker must not sit in a corner"

    if c == w - 1:
        return (r, c), "right"
    if c == 0:
        return (r, c), "left"
    if r == h - 1:
        return (r, c), "down"
    return (r, c), "up"


def _shadow_slice(
    bbox_: tuple[int, int, int, int], direction: str, step: int
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox_
    if direction == "right":
        return [(int(r), int(c1 + step)) for r in range(r0, r1 + 1)]
    if direction == "left":
        return [(int(r), int(c0 - step)) for r in range(r0, r1 + 1)]
    if direction == "down":
        return [(int(r1 + step), int(c)) for c in range(c0, c1 + 1)]
    return [(int(r0 - step), int(c)) for c in range(c0, c1 + 1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each object's rectangular shadow in the direction indicated by the edge marker.

    The shadow length is the object's area. Perfect solid rectangles cast grey
    shadows; all other connected shapes cast maroon shadows.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    marker, direction = _marker_and_direction(g)
    blocker_mask = g != BACKGROUND
    object_mask = blocker_mask.copy()
    object_mask[marker] = False

    comps = _components(object_mask)
    comps.sort(key=lambda pts: _bbox(pts)[:2])

    out = g.copy()
    for coords in comps:
        area = int(len(coords))
        bbox_ = _bbox(coords)
        r0, c0, r1, c1 = bbox_
        bbox_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        shadow_color = RECT_SHADOW if area == bbox_area else MARKER_COLOR

        for step in range(1, area + 1):
            cells = _shadow_slice(bbox_, direction, step)
            if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in cells):
                break
            if any(bool(blocker_mask[rr, cc]) for rr, cc in cells):
                break
            for rr, cc in cells:
                out[rr, cc] = shadow_color

    return out


def _positive_partition(total: int, parts: int) -> list[int]:
    values = [1] * parts
    for _ in range(total - parts):
        values[random.randrange(parts)] += 1
    return values


def _grow_connected_mask(height: int, width: int, target_area: int) -> np.ndarray | None:
    mask = np.zeros((height, width), dtype=bool)
    start = (random.randrange(height), random.randrange(width))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], height, width))

    while int(mask.sum()) < target_area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if bool(mask[cell]):
            continue
        mask[cell] = True
        for nbr in _neighbors4(cell[0], cell[1], height, width):
            if not bool(mask[nbr]):
                frontier.add(nbr)

    return mask


def _make_rectangle() -> np.ndarray:
    while True:
        height = random.randint(1, 3)
        width = random.randint(2, 4)
        area = height * width
        if 2 <= area <= 8:
            return np.ones((height, width), dtype=bool)


def _make_irregular() -> np.ndarray:
    for _ in range(200):
        height = random.randint(2, 4)
        width = random.randint(2, 4)
        min_area = max(3, height + width - 1)
        max_area = min(height * width - 1, 8)
        if min_area > max_area:
            continue
        target_area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(height, width, target_area)
        if mask is None:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape != (height, width):
            continue
        if bool(np.all(cropped)):
            continue
        return cropped
    raise RuntimeError("failed to build irregular object")


def _rotate_from_right(grid: np.ndarray, direction: str) -> np.ndarray:
    if direction == "right":
        return grid
    if direction == "down":
        return np.rot90(grid, 3)
    if direction == "left":
        return np.rot90(grid, 2)
    return np.rot90(grid, 1)


def _gen_rightward_input() -> np.ndarray | None:
    n_objects = random.randint(2, 4)
    kinds = [True, False] + [random.choice([True, False]) for _ in range(n_objects - 2)]
    random.shuffle(kinds)
    modes = ["full", "trunc"] + [random.choice(["full", "trunc"]) for _ in range(n_objects - 2)]
    random.shuffle(modes)

    objects = []
    for is_rect, mode in zip(kinds, modes):
        mask = _make_rectangle() if is_rect else _make_irregular()
        height, width = map(int, mask.shape)
        area = int(mask.sum())
        objects.append(
            {
                "mask": mask,
                "is_rect": bool(is_rect),
                "mode": mode,
                "height": height,
                "width": width,
                "area": area,
            }
        )

    total_heights = sum(obj["height"] for obj in objects)
    min_h = max(GRID_MIN, total_heights + n_objects + 1)
    if min_h > GRID_MAX:
        return None
    grid_h = random.randint(min_h, GRID_MAX)
    gap_sizes = _positive_partition(grid_h - total_heights, n_objects + 1)

    min_w = max(
        GRID_MIN,
        max(
            obj["width"] + (obj["area"] if obj["mode"] == "full" else 1)
            for obj in objects
        ),
    )
    if min_w > GRID_MAX:
        return None
    grid_w = random.randint(min_w, GRID_MAX)

    colors = random.sample(OBJECT_COLORS, n_objects)
    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)

    row = gap_sizes[0]
    used_rows: set[int] = set()
    for obj, color, gap_after in zip(objects, colors, gap_sizes[1:]):
        height = int(obj["height"])
        width = int(obj["width"])
        area = int(obj["area"])
        top = int(row)

        if obj["mode"] == "full":
            max_left = grid_w - width - area
            if max_left < 0:
                return None
            left = random.randint(0, max_left)
        else:
            max_available = min(area - 1, grid_w - width)
            if max_available < 1:
                return None
            available = random.randint(1, max_available)
            left = grid_w - width - available

        mask = obj["mask"]
        coords = np.argwhere(mask)
        for dr, dc in coords:
            grid[top + int(dr), left + int(dc)] = int(color)

        obj["top"] = top
        obj["left"] = left
        obj["color"] = int(color)
        used_rows.update(range(top, top + height))
        row += height + gap_after

    free_rows = [r for r in range(grid_h) if r not in used_rows]
    if not free_rows:
        return None
    marker_row = random.choice(free_rows)
    grid[marker_row, grid_w - 1] = MARKER_COLOR
    return grid


def _valid_generated_input(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    if np.array_equal(grid, out):
        return False
    if int(np.count_nonzero(grid == MARKER_COLOR)) != 1:
        return False
    if not bool(np.any(out == RECT_SHADOW)):
        return False
    if int(np.count_nonzero(out == MARKER_COLOR)) <= 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        base = _gen_rightward_input()
        if base is None:
            continue
        direction = random.choice(("right", "down", "left", "up"))
        grid = _rotate_from_right(base, direction)
        if _valid_generated_input(grid):
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GEN_ATTEMPTS} attempts")
