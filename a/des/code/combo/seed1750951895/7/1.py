import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = (2, 3, 1)  # red, green, blue
SOURCE_COLORS = (4, 5, 6, 8, 9)

MIN_SIDE = 10
MAX_SIDE = 22
MIN_GROUPS = 2
MAX_GROUPS = 4
MIN_PIXELS = 3
MAX_PIXELS = 7
MAX_ATTEMPTS = 400
PLACEMENT_GAP = 2


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _sorted_group_coords(coords: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    pts = [tuple(map(int, rc)) for rc in coords]
    if height >= width:
        return sorted(pts, key=lambda rc: (rc[0], rc[1]))
    return sorted(pts, key=lambda rc: (rc[1], rc[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group scattered foreground pixels by color, sort each color-group along its
    bbox orientation, and recolor each ordered sequence with a repeating RGB
    palette.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.zeros_like(g)
    colors = [int(color) for color in np.unique(g) if int(color) != BG]
    assert 2 <= len(colors) <= 4, "expected 2-4 foreground colors"

    for color in colors:
        coords = np.argwhere(g == color)
        assert len(coords) > 0
        ordered = _sorted_group_coords(coords)
        for i, (r, c) in enumerate(ordered):
            out[r, c] = PALETTE[i % len(PALETTE)]

    return out


def _far_enough(cell: tuple[int, int], used: set[tuple[int, int]]) -> bool:
    r, c = cell
    for ur, uc in used:
        if abs(r - ur) <= 1 and abs(c - uc) <= 1:
            return False
    return True


def _box_far_enough(
    box: tuple[int, int, int, int], boxes: Iterable[tuple[int, int, int, int]]
) -> bool:
    r0, c0, r1, c1 = box
    for or0, oc0, or1, oc1 in boxes:
        if not (r1 + PLACEMENT_GAP < or0 or or1 + PLACEMENT_GAP < r0 or c1 + PLACEMENT_GAP < oc0 or oc1 + PLACEMENT_GAP < c0):
            return False
    return True


def _choose_dimensions(orientation: str) -> tuple[int, int]:
    if orientation == "vertical":
        h = random.randint(4, 8)
        w = random.randint(2, min(5, h))
        if random.random() < 0.2:
            w = h
        else:
            w = min(w, h)
    else:
        w = random.randint(4, 8)
        h = random.randint(2, min(5, w))
        if h == w:
            h = max(2, h - 1)
    if orientation == "vertical" and h < w:
        h, w = w, h
    if orientation == "horizontal" and w <= h:
        w = h + random.randint(1, 3)
    return h, w


def _sample_cells_in_box(
    top: int, left: int, height: int, width: int, count: int, used: set[tuple[int, int]]
) -> list[tuple[int, int]] | None:
    candidates = [(r, c) for r in range(top, top + height) for c in range(left, left + width)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    local_used = set(used)

    must_have = {
        (top, left),
        (top + height - 1, left + width - 1),
    }
    if height >= 2:
        must_have.add((top + height - 1, left))
    if width >= 2:
        must_have.add((top, left + width - 1))

    for cell in sorted(must_have):
        if cell not in candidates or not _far_enough(cell, local_used):
            return None
        chosen.append(cell)
        local_used.add(cell)

    for cell in candidates:
        if len(chosen) >= count:
            break
        if cell in must_have:
            continue
        if _far_enough(cell, local_used):
            chosen.append(cell)
            local_used.add(cell)

    if len(chosen) != count:
        return None

    coords = np.array(chosen, dtype=int)
    r0, c0, r1, c1 = _bbox(coords)
    if r0 != top or c0 != left or r1 != top + height - 1 or c1 != left + width - 1:
        return None
    return chosen


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_SIDE, MAX_SIDE)
    width = random.randint(MIN_SIDE, MAX_SIDE)
    group_count = random.randint(MIN_GROUPS, MAX_GROUPS)

    if group_count == 2:
        orientations = ["vertical", "horizontal"]
    else:
        base = ["vertical", "horizontal"]
        orientations = base + random.choices(base, k=group_count - 2)
        random.shuffle(orientations)

    grid = np.zeros((height, width), dtype=int)
    colors = random.sample(SOURCE_COLORS, group_count)
    used_cells: set[tuple[int, int]] = set()
    placed_boxes: list[tuple[int, int, int, int]] = []

    for color, orientation in zip(colors, orientations):
        pixel_count = random.randint(MIN_PIXELS, MAX_PIXELS)
        for _ in range(80):
            box_h, box_w = _choose_dimensions(orientation)
            if box_h > height - 2 or box_w > width - 2:
                continue
            top = random.randint(1, height - box_h - 1)
            left = random.randint(1, width - box_w - 1)
            box = (top, left, top + box_h - 1, left + box_w - 1)
            if not _box_far_enough(box, placed_boxes):
                continue
            cells = _sample_cells_in_box(top, left, box_h, box_w, pixel_count, used_cells)
            if cells is None:
                continue
            for r, c in cells:
                grid[r, c] = color
                used_cells.add((r, c))
            placed_boxes.append(box)
            break
        else:
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if len({grid.shape, out.shape}) != 1:
        return None
    if len(np.unique(grid)) - 1 != group_count:
        return None
    if not ({2, 3, 1} & set(np.unique(out))):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
