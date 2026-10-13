from __future__ import annotations

import random
from collections import deque

import numpy as np

BACKGROUND = 0
COUNTER_COLOR = 5
GAP_COLOR = 8
OBJECT_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _same_color_components(grid: np.ndarray, forbidden: set[int]) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in forbidden or visited[r, c]:
                continue

            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = [cell[0] for cell in cells]
            cols = [cell[1] for cell in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            components.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "patch": grid[r0 : r1 + 1, c0 : c1 + 1].copy(),
                    "size": len(cells),
                }
            )

    return components


def _is_hollow_rectangle(component: dict[str, object]) -> bool:
    patch = np.asarray(component["patch"], dtype=int)
    color = int(component["color"])
    if patch.shape != (4, 4):
        return False
    return bool(
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
        and np.all(patch[1:3, 1:3] == BACKGROUND)
    )


def _extract_hollow_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    components = _same_color_components(grid, forbidden={BACKGROUND, COUNTER_COLOR})
    hollows = [component for component in components if _is_hollow_rectangle(component)]
    hollows.sort(key=lambda component: int(component["bbox"][1]))
    return hollows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort hollow 4x4 rectangles, concatenate them with counter-sized gaps, and fill every gap teal.

    Input structure:
    - The grid has a black background, several isolated 4x4 monochrome objects, and a small grey counter near the lower-right.
    - Some 4x4 objects are hollow rectangular frames whose inner 2x2 cells are black, while others are solid 4x4 squares used as distractors.
    - Object colors are distinct, and the grey counter consists of 1 to 4 pixels in a horizontal or vertical line.

    Rule:
    - Count the grey counter pixels to get N, then ignore the counter itself.
    - Keep only the hollow 4x4 rectangles and sort them by their original left-edge column.
    - Build a 4-row output strip by placing the sorted rectangles left to right, using exactly N columns of separation between consecutive rectangles.
    - Fill every separator gap completely with teal and leave all other cells unchanged from the retained hollow rectangles.

    Key insight:
    - The grey counter does not move an object directly; it sets the uniform gap width used when the sorted hollow rectangles are reassembled into a single strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    gap_width = int(np.sum(g == COUNTER_COLOR))
    assert 1 <= gap_width <= 4, "expected a 1-4 pixel grey counter"

    hollows = _extract_hollow_rectangles(g)
    assert hollows, "expected at least one hollow rectangle"

    out_width = len(hollows) * 4 + (len(hollows) - 1) * gap_width
    assert out_width <= 30, "output must stay within ARC size limits"

    out = np.zeros((4, out_width), dtype=int)
    cursor = 0
    for idx, component in enumerate(hollows):
        out[:, cursor : cursor + 4] = np.asarray(component["patch"], dtype=int)
        cursor += 4
        if idx + 1 < len(hollows):
            out[:, cursor : cursor + gap_width] = GAP_COLOR
            cursor += gap_width

    return out


def _hollow_patch(color: int) -> np.ndarray:
    patch = np.full((4, 4), int(color), dtype=int)
    patch[1:3, 1:3] = BACKGROUND
    return patch


def _solid_patch(color: int) -> np.ndarray:
    return np.full((4, 4), int(color), dtype=int)


def _boxes_too_close(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], padding: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (
        ar1 + padding < br0
        or br1 + padding < ar0
        or ac1 + padding < bc0
        or bc1 + padding < ac0
    )


def _counter_spec(height: int, width: int, count: int) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    orientation = random.choice(("h", "v"))
    if orientation == "h":
        row = height - 1 - random.randint(0, min(2, height - 1))
        start_col = width - count - random.randint(0, min(2, width - count))
        cells = [(row, start_col + offset) for offset in range(count)]
    else:
        col = width - 1 - random.randint(0, min(2, width - 1))
        start_row = height - count - random.randint(0, min(2, height - count))
        cells = [(start_row + offset, col) for offset in range(count)]

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    bbox = (min(rows), min(cols), max(rows), max(cols))
    return cells, bbox


def _find_slot(
    positions: list[tuple[int, int]],
    boxes: list[tuple[int, int, int, int]],
    used_hollow_cols: set[int],
    require_new_hollow_col: bool,
) -> tuple[int, int] | None:
    candidates = positions[:]
    random.shuffle(candidates)
    for r, c in candidates:
        if require_new_hollow_col and c in used_hollow_cols:
            continue
        box = (r, c, r + 3, c + 3)
        if any(_boxes_too_close(box, existing) for existing in boxes):
            continue
        return r, c
    return None


def _build_input() -> np.ndarray | None:
    counter_count = random.randint(1, 4)
    max_hollows = min(5, (30 + counter_count) // (4 + counter_count))
    if max_hollows < 3:
        return None

    hollow_count = random.randint(3, max_hollows)
    solid_count = random.randint(1, min(2, len(OBJECT_COLORS) - hollow_count))
    total_objects = hollow_count + solid_count

    height = random.randint(10, 18)
    width = random.randint(20, 28)
    grid = np.zeros((height, width), dtype=int)

    counter_cells, counter_box = _counter_spec(height, width, counter_count)
    boxes = [counter_box]
    used_hollow_cols: set[int] = set()

    positions = [(r, c) for r in range(height - 3) for c in range(width - 3)]
    colors = random.sample(OBJECT_COLORS, total_objects)
    hollow_colors = colors[:hollow_count]
    solid_colors = colors[hollow_count:]

    placements: list[tuple[np.ndarray, int, int]] = []

    for color in hollow_colors:
        slot = _find_slot(positions, boxes, used_hollow_cols, require_new_hollow_col=True)
        if slot is None:
            return None
        r, c = slot
        placements.append((_hollow_patch(color), r, c))
        boxes.append((r, c, r + 3, c + 3))
        used_hollow_cols.add(c)

    for color in solid_colors:
        slot = _find_slot(positions, boxes, used_hollow_cols, require_new_hollow_col=False)
        if slot is None:
            return None
        r, c = slot
        placements.append((_solid_patch(color), r, c))
        boxes.append((r, c, r + 3, c + 3))

    random.shuffle(placements)
    for patch, r, c in placements:
        grid[r : r + 4, c : c + 4] = patch

    for r, c in counter_cells:
        if grid[r, c] != BACKGROUND:
            return None
        grid[r, c] = COUNTER_COLOR

    if max(grid.shape) > 30:
        return None
    if len(_extract_hollow_rectangles(grid)) != hollow_count:
        return None
    out = main(grid)
    if out.shape[0] != 4 or max(out.shape) > 30:
        return None
    if out.shape[1] != hollow_count * 4 + (hollow_count - 1) * counter_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
