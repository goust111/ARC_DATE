import random
from collections import Counter

import numpy as np


NEIGHBOR_TO_COLOR = {
    0: 2,  # red
    1: 1,  # blue
    2: 3,  # green
    3: 4,  # yellow
}
HIGH_NEIGHBOR_COLOR = 6  # magenta


GENERATION_ATTEMPTS = 300
SAFE_INPUT_COLORS = (0, 5, 7, 8, 9)
BASE_TEMPLATES = (
    ((0, 0),),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (1, 1)),
)
EXTRA_TEMPLATES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3), (0, 4)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
)


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values, minlength=10)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((int(cr), int(cc)))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(cr + dr)
                nc = int(cc + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if seen[nr, nc] or not mask[nr, nc]:
                    continue
                seen[nr, nc] = True
                stack.append((nr, nc))
        parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return (
        int(rows.min()),
        int(rows.max()),
        int(cols.min()),
        int(cols.max()),
    )


def _ranges_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _count_neighbors(boxes: list[tuple[int, int, int, int]]) -> list[int]:
    counts: list[int] = []
    for i, (r0, r1, c0, c1) in enumerate(boxes):
        neighbors = 0
        for j, (rr0, rr1, cc0, cc1) in enumerate(boxes):
            if i == j:
                continue
            if _ranges_overlap(r0, r1, rr0, rr1) or _ranges_overlap(c0, c1, cc0, cc1):
                neighbors += 1
        counts.append(int(neighbors))
    return counts


def _color_for_neighbors(count: int) -> int:
    return int(NEIGHBOR_TO_COLOR.get(int(count), HIGH_NEIGHBOR_COLOR))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by the number of other objects whose row-range
    or column-range bbox projection overlaps its own.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    objects = _components(g != bg)
    if not objects:
        return g.copy()

    boxes = [_bbox(coords) for coords in objects]
    neighbor_counts = _count_neighbors(boxes)

    out = g.copy()
    for coords, count in zip(objects, neighbor_counts):
        color = _color_for_neighbors(int(count))
        out[coords[:, 0], coords[:, 1]] = int(color)
    return out


def _template_degrees(template: tuple[tuple[int, int], ...]) -> list[int]:
    row_counts = Counter(r for r, _c in template)
    col_counts = Counter(c for _r, c in template)
    return [int(row_counts[r] + col_counts[c] - 2) for r, c in template]


def _random_component_mask(height: int, width: int) -> np.ndarray:
    if height == 1 or width == 1:
        return np.ones((height, width), dtype=bool)

    start_left = random.random() < 0.5
    row = 0
    col = 0 if start_left else width - 1
    col_step = 1 if start_left else -1
    cells = {(row, col)}
    down_moves = height - 1
    side_moves = width - 1

    while down_moves or side_moves:
        if down_moves and side_moves:
            move_down = random.random() < 0.5
        else:
            move_down = bool(down_moves)
        if move_down:
            row += 1
            down_moves -= 1
        else:
            col += col_step
            side_moves -= 1
        cells.add((row, col))

    target = random.randint(len(cells), height * width)
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _build_gadget(template: tuple[tuple[int, int], ...]) -> tuple[list[tuple[int, int, int, int]], list[int], int, int]:
    n_rows = 1 + max(r for r, _c in template)
    n_cols = 1 + max(c for _r, c in template)
    row_sizes = [random.randint(1, 3) for _ in range(n_rows)]
    col_sizes = [random.randint(1, 3) for _ in range(n_cols)]
    row_gaps = [random.randint(1, 2) for _ in range(max(0, n_rows - 1))]
    col_gaps = [random.randint(1, 2) for _ in range(max(0, n_cols - 1))]

    row_starts: list[int] = []
    cursor = 0
    for idx, size in enumerate(row_sizes):
        row_starts.append(int(cursor))
        cursor += int(size)
        if idx < len(row_gaps):
            cursor += int(row_gaps[idx])
    zone_h = int(cursor)

    col_starts: list[int] = []
    cursor = 0
    for idx, size in enumerate(col_sizes):
        col_starts.append(int(cursor))
        cursor += int(size)
        if idx < len(col_gaps):
            cursor += int(col_gaps[idx])
    zone_w = int(cursor)

    boxes = []
    for row_group, col_group in template:
        r0 = int(row_starts[row_group])
        c0 = int(col_starts[col_group])
        h = int(row_sizes[row_group])
        w = int(col_sizes[col_group])
        boxes.append((r0, r0 + h - 1, c0, c0 + w - 1))
    return boxes, _template_degrees(template), zone_h, zone_w


def _gen_attempt() -> np.ndarray | None:
    bg, fg = random.sample(SAFE_INPUT_COLORS, 2)
    templates = list(BASE_TEMPLATES)
    extra_count = random.randint(0, 2)
    if extra_count:
        templates.extend(random.sample(EXTRA_TEMPLATES, extra_count))

    gadgets = []
    for template in templates:
        gadgets.append(_build_gadget(template))

    top = random.randint(1, 3)
    left = random.randint(1, 3)
    placements: list[tuple[int, int, int, int]] = []
    expected_counts: list[int] = []

    for index, (boxes, degrees, zone_h, zone_w) in enumerate(gadgets):
        for r0, r1, c0, c1 in boxes:
            placements.append((r0 + top, r1 + top, c0 + left, c1 + left))
        expected_counts.extend(int(v) for v in degrees)
        if index < len(gadgets) - 1:
            top += zone_h + random.randint(2, 4)
            left += zone_w + random.randint(2, 4)

    max_row = max(r1 for _r0, r1, _c0, _c1 in placements)
    max_col = max(c1 for _r0, _r1, _c0, c1 in placements)
    grid_h = int(max_row + 1 + random.randint(1, 4))
    grid_w = int(max_col + 1 + random.randint(1, 4))
    if grid_h > 40 or grid_w > 40:
        return None

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    for r0, r1, c0, c1 in placements:
        mask = _random_component_mask(r1 - r0 + 1, c1 - c0 + 1)
        view = grid[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(view[mask] != bg):
            return None
        view[mask] = int(fg)

    if int(np.count_nonzero(grid == bg)) <= int(np.count_nonzero(grid != bg)):
        return None

    comps = _components(grid != bg)
    if len(comps) != len(placements):
        return None

    actual_boxes = [_bbox(coords) for coords in comps]
    actual_counts = _count_neighbors(actual_boxes)
    if sorted(actual_counts) != sorted(expected_counts):
        return None
    if set(actual_counts) != {0, 1, 2, 3, 4}:
        return None

    out = main(grid)
    present_output_colors = set(map(int, np.unique(out[out != bg])))
    if present_output_colors != {1, 2, 3, 4, 6}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
