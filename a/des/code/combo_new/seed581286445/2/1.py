import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)

CELL_INNER_W = 3
MIN_ROOM_H = 3
MIN_ROOM_W = 3
ROOM_COUNT_RANGE = (2, 7)
UPPER_H_RANGE = (10, 23)
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _room_components(grid: np.ndarray, wall: int) -> list[np.ndarray]:
    upper = grid[:-3, :]
    return _components(upper != wall)


def _parse_legend(grid: np.ndarray, bg: int, wall: int) -> dict[int, np.ndarray]:
    h, w = grid.shape
    assert h >= 5

    swatch_row = h - 3
    pattern_rows = grid[h - 2 : h, :]
    sep_cols = [c for c in range(w) if np.all(pattern_rows[:, c] == wall)]
    assert len(sep_cols) >= 2, "legend must have vertical separators"

    mapping: dict[int, np.ndarray] = {}
    for left, right in zip(sep_cols, sep_cols[1:]):
        if right - left <= 2:
            continue
        inner_cols = list(range(left + 1, right))
        pat_cols = [c for c in inner_cols if np.any((pattern_rows[:, c] != bg) & (pattern_rows[:, c] != wall))]
        swatch_cols = [c for c in inner_cols if int(grid[swatch_row, c]) not in (bg, wall)]
        if len(pat_cols) != 2 or len(swatch_cols) != 1:
            continue
        color = int(grid[swatch_row, swatch_cols[0]])
        mapping[color] = grid[h - 2 : h, pat_cols].copy()

    assert mapping, "legend mapping must be non-empty"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-bounded room with the checkerboard template keyed by its internal marker color.

    Input structure:
    - The background is the most common color.
    - A connected grey wall network in the upper part of the grid partitions the background into closed rooms.
    - Each room contains exactly one non-background, non-grey marker pixel whose color is unique across rooms.
    - Just above the bottom two template rows, a legend row provides one color swatch per room color; below it, each swatch's cell contains a 2×2 alternating-color template, with cells separated by grey vertical lines.

    Rule:
    - Read each room's single marker color C.
    - In the legend, find the cell whose swatch color is C and extract that cell's 2×2 template.
    - For the entire room interior, use the room component's bbox top-left corner as phase origin and tile the 2×2 template periodically over all non-grey cells of the room, covering the marker pixel too.
    - Leave the grey wall network and the legend area unchanged.

    Key insight:
    - The lone room marker is an index into the bottom legend, and the retrieved 2×2 template must be tiled with a bbox-aligned phase rather than copied once.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)

    non_bg = g[g != bg]
    assert non_bg.size > 0
    wall_vals, wall_counts = np.unique(non_bg, return_counts=True)
    wall = int(wall_vals[np.argmax(wall_counts)])

    legend = _parse_legend(g, bg, wall)
    rooms = _room_components(g, wall)
    assert rooms, "expected at least one room"

    out = g.copy()
    upper = g[:-3, :]
    for coords in rooms:
        values = upper[coords[:, 0], coords[:, 1]]
        marker_values = [int(v) for v in values if int(v) != bg]
        assert len(marker_values) == 1, "each room must contain exactly one marker"
        marker = marker_values[0]
        assert marker in legend, "every room marker must appear in the legend"
        template = legend[marker]

        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        for r, c in coords:
            out[int(r), int(c)] = int(template[(int(r) - r0) % 2, (int(c) - c0) % 2])

    return out


def _can_split_vert(rect: tuple[int, int, int, int]) -> bool:
    _r0, c0, _r1, c1 = rect
    return c1 - c0 + 1 >= 2 * MIN_ROOM_W + 1


def _can_split_horiz(rect: tuple[int, int, int, int]) -> bool:
    r0, _c0, r1, _c1 = rect
    return r1 - r0 + 1 >= 2 * MIN_ROOM_H + 1


def _split_rooms(upper_h: int, width: int, room_count: int) -> np.ndarray | None:
    grid = np.full((upper_h, width), BG, dtype=int)
    grid[0, :] = WALL
    grid[-1, :] = WALL
    grid[:, 0] = WALL
    grid[:, -1] = WALL

    rects: list[tuple[int, int, int, int]] = [(1, 1, upper_h - 2, width - 2)]
    while len(rects) < room_count:
        options: list[tuple[int, str]] = []
        for idx, rect in enumerate(rects):
            if _can_split_vert(rect):
                options.append((idx, "v"))
            if _can_split_horiz(rect):
                options.append((idx, "h"))
        if not options:
            return None

        idx, axis = random.choice(options)
        r0, c0, r1, c1 = rects.pop(idx)
        if axis == "v":
            split_c = random.randint(c0 + MIN_ROOM_W, c1 - MIN_ROOM_W)
            grid[r0 - 1 : r1 + 2, split_c] = WALL
            rects.extend([(r0, c0, r1, split_c - 1), (r0, split_c + 1, r1, c1)])
        else:
            split_r = random.randint(r0 + MIN_ROOM_H, r1 - MIN_ROOM_H)
            grid[split_r, c0 - 1 : c1 + 2] = WALL
            rects.extend([(r0, c0, split_r - 1, c1), (split_r + 1, c0, r1, c1)])

    return grid


def _add_notch(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    if random.random() < 0.25:
        return

    r0, c0, r1, c1 = rect
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    sides: list[str] = []
    if height >= 3:
        sides.extend(["top", "bottom"])
    if width >= 3:
        sides.extend(["left", "right"])
    if not sides:
        return

    side = random.choice(sides)
    if side == "top":
        col = random.randint(c0, c1)
        length = random.randint(1, max(1, height - 1))
        grid[r0 : r0 + length, col] = WALL
    elif side == "bottom":
        col = random.randint(c0, c1)
        length = random.randint(1, max(1, height - 1))
        grid[r1 - length + 1 : r1 + 1, col] = WALL
    elif side == "left":
        row = random.randint(r0, r1)
        length = random.randint(1, max(1, width - 1))
        grid[row, c0 : c0 + length] = WALL
    else:
        row = random.randint(r0, r1)
        length = random.randint(1, max(1, width - 1))
        grid[row, c1 - length + 1 : c1 + 1] = WALL


def _checkerboard(color_a: int, color_b: int, flip: bool) -> np.ndarray:
    if flip:
        return np.array([[color_b, color_a], [color_a, color_b]], dtype=int)
    return np.array([[color_a, color_b], [color_b, color_a]], dtype=int)


def _expected_output(grid: np.ndarray, room_templates: dict[int, np.ndarray]) -> np.ndarray:
    out = grid.copy()
    rooms = _room_components(grid, WALL)
    upper = grid[:-3, :]
    for coords in rooms:
        values = upper[coords[:, 0], coords[:, 1]]
        marker_values = [int(v) for v in values if int(v) != BG]
        if len(marker_values) != 1:
            raise AssertionError("generator created room without exactly one marker")
        marker = marker_values[0]
        template = room_templates[marker]
        r0 = int(coords[:, 0].min())
        c0 = int(coords[:, 1].min())
        for r, c in coords:
            out[int(r), int(c)] = int(template[(int(r) - r0) % 2, (int(c) - c0) % 2])
    return out


def _gen_attempt() -> np.ndarray | None:
    room_count = random.randint(*ROOM_COUNT_RANGE)
    width = 1 + room_count * (CELL_INNER_W + 1)
    upper_h = random.randint(*UPPER_H_RANGE)
    if upper_h + 3 > 30 or width > 30:
        return None

    upper = _split_rooms(upper_h, width, room_count)
    if upper is None:
        return None

    rects = [(1, 1, upper_h - 2, width - 2)]
    # Reconstruct the leaf rooms from the current upper grid rather than from split history.
    base_rooms = _components(upper != WALL)
    if len(base_rooms) != room_count:
        return None
    for comp in base_rooms:
        rr0 = int(comp[:, 0].min())
        cc0 = int(comp[:, 1].min())
        rr1 = int(comp[:, 0].max())
        cc1 = int(comp[:, 1].max())
        rects.append((rr0, cc0, rr1, cc1))
    for rect in rects[1:]:
        _add_notch(upper, rect)
        if random.random() < 0.35:
            _add_notch(upper, rect)

    rooms = _components(upper != WALL)
    if len(rooms) != room_count:
        return None
    if any(len(comp) < 5 for comp in rooms):
        return None

    grid = np.full((upper_h + 3, width), BG, dtype=int)
    grid[:upper_h, :] = upper

    marker_colors = random.sample(PALETTE, room_count)
    room_templates: dict[int, np.ndarray] = {}
    room_order = rooms[:]
    random.shuffle(room_order)
    for color, comp in zip(marker_colors, room_order):
        idx = random.randrange(len(comp))
        r, c = map(int, comp[idx])
        grid[r, c] = int(color)

    swatch_row = upper_h
    sep_cols = list(range(0, width, CELL_INNER_W + 1))
    grid[swatch_row:, sep_cols] = WALL

    for i, color in enumerate(marker_colors):
        cell_left = 1 + i * (CELL_INNER_W + 1)
        pattern_offset = random.randint(0, CELL_INNER_W - 2)
        pattern_cols = [cell_left + pattern_offset, cell_left + pattern_offset + 1]
        grid[swatch_row, pattern_cols[0]] = int(color)

        a, b = random.sample(PALETTE, 2)
        template = _checkerboard(a, b, flip=bool(random.getrandbits(1)))
        room_templates[int(color)] = template
        grid[upper_h + 1 : upper_h + 3, pattern_cols] = template

    rooms_with_markers = _room_components(grid, WALL)
    if len(rooms_with_markers) != room_count:
        return None
    for comp in rooms_with_markers:
        vals = grid[:-3, :][comp[:, 0], comp[:, 1]]
        if np.count_nonzero(vals != BG) != 1:
            return None

    counts = Counter(int(v) for v in grid.ravel())
    if counts.most_common(1)[0][0] != BG:
        return None
    wall_counts = Counter({k: v for k, v in counts.items() if k != BG})
    if wall_counts.most_common(1)[0][0] != WALL:
        return None

    expected = _expected_output(grid, room_templates)
    solved = main(grid)
    if not np.array_equal(expected, solved):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
