from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
WALL = 5
LEGEND = 9
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8)

SOLID = np.ones((3, 3), dtype=bool)
STRIPES = np.array(
    [
        [True, False, True],
        [True, False, True],
        [True, False, True],
    ],
    dtype=bool,
)
CHECKER = (np.indices((3, 3)).sum(axis=0) % 2) == 0

LAYOUTS = ((1, 3), (2, 2), (2, 3), (3, 2))
ROOM_H_RANGE = (3, 6)
ROOM_W_RANGE = (3, 7)
TOP_MARGIN_RANGE = (1, 3)
GAP_RANGE = (1, 3)
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _border_mode(grid: np.ndarray) -> int:
    h, w = grid.shape
    border = list(map(int, grid[0, :])) + list(map(int, grid[h - 1, :]))
    if h > 2:
        border += list(map(int, grid[1 : h - 1, 0]))
        border += list(map(int, grid[1 : h - 1, w - 1]))
    return Counter(border).most_common(1)[0][0]


def _pattern_block(pattern: np.ndarray, color: int, bg: int) -> np.ndarray:
    return np.where(pattern, int(color), int(bg)).astype(int)


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    patterns = (SOLID, STRIPES, CHECKER)
    candidates: list[tuple[int, int, int]] = []

    row_order = [h - 3] + [r for r in range(h - 2) if r != h - 3]
    for r0 in row_order:
        for c0 in range(w - 10):
            solid = grid[r0 : r0 + 3, c0 : c0 + 3]
            colors = np.unique(solid)
            if len(colors) != 1:
                continue
            legend_color = int(colors[0])
            if legend_color == bg:
                continue
            if not np.array_equal(solid, _pattern_block(patterns[0], legend_color, bg)):
                continue
            if not np.all(grid[r0 : r0 + 3, c0 + 3] == bg):
                continue
            if not np.array_equal(
                grid[r0 : r0 + 3, c0 + 4 : c0 + 7],
                _pattern_block(patterns[1], legend_color, bg),
            ):
                continue
            if not np.all(grid[r0 : r0 + 3, c0 + 7] == bg):
                continue
            if not np.array_equal(
                grid[r0 : r0 + 3, c0 + 8 : c0 + 11],
                _pattern_block(patterns[2], legend_color, bg),
            ):
                continue
            candidates.append((r0, c0, legend_color))

    assert len(candidates) == 1, "expected one bottom legend"
    return candidates[0]


def _component_from_mask(mask: np.ndarray) -> list[tuple[list[tuple[int, int]], bool]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[tuple[list[tuple[int, int]], bool]] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        touches_border = False
        while q:
            r, c = q.popleft()
            coords.append((r, c))
            if r in (0, h - 1) or c in (0, w - 1):
                touches_border = True
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        components.append((coords, touches_border))
    return components


def _pick_wall_color(grid: np.ndarray, bg: int, legend_color: int) -> int:
    counts = Counter(map(int, grid.reshape(-1)))
    candidates = [color for color in counts if color not in {bg, legend_color}]
    assert candidates, "expected a wall color"
    wall = max(candidates, key=lambda color: counts[color])
    wall_mask = grid == wall
    wall_components = _component_from_mask(wall_mask)
    assert len(wall_components) == 1, "grey walls must be one connected structure"
    assert len(wall_components[0][0]) == counts[wall], "wall color must be fully connected"
    return int(wall)


def _room_boxes(grid: np.ndarray, wall_color: int) -> list[tuple[int, int, int, int, int, int]]:
    rooms: list[tuple[int, int, int, int, int, int]] = []
    for coords, touches_border in _component_from_mask(grid != wall_color):
        if touches_border:
            continue

        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        rect = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            rect[r - r0, c - c0] = True
        assert np.all(rect), "each room interior must be a rectangle"

        rooms.append((r0, r1, c0, c1, len(coords), len(rect.reshape(-1))))
    return rooms


def _pattern_for_touch_count(touch_count: int) -> np.ndarray:
    if touch_count <= 1:
        return SOLID
    if touch_count == 2:
        return STRIPES
    return CHECKER


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room with a tiled pattern chosen by the seed's wall contacts.

    Input structure:
    - Black background surrounds one connected grey wall structure and a bottom legend.
    - The grey walls partition the upper area into enclosed rectangular rooms.
    - Each room contains exactly one non-grey seed pixel on an otherwise black interior.
    - The grid bottom shows three 3x3 maroon template blocks: solid, vertical stripes,
      and checkerboard, separated by one black column.

    Rule:
    - Find each enclosed room and its unique seed pixel.
    - Count how many of the seed's four orthogonal neighbors are grey wall cells.
    - Use the solid template for counts 0 or 1, the striped template for count 2,
      and the checkerboard template for counts 3 or 4.
    - Recolor the chosen template with the seed color and tile it across the whole
      room interior with the template's top-left aligned to the room's top-left cell.
    - Keep the grey walls and the bottom legend unchanged.

    Key insight:
    - The filled motif depends on the seed's local wall adjacency, not on the room's
      size or position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _border_mode(g)
    _legend_r, _legend_c, legend_color = _find_legend(g, bg)
    wall_color = _pick_wall_color(g, bg, legend_color)

    out = g.copy()
    for r0, r1, c0, c1, _area, _bbox_area in _room_boxes(g, wall_color):
        room = g[r0 : r1 + 1, c0 : c1 + 1]
        room_colors = {int(v) for v in room.reshape(-1)} - {bg}
        assert len(room_colors) == 1, "each room must contain exactly one seed color"
        seed_color = room_colors.pop()

        seed_positions = np.argwhere(room == seed_color)
        assert len(seed_positions) == 1, "each room must contain one seed pixel"
        sr = int(seed_positions[0, 0]) + r0
        sc = int(seed_positions[0, 1]) + c0

        touch_count = sum(
            0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == wall_color
            for nr, nc in _neighbors4(sr, sc)
        )
        pattern = _pattern_for_touch_count(int(touch_count))

        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r, c] = seed_color if pattern[(r - r0) % 3, (c - c0) % 3] else bg

    return out


def _wall_boundaries(sizes: list[int], start: int) -> list[int]:
    bounds = [start]
    cursor = start
    for size in sizes:
        cursor += size + 1
        bounds.append(cursor)
    return bounds


def _seed_options(box: tuple[int, int, int, int], touch_count: int) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = box
    if touch_count == 0:
        return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
    if touch_count == 1:
        options = [(r0, c) for c in range(c0 + 1, c1)]
        options += [(r1, c) for c in range(c0 + 1, c1)]
        options += [(r, c0) for r in range(r0 + 1, r1)]
        options += [(r, c1) for r in range(r0 + 1, r1)]
        return options
    return [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]


def _draw_structure(
    grid: np.ndarray,
    top: int,
    left: int,
    room_heights: list[int],
    room_widths: list[int],
) -> list[tuple[int, int, int, int]]:
    row_walls = _wall_boundaries(room_heights, top)
    col_walls = _wall_boundaries(room_widths, left)
    bottom = row_walls[-1]
    right = col_walls[-1]

    for r in row_walls:
        grid[r, left : right + 1] = WALL
    for c in col_walls:
        grid[top : bottom + 1, c] = WALL

    rooms: list[tuple[int, int, int, int]] = []
    for i, room_h in enumerate(room_heights):
        r0 = row_walls[i] + 1
        r1 = r0 + room_h - 1
        for j, room_w in enumerate(room_widths):
            c0 = col_walls[j] + 1
            c1 = c0 + room_w - 1
            rooms.append((r0, r1, c0, c1))
    return rooms


def _place_legend(grid: np.ndarray, left: int) -> None:
    r0 = grid.shape[0] - 3
    grid[r0 : r0 + 3, left : left + 3] = LEGEND
    grid[r0 : r0 + 3, left + 4 : left + 7] = _pattern_block(STRIPES, LEGEND, BG)
    grid[r0 : r0 + 3, left + 8 : left + 11] = _pattern_block(CHECKER, LEGEND, BG)


def _gen_attempt() -> np.ndarray | None:
    room_rows, room_cols = random.choice(LAYOUTS)
    room_heights = [random.randint(*ROOM_H_RANGE) for _ in range(room_rows)]
    room_widths = [random.randint(*ROOM_W_RANGE) for _ in range(room_cols)]

    struct_h = sum(room_heights) + room_rows + 1
    struct_w = sum(room_widths) + room_cols + 1

    top_margin = random.randint(*TOP_MARGIN_RANGE)
    gap = random.randint(*GAP_RANGE)
    total_h = top_margin + struct_h + gap + 3
    min_w = max(11, struct_w + 2)
    if total_h > 30 or min_w > 30:
        return None

    total_w = random.randint(min_w, 30)
    wall_left = random.randint(1, total_w - struct_w - 1)
    legend_left = random.randint(0, total_w - 11)

    grid = np.full((total_h, total_w), BG, dtype=int)
    rooms = _draw_structure(grid, top_margin, wall_left, room_heights, room_widths)
    _place_legend(grid, legend_left)

    num_rooms = len(rooms)
    assert 3 <= num_rooms <= len(SEED_COLORS)
    colors = random.sample(SEED_COLORS, num_rooms)

    touches = [0, 1, 2] * (num_rooms // 3)
    if num_rooms % 3:
        touches += random.sample([0, 1, 2], num_rooms % 3)
    random.shuffle(touches)

    room_order = list(range(num_rooms))
    random.shuffle(room_order)
    for idx, color in zip(room_order, colors):
        box = rooms[idx]
        options = _seed_options(box, touches[idx])
        if not options:
            return None
        r, c = random.choice(options)
        grid[r, c] = color

    try:
        main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
