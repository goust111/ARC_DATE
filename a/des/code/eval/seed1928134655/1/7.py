import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))
MIN_GRID_SIZE = 16
MAX_GRID_SIZE = 24
MIN_RECT_SIZE = 5
MIN_GAP = 2
MAX_GAP = 4
MAX_GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray):
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[np.ndarray] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != -1:
                continue
            stack = [(r, c)]
            labels[r, c] = next_id
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == -1:
                        labels[nr, nc] = next_id
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
            next_id += 1
    return comps, labels


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == h - 1)
        or np.any(cols == 0)
        or np.any(cols == w - 1)
    )


def _draw_rectangle(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _sample_child_bbox(parent: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    pr0, pc0, pr1, pc1 = parent
    parent_h = pr1 - pr0 + 1
    parent_w = pc1 - pc0 + 1
    for _ in range(80):
        gap_top = random.randint(MIN_GAP, MAX_GAP)
        gap_bottom = random.randint(MIN_GAP, MAX_GAP)
        gap_left = random.randint(MIN_GAP, MAX_GAP)
        gap_right = random.randint(MIN_GAP, MAX_GAP)
        child_h = parent_h - gap_top - gap_bottom
        child_w = parent_w - gap_left - gap_right
        if child_h < MIN_RECT_SIZE or child_w < MIN_RECT_SIZE:
            continue
        return (
            pr0 + gap_top,
            pc0 + gap_left,
            pr1 - gap_bottom,
            pc1 - gap_right,
        )
    return None


def _draw_bridge(
    grid: np.ndarray,
    parent: tuple[int, int, int, int],
    child: tuple[int, int, int, int],
    color: int,
) -> str:
    pr0, pc0, pr1, pc1 = parent
    cr0, cc0, cr1, cc1 = child
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides:
        if side in {"top", "bottom"} and cc1 - cc0 + 1 >= 3:
            c = random.randint(cc0 + 1, cc1 - 1)
            if side == "top":
                grid[pr0 : cr0 + 1, c] = color
            else:
                grid[cr1 : pr1 + 1, c] = color
            return side
        if side in {"left", "right"} and cr1 - cr0 + 1 >= 3:
            r = random.randint(cr0 + 1, cr1 - 1)
            if side == "left":
                grid[r, pc0 : cc0 + 1] = color
            else:
                grid[r, cc1 : pc1 + 1] = color
            return side
    raise RuntimeError("unable to draw bridge")


def _pick_marker_cell(labels: np.ndarray, room_id: int, coords: np.ndarray) -> tuple[int, int] | None:
    h, w = labels.shape
    primary = []
    secondary = []
    fallback = []
    for r, c in coords:
        same_neighbors = 0
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] == room_id:
                same_neighbors += 1
        on_edge = r == 0 or r == h - 1 or c == 0 or c == w - 1
        if not on_edge and same_neighbors == 4:
            primary.append((int(r), int(c)))
        elif not on_edge and same_neighbors >= 3:
            secondary.append((int(r), int(c)))
        elif same_neighbors >= 2:
            fallback.append((int(r), int(c)))
    pool = primary or secondary or fallback
    if not pool:
        return None
    return random.choice(pool)


def _room_info(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)
    counts = Counter(map(int, g.ravel()))
    non_bg = [color for color in counts if color != bg]
    assert non_bg, "expected walls and markers"
    wall_color = max(non_bg, key=lambda color: counts[color])
    bg_comps, labels = _components(g == bg)
    marker_mask = (g != bg) & (g != wall_color)
    rooms = []
    for room_id, coords in enumerate(bg_comps):
        marker_colors = set()
        marker_cells = set()
        for r, c in coords:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and marker_mask[nr, nc]:
                    marker_colors.add(int(g[nr, nc]))
                    marker_cells.add((int(nr), int(nc)))
        assert len(marker_colors) == 1, "each room must have exactly one adjacent marker color"
        assert len(marker_cells) == 1, "each room must contain exactly one marker pixel"
        rooms.append(
            {
                "external": _touches_edge(coords, h, w),
                "marker_color": next(iter(marker_colors)),
            }
        )
    return bg, wall_color, rooms, labels


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color the wall side that belongs to an edge-touching room with that room's marker color.

    Rooms are the 4-connected background components. A room is external if it
    touches the grid boundary. For each gray wall pixel, inspect the rooms in
    its 4-neighborhood; when exactly one adjacent room is external, repaint the
    wall pixel with that room's marker color. Background and marker pixels stay
    unchanged, and walls bordering only internal rooms remain gray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    _, wall_color, rooms, labels = _room_info(g)
    h, w = g.shape

    out = g.copy()
    for r, c in np.argwhere(g == wall_color):
        adjacent_rooms = set()
        for nr, nc in _neighbors4(int(r), int(c)):
            if 0 <= nr < h and 0 <= nc < w and labels[nr, nc] != -1:
                adjacent_rooms.add(int(labels[nr, nc]))
        external_neighbors = [room_id for room_id in adjacent_rooms if rooms[room_id]["external"]]
        if len(external_neighbors) == 1:
            out[r, c] = int(rooms[external_neighbors[0]]["marker_color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    w = random.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
    bg, wall = random.sample(PALETTE, 2)
    grid = np.full((h, w), bg, dtype=int)

    depth = random.choice([1, 2])
    min_outer = 10 if depth == 1 else 13
    if h - 4 < min_outer or w - 4 < min_outer:
        return None

    outer_h = random.randint(min_outer, h - 4)
    outer_w = random.randint(min_outer, w - 4)
    outer_r0 = random.randint(2, h - outer_h - 2)
    outer_c0 = random.randint(2, w - outer_w - 2)
    outer = (outer_r0, outer_c0, outer_r0 + outer_h - 1, outer_c0 + outer_w - 1)
    _draw_rectangle(grid, outer, wall)

    parent = outer
    for _ in range(depth):
        child = _sample_child_bbox(parent)
        if child is None:
            return None
        _draw_rectangle(grid, child, wall)
        _draw_bridge(grid, parent, child, wall)
        parent = child

    bg_comps, labels = _components(grid == bg)
    expected_rooms = depth + 2
    if len(bg_comps) != expected_rooms:
        return None
    if sum(_touches_edge(comp, h, w) for comp in bg_comps) != 1:
        return None

    bg_count = int(np.count_nonzero(grid == bg))
    wall_count = int(np.count_nonzero(grid == wall))
    if bg_count <= wall_count:
        return None

    available_marker_colors = [color for color in PALETTE if color not in (bg, wall)]
    if len(available_marker_colors) < len(bg_comps):
        return None
    random.shuffle(available_marker_colors)

    marker_positions = []
    for room_id, coords in enumerate(bg_comps):
        pos = _pick_marker_cell(labels, room_id, coords)
        if pos is None:
            return None
        marker_positions.append(pos)

    for room_id, pos in enumerate(marker_positions):
        grid[pos] = available_marker_colors[room_id]

    if len(_components(grid == bg)[0]) != expected_rooms:
        return None

    try:
        transformed = main(grid)
    except AssertionError:
        return None

    if transformed.shape != grid.shape:
        return None
    if not np.any(transformed != grid):
        return None

    non_wall_mask = grid != wall
    if not np.array_equal(transformed[non_wall_mask], grid[non_wall_mask]):
        return None

    if int(np.count_nonzero(transformed == wall)) >= wall_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
