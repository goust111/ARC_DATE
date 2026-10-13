import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
SEED_COLORS = (RED, BLUE)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_gap_candidates(grid: np.ndarray) -> list[tuple[int, int, str]]:
    h, w = map(int, grid.shape)
    gaps: list[tuple[int, int, str]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != BLACK:
                continue
            vertical = (
                int(grid[r - 1, c]) == GREY
                and int(grid[r + 1, c]) == GREY
                and int(grid[r, c - 1]) != GREY
                and int(grid[r, c + 1]) != GREY
            )
            horizontal = (
                int(grid[r, c - 1]) == GREY
                and int(grid[r, c + 1]) == GREY
                and int(grid[r - 1, c]) != GREY
                and int(grid[r + 1, c]) != GREY
            )
            if vertical ^ horizontal:
                gaps.append((r, c, "vertical" if vertical else "horizontal"))
    return gaps


def _label_components(passable: np.ndarray) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = map(int, passable.shape)
    labels = -np.ones((h, w), dtype=int)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(passable[r, c]) or int(labels[r, c]) != -1:
                continue
            label = len(components)
            labels[r, c] = label
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(passable[nr, nc]) and int(labels[nr, nc]) == -1:
                        labels[nr, nc] = label
                        queue.append((nr, nc))
            components.append(cells)
    return labels, components


def _room_seed_colors(grid: np.ndarray, rooms: list[list[tuple[int, int]]]) -> dict[int, int]:
    colors: dict[int, int] = {}
    for label, cells in enumerate(rooms):
        seen = {
            int(grid[r, c])
            for r, c in cells
            if int(grid[r, c]) not in (BLACK, GREY)
        }
        assert len(seen) == 1, "each room must contain exactly one seed color"
        color = seen.pop()
        assert color in SEED_COLORS, "room seeds must be red or blue"
        colors[label] = color
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """Decide every wall gap from the seed colors of the two rooms it separates.

    Input structure:
    - The background is black.
    - Grey one-pixel orthogonal walls partition the grid into rectangular rooms.
    - Some wall cells are missing as one-pixel gaps between adjacent rooms.
    - Each room contains exactly one colored seed pixel, either red or blue.

    Rule:
    - Temporarily treat every one-pixel wall gap as closed so the open area splits back into the original rooms.
    - Read the unique red/blue seed color inside each recovered room.
    - For each gap, identify the two rooms on opposite sides of that gap.
    - Fill the gap with grey when the two seed colors match; otherwise fill it with green. Leave all other pixels unchanged.

    Key insight:
    - The missing wall cells are resolved by room identity rather than by position: reconstruct the two rooms flanking each gap, then apply a same-color versus different-color test.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    out = g.copy()
    gaps = _find_gap_candidates(g)

    wall_mask = g == GREY
    for r, c, _axis in gaps:
        wall_mask[r, c] = True

    room_labels, rooms = _label_components(~wall_mask)
    room_colors = _room_seed_colors(g, rooms)

    for r, c, axis in gaps:
        if axis == "vertical":
            a = int(room_labels[r, c - 1])
            b = int(room_labels[r, c + 1])
        else:
            a = int(room_labels[r - 1, c])
            b = int(room_labels[r + 1, c])
        assert a >= 0 and b >= 0 and a != b, "gap must separate two distinct rooms"
        out[r, c] = GREY if room_colors[a] == room_colors[b] else GREEN

    return out


GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
MIN_ROOM_SPAN = 3
TARGET_ROOM_COUNT = (4, 8)
MAX_GAPS_PER_TYPE = 4
GENERATION_ATTEMPTS = 500


def _split_orientations(room: tuple[int, int, int, int]) -> list[str]:
    r0, r1, c0, c1 = room
    options: list[str] = []
    if r1 - r0 + 1 >= 2 * MIN_ROOM_SPAN + 1:
        options.append("H")
    if c1 - c0 + 1 >= 2 * MIN_ROOM_SPAN + 1:
        options.append("V")
    return options


def _partition_rooms(height: int, width: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, :] = GREY
    grid[-1, :] = GREY
    grid[:, 0] = GREY
    grid[:, -1] = GREY

    rooms = [(1, height - 2, 1, width - 2)]
    target = random.randint(*TARGET_ROOM_COUNT)

    while len(rooms) < target:
        splittable = [(idx, room, _split_orientations(room)) for idx, room in enumerate(rooms)]
        splittable = [item for item in splittable if item[2]]
        if not splittable:
            break

        idx, room, options = random.choice(splittable)
        r0, r1, c0, c1 = room
        orientation = random.choice(options)
        rooms.pop(idx)

        if orientation == "H":
            wall_r = random.randint(r0 + MIN_ROOM_SPAN, r1 - MIN_ROOM_SPAN)
            grid[wall_r, c0 : c1 + 1] = GREY
            rooms.extend([(r0, wall_r - 1, c0, c1), (wall_r + 1, r1, c0, c1)])
        else:
            wall_c = random.randint(c0 + MIN_ROOM_SPAN, c1 - MIN_ROOM_SPAN)
            grid[r0 : r1 + 1, wall_c] = GREY
            rooms.extend([(r0, r1, c0, wall_c - 1), (r0, r1, wall_c + 1, c1)])

    return grid, rooms


def _room_id_map(shape: tuple[int, int], rooms: list[tuple[int, int, int, int]]) -> np.ndarray:
    room_id = -np.ones(shape, dtype=int)
    for idx, (r0, r1, c0, c1) in enumerate(rooms):
        room_id[r0 : r1 + 1, c0 : c1 + 1] = idx
    return room_id


def _edge_candidates(grid: np.ndarray, room_id: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    h, w = map(int, grid.shape)
    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != GREY:
                continue
            vertical = (
                int(grid[r - 1, c]) == GREY
                and int(grid[r + 1, c]) == GREY
                and int(room_id[r, c - 1]) >= 0
                and int(room_id[r, c + 1]) >= 0
            )
            horizontal = (
                int(grid[r, c - 1]) == GREY
                and int(grid[r, c + 1]) == GREY
                and int(room_id[r - 1, c]) >= 0
                and int(room_id[r + 1, c]) >= 0
            )
            if not (vertical ^ horizontal):
                continue
            if vertical:
                a = int(room_id[r, c - 1])
                b = int(room_id[r, c + 1])
            else:
                a = int(room_id[r - 1, c])
                b = int(room_id[r + 1, c])
            if a == b:
                continue
            key = (a, b) if a < b else (b, a)
            edges.setdefault(key, []).append((r, c))
    return edges


def _sample_room_colors(room_count: int, edges: list[tuple[int, int]]) -> tuple[list[int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    for _ in range(200):
        colors = [random.choice(SEED_COLORS) for _ in range(room_count)]
        same = [edge for edge in edges if colors[edge[0]] == colors[edge[1]]]
        diff = [edge for edge in edges if colors[edge[0]] != colors[edge[1]]]
        if same and diff:
            return colors, same, diff
    return None


def _validate_generated(grid: np.ndarray) -> None:
    gaps = _find_gap_candidates(grid)
    assert gaps and len(gaps) % 2 == 0
    gap_cells = {(r, c) for r, c, _axis in gaps}
    out = main(grid)
    changed = set(map(tuple, np.argwhere(out != grid)))
    assert changed == gap_cells
    for r, c in changed:
        assert int(out[r, c]) in (GREY, GREEN)


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid, rooms = _partition_rooms(height, width)
    if len(rooms) < TARGET_ROOM_COUNT[0]:
        return None

    room_id = _room_id_map(grid.shape, rooms)
    edge_map = _edge_candidates(grid, room_id)
    if len(edge_map) < 2:
        return None

    sampled = _sample_room_colors(len(rooms), list(edge_map))
    if sampled is None:
        return None
    room_colors, same_edges, diff_edges = sampled

    per_type = random.randint(1, min(len(same_edges), len(diff_edges), MAX_GAPS_PER_TYPE))
    selected_edges = random.sample(same_edges, per_type) + random.sample(diff_edges, per_type)
    for edge in selected_edges:
        gap_r, gap_c = random.choice(edge_map[edge])
        grid[gap_r, gap_c] = BLACK

    for idx, (r0, r1, c0, c1) in enumerate(rooms):
        seed_r = random.randint(r0, r1)
        seed_c = random.randint(c0, c1)
        grid[seed_r, seed_c] = room_colors[idx]

    try:
        _validate_generated(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-gap puzzle")
