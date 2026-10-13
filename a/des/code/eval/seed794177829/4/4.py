import random
from collections import Counter, deque

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = (2, 1, 3, 4, 7, 9)
OUTER_MARGIN = 2
ROOM_COUNT_RANGE = (2, 4)
ROOM_SIZE_RANGE = (2, 4)
GEN_RETRIES = 400
GRAPH_RETRIES = 200

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SCENARIOS = (
    "full_cover_connected",
    "full_cover_disconnected",
    "partial_cover_connected",
    "partial_cover_disconnected",
)


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _corner_background(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _mask_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(cells)
    return components


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    options = [(int(color), int(count)) for color, count in zip(colors, counts) if int(color) != bg]
    assert options, "expected a wall color"
    return max(options, key=lambda item: item[1])[0]


def _detect_doors(grid: np.ndarray, bg: int, wall: int) -> dict[tuple[int, int], str]:
    h, w = grid.shape
    doors: dict[tuple[int, int], str] = {}
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != bg:
                continue
            vertical = (
                int(grid[r - 1, c]) == wall
                and int(grid[r + 1, c]) == wall
                and int(grid[r, c - 1]) == bg
                and int(grid[r, c + 1]) == bg
            )
            horizontal = (
                int(grid[r, c - 1]) == wall
                and int(grid[r, c + 1]) == wall
                and int(grid[r - 1, c]) == bg
                and int(grid[r + 1, c]) == bg
            )
            if vertical ^ horizontal:
                doors[(r, c)] = "v" if vertical else "h"
    return doors


def _label_rooms(closed_grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[list[tuple[int, int]]]]:
    h, w = closed_grid.shape
    room_ids = np.full((h, w), -1, dtype=int)
    room_cells: list[list[tuple[int, int]]] = []
    for cells in _mask_components(closed_grid == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in cells):
            continue
        room_id = len(room_cells)
        room_cells.append(cells)
        for r, c in cells:
            room_ids[r, c] = room_id
    assert room_cells, "expected at least one room"
    return room_ids, room_cells


def _find_markers(
    grid: np.ndarray,
    bg: int,
    wall: int,
    room_ids: np.ndarray,
) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    markers: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (bg, wall):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            assert len(cells) == 2, "marker must have area 2"
            assert (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1) in {(1, 2), (2, 1)}
            assert any(rr in (0, h - 1) or cc in (0, w - 1) for rr, cc in cells), "marker must touch edge"

            start_rooms = set()
            for cr, cc in cells:
                for dr, dc in DIR4:
                    wr, wc = cr + dr, cc + dc
                    br, bc = cr + 2 * dr, cc + 2 * dc
                    if not (0 <= wr < h and 0 <= wc < w and 0 <= br < h and 0 <= bc < w):
                        continue
                    if int(grid[wr, wc]) != wall:
                        continue
                    room_id = int(room_ids[br, bc])
                    if room_id >= 0:
                        start_rooms.add(room_id)
            assert len(start_rooms) == 1, "marker must identify exactly one start room"
            markers.append(
                {
                    "color": color,
                    "room": next(iter(start_rooms)),
                    "anchor": min(cells),
                }
            )

    markers.sort(key=lambda marker: marker["anchor"])
    assert markers, "expected at least one marker"
    return markers


def _analyze(grid: np.ndarray):
    bg = _corner_background(grid)
    wall = _detect_wall_color(grid, bg)
    doors = _detect_doors(grid, bg, wall)

    closed = np.array(grid, copy=True)
    for r, c in doors:
        closed[r, c] = wall

    room_ids, room_cells = _label_rooms(closed, bg)
    graph = {room_id: set() for room_id in range(len(room_cells))}
    for (r, c), orientation in doors.items():
        if orientation == "v":
            a = int(room_ids[r, c - 1])
            b = int(room_ids[r, c + 1])
        else:
            a = int(room_ids[r - 1, c])
            b = int(room_ids[r + 1, c])
        if a >= 0 and b >= 0 and a != b:
            graph[a].add(b)
            graph[b].add(a)

    markers = _find_markers(grid, bg, wall, room_ids)
    return room_cells, graph, markers


def _reachable_rooms(graph: dict[int, set[int]], start: int) -> set[int]:
    seen = {start}
    queue = deque([start])
    while queue:
        room = queue.popleft()
        for nxt in graph[room]:
            if nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color each marker-labeled room-domain by following room-to-room door connectivity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    room_cells, graph, markers = _analyze(g)
    room_colors: dict[int, int] = {}
    for marker in markers:
        color = int(marker["color"])
        for room_id in _reachable_rooms(graph, int(marker["room"])):
            prior = room_colors.get(room_id)
            assert prior in (None, color), "different markers must not target the same room-domain"
            room_colors[room_id] = color

    out = np.array(g, copy=True)
    for room_id, color in room_colors.items():
        for r, c in room_cells[room_id]:
            out[r, c] = color
    return out


def _room_id(i: int, j: int, cols: int) -> int:
    return i * cols + j


def _sample_room_grid(scenario: str, rng: random.Random) -> tuple[int, int]:
    if scenario == "full_cover_disconnected":
        return rng.choice([(2, 2), (2, 3), (3, 2)])
    if scenario == "full_cover_connected":
        return rng.choice([(2, 2), (2, 3), (3, 2), (3, 3)])
    return rng.randint(*ROOM_COUNT_RANGE), rng.randint(*ROOM_COUNT_RANGE)


def _boundary_positions(sizes: list[int]) -> list[int]:
    positions = [OUTER_MARGIN]
    current = OUTER_MARGIN
    for size in sizes:
        current += size + 1
        positions.append(current)
    return positions


def _room_layout(rows: int, cols: int, rng: random.Random):
    row_sizes = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(rows)]
    col_sizes = [rng.randint(*ROOM_SIZE_RANGE) for _ in range(cols)]
    wall_rows = _boundary_positions(row_sizes)
    wall_cols = _boundary_positions(col_sizes)
    height = wall_rows[-1] + OUTER_MARGIN + 1
    width = wall_cols[-1] + OUTER_MARGIN + 1

    boxes: list[tuple[int, int, int, int]] = []
    coords: list[tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            boxes.append((wall_rows[i] + 1, wall_rows[i + 1] - 1, wall_cols[j] + 1, wall_cols[j + 1] - 1))
            coords.append((i, j))
    return row_sizes, col_sizes, wall_rows, wall_cols, height, width, boxes, coords


def _all_edges(rows: int, cols: int) -> list[dict[str, int | str]]:
    edges: list[dict[str, int | str]] = []
    for i in range(rows):
        for j in range(cols - 1):
            edges.append({"a": _room_id(i, j, cols), "b": _room_id(i, j + 1, cols), "kind": "v", "i": i, "j": j})
    for i in range(rows - 1):
        for j in range(cols):
            edges.append({"a": _room_id(i, j, cols), "b": _room_id(i + 1, j, cols), "kind": "h", "i": i, "j": j})
    return edges


def _graph_components(n_rooms: int, open_edges: set[int], edges: list[dict[str, int | str]]):
    graph = {room: set() for room in range(n_rooms)}
    for edge_index in open_edges:
        edge = edges[edge_index]
        a = int(edge["a"])
        b = int(edge["b"])
        graph[a].add(b)
        graph[b].add(a)

    components: list[list[int]] = []
    room_to_component = [-1] * n_rooms
    for room in range(n_rooms):
        if room_to_component[room] != -1:
            continue
        queue = deque([room])
        room_to_component[room] = len(components)
        cells: list[int] = []
        while queue:
            cur = queue.popleft()
            cells.append(cur)
            for nxt in graph[cur]:
                if room_to_component[nxt] == -1:
                    room_to_component[nxt] = room_to_component[room]
                    queue.append(nxt)
        components.append(cells)
    return graph, components, room_to_component


def _perimeter_rooms(rows: int, cols: int) -> set[int]:
    perimeter = set()
    for i in range(rows):
        for j in range(cols):
            if i in (0, rows - 1) or j in (0, cols - 1):
                perimeter.add(_room_id(i, j, cols))
    return perimeter


def _sample_open_edges(
    scenario: str,
    rows: int,
    cols: int,
    perimeter: set[int],
    rng: random.Random,
):
    n_rooms = rows * cols
    edges = _all_edges(rows, cols)

    if scenario.endswith("disconnected"):
        open_edges: set[int] = set()
        graph, components, room_to_component = _graph_components(n_rooms, open_edges, edges)
        if scenario == "full_cover_disconnected":
            if any(len(component) != 1 for component in components):
                return None
            if any(not (set(component) & perimeter) for component in components):
                return None
            if len(components) > len(MARKER_COLORS):
                return None
        return edges, open_edges, graph, components, room_to_component

    probabilities = [0.4, 0.55, 0.7] if scenario == "full_cover_connected" else [0.25, 0.45, 0.65]
    for _ in range(GRAPH_RETRIES):
        p_open = rng.choice(probabilities)
        open_edges = {idx for idx in range(len(edges)) if rng.random() < p_open}
        if not open_edges:
            continue
        graph, components, room_to_component = _graph_components(n_rooms, open_edges, edges)
        if not any(len(component) > 1 for component in components):
            continue

        if scenario == "full_cover_connected":
            if len(components) > len(MARKER_COLORS):
                continue
            if any(not (set(component) & perimeter) for component in components):
                continue
        else:
            if len(components) < 2:
                continue
        return edges, open_edges, graph, components, room_to_component
    return None


def _select_labeled_components(
    scenario: str,
    components: list[list[int]],
    perimeter: set[int],
    rng: random.Random,
) -> list[int] | None:
    eligible = [idx for idx, component in enumerate(components) if set(component) & perimeter]
    if not eligible:
        return None

    if scenario.startswith("full_cover"):
        if len(eligible) != len(components) or len(components) > len(MARKER_COLORS):
            return None
        return eligible

    interior_only = [idx for idx in range(len(components)) if idx not in eligible]
    max_labels = min(len(eligible), len(MARKER_COLORS))
    if max_labels == 0:
        return None

    if interior_only:
        max_pick = max_labels
    else:
        if len(eligible) < 2:
            return None
        max_pick = min(max_labels, len(eligible) - 1)
    if max_pick <= 0:
        return None

    label_count = rng.randint(1, max_pick)
    return rng.sample(eligible, label_count)


def _draw_walls(grid: np.ndarray, wall_rows: list[int], wall_cols: list[int]) -> None:
    top = wall_rows[0]
    bottom = wall_rows[-1]
    left = wall_cols[0]
    right = wall_cols[-1]
    for row in wall_rows:
        grid[row, left : right + 1] = GREY
    for col in wall_cols:
        grid[top : bottom + 1, col] = GREY


def _draw_doors(
    grid: np.ndarray,
    edges: list[dict[str, int | str]],
    open_edges: set[int],
    wall_rows: list[int],
    wall_cols: list[int],
    rng: random.Random,
) -> None:
    for edge_index in open_edges:
        edge = edges[edge_index]
        kind = str(edge["kind"])
        i = int(edge["i"])
        j = int(edge["j"])
        if kind == "v":
            door_r = rng.randint(wall_rows[i] + 1, wall_rows[i + 1] - 1)
            door_c = wall_cols[j + 1]
        else:
            door_r = wall_rows[i + 1]
            door_c = rng.randint(wall_cols[j] + 1, wall_cols[j + 1] - 1)
        grid[door_r, door_c] = BLACK


def _room_sides(room_coord: tuple[int, int], rows: int, cols: int) -> list[str]:
    i, j = room_coord
    sides = []
    if i == 0:
        sides.append("top")
    if i == rows - 1:
        sides.append("bottom")
    if j == 0:
        sides.append("left")
    if j == cols - 1:
        sides.append("right")
    return sides


def _marker_cells(
    side: str,
    room_box: tuple[int, int, int, int],
    height: int,
    width: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = room_box
    if side == "top":
        col = rng.randint(c0, c1)
        return [(0, col), (1, col)]
    if side == "bottom":
        col = rng.randint(c0, c1)
        return [(height - 2, col), (height - 1, col)]
    if side == "left":
        row = rng.randint(r0, r1)
        return [(row, 0), (row, 1)]
    row = rng.randint(r0, r1)
    return [(row, width - 2), (row, width - 1)]


def _place_markers(
    grid: np.ndarray,
    rows: int,
    cols: int,
    room_boxes: list[tuple[int, int, int, int]],
    room_coords: list[tuple[int, int]],
    components: list[list[int]],
    labeled_components: list[int],
    perimeter: set[int],
    rng: random.Random,
) -> dict[int, int]:
    colors = rng.sample(MARKER_COLORS, len(labeled_components))
    component_colors: dict[int, int] = {}
    for component_index, color in zip(labeled_components, colors):
        rooms = components[component_index]
        start_room = rng.choice([room for room in rooms if room in perimeter])
        side = rng.choice(_room_sides(room_coords[start_room], rows, cols))
        cells = _marker_cells(side, room_boxes[start_room], *grid.shape, rng)
        assert all(int(grid[r, c]) == BLACK for r, c in cells), "marker cells must be empty"
        for r, c in cells:
            grid[r, c] = color
        component_colors[component_index] = color
    return component_colors


def _oracle_output(
    grid: np.ndarray,
    room_boxes: list[tuple[int, int, int, int]],
    room_to_component: list[int],
    component_colors: dict[int, int],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for room_id, component_index in enumerate(room_to_component):
        color = component_colors.get(component_index)
        if color is None:
            continue
        r0, r1, c0, c1 = room_boxes[room_id]
        out[r0 : r1 + 1, c0 : c1 + 1] = color
    return out


def _gen_attempt(rng: random.Random):
    scenario = rng.choice(SCENARIOS)
    rows, cols = _sample_room_grid(scenario, rng)
    layout = _room_layout(rows, cols, rng)
    _, _, wall_rows, wall_cols, height, width, room_boxes, room_coords = layout

    perimeter = _perimeter_rooms(rows, cols)
    graph_info = _sample_open_edges(scenario, rows, cols, perimeter, rng)
    if graph_info is None:
        return None
    edges, open_edges, _, components, room_to_component = graph_info

    labeled_components = _select_labeled_components(scenario, components, perimeter, rng)
    if not labeled_components:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    _draw_walls(grid, wall_rows, wall_cols)
    _draw_doors(grid, edges, open_edges, wall_rows, wall_cols, rng)
    component_colors = _place_markers(
        grid,
        rows,
        cols,
        room_boxes,
        room_coords,
        components,
        labeled_components,
        perimeter,
        rng,
    )
    expected = _oracle_output(grid, room_boxes, room_to_component, component_colors)

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """
    Generate a wall-and-door floorplan with edge markers in balanced scenario families.
    """
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
