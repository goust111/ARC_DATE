import math
import random

import numpy as np


BG = 0
WALL = 5
GATE = 7
ROOM_COLORS = (1, 2, 3, 4, 6, 8, 9)

# Generator configuration.
LAYOUT_CHOICES = ((2, 3), (3, 2), (2, 2))
LAYOUT_WEIGHTS = (0.4, 0.4, 0.2)
ROOM_SIZE_RANGE = (4, 7)
MIN_FRAGMENT_AREA = 4
GENERATION_ATTEMPTS = 300
SHAPE_ATTEMPTS = 80


def _compact_dims(area: int) -> tuple[int, int]:
    """Pick the factor pair closest to square, then orient it wide."""
    best_h = 1
    for h in range(1, int(math.isqrt(int(area))) + 1):
        if int(area) % h == 0:
            best_h = h
    return best_h, int(area) // best_h


def _wall_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    wall_rows = [
        r for r in range(g.shape[0]) if bool(np.all(np.isin(g[r, :], (WALL, GATE))))
    ]
    wall_cols = [
        c for c in range(g.shape[1]) if bool(np.all(np.isin(g[:, c], (WALL, GATE))))
    ]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected room-separating walls"
    return wall_rows, wall_cols


def _spans_from_lines(lines: list[int], length: int) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(set(int(x) for x in lines)) + [int(length)]
    spans: list[tuple[int, int]] = []
    for a, b in zip(bounds, bounds[1:]):
        start = int(a) + 1
        end = int(b) - 1
        if start <= end:
            spans.append((start, end))
    return spans


def _extract_rooms(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    wall_rows, wall_cols = _wall_lines(g)
    row_spans = _spans_from_lines(wall_rows, g.shape[0])
    col_spans = _spans_from_lines(wall_cols, g.shape[1])
    assert row_spans and col_spans, "expected at least one room"

    room_ids = np.full(g.shape, -1, dtype=int)
    rooms: list[dict] = []

    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            room_id = len(rooms)
            room = g[r0 : r1 + 1, c0 : c1 + 1]
            assert not np.any(room == GATE), "gates must stay on walls"

            colors = [int(v) for v in np.unique(room) if int(v) not in (BG, WALL, GATE)]
            assert len(colors) == 1, "each room must contain exactly one fragment color"
            color = colors[0]
            area = int(np.sum(room == color))
            assert area > 0, "fragment must be non-empty"

            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            rect_h, rect_w = _compact_dims(area)
            assert rect_h <= height and rect_w <= width, "repacked rectangle must fit its room"

            rooms.append(
                {
                    "id": room_id,
                    "row_band": ri,
                    "col_band": ci,
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "height": height,
                    "width": width,
                    "color": color,
                    "area": area,
                    "rect_h": rect_h,
                    "rect_w": rect_w,
                }
            )
            room_ids[r0 : r1 + 1, c0 : c1 + 1] = room_id

    return rooms, room_ids


def _gate_edges(grid: np.ndarray, room_ids: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    edges: set[tuple[int, int]] = set()
    for r, c in np.argwhere(g == GATE):
        adjacent: set[int] = set()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr = int(r) + dr
            cc = int(c) + dc
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]:
                room_id = int(room_ids[rr, cc])
                if room_id >= 0:
                    adjacent.add(room_id)
        if len(adjacent) == 2:
            a, b = sorted(adjacent)
            edges.add((a, b))
    return sorted(edges)


def _adjacency(room_count: int, edges: list[tuple[int, int]]) -> dict[int, set[int]]:
    graph: dict[int, set[int]] = {room_id: set() for room_id in range(int(room_count))}
    for a, b in edges:
        graph[int(a)].add(int(b))
        graph[int(b)].add(int(a))
    return graph


def _final_colors(rooms: list[dict], edges: list[tuple[int, int]]) -> list[int]:
    graph = _adjacency(len(rooms), edges)
    final: list[int] = []
    for room in rooms:
        room_id = int(room["id"])
        own_area = int(room["area"])
        candidates = [
            other_id for other_id in graph[room_id] if int(rooms[other_id]["area"]) > own_area
        ]
        if not candidates:
            final.append(int(room["color"]))
            continue

        max_area = max(int(rooms[other_id]["area"]) for other_id in candidates)
        best_colors = {
            int(rooms[other_id]["color"])
            for other_id in candidates
            if int(rooms[other_id]["area"]) == max_area
        }
        final.append(next(iter(best_colors)) if len(best_colors) == 1 else int(room["color"]))
    return final


def _paint_solution(shape: tuple[int, int], rooms: list[dict], edges: list[tuple[int, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)

    wall_rows = {0, int(shape[0]) - 1}
    wall_cols = {0, int(shape[1]) - 1}
    for room in rooms:
        wall_rows.add(int(room["r0"]) - 1)
        wall_rows.add(int(room["r1"]) + 1)
        wall_cols.add(int(room["c0"]) - 1)
        wall_cols.add(int(room["c1"]) + 1)

    for r in wall_rows:
        if 0 <= r < shape[0]:
            out[r, :] = WALL
    for c in wall_cols:
        if 0 <= c < shape[1]:
            out[:, c] = WALL

    colors = _final_colors(rooms, edges)
    for room, color in zip(rooms, colors):
        r0 = int(room["r0"])
        c0 = int(room["c0"])
        h = int(room["rect_h"])
        w = int(room["rect_w"])
        out[r0 : r0 + h, c0 : c0 + w] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack each room fragment into a compact top-left rectangle, then recolor
    smaller connected rooms with the larger neighbor's color. Orange gate pixels
    are removed back into grey walls in the output.
    """
    g = np.asarray(grid, dtype=int)
    rooms, room_ids = _extract_rooms(g)
    edges = _gate_edges(g, room_ids)
    return _paint_solution(g.shape, rooms, edges)


def _room_layout() -> tuple[list[int], list[int]]:
    room_rows, room_cols = random.choices(LAYOUT_CHOICES, weights=LAYOUT_WEIGHTS, k=1)[0]
    heights = [random.randint(ROOM_SIZE_RANGE[0], ROOM_SIZE_RANGE[1]) for _ in range(room_rows)]
    widths = [random.randint(ROOM_SIZE_RANGE[0], ROOM_SIZE_RANGE[1]) for _ in range(room_cols)]
    return heights, widths


def _room_spans(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int]]:
    spans: list[tuple[int, int]] = []
    walls = [0]
    cursor = 0
    for size in sizes:
        start = cursor + 1
        end = start + int(size) - 1
        spans.append((start, end))
        cursor = end + 1
        walls.append(cursor)
    return spans, walls


def _grid_graph(room_rows: int, room_cols: int) -> dict[int, list[int]]:
    graph: dict[int, list[int]] = {}
    for r in range(room_rows):
        for c in range(room_cols):
            room_id = r * room_cols + c
            neighbors: list[int] = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < room_rows and 0 <= cc < room_cols:
                    neighbors.append(rr * room_cols + cc)
            graph[room_id] = neighbors
    return graph


def _perfect_matching(room_rows: int, room_cols: int) -> list[tuple[int, int]] | None:
    graph = _grid_graph(room_rows, room_cols)
    unmatched = set(graph)

    def dfs(current: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if not unmatched:
            return list(current)

        room_id = min(unmatched)
        candidates = [other for other in graph[room_id] if other in unmatched]
        random.shuffle(candidates)

        unmatched.remove(room_id)
        for other in candidates:
            unmatched.remove(other)
            current.append((room_id, other))
            solved = dfs(current)
            if solved is not None:
                return solved
            current.pop()
            unmatched.add(other)
        unmatched.add(room_id)
        return None

    return dfs([])


def _feasible_areas(height: int, width: int) -> list[int]:
    areas: list[int] = []
    for area in range(MIN_FRAGMENT_AREA, int(height) * int(width)):
        rect_h, rect_w = _compact_dims(area)
        if rect_h <= height and rect_w <= width and not (rect_h == height and rect_w == width):
            areas.append(area)
    return areas


def _relation_order(pair_count: int) -> list[str]:
    relations = ["a_gt_b", "b_gt_a", "equal"]
    if pair_count <= len(relations):
        return random.sample(relations, pair_count)
    order: list[str] = []
    while len(order) < pair_count:
        order.extend(random.sample(relations, len(relations)))
    return order[:pair_count]


def _choose_pair_areas(areas_a: list[int], areas_b: list[int], relation: str) -> tuple[int, int] | None:
    if relation == "equal":
        options = [(area, area) for area in sorted(set(areas_a) & set(areas_b))]
    elif relation == "a_gt_b":
        options = [(a, b) for a in areas_a for b in areas_b if a > b]
    else:
        options = [(a, b) for a in areas_a for b in areas_b if b > a]

    if not options:
        return None

    options.sort(key=lambda pair: abs(pair[0] - pair[1]), reverse=(relation != "equal"))
    top = options[: min(20, len(options))]
    return random.choice(top)


def _neighbors4(r: int, c: int, height: int, width: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        rr = int(r) + dr
        cc = int(c) + dc
        if 0 <= rr < height and 0 <= cc < width:
            out.append((rr, cc))
    return out


def _random_fragment(height: int, width: int, area: int, rect_h: int, rect_w: int) -> np.ndarray | None:
    forbidden = {(r, c) for r in range(rect_h) for c in range(rect_w)}

    for _ in range(SHAPE_ATTEMPTS):
        start = (random.randrange(height), random.randrange(width))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], height, width))

        while len(cells) < area:
            pick = random.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            for nb in _neighbors4(pick[0], pick[1], height, width):
                if nb not in cells:
                    frontier.add(nb)

        if cells == forbidden:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
        if bbox_area == area:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        return mask

    return None


def _gate_cell(room_a: dict, room_b: dict) -> tuple[int, int]:
    if int(room_a["row_band"]) == int(room_b["row_band"]):
        left, right = sorted((room_a, room_b), key=lambda room: int(room["col_band"]))
        row = random.randint(int(left["r0"]), int(left["r1"]))
        col = int(left["c1"]) + 1
        return row, col

    top, bottom = sorted((room_a, room_b), key=lambda room: int(room["row_band"]))
    row = int(top["r1"]) + 1
    col = random.randint(int(top["c0"]), int(top["c1"]))
    return row, col


def _sample_rooms() -> tuple[np.ndarray, list[dict], list[tuple[int, int]]] | None:
    heights, widths = _room_layout()
    row_spans, wall_rows = _room_spans(heights)
    col_spans, wall_cols = _room_spans(widths)

    room_rows = len(heights)
    room_cols = len(widths)
    room_count = room_rows * room_cols
    if room_count > len(ROOM_COLORS):
        return None

    matching = _perfect_matching(room_rows, room_cols)
    if matching is None:
        return None

    rooms: list[dict] = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            height = int(r1 - r0 + 1)
            width = int(c1 - c0 + 1)
            areas = _feasible_areas(height, width)
            if not areas:
                return None
            rooms.append(
                {
                    "id": len(rooms),
                    "row_band": ri,
                    "col_band": ci,
                    "r0": int(r0),
                    "r1": int(r1),
                    "c0": int(c0),
                    "c1": int(c1),
                    "height": height,
                    "width": width,
                    "areas": areas,
                }
            )

    relations = _relation_order(len(matching))
    for relation, (a, b) in zip(relations, matching):
        chosen = _choose_pair_areas(rooms[a]["areas"], rooms[b]["areas"], relation)
        if chosen is None:
            other_relations = ["equal", "a_gt_b", "b_gt_a"]
            for fallback in other_relations:
                chosen = _choose_pair_areas(rooms[a]["areas"], rooms[b]["areas"], fallback)
                if chosen is not None:
                    break
        if chosen is None:
            return None
        rooms[a]["area"] = int(chosen[0])
        rooms[b]["area"] = int(chosen[1])

    if any("area" not in room for room in rooms):
        return None

    colors = random.sample(ROOM_COLORS, room_count)
    for room, color in zip(rooms, colors):
        rect_h, rect_w = _compact_dims(int(room["area"]))
        room["rect_h"] = rect_h
        room["rect_w"] = rect_w
        room["color"] = int(color)

    height = sum(heights) + room_rows + 1
    width = sum(widths) + room_cols + 1
    grid = np.full((height, width), BG, dtype=int)
    grid[wall_rows, :] = WALL
    grid[:, wall_cols] = WALL

    edges: list[tuple[int, int]] = []
    for a, b in matching:
        gate_r, gate_c = _gate_cell(rooms[a], rooms[b])
        grid[gate_r, gate_c] = GATE
        edges.append(tuple(sorted((a, b))))

    for room in rooms:
        mask = _random_fragment(
            int(room["height"]),
            int(room["width"]),
            int(room["area"]),
            int(room["rect_h"]),
            int(room["rect_w"]),
        )
        if mask is None:
            return None
        sub = grid[int(room["r0"]) : int(room["r1"]) + 1, int(room["c0"]) : int(room["c1"]) + 1]
        sub[mask] = int(room["color"])

    return grid, rooms, sorted(set(edges))


def _validate_sample(grid: np.ndarray, rooms: list[dict], edges: list[tuple[int, int]]) -> bool:
    expected = _paint_solution(grid.shape, rooms, edges)
    try:
        solved = main(grid)
    except Exception:
        return False

    if not np.array_equal(solved, expected):
        return False
    if np.any(solved == GATE):
        return False

    for room, color in zip(rooms, _final_colors(rooms, edges)):
        sub = solved[int(room["r0"]) : int(room["r1"]) + 1, int(room["c0"]) : int(room["c1"]) + 1]
        mask = np.zeros_like(sub, dtype=bool)
        mask[: int(room["rect_h"]), : int(room["rect_w"])] = True
        if not bool(np.all(sub[mask] == int(color))):
            return False
        if not bool(np.all(sub[~mask] == BG)):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        sampled = _sample_rooms()
        if sampled is None:
            continue
        grid, rooms, edges = sampled
        if _validate_sample(grid, rooms, edges):
            return grid
    raise RuntimeError("failed to generate a valid room-fragment puzzle")
