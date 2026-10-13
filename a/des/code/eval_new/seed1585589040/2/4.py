from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
import random

import numpy as np


BLACK = 0
ROOM_AXIS_CHOICES = (2, 3, 4)
ROOM_SIDE_MIN = 3
ROOM_SIDE_MAX = 6
WALL_COLOR_COUNT_CHOICES = (1, 2, 3)
FLOOR_COLOR_COUNT_CHOICES = (3, 4, 5)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _same_color_components(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, list[tuple[int, int]]]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append((int(color), coords))
    return comps


def _extract_rooms(grid: np.ndarray) -> tuple[list[dict], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    room_id = np.full((h, w), -1, dtype=int)
    rooms: list[dict] = []

    for color, coords in _same_color_components(g):
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        box_h = int(r1 - r0 + 1)
        box_w = int(c1 - c0 + 1)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if box_h < 2 or box_w < 2:
            continue
        if len(coords) != box_h * box_w:
            continue

        rid = len(rooms)
        for rr, cc in coords:
            room_id[rr, cc] = rid
        rooms.append(
            {
                "id": int(rid),
                "color": int(color),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "area": int(len(coords)),
            }
        )

    return rooms, room_id


def _find_doors(grid: np.ndarray, room_id: np.ndarray) -> tuple[list[tuple[int, int, int, int]], dict[int, list[int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = map(int, g.shape)
    doors: list[tuple[int, int, int, int]] = []
    adjacency: dict[int, list[int]] = defaultdict(list)

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BLACK:
                continue

            left = int(room_id[r, c - 1])
            right = int(room_id[r, c + 1])
            up = int(room_id[r - 1, c])
            down = int(room_id[r + 1, c])

            vertical = left >= 0 and right >= 0 and left != right and up < 0 and down < 0
            horizontal = up >= 0 and down >= 0 and up != down and left < 0 and right < 0
            if vertical == horizontal:
                continue

            a, b = (left, right) if vertical else (up, down)
            doors.append((int(r), int(c), int(a), int(b)))
            adjacency[int(a)].append(int(b))
            adjacency[int(b)].append(int(a))

    return doors, adjacency


def _resolve_target_color(room_index: int, neighbor_ids: list[int], rooms: list[dict]) -> int:
    if not neighbor_ids:
        return int(rooms[room_index]["color"])
    if len(neighbor_ids) == 1:
        return int(rooms[neighbor_ids[0]]["color"])

    color_counts = Counter(int(rooms[n]["color"]) for n in neighbor_ids)
    best_count = max(color_counts.values())
    tied_colors = [int(color) for color, count in color_counts.items() if int(count) == int(best_count)]
    if len(tied_colors) == 1:
        return int(tied_colors[0])

    best_color = int(tied_colors[0])
    best_key: tuple[int, int, int] | None = None
    for color in tied_colors:
        top_area = -1
        top_pos = None
        for nid in set(neighbor_ids):
            neighbor = rooms[int(nid)]
            if int(neighbor["color"]) != int(color):
                continue
            pos = neighbor["bbox"][:2]
            area = int(neighbor["area"])
            if area > top_area or (area == top_area and (top_pos is None or pos < top_pos)):
                top_area = area
                top_pos = pos
        assert top_pos is not None
        key = (int(top_area), -int(top_pos[0]), -int(top_pos[1]))
        if best_key is None or key > best_key:
            best_key = key
            best_color = int(color)
    return int(best_color)


def _room_analysis(grid: np.ndarray) -> tuple[list[dict], np.ndarray, list[tuple[int, int, int, int]], dict[int, list[int]], dict[int, int]]:
    rooms, room_id = _extract_rooms(grid)
    doors, adjacency = _find_doors(grid, room_id)
    targets = {
        int(room["id"]): _resolve_target_color(int(room["id"]), list(adjacency.get(int(room["id"]), [])), rooms)
        for room in rooms
    }
    return rooms, room_id, doors, adjacency, targets


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint each room floor by the door-weighted majority floor color of its neighboring rooms.

    Input structure:
    - bg = black.
    - One-cell-thick colored border segments partition the grid into rectangular rooms.
    - Each room interior is a solid monochrome rectangle whose floor color differs from the border colors.
    - Some border segments contain single black pixels that act as doors between adjacent rooms.

    Rule:
    - Identify every rectangular room interior and its floor color.
    - For each black door pixel, read the two room interiors on opposite sides to determine the neighboring rooms linked by that door.
    - If a room has exactly one door, recolor its entire floor to its lone neighbor's floor color.
    - If a room has multiple doors, recolor its floor to the floor color appearing through the most doors; if the top counts tie, use the tied color whose neighboring room has the largest area.
    - Leave all border pixels and door pixels unchanged.

    Key insight: The black door gaps are the whole adjacency signal, and every door contributes one vote even when several doors lead to the same neighboring room.
    """
    g = np.asarray(grid, dtype=int)
    rooms, _room_id, _doors, _adjacency, targets = _room_analysis(g)

    out = g.copy()
    for room in rooms:
        rid = int(room["id"])
        r0, c0, r1, c1 = room["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(targets[rid])
    return out


def _build_room_layout(row_heights: list[int], col_widths: list[int]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[dict]]:
    row_ranges: list[tuple[int, int]] = []
    col_ranges: list[tuple[int, int]] = []

    cursor = 1
    for height in row_heights:
        row_ranges.append((int(cursor), int(cursor + height - 1)))
        cursor += int(height) + 1

    cursor = 1
    for width in col_widths:
        col_ranges.append((int(cursor), int(cursor + width - 1)))
        cursor += int(width) + 1

    rooms: list[dict] = []
    n_cols = len(col_widths)
    for rr, (r0, r1) in enumerate(row_ranges):
        for cc, (c0, c1) in enumerate(col_ranges):
            rid = rr * n_cols + cc
            rooms.append(
                {
                    "id": int(rid),
                    "grid_pos": (int(rr), int(cc)),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int((r1 - r0 + 1) * (c1 - c0 + 1)),
                    "color": -1,
                }
            )
    return row_ranges, col_ranges, rooms


def _edge_slots(
    row_ranges: list[tuple[int, int]],
    col_ranges: list[tuple[int, int]],
) -> tuple[list[dict], list[list[int]]]:
    edges: list[dict] = []
    room_count = len(row_ranges) * len(col_ranges)
    incident: list[list[int]] = [[] for _ in range(room_count)]
    n_cols = len(col_ranges)

    for rr, (r0, r1) in enumerate(row_ranges):
        for cc in range(len(col_ranges) - 1):
            a = rr * n_cols + cc
            b = a + 1
            wall_c = int(col_ranges[cc][1] + 1)
            slots = [(int(r), int(wall_c)) for r in range(r0 + 1, r1)]
            idx = len(edges)
            edges.append({"a": int(a), "b": int(b), "slots": slots})
            incident[a].append(idx)
            incident[b].append(idx)

    for rr in range(len(row_ranges) - 1):
        wall_r = int(row_ranges[rr][1] + 1)
        for cc, (c0, c1) in enumerate(col_ranges):
            a = rr * n_cols + cc
            b = (rr + 1) * n_cols + cc
            slots = [(int(wall_r), int(c)) for c in range(c0 + 1, c1)]
            idx = len(edges)
            edges.append({"a": int(a), "b": int(b), "slots": slots})
            incident[a].append(idx)
            incident[b].append(idx)

    return edges, incident


def _max_single_pixel_doors(slots: list[tuple[int, int]]) -> int:
    return (len(slots) + 1) // 2


def _choose_door_cells(slots: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    ordered = sorted(slots)
    if count == 0:
        return []
    valid = []
    for picks in combinations(range(len(ordered)), count):
        if all(picks[i + 1] - picks[i] > 1 for i in range(len(picks) - 1)):
            valid.append(picks)
    assert valid, "requested more doors than a wall can hold as isolated pixels"
    chosen = random.choice(valid)
    return [ordered[i] for i in chosen]


def _tree_door_counts(room_count: int, edges: list[dict]) -> list[int] | None:
    parent = list(range(room_count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    order = list(range(len(edges)))
    random.shuffle(order)
    door_counts = [0] * len(edges)
    for idx in order:
        a = int(edges[idx]["a"])
        b = int(edges[idx]["b"])
        if union(a, b):
            door_counts[idx] = 1

    if sum(door_counts) != room_count - 1:
        return None
    return door_counts


def _neighbor_lists(room_count: int, edges: list[dict], door_counts: list[int]) -> dict[int, list[int]]:
    adjacency: dict[int, list[int]] = defaultdict(list)
    for idx, count in enumerate(door_counts):
        if count <= 0:
            continue
        a = int(edges[idx]["a"])
        b = int(edges[idx]["b"])
        adjacency[a].extend([int(b)] * int(count))
        adjacency[b].extend([int(a)] * int(count))
    for rid in range(room_count):
        adjacency.setdefault(int(rid), [])
    return adjacency


def _branch_profile(rooms: list[dict], neighbors: dict[int, list[int]]) -> dict[str, int]:
    stats = {"single": 0, "majority": 0, "tie": 0, "weighted": 0}
    for room in rooms:
        rid = int(room["id"])
        adj = list(neighbors[rid])
        if len(adj) == 1:
            stats["single"] += 1
            continue
        if len(adj) < 2:
            continue
        if len(adj) > len(set(adj)):
            stats["weighted"] += 1
        counts = Counter(int(rooms[n]["color"]) for n in adj)
        top = max(counts.values())
        winner_count = sum(1 for count in counts.values() if int(count) == int(top))
        if winner_count == 1:
            stats["majority"] += 1
        else:
            stats["tie"] += 1
    return stats


def _sample_case() -> np.ndarray | None:
    n_rows = random.choice(ROOM_AXIS_CHOICES)
    n_cols = random.choice(ROOM_AXIS_CHOICES)
    row_heights = [random.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(n_rows)]
    col_widths = [random.randint(ROOM_SIDE_MIN, ROOM_SIDE_MAX) for _ in range(n_cols)]
    row_ranges, col_ranges, rooms = _build_room_layout(row_heights, col_widths)

    height = sum(row_heights) + n_rows + 1
    width = sum(col_widths) + n_cols + 1
    if max(height, width) > 30:
        return None

    edges, incident = _edge_slots(row_ranges, col_ranges)
    room_count = len(rooms)
    door_counts = _tree_door_counts(room_count, edges)
    if door_counts is None:
        return None

    tree_neighbors = _neighbor_lists(room_count, edges, door_counts)
    tie_candidates = []
    majority_candidates = []
    for room in rooms:
        rid = int(room["id"])
        used_edges = [idx for idx in incident[rid] if int(door_counts[idx]) > 0]
        if len(used_edges) != 2:
            continue

        neighbor_ids = []
        for idx in used_edges:
            edge = edges[idx]
            neighbor_ids.append(int(edge["b"] if int(edge["a"]) == rid else edge["a"]))

        if rooms[neighbor_ids[0]]["area"] != rooms[neighbor_ids[1]]["area"]:
            tie_candidates.append((int(rid), neighbor_ids))

        expandable = [idx for idx in used_edges if _max_single_pixel_doors(list(edges[idx]["slots"])) >= 2]
        if expandable:
            majority_candidates.append((int(rid), neighbor_ids, expandable))

    scenario = random.choice(("tie_break", "weighted_majority"))
    if scenario == "tie_break":
        if not tie_candidates:
            return None
        focus_room, focus_neighbors = random.choice(tie_candidates)
        boosted_edge = None
    else:
        if not majority_candidates:
            return None
        focus_room, focus_neighbors, expandable = random.choice(majority_candidates)
        boosted_edge = random.choice(expandable)
        door_counts[boosted_edge] += 1

    all_colors = list(range(1, 10))
    random.shuffle(all_colors)
    wall_color_count = random.choice(WALL_COLOR_COUNT_CHOICES)
    wall_colors = [int(c) for c in all_colors[:wall_color_count]]
    floor_pool = [int(c) for c in all_colors[wall_color_count:]]
    floor_color_count = random.choice(
        [n for n in FLOOR_COLOR_COUNT_CHOICES if n <= len(floor_pool)]
    )
    floor_palette = random.sample(floor_pool, floor_color_count)

    room_colors = [int(random.choice(floor_palette)) for _ in range(room_count)]
    seed_rooms = random.sample(range(room_count), k=min(room_count, len(floor_palette)))
    for rid, color in zip(seed_rooms, floor_palette, strict=False):
        room_colors[int(rid)] = int(color)

    distinct_focus_colors = random.sample(floor_palette, k=min(3, len(floor_palette)))
    while len(distinct_focus_colors) < 3:
        distinct_focus_colors.append(int(floor_palette[0]))

    if scenario == "tie_break":
        n1, n2 = map(int, focus_neighbors)
        room_colors[n1] = int(distinct_focus_colors[0])
        room_colors[n2] = int(distinct_focus_colors[1])
        room_colors[focus_room] = int(distinct_focus_colors[2])
    else:
        n_major = int(edges[boosted_edge]["b"] if int(edges[boosted_edge]["a"]) == focus_room else edges[boosted_edge]["a"])
        n_other = next(int(n) for n in focus_neighbors if int(n) != n_major)
        room_colors[n_major] = int(distinct_focus_colors[0])
        room_colors[n_other] = int(distinct_focus_colors[1])
        room_colors[focus_room] = int(distinct_focus_colors[2])

    for rid, color in enumerate(room_colors):
        rooms[rid]["color"] = int(color)

    wall_row_colors = [int(random.choice(wall_colors)) for _ in range(n_rows + 1)]
    wall_col_colors = [int(random.choice(wall_colors)) for _ in range(n_cols + 1)]
    grid = np.full((height, width), BLACK, dtype=int)

    wall_rows = [0] + [int(r1 + 1) for _, r1 in row_ranges]
    wall_cols = [0] + [int(c1 + 1) for _, c1 in col_ranges]
    for idx, rr in enumerate(wall_rows):
        grid[int(rr), :] = int(wall_row_colors[idx])
    for idx, cc in enumerate(wall_cols):
        grid[:, int(cc)] = int(wall_col_colors[idx])

    for room in rooms:
        r0, c0, r1, c1 = room["bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(room["color"])

    for idx, count in enumerate(door_counts):
        if count <= 0:
            continue
        slots = list(edges[idx]["slots"])
        if _max_single_pixel_doors(slots) < int(count):
            return None
        for r, c in _choose_door_cells(slots, int(count)):
            grid[int(r), int(c)] = BLACK

    neighbors = _neighbor_lists(room_count, edges, door_counts)
    targets = {
        rid: _resolve_target_color(rid, list(neighbors[rid]), rooms)
        for rid in range(room_count)
    }

    expected = grid.copy()
    changed_rooms = 0
    for room in rooms:
        rid = int(room["id"])
        r0, c0, r1, c1 = room["bbox"]
        expected[r0 : r1 + 1, c0 : c1 + 1] = int(targets[rid])
        if int(targets[rid]) != int(room["color"]):
            changed_rooms += 1

    if changed_rooms < 2:
        return None

    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    parsed_rooms, _room_id, parsed_doors, parsed_neighbors, _targets = _room_analysis(grid)
    if len(parsed_rooms) != room_count or len(parsed_doors) != sum(door_counts):
        return None
    if any(len(parsed_neighbors[rid]) == 0 for rid in range(room_count)):
        return None

    stats = _branch_profile(parsed_rooms, parsed_neighbors)
    if stats["single"] == 0:
        return None
    if scenario == "tie_break" and stats["tie"] == 0:
        return None
    if scenario == "weighted_majority" and (stats["majority"] == 0 or stats["weighted"] == 0):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _sample_case()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid room-door voting puzzle")
