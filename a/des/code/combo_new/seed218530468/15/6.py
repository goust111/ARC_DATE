import random
from collections import defaultdict
from itertools import combinations

import numpy as np


BLACK = 0
GREY = 5
DOOR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400

ROOM_GRID_CHOICES = ((2, 3), (3, 2), (2, 4), (4, 2), (3, 3))
ROOM_SIZE_MIN = 2
ROOM_SIZE_MAX = 6
SCENARIO_ORDER = ("pair_pairs", "chain2", "mixed3", "chain3", "mixed4")


class DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _black_components(grid: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = grid.shape
    room_of = np.full((h, w), -1, dtype=int)
    rooms: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if grid[r, c] != BLACK or room_of[r, c] != -1:
                continue
            stack = [(r, c)]
            room_id = len(rooms)
            room_of[r, c] = room_id
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BLACK and room_of[nr, nc] == -1:
                        room_of[nr, nc] = room_id
                        stack.append((nr, nc))
            rooms.append(np.array(coords, dtype=int))
    return rooms, room_of


def _side_room_ids(room_of: np.ndarray, cells: list[tuple[int, int]], dr: int, dc: int) -> int:
    h, w = room_of.shape
    ids = {
        int(room_of[r + dr, c + dc])
        for r, c in cells
        if 0 <= r + dr < h and 0 <= c + dc < w and int(room_of[r + dr, c + dc]) >= 0
    }
    assert len(ids) == 1, "door side must touch exactly one room"
    return ids.pop()


def _extract_doors(grid: np.ndarray, room_of: np.ndarray) -> list[dict]:
    door_infos: list[dict] = []
    marker_colors = [int(color) for color in np.unique(grid) if int(color) not in {BLACK, GREY}]
    assert 2 <= len(marker_colors) <= 4, "expected 2 to 4 door colors"

    for color in marker_colors:
        coords = np.argwhere(grid == color)
        assert coords.shape == (2, 2), "each door color must appear exactly twice"
        pts = sorted((int(r), int(c)) for r, c in coords)
        (r0, c0), (r1, c1) = pts
        if r0 == r1 and c1 == c0 + 1:
            cells = [(r0, c0), (r1, c1)]
            room_a = _side_room_ids(room_of, cells, -1, 0)
            room_b = _side_room_ids(room_of, cells, 1, 0)
        elif c0 == c1 and r1 == r0 + 1:
            cells = [(r0, c0), (r1, c1)]
            room_a = _side_room_ids(room_of, cells, 0, -1)
            room_b = _side_room_ids(room_of, cells, 0, 1)
        else:
            raise AssertionError("door markers must form an adjacent pair")

        assert room_a != room_b, "a door must connect two distinct rooms"
        door_infos.append(
            {
                "color": color,
                "cells": cells,
                "rooms": (room_a, room_b),
                "rank": (min(r0, r1), min(c0, c1)),
            }
        )
    return door_infos


def _solve_room_groups(num_rooms: int, doors: list[dict]) -> tuple[dict[int, list[int]], dict[int, list[dict]]]:
    dsu = DSU(num_rooms)
    for door in doors:
        room_a, room_b = door["rooms"]
        dsu.union(int(room_a), int(room_b))

    rooms_by_root: dict[int, list[int]] = defaultdict(list)
    for room_id in range(num_rooms):
        rooms_by_root[dsu.find(room_id)].append(room_id)

    doors_by_root: dict[int, list[dict]] = defaultdict(list)
    for door in doors:
        doors_by_root[dsu.find(int(door["rooms"][0]))].append(door)
    return rooms_by_root, doors_by_root


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open the marked wall doors and fill each connected room group with the winning door color.

    Input structure:
    - Black pixels are room interiors.
    - Grey one-pixel walls form a complete orthogonal floor plan that partitions the grid into rectangular rooms.
    - Each non-black, non-grey color appears exactly twice as an adjacent marker pair on one wall segment; each pair marks one door location.

    Rule:
    - Replace every colored marker pair with black to open a two-pixel-wide doorway in that wall.
    - Treat the original black rooms as nodes and every doorway as a connection between the two rooms on opposite sides of that wall.
    - For each connected room group with at least two rooms, fill all of those rooms' original interior black pixels with the color of the topmost door pair in that group; isolated rooms stay black and remaining walls stay grey.

    Key insight:
    - The colored wall pairs are not pixels to preserve; they are door instructions that first change room connectivity and then choose the fill color for the entire connected room component.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    rooms, room_of = _black_components(g)
    assert rooms, "expected at least one room"
    doors = _extract_doors(g, room_of)

    out = g.copy()
    for door in doors:
        for r, c in door["cells"]:
            out[r, c] = BLACK

    rooms_by_root, doors_by_root = _solve_room_groups(len(rooms), doors)
    for root, room_ids in rooms_by_root.items():
        if len(room_ids) < 2:
            continue
        fill_color = min(doors_by_root[root], key=lambda item: item["rank"])["color"]
        for room_id in room_ids:
            coords = rooms[int(room_id)]
            out[coords[:, 0], coords[:, 1]] = int(fill_color)
    return out


def _sample_room_sizes(count: int) -> list[int]:
    return [random.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(count)]


def _build_scene() -> tuple[np.ndarray, list[dict], list[dict]]:
    room_rows, room_cols = random.choice(ROOM_GRID_CHOICES)
    heights = _sample_room_sizes(room_rows)
    widths = _sample_room_sizes(room_cols)

    h = sum(heights) + room_rows + 1
    w = sum(widths) + room_cols + 1
    assert h <= 30 and w <= 30

    grid = np.full((h, w), BLACK, dtype=int)

    row_starts: list[int] = []
    row_walls = [0]
    cur = 1
    for size in heights:
        row_starts.append(cur)
        cur += size
        row_walls.append(cur)
        cur += 1

    col_starts: list[int] = []
    col_walls = [0]
    cur = 1
    for size in widths:
        col_starts.append(cur)
        cur += size
        col_walls.append(cur)
        cur += 1

    grid[row_walls, :] = GREY
    grid[:, col_walls] = GREY

    rooms: list[dict] = []
    room_ids = np.full((room_rows, room_cols), -1, dtype=int)
    for rr in range(room_rows):
        for cc in range(room_cols):
            room_id = len(rooms)
            r0 = row_starts[rr]
            r1 = r0 + heights[rr] - 1
            c0 = col_starts[cc]
            c1 = c0 + widths[cc] - 1
            rooms.append({"bbox": (r0, r1, c0, c1), "grid_pos": (rr, cc)})
            room_ids[rr, cc] = room_id

    edges: list[dict] = []
    for rr in range(room_rows - 1):
        wall_r = row_walls[rr + 1]
        for cc in range(room_cols):
            c0 = col_starts[cc]
            c1 = c0 + widths[cc] - 1
            edges.append(
                {
                    "rooms": (int(room_ids[rr, cc]), int(room_ids[rr + 1, cc])),
                    "orientation": "horizontal",
                    "line": wall_r,
                    "span": (c0, c1),
                }
            )
    for rr in range(room_rows):
        r0 = row_starts[rr]
        r1 = r0 + heights[rr] - 1
        for cc in range(room_cols - 1):
            wall_c = col_walls[cc + 1]
            edges.append(
                {
                    "rooms": (int(room_ids[rr, cc]), int(room_ids[rr, cc + 1])),
                    "orientation": "vertical",
                    "line": wall_c,
                    "span": (r0, r1),
                }
            )
    return grid, rooms, edges


def _edge_signature(edge_ids: tuple[int, ...], edges: list[dict]) -> tuple[int, ...] | None:
    adjacency: dict[int, set[int]] = defaultdict(set)
    nodes: set[int] = set()
    for edge_id in edge_ids:
        room_a, room_b = edges[edge_id]["rooms"]
        adjacency[int(room_a)].add(int(room_b))
        adjacency[int(room_b)].add(int(room_a))
        nodes.add(int(room_a))
        nodes.add(int(room_b))

    if any(len(adjacency[node]) > 2 for node in nodes):
        return None

    seen: set[int] = set()
    component_edge_counts: list[int] = []
    for node in nodes:
        if node in seen:
            continue
        stack = [node]
        seen.add(node)
        comp_nodes = 0
        comp_degree_sum = 0
        while stack:
            cur = stack.pop()
            comp_nodes += 1
            comp_degree_sum += len(adjacency[cur])
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        comp_edges = comp_degree_sum // 2
        if comp_edges != comp_nodes - 1:
            return None
        component_edge_counts.append(comp_edges)
    return tuple(sorted(component_edge_counts, reverse=True))


def _scenario_candidates(edges: list[dict]) -> dict[str, list[tuple[int, ...]]]:
    mapping: dict[str, list[tuple[int, ...]]] = defaultdict(list)
    for door_count in (2, 3, 4):
        for edge_ids in combinations(range(len(edges)), door_count):
            signature = _edge_signature(edge_ids, edges)
            if signature == (1, 1):
                mapping["pair_pairs"].append(edge_ids)
            elif signature == (2,):
                mapping["chain2"].append(edge_ids)
            elif signature == (2, 1):
                mapping["mixed3"].append(edge_ids)
            elif signature == (3,):
                mapping["chain3"].append(edge_ids)
            elif signature == (3, 1):
                mapping["mixed4"].append(edge_ids)
    return mapping


def _rooms_by_selected_component(selected_edges: list[dict]) -> dict[int, list[dict]]:
    room_ids = sorted({room for edge in selected_edges for room in edge["rooms"]})
    remap = {room_id: idx for idx, room_id in enumerate(room_ids)}
    dsu = DSU(len(room_ids))
    for edge in selected_edges:
        room_a, room_b = edge["rooms"]
        dsu.union(remap[int(room_a)], remap[int(room_b)])
    groups: dict[int, list[dict]] = defaultdict(list)
    for edge in selected_edges:
        root = dsu.find(remap[int(edge["rooms"][0])])
        groups[root].append(edge)
    return groups


def _sample_doors(selected_edges: list[dict]) -> list[dict] | None:
    edge_groups = _rooms_by_selected_component(selected_edges)
    for _ in range(80):
        colors = random.sample(DOOR_COLORS, len(selected_edges))
        random.shuffle(colors)
        doors: list[dict] = []
        for color, edge in zip(colors, selected_edges):
            start, end = edge["span"]
            door_start = random.randint(int(start), int(end) - 1)
            if edge["orientation"] == "horizontal":
                cells = [(int(edge["line"]), door_start), (int(edge["line"]), door_start + 1)]
                rank = (int(edge["line"]), int(door_start))
            else:
                cells = [(door_start, int(edge["line"])), (door_start + 1, int(edge["line"]))]
                rank = (int(door_start), int(edge["line"]))
            doors.append(
                {
                    "color": int(color),
                    "cells": cells,
                    "rooms": edge["rooms"],
                    "rank": rank,
                }
            )

        edge_to_door = {tuple(edge["rooms"]): door for edge, door in zip(selected_edges, doors)}
        ok = True
        for group_edges in edge_groups.values():
            if len(group_edges) == 1:
                continue
            rows = [edge_to_door[tuple(edge["rooms"])]["rank"][0] for edge in group_edges]
            if rows.count(min(rows)) != 1:
                ok = False
                break
        if ok:
            return doors
    return None


def _render_expected(base: np.ndarray, rooms: list[dict], doors: list[dict]) -> np.ndarray:
    out = base.copy()
    for door in doors:
        for r, c in door["cells"]:
            out[r, c] = BLACK

    rooms_by_root, doors_by_root = _solve_room_groups(len(rooms), doors)
    for root, room_ids in rooms_by_root.items():
        if len(room_ids) < 2:
            continue
        fill_color = min(doors_by_root[root], key=lambda item: item["rank"])["color"]
        for room_id in room_ids:
            r0, r1, c0, c1 = rooms[int(room_id)]["bbox"]
            out[r0 : r1 + 1, c0 : c1 + 1] = int(fill_color)
    return out


def _make_input() -> np.ndarray | None:
    base, rooms, edges = _build_scene()
    candidates = _scenario_candidates(edges)
    available = [name for name in SCENARIO_ORDER if candidates.get(name)]
    if not available:
        return None

    scenario = random.choice(available)
    selected_ids = random.choice(candidates[scenario])
    selected_edges = [edges[idx] for idx in selected_ids]
    doors = _sample_doors(selected_edges)
    if doors is None:
        return None

    inp = base.copy()
    for door in doors:
        for r, c in door["cells"]:
            inp[r, c] = int(door["color"])

    expected = _render_expected(base, rooms, doors)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _make_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid door-room puzzle")
