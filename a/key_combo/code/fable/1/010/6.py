import random
import sys
from collections import Counter, deque
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, components_mask


PALETTE_POOL = [Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_RETRIES = 400


def _palette_from_row(row: np.ndarray) -> list[int]:
    palette: list[int] = []
    for value in row:
        color = int(value)
        if color == Color.BLACK:
            break
        palette.append(color)
    assert palette, "palette row must begin with at least one color"
    assert all(int(v) == Color.BLACK for v in row[len(palette) :]), "palette tail must be black"
    return palette


def _door_orientation(area: np.ndarray, r: int, c: int) -> str | None:
    h, w = area.shape
    if int(area[r, c]) == Color.BLUE:
        return None
    lr = (
        0 < r < h - 1
        and 0 < c < w - 1
        and int(area[r - 1, c]) == Color.BLUE
        and int(area[r + 1, c]) == Color.BLUE
        and int(area[r, c - 1]) != Color.BLUE
        and int(area[r, c + 1]) != Color.BLUE
    )
    ud = (
        0 < r < h - 1
        and 0 < c < w - 1
        and int(area[r, c - 1]) == Color.BLUE
        and int(area[r, c + 1]) == Color.BLUE
        and int(area[r - 1, c]) != Color.BLUE
        and int(area[r + 1, c]) != Color.BLUE
    )
    if lr == ud:
        return None
    return "lr" if lr else "ud"


def _build_graph(room_count: int, edges: set[tuple[int, int]]) -> list[set[int]]:
    graph = [set() for _ in range(room_count)]
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return graph


def _bfs_distances(graph: list[set[int]], seed: int) -> dict[int, int]:
    dist = {int(seed): 0}
    queue = deque([int(seed)])
    while queue:
        cur = queue.popleft()
        for nxt in sorted(graph[cur]):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                queue.append(nxt)
    return dist


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    sep_rows = np.where(np.all(g == Color.GREY, axis=1))[0]
    assert len(sep_rows) == 1, "expected exactly one full grey separator row"
    sep_r = int(sep_rows[0])
    assert sep_r >= 1, "palette row must exist above the separator"

    palette = _palette_from_row(g[sep_r - 1])
    assert len(set(palette)) == len(palette), "palette colors should be unique"
    assert all(color in PALETTE_POOL for color in palette), "palette uses only non-special colors"

    area = g[sep_r + 1 :, :]
    assert area.size > 0, "main area must exist below the separator"

    allowed_area_colors = {Color.BLACK, Color.BLUE, palette[0]}
    assert set(map(int, np.unique(area))) <= allowed_area_colors, "main area may contain only walls, darkness, and the seed"

    seed_cells = np.argwhere(area == palette[0])
    assert seed_cells.shape[0] == 1, "expected a unique seed pixel in the main area"
    seed_r, seed_c = map(int, seed_cells[0])

    door_mask = np.zeros(area.shape, dtype=bool)
    door_cells: list[tuple[int, int, str]] = []
    for r in range(area.shape[0]):
        for c in range(area.shape[1]):
            orientation = _door_orientation(area, r, c)
            if orientation is not None:
                door_mask[r, c] = True
                door_cells.append((r, c, orientation))

    assert door_cells, "expected at least one doorway"
    assert not door_mask[seed_r, seed_c], "seed cannot sit on a doorway"

    room_mask = (area != Color.BLUE) & ~door_mask
    rooms = components_mask(room_mask, connectivity=4)
    assert len(rooms) >= 2, "expected multiple rooms"

    room_ids = np.full(area.shape, -1, dtype=int)
    for room_id, coords in enumerate(rooms):
        room_ids[coords[:, 0], coords[:, 1]] = int(room_id)

    edges: set[tuple[int, int]] = set()
    for r, c, orientation in door_cells:
        if orientation == "lr":
            a = int(room_ids[r, c - 1])
            b = int(room_ids[r, c + 1])
        else:
            a = int(room_ids[r - 1, c])
            b = int(room_ids[r + 1, c])
        assert a >= 0 and b >= 0 and a != b, "doorway must connect exactly two distinct rooms"
        edge = (a, b) if a < b else (b, a)
        edges.add(edge)

    graph = _build_graph(len(rooms), edges)
    seed_room = int(room_ids[seed_r, seed_c])
    assert seed_room >= 0
    distances = _bfs_distances(graph, seed_room)
    assert len(distances) == len(rooms), "every room must be reachable from the seed room"

    return {
        "grid": g,
        "sep_r": sep_r,
        "palette": palette,
        "area": area,
        "door_mask": door_mask,
        "door_cells": door_cells,
        "rooms": rooms,
        "room_ids": room_ids,
        "edges": edges,
        "graph": graph,
        "seed_room": seed_room,
        "distances": distances,
        "seed_cell": (seed_r, seed_c),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Flood rooms from the seed by doorway depth and stop once the palette runs out.

    Input structure:
    - The top row begins with a left-aligned palette strip and is black afterward; the row directly below it is a full grey separator.
    - Below the separator, blue one-cell-thick walls split a black region into multiple rooms, and some wall cells are replaced by one-cell black doorways.
    - Exactly one room contains a single seed pixel whose color matches the first palette color.

    Rule:
    - Color the entire seed room with the first palette color.
    - For every other room, compute the minimum number of doorways on any path from the seed room to that room; a room at depth d uses palette color d+1.
    - Rooms whose minimum doorway count is at least the palette length stay black; doorway cells stay black; the palette row, separator row, and blue walls stay unchanged.

    Key insight:
    - Distance is measured on the room graph, not by cell steps inside the floor plan, and the palette length is also the maximum flood depth.
    """
    info = _analyze_grid(grid)
    out = info["grid"].copy()
    area_out = out[info["sep_r"] + 1 :, :]
    palette = info["palette"]

    for room_id, coords in enumerate(info["rooms"]):
        depth = info["distances"][room_id]
        if depth >= len(palette):
            continue
        area_out[coords[:, 0], coords[:, 1]] = int(palette[depth])

    area_out[info["door_mask"]] = Color.BLACK
    return out


def _room_index(rr: int, cc: int, room_cols: int) -> int:
    return int(rr * room_cols + cc)


def _room_coord(idx: int, room_cols: int) -> tuple[int, int]:
    return int(idx // room_cols), int(idx % room_cols)


def _all_room_adjacencies(room_rows: int, room_cols: int) -> list[tuple[int, int, str, int, int]]:
    edges: list[tuple[int, int, str, int, int]] = []
    for rr in range(room_rows):
        for cc in range(room_cols):
            idx = _room_index(rr, cc, room_cols)
            if cc + 1 < room_cols:
                edges.append((idx, _room_index(rr, cc + 1, room_cols), "vertical", rr, cc))
            if rr + 1 < room_rows:
                edges.append((idx, _room_index(rr + 1, cc, room_cols), "horizontal", rr, cc))
    return edges


def _spanning_tree(room_rows: int, room_cols: int) -> set[tuple[int, int]]:
    room_count = room_rows * room_cols
    neighbors = {idx: [] for idx in range(room_count)}
    for a, b, _kind, _rr, _cc in _all_room_adjacencies(room_rows, room_cols):
        neighbors[a].append(b)
        neighbors[b].append(a)

    start = random.randrange(room_count)
    seen = {start}
    stack = [start]
    tree: set[tuple[int, int]] = set()
    while stack:
        cur = stack[-1]
        choices = [nxt for nxt in neighbors[cur] if nxt not in seen]
        if not choices:
            stack.pop()
            continue
        nxt = random.choice(choices)
        seen.add(nxt)
        stack.append(nxt)
        edge = (cur, nxt) if cur < nxt else (nxt, cur)
        tree.add(edge)
    assert len(tree) == room_count - 1
    return tree


def _sample_geometry() -> tuple[list[int], list[int]] | None:
    room_rows = random.randint(2, 4)
    room_cols = random.randint(2, 4)
    if room_rows * room_cols < 6:
        return None

    heights = [random.randint(2, 5) for _ in range(room_rows)]
    widths = [random.randint(2, 5) for _ in range(room_cols)]
    main_h = sum(heights) + room_rows + 1
    main_w = sum(widths) + room_cols + 1
    if main_h + 2 > 30 or main_w > 30:
        return None
    if len(set(heights)) == 1 and len(set(widths)) == 1:
        return None
    return heights, widths


def _room_ranges(sizes: list[int]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 1
    for size in sizes:
        spans.append((cursor, cursor + size - 1))
        cursor += size + 1
    return spans


def _seed_candidates(room_rows: int, room_cols: int, graph: list[set[int]]) -> list[tuple[int, dict[int, int]]]:
    candidates: list[tuple[int, dict[int, int]]] = []
    for seed in range(room_rows * room_cols):
        distances = _bfs_distances(graph, seed)
        if len(distances) != room_rows * room_cols:
            continue
        if max(distances.values()) < 2:
            continue
        counts = Counter(distances.values())
        if max((count for depth, count in counts.items() if depth > 0), default=0) < 2:
            continue
        if not any(
            distances[idx] > abs(_room_coord(seed, room_cols)[0] - _room_coord(idx, room_cols)[0]) + abs(_room_coord(seed, room_cols)[1] - _room_coord(idx, room_cols)[1])
            for idx in range(room_rows * room_cols)
        ):
            continue
        candidates.append((seed, distances))
    return candidates


def _choose_palette_length(max_depth: int) -> int:
    if max_depth >= 2 and random.random() < 0.65:
        return random.randint(2, min(len(PALETTE_POOL), max_depth))
    full_cover_min = max_depth + 1
    if full_cover_min <= len(PALETTE_POOL):
        return random.randint(full_cover_min, len(PALETTE_POOL))
    return random.randint(2, len(PALETTE_POOL))


def _door_cells_for_layout(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    edge_specs: list[tuple[int, int, str, int, int]],
    open_edges: set[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, int]]:
    door_cells: dict[tuple[int, int], tuple[int, int]] = {}
    for a, b, kind, rr, cc in edge_specs:
        edge = (a, b) if a < b else (b, a)
        if edge not in open_edges:
            continue
        if kind == "vertical":
            wall_c = col_spans[cc][1] + 1
            door_r = random.randint(row_spans[rr][0], row_spans[rr][1])
            door_cells[edge] = (door_r, wall_c)
        else:
            wall_r = row_spans[rr][1] + 1
            door_c = random.randint(col_spans[cc][0], col_spans[cc][1])
            door_cells[edge] = (wall_r, door_c)
    return door_cells


def _render_input(
    heights: list[int],
    widths: list[int],
    palette: list[int],
    seed_room: int,
    door_cells: dict[tuple[int, int], tuple[int, int]],
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    row_spans = _room_ranges(heights)
    col_spans = _room_ranges(widths)
    main_h = sum(heights) + len(heights) + 1
    main_w = sum(widths) + len(widths) + 1

    grid = np.full((main_h + 2, main_w), Color.BLACK, dtype=int)
    grid[1, :] = Color.GREY
    grid[0, : len(palette)] = np.asarray(palette, dtype=int)

    area = grid[2:, :]
    wall_rows = [0] + [span[1] + 1 for span in row_spans]
    wall_cols = [0] + [span[1] + 1 for span in col_spans]
    area[wall_rows, :] = Color.BLUE
    area[:, wall_cols] = Color.BLUE

    room_boxes: dict[int, tuple[int, int, int, int]] = {}
    for rr, (r0, r1) in enumerate(row_spans):
        for cc, (c0, c1) in enumerate(col_spans):
            room_boxes[_room_index(rr, cc, len(widths))] = (r0, r1, c0, c1)

    for door_r, door_c in door_cells.values():
        area[door_r, door_c] = Color.BLACK

    sr0, sr1, sc0, sc1 = room_boxes[seed_room]
    seed_r = random.randint(sr0, sr1)
    seed_c = random.randint(sc0, sc1)
    area[seed_r, seed_c] = int(palette[0])
    return grid, room_boxes


def _expected_output(
    inp: np.ndarray,
    room_boxes: dict[int, tuple[int, int, int, int]],
    distances: dict[int, int],
    palette: list[int],
) -> np.ndarray:
    out = inp.copy()
    area = out[2:, :]
    for room_id, (r0, r1, c0, c1) in room_boxes.items():
        depth = distances[room_id]
        if depth >= len(palette):
            continue
        area[r0 : r1 + 1, c0 : c1 + 1] = int(palette[depth])
    return out


def _parsed_model_matches(
    inp: np.ndarray,
    room_boxes: dict[int, tuple[int, int, int, int]],
    open_edges: set[tuple[int, int]],
    distances: dict[int, int],
    palette: list[int],
    seed_room: int,
) -> bool:
    try:
        parsed = _analyze_grid(inp)
    except AssertionError:
        return False

    if parsed["palette"] != palette:
        return False
    if len(parsed["rooms"]) != len(room_boxes):
        return False

    parsed_room_for_generated: dict[int, int] = {}
    generated_for_parsed: dict[int, int] = {}
    for room_id, (r0, _r1, c0, _c1) in room_boxes.items():
        parsed_room = int(parsed["room_ids"][r0, c0])
        if parsed_room < 0 or parsed_room in generated_for_parsed:
            return False
        parsed_room_for_generated[room_id] = parsed_room
        generated_for_parsed[parsed_room] = room_id

    seed_r, seed_c = parsed["seed_cell"]
    if parsed_room_for_generated[seed_room] != int(parsed["room_ids"][seed_r, seed_c]):
        return False

    parsed_edges: set[tuple[int, int]] = set()
    for a, b in parsed["edges"]:
        ga = generated_for_parsed[int(a)]
        gb = generated_for_parsed[int(b)]
        parsed_edges.add((ga, gb) if ga < gb else (gb, ga))
    if parsed_edges != open_edges:
        return False

    for room_id, parsed_room in parsed_room_for_generated.items():
        if parsed["distances"][parsed_room] != distances[room_id]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    geometry = _sample_geometry()
    if geometry is None:
        return None
    heights, widths = geometry
    room_rows, room_cols = len(heights), len(widths)
    room_count = room_rows * room_cols

    all_edges = _all_room_adjacencies(room_rows, room_cols)
    tree_edges = _spanning_tree(room_rows, room_cols)
    remaining = [tuple(sorted((a, b))) for a, b, _kind, _rr, _cc in all_edges if tuple(sorted((a, b))) not in tree_edges]
    random.shuffle(remaining)
    extra_count = random.choices([0, 1, 2], weights=[4, 3, 1], k=1)[0]
    open_edges = set(tree_edges)
    for edge in remaining[: min(extra_count, len(remaining))]:
        open_edges.add(edge)

    graph = _build_graph(room_count, open_edges)
    candidates = _seed_candidates(room_rows, room_cols, graph)
    if not candidates:
        return None
    seed_room, distances = random.choice(candidates)
    max_depth = max(distances.values())

    palette_len = _choose_palette_length(max_depth)
    palette = random.sample(PALETTE_POOL, palette_len)

    row_spans = _room_ranges(heights)
    col_spans = _room_ranges(widths)
    door_cells = _door_cells_for_layout(row_spans, col_spans, all_edges, open_edges)
    inp, room_boxes = _render_input(heights, widths, palette, seed_room, door_cells)
    expected = _expected_output(inp, room_boxes, distances, palette)

    if not _parsed_model_matches(inp, room_boxes, open_edges, distances, palette, seed_room):
        return None
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    dark_rooms = [room_id for room_id, depth in distances.items() if depth >= palette_len]
    if palette_len <= max_depth and not dark_rooms:
        return None
    if palette_len > max_depth and dark_rooms:
        return None

    if len(set(int(actual[2 + box[0], box[2]]) for box in room_boxes.values())) < 3:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid doorway-depth flood-fill input")
