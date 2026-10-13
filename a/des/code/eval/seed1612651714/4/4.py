import itertools
import random
from collections import Counter

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _most_common_non_bg(grid: np.ndarray, bg: int) -> int:
    values = [int(v) for v in grid.ravel() if int(v) != int(bg)]
    if not values:
        return int(bg)
    return int(Counter(values).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1 + 1) // 2, (c0 + c1 + 1) // 2


def _mask_components(mask: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[dict] = []
    next_id = 0
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or int(labels[r, c]) != -1:
                continue
            stack = [(int(r), int(c))]
            labels[r, c] = int(next_id)
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and int(labels[nr, nc]) == -1:
                        labels[nr, nc] = int(next_id)
                        stack.append((int(nr), int(nc)))
            comps.append(
                {
                    "id": int(next_id),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
            next_id += 1
    return labels, comps


def _monochrome_components(grid: np.ndarray, mask: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if bool(seen[nr, nc]) or not bool(mask[nr, nc]) or int(grid[nr, nc]) != int(color):
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))
            comps.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _detect_doors(grid: np.ndarray, bg: int, wall: int) -> tuple[np.ndarray, list[dict]]:
    h, w = grid.shape
    door_mask = np.zeros((h, w), dtype=bool)
    doors: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            value = int(grid[r, c])
            if value in {int(bg), int(wall)}:
                continue
            vertical = (
                int(grid[r - 1, c]) == int(wall)
                and int(grid[r + 1, c]) == int(wall)
                and int(grid[r, c - 1]) != int(wall)
                and int(grid[r, c + 1]) != int(wall)
            )
            horizontal = (
                int(grid[r, c - 1]) == int(wall)
                and int(grid[r, c + 1]) == int(wall)
                and int(grid[r - 1, c]) != int(wall)
                and int(grid[r + 1, c]) != int(wall)
            )
            if not (vertical or horizontal):
                continue
            door_mask[r, c] = True
            doors.append({"pos": (int(r), int(c)), "orientation": "v" if vertical else "h"})
    return door_mask, doors


def _room_doors(doors: list[dict], room_labels: np.ndarray) -> dict[int, list[dict]]:
    by_room: dict[int, list[dict]] = {}
    for door in doors:
        r, c = door["pos"]
        if door["orientation"] == "v":
            left = int(room_labels[r, c - 1])
            right = int(room_labels[r, c + 1])
            if left < 0 or right < 0 or left == right:
                continue
            by_room.setdefault(left, []).append({"pos": (int(r), int(c)), "other": right})
            by_room.setdefault(right, []).append({"pos": (int(r), int(c)), "other": left})
        else:
            top = int(room_labels[r - 1, c])
            bottom = int(room_labels[r + 1, c])
            if top < 0 or bottom < 0 or top == bottom:
                continue
            by_room.setdefault(top, []).append({"pos": (int(r), int(c)), "other": bottom})
            by_room.setdefault(bottom, []).append({"pos": (int(r), int(c)), "other": top})
    for room_id in list(by_room):
        by_room[room_id].sort(key=lambda item: (int(item["pos"][1]), int(item["pos"][0])))
    return by_room


def _translate_to_room_center(
    coords: list[tuple[int, int]],
    target_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    obj_bbox = _bbox_from_coords(coords)
    obj_center = _bbox_center(obj_bbox)
    target_center = _bbox_center(target_bbox)
    dr = int(target_center[0] - obj_center[0])
    dc = int(target_center[1] - obj_center[1])
    return [(int(r + dr), int(c + dc)) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each small monochrome room object through a selected colored door into the adjacent room.

    Input structure:
    - The dominant color is the room interior background.
    - Single-cell-thick grey lines partition the grid into rectangular rooms.
    - Non-grey pixels embedded in those separator lines are doors shared by two adjacent rooms.
    - Each room contains at most one small monochrome 4-connected object of size 2-4.

    Rule:
    - Restore the separator network by treating door pixels as part of the walls.
    - For each room containing an object, sort that room's doors by column, then row.
    - If the room has one door, use it. Otherwise size 2 uses the first door, size 3 the second,
      and size 4 the third (falling back to the last available door if needed).
    - Move the whole object into the adjacent room reached through that door, preserving shape
      and aligning the object's bbox center to the target room interior center.
    - All door pixels become grey in the output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    wall = _most_common_non_bg(g, bg)

    door_mask, doors = _detect_doors(g, bg, wall)
    structure = (g == int(wall)) | door_mask
    room_labels, rooms = _mask_components(~structure)
    room_map = {int(room["id"]): room for room in rooms}
    doors_by_room = _room_doors(doors, room_labels)

    object_mask = (g != int(bg)) & (g != int(wall)) & (~door_mask)
    objects = _monochrome_components(g, object_mask)
    objects_by_room: dict[int, dict] = {}
    for obj in objects:
        room_id = int(room_labels[obj["coords"][0][0], obj["coords"][0][1]])
        if room_id >= 0:
            objects_by_room[room_id] = obj

    out = g.copy()
    out[door_mask] = int(wall)
    for obj in objects:
        for r, c in obj["coords"]:
            out[int(r), int(c)] = int(bg)

    size_to_rank = {2: 0, 3: 1, 4: 2}
    for room_id, obj in objects_by_room.items():
        room_doors = doors_by_room.get(int(room_id), [])
        if not room_doors:
            continue
        if len(room_doors) == 1:
            chosen = room_doors[0]
        else:
            rank = int(size_to_rank.get(int(obj["area"]), 0))
            chosen = room_doors[min(rank, len(room_doors) - 1)]
        target_room = room_map[int(chosen["other"])]
        translated = _translate_to_room_center(obj["coords"], target_room["bbox"])
        for rr, cc in translated:
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(room_labels[rr, cc]) == int(target_room["id"]):
                out[int(rr), int(cc)] = int(obj["color"])

    return out


SHAPES = {
    2: [
        {(0, 0), (0, 1)},
        {(0, 0), (1, 0)},
    ],
    3: [
        {(0, 0), (0, 1), (0, 2)},
        {(0, 0), (1, 0), (2, 0)},
        {(0, 0), (1, 0), (1, 1)},
        {(0, 1), (1, 0), (1, 1)},
    ],
    4: [
        {(0, 0), (0, 1), (1, 0), (1, 1)},
        {(0, 0), (1, 0), (2, 0), (2, 1)},
        {(0, 1), (1, 1), (2, 0), (2, 1)},
        {(0, 0), (0, 1), (1, 1), (2, 1)},
        {(0, 1), (1, 1), (2, 0), (2, 1)},
    ],
}


BRANCHES = {
    "single": {"sizes": (2, 3, 4), "rank": 0, "door_counts": (1,)},
    "first": {"sizes": (2,), "rank": 0, "door_counts": (2, 3, 4)},
    "second": {"sizes": (3,), "rank": 1, "door_counts": (2, 3, 4)},
    "third": {"sizes": (4,), "rank": 2, "door_counts": (3, 4)},
}


GEN_RETRIES = 600


def _normalize_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    return {(int(r - r0), int(c - c0)) for r, c in cells}


def _rotate_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_r = max(r for r, _ in cells)
    return _normalize_shape({(int(c), int(max_r - r)) for r, c in cells})


def _flip_shape(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    max_c = max(c for _, c in cells)
    return _normalize_shape({(int(r), int(max_c - c)) for r, c in cells})


def _random_shape(size: int) -> set[tuple[int, int]]:
    cells = set(random.choice(SHAPES[int(size)]))
    for _ in range(random.randrange(4)):
        cells = _rotate_shape(cells)
    if random.random() < 0.5:
        cells = _flip_shape(cells)
    return _normalize_shape(cells)


def _shape_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _build_layout(
    room_rows: int,
    room_cols: int,
    row_heights: list[int],
    col_widths: list[int],
    bg: int,
    wall: int,
) -> tuple[np.ndarray, list[dict], dict[frozenset[int], dict], dict[int, list[frozenset[int]]]]:
    row_lines = [0]
    col_lines = [0]
    pos = 0
    for height in row_heights:
        pos += int(height) + 1
        row_lines.append(int(pos))
    pos = 0
    for width in col_widths:
        pos += int(width) + 1
        col_lines.append(int(pos))

    grid = np.full((row_lines[-1] + 1, col_lines[-1] + 1), int(bg), dtype=int)
    grid[row_lines, :] = int(wall)
    grid[:, col_lines] = int(wall)

    rooms: list[dict] = []
    room_neighbors: dict[int, list[frozenset[int]]] = {}

    def room_id(i: int, j: int) -> int:
        return int(i * room_cols + j)

    for i in range(room_rows):
        for j in range(room_cols):
            rid = room_id(i, j)
            bbox = (
                int(row_lines[i] + 1),
                int(col_lines[j] + 1),
                int(row_lines[i + 1] - 1),
                int(col_lines[j + 1] - 1),
            )
            rooms.append({"id": rid, "rc": (int(i), int(j)), "bbox": bbox})
            room_neighbors[rid] = []

    edges: dict[frozenset[int], dict] = {}
    for i in range(room_rows):
        for j in range(room_cols - 1):
            a = room_id(i, j)
            b = room_id(i, j + 1)
            edge_id = frozenset((int(a), int(b)))
            cells = [(int(r), int(col_lines[j + 1])) for r in range(int(row_lines[i] + 1), int(row_lines[i + 1]))]
            edges[edge_id] = {"rooms": (int(a), int(b)), "orientation": "v", "cells": cells}
            room_neighbors[a].append(edge_id)
            room_neighbors[b].append(edge_id)
    for i in range(room_rows - 1):
        for j in range(room_cols):
            a = room_id(i, j)
            b = room_id(i + 1, j)
            edge_id = frozenset((int(a), int(b)))
            cells = [(int(row_lines[i + 1]), int(c)) for c in range(int(col_lines[j] + 1), int(col_lines[j + 1]))]
            edges[edge_id] = {"rooms": (int(a), int(b)), "orientation": "h", "cells": cells}
            room_neighbors[a].append(edge_id)
            room_neighbors[b].append(edge_id)
    return grid, rooms, edges, room_neighbors


def _other_room(edge: dict, room_id: int) -> int:
    a, b = edge["rooms"]
    return int(b if int(a) == int(room_id) else a)


def _sample_branch_sequence() -> list[str]:
    count = random.randint(1, 3)
    primary = random.choice(tuple(BRANCHES))
    branches = [primary]
    while len(branches) < count:
        branches.append(random.choice(tuple(BRANCHES)))
    random.shuffle(branches)
    branches.sort(key=lambda name: max(BRANCHES[name]["door_counts"]), reverse=True)
    return branches


def _choose_source_configs(
    rooms: list[dict],
    edges: dict[frozenset[int], dict],
    room_neighbors: dict[int, list[frozenset[int]]],
) -> list[dict] | None:
    selected_sources: set[int] = set()
    reserved_targets: set[int] = set()
    blocked_sources: set[int] = set()
    configs: list[dict] = []

    for branch in _sample_branch_sequence():
        spec = BRANCHES[branch]
        size = int(random.choice(spec["sizes"]))
        rank = int(spec["rank"])
        candidates = []
        for room in rooms:
            rid = int(room["id"])
            if rid in blocked_sources or rid in reserved_targets:
                continue
            degree = len(room_neighbors[rid])
            allowed_counts = [k for k in spec["door_counts"] if int(k) <= degree]
            if allowed_counts:
                candidates.append((rid, allowed_counts))
        random.shuffle(candidates)

        chosen_config = None
        for rid, allowed_counts in candidates:
            counts = list(allowed_counts)
            random.shuffle(counts)
            incident_edges = list(room_neighbors[rid])
            for door_count in counts:
                subsets = list(itertools.combinations(incident_edges, int(door_count)))
                random.shuffle(subsets)
                for subset in subsets:
                    sampled_positions = {edge_id: random.choice(edges[edge_id]["cells"]) for edge_id in subset}
                    ordered = sorted(subset, key=lambda edge_id: (sampled_positions[edge_id][1], sampled_positions[edge_id][0]))
                    chosen_edge = ordered[min(rank, len(ordered) - 1)]
                    target = _other_room(edges[chosen_edge], rid)
                    if target in reserved_targets or target in selected_sources:
                        continue
                    chosen_config = {
                        "source": int(rid),
                        "target": int(target),
                        "branch": branch,
                        "size": int(size),
                        "doors": {
                            edge_id: {"pos": tuple(map(int, sampled_positions[edge_id]))}
                            for edge_id in subset
                        },
                    }
                    break
                if chosen_config is not None:
                    break
            if chosen_config is not None:
                break

        if chosen_config is None:
            return None

        source = int(chosen_config["source"])
        target = int(chosen_config["target"])
        selected_sources.add(source)
        reserved_targets.add(target)
        blocked_sources.add(source)
        blocked_sources.add(target)
        for edge_id in room_neighbors[source]:
            blocked_sources.add(_other_room(edges[edge_id], source))
        configs.append(chosen_config)

    return configs


def _paint_doors(
    grid: np.ndarray,
    configs: list[dict],
    edges: dict[frozenset[int], dict],
    door_palette: list[int],
) -> dict[frozenset[int], tuple[int, int]]:
    placed: dict[frozenset[int], tuple[int, int]] = {}
    for config in configs:
        for edge_id, meta in config["doors"].items():
            if edge_id in placed:
                continue
            r, c = meta["pos"]
            grid[int(r), int(c)] = int(random.choice(door_palette))
            placed[edge_id] = (int(r), int(c))
    return placed


def _room_bbox(rooms: list[dict], room_id: int) -> tuple[int, int, int, int]:
    return next(room["bbox"] for room in rooms if int(room["id"]) == int(room_id))


def _room_can_host_shape(shape: set[tuple[int, int]], bbox: tuple[int, int, int, int], margin: int) -> bool:
    r0, c0, r1, c1 = bbox
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    sh, sw = _shape_dims(shape)
    return sh <= height - 2 * margin and sw <= width - 2 * margin


def _place_shape_in_room(
    grid: np.ndarray,
    shape: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    color: int,
    bg: int,
) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    sh, sw = _shape_dims(shape)
    top_choices = list(range(int(r0 + 1), int(r1 - sh + 1)))
    left_choices = list(range(int(c0 + 1), int(c1 - sw + 1)))
    if not top_choices or not left_choices:
        return None
    random.shuffle(top_choices)
    random.shuffle(left_choices)
    for top in top_choices:
        for left in left_choices:
            coords = [(int(top + r), int(left + c)) for r, c in shape]
            if any(int(grid[r, c]) != int(bg) for r, c in coords):
                continue
            for r, c in coords:
                grid[int(r), int(c)] = int(color)
            return coords
    return None


def _build_expected_output(
    inp: np.ndarray,
    wall: int,
    door_positions: list[tuple[int, int]],
    placements: list[dict],
    rooms: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for r, c in door_positions:
        out[int(r), int(c)] = int(wall)
    for item in placements:
        for r, c in item["coords"]:
            out[int(r), int(c)] = int(item["bg"])
    for item in placements:
        target_bbox = _room_bbox(rooms, item["target"])
        translated = _translate_to_room_center(item["coords"], target_bbox)
        for r, c in translated:
            out[int(r), int(c)] = int(item["color"])
    return out


def _generated_grid_valid(grid: np.ndarray, bg: int, wall: int) -> bool:
    counts = Counter(int(v) for v in grid.ravel())
    non_bg_counts = {color: count for color, count in counts.items() if int(color) != int(bg)}
    return counts[int(bg)] > max(non_bg_counts.values(), default=0) and int(wall) == _most_common_non_bg(grid, bg)


def _gen_attempt() -> np.ndarray | None:
    colors = list(range(10))
    bg, wall = random.sample(colors, 2)
    room_rows = random.randint(3, 4)
    room_cols = random.randint(3, 4)
    row_heights = [random.randint(5, 8) for _ in range(room_rows)]
    col_widths = [random.randint(5, 8) for _ in range(room_cols)]

    inp, rooms, edges, room_neighbors = _build_layout(room_rows, room_cols, row_heights, col_widths, bg, wall)
    configs = _choose_source_configs(rooms, edges, room_neighbors)
    if configs is None:
        return None

    door_palette_pool = [c for c in colors if c not in {int(bg), int(wall)}]
    if len(door_palette_pool) < 3:
        return None
    door_palette = random.sample(door_palette_pool, 3)
    object_color_pool = [c for c in colors if c not in {int(bg), int(wall)} and c not in set(door_palette)]
    if len(object_color_pool) < len(configs):
        return None
    object_colors = random.sample(object_color_pool, len(configs))

    door_locations = _paint_doors(inp, configs, edges, door_palette)

    placements: list[dict] = []
    for config, color in zip(configs, object_colors):
        shape = _random_shape(int(config["size"]))
        source_bbox = _room_bbox(rooms, int(config["source"]))
        target_bbox = _room_bbox(rooms, int(config["target"]))
        if not _room_can_host_shape(shape, source_bbox, margin=1):
            return None
        if not _room_can_host_shape(shape, target_bbox, margin=0):
            return None
        coords = _place_shape_in_room(inp, shape, source_bbox, int(color), int(bg))
        if coords is None:
            return None
        placements.append(
            {
                "coords": coords,
                "color": int(color),
                "source": int(config["source"]),
                "target": int(config["target"]),
                "bg": int(bg),
            }
        )

    if not _generated_grid_valid(inp, bg, wall):
        return None

    expected = _build_expected_output(inp, wall, list(door_locations.values()), placements, rooms)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-door transport puzzle")
