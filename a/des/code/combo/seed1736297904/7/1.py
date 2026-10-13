from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
FILL_COLORS = (1, 2, 3, 6, 7, 8, 9)

GRID_LAYOUTS = ((2, 2), (2, 3), (3, 2))
ROOM_SIZE_MIN = 3
ROOM_SIZE_MAX = 6
MARGIN_MIN = 1
MARGIN_MAX = 3
CONFLICT_PROB = 0.55
THREE_WAY_CONFLICT_PROB = 0.25
GEN_RETRIES = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                if rr > 0 and mask[rr - 1, cc] and not seen[rr - 1, cc]:
                    seen[rr - 1, cc] = True
                    stack.append((rr - 1, cc))
                if rr + 1 < h and mask[rr + 1, cc] and not seen[rr + 1, cc]:
                    seen[rr + 1, cc] = True
                    stack.append((rr + 1, cc))
                if cc > 0 and mask[rr, cc - 1] and not seen[rr, cc - 1]:
                    seen[rr, cc - 1] = True
                    stack.append((rr, cc - 1))
                if cc + 1 < w and mask[rr, cc + 1] and not seen[rr, cc + 1]:
                    seen[rr, cc + 1] = True
                    stack.append((rr, cc + 1))
            comps.append(pts)
    return comps


def _marker_direction(marker_cells: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> str | None:
    r0, c0, r1, c1 = bbox
    rows = sorted({r for r, _ in marker_cells})
    cols = sorted({c for _, c in marker_cells})

    if len(rows) == 1 and len(cols) == 2 and cols[1] == cols[0] + 1:
        left_gap = cols[0] - c0
        right_gap = c1 - cols[1]
        if left_gap == right_gap:
            return None
        return "left" if left_gap < right_gap else "right"

    if len(rows) == 2 and len(cols) == 1 and rows[1] == rows[0] + 1:
        top_gap = rows[0] - r0
        bottom_gap = r1 - rows[1]
        if top_gap == bottom_gap:
            return None
        return "up" if top_gap < bottom_gap else "down"

    return None


def _extract_regions(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = (g != BLACK) & (g != GREY)
    regions: list[dict] = []
    cell_to_region: dict[tuple[int, int], int] = {}

    for pts in _components(mask):
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if not np.all(g[r0 - 1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(g[r1 + 1, c0 : c1 + 1] == GREY):
            continue
        if not np.all(g[r0 : r1 + 1, c0 - 1] == GREY):
            continue
        if not np.all(g[r0 : r1 + 1, c1 + 1] == GREY):
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any((sub == BLACK) | (sub == GREY)):
            continue

        marker_local = np.argwhere(sub == YELLOW)
        if len(marker_local) != 2:
            continue
        marker_cells = [(r0 + int(r), c0 + int(c)) for r, c in marker_local]
        if abs(marker_cells[0][0] - marker_cells[1][0]) + abs(marker_cells[0][1] - marker_cells[1][1]) != 1:
            continue

        fill_colors = {int(v) for v in np.unique(sub) if int(v) != YELLOW}
        if len(fill_colors) != 1:
            continue

        direction = _marker_direction(marker_cells, (r0, c0, r1, c1))
        if direction is None:
            continue

        region_id = len(regions)
        region = {
            "id": region_id,
            "bbox": (r0, c0, r1, c1),
            "area": int(sub.size),
            "fill_color": int(next(iter(fill_colors))),
            "marker_cells": marker_cells,
            "direction": direction,
        }
        regions.append(region)
        for cell in pts:
            cell_to_region[cell] = region_id

    return regions, cell_to_region


def _target_region(
    grid: np.ndarray,
    region: dict,
    cell_to_region: dict[tuple[int, int], int],
) -> int | None:
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = region["bbox"]
    direction = region["direction"]
    counts: Counter[int] = Counter()

    if direction == "left":
        wall_c = c0 - 1
        src_c = c0 - 2
        if src_c < 0:
            return None
        for r in range(r0, r1 + 1):
            if int(g[r, wall_c]) == GREY and (r, src_c) in cell_to_region:
                counts[cell_to_region[(r, src_c)]] += 1
    elif direction == "right":
        wall_c = c1 + 1
        src_c = c1 + 2
        if src_c >= g.shape[1]:
            return None
        for r in range(r0, r1 + 1):
            if int(g[r, wall_c]) == GREY and (r, src_c) in cell_to_region:
                counts[cell_to_region[(r, src_c)]] += 1
    elif direction == "up":
        wall_r = r0 - 1
        src_r = r0 - 2
        if src_r < 0:
            return None
        for c in range(c0, c1 + 1):
            if int(g[wall_r, c]) == GREY and (src_r, c) in cell_to_region:
                counts[cell_to_region[(src_r, c)]] += 1
    else:
        wall_r = r1 + 1
        src_r = r1 + 2
        if src_r >= g.shape[0]:
            return None
        for c in range(c0, c1 + 1):
            if int(g[wall_r, c]) == GREY and (src_r, c) in cell_to_region:
                counts[cell_to_region[(src_r, c)]] += 1

    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _incoming_pushes(grid: np.ndarray) -> tuple[list[dict], dict[int, list[tuple[int, int, int]]]]:
    regions, cell_to_region = _extract_regions(grid)
    incoming: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for region in regions:
        target = _target_region(grid, region, cell_to_region)
        if target is None or target == region["id"]:
            continue
        incoming[target].append((int(region["area"]), int(region["id"]), int(region["fill_color"])))
    return regions, incoming


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    regions, incoming = _incoming_pushes(g)

    for target_id, pushes in incoming.items():
        winning_area, winning_source, winning_color = max(pushes, key=lambda item: (item[0], -item[1]))
        _ = winning_area, winning_source
        r0, c0, r1, c1 = regions[target_id]["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[g[r0 : r1 + 1, c0 : c1 + 1] != YELLOW] = winning_color

    return out


def _room_index(row: int, col: int, ncols: int) -> int:
    return row * ncols + col


def _direction_between(src: dict, dst: dict) -> str:
    if dst["row"] == src["row"] and dst["col"] == src["col"] - 1:
        return "left"
    if dst["row"] == src["row"] and dst["col"] == src["col"] + 1:
        return "right"
    if dst["col"] == src["col"] and dst["row"] == src["row"] - 1:
        return "up"
    return "down"


def _neighbor_map(rooms: list[dict], nrows: int, ncols: int) -> list[dict[str, int]]:
    neighbors: list[dict[str, int]] = []
    for room in rooms:
        row = int(room["row"])
        col = int(room["col"])
        entry: dict[str, int] = {}
        if col > 0:
            entry["left"] = _room_index(row, col - 1, ncols)
        if col + 1 < ncols:
            entry["right"] = _room_index(row, col + 1, ncols)
        if row > 0:
            entry["up"] = _room_index(row - 1, col, ncols)
        if row + 1 < nrows:
            entry["down"] = _room_index(row + 1, col, ncols)
        neighbors.append(entry)
    return neighbors


def _choose_directions(
    rng: random.Random,
    rooms: list[dict],
    neighbors: list[dict[str, int]],
) -> tuple[dict[int, str], bool] | None:
    directions: dict[int, str] = {}
    force_conflict = rng.random() < CONFLICT_PROB

    if force_conflict:
        candidates: list[tuple[int, list[int]]] = []
        for target in rooms:
            inbound = []
            for source in rooms:
                if target["id"] in neighbors[source["id"]].values():
                    inbound.append(int(source["id"]))
            if len({int(rooms[sid]["area"]) for sid in inbound}) >= 2:
                candidates.append((int(target["id"]), inbound))
        if not candidates:
            return None

        target_id, inbound = rng.choice(candidates)
        area_to_sources: dict[int, list[int]] = defaultdict(list)
        for sid in inbound:
            area_to_sources[int(rooms[sid]["area"])].append(sid)
        distinct_areas = list(area_to_sources)
        rng.shuffle(distinct_areas)
        take = 3 if len(distinct_areas) >= 3 and rng.random() < THREE_WAY_CONFLICT_PROB else 2
        chosen_sources = [rng.choice(area_to_sources[area]) for area in distinct_areas[:take]]
        for sid in chosen_sources:
            directions[sid] = _direction_between(rooms[sid], rooms[target_id])

    for room in rooms:
        sid = int(room["id"])
        if sid in directions:
            continue
        directions[sid] = rng.choice(list(neighbors[sid].keys()))

    return directions, force_conflict


def _incoming_from_assignments(
    rooms: list[dict],
    neighbors: list[dict[str, int]],
    directions: dict[int, str],
) -> dict[int, list[int]]:
    incoming: dict[int, list[int]] = defaultdict(list)
    for sid, direction in directions.items():
        incoming[neighbors[sid][direction]].append(sid)
    return incoming


def _biased_offsets(span: int, toward_low_side: bool) -> list[int]:
    offsets = []
    for start in range(span - 1):
        low_gap = start
        high_gap = span - (start + 2)
        if toward_low_side and low_gap < high_gap:
            offsets.append(start)
        if not toward_low_side and high_gap < low_gap:
            offsets.append(start)
    return offsets


def _place_marker(rng: random.Random, grid: np.ndarray, room: dict, direction: str) -> None:
    r0, c0, r1, c1 = room["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1

    if direction in ("left", "right"):
        row = r0 + rng.randrange(height)
        choices = _biased_offsets(width, toward_low_side=(direction == "left"))
        col = c0 + rng.choice(choices)
        grid[row, col : col + 2] = YELLOW
    else:
        col = c0 + rng.randrange(width)
        choices = _biased_offsets(height, toward_low_side=(direction == "up"))
        row = r0 + rng.choice(choices)
        grid[row : row + 2, col] = YELLOW


def _build_rooms(
    rng: random.Random,
    nrows: int,
    ncols: int,
    top: int,
    left: int,
    heights: list[int],
    widths: list[int],
) -> list[dict]:
    rooms: list[dict] = []
    cursor_r = top + 1
    for row in range(nrows):
        cursor_c = left + 1
        for col in range(ncols):
            height = heights[row]
            width = widths[col]
            r0 = cursor_r
            c0 = cursor_c
            room_id = _room_index(row, col, ncols)
            rooms.append(
                {
                    "id": room_id,
                    "row": row,
                    "col": col,
                    "bbox": (r0, c0, r0 + height - 1, c0 + width - 1),
                    "area": height * width,
                }
            )
            cursor_c += width + 1
        cursor_r += heights[row] + 1
    return rooms


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    nrows, ncols = rng.choice(GRID_LAYOUTS)
    room_count = nrows * ncols

    heights = [rng.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(nrows)]
    widths = [rng.randint(ROOM_SIZE_MIN, ROOM_SIZE_MAX) for _ in range(ncols)]
    margin_top = rng.randint(MARGIN_MIN, MARGIN_MAX)
    margin_bottom = rng.randint(MARGIN_MIN, MARGIN_MAX)
    margin_left = rng.randint(MARGIN_MIN, MARGIN_MAX)
    margin_right = rng.randint(MARGIN_MIN, MARGIN_MAX)

    frame_height = sum(heights) + nrows + 1
    frame_width = sum(widths) + ncols + 1
    total_height = frame_height + margin_top + margin_bottom
    total_width = frame_width + margin_left + margin_right

    rooms = _build_rooms(rng, nrows, ncols, margin_top, margin_left, heights, widths)
    neighbors = _neighbor_map(rooms, nrows, ncols)
    chosen = _choose_directions(rng, rooms, neighbors)
    if chosen is None:
        return None
    directions, force_conflict = chosen
    incoming = _incoming_from_assignments(rooms, neighbors, directions)

    if force_conflict and not any(len(srcs) >= 2 for srcs in incoming.values()):
        return None
    for srcs in incoming.values():
        areas = sorted((int(rooms[sid]["area"]) for sid in srcs), reverse=True)
        if len(areas) >= 2 and areas[0] == areas[1]:
            return None

    colors = rng.sample(list(FILL_COLORS), room_count)

    grid = np.full((total_height, total_width), BLACK, dtype=int)
    grid[margin_top : margin_top + frame_height, margin_left : margin_left + frame_width] = GREY

    for room, color in zip(rooms, colors, strict=False):
        r0, c0, r1, c1 = room["bbox"]
        room["fill_color"] = color
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    for room in rooms:
        _place_marker(rng, grid, room, directions[room["id"]])

    parsed_regions, parsed_incoming = _incoming_pushes(grid)
    if len(parsed_regions) != room_count:
        return None
    if any(len(pushes) >= 2 and pushes[0][0] == pushes[1][0] for pushes in (sorted(v, reverse=True) for v in parsed_incoming.values())):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
