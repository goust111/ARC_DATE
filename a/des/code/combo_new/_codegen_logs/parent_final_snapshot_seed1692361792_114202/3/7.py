from __future__ import annotations

from collections import deque
import heapq
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 0
MAX_GENERATION_ATTEMPTS = 400
GRID_MIN = 18
GRID_MAX = 28
ROOM_COUNT_CHOICES = (2, 3, 4)
SEED_COUNT_CHOICES = (2, 3, 4)
SEED_SIZE_OPTIONS = (1, 2, 3, 4, 5)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(int(r), int(c))])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _bg_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return max(set(corners), key=corners.count)


def _detect_wall_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.unique(grid) if int(v) != int(bg)]
    best_color = None
    best_score = (-1, -1)
    for color in colors:
        room_count = 0
        room_area = 0
        for comp in _components(grid != int(color)):
            if _touches_border(comp, *grid.shape):
                continue
            if not any(int(grid[r, c]) == int(bg) for r, c in comp):
                continue
            room_count += 1
            room_area += len(comp)
        score = (int(room_count), int(room_area))
        if score > best_score:
            best_score = score
            best_color = int(color)
    assert best_color is not None and best_score[0] > 0, "failed to identify the wall color"
    return int(best_color)


def _room_components(grid: np.ndarray, bg: int, wall_color: int) -> list[list[tuple[int, int]]]:
    rooms: list[list[tuple[int, int]]] = []
    for comp in _components(grid != int(wall_color)):
        if _touches_border(comp, *grid.shape):
            continue
        if any(int(grid[r, c]) == int(bg) for r, c in comp):
            rooms.append(comp)
    return rooms


def _seeds_in_room(grid: np.ndarray, room: list[tuple[int, int]], bg: int, wall_color: int) -> list[dict]:
    h, w = grid.shape
    room_mask = np.zeros((h, w), dtype=bool)
    for r, c in room:
        room_mask[r, c] = True

    seen = np.zeros((h, w), dtype=bool)
    seeds: list[dict] = []
    for r, c in room:
        color = int(grid[r, c])
        if color in (int(bg), int(wall_color)) or bool(seen[r, c]):
            continue
        seen[r, c] = True
        q = deque([(int(r), int(c))])
        cells: list[tuple[int, int]] = []
        while q:
            rr, cc = q.popleft()
            cells.append((int(rr), int(cc)))
            for dr, dc in DIR4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if (
                    0 <= nr < h
                    and 0 <= nc < w
                    and bool(room_mask[nr, nc])
                    and not bool(seen[nr, nc])
                    and int(grid[nr, nc]) == color
                ):
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
        cells.sort()
        seeds.append(
            {
                "color": int(color),
                "cells": cells,
                "size": int(len(cells)),
                "anchor": cells[0],
            }
        )
    return seeds


def _priority_order(seeds: list[dict]) -> dict[int, int]:
    order = sorted(range(len(seeds)), key=lambda idx: (-int(seeds[idx]["size"]), seeds[idx]["anchor"]))
    return {int(seed_idx): int(rank) for rank, seed_idx in enumerate(order)}


def _fill_room(grid: np.ndarray, out: np.ndarray, room: list[tuple[int, int]], seeds: list[dict]) -> None:
    h, w = grid.shape
    room_mask = np.zeros((h, w), dtype=bool)
    for r, c in room:
        room_mask[r, c] = True

    priority = _priority_order(seeds)
    inf = 10**9
    best_dist = np.full((h, w), inf, dtype=int)
    best_rank = np.full((h, w), inf, dtype=int)
    owner = np.full((h, w), -1, dtype=int)
    heap: list[tuple[int, int, int, int, int]] = []

    for seed_idx, seed in enumerate(seeds):
        rank = int(priority[seed_idx])
        for r, c in seed["cells"]:
            best_dist[r, c] = 0
            best_rank[r, c] = rank
            owner[r, c] = int(seed_idx)
            heapq.heappush(heap, (0, rank, int(seed_idx), int(r), int(c)))

    while heap:
        dist, rank, seed_idx, r, c = heapq.heappop(heap)
        if dist != int(best_dist[r, c]) or rank != int(best_rank[r, c]) or seed_idx != int(owner[r, c]):
            continue
        for dr, dc in DIR4:
            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w and bool(room_mask[nr, nc])):
                continue
            cand_dist = int(dist + 1)
            if cand_dist < int(best_dist[nr, nc]) or (
                cand_dist == int(best_dist[nr, nc]) and rank < int(best_rank[nr, nc])
            ):
                best_dist[nr, nc] = cand_dist
                best_rank[nr, nc] = rank
                owner[nr, nc] = int(seed_idx)
                heapq.heappush(heap, (cand_dist, rank, int(seed_idx), int(nr), int(nc)))

    for r, c in room:
        assert int(owner[r, c]) >= 0, "every room cell must be claimed by a seed"
        out[r, c] = int(seeds[int(owner[r, c])]["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed room by simultaneous seed growth, breaking same-step collisions by seed area.

    Input structure:
    - The background is black.
    - One non-black color forms closed walls that split the grid into multiple disconnected rooms.
    - Inside each room are 2 to 4 differently colored seed blobs, each a small 4-connected component.

    Rule:
    - Keep the wall pixels unchanged.
    - Process each room independently: every seed expands one step at a time in the four cardinal directions across that room.
    - A room pixel belongs to the seed that reaches it in the fewest expansion steps; if multiple seeds reach it in the same step, the larger original seed blob wins that tie.
    - Continue until every non-wall pixel inside the room has been assigned to one seed color.

    Key insight:
    - The puzzle combines three layers: detect enclosed rooms, measure each seed's initial area to get a priority order, then apply synchronous multi-source growth with area-based tie-breaking inside each room.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _bg_color(g)
    wall_color = _detect_wall_color(g, bg)
    rooms = _room_components(g, bg, wall_color)
    assert rooms, "expected at least one enclosed room"

    out = g.copy()
    for room in rooms:
        seeds = _seeds_in_room(g, room, bg, wall_color)
        assert len(seeds) >= 2, "each room must contain at least two seeds"
        _fill_room(g, out, room, seeds)
    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return sorted((int(r - min_r), int(c - min_c)) for r, c in cells)


def _transform_shape(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = np.array(cells, dtype=int)
    if random.random() < 0.5:
        pts[:, 1] *= -1
    for _ in range(random.randint(0, 3)):
        pts = np.stack([pts[:, 1], -pts[:, 0]], axis=1)
    return _normalize_shape({(int(r), int(c)) for r, c in pts})


def _random_polyomino(size: int) -> list[tuple[int, int]]:
    cells = {(0, 0)}
    while len(cells) < int(size):
        frontier = set()
        for r, c in cells:
            for dr, dc in DIR4:
                cand = (int(r + dr), int(c + dc))
                if cand not in cells:
                    frontier.add(cand)
        cells.add(random.choice(sorted(frontier)))
    return _transform_shape(_normalize_shape(cells))


def _shape_bbox(shape: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _can_place(shape_cells: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in shape_cells:
        for rr in range(int(r - 1), int(r + 2)):
            for cc in range(int(c - 1), int(c + 2)):
                if (int(rr), int(cc)) in occupied:
                    return False
    return True


def _room_has_priority_tie(inner_h: int, inner_w: int, seeds: list[dict]) -> bool:
    occupied = {cell for seed in seeds for cell in seed["cells"]}
    for r in range(inner_h):
        for c in range(inner_w):
            if (int(r), int(c)) in occupied:
                continue
            dists = [
                min(abs(int(r - sr)) + abs(int(c - sc)) for sr, sc in seed["cells"])
                for seed in seeds
            ]
            best = min(dists)
            if sum(dist == best for dist in dists) >= 2:
                return True
    return False


def _sample_inner_size(seed_sizes: list[int], sector_h: int, sector_w: int) -> tuple[int, int] | None:
    needed_area = sum(seed_sizes) + 4 * len(seed_sizes)
    min_side = max(4, len(seed_sizes) + 2)
    choices = []
    for inner_h in range(min_side, min(10, sector_h - 2) + 1):
        for inner_w in range(min_side, min(10, sector_w - 2) + 1):
            if inner_h * inner_w >= needed_area:
                choices.append((int(inner_h), int(inner_w)))
    if not choices:
        return None
    return random.choice(choices)


def _build_room_payload(inner_h: int, inner_w: int, wall_color: int) -> tuple[np.ndarray, bool] | None:
    seed_count = int(random.choice(SEED_COUNT_CHOICES))
    seed_sizes = random.sample(list(SEED_SIZE_OPTIONS), seed_count)
    random.shuffle(seed_sizes)

    local_colors = random.sample([color for color in range(1, 10) if color != int(wall_color)], seed_count)
    occupied: set[tuple[int, int]] = set()
    seeds: list[dict] = []

    for size, color in zip(seed_sizes, local_colors, strict=False):
        placed = False
        for _ in range(120):
            shape = _random_polyomino(int(size))
            shape_h, shape_w = _shape_bbox(shape)
            if shape_h > inner_h or shape_w > inner_w:
                continue
            r0 = random.randint(0, inner_h - shape_h)
            c0 = random.randint(0, inner_w - shape_w)
            cells = [(int(r0 + r), int(c0 + c)) for r, c in shape]
            if not _can_place(cells, occupied):
                continue
            for cell in cells:
                occupied.add(cell)
            cells.sort()
            seeds.append({"color": int(color), "cells": cells, "size": int(size)})
            placed = True
            break
        if not placed:
            return None

    if not _room_has_priority_tie(inner_h, inner_w, seeds):
        return None

    room = np.full((inner_h + 2, inner_w + 2), int(wall_color), dtype=int)
    room[1:-1, 1:-1] = int(BG)
    for seed in seeds:
        for r, c in seed["cells"]:
            room[int(r + 1), int(c + 1)] = int(seed["color"])
    return room, True


def _sector_boxes(height: int, width: int) -> list[tuple[int, int, int, int]]:
    split_r = random.randint(8, height - 9)
    split_c = random.randint(8, width - 9)
    return [
        (1, split_r - 1, 1, split_c - 1),
        (1, split_r - 1, split_c + 1, width - 2),
        (split_r + 1, height - 2, 1, split_c - 1),
        (split_r + 1, height - 2, split_c + 1, width - 2),
    ]


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    grid = np.full((height, width), int(BG), dtype=int)

    wall_color = random.randint(1, 9)
    sectors = _sector_boxes(height, width)
    room_count = int(random.choice(ROOM_COUNT_CHOICES))
    chosen_sectors = random.sample(sectors, room_count)
    tie_count = 0

    for r0, r1, c0, c1 in chosen_sectors:
        sector_h = int(r1 - r0 + 1)
        sector_w = int(c1 - c0 + 1)
        room = None
        for _ in range(80):
            seed_count = int(random.choice(SEED_COUNT_CHOICES))
            seed_sizes = random.sample(list(SEED_SIZE_OPTIONS), seed_count)
            dims = _sample_inner_size(seed_sizes, sector_h, sector_w)
            if dims is None:
                continue
            inner_h, inner_w = dims
            payload = _build_room_payload(int(inner_h), int(inner_w), int(wall_color))
            if payload is None:
                continue
            room, has_tie = payload
            tie_count += int(has_tie)
            break
        if room is None:
            return None

        room_h, room_w = map(int, room.shape)
        top = random.randint(int(r0), int(r1 - room_h + 1))
        left = random.randint(int(c0), int(c1 - room_w + 1))
        grid[top : top + room_h, left : left + room_w] = room

    if tie_count == 0:
        return None

    try:
        detected_wall = _detect_wall_color(grid, BG)
        rooms = _room_components(grid, BG, detected_wall)
        if int(detected_wall) != int(wall_color) or len(rooms) != room_count:
            return None
        out = main(grid)
        if out.shape != grid.shape or np.array_equal(out, grid):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-growth puzzle input")
