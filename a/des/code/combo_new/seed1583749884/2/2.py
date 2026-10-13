import random
from collections import deque

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS = ((-1, -1), (-1, 1), (1, -1), (1, 1))

GRID_H_RANGE = (12, 22)
GRID_W_RANGE = (12, 22)
SPRITE_COUNT_RANGE = (1, 3)
EXTRA_OBSTACLE_RANGE = (0, 2)
MIN_SEGMENT_LEN = 1
MAX_SEGMENT_LEN = 5
MAX_BOUNCES = 3
RAY_RETRIES = 160
GENERATE_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _reflect(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    if dr == dc:
        return -dr, dc
    return dr, -dc


def _component4(grid: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> list[tuple[int, int]]:
    color = int(grid[start])
    q: deque[tuple[int, int]] = deque([start])
    seen[start] = True
    coords: list[tuple[int, int]] = []
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))
    return coords


def _parse_sprite(coords: list[tuple[int, int]], color: int) -> dict[str, object] | None:
    if len(coords) != 3:
        return None
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) != 2 or len(cols) != 2:
        return None

    cell_set = set(coords)
    box = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    if len(box - cell_set) != 1:
        return None

    missing = next(iter(box - cell_set))
    apex = (rows[0] + rows[1] - missing[0], cols[0] + cols[1] - missing[1])
    direction = (apex[0] - missing[0], apex[1] - missing[1])
    assert direction in DIRS, "sprite direction must be diagonal"

    expected = {apex, (apex[0] - direction[0], apex[1]), (apex[0], apex[1] - direction[1])}
    assert expected == cell_set, "invalid L sprite geometry"
    return {
        "color": color,
        "coords": tuple(sorted(coords)),
        "apex": apex,
        "direction": direction,
        "box": tuple(sorted(box)),
    }


def _extract_scene(grid: np.ndarray) -> tuple[list[dict[str, object]], dict[tuple[int, int], int]]:
    seen = np.zeros_like(grid, dtype=bool)
    sprites: list[dict[str, object]] = []
    obstacles: dict[tuple[int, int], int] = {}

    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if seen[r, c] or int(grid[r, c]) == BG:
                continue
            coords = _component4(grid, (r, c), seen)
            color = int(grid[r, c])
            if len(coords) == 1:
                obstacles[coords[0]] = color
                continue
            sprite = _parse_sprite(coords, color)
            assert sprite is not None, "foreground objects must be single-pixel obstacles or 3-pixel L sprites"
            sprites.append(sprite)

    assert sprites, "expected at least one sprite"
    return sprites, obstacles


def _trace_ray(out: np.ndarray, grid: np.ndarray, obstacles: dict[tuple[int, int], int], sprite: dict[str, object]) -> None:
    current = tuple(sprite["apex"])
    direction = tuple(sprite["direction"])
    color = int(sprite["color"])
    seen_states: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for _ in range(grid.size * 8):
        state = (current, direction, color)
        assert state not in seen_states, "ray path must not enter an obstacle loop"
        seen_states.add(state)

        nr = current[0] + direction[0]
        nc = current[1] + direction[1]
        if not _in_bounds(grid.shape, nr, nc):
            return

        if (nr, nc) in obstacles:
            current = (nr, nc)
            color = int(obstacles[current])
            direction = _reflect(direction)
            continue

        assert int(grid[nr, nc]) == BG, "ray must travel through background until an obstacle"
        out[nr, nc] = color
        current = (nr, nc)

    raise AssertionError("ray tracing exceeded safety limit")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend diagonal rays from L sprites, bouncing off single-pixel obstacles and inheriting each obstacle's color.

    Input structure:
    - The background is black.
    - Foreground objects are either monochrome 3-pixel L sprites occupying a 2x2 box with one missing corner, or isolated single-pixel colored obstacles.
    - The missing corner of each L sprite determines the initial 45-degree emission direction, and the diagonally opposite occupied corner is the ray apex.

    Rule:
    - Keep every sprite and every obstacle pixel in place.
    - From each sprite apex, step along the encoded diagonal and paint background cells with the sprite color.
    - When the next diagonal cell is an obstacle, do not overwrite that obstacle; instead, move the ray origin to the obstacle, turn it by 90 degrees using the direction-dependent reflection, and switch the ray color to the obstacle's color.
    - Continue until the ray leaves the grid, repeating the same bounce-and-recolor process at every new obstacle.

    Key insight:
    - The L shape encodes both the apex and the initial diagonal, while each obstacle acts like a colored mirror that both redirects the path and changes the color used for the remaining segment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sprites, obstacles = _extract_scene(g)
    out = g.copy()
    for sprite in sorted(sprites, key=lambda item: tuple(item["apex"])):
        _trace_ray(out, g, obstacles, sprite)
    return out


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = apex
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc)}


def _sprite_box(apex: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = apex
    dr, dc = direction
    return {(r, c), (r - dr, c), (r, c - dc), (r - dr, c - dc)}


def _candidate_apices(
    shape: tuple[int, int], direction: tuple[int, int], reserved: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(shape[0]):
        for c in range(shape[1]):
            apex = (r, c)
            box = _sprite_box(apex, direction)
            if any(not _in_bounds(shape, rr, cc) or (rr, cc) in reserved for rr, cc in box):
                continue
            first = (r + direction[0], c + direction[1])
            if not _in_bounds(shape, *first) or first in reserved:
                continue
            candidates.append(apex)
    return candidates


def _free_run(
    start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int], reserved: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r, c = start
    while True:
        r += direction[0]
        c += direction[1]
        if not _in_bounds(shape, r, c):
            return cells
        if (r, c) in reserved:
            return []
        cells.append((r, c))


def _place_ray(
    shape: tuple[int, int], reserved: set[tuple[int, int]], bounce_count: int
) -> dict[str, object] | None:
    for _ in range(RAY_RETRIES):
        direction = random.choice(DIRS)
        apices = _candidate_apices(shape, direction, reserved)
        if not apices:
            continue
        apex = random.choice(apices)

        local_reserved = set(reserved)
        sprite_box = _sprite_box(apex, direction)
        local_reserved.update(sprite_box)
        current = apex
        current_dir = direction
        path_cells: list[tuple[int, int]] = []
        obstacle_positions: list[tuple[int, int]] = []
        ok = True

        for _bounce_idx in range(bounce_count):
            max_clear = 0
            probe_r, probe_c = current
            while True:
                probe_r += current_dir[0]
                probe_c += current_dir[1]
                if not _in_bounds(shape, probe_r, probe_c) or (probe_r, probe_c) in local_reserved:
                    break
                max_clear += 1

            if max_clear < MIN_SEGMENT_LEN + 1:
                ok = False
                break

            candidates: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
            max_segment = min(MAX_SEGMENT_LEN, max_clear - 1)
            for seg_len in range(MIN_SEGMENT_LEN, max_segment + 1):
                segment = [
                    (current[0] + step * current_dir[0], current[1] + step * current_dir[1])
                    for step in range(1, seg_len + 1)
                ]
                obstacle = (current[0] + (seg_len + 1) * current_dir[0], current[1] + (seg_len + 1) * current_dir[1])
                if any(cell in local_reserved for cell in segment) or obstacle in local_reserved:
                    continue
                next_dir = _reflect(current_dir)
                next_step = (obstacle[0] + next_dir[0], obstacle[1] + next_dir[1])
                if not _in_bounds(shape, *next_step) or next_step in local_reserved:
                    continue
                candidates.append((segment, obstacle))

            if not candidates:
                ok = False
                break

            segment, obstacle = random.choice(candidates)
            path_cells.extend(segment)
            obstacle_positions.append(obstacle)
            local_reserved.update(segment)
            local_reserved.add(obstacle)
            current = obstacle
            current_dir = _reflect(current_dir)

        if not ok:
            continue

        final_cells = _free_run(current, current_dir, shape, local_reserved)
        if len(final_cells) < 1:
            continue
        path_cells.extend(final_cells)
        return {
            "apex": apex,
            "direction": direction,
            "sprite_cells": _sprite_cells(apex, direction),
            "sprite_box": sprite_box,
            "path_cells": tuple(path_cells),
            "obstacles": tuple(obstacle_positions),
        }
    return None


def _choose_bounce_counts(sprite_count: int) -> list[int]:
    counts: list[int] = []
    color_budget = len(PALETTE)
    used_colors = 0
    for idx in range(sprite_count):
        sprites_left = sprite_count - idx
        remaining_budget = color_budget - used_colors
        max_for_this = min(MAX_BOUNCES, remaining_budget - sprites_left)
        if idx == sprite_count - 1 and not any(counts) and max_for_this >= 1 and random.random() < 0.75:
            count = random.randint(1, max_for_this)
        else:
            count = random.randint(0, max_for_this)
        counts.append(count)
        used_colors += 1 + count
    random.shuffle(counts)
    return counts


def _place_extra_obstacles(
    grid: np.ndarray, reserved: set[tuple[int, int]], free_colors: list[int]
) -> list[tuple[tuple[int, int], int]]:
    extras: list[tuple[tuple[int, int], int]] = []
    max_extra = min(random.randint(*EXTRA_OBSTACLE_RANGE), len(free_colors))
    free_cells = [
        (r, c)
        for r in range(grid.shape[0])
        for c in range(grid.shape[1])
        if (r, c) not in reserved
    ]
    random.shuffle(free_cells)
    for _ in range(max_extra):
        if not free_cells or not free_colors:
            break
        cell = free_cells.pop()
        extras.append((cell, free_colors.pop()))
        reserved.add(cell)
    return extras


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    shape = (h, w)
    grid = np.zeros(shape, dtype=int)

    sprite_count = random.randint(*SPRITE_COUNT_RANGE)
    bounce_counts = _choose_bounce_counts(sprite_count)

    reserved: set[tuple[int, int]] = set()
    rays: list[dict[str, object]] = []
    for bounce_count in bounce_counts:
        ray = _place_ray(shape, reserved, bounce_count)
        if ray is None:
            return None
        reserved.update(ray["sprite_box"])
        reserved.update(ray["path_cells"])
        reserved.update(ray["obstacles"])
        rays.append(ray)

    color_pool = list(PALETTE)
    random.shuffle(color_pool)
    used_colors: set[int] = set()

    for ray in rays:
        sprite_color = color_pool.pop()
        used_colors.add(sprite_color)
        ray["sprite_color"] = sprite_color
        obstacle_colors: list[int] = []
        current_color = sprite_color
        for _ in ray["obstacles"]:
            candidates = [color for color in color_pool if color != current_color]
            if not candidates:
                return None
            color = candidates[-1]
            color_pool.remove(color)
            used_colors.add(color)
            obstacle_colors.append(color)
            current_color = color
        ray["obstacle_colors"] = tuple(obstacle_colors)

    for ray in rays:
        sprite_color = int(ray["sprite_color"])
        for r, c in ray["sprite_cells"]:
            grid[r, c] = sprite_color
        for obstacle, color in zip(ray["obstacles"], ray["obstacle_colors"]):
            grid[obstacle] = int(color)

    extras = _place_extra_obstacles(grid, reserved, color_pool)
    for (r, c), color in extras:
        grid[r, c] = int(color)

    try:
        parsed_sprites, parsed_obstacles = _extract_scene(grid)
        assert len(parsed_sprites) == len(rays), "sprite count changed after placement"
        expected_obstacles = sum(len(ray["obstacles"]) for ray in rays) + len(extras)
        assert len(parsed_obstacles) == expected_obstacles, "obstacle count changed after placement"
        out = main(grid)
    except AssertionError:
        return None

    assert out.shape == grid.shape
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-ray input")
