from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = list(range(1, 10))
MAX_GENERATION_ATTEMPTS = 600
GRID_HEIGHT_RANGE = (16, 26)
GRID_WIDTH_RANGE = (18, 30)
PLATFORM_COUNT_RANGE = (1, 4)
LAUNCHER_COUNT_RANGE = (2, 4)
MIN_RAY_CELLS = 4


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _analyze_launcher(coords: np.ndarray) -> dict | None:
    pts = {(int(r), int(c)) for r, c in coords}
    rows = Counter(r for r, _ in pts)
    cols = Counter(c for _, c in pts)

    bend_rows = [r for r, count in rows.items() if count >= 2]
    bend_cols = [c for c, count in cols.items() if count >= 2]
    if len(bend_rows) != 1 or len(bend_cols) != 1:
        return None

    bend_r = bend_rows[0]
    bend_c = bend_cols[0]
    if any(r != bend_r and c != bend_c for r, c in pts):
        return None

    row_cols = sorted(c for r, c in pts if r == bend_r)
    col_rows = sorted(r for r, c in pts if c == bend_c)
    if row_cols != list(range(row_cols[0], row_cols[-1] + 1)):
        return None
    if col_rows != list(range(col_rows[0], col_rows[-1] + 1)):
        return None

    if bend_c == row_cols[0]:
        direction = 1
        end_c = row_cols[-1]
    elif bend_c == row_cols[-1]:
        direction = -1
        end_c = row_cols[0]
    else:
        return None

    if bend_r == col_rows[0]:
        vertical_dir = 1
    elif bend_r == col_rows[-1]:
        vertical_dir = -1
    else:
        return None

    horiz_len = len(row_cols) - 1
    vert_len = len(col_rows) - 1
    if horiz_len <= vert_len or vert_len < 1:
        return None

    return {
        "bend": (bend_r, bend_c),
        "direction": direction,
        "vertical_dir": vertical_dir,
        "long_len": horiz_len,
        "short_len": vert_len,
        "end": (bend_r, end_c),
        "bbox": _bbox_coords(coords),
        "cells": pts,
    }


def _find_launchers(grid: np.ndarray, bg: int) -> list[dict]:
    launchers: list[dict] = []
    for comp in _components_mask(grid != bg):
        colors = {int(grid[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        info = _analyze_launcher(comp)
        if info is None:
            continue
        info["color"] = int(next(iter(colors)))
        launchers.append(info)
    launchers.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return launchers


def _trace_ray(
    shape: tuple[int, int],
    occupied: np.ndarray,
    launcher: dict,
) -> tuple[list[tuple[int, int]], str]:
    r, c = launcher["end"]
    direction = int(launcher["direction"])
    span = int(launcher["long_len"])
    ray: list[tuple[int, int]] = []

    while True:
        for _ in range(span):
            c += direction
            if not _in_bounds(shape, r, c):
                return ray, "boundary"
            if occupied[r, c]:
                return ray, "obstacle"
            ray.append((r, c))

        r += 1
        if not _in_bounds(shape, r, c):
            return ray, "boundary"
        if occupied[r, c]:
            return ray, "obstacle"
        ray.append((r, c))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw each launcher's stair-step ballistic ray using its own horizontal arm length as the descent period.

    Input structure:
    - A dominant background fills most of the grid.
    - The bottom row is a solid ground color, and a few solid rectangular platforms float above it.
    - Several monochrome one-pixel-thick L launchers appear elsewhere; each has one vertical short arm and one strictly longer horizontal arm pointing left or right.

    Rule:
    - Detect every L launcher and measure L as the number of horizontal-arm pixels excluding the bend.
    - Starting from the first cell just outside that arm's far end, paint the launcher's color along a repeated path: move horizontally for L cells in the arm direction, then move down one cell, and repeat.
    - Stop a ray before any step that would enter a non-background cell or leave the grid; preserve all previously painted cells.

    Key insight:
    - The launcher's longest arm determines both emission direction and the ray's glide period, so different launchers can produce different stair slopes in the same scene.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    occupied = g != bg
    out = g.copy()

    for launcher in _find_launchers(g, bg):
        for r, c in _trace_ray(g.shape, occupied, launcher)[0]:
            out[r, c] = int(launcher["color"])

    return out


def _cells_clear(mask: np.ndarray, cells: list[tuple[int, int]], margin: int = 1) -> bool:
    h, w = mask.shape
    for r, c in cells:
        if not _in_bounds(mask.shape, r, c) or mask[r, c]:
            return False
        for rr in range(r - margin, r + margin + 1):
            for cc in range(c - margin, c + margin + 1):
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc]:
                    return False
    return True


def _launcher_cells(
    bend_r: int,
    bend_c: int,
    direction: int,
    vertical_dir: int,
    long_len: int,
    short_len: int,
) -> list[tuple[int, int]]:
    cells = {(bend_r, bend_c)}
    for i in range(1, long_len + 1):
        cells.add((bend_r, bend_c + direction * i))
    for i in range(1, short_len + 1):
        cells.add((bend_r + vertical_dir * i, bend_c))
    return sorted(cells)


def _paint(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _build_platforms(
    h: int,
    w: int,
    grid: np.ndarray,
    colors: list[int],
) -> tuple[list[dict], np.ndarray] | None:
    occupied = grid != 0
    platforms: list[dict] = []
    target = random.randint(*PLATFORM_COUNT_RANGE)
    attempts = 0
    while len(platforms) < target and attempts < 240:
        attempts += 1
        ph = random.randint(1, min(3, h - 4))
        pw = random.randint(3, min(8, w - 2))
        r0 = random.randint(1, h - ph - 3)
        c0 = random.randint(1, w - pw - 1)
        cells = [(r, c) for r in range(r0, r0 + ph) for c in range(c0, c0 + pw)]
        if not _cells_clear(occupied, cells, margin=1):
            continue
        color = colors[len(platforms)]
        _paint(grid, cells, color)
        occupied = grid != 0
        platforms.append({"cells": cells, "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1), "color": color})

    if not platforms:
        return None
    return platforms, occupied


def _sample_launcher_specs(count: int) -> list[tuple[int, int, int]]:
    directions = [1, -1]
    while len(directions) < count:
        directions.append(random.choice([-1, 1]))
    random.shuffle(directions)

    lengths_pool = [2, 3, 4, 5, 6]
    long_lengths = [random.choice(lengths_pool) for _ in range(count)]
    if len(set(long_lengths)) == 1:
        idx = random.randrange(count)
        choices = [v for v in lengths_pool if v != long_lengths[idx]]
        long_lengths[idx] = random.choice(choices)

    specs = []
    for i in range(count):
        long_len = long_lengths[i]
        short_len = random.randint(1, min(3, long_len - 1))
        vertical_dir = random.choice([-1, 1])
        specs.append((directions[i], vertical_dir, long_len, short_len))
    return specs


def _place_launchers(
    h: int,
    w: int,
    grid: np.ndarray,
    occupied: np.ndarray,
    colors: list[int],
) -> tuple[list[dict], list[str]] | None:
    launcher_count = random.randint(*LAUNCHER_COUNT_RANGE)
    specs = _sample_launcher_specs(launcher_count)
    reserved = occupied.copy()
    launchers: list[dict] = []
    stop_reasons: list[str] = []

    for idx, (direction, vertical_dir, long_len, short_len) in enumerate(specs):
        placed = False
        for _ in range(280):
            if direction == 1:
                c_min, c_max = 1, w - long_len - 3
            else:
                c_min, c_max = long_len + 2, w - 2
            if vertical_dir == 1:
                r_min, r_max = 1, h - short_len - 3
            else:
                r_min, r_max = short_len + 1, h - 3
            if c_min > c_max or r_min > r_max:
                return None

            bend_r = random.randint(r_min, r_max)
            bend_c = random.randint(c_min, c_max)
            cells = _launcher_cells(bend_r, bend_c, direction, vertical_dir, long_len, short_len)
            if not _cells_clear(reserved, cells, margin=1):
                continue

            launcher = {
                "bend": (bend_r, bend_c),
                "direction": direction,
                "vertical_dir": vertical_dir,
                "long_len": long_len,
                "short_len": short_len,
                "end": (bend_r, bend_c + direction * long_len),
                "color": colors[idx],
                "cells": cells,
            }
            ray, stop_reason = _trace_ray((h, w), occupied, launcher)
            if len(ray) < MIN_RAY_CELLS:
                continue
            if any(reserved[r, c] for r, c in ray):
                continue

            _paint(grid, cells, launcher["color"])
            for r, c in cells:
                occupied[r, c] = True
                reserved[r, c] = True
            for r, c in ray:
                reserved[r, c] = True

            launcher["ray"] = ray
            launchers.append(launcher)
            stop_reasons.append(stop_reason)
            placed = True
            break

        if not placed:
            return None

    return launchers, stop_reasons


def _render_output(inp: np.ndarray, launchers: list[dict]) -> np.ndarray:
    out = inp.copy()
    for launcher in launchers:
        for r, c in launcher["ray"]:
            out[r, c] = int(launcher["color"])
    return out


def _semantic_ok(inp: np.ndarray, launchers: list[dict], stop_reasons: list[str]) -> bool:
    parsed = _find_launchers(inp, 0)
    if len(parsed) != len(launchers):
        return False

    expected = sorted(
        (item["color"], item["long_len"], item["direction"], item["bend"][0], item["bend"][1])
        for item in launchers
    )
    actual = sorted(
        (item["color"], item["long_len"], item["direction"], item["bend"][0], item["bend"][1])
        for item in parsed
    )
    if actual != expected:
        return False

    lengths = {item["long_len"] for item in launchers}
    directions = {item["direction"] for item in launchers}
    if len(lengths) < 2 or len(directions) < 2:
        return False

    if "boundary" not in stop_reasons or "obstacle" not in stop_reasons:
        return False

    total_ray = sum(len(item["ray"]) for item in launchers)
    if total_ray < 12:
        return False

    expected_out = _render_output(inp, launchers)
    actual_out = main(inp)
    if not np.array_equal(actual_out, expected_out):
        return False

    # Reject constant-slope shortcuts: at least one launcher must have glide span > 2.
    if max(item["long_len"] for item in launchers) <= 2:
        return False

    # Rays must all be distinct and non-empty.
    ray_sets = [set(item["ray"]) for item in launchers]
    if any(not cells for cells in ray_sets):
        return False
    for i in range(len(ray_sets)):
        for j in range(i + 1, len(ray_sets)):
            if ray_sets[i] & ray_sets[j]:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.zeros((h, w), dtype=int)

    colors = random.sample(PALETTE, k=min(len(PALETTE), 1 + PLATFORM_COUNT_RANGE[1] + LAUNCHER_COUNT_RANGE[1]))
    floor_color = colors[0]
    grid[-1, :] = floor_color

    platform_info = _build_platforms(h, w, grid, colors[1 : 1 + PLATFORM_COUNT_RANGE[1]])
    if platform_info is None:
        return None
    _platforms, occupied = platform_info

    launcher_colors = colors[1 + PLATFORM_COUNT_RANGE[1] :]
    launch_info = _place_launchers(h, w, grid, occupied, launcher_colors)
    if launch_info is None:
        return None
    launchers, stop_reasons = launch_info

    if not _semantic_ok(grid, launchers, stop_reasons):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid long-arm ballistic input")
