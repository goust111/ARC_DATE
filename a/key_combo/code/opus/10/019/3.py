from __future__ import annotations

import random

import numpy as np


BG = 0
BAR = 5
SNAKE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_snakes(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    snakes: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) in (BG, BAR):
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            snakes.append({"color": color, "cells": cells})
    return snakes


def _degree_map(cells: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    cell_set = set(cells)
    return {
        cell: sum((nr, nc) in cell_set for nr, nc in _neighbors4(cell[0], cell[1]))
        for cell in cell_set
    }


def _simple_path_endpoints(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    degrees = _degree_map(cells)
    assert all(deg <= 2 for deg in degrees.values()), "snake must be a simple chain"
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "snake must have exactly two endpoints"
    return sorted(endpoints)


def _endpoint_rays(
    grid: np.ndarray, cells: list[tuple[int, int]], endpoint: tuple[int, int]
) -> list[dict]:
    cell_set = set(cells)
    er, ec = endpoint
    neighbors = [(nr, nc) for nr, nc in _neighbors4(er, ec) if (nr, nc) in cell_set]
    assert len(neighbors) == 1
    nr, nc = neighbors[0]
    dr = nr - er
    dc = nc - ec
    perps = [(-dc, dr), (dc, -dr)]
    h, w = grid.shape
    rays: list[dict] = []

    for pdr, pdc in perps:
        rr, cc = er + pdr, ec + pdc
        segment: list[tuple[int, int]] = []
        hit = None
        while 0 <= rr < h and 0 <= cc < w:
            if int(grid[rr, cc]) == BAR:
                hit = (rr, cc)
                break
            segment.append((rr, cc))
            rr += pdr
            cc += pdc
        rays.append(
            {
                "local_dir": (dr, dc),
                "cast_dir": (pdr, pdc),
                "segment": segment,
                "hit": hit,
            }
        )
    return rays


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each snake endpoint's local tangent to decide where vertical-to-the-tangent rays stamp the horizontal bars.

    Input structure:
    - The background is black.
    - Two or three complete gray horizontal bars span the full width of the grid.
    - Every other connected colored object is a single 4-connected snake path with exactly two endpoints.

    Rule:
    - For each snake endpoint, look at its unique neighboring snake cell to get the endpoint's local travel direction.
    - Cast two rays from the endpoint along the two directions perpendicular to that local direction.
    - When a ray first reaches a gray bar, paint the whole endpoint-to-hit segment in the snake color, then stamp a 3x3 gray block centered on the hit and restore the center to the snake color.
    - Rays that never reach a gray bar produce nothing.

    Key insight:
    - The bars are fixed, but the active casting axis is not; it is encoded locally by the snake geometry at each endpoint, so vertical-endpoint decoys must stay silent while horizontal-endpoint decoys fire.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert g.ndim == 2

    bar_rows = [r for r in range(h) if np.all(g[r, :] == BAR)]
    assert 2 <= len(bar_rows) <= 3

    snakes = _find_snakes(g)
    assert snakes

    out = g.copy()
    hits: dict[tuple[int, int], int] = {}

    for snake in snakes:
        color = int(snake["color"])
        cells = snake["cells"]
        for endpoint in _simple_path_endpoints(cells):
            for ray in _endpoint_rays(g, cells, endpoint):
                hit = ray["hit"]
                if hit is None:
                    continue
                for rr, cc in ray["segment"]:
                    out[rr, cc] = color
                hr, hc = hit
                out[hr, hc] = color
                prev = hits.get(hit)
                assert prev is None or prev == color
                hits[hit] = color

    for (hr, hc), color in hits.items():
        r0 = max(0, hr - 1)
        r1 = min(h, hr + 2)
        c0 = max(0, hc - 1)
        c1 = min(w, hc + 2)
        out[r0:r1, c0:c1] = BAR
        out[hr, hc] = color

    return out


def _append_line(path: list[tuple[int, int]], dr: int, dc: int, steps: int) -> None:
    for _ in range(steps):
        r, c = path[-1]
        path.append((r + dr, c + dc))


def _build_template(
    kind: str, usable_rows: list[int], slot: tuple[int, int]
) -> list[tuple[int, int]] | None:
    c0, c1 = slot
    width = c1 - c0 + 1
    hdir = random.choice([-1, 1])
    start_c = c0 if hdir == 1 else c1

    if kind == "H":
        if width < 5:
            return None
        row = random.choice(usable_rows)
        h_steps = random.randint(4, min(9, width - 1))
        path = [(row, start_c)]
        _append_line(path, 0, hdir, h_steps)
        return path

    if len(usable_rows) < 2:
        return None

    row_a, row_b = random.sample(usable_rows, 2)
    v_steps = abs(row_b - row_a)
    v_dir = 1 if row_b > row_a else -1

    if kind in {"HV", "VH"}:
        low = max(2, 4 - v_steps)
        high = min(width - 1, 9 - v_steps)
        if low > high:
            return None
        h_steps = random.randint(low, high)
        path = [(row_a, start_c)]
        if kind == "HV":
            _append_line(path, 0, hdir, h_steps)
            _append_line(path, v_dir, 0, v_steps)
        else:
            _append_line(path, v_dir, 0, v_steps)
            _append_line(path, 0, hdir, h_steps)
        return path

    if kind == "HVH":
        low = max(2, 4 - v_steps)
        high = min(width - 1, 9 - v_steps)
        if low > high:
            return None
        total_h = random.randint(low, high)
        first_h = random.randint(1, total_h - 1)
        second_h = total_h - first_h
        path = [(row_a, start_c)]
        _append_line(path, 0, hdir, first_h)
        _append_line(path, v_dir, 0, v_steps)
        _append_line(path, 0, hdir, second_h)
        return path

    if kind == "HVHVH":
        if len(usable_rows) < 3:
            return None
        row0, row1, row2 = random.sample(usable_rows, 3)
        v1 = abs(row1 - row0)
        v2 = abs(row2 - row1)
        total_v = v1 + v2
        low = max(3, 4 - total_v)
        high = min(width - 1, 9 - total_v)
        if low > high:
            return None
        total_h = random.randint(low, high)
        cuts = sorted(random.sample(range(1, total_h), 2))
        h1 = cuts[0]
        h2 = cuts[1] - cuts[0]
        h3 = total_h - cuts[1]
        path = [(row0, start_c)]
        _append_line(path, 0, hdir, h1)
        _append_line(path, 1 if row1 > row0 else -1, 0, v1)
        _append_line(path, 0, hdir, h2)
        _append_line(path, 1 if row2 > row1 else -1, 0, v2)
        _append_line(path, 0, hdir, h3)
        return path

    return None


def _path_valid(
    path: list[tuple[int, int]],
    slot: tuple[int, int],
    band_rows: set[int],
) -> bool:
    if path is None or not (5 <= len(path) <= 10):
        return False
    if len(set(path)) != len(path):
        return False
    c0, c1 = slot
    for r, c in path:
        if r not in band_rows or c < c0 or c > c1:
            return False
    for (r0, c0p), (r1, c1p) in zip(path, path[1:]):
        if abs(r0 - r1) + abs(c0p - c1p) != 1:
            return False
    try:
        _simple_path_endpoints(path)
    except AssertionError:
        return False
    return True


def _build_snake(
    kind_choices: list[str],
    usable_rows: list[int],
    slot: tuple[int, int],
) -> list[tuple[int, int]] | None:
    band_rows = set(usable_rows)
    for _ in range(80):
        kind = random.choice(kind_choices)
        path = _build_template(kind, usable_rows, slot)
        if _path_valid(path, slot, band_rows):
            return path
    return None


def _sample_slot_widths(count: int) -> list[int]:
    max_total = 30 - 2 - 2 * (count - 1)
    min_width = 4
    widths = [min_width] * count
    remaining = max_total - min_width * count
    per_slot_cap = 4 if count == 5 else 5 if count == 4 else 6

    while remaining > 0:
        choices = [i for i, width in enumerate(widths) if width < per_slot_cap]
        if not choices:
            break
        widths[random.choice(choices)] += 1
        remaining -= 1

    if count <= 4 and random.random() < 0.5:
        slack = random.randint(0, min(2, 30 - (2 + sum(widths) + 2 * (count - 1))))
        if slack > 0:
            widths[random.randrange(count)] += slack

    random.shuffle(widths)
    return widths


def _build_layout() -> tuple[int, list[int], list[dict], list[tuple[int, int]], list[int]]:
    n_bars = random.choice([2, 2, 3])
    n_bands = n_bars + 1
    band_heights = [random.randint(4, 5) for _ in range(n_bands)]
    total_rows = sum(band_heights) + n_bars

    bands: list[dict] = []
    cursor = 0
    for i, height in enumerate(band_heights):
        start = cursor
        end = cursor + height - 1
        if i == 0:
            usable = list(range(start, end))
        elif i == n_bands - 1:
            usable = list(range(start + 1, end + 1))
        else:
            usable = list(range(start + 1, end))
        bands.append({"start": start, "end": end, "usable": usable})
        cursor = end + 2

    bar_rows: list[int] = []
    cursor = band_heights[0]
    for i in range(n_bars):
        bar_rows.append(cursor)
        if i + 1 < len(band_heights):
            cursor += 1 + band_heights[i + 1]

    counts = [1] * n_bands
    extra_limit = 1 if n_bands == 4 else 2
    extra = random.randint(0, extra_limit)
    for idx in random.sample(range(n_bands), extra):
        counts[idx] += 1

    total_snakes = sum(counts)
    widths = _sample_slot_widths(total_snakes)

    slots: list[tuple[int, int]] = []
    col = 1
    for width in widths:
        slots.append((col, col + width - 1))
        col += width + 2

    return total_rows, bar_rows, bands, slots, counts


def _render_input(
    total_rows: int,
    bar_rows: list[int],
    bands: list[dict],
    slots: list[tuple[int, int]],
    counts: list[int],
) -> np.ndarray | None:
    snake_bands: list[int] = []
    for band_idx, count in enumerate(counts):
        snake_bands.extend([band_idx] * count)
    random.shuffle(snake_bands)

    if len(snake_bands) != len(slots):
        return None

    middle_candidates = [i for i, band_idx in enumerate(snake_bands) if 0 < band_idx < len(bands) - 1]
    if not middle_candidates:
        return None
    mixed_idx = random.choice(middle_candidates)
    hh_candidates = [i for i in range(len(slots)) if i != mixed_idx]
    hh_idx = random.choice(hh_candidates) if hh_candidates else mixed_idx

    snakes: list[dict] = []
    used_colors = random.sample(SNAKE_COLORS, len(slots))

    for i, (slot, band_idx, color) in enumerate(zip(slots, snake_bands, used_colors)):
        usable_rows = bands[band_idx]["usable"]
        if len(usable_rows) < 1:
            return None

        slot_width = slot[1] - slot[0] + 1
        kinds = []
        if slot_width >= 5:
            kinds.append("H")
        if slot_width >= 4 and len(usable_rows) >= 2:
            kinds.extend(["HV", "VH", "HVH"])
        if slot_width >= 4 and len(usable_rows) >= 3:
            kinds.append("HVHVH")
        if not kinds:
            return None

        if i == mixed_idx:
            required = [kind for kind in kinds if kind in {"HV", "VH"}]
        elif i == hh_idx:
            required = [kind for kind in kinds if kind in {"H", "HVH", "HVHVH"}]
        else:
            required = kinds

        path = _build_snake(required, usable_rows, slot)
        if path is None:
            return None
        snakes.append({"color": color, "cells": path, "band": band_idx})

    width = max(c1 for _c0, c1 in slots) + 2
    grid = np.full((total_rows, width), BG, dtype=int)
    grid[bar_rows, :] = BAR

    for snake in snakes:
        color = int(snake["color"])
        for r, c in snake["cells"]:
            if int(grid[r, c]) != BG:
                return None
            grid[r, c] = color

    if _semantic_ok(grid):
        return grid
    return None


def _semantic_ok(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False

    bar_rows = [r for r in range(h) if np.all(grid[r, :] == BAR)]
    if len(bar_rows) not in (2, 3):
        return False
    if any(b <= 0 or b >= h - 1 for b in bar_rows):
        return False
    if any(bar_rows[i + 1] - bar_rows[i] < 5 for i in range(len(bar_rows) - 1)):
        return False

    snakes = _find_snakes(grid)
    if len(snakes) < len(bar_rows) + 1:
        return False

    endpoint_columns: list[int] = []
    hit_points: list[tuple[int, int]] = []
    saw_vertical_endpoint = False
    saw_one_hit = False
    saw_two_hit = False

    for snake in snakes:
        cells = snake["cells"]
        color = int(snake["color"])
        if not (5 <= len(cells) <= 10):
            return False
        try:
            endpoints = _simple_path_endpoints(cells)
        except AssertionError:
            return False

        degrees = _degree_map(cells)
        if any(deg == 0 for deg in degrees.values()):
            return False

        for endpoint in endpoints:
            endpoint_columns.append(endpoint[1])
            rays = _endpoint_rays(grid, cells, endpoint)
            neighbors = [(nr, nc) for nr, nc in _neighbors4(endpoint[0], endpoint[1]) if (nr, nc) in set(cells)]
            dr = neighbors[0][0] - endpoint[0]
            dc = neighbors[0][1] - endpoint[1]
            if dr != 0:
                saw_vertical_endpoint = True

            hits_here = 0
            for ray in rays:
                for rr, cc in ray["segment"]:
                    if int(grid[rr, cc]) != BG:
                        return False
                hit = ray["hit"]
                if hit is not None:
                    hits_here += 1
                    if int(grid[hit[0], hit[1]]) != BAR:
                        return False
                    hit_points.append(hit)
            if hits_here == 1:
                saw_one_hit = True
            if hits_here == 2:
                saw_two_hit = True

        if len({int(grid[r, c]) for r, c in cells}) != 1 or int(grid[cells[0][0], cells[0][1]]) != color:
            return False

    endpoint_columns.sort()
    if any(b - a < 3 for a, b in zip(endpoint_columns, endpoint_columns[1:])):
        return False

    for i, (r0, c0) in enumerate(hit_points):
        for r1, c1 in hit_points[i + 1 :]:
            if abs(r0 - r1) <= 2 and abs(c0 - c1) <= 2:
                return False

    if not (saw_vertical_endpoint and saw_one_hit and saw_two_hit):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if max(out.shape) > 30:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    total_rows, bar_rows, bands, slots, counts = _build_layout()
    return _render_input(total_rows, bar_rows, bands, slots, counts)


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for snake-ray stamp task")
