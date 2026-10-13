import random
from collections import Counter, deque

import numpy as np


BG = 0
CCW_MARKER = 1  # blue
CW_MARKER = 2   # red
STATION = 3     # green
TRACK_COLORS = [4, 5, 6, 7, 8, 9]

GRID_TRACK_COUNT = [1, 2, 3]
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
TRACK_ATTEMPTS = 80
MAX_TRACK_CELLS = 28
MIN_TRACK_SIZE = 4
MAX_TRACK_SIZE = 8

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _normalize_order(order: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in order)
    min_c = min(c for _, c in order)
    return [(r - min_r, c - min_c) for r, c in order]


def _signed_area(order: list[tuple[int, int]]) -> int:
    area2 = 0
    for (r0, c0), (r1, c1) in zip(order, order[1:] + order[:1]):
        area2 += c0 * r1 - c1 * r0
    return area2


def _clockwise_order(order: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if _signed_area(order) > 0:
        return order
    return [order[0]] + order[:0:-1]


def _cycle_order(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    neigh: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in pts:
        here = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in pts]
        if len(here) != 2:
            raise AssertionError("track component must be a simple 4-neighbor cycle")
        neigh[(r, c)] = sorted(here)

    start = min(pts)
    prev = start
    cur = neigh[start][0]
    order = [start]
    while cur != start:
        order.append(cur)
        a, b = neigh[cur]
        nxt = b if a == prev else a
        prev, cur = cur, nxt
        if len(order) > len(pts):
            raise AssertionError("cycle traversal did not terminate cleanly")
    if len(order) != len(pts):
        raise AssertionError("cycle traversal missed cells")
    return order


def _interior_cells_for_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)

    blocked = np.zeros((max_r - min_r + 3, max_c - min_c + 3), dtype=bool)
    for r, c in coords:
        blocked[r - min_r + 1, c - min_c + 1] = True

    outside = np.zeros_like(blocked, dtype=bool)
    q = deque()
    bh, bw = blocked.shape
    for r in range(bh):
        for c in range(bw):
            if r not in (0, bh - 1) and c not in (0, bw - 1):
                continue
            if blocked[r, c] or outside[r, c]:
                continue
            outside[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < bh and 0 <= nc < bw and not blocked[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    interior_mask = (~blocked) & (~outside)
    interior_comps = _components(interior_mask)
    if len(interior_comps) != 1:
        raise AssertionError("track must enclose exactly one interior region")

    cells = []
    for r, c in interior_comps[0]:
        cells.append((r + min_r - 1, c + min_c - 1))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move cars around orthogonal one-pixel loops by a step count equal to the
    number of cars on the same loop; red center markers mean clockwise and blue
    markers mean counterclockwise. A visible green station remains green unless
    a moved car occupies it, in which case the occupying car becomes green.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    marker_mask = (g == CW_MARKER) | (g == CCW_MARKER)
    track_mask = (g != BG) & (~marker_mask)

    for comp in _components(track_mask):
        values = [int(g[r, c]) for r, c in comp]
        color_counts = Counter(v for v in values if v != STATION)
        if not color_counts:
            raise AssertionError("track component must contain a base track color")
        track_color = color_counts.most_common(1)[0][0]

        stations = [(r, c) for r, c in comp if int(g[r, c]) == STATION]
        if len(stations) > 1:
            raise AssertionError("each track may have at most one visible station")
        station = stations[0] if stations else None

        cars = [(r, c) for r, c in comp if int(g[r, c]) not in (track_color, STATION)]
        if not (1 <= len(cars) <= 3):
            raise AssertionError("each track must contain between 1 and 3 cars")

        interior = set(_interior_cells_for_coords(comp))
        markers = [(r, c, int(g[r, c])) for r, c in interior if int(g[r, c]) in (CW_MARKER, CCW_MARKER)]
        if len(markers) != 1:
            raise AssertionError("each track must contain exactly one center marker")
        marker_color = markers[0][2]

        order = _clockwise_order(_cycle_order(comp))
        index_of = {pt: i for i, pt in enumerate(order)}
        if len(index_of) != len(order):
            raise AssertionError("track order should not repeat cells")

        for r, c in comp:
            out[r, c] = track_color
        if station is not None:
            out[station] = STATION

        shift = len(cars) if marker_color == CW_MARKER else -len(cars)
        loop_len = len(order)
        for r, c in cars:
            dst = order[(index_of[(r, c)] + shift) % loop_len]
            color = STATION if station is not None and dst == station else int(g[r, c])
            out[dst] = color

    return out


def _rectangle_cycle(height: int, width: int) -> list[tuple[int, int]]:
    order: list[tuple[int, int]] = []
    for c in range(width):
        order.append((0, c))
    for r in range(1, height):
        order.append((r, width - 1))
    for c in range(width - 2, -1, -1):
        order.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        order.append((r, 0))
    return order


def _is_valid_cycle(order: list[tuple[int, int]]) -> bool:
    if len(order) < 8 or len(set(order)) != len(order):
        return False
    for a, b in zip(order, order[1:] + order[:1]):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) != 1:
            return False
    pts = set(order)
    for r, c in pts:
        deg = sum((nr, nc) in pts for nr, nc in _neighbors4(r, c))
        if deg != 2:
            return False
    try:
        _interior_cells_for_coords(order)
    except AssertionError:
        return False
    return True


def _bulge(order: list[tuple[int, int]], edge_idx: int, inward: bool) -> list[tuple[int, int]] | None:
    p = order[edge_idx]
    q = order[(edge_idx + 1) % len(order)]
    dr = q[0] - p[0]
    dc = q[1] - p[1]
    if abs(dr) + abs(dc) != 1:
        return None

    if inward:
        nr, nc = dc, -dr
    else:
        nr, nc = -dc, dr

    a = (p[0] + nr, p[1] + nc)
    b = (q[0] + nr, q[1] + nc)
    if a == b:
        return None
    if a in order or b in order:
        return None

    cand = order[: edge_idx + 1] + [a, b] + order[edge_idx + 1 :]
    cand = _normalize_order(cand)
    if len(cand) > MAX_TRACK_CELLS:
        return None
    if not _is_valid_cycle(cand):
        return None
    return cand


def _rotate_coords(coords: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    out = coords[:]
    out = _normalize_order(out)
    h = max(r for r, _ in out) + 1
    w = max(c for _, c in out) + 1
    for _ in range(turns % 4):
        out = [(c, h - 1 - r) for r, c in out]
        h, w = w, h
        out = _normalize_order(out)
    return out


def _choose_marker_cell(interior: list[tuple[int, int]]) -> tuple[int, int]:
    mean_r = sum(r for r, _ in interior) / len(interior)
    mean_c = sum(c for _, c in interior) / len(interior)
    return min(interior, key=lambda pt: (abs(pt[0] - mean_r) + abs(pt[1] - mean_c), pt))


def _gen_track(rng: random.Random, track_color: int) -> dict | None:
    for _ in range(TRACK_ATTEMPTS):
        height = rng.randint(MIN_TRACK_SIZE, MAX_TRACK_SIZE)
        width = rng.randint(MIN_TRACK_SIZE, MAX_TRACK_SIZE)
        order = _rectangle_cycle(height, width)

        target_bulges = rng.randint(0, 3)
        applied = 0
        local_attempts = 0
        while applied < target_bulges and local_attempts < 40:
            local_attempts += 1
            cand = _bulge(order, rng.randrange(len(order)), inward=bool(rng.getrandbits(1)))
            if cand is None:
                continue
            order = cand
            applied += 1

        order = _rotate_coords(order, rng.randrange(4))
        order = _clockwise_order(order)
        if not _is_valid_cycle(order):
            continue

        interior = _interior_cells_for_coords(order)
        marker = _choose_marker_cell(interior)
        marker_color = CW_MARKER if rng.random() < 0.5 else CCW_MARKER

        car_count = rng.randint(1, 3)
        loop_len = len(order)
        if loop_len <= car_count:
            continue

        station_present = rng.random() < 0.5
        station_should_hit = rng.random() < 0.5
        delta = car_count if marker_color == CW_MARKER else -car_count

        car_indices = None
        station_index = None
        for _ in range(25):
            chosen = sorted(rng.sample(range(loop_len), car_count))
            chosen_set = set(chosen)
            free_now = set(range(loop_len)) - chosen_set
            future = {(idx + delta) % loop_len for idx in chosen}

            if not station_present:
                car_indices = chosen
                break

            if station_should_hit:
                options = list((future & free_now))
                if not options:
                    continue
                station_index = rng.choice(options)
            else:
                options = list(free_now - future)
                if not options:
                    continue
                station_index = rng.choice(options)
            car_indices = chosen
            break

        if car_indices is None:
            continue

        car_palette = [c for c in TRACK_COLORS if c != track_color]
        car_colors = rng.sample(car_palette, car_count)
        bbox_h = max(r for r, _ in order) + 1
        bbox_w = max(c for _, c in order) + 1
        return {
            "order": order,
            "cells": set(order),
            "height": bbox_h,
            "width": bbox_w,
            "track_color": track_color,
            "marker": marker,
            "marker_color": marker_color,
            "car_indices": car_indices,
            "car_colors": car_colors,
            "station_index": station_index,
        }
    return None


def _paint_track(grid: np.ndarray, top: int, left: int, spec: dict) -> None:
    order = spec["order"]
    for r, c in order:
        grid[top + r, left + c] = spec["track_color"]

    if spec["station_index"] is not None:
        sr, sc = order[spec["station_index"]]
        grid[top + sr, left + sc] = STATION

    for idx, color in zip(spec["car_indices"], spec["car_colors"]):
        r, c = order[idx]
        grid[top + r, left + c] = color

    mr, mc = spec["marker"]
    grid[top + mr, left + mc] = spec["marker_color"]


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    track_mask = (grid != BG) & (grid != CW_MARKER) & (grid != CCW_MARKER)
    comps = _components(track_mask)
    if not (1 <= len(comps) <= 3):
        return False

    seen_track_colors = set()
    for comp in comps:
        values = [int(grid[r, c]) for r, c in comp]
        counts = Counter(v for v in values if v != STATION)
        if not counts:
            return False
        track_color = counts.most_common(1)[0][0]
        if track_color in seen_track_colors:
            return False
        seen_track_colors.add(track_color)
        stations = sum(int(grid[r, c]) == STATION for r, c in comp)
        cars = sum(int(grid[r, c]) not in (track_color, STATION) for r, c in comp)
        if stations > 1 or not (1 <= cars <= 3):
            return False
        try:
            interior = set(_interior_cells_for_coords(comp))
            marker_count = sum(int(grid[r, c]) in (CW_MARKER, CCW_MARKER) for r, c in interior)
            if marker_count != 1:
                return False
            _cycle_order(comp)
        except AssertionError:
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == grid.shape and not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    track_count = rng.choice(GRID_TRACK_COUNT)
    size_base = 12 + 5 * track_count
    height = rng.randint(size_base, size_base + 6)
    width = rng.randint(size_base, size_base + 6)
    grid = np.full((height, width), BG, dtype=int)

    track_colors = rng.sample(TRACK_COLORS, track_count)
    specs = []
    for color in track_colors:
        spec = _gen_track(rng, color)
        if spec is None:
            return None
        specs.append(spec)

    blocked = np.zeros_like(grid, dtype=bool)
    gap = 2
    for spec in specs:
        h = spec["height"]
        w = spec["width"]
        if h + 2 >= height or w + 2 >= width:
            return None
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(1, height - h - 1)
            left = rng.randint(1, width - w - 1)
            r0 = max(0, top - gap)
            r1 = min(height, top + h + gap)
            c0 = max(0, left - gap)
            c1 = min(width, left + w + gap)
            if blocked[r0:r1, c0:c1].any():
                continue
            _paint_track(grid, top, left, spec)
            blocked[r0:r1, c0:c1] = True
            placed = True
            break
        if not placed:
            return None

    return grid if _validate_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
