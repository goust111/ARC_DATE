from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400
MIN_BOX = 7
BOX_GAP = 2


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], pos: tuple[int, int]) -> bool:
    r, c = pos
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _neighbors4(shape: tuple[int, int], pos: tuple[int, int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        nxt = int(pos[0] + dr), int(pos[1] + dc)
        if _in_bounds(shape, nxt):
            out.append(nxt)
    return out


def _perp_offsets(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    return (-dc, dr), (dc, -dr)


def _line_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(int(c0), int(c1 + step), step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(int(r), int(c0)) for r in range(int(r0), int(r1 + step), step)]
    raise ValueError("polyline segments must be axis-aligned")


def _polyline_cells(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for i in range(len(points) - 1):
        seg = _line_cells(points[i], points[i + 1])
        if cells:
            seg = seg[1:]
        cells.extend(seg)
    if len(cells) != len(set(cells)):
        raise ValueError("path self-overlaps")
    return cells


def _find_signal_paths(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    counts = Counter(int(v) for v in g.ravel())
    signals: list[dict[str, object]] = []

    for color, count in counts.items():
        if color == bg or count != 1:
            continue

        pos_arr = np.argwhere(g == color)
        if pos_arr.shape[0] != 1:
            continue
        signal_pos = int(pos_arr[0, 0]), int(pos_arr[0, 1])

        neighbor_colors = [
            int(g[r, c])
            for r, c in _neighbors4(g.shape, signal_pos)
            if int(g[r, c]) not in {bg, color}
        ]
        if len(neighbor_colors) != 2 or neighbor_colors[0] != neighbor_colors[1]:
            continue

        path_color = int(neighbor_colors[0])
        allowed = {int(color), int(path_color)}
        queue = deque([signal_pos])
        seen = {signal_pos}
        while queue:
            cur = queue.popleft()
            for nxt in _neighbors4(g.shape, cur):
                if nxt in seen or int(g[nxt]) not in allowed:
                    continue
                seen.add(nxt)
                queue.append(nxt)

        degrees: dict[tuple[int, int], int] = {}
        for cell in seen:
            deg = sum(1 for nxt in _neighbors4(g.shape, cell) if nxt in seen)
            degrees[cell] = deg
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]

        assert len(endpoints) == 2, "each signal-bearing path must have two endpoints"
        assert all(deg in {1, 2} for deg in degrees.values()), "paths must be simple non-branching chains"

        signals.append(
            {
                "signal_color": int(color),
                "signal_pos": signal_pos,
                "path_color": int(path_color),
                "cells": seen,
                "endpoints": endpoints,
            }
        )

    assert signals, "expected at least one signal pixel"
    return signals


def _gate_info(
    grid: np.ndarray,
    path_cells: set[tuple[int, int]],
    endpoint: tuple[int, int],
) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    in_path_neighbors = [nxt for nxt in _neighbors4(g.shape, endpoint) if nxt in path_cells]
    assert len(in_path_neighbors) == 1
    prev = in_path_neighbors[0]
    direction = int(endpoint[0] - prev[0]), int(endpoint[1] - prev[1])
    middle = _add(endpoint, direction)
    assert _in_bounds(g.shape, middle)

    p0, p1 = _perp_offsets(direction)
    frame_a = _add(middle, p0)
    frame_b = _add(middle, p1)
    assert _in_bounds(g.shape, frame_a) and _in_bounds(g.shape, frame_b)

    gate_color = int(g[frame_a])
    assert gate_color != _get_bg(g)
    assert int(g[frame_b]) == gate_color

    middle_color = int(g[middle])
    assert middle_color in {_get_bg(g), gate_color}

    return {
        "endpoint": endpoint,
        "middle": middle,
        "open": middle_color == _get_bg(g),
        "gate_color": gate_color,
    }


def _distances_from(
    shape: tuple[int, int],
    cells: set[tuple[int, int]],
    start: tuple[int, int],
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], tuple[int, int]]]:
    queue = deque([start])
    dist = {start: 0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    while queue:
        cur = queue.popleft()
        for nxt in _neighbors4(shape, cur):
            if nxt not in cells or nxt in dist:
                continue
            dist[nxt] = int(dist[cur] + 1)
            parent[nxt] = cur
            queue.append(nxt)

    return dist, parent


def _path_to_start(
    parent: dict[tuple[int, int], tuple[int, int]],
    start: tuple[int, int],
    goal: tuple[int, int],
) -> list[tuple[int, int]]:
    out = [goal]
    cur = goal
    while cur != start:
        cur = parent[cur]
        out.append(cur)
    out.reverse()
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Propagate each path's signal toward the only reachable or nearest open gate and repaint the traversed path segment.

    Input structure:
    - The background is the dominant color.
    - One or more thin orthogonal paths are drawn in single colors, with exactly one distinct single-pixel signal embedded on each path.
    - Each path has a same-colored gate at both ends; each gate is a 3x1 or 1x3 bar perpendicular to the path, with two frame pixels around a middle pixel.
    - A gate is open when its middle pixel is background and closed when that middle pixel matches the gate frame color.

    Rule:
    - Recover each simple path by pairing the unique signal pixel with its surrounding path color, then identify the two endpoint gates from the endpoint geometry.
    - If both gates are closed, leave that path unchanged; if exactly one gate is open, the signal moves toward that open gate; if both gates are open, it moves toward the nearer endpoint.
    - Recolor every traversed path pixel between the signal's original location and the chosen endpoint to the signal color, restore the original signal cell to the path color, and place the signal into the chosen gate's middle cell.
    - All untouched path cells, closed gates, and non-participating objects remain unchanged.

    Key insight:
    - The gate state is encoded geometrically by the gate middle pixel, so the solver must first decode endpoint gates before deciding direction and then repainting the exact path suffix the signal traverses.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    path_infos = _find_signal_paths(g, bg)
    for info in path_infos:
        signal_pos = info["signal_pos"]
        signal_color = int(info["signal_color"])
        path_color = int(info["path_color"])
        path_cells = set(info["cells"])
        endpoints = list(info["endpoints"])

        gate_a = _gate_info(g, path_cells, endpoints[0])
        gate_b = _gate_info(g, path_cells, endpoints[1])
        assert int(gate_a["gate_color"]) == int(gate_b["gate_color"])

        if not bool(gate_a["open"]) and not bool(gate_b["open"]):
            continue

        dist, parent = _distances_from(g.shape, path_cells, signal_pos)
        da = int(dist[gate_a["endpoint"]])
        db = int(dist[gate_b["endpoint"]])

        if bool(gate_a["open"]) and bool(gate_b["open"]):
            assert da != db, "two open gates need a unique nearer endpoint"
            target = gate_a if da < db else gate_b
        else:
            target = gate_a if bool(gate_a["open"]) else gate_b

        target_path = _path_to_start(parent, signal_pos, target["endpoint"])
        for cell in target_path[1:]:
            out[cell] = signal_color
        out[signal_pos] = path_color
        out[target["middle"]] = signal_color

    return out


def _split_lengths(total: int, count: int, min_size: int) -> list[int] | None:
    if total < count * min_size:
        return None
    parts = [min_size] * count
    extra = total - count * min_size
    for _ in range(extra):
        parts[random.randrange(count)] += 1
    random.shuffle(parts)
    return parts


def _boxes_for_scene(h: int, w: int, count: int) -> list[tuple[int, int, int, int]] | None:
    if count == 1:
        return [(0, h - 1, 0, w - 1)]

    layouts: list[str] = []
    if h >= count * MIN_BOX + (count - 1) * BOX_GAP:
        layouts.append("rows")
    if w >= count * MIN_BOX + (count - 1) * BOX_GAP:
        layouts.append("cols")
    if not layouts:
        return None

    layout = random.choice(layouts)
    boxes: list[tuple[int, int, int, int]] = []
    if layout == "rows":
        heights = _split_lengths(h - (count - 1) * BOX_GAP, count, MIN_BOX)
        assert heights is not None
        top = 0
        for i, band_h in enumerate(heights):
            boxes.append((top, top + band_h - 1, 0, w - 1))
            top += band_h
            if i < count - 1:
                top += BOX_GAP
    else:
        widths = _split_lengths(w - (count - 1) * BOX_GAP, count, MIN_BOX)
        assert widths is not None
        left = 0
        for i, band_w in enumerate(widths):
            boxes.append((0, h - 1, left, left + band_w - 1))
            left += band_w
            if i < count - 1:
                left += BOX_GAP
    return boxes


def _build_straight_h(box: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = box
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 6:
        return None
    row = random.randint(r0 + 1, r1 - 1)
    start = random.randint(c0 + 1, c1 - 5)
    end = random.randint(start + 4, c1 - 1)
    return _polyline_cells([(row, start), (row, end)])


def _build_straight_v(box: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = box
    if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 3:
        return None
    col = random.randint(c0 + 1, c1 - 1)
    start = random.randint(r0 + 1, r1 - 5)
    end = random.randint(start + 4, r1 - 1)
    return _polyline_cells([(start, col), (end, col)])


def _build_l_shape(box: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = box
    for _ in range(20):
        start_left = random.choice([True, False])
        end_top = random.choice([True, False])

        ct_lo = c0 + 3 if start_left else c0 + 2
        ct_hi = c1 - 2 if start_left else c1 - 3
        rt_lo = r0 + 3 if end_top else r0 + 2
        rt_hi = r1 - 2 if end_top else r1 - 3
        if ct_lo > ct_hi or rt_lo > rt_hi:
            continue

        turn = random.randint(rt_lo, rt_hi), random.randint(ct_lo, ct_hi)
        if start_left:
            start = turn[0], random.randint(c0 + 1, turn[1] - 2)
        else:
            start = turn[0], random.randint(turn[1] + 2, c1 - 1)
        if end_top:
            end = random.randint(r0 + 1, turn[0] - 2), turn[1]
        else:
            end = random.randint(turn[0] + 2, r1 - 1), turn[1]

        path = _polyline_cells([start, turn, end])
        if len(path) >= 6:
            return path
    return None


def _build_dogleg(box: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    r0, r1, c0, c1 = box
    for _ in range(20):
        if random.choice([True, False]):
            if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 7:
                continue
            start_left = random.choice([True, False])
            rows = [r for r in range(r0 + 1, r1)]
            if len(rows) < 2:
                continue
            row_a = random.choice(rows)
            far_rows = [r for r in rows if abs(r - row_a) >= 2]
            if not far_rows:
                continue
            row_b = random.choice(far_rows)
            turn_col = random.randint(c0 + 3, c1 - 3)
            if start_left:
                start = row_a, random.randint(c0 + 1, turn_col - 2)
                end = row_b, random.randint(turn_col + 2, c1 - 1)
            else:
                start = row_a, random.randint(turn_col + 2, c1 - 1)
                end = row_b, random.randint(c0 + 1, turn_col - 2)
            turn1 = row_a, turn_col
            turn2 = row_b, turn_col
            path = _polyline_cells([start, turn1, turn2, end])
        else:
            if r1 - r0 + 1 < 7 or c1 - c0 + 1 < 5:
                continue
            start_top = random.choice([True, False])
            cols = [c for c in range(c0 + 1, c1)]
            if len(cols) < 2:
                continue
            col_a = random.choice(cols)
            far_cols = [c for c in cols if abs(c - col_a) >= 2]
            if not far_cols:
                continue
            col_b = random.choice(far_cols)
            turn_row = random.randint(r0 + 3, r1 - 3)
            if start_top:
                start = random.randint(r0 + 1, turn_row - 2), col_a
                end = random.randint(turn_row + 2, r1 - 1), col_b
            else:
                start = random.randint(turn_row + 2, r1 - 1), col_a
                end = random.randint(r0 + 1, turn_row - 2), col_b
            turn1 = turn_row, col_a
            turn2 = turn_row, col_b
            path = _polyline_cells([start, turn1, turn2, end])

        if len(path) >= 6:
            return path
    return None


def _random_path(box: tuple[int, int, int, int]) -> list[tuple[int, int]] | None:
    builders = [_build_straight_h, _build_straight_v, _build_l_shape, _build_dogleg]
    for _ in range(40):
        path = random.choice(builders)(box)
        if path is None or len(path) < 6:
            continue
        return path
    return None


def _gate_cells_for_path(path: list[tuple[int, int]]) -> tuple[dict[str, object], dict[str, object]]:
    start = path[0]
    start_dir = int(path[0][0] - path[1][0]), int(path[0][1] - path[1][1])
    mid_a = _add(start, start_dir)
    pa0, pa1 = _perp_offsets(start_dir)
    gate_a = {
        "endpoint": start,
        "middle": mid_a,
        "frames": [_add(mid_a, pa0), _add(mid_a, pa1)],
    }

    end = path[-1]
    end_dir = int(path[-1][0] - path[-2][0]), int(path[-1][1] - path[-2][1])
    mid_b = _add(end, end_dir)
    pb0, pb1 = _perp_offsets(end_dir)
    gate_b = {
        "endpoint": end,
        "middle": mid_b,
        "frames": [_add(mid_b, pb0), _add(mid_b, pb1)],
    }
    return gate_a, gate_b


def _valid_geometry(shape: tuple[int, int], path: list[tuple[int, int]]) -> bool:
    path_set = set(path)
    gate_a, gate_b = _gate_cells_for_path(path)
    reserved = set(path)
    for gate in (gate_a, gate_b):
        for cell in gate["frames"] + [gate["middle"]]:
            if not _in_bounds(shape, cell):
                return False
            if cell in reserved:
                return False
            reserved.add(cell)
    return True


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    path_color: int,
    gate_color: int,
    signal_color: int,
    open_a: bool,
    open_b: bool,
) -> None:
    for r, c in path:
        grid[r, c] = path_color

    gate_a, gate_b = _gate_cells_for_path(path)
    for cell in gate_a["frames"] + gate_b["frames"]:
        grid[cell] = gate_color
    if not open_a:
        grid[gate_a["middle"]] = gate_color
    if not open_b:
        grid[gate_b["middle"]] = gate_color

    candidates = list(range(1, len(path) - 1))
    core = [i for i in candidates if 1 < i < len(path) - 2]
    if core:
        candidates = core
    if open_a and open_b:
        candidates = [i for i in candidates if i != len(path) - 1 - i]
    signal_idx = random.choice(candidates)
    sr, sc = path[signal_idx]
    grid[sr, sc] = signal_color


def _scene_attempt() -> np.ndarray | None:
    path_count = random.randint(1, 3)
    h = random.randint(max(10, path_count * 4 + 6), 30)
    w = random.randint(max(10, path_count * 4 + 6), 30)
    boxes = _boxes_for_scene(h, w, path_count)
    if boxes is None:
        return None

    palette = random.sample(list(range(10)), 1 + 3 * path_count)
    bg = int(palette[0])
    grid = np.full((h, w), bg, dtype=int)

    for i, box in enumerate(boxes):
        path_color = int(palette[1 + 3 * i])
        gate_color = int(palette[2 + 3 * i])
        signal_color = int(palette[3 + 3 * i])
        branch = random.choice(
            [
                (False, False),
                (True, False),
                (False, True),
                (True, True),
            ]
        )

        path = _random_path(box)
        if path is None or not _valid_geometry(grid.shape, path):
            return None
        _paint_path(grid, path, path_color, gate_color, signal_color, branch[0], branch[1])

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _scene_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-signal ARC input")
