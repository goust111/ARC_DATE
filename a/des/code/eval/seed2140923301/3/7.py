from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CLOSED_VALVE = 2
OPEN_VALVE = 3
SOURCE_MARKER = 4
CONTAINER_WALL = 5
PIPE_COLORS = (1, 6, 7, 8, 9)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 500


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _find_segments(grid: np.ndarray) -> tuple[list[dict], dict[tuple[int, int], tuple[int, int]]]:
    reserved = {BG, CLOSED_VALVE, OPEN_VALVE, SOURCE_MARKER, CONTAINER_WALL}
    segments: list[dict] = []
    endpoint_by_cell: dict[tuple[int, int], tuple[int, int]] = {}

    for color in np.unique(grid):
        color = int(color)
        if color in reserved:
            continue
        for comp in _components(grid == color):
            cell_set = set(comp)
            rows = {r for r, _c in comp}
            cols = {c for _r, c in comp}
            if len(rows) != 1 and len(cols) != 1:
                continue

            endpoints = []
            for r, c in comp:
                degree = 0
                for dr, dc in DIRS4:
                    if (r + dr, c + dc) in cell_set:
                        degree += 1
                if degree <= 1:
                    endpoints.append((r, c))

            if len(comp) == 1:
                endpoints = [comp[0], comp[0]]
            elif len(endpoints) != 2:
                if len(rows) == 1:
                    endpoints = [
                        min(comp, key=lambda p: p[1]),
                        max(comp, key=lambda p: p[1]),
                    ]
                else:
                    endpoints = [
                        min(comp, key=lambda p: p[0]),
                        max(comp, key=lambda p: p[0]),
                    ]

            seg_id = len(segments)
            segments.append(
                {
                    "color": color,
                    "cells": cell_set,
                    "endpoints": (endpoints[0], endpoints[1]),
                }
            )
            for idx, cell in enumerate(endpoints[:2]):
                endpoint_by_cell[cell] = (seg_id, idx)

    return segments, endpoint_by_cell


def _find_valves(grid: np.ndarray) -> list[dict]:
    valves: list[dict] = []
    for color, is_open in ((OPEN_VALVE, True), (CLOSED_VALVE, False)):
        for comp in _components(grid == color):
            if len(comp) != 4:
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if r1 - r0 != 1 or c1 - c0 != 1:
                continue
            if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
                continue
            valves.append(
                {
                    "open": is_open,
                    "cells": set(comp),
                }
            )
    return valves


def _find_containers(grid: np.ndarray) -> list[dict]:
    containers: list[dict] = []
    for comp in _components(grid == CONTAINER_WALL):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(box[0, :] == CONTAINER_WALL):
            continue
        if not np.all(box[-1, :] == CONTAINER_WALL):
            continue
        if not np.all(box[:, 0] == CONTAINER_WALL):
            continue
        if not np.all(box[:, -1] == CONTAINER_WALL):
            continue
        if not np.all(box[1:-1, 1:-1] == BG):
            continue
        containers.append(
            {
                "cells": set(comp),
                "bbox": (r0, c0, r1, c1),
            }
        )
    return containers


def _find_sources(
    grid: np.ndarray, endpoint_by_cell: dict[tuple[int, int], tuple[int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = grid.shape
    sources: list[tuple[tuple[int, int], tuple[int, int]]] = []
    yellow_cells = sorted(map(tuple, np.argwhere(grid == SOURCE_MARKER)))
    for marker in yellow_cells:
        r, c = marker
        if r not in (0, h - 1) and c not in (0, w - 1):
            continue
        attached = []
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in endpoint_by_cell:
                attached.append(endpoint_by_cell[(nr, nc)])
        if len(attached) == 1:
            sources.append((marker, attached[0]))
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace colored fluid from border-marked pipe sources through open valves and
    fill any connected container interiors. Earlier sources in row-major scan
    order win conflicts on shared containers.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    segments, endpoint_by_cell = _find_segments(g)
    valves = _find_valves(g)
    containers = _find_containers(g)
    sources = _find_sources(g, endpoint_by_cell)

    other_end: dict[tuple[int, int], tuple[int, int]] = {}
    for seg_id, seg in enumerate(segments):
        other_end[(seg_id, 0)] = (seg_id, 1)
        other_end[(seg_id, 1)] = (seg_id, 0)

    valve_by_cell: dict[tuple[int, int], int] = {}
    for valve_id, valve in enumerate(valves):
        for cell in valve["cells"]:
            valve_by_cell[cell] = valve_id

    container_by_wall: dict[tuple[int, int], int] = {}
    for container_id, container in enumerate(containers):
        for cell in container["cells"]:
            container_by_wall[cell] = container_id

    valve_attachments: list[set[tuple[int, int]]] = [set() for _ in valves]
    endpoint_container: dict[tuple[int, int], int] = {}
    for cell, endpoint_id in endpoint_by_cell.items():
        r, c = cell
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in valve_by_cell:
                valve_attachments[valve_by_cell[(nr, nc)]].add(endpoint_id)
            if (nr, nc) in container_by_wall:
                endpoint_container[endpoint_id] = container_by_wall[(nr, nc)]

    valve_jump: dict[tuple[int, int], tuple[int, int]] = {}
    for valve, attached in zip(valves, valve_attachments, strict=False):
        if not valve["open"] or len(attached) != 2:
            continue
        a, b = sorted(attached)
        valve_jump[a] = b
        valve_jump[b] = a

    filled: set[int] = set()
    for _marker, start in sources:
        source_color = int(segments[start[0]]["color"])
        current = other_end.get(start)
        if current is None:
            continue
        visited_segments = {start[0]}

        while True:
            container_id = endpoint_container.get(current)
            if container_id is not None:
                if container_id not in filled:
                    r0, c0, r1, c1 = containers[container_id]["bbox"]
                    out[r0 + 1 : r1, c0 + 1 : c1] = source_color
                    filled.add(container_id)
                break

            nxt = valve_jump.get(current)
            if nxt is None:
                break

            seg_id = nxt[0]
            if seg_id in visited_segments:
                break
            visited_segments.add(seg_id)
            current = other_end[nxt]

    return out


def _rect_ring_cells(top: int, left: int, height: int, width: int) -> list[tuple[int, int]]:
    cells = []
    bottom = top + height - 1
    right = left + width - 1
    for c in range(left, right + 1):
        cells.append((top, c))
        cells.append((bottom, c))
    for r in range(top + 1, bottom):
        cells.append((r, left))
        cells.append((r, right))
    return cells


def _line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("line must be axis-aligned")


def _valve_cells(top: int, left: int) -> list[tuple[int, int]]:
    return [(top, left), (top, left + 1), (top + 1, left), (top + 1, left + 1)]


def _in_bounds(shape: tuple[int, int], cells: list[tuple[int, int]]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _draw_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> bool:
    if not _in_bounds(grid.shape, cells):
        return False
    for r, c in cells:
        if grid[r, c] != BG:
            return False
    for r, c in cells:
        grid[r, c] = int(color)
    return True


def _place_container(grid: np.ndarray, top: int, left: int, height: int, width: int) -> dict | None:
    cells = _rect_ring_cells(top, left, height, width)
    if not _draw_cells(grid, cells, CONTAINER_WALL):
        return None
    return {"top": top, "left": left, "height": height, "width": width}


def _place_horizontal_path(
    grid: np.ndarray,
    container: dict,
    row: int,
    valve_col: int,
    colors: tuple[int, int],
    valve_open: bool,
) -> bool:
    left = int(container["left"])
    top = int(container["top"])
    height = int(container["height"])
    if not (top + 1 <= row <= top + height - 2):
        return False
    if valve_col < 3 or valve_col > left - 4:
        return False

    valve_color = OPEN_VALVE if valve_open else CLOSED_VALVE
    items = [
        ([(row, 0)], SOURCE_MARKER),
        (_line_cells(row, 1, row, valve_col - 1), colors[0]),
        (_valve_cells(row, valve_col), valve_color),
        (_line_cells(row, valve_col + 2, row, left - 1), colors[1]),
    ]
    for cells, color in items:
        if not _draw_cells(grid, cells, color):
            return False
    return True


def _place_zigzag_path(
    grid: np.ndarray,
    container: dict,
    source_row: int,
    target_row: int,
    valve_col: int,
    colors: tuple[int, int, int],
    valve_open_1: bool,
    valve_open_2: bool,
) -> bool:
    left = int(container["left"])
    top = int(container["top"])
    height = int(container["height"])
    if not (top + 1 <= target_row <= top + height - 2):
        return False
    if source_row < 2 or target_row < source_row + 4:
        return False
    if valve_col < 4 or valve_col > left - 5:
        return False

    items = [
        ([(source_row, 0)], SOURCE_MARKER),
        (_line_cells(source_row, 1, source_row, valve_col - 1), colors[0]),
        (_valve_cells(source_row, valve_col), OPEN_VALVE if valve_open_1 else CLOSED_VALVE),
        (_line_cells(source_row + 2, valve_col, target_row - 1, valve_col), colors[1]),
        (_valve_cells(target_row, valve_col), OPEN_VALVE if valve_open_2 else CLOSED_VALVE),
        (_line_cells(target_row, valve_col + 2, target_row, left - 1), colors[2]),
    ]
    for cells, color in items:
        if not _draw_cells(grid, cells, color):
            return False
    return True


def _place_top_turn_path(
    grid: np.ndarray,
    container: dict,
    source_col: int,
    target_row: int,
    colors: tuple[int, int],
    valve_open: bool,
) -> bool:
    left = int(container["left"])
    top = int(container["top"])
    height = int(container["height"])
    if not (top + 1 <= target_row <= top + height - 2):
        return False
    if source_col < 2 or source_col > left - 5:
        return False

    items = [
        ([(0, source_col)], SOURCE_MARKER),
        (_line_cells(1, source_col, target_row - 1, source_col), colors[0]),
        (_valve_cells(target_row, source_col), OPEN_VALVE if valve_open else CLOSED_VALVE),
        (_line_cells(target_row, source_col + 2, target_row, left - 1), colors[1]),
    ]
    for cells, color in items:
        if not _draw_cells(grid, cells, color):
            return False
    return True


def _sample_pipe_colors(count: int) -> tuple[int, ...]:
    return tuple(random.sample(PIPE_COLORS, count))


def _build_solo_horizontal(h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    ch = random.randint(5, 7)
    cw = random.randint(5, 7)
    top = random.randint(2, h - ch - 3)
    left = random.randint(w // 2, w - cw - 2)
    container = _place_container(grid, top, left, ch, cw)
    if container is None:
        return None
    row = random.randint(top + 1, top + ch - 2)
    valve_col = random.randint(3, left - 4)
    colors = _sample_pipe_colors(2)
    if not _place_horizontal_path(grid, container, row, valve_col, colors, random.choice([True, False])):
        return None
    return grid


def _build_solo_zigzag(h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    ch = random.randint(5, 7)
    cw = random.randint(5, 7)
    top = random.randint(h // 2, h - ch - 3)
    left = random.randint(w // 2, w - cw - 2)
    container = _place_container(grid, top, left, ch, cw)
    if container is None:
        return None
    target_row = random.randint(top + 1, top + ch - 2)
    if target_row - 4 < 2:
        return None
    source_row = random.randint(2, target_row - 4)
    valve_col = random.randint(4, left - 5)
    colors = _sample_pipe_colors(3)
    if not _place_zigzag_path(
        grid,
        container,
        source_row,
        target_row,
        valve_col,
        colors,
        random.choice([True, False]),
        random.choice([True, False]),
    ):
        return None
    return grid


def _build_dual(h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    ch1 = random.randint(5, 7)
    cw1 = random.randint(5, 7)
    ch2 = random.randint(5, 7)
    cw2 = random.randint(5, 7)

    top1_hi = h // 2 - ch1 - 4
    if top1_hi < 2:
        return None
    top1 = random.randint(2, top1_hi)
    top2_lo = top1 + ch1 + 4
    top2_hi = h - ch2 - 3
    if top2_lo > top2_hi:
        return None
    top2 = random.randint(top2_lo, top2_hi)

    left1 = random.randint(w // 2, w - cw1 - 2)
    left2 = random.randint(w // 2, w - cw2 - 2)
    container1 = _place_container(grid, top1, left1, ch1, cw1)
    container2 = _place_container(grid, top2, left2, ch2, cw2)
    if container1 is None or container2 is None:
        return None

    row1 = random.randint(top1 + 1, top1 + ch1 - 2)
    valve_col1 = random.randint(3, left1 - 4)

    use_zigzag_second = random.choice([True, False])
    color_count = 5 if use_zigzag_second else 4
    colors = list(_sample_pipe_colors(color_count))

    if not _place_horizontal_path(
        grid,
        container1,
        row1,
        valve_col1,
        (colors[0], colors[1]),
        random.choice([True, False]),
    ):
        return None

    if use_zigzag_second:
        target_row2 = random.randint(top2 + 1, top2 + ch2 - 2)
        source_lo = max(top1 + ch1 + 2, 2)
        source_hi = target_row2 - 4
        if source_lo > source_hi:
            return None
        source_row2 = random.randint(source_lo, source_hi)
        valve_col2 = random.randint(4, left2 - 5)
        ok = _place_zigzag_path(
            grid,
            container2,
            source_row2,
            target_row2,
            valve_col2,
            (colors[2], colors[3], colors[4]),
            random.choice([True, False]),
            random.choice([True, False]),
        )
    else:
        row2 = random.randint(top2 + 1, top2 + ch2 - 2)
        valve_col2 = random.randint(3, left2 - 4)
        ok = _place_horizontal_path(
            grid,
            container2,
            row2,
            valve_col2,
            (colors[2], colors[3]),
            random.choice([True, False]),
        )
    if not ok:
        return None
    return grid


def _build_shared(h: int, w: int) -> np.ndarray | None:
    grid = np.full((h, w), BG, dtype=int)
    ch = random.randint(6, 7)
    cw = random.randint(5, 7)
    top = random.randint(h // 3, h - ch - 3)
    left = random.randint(w // 2, w - cw - 2)
    container = _place_container(grid, top, left, ch, cw)
    if container is None:
        return None

    row_top = random.randint(top + 1, top + ch - 4)
    row_left = random.randint(row_top + 2, top + ch - 2)
    source_col = random.randint(2, left - 5)

    use_zigzag_left = random.choice([True, False])
    color_count = 5 if use_zigzag_left else 4
    colors = list(_sample_pipe_colors(color_count))

    if not _place_top_turn_path(
        grid,
        container,
        source_col,
        row_top,
        (colors[0], colors[1]),
        random.choice([True, False]),
    ):
        return None

    if use_zigzag_left:
        source_lo = row_top + 3
        source_hi = row_left - 4
        if source_lo > source_hi:
            return None
        source_row = random.randint(source_lo, source_hi)
        valve_col = random.randint(4, left - 5)
        ok = _place_zigzag_path(
            grid,
            container,
            source_row,
            row_left,
            valve_col,
            (colors[2], colors[3], colors[4]),
            random.choice([True, False]),
            random.choice([True, False]),
        )
    else:
        valve_col = random.randint(3, left - 4)
        ok = _place_horizontal_path(
            grid,
            container,
            row_left,
            valve_col,
            (colors[2], colors[3]),
            random.choice([True, False]),
        )
    if not ok:
        return None
    return grid


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.rot90(grid, random.randint(0, 3))
    if random.choice([True, False]):
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def generate_input() -> np.ndarray:
    builders = [
        _build_solo_horizontal,
        _build_solo_zigzag,
        _build_dual,
        _build_shared,
    ]
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(20, 30)
        w = random.randint(20, 30)
        grid = random.choice(builders)(h, w)
        if grid is None:
            continue
        grid = _random_transform(grid)

        try:
            out = main(grid)
        except Exception:
            continue

        if not isinstance(out, np.ndarray) or out.shape != grid.shape:
            continue
        if np.count_nonzero(grid == SOURCE_MARKER) == 0:
            continue
        if np.count_nonzero(grid == CONTAINER_WALL) == 0:
            continue
        return grid
    raise RuntimeError("generate_input failed after repeated attempts")
