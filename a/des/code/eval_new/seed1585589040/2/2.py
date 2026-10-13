from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


START_COLOR = 3
END_COLOR = 2
WORK_SIZE = 30
CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
PLACE_TRIES = 140
GEN_TRIES = 400


def _most_common_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _perpendiculars(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def _shift(point: tuple[int, int], direction: tuple[int, int], steps: int) -> tuple[int, int]:
    return point[0] + direction[0] * steps, point[1] + direction[1] * steps


def _in_bounds(point: tuple[int, int], h: int, w: int) -> bool:
    return 0 <= point[0] < h and 0 <= point[1] < w


def _open_interval(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] > a[1] else -1
        return [(a[0], c) for c in range(a[1] + step, b[1], step)]
    if a[1] == b[1]:
        step = 1 if b[0] > a[0] else -1
        return [(r, a[1]) for r in range(a[0] + step, b[0], step)]
    return []


def _clear_straight_line(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    bg: int,
) -> bool:
    if a[0] != b[0] and a[1] != b[1]:
        return False
    for r, c in _open_interval(a, b):
        if int(grid[r, c]) != bg:
            return False
    return True


def _paint_open_line(
    grid: np.ndarray,
    a: tuple[int, int],
    b: tuple[int, int],
    color: int,
) -> None:
    for r, c in _open_interval(a, b):
        grid[r, c] = color


def _component_info(color: int, cells: list[tuple[int, int]]) -> dict:
    cellset = set(cells)
    degrees = {}
    for r, c in cells:
        deg = sum((nr, nc) in cellset for nr, nc in _neighbors4(r, c))
        assert deg in {1, 2}, "fragment must be a simple path"
        degrees[(r, c)] = deg

    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "fragment must have exactly two endpoints"

    return {
        "color": int(color),
        "cells": list(cells),
        "size": len(cells),
        "top": min(r for r, _ in cells),
        "left": min(c for _, c in cells),
        "endpoints": tuple(sorted(endpoints)),
    }


def _extract_fragments(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    fragments = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if seen[r, c] or color in {bg, START_COLOR, END_COLOR}:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if not _in_bounds((nr, nc), h, w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            fragments.append(_component_info(color, cells))

    return fragments


def _other_endpoint(fragment: dict, endpoint: tuple[int, int]) -> tuple[int, int]:
    a, b = fragment["endpoints"]
    return b if endpoint == a else a


def _unique_connectable_endpoint(
    source: tuple[int, int],
    fragment: dict,
    grid: np.ndarray,
    bg: int,
) -> tuple[int, int]:
    matches = [
        endpoint
        for endpoint in fragment["endpoints"]
        if _clear_straight_line(grid, source, endpoint, bg)
    ]
    assert len(matches) == 1, "connection to fragment must be unique"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect the scattered colored path fragments into one routed chain from the green start to the red end.

    Input structure:
    - The background is the most common color.
    - Several disconnected one-pixel-wide path fragments are drawn in distinct non-background colors; each fragment is either a straight segment or an L-shaped segment.
    - A single green pixel marks the start and a single red pixel marks the end.
    - The start and end markers are isolated from the colored fragments.

    Rule:
    - Sort the path fragments by increasing pixel count; when counts tie, sort by the fragment's topmost row.
    - Connect the green start marker to the reachable endpoint of the first fragment with a straight green horizontal or vertical line.
    - For each next fragment in that order, connect the previous fragment's remaining endpoint to the reachable endpoint of the next fragment with a straight green line.
    - Connect the last fragment's remaining endpoint to the red end marker, while leaving every original fragment in its own color.

    Key insight:
    - The assembly order comes from each fragment's own size rather than from spatial position, and every new bridge is the unique row- or column-aligned gap between consecutive endpoints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    start_pts = np.argwhere(g == START_COLOR)
    end_pts = np.argwhere(g == END_COLOR)
    assert start_pts.shape == (1, 2)
    assert end_pts.shape == (1, 2)
    start = tuple(map(int, start_pts[0]))
    end = tuple(map(int, end_pts[0]))

    fragments = _extract_fragments(g, bg)
    assert fragments, "expected at least one fragment"
    fragments.sort(key=lambda frag: (frag["size"], frag["top"], frag["left"]))

    out = g.copy()

    first = fragments[0]
    incoming = _unique_connectable_endpoint(start, first, g, bg)
    _paint_open_line(out, start, incoming, START_COLOR)
    current = _other_endpoint(first, incoming)

    for fragment in fragments[1:]:
        incoming = _unique_connectable_endpoint(current, fragment, g, bg)
        _paint_open_line(out, current, incoming, START_COLOR)
        current = _other_endpoint(fragment, incoming)

    assert _clear_straight_line(g, current, end, bg), "final connection must be straight and clear"
    _paint_open_line(out, current, end, START_COLOR)
    return out


def _touches_existing(candidate: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if (nr, nc) in occupied and (nr, nc) not in candidate:
                    return True
    return False


def _shape_options(
    start: tuple[int, int],
    size: int,
    prefer_l: bool,
) -> list[tuple[list[tuple[int, int]], tuple[int, int]]]:
    options: list[tuple[list[tuple[int, int]], tuple[int, int]]] = []
    straight_dirs = list(CARDINALS)
    random.shuffle(straight_dirs)
    l_dirs = list(CARDINALS)
    random.shuffle(l_dirs)

    if prefer_l and size >= 3:
        splits = [(a, size - a + 1) for a in range(2, size) if size - a + 1 >= 2]
        random.shuffle(splits)
        for arm1, arm2 in splits:
            for d1 in l_dirs:
                d2s = _perpendiculars(d1)
                random.shuffle(d2s)
                corner = _shift(start, d1, arm1 - 1)
                arm = [_shift(start, d1, k) for k in range(arm1)]
                for d2 in d2s:
                    cells = arm + [_shift(corner, d2, k) for k in range(1, arm2)]
                    options.append((cells, cells[-1]))

    for d in straight_dirs:
        cells = [_shift(start, d, k) for k in range(size)]
        options.append((cells, cells[-1]))

    if not prefer_l and size >= 3:
        splits = [(a, size - a + 1) for a in range(2, size) if size - a + 1 >= 2]
        random.shuffle(splits)
        for arm1, arm2 in splits:
            for d1 in l_dirs:
                d2s = _perpendiculars(d1)
                random.shuffle(d2s)
                corner = _shift(start, d1, arm1 - 1)
                arm = [_shift(start, d1, k) for k in range(arm1)]
                for d2 in d2s:
                    cells = arm + [_shift(corner, d2, k) for k in range(1, arm2)]
                    options.append((cells, cells[-1]))

    return options


def _sample_sizes(count: int) -> list[int]:
    sizes = [random.randint(2, 4)]
    while len(sizes) < count:
        prev = sizes[-1]
        if random.random() < 0.35 and prev < 8:
            sizes.append(prev)
        else:
            sizes.append(min(8, prev + random.randint(1, 2)))
    return sizes


def _crop_layout(layout: dict) -> dict:
    points = [layout["start"], layout["end"]]
    for fragment in layout["fragments"]:
        points.extend(fragment["cells"])

    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)

    top_margin = random.randint(0, min(3, r0))
    left_margin = random.randint(0, min(3, c0))
    bottom_margin = random.randint(0, min(3, WORK_SIZE - 1 - r1))
    right_margin = random.randint(0, min(3, WORK_SIZE - 1 - c1))

    rr0 = r0 - top_margin
    rr1 = r1 + bottom_margin
    cc0 = c0 - left_margin
    cc1 = c1 + right_margin

    def shift_point(point: tuple[int, int]) -> tuple[int, int]:
        return point[0] - rr0, point[1] - cc0

    cropped = {
        "bg": layout["bg"],
        "start": shift_point(layout["start"]),
        "end": shift_point(layout["end"]),
        "connectors": [(shift_point(a), shift_point(b)) for a, b in layout["connectors"]],
        "fragments": [],
        "shape": (rr1 - rr0 + 1, cc1 - cc0 + 1),
    }

    for fragment in layout["fragments"]:
        cropped["fragments"].append(
            {
                "color": fragment["color"],
                "size": fragment["size"],
                "cells": [shift_point(cell) for cell in fragment["cells"]],
            }
        )

    return cropped


def _render_input_and_output(layout: dict) -> tuple[np.ndarray, np.ndarray]:
    h, w = layout["shape"]
    bg = layout["bg"]
    inp = np.full((h, w), bg, dtype=int)
    inp[layout["start"]] = START_COLOR
    inp[layout["end"]] = END_COLOR

    for fragment in layout["fragments"]:
        for r, c in fragment["cells"]:
            inp[r, c] = fragment["color"]

    out = inp.copy()
    for a, b in layout["connectors"]:
        _paint_open_line(out, a, b, START_COLOR)

    return inp, out


def _build_layout() -> tuple[np.ndarray, np.ndarray] | None:
    bg_choices = [color for color in range(10) if color not in {START_COLOR, END_COLOR}]
    bg = random.choice(bg_choices)
    frag_count = random.randint(3, 5)
    sizes = _sample_sizes(frag_count)
    frag_colors = random.sample([c for c in range(10) if c not in {bg, START_COLOR, END_COLOR}], frag_count)

    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()
    connectors: list[tuple[tuple[int, int], tuple[int, int]]] = []
    fragments = []

    start = (random.randint(7, 22), random.randint(7, 22))
    occupied.add(start)
    blocked.add(start)
    current = start
    prev_same_size_top: int | None = None
    prev_size: int | None = None

    for idx, size in enumerate(sizes):
        placed = False
        equal_size_top_floor = prev_same_size_top if prev_size == size else None

        for _ in range(PLACE_TRIES):
            conn_dir = random.choice(CARDINALS)
            conn_len = random.randint(2, 5)
            endpoint_in = _shift(current, conn_dir, conn_len)
            gap = _open_interval(current, endpoint_in)

            if not _in_bounds(endpoint_in, WORK_SIZE, WORK_SIZE):
                continue
            if any(not _in_bounds(cell, WORK_SIZE, WORK_SIZE) for cell in gap):
                continue
            if endpoint_in in blocked or any(cell in blocked for cell in gap):
                continue

            prefer_l = size >= 3 and random.random() < 0.5
            for cells, endpoint_out in _shape_options(endpoint_in, size, prefer_l):
                cellset = set(cells)
                if len(cellset) != len(cells):
                    continue
                if any(not _in_bounds(cell, WORK_SIZE, WORK_SIZE) for cell in cellset):
                    continue
                if cellset & blocked:
                    continue
                if _touches_existing(cellset, occupied):
                    continue

                top = min(r for r, _ in cellset)
                if equal_size_top_floor is not None and top <= equal_size_top_floor:
                    continue

                blocked.update(gap)
                blocked.update(cellset)
                occupied.update(cellset)
                connectors.append((current, endpoint_in))
                fragments.append(
                    {
                        "color": frag_colors[idx],
                        "size": size,
                        "cells": cells,
                    }
                )
                current = endpoint_out
                prev_same_size_top = top
                prev_size = size
                placed = True
                break

            if placed:
                break

        if not placed:
            return None

    end = None
    for _ in range(PLACE_TRIES):
        conn_dir = random.choice(CARDINALS)
        conn_len = random.randint(2, 5)
        candidate_end = _shift(current, conn_dir, conn_len)
        gap = _open_interval(current, candidate_end)
        if not _in_bounds(candidate_end, WORK_SIZE, WORK_SIZE):
            continue
        if any(not _in_bounds(cell, WORK_SIZE, WORK_SIZE) for cell in gap):
            continue
        if candidate_end in blocked or any(cell in blocked for cell in gap):
            continue
        if _touches_existing({candidate_end}, occupied):
            continue
        blocked.update(gap)
        blocked.add(candidate_end)
        occupied.add(candidate_end)
        connectors.append((current, candidate_end))
        end = candidate_end
        break

    if end is None:
        return None

    layout = {
        "bg": bg,
        "start": start,
        "end": end,
        "connectors": connectors,
        "fragments": fragments,
    }
    cropped = _crop_layout(layout)
    inp, expected = _render_input_and_output(cropped)

    if _most_common_color(inp) != bg:
        return None
    if inp.shape[0] > 30 or inp.shape[1] > 30:
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        built = _build_layout()
        if built is None:
            continue
        inp, expected = built
        try:
            out = main(inp)
        except AssertionError:
            continue
        if np.array_equal(out, expected):
            return inp
    raise RuntimeError("failed to generate a valid input after repeated attempts")
