from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))
UP, RIGHT, DOWN, LEFT = DIRS
GEN_RETRIES = 800


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _sub(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] - b[0], a[1] - b[1]


def _neg(d: tuple[int, int]) -> tuple[int, int]:
    return -d[0], -d[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS:
        yield r + dr, c + dc


def _line_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    sr, sc = start
    dr, dc = direction
    return [(sr + k * dr, sc + k * dc) for k in range(length)]


def _components(g: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for rr, cc in np.argwhere(g == color):
        r, c = int(rr), int(cc)
        if seen[r, c]:
            continue
        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)
    return comps


def _parse_segment(g: np.ndarray, comp: list[tuple[int, int]], color: int) -> dict:
    rows = [r for r, _c in comp]
    cols = [c for _r, c in comp]
    assert len(comp) >= 2, "line segments must have length at least two"

    if len(set(rows)) == 1:
        r = rows[0]
        c0, c1 = min(cols), max(cols)
        expected = {(r, c) for c in range(c0, c1 + 1)}
        endpoints = ((r, c0), (r, c1))
    elif len(set(cols)) == 1:
        c = cols[0]
        r0, r1 = min(rows), max(rows)
        expected = {(r, c) for r in range(r0, r1 + 1)}
        endpoints = ((r0, c), (r1, c))
    else:
        raise AssertionError("each colored component must be one horizontal or vertical segment")

    assert set(comp) == expected, "line segment cells must be contiguous"

    h, w = g.shape
    marked: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for endpoint in endpoints:
        er, ec = endpoint
        grey_neighbors = []
        for nr, nc in _neighbors4(er, ec):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == GREY:
                grey_neighbors.append((nr, nc))
        if grey_neighbors:
            assert len(grey_neighbors) == 1, "an endpoint may touch only one Grey direction marker"
            marked.append((endpoint, _sub(grey_neighbors[0], endpoint)))

    assert len(marked) == 1, "each segment must have exactly one marked endpoint"
    marked_endpoint, marker_dir = marked[0]
    unmarked_endpoint = endpoints[1] if endpoints[0] == marked_endpoint else endpoints[0]

    body_delta = _sub(unmarked_endpoint, marked_endpoint)
    if body_delta[0] == 0:
        body_dir = (0, 1 if body_delta[1] > 0 else -1)
        length = abs(body_delta[1]) + 1
    else:
        body_dir = (1 if body_delta[0] > 0 else -1, 0)
        length = abs(body_delta[0]) + 1

    assert marker_dir in DIRS, "Grey marker must be in a four-neighbor direction"
    assert marker_dir != body_dir, "direction marker must not overlap the segment body"
    return {
        "color": int(color),
        "marker_dir": marker_dir,
        "body_dir": body_dir,
        "length": int(length),
        "marked_endpoint": marked_endpoint,
    }


def _order_segments(segments: list[dict]) -> list[dict]:
    valid_orders: list[list[int]] = []
    n = len(segments)

    for start in range(n):
        order = [start]
        unused = set(range(n)) - {start}
        current = start
        while unused:
            out_dir = segments[current]["body_dir"]
            matches = [idx for idx in unused if segments[idx]["marker_dir"] == out_dir]
            if len(matches) != 1:
                break
            current = matches[0]
            order.append(current)
            unused.remove(current)
        if not unused:
            valid_orders.append(order)

    assert len(valid_orders) == 1, "direction markers must encode one unique segment order"
    return [segments[i] for i in valid_orders[0]]


def _relative_polyline(ordered: list[dict]) -> list[tuple[int, int]]:
    cur = (0, 0)
    cells: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    for seg_i, seg in enumerate(ordered):
        direction = seg["body_dir"]
        length = int(seg["length"])
        for k, cell in enumerate(_line_cells(cur, direction, length)):
            if seg_i > 0 and k == 0:
                assert cell in seen, "adjacent segments must share one endpoint"
                continue
            assert cell not in seen, "assembled polyline must not overlap itself"
            cells.append(cell)
            seen.add(cell)
        cur = _add(cur, ((length - 1) * direction[0], (length - 1) * direction[1]))
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stitch same-colored scattered line segments by Grey endpoint directions and move each stitched polyline to its anchor.

    Input structure:
    - The background is black.
    - Away from the bottom row, each foreground color forms two to four separated
      one-cell-wide horizontal or vertical line segments.
    - Each segment has exactly one Grey pixel in a four-neighbor cell beside one
      of its endpoints; the offset from that endpoint is the segment's direction
      marker.
    - The bottom row contains exactly one isolated anchor pixel for each segment
      color.

    Rule:
    - Group line segments by color and orient each segment from its Grey-marked
      endpoint to its unmarked endpoint.
    - For each color, find the only ordering in which the previous segment's
      body direction equals the next segment's Grey marker direction.
    - Assemble the ordered segments into one polyline, overlapping consecutive
      segments at the shared endpoint pixel.
    - Translate the whole polyline so its starting endpoint lands on the bottom
      row anchor of the same color.
    - Return a black canvas with only the translated polylines; discard Grey
      markers, scattered source segments, and anchor-only pixels.

    Key insight:
    - The Grey pixel is not output content; its offset is a symbolic direction
      that orders each color's segments before the completed polyline is moved
      onto the visible color anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must fit within 30x30"

    anchor_row = g[h - 1]
    anchors: dict[int, tuple[int, int]] = {}
    for c, value in enumerate(anchor_row):
        color = int(value)
        if color in (BG, GREY):
            continue
        assert color not in anchors, "each color must have exactly one bottom-row anchor"
        anchors[color] = (h - 1, c)
    assert anchors, "expected at least one anchor"

    work = g.copy()
    work[h - 1, :] = BG
    source_colors = {int(v) for v in np.unique(work) if int(v) not in (BG, GREY)}
    assert source_colors == set(anchors), "anchors and segment colors must match"

    out = np.zeros_like(g)
    for color, anchor in anchors.items():
        comps = _components(work, color)
        assert 2 <= len(comps) <= 4, "each color must have two to four segments"
        segments = [_parse_segment(work, comp, color) for comp in comps]
        ordered = _order_segments(segments)
        rel_cells = _relative_polyline(ordered)

        ar, ac = anchor
        for rr, cc in rel_cells:
            dst = (ar + rr, ac + cc)
            dr, dc = dst
            assert 0 <= dr < h and 0 <= dc < w, "translated polyline must remain inside the canvas"
            assert int(out[dr, dc]) in (BG, color), "translated polylines must not overlap"
            out[dr, dc] = color

    return out


def _chain_is_unique(directions: list[tuple[int, int]], marker_dirs: list[tuple[int, int]]) -> bool:
    segments = [
        {"body_dir": body_dir, "marker_dir": marker_dir, "length": 2}
        for body_dir, marker_dir in zip(directions, marker_dirs)
    ]
    try:
        return [seg["body_dir"] for seg in _order_segments(segments)] == directions
    except AssertionError:
        return False


def _try_relative_path(n_segments: int, h: int, w: int):
    directions: list[tuple[int, int]] = []
    lengths: list[int] = []
    cells = {(0, 0)}
    cur = (0, 0)

    for seg_i in range(n_segments):
        if seg_i == 0:
            choices = [UP, LEFT, RIGHT]
        else:
            prev = directions[-1]
            choices = [d for d in DIRS if d != prev and d != _neg(prev)]
        random.shuffle(choices)

        picked = None
        for direction in choices:
            length_values = list(range(2, 6))
            random.shuffle(length_values)
            for length in length_values:
                new_cells = _line_cells(cur, direction, length)[1:]
                if any(r > 0 or r < -(h - 1) or abs(c) >= w for r, c in new_cells):
                    continue
                if any(cell in cells for cell in new_cells):
                    continue
                picked = (direction, length, new_cells)
                break
            if picked is not None:
                break

        if picked is None:
            return None

        direction, length, new_cells = picked
        directions.append(direction)
        lengths.append(length)
        cells.update(new_cells)
        cur = new_cells[-1]

    start_marker_choices = [d for d in DIRS if d != directions[0] and d != directions[-1]]
    random.shuffle(start_marker_choices)
    for start_marker in start_marker_choices:
        marker_dirs = [start_marker] + directions[:-1]
        if _chain_is_unique(directions, marker_dirs):
            return {
                "directions": directions,
                "lengths": lengths,
                "marker_dirs": marker_dirs,
                "cells": sorted(cells),
            }
    return None


def _translated_cells(rel_cells: list[tuple[int, int]], anchor_col: int, h: int) -> set[tuple[int, int]]:
    return {(h - 1 + r, anchor_col + c) for r, c in rel_cells}


def _sample_specs(h: int, w: int, colors: list[int]):
    specs = []
    occupied_out: set[tuple[int, int]] = set()
    anchor_cols: set[int] = set()

    for color in colors:
        placed = None
        for _ in range(200):
            n_segments = random.randint(2, 4)
            path = _try_relative_path(n_segments, h, w)
            if path is None:
                continue

            rel_cells = path["cells"]
            min_c = min(c for _r, c in rel_cells)
            max_c = max(c for _r, c in rel_cells)
            candidates = [
                c
                for c in range(-min_c, w - max_c)
                if c not in anchor_cols
                and all(abs(c - old) >= 2 for old in anchor_cols)
                and _translated_cells(rel_cells, c, h).isdisjoint(occupied_out)
            ]
            if not candidates:
                continue

            anchor_col = random.choice(candidates)
            out_cells = _translated_cells(rel_cells, anchor_col, h)
            placed = {
                "color": color,
                "anchor": (h - 1, anchor_col),
                "directions": path["directions"],
                "lengths": path["lengths"],
                "marker_dirs": path["marker_dirs"],
                "out_cells": out_cells,
            }
            break

        if placed is None:
            return None
        occupied_out.update(placed["out_cells"])
        anchor_cols.add(placed["anchor"][1])
        specs.append(placed)

    return specs


def _chebyshev_ring(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    ring = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    ring.add((nr, nc))
    return ring


def _segment_package(
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    marker_dir: tuple[int, int],
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    cells = _line_cells(start, direction, length)
    marker = _add(start, marker_dir)
    return cells, marker


def _place_scattered_segments(grid: np.ndarray, specs: list[dict]) -> bool:
    h, w = grid.shape
    blocked = {(h - 1, c) for c in range(w)}
    blocked.update({(h - 2, c) for c in range(w)})
    for spec in specs:
        blocked.update(_chebyshev_ring({spec["anchor"]}, h, w))

    items = []
    for spec in specs:
        color = int(spec["color"])
        for direction, length, marker_dir in zip(spec["directions"], spec["lengths"], spec["marker_dirs"]):
            items.append((color, direction, int(length), marker_dir))
    random.shuffle(items)

    for color, direction, length, marker_dir in items:
        starts = [(r, c) for r in range(h - 2) for c in range(w)]
        random.shuffle(starts)
        placed = False

        for start in starts:
            cells, marker = _segment_package(start, direction, length, marker_dir)
            package = set(cells) | {marker}
            if any(not (0 <= r < h - 2 and 0 <= c < w) for r, c in package):
                continue
            if marker in cells:
                continue
            if any(cell in blocked for cell in package):
                continue

            for r, c in cells:
                grid[r, c] = color
            mr, mc = marker
            grid[mr, mc] = GREY
            blocked.update(_chebyshev_ring(package, h, w))
            placed = True
            break

        if not placed:
            return False

    return True


def _expected_output(shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["out_cells"]:
            if int(out[r, c]) not in (BG, color):
                raise AssertionError("generated output paths overlap")
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    n_colors = random.randint(1, 3)
    colors = random.sample(COLORS, n_colors)

    specs = _sample_specs(h, w, colors)
    if specs is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        ar, ac = spec["anchor"]
        grid[ar, ac] = int(spec["color"])

    if not _place_scattered_segments(grid, specs):
        return None

    try:
        expected = _expected_output(grid.shape, specs)
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for stitched direction-marker polylines")
