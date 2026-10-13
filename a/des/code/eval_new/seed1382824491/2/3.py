from __future__ import annotations

import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _connected_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append({"color": color, "coords": coords})
    return components


def _segment_from_component(component: dict[str, object]) -> dict[str, object]:
    coords = list(component["coords"])
    color = int(component["color"])
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        r = rows[0]
        c0, c1 = min(cols), max(cols)
        expected = {(r, c) for c in range(c0, c1 + 1)}
        assert set(coords) == expected, "horizontal segment must be contiguous"
        cells = [(r, c) for c in range(c0, c1 + 1)]
        endpoints = (cells[0], cells[-1])
        anchor = cells[0]
        orientation = "h"
    else:
        assert len(cols) == 1, "segment must be axis-aligned"
        c = cols[0]
        r0, r1 = min(rows), max(rows)
        expected = {(r, c) for r in range(r0, r1 + 1)}
        assert set(coords) == expected, "vertical segment must be contiguous"
        cells = [(r, c) for r in range(r0, r1 + 1)]
        endpoints = (cells[0], cells[-1])
        anchor = cells[0]
        orientation = "v"

    return {
        "color": color,
        "cells": tuple(cells),
        "length": len(cells),
        "endpoints": endpoints,
        "anchor": anchor,
        "orientation": orientation,
    }


def _corner_fill_color(seg_a: dict[str, object], seg_b: dict[str, object]) -> int:
    len_a = int(seg_a["length"])
    len_b = int(seg_b["length"])
    if len_a < len_b:
        return int(seg_a["color"])
    if len_b < len_a:
        return int(seg_b["color"])
    return int(seg_a["color"]) if tuple(seg_a["anchor"]) < tuple(seg_b["anchor"]) else int(seg_b["color"])


def _endpoint_corner_targets(segment: dict[str, object]) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    cells = list(segment["cells"])
    if segment["orientation"] == "h":
        left = cells[0]
        right = cells[-1]
        return (
            (left, (int(left[0]), int(left[1] - 1))),
            (right, (int(right[0]), int(right[1] + 1))),
        )
    top = cells[0]
    bottom = cells[-1]
    return (
        (top, (int(top[0] - 1), int(top[1]))),
        (bottom, (int(bottom[0] + 1), int(bottom[1]))),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each missing pipe corner with the color determined by the two segments it connects.

    Input structure:
    - The background is the dominant color.
    - The foreground consists of several single-color straight segments, each one
      horizontal or vertical and at least two pixels long.
    - Segment endpoints are paired through currently empty background cells so the
      segments would form one or more closed orthogonal loops if those corner cells
      were filled.

    Rule:
    - Identify every maximal straight segment and its two endpoints.
    - For each endpoint, follow the one-cell continuation that lies away from the
      rest of its segment; matching endpoints from two segments point to the same
      background cell, which is the missing corner.
    - Color the corner with the shorter of the two connected segments; if the segment
      lengths are equal, use the segment whose top-left pixel is farther up-left.
    - Leave every original segment pixel unchanged.

    Key insight:
    The task is solved by reconstructing endpoint-to-corner connectivity and then
    applying a local comparison rule based on segment length, with a top-left
    tie-breaker when the two lengths match.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _most_common_color(g)

    segments = [_segment_from_component(comp) for comp in _connected_components(g, bg)]
    assert segments, "expected at least one segment"

    endpoint_to_segment: dict[tuple[int, int], int] = {}
    corner_to_endpoints: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = {}
    for idx, segment in enumerate(segments):
        assert 2 <= int(segment["length"]) <= 6, "segment length must be between 2 and 6"
        for endpoint in segment["endpoints"]:
            assert endpoint not in endpoint_to_segment, "segment endpoints must be unique"
            endpoint_to_segment[endpoint] = idx
        for endpoint, corner in _endpoint_corner_targets(segment):
            assert 0 <= corner[0] < g.shape[0] and 0 <= corner[1] < g.shape[1], "corner must stay in bounds"
            corner_to_endpoints.setdefault(corner, []).append((idx, endpoint))

    out = g.copy()
    endpoint_use_count = {point: 0 for point in endpoint_to_segment}

    for corner, touching in corner_to_endpoints.items():
        assert len(touching) == 2, "each corner must join exactly two endpoints"
        r, c = corner
        assert int(g[r, c]) == bg, "corner cell must start as background"
        seg_a_idx, pt_a = touching[0]
        seg_b_idx, pt_b = touching[1]
        assert seg_a_idx != seg_b_idx, "corner must connect two different segments"
        seg_a = segments[seg_a_idx]
        seg_b = segments[seg_b_idx]
        assert seg_a["orientation"] != seg_b["orientation"], "corner must join perpendicular segments"
        out[r, c] = _corner_fill_color(seg_a, seg_b)
        endpoint_use_count[pt_a] += 1
        endpoint_use_count[pt_b] += 1

    assert all(count == 1 for count in endpoint_use_count.values()), "each endpoint must participate in one corner"
    return out


def _normalize_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in vertices)
    min_c = min(c for _, c in vertices)
    return [(int(r - min_r), int(c - min_c)) for r, c in vertices]


def _transform_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    transformed = [(int(r), int(c)) for r, c in vertices]
    for _ in range(random.randint(0, 3)):
        height = max(r for r, _ in transformed)
        transformed = [(int(c), int(height - r)) for r, c in transformed]
    if random.random() < 0.5:
        width = max(c for _, c in transformed)
        transformed = [(int(r), int(width - c)) for r, c in transformed]
    return _normalize_vertices(transformed)


def _rectangle_vertices(equal: bool) -> list[tuple[int, int]]:
    if equal:
        side = random.randint(3, 7)
        height = side
        width = side
    else:
        while True:
            height = random.randint(3, 7)
            width = random.randint(3, 7)
            if height != width:
                break
    return [(0, 0), (0, width), (height, width), (height, 0)]


def _l_vertices() -> list[tuple[int, int]]:
    while True:
        p = random.randint(3, 4)
        q = random.randint(3, 4)
        r = random.randint(3, 4)
        s = random.randint(3, 4)
        if p + q > 7 or r + s > 7:
            continue
        lengths = [p + q, r, p, s, q, r + s]
        adjacent = [(lengths[i], lengths[(i + 1) % 6]) for i in range(6)]
        if any(a == b for a, b in adjacent):
            return [(0, 0), (0, p + q), (r, p + q), (r, q), (r + s, q), (r + s, 0)]


def _abstract_loops() -> list[list[tuple[int, int]]]:
    scenario = random.choice(("tie_only", "shorter_only", "mixed"))
    if scenario == "tie_only":
        count = random.choice((1, 2))
        return [_transform_vertices(_rectangle_vertices(equal=True)) for _ in range(count)]
    if scenario == "shorter_only":
        count = random.choice((1, 2))
        return [_transform_vertices(_rectangle_vertices(equal=False)) for _ in range(count)]
    if random.random() < 0.5:
        return [_transform_vertices(_l_vertices())]
    return [
        _transform_vertices(_rectangle_vertices(equal=True)),
        _transform_vertices(_rectangle_vertices(equal=False)),
    ]


def _line_points_inclusive(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(int(r0), int(c)) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(int(r), int(c0)) for r in range(r0, r1 + step, step)]


def _line_points_exclusive(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    pts = _line_points_inclusive(a, b)
    return pts[1:-1]


def _loop_bbox_shape(vertices: list[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in vertices) + 1, max(c for _, c in vertices) + 1


def _distribute_extra(total: int, mins: list[int]) -> tuple[int, ...]:
    values = list(mins)
    extra = int(total - sum(mins))
    for _ in range(extra):
        values[random.randrange(len(values))] += 1
    return tuple(int(v) for v in values)


def _choose_offsets(shapes: list[tuple[int, int]]) -> tuple[tuple[int, int], ...] | None:
    count = len(shapes)
    if count == 1:
        h0, w0 = shapes[0]
        min_h = max(8, h0 + 2)
        min_w = max(8, w0 + 2)
        if min_h > 30 or min_w > 30:
            return None
        grid_h = random.randint(min_h, 30)
        grid_w = random.randint(min_w, 30)
        r0 = random.randint(1, grid_h - h0 - 1)
        c0 = random.randint(1, grid_w - w0 - 1)
        return ((r0, c0),)

    assert count == 2, "generator only supports one or two loops"
    if random.random() < 0.5:
        h0, w0 = shapes[0]
        h1, w1 = shapes[1]
        min_h = max(8, max(h0, h1) + 2)
        min_w = max(10, w0 + w1 + 4)
        if min_h > 30 or min_w > 30:
            return None
        grid_h = random.randint(min_h, 30)
        grid_w = random.randint(min_w, 30)
        left, gap, _ = _distribute_extra(grid_w - w0 - w1, [1, 2, 1])
        r_a = random.randint(1, grid_h - h0 - 1)
        r_b = random.randint(1, grid_h - h1 - 1)
        c_a = left
        c_b = left + w0 + gap
        return ((r_a, c_a), (r_b, c_b))

    h0, w0 = shapes[0]
    h1, w1 = shapes[1]
    min_h = max(10, h0 + h1 + 4)
    min_w = max(8, max(w0, w1) + 2)
    if min_h > 30 or min_w > 30:
        return None
    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)
    top, gap, _ = _distribute_extra(grid_h - h0 - h1, [1, 2, 1])
    c_a = random.randint(1, grid_w - w0 - 1)
    c_b = random.randint(1, grid_w - w1 - 1)
    r_a = top
    r_b = top + h0 + gap
    return ((r_a, c_a), (r_b, c_b))


def _build_loop(
    vertices: list[tuple[int, int]], offset: tuple[int, int], colors: list[int]
) -> tuple[list[dict[str, object]], list[tuple[int, int]], np.ndarray]:
    r0, c0 = offset
    placed_vertices = [(int(r + r0), int(c + c0)) for r, c in vertices]
    bbox_h, bbox_w = _loop_bbox_shape(vertices)
    boundary = np.zeros((bbox_h, bbox_w), dtype=bool)
    segments: list[dict[str, object]] = []
    count = len(placed_vertices)

    for idx in range(count):
        a_local = vertices[idx]
        b_local = vertices[(idx + 1) % count]
        for rr, cc in _line_points_inclusive(a_local, b_local):
            boundary[rr, cc] = True

        a = placed_vertices[idx]
        b = placed_vertices[(idx + 1) % count]
        cells = _line_points_exclusive(a, b)
        assert 2 <= len(cells) <= 6, "generated segment length must stay in range"
        segments.append(
            {
                "color": int(colors[idx]),
                "cells": tuple(cells),
                "length": len(cells),
                "anchor": min(cells),
                "endpoints": (cells[0], cells[-1]),
            }
        )

    return segments, placed_vertices, boundary


def _expected_output(inp: np.ndarray, loops: list[tuple[list[dict[str, object]], list[tuple[int, int]]]]) -> np.ndarray:
    out = inp.copy()
    for segments, corners in loops:
        count = len(segments)
        for idx, corner in enumerate(corners):
            prev_seg = segments[(idx - 1) % count]
            next_seg = segments[idx]
            out[corner] = _corner_fill_color(prev_seg, next_seg)
    return out


def _gen_attempt() -> np.ndarray | None:
    abstract = _abstract_loops()
    shapes = [_loop_bbox_shape(vertices) for vertices in abstract]
    offsets = _choose_offsets(shapes)
    if offsets is None:
        return None

    total_segments = sum(len(vertices) for vertices in abstract)
    if total_segments > 9:
        return None

    colors = random.sample(range(1, 10), total_segments)
    grid_h = max(offset[0] + shape[0] for offset, shape in zip(offsets, shapes)) + 1
    grid_w = max(offset[1] + shape[1] for offset, shape in zip(offsets, shapes)) + 1
    if grid_h > 30 or grid_w > 30:
        return None

    inp = np.zeros((grid_h, grid_w), dtype=int)
    loops: list[tuple[list[dict[str, object]], list[tuple[int, int]]]] = []
    color_index = 0
    for vertices, offset in zip(abstract, offsets):
        segment_count = len(vertices)
        loop_colors = colors[color_index : color_index + segment_count]
        color_index += segment_count
        segments, corners, _ = _build_loop(vertices, offset, loop_colors)
        for segment in segments:
            for r, c in segment["cells"]:
                assert inp[r, c] == 0, "segments must not overlap"
                inp[r, c] = int(segment["color"])
        loops.append((segments, corners))

    expected = _expected_output(inp, loops)
    try:
        if not np.array_equal(main(inp), expected):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pipe-corner input")
