from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]

MIN_DIM = 14
MAX_DIM = 30
SEGMENT_COUNT_RANGE = (2, 5)
SEGMENT_LENGTH_RANGE = (3, 5)
GAP_LENGTH_RANGE = (1, 4)
MAX_ROUTE_SPAN = 26
MAX_GENERATION_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(grid: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _ordered_segment_cells(coords: np.ndarray) -> tuple[list[tuple[int, int]], str]:
    rows = sorted({int(r) for r, _c in coords})
    cols = sorted({int(c) for _r, c in coords})
    if len(rows) == 1:
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        assert len(cols) == c1 - c0 + 1, "horizontal segment must be contiguous"
        return ([(row, c) for c in range(c0, c1 + 1)], "H")
    if len(cols) == 1:
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        assert len(rows) == r1 - r0 + 1, "vertical segment must be contiguous"
        return ([(r, col) for r in range(r0, r1 + 1)], "V")
    raise AssertionError("component must be a straight line segment")


def _analyze_segments(grid: np.ndarray, bg: int) -> list[dict]:
    segments: list[dict] = []
    for seg_id, coords in enumerate(_find_components(grid, bg)):
        ordered, orientation = _ordered_segment_cells(coords)
        length = len(ordered)
        assert length >= 2, "segment must have at least two cells"
        endpoint_colors = [int(grid[ordered[0]]), int(grid[ordered[-1]])]
        assert endpoint_colors[0] != endpoint_colors[1], "segment endpoints must differ"
        body_len = max(0, length - 2)
        if body_len:
            body_colors = {int(grid[cell]) for cell in ordered[1:-1]}
            assert len(body_colors) == 1, "segment body must be monochrome"
            body_color = next(iter(body_colors))
            assert body_color not in endpoint_colors, "body color must differ from endpoints"
        else:
            body_color = None
        if orientation == "H":
            outward = [(0, -1), (0, 1)]
        else:
            outward = [(-1, 0), (1, 0)]
        endpoints = []
        for idx, cell in enumerate((ordered[0], ordered[-1])):
            endpoints.append(
                {
                    "segment": seg_id,
                    "pos": cell,
                    "color": int(grid[cell]),
                    "outward": outward[idx],
                }
            )
        segments.append(
            {
                "id": seg_id,
                "cells": ordered,
                "orientation": orientation,
                "length": length,
                "body_len": body_len,
                "body_color": body_color,
                "endpoints": endpoints,
            }
        )
    assert segments, "expected at least one segment"
    return segments


def _clear_between(grid: np.ndarray, bg: int, a: tuple[int, int], b: tuple[int, int]) -> bool:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if ac < bc else -1
        for c in range(ac + step, bc, step):
            if int(grid[ar, c]) != bg:
                return False
        return True
    if ac == bc:
        step = 1 if ar < br else -1
        for r in range(ar + step, br, step):
            if int(grid[r, ac]) != bg:
                return False
        return True
    return False


def _matching_forms_chain(endpoints: list[dict], chosen: list[tuple[int, int]], segment_count: int) -> bool:
    if segment_count == 1:
        return not chosen
    deg = [0] * segment_count
    graph = [set() for _ in range(segment_count)]
    for a, b in chosen:
        sa = int(endpoints[a]["segment"])
        sb = int(endpoints[b]["segment"])
        if sa == sb:
            return False
        deg[sa] += 1
        deg[sb] += 1
        graph[sa].add(sb)
        graph[sb].add(sa)
    if any(d not in (1, 2) for d in deg):
        return False
    if sum(d == 1 for d in deg) != 2:
        return False
    stack = [0]
    seen = set()
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(graph[cur] - seen)
    return len(seen) == segment_count


def _choose_pairs(endpoints: list[dict], segment_count: int, candidates: list[tuple[int, int]]) -> list[tuple[int, int]]:
    target_edges = max(0, segment_count - 1)
    total_endpoints = len(endpoints)
    target_skips = total_endpoints - 2 * target_edges
    by_endpoint: dict[int, list[int]] = defaultdict(list)
    for edge_idx, (a, b) in enumerate(candidates):
        by_endpoint[a].append(edge_idx)
        by_endpoint[b].append(edge_idx)

    solution: list[tuple[int, int]] | None = None

    def rec(matched_mask: int, skipped_mask: int, chosen: list[tuple[int, int]]) -> bool:
        nonlocal solution
        matched_count = matched_mask.bit_count()
        skipped_count = skipped_mask.bit_count()
        handled = matched_count + skipped_count
        remaining = total_endpoints - handled
        if len(chosen) > target_edges or skipped_count > target_skips:
            return False
        if len(chosen) + remaining // 2 < target_edges:
            return False
        if handled == total_endpoints:
            if len(chosen) == target_edges and skipped_count == target_skips:
                if _matching_forms_chain(endpoints, chosen, segment_count):
                    solution = chosen.copy()
                    return True
            return False

        first = next(
            idx
            for idx in range(total_endpoints)
            if not (matched_mask >> idx) & 1 and not (skipped_mask >> idx) & 1
        )

        if skipped_count < target_skips:
            if rec(matched_mask, skipped_mask | (1 << first), chosen):
                return True

        for edge_idx in by_endpoint[first]:
            a, b = candidates[edge_idx]
            other = b if a == first else a
            if (matched_mask >> other) & 1 or (skipped_mask >> other) & 1:
                continue
            new_mask = matched_mask | (1 << first) | (1 << other)
            if rec(new_mask, skipped_mask, chosen + [(first, other)]):
                return True
        return False

    found = rec(0, 0, [])
    assert found and solution is not None, "failed to find a unique chain matching"
    return solution


def _draw_line(out: np.ndarray, a: tuple[int, int], b: tuple[int, int], color: int) -> None:
    ar, ac = a
    br, bc = b
    if ar == br:
        c0, c1 = sorted((ac, bc))
        out[ar, c0 : c1 + 1] = color
        return
    if ac == bc:
        r0, r1 = sorted((ar, br))
        out[r0 : r1 + 1, ac] = color
        return
    raise AssertionError("line must be horizontal or vertical")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Join endpoint-colored line segments into one chain, then extend the two dangling ends.

    Input structure:
    - Black background.
    - Each foreground component is one short straight segment.
    - Endpoint colors encode which segment ends can connect.

    Rule:
    - Pair endpoints that share a color and can be linked by a clear horizontal or vertical line.
    - Draw each connector in the shared endpoint color.
    - After all pairings, extend the two unmatched endpoints outward along their own segment
      orientation by a length equal to that segment's body length.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    segments = _analyze_segments(g, bg)

    endpoints: list[dict] = []
    for segment in segments:
        endpoints.extend(segment["endpoints"])

    candidates: list[tuple[int, int]] = []
    for i in range(len(endpoints)):
        for j in range(i + 1, len(endpoints)):
            ep_i = endpoints[i]
            ep_j = endpoints[j]
            if int(ep_i["segment"]) == int(ep_j["segment"]):
                continue
            if int(ep_i["color"]) != int(ep_j["color"]):
                continue
            if _clear_between(g, bg, ep_i["pos"], ep_j["pos"]):
                candidates.append((i, j))

    chosen = _choose_pairs(endpoints, len(segments), candidates)
    matched = {idx for pair in chosen for idx in pair}

    out = g.copy()
    for a, b in chosen:
        color = int(endpoints[a]["color"])
        _draw_line(out, endpoints[a]["pos"], endpoints[b]["pos"], color)

    for idx, endpoint in enumerate(endpoints):
        if idx in matched:
            continue
        steps = int(segments[int(endpoint["segment"])]["body_len"])
        r, c = endpoint["pos"]
        dr, dc = endpoint["outward"]
        for _ in range(steps):
            r += dr
            c += dc
            assert _in_bounds(out.shape, r, c), "extension ray left the grid"
            out[r, c] = int(endpoint["color"])

    return out


def _turn_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (-dc, dr)


def _turn_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _add_step(a: tuple[int, int], direction: tuple[int, int]) -> tuple[int, int]:
    return (a[0] + direction[0], a[1] + direction[1])


def _split_margin(total_extra: int) -> tuple[int, int]:
    low = 1
    high = total_extra - 1
    left = random.randint(low, high)
    return left, total_extra - left


def _choose_next_direction(prev: tuple[int, int]) -> tuple[int, int]:
    candidates = [prev, _turn_left(prev), _turn_right(prev)]
    weights = [2, 3, 3]
    return random.choices(candidates, weights=weights, k=1)[0]


def _build_route(
    segment_lengths: list[int], gap_lengths: list[int]
) -> tuple[list[list[tuple[int, int]]], list[tuple[tuple[int, int], tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    n_segments = len(segment_lengths)
    directions = [random.choice(DIRS4)]
    for _ in range(2 * n_segments - 2):
        directions.append(_choose_next_direction(directions[-1]))

    current = (0, 0)
    occupied = {current}
    segments: list[list[tuple[int, int]]] = []
    connectors: list[tuple[tuple[int, int], tuple[int, int]]] = []
    min_r = max_r = 0
    min_c = max_c = 0

    def register(cell: tuple[int, int]) -> bool:
        nonlocal min_r, max_r, min_c, max_c
        if cell in occupied:
            return False
        occupied.add(cell)
        min_r = min(min_r, cell[0])
        max_r = max(max_r, cell[0])
        min_c = min(min_c, cell[1])
        max_c = max(max_c, cell[1])
        return (max_r - min_r + 1) <= MAX_ROUTE_SPAN and (max_c - min_c + 1) <= MAX_ROUTE_SPAN

    for idx, seg_len in enumerate(segment_lengths):
        seg_dir = directions[2 * idx]
        cells = [current]
        for _ in range(seg_len - 1):
            current = _add_step(current, seg_dir)
            if not register(current):
                return None
            cells.append(current)
        segments.append(cells)

        if idx == n_segments - 1:
            continue

        gap_dir = directions[2 * idx + 1]
        start = current
        for _ in range(gap_lengths[idx] + 1):
            current = _add_step(current, gap_dir)
            if not register(current):
                return None
        connectors.append((start, current))

    start_ray: list[tuple[int, int]] = []
    ray_dir = (-directions[0][0], -directions[0][1])
    probe = segments[0][0]
    for _ in range(segment_lengths[0] - 2):
        probe = _add_step(probe, ray_dir)
        if not register(probe):
            return None
        start_ray.append(probe)

    end_ray: list[tuple[int, int]] = []
    ray_dir = directions[-1]
    probe = segments[-1][-1]
    for _ in range(segment_lengths[-1] - 2):
        probe = _add_step(probe, ray_dir)
        if not register(probe):
            return None
        end_ray.append(probe)

    return segments, connectors, start_ray, end_ray


def _render_input(
    shape: tuple[int, int],
    segments: list[list[tuple[int, int]]],
    body_colors: list[int],
    endpoint_colors: list[int],
    offset: tuple[int, int],
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    off_r, off_c = offset
    for idx, cells in enumerate(segments):
        shifted = [(r + off_r, c + off_c) for r, c in cells]
        out[shifted[0]] = endpoint_colors[idx]
        out[shifted[-1]] = endpoint_colors[idx + 1]
        for cell in shifted[1:-1]:
            out[cell] = body_colors[idx]
    return out


def _render_expected(
    inp: np.ndarray,
    segments: list[list[tuple[int, int]]],
    connectors: list[tuple[tuple[int, int], tuple[int, int]]],
    start_ray: list[tuple[int, int]],
    end_ray: list[tuple[int, int]],
    endpoint_colors: list[int],
    offset: tuple[int, int],
) -> np.ndarray:
    out = inp.copy()
    off_r, off_c = offset
    for idx, (a, b) in enumerate(connectors):
        aa = (a[0] + off_r, a[1] + off_c)
        bb = (b[0] + off_r, b[1] + off_c)
        _draw_line(out, aa, bb, endpoint_colors[idx + 1])
    for cell in start_ray:
        out[cell[0] + off_r, cell[1] + off_c] = endpoint_colors[0]
    for cell in end_ray:
        out[cell[0] + off_r, cell[1] + off_c] = endpoint_colors[-1]
    return out


def _choose_colors(n_segments: int) -> tuple[list[int], list[int]]:
    endpoint_colors = random.sample(PALETTE, n_segments + 1)
    body_colors = []
    for idx in range(n_segments):
        forbidden = {endpoint_colors[idx], endpoint_colors[idx + 1]}
        choices = [color for color in PALETTE if color not in forbidden]
        body_colors.append(random.choice(choices))
    return endpoint_colors, body_colors


def _scene_bounds(
    segments: list[list[tuple[int, int]]], start_ray: list[tuple[int, int]], end_ray: list[tuple[int, int]]
) -> tuple[int, int, int, int]:
    cells = [cell for segment in segments for cell in segment] + start_ray + end_ray
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _gen_attempt() -> np.ndarray | None:
    n_segments = random.randint(*SEGMENT_COUNT_RANGE)
    segment_lengths = [random.randint(*SEGMENT_LENGTH_RANGE) for _ in range(n_segments)]
    gap_lengths = [random.randint(*GAP_LENGTH_RANGE) for _ in range(n_segments - 1)]

    built = _build_route(segment_lengths, gap_lengths)
    if built is None:
        return None
    segments, connectors, start_ray, end_ray = built

    min_r, min_c, max_r, max_c = _scene_bounds(segments, start_ray, end_ray)
    bbox_h = max_r - min_r + 1
    bbox_w = max_c - min_c + 1
    if bbox_h + 2 > MAX_DIM or bbox_w + 2 > MAX_DIM:
        return None

    min_h = max(MIN_DIM, bbox_h + 2)
    min_w = max(MIN_DIM, bbox_w + 2)
    max_h = min(MAX_DIM, bbox_h + 8)
    max_w = min(MAX_DIM, bbox_w + 8)
    if min_h > max_h or min_w > max_w:
        return None

    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)
    if bbox_h >= height - 1 or bbox_w >= width - 1:
        return None

    top_margin, bottom_margin = _split_margin(height - bbox_h)
    left_margin, right_margin = _split_margin(width - bbox_w)
    assert bottom_margin >= 1 and right_margin >= 1
    offset = (top_margin - min_r, left_margin - min_c)

    endpoint_colors, body_colors = _choose_colors(n_segments)
    inp = _render_input((height, width), segments, body_colors, endpoint_colors, offset)
    expected = _render_expected(inp, segments, connectors, start_ray, end_ray, endpoint_colors, offset)

    if int(np.count_nonzero(inp)) >= (height * width) // 2:
        return None
    if len(_find_components(inp, BG)) != n_segments:
        return None
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
