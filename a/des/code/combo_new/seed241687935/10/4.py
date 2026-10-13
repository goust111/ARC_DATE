import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if dr or dc]


# Generator configuration.
COLOR_POOL = tuple(range(1, 10))
SEGMENT_COUNT_RANGE = (4, 5)
SEGMENT_STEP_RANGE = (4, 6)
PATH_SPAN_LIMIT = 22
RECT_LONG_MAX = 4
RECT_SHORT_MAX = 4
GENERATION_ATTEMPTS = 400


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _bbox(cells) -> tuple[int, int, int, int]:
    pts = list(cells)
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _same_color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if visited[r, c] or color == bg:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(pts, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(pts),
                }
            )
    return components


def _is_filled_rectangle(component: dict) -> bool:
    r0, c0, r1, c1 = component["bbox"]
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    return height >= 2 and width >= 2 and len(component["coords"]) == height * width


def _ordered_path(path_mask: np.ndarray) -> list[tuple[int, int]]:
    coords = [tuple(x) for x in np.argwhere(path_mask)]
    coord_set = set(coords)
    assert coords, "path must contain at least one cell"

    neighbors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for cell in coords:
        r, c = cell
        nbrs = []
        for dr, dc in DIR8:
            q = (r + dr, c + dc)
            if q in coord_set:
                nbrs.append(q)
        neighbors[cell] = nbrs

    endpoints = [cell for cell, nbrs in neighbors.items() if len(nbrs) == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    for cell, nbrs in neighbors.items():
        assert len(nbrs) in (1, 2), "path must be a simple chain under 8-connectivity"

    start = min(endpoints)
    order = []
    prev = None
    cur = start
    while True:
        order.append(cur)
        next_cells = [q for q in neighbors[cur] if q != prev]
        assert len(next_cells) <= 1, "path must not branch"
        if not next_cells:
            break
        prev, cur = cur, next_cells[0]

    assert len(order) == len(coords), "path traversal must visit every path cell"
    return order


def _segments_from_order(order: list[tuple[int, int]], grid: np.ndarray) -> list[dict]:
    segments = []
    current_color = int(grid[order[0]])
    current_cells = [order[0]]

    for cell in order[1:]:
        color = int(grid[cell])
        if color != current_color:
            segments.append({"color": current_color, "cells": current_cells})
            current_color = color
            current_cells = [cell]
        else:
            current_cells.append(cell)
    segments.append({"color": current_color, "cells": current_cells})

    assert all(len(segment["cells"]) >= 3 for segment in segments)
    return segments


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    components = _same_color_components(g, bg)
    rectangles = [component for component in components if _is_filled_rectangle(component)]
    assert rectangles, "input must contain at least one rectangle"

    rect_mask = np.zeros_like(g, dtype=bool)
    for rect in rectangles:
        coords = rect["coords"]
        rect_mask[coords[:, 0], coords[:, 1]] = True

    path_mask = (g != bg) & (~rect_mask)
    order = _ordered_path(path_mask)
    segments = _segments_from_order(order, g)
    coord_to_segment = {cell: idx for idx, segment in enumerate(segments) for cell in segment["cells"]}

    segment_to_rect: dict[int, dict] = {}
    path_colors = {segment["color"] for segment in segments}
    for rect in rectangles:
        assert rect["color"] in path_colors
        touched_segments = set()
        for r, c in rect["coords"]:
            for dr, dc in DIR4:
                q = (int(r + dr), int(c + dc))
                if q in coord_to_segment:
                    touched_segments.add(coord_to_segment[q])
        assert len(touched_segments) == 1, "each rectangle must touch exactly one path segment"
        seg_idx = next(iter(touched_segments))
        assert seg_idx not in segment_to_rect, "a segment must not control multiple rectangles"
        segment_to_rect[seg_idx] = rect

    out = g.copy()
    for seg_idx, rect in segment_to_rect.items():
        seg_color = segments[seg_idx]["color"]
        rect_color = rect["color"]
        rect_coords = rect["coords"]
        out[rect_coords[:, 0], rect_coords[:, 1]] = seg_color
        for r, c in segments[seg_idx]["cells"]:
            out[r, c] = rect_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap colors between each touched path segment and its adjacent filled rectangle.

    Input structure:
    - The background is black.
    - One non-branching one-pixel path is 8-connected from end to end and is split into consecutive color segments.
    - Several separate monochrome filled rectangles sit beside the path; each rectangle's color matches some path segment color.

    Rule:
    - Remove the rectangles conceptually, recover the unique path, and traverse it from the endpoint with smaller (row, col).
    - Split the ordered path into maximal consecutive same-color segments.
    - For every rectangle, find the unique path segment touching it in 4-connectivity.
    - Recolor that whole rectangle to the segment's color and recolor the whole touched segment to the rectangle's original color.
    - Leave untouched path segments unchanged, and evaluate all swaps from the original input without chaining.

    Key insight:
    - The pairing is determined by spatial adjacency, not by matching equal colors, so the solver must separate rectangles from the snake path first and then swap colors bidirectionally.
    """
    return _transform(np.asarray(grid, dtype=int))


def _iter8(cell: tuple[int, int]):
    r, c = cell
    for dr, dc in DIR8:
        yield r + dr, c + dc


def _extend_segment(
    end: tuple[int, int],
    occupied: set[tuple[int, int]],
    direction: tuple[int, int],
    length: int,
):
    cur = end
    local = set(occupied)
    new_cells = []
    for _ in range(length):
        nxt = _add(cur, direction)
        if nxt in local:
            return None
        for neighbor in _iter8(nxt):
            if neighbor in local and neighbor != cur:
                return None
        new_cells.append(nxt)
        local.add(nxt)
        cur = nxt
    return new_cells


def _build_path(step_lengths: list[int]):
    def dfs(
        index: int,
        coords: list[tuple[int, int]],
        occupied: set[tuple[int, int]],
        directions: list[tuple[int, int]],
    ):
        if index == len(step_lengths):
            return coords, directions

        prev = directions[-1] if directions else None
        candidates = [d for d in DIR4 if prev is None or d != (-prev[0], -prev[1])]
        if prev is not None and random.random() < 0.8:
            perpendicular = [d for d in candidates if d != prev]
            straight = [d for d in candidates if d == prev]
            random.shuffle(perpendicular)
            random.shuffle(straight)
            candidates = perpendicular + straight
        else:
            random.shuffle(candidates)

        for direction in candidates:
            extension = _extend_segment(coords[-1], occupied, direction, step_lengths[index])
            if extension is None:
                continue
            new_coords = coords + extension
            r0, c0, r1, c1 = _bbox(new_coords)
            if r1 - r0 + 1 > PATH_SPAN_LIMIT or c1 - c0 + 1 > PATH_SPAN_LIMIT:
                continue
            result = dfs(index + 1, new_coords, occupied | set(extension), directions + [direction])
            if result is not None:
                return result
        return None

    return dfs(0, [(0, 0)], {(0, 0)}, [])


def _segments_from_directions(
    step_lengths: list[int], directions: list[tuple[int, int]]
) -> tuple[list[tuple[int, int]], list[list[tuple[int, int]]]]:
    coords = [(0, 0)]
    segments = []
    cur = (0, 0)

    for idx, (length, direction) in enumerate(zip(step_lengths, directions)):
        segment_cells = [cur] if idx == 0 else []
        for _ in range(length):
            cur = _add(cur, direction)
            coords.append(cur)
            segment_cells.append(cur)
        segments.append(segment_cells)
    return coords, segments


def _rectangle_candidates(
    path_set: set[tuple[int, int]], segment: list[tuple[int, int]]
) -> list[frozenset[tuple[int, int]]]:
    segment_set = set(segment)
    out: list[frozenset[tuple[int, int]]] = []
    rows = {r for r, _ in segment}
    cols = {c for _, c in segment}
    seg_len = len(segment)

    if len(rows) == 1:
        row = next(iter(rows))
        ordered_cols = sorted(c for _, c in segment)
        for rect_w in range(2, min(RECT_LONG_MAX, seg_len - 2) + 1):
            for start in range(1, seg_len - rect_w):
                cols_run = tuple(ordered_cols[start : start + rect_w])
                for rect_h in range(2, RECT_SHORT_MAX + 1):
                    for side in (-1, 1):
                        rows_run = (
                            range(row - rect_h, row)
                            if side == -1
                            else range(row + 1, row + 1 + rect_h)
                        )
                        cells = frozenset((rr, cc) for rr in rows_run for cc in cols_run)
                        if cells & path_set:
                            continue
                        touched4 = set()
                        touched8 = set()
                        for rr, cc in cells:
                            for dr, dc in DIR4:
                                q = (rr + dr, cc + dc)
                                if q in path_set:
                                    touched4.add(q)
                            for dr, dc in DIR8:
                                q = (rr + dr, cc + dc)
                                if q in path_set:
                                    touched8.add(q)
                        target = {(row, cc) for cc in cols_run}
                        if touched4 == target and touched8 <= segment_set:
                            out.append(cells)
    elif len(cols) == 1:
        col = next(iter(cols))
        ordered_rows = sorted(r for r, _ in segment)
        for rect_h in range(2, min(RECT_LONG_MAX, seg_len - 2) + 1):
            for start in range(1, seg_len - rect_h):
                rows_run = tuple(ordered_rows[start : start + rect_h])
                for rect_w in range(2, RECT_SHORT_MAX + 1):
                    for side in (-1, 1):
                        cols_run = (
                            range(col - rect_w, col)
                            if side == -1
                            else range(col + 1, col + 1 + rect_w)
                        )
                        cells = frozenset((rr, cc) for rr in rows_run for cc in cols_run)
                        if cells & path_set:
                            continue
                        touched4 = set()
                        touched8 = set()
                        for rr, cc in cells:
                            for dr, dc in DIR4:
                                q = (rr + dr, cc + dc)
                                if q in path_set:
                                    touched4.add(q)
                            for dr, dc in DIR8:
                                q = (rr + dr, cc + dc)
                                if q in path_set:
                                    touched8.add(q)
                        target = {(rr, col) for rr in rows_run}
                        if touched4 == target and touched8 <= segment_set:
                            out.append(cells)

    return list(dict.fromkeys(out))


def _margin8(cells: frozenset[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in cells:
        for dr, dc in DIR8:
            out.add((r + dr, c + dc))
    return out


def _place_rectangles(
    base_candidates: list[list[frozenset[tuple[int, int]]]],
    chosen_segments: list[int],
):
    order = sorted(chosen_segments, key=lambda idx: len(base_candidates[idx]))

    def dfs(index: int, reserved: set[tuple[int, int]], placed: list[tuple[int, frozenset[tuple[int, int]]]]):
        if index == len(order):
            return placed

        seg_idx = order[index]
        candidates = list(base_candidates[seg_idx])
        random.shuffle(candidates)
        for cells in candidates:
            if cells & reserved:
                continue
            result = dfs(index + 1, reserved | _margin8(cells), placed + [(seg_idx, cells)])
            if result is not None:
                return result
        return None

    return dfs(0, set(), [])


def _split_padding(content_size: int) -> tuple[int, int]:
    max_total = min(6, 30 - content_size)
    if max_total <= 0:
        return 0, 0
    min_total = 2 if max_total >= 2 else 0
    total = random.randint(min_total, max_total)
    first = random.randint(0, total)
    return first, total - first


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int)
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k)
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=int, copy=True)


def _gen_attempt() -> np.ndarray | None:
    segment_count = random.randint(*SEGMENT_COUNT_RANGE)
    step_lengths = [random.randint(*SEGMENT_STEP_RANGE) for _ in range(segment_count)]
    built = _build_path(step_lengths)
    if built is None:
        return None

    _, directions = built
    path_coords, segments = _segments_from_directions(step_lengths, directions)
    path_set = set(path_coords)

    base_candidates = [_rectangle_candidates(path_set, segment) for segment in segments]
    feasible_segments = [idx for idx, candidates in enumerate(base_candidates) if candidates]
    if len(feasible_segments) < 2:
        return None

    rect_count = random.randint(2, min(len(feasible_segments), segment_count - 1))
    chosen_segments = random.sample(feasible_segments, rect_count)
    placements = _place_rectangles(base_candidates, chosen_segments)
    if placements is None:
        return None

    segment_colors = random.sample(COLOR_POOL, segment_count)
    shifted = chosen_segments[1:] + chosen_segments[:1]
    rect_color_for_segment = {
        src: segment_colors[dst] for src, dst in zip(chosen_segments, shifted)
    }

    all_cells = set(path_coords)
    for _, cells in placements:
        all_cells |= set(cells)
    r0, c0, r1, c1 = _bbox(all_cells)
    content_h = r1 - r0 + 1
    content_w = c1 - c0 + 1
    if content_h > 30 or content_w > 30:
        return None

    top, bottom = _split_padding(content_h)
    left, right = _split_padding(content_w)
    h = content_h + top + bottom
    w = content_w + left + right
    grid = np.zeros((h, w), dtype=int)
    shift_r = top - r0
    shift_c = left - c0

    for seg_idx, cells in enumerate(segments):
        color = segment_colors[seg_idx]
        for r, c in cells:
            grid[r + shift_r, c + shift_c] = color

    for seg_idx, cells in placements:
        color = rect_color_for_segment[seg_idx]
        for r, c in cells:
            grid[r + shift_r, c + shift_c] = color

    grid = _random_transform(grid)
    out = _transform(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 4")
