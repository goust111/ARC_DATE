from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


Coord = tuple[int, int]
ORTHO_DIRS: tuple[Coord, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int) -> tuple[Coord, ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    bg = int(values[int(order[0])])
    if len(order) > 1:
        assert int(counts[int(order[0])]) > int(counts[int(order[1])]), "background must be uniquely dominant"
    return bg


def _monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            queue: deque[Coord] = deque([(r, c)])
            seen[r, c] = True
            cells: list[Coord] = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))

            components.append({"color": color, "cells": cells})

    return components


def _path_endpoints(cells: list[Coord]) -> list[Coord] | None:
    if len(cells) < 2:
        return None

    cell_set = set(cells)
    endpoints: list[Coord] = []
    for r, c in cells:
        degree = sum((nr, nc) in cell_set for nr, nc in _neighbors4(r, c))
        if degree == 1:
            endpoints.append((r, c))
        elif degree != 2:
            return None

    if len(endpoints) != 2:
        return None
    endpoints.sort()
    return endpoints


def _horizontal_then_vertical_path(start: Coord, end: Coord) -> list[Coord]:
    sr, sc = start
    er, ec = end

    path: list[Coord] = []
    if sc == ec:
        path.append((sr, sc))
    else:
        step_c = 1 if ec > sc else -1
        for col in range(sc, ec + step_c, step_c):
            path.append((sr, col))

    if er != sr:
        step_r = 1 if er > sr else -1
        for row in range(sr + step_r, er + step_r, step_r):
            path.append((row, ec))

    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Straighten each colored path into a horizontal-then-vertical polyline and recolor it from the endpoint labels.

    Input structure:
    - A dominant background contains several 4-connected single-color path components of width 1.
    - Each path has no branches, has exactly two endpoints, and each endpoint is 4-adjacent to a single colored label pixel that is not part of any path.
    - Label pixels may reuse colors across different paths, while path colors are only structural cues for locating the original paths.

    Rule:
    - Identify every path component, its two endpoints, and the label color adjacent to each endpoint.
    - Erase the original path pixels to background, keeping the label pixels unchanged.
    - For each path, order the endpoints by row and then column, then draw the shortest axis-aligned polyline from the first endpoint to the second, moving horizontally first and vertically second.
    - Color the horizontal-first segment with the first endpoint's label color and the vertical-second segment with the second endpoint's label color.
    - Color the turning cell with whichever of those two label colors appears more often in the whole input grid.

    Key insight:
    - The task separates object recognition from redraw logic: first recover thin path objects and their endpoint labels, then replace each winding path by a canonical L-shaped route whose colors are determined entirely by the visible labels and their global frequencies.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    h, w = g.shape

    bg = _background_color(g)
    components = _monochrome_components(g, bg)

    path_infos: list[dict[str, object]] = []
    label_cells: set[Coord] = set()
    path_cells: set[Coord] = set()

    for comp in components:
        cells = comp["cells"]
        endpoints = _path_endpoints(cells)
        if endpoints is None:
            assert len(cells) == 1, "non-path foreground objects must be single label pixels"
            label_cells.add(cells[0])
            continue
        path_infos.append({"cells": cells, "endpoints": endpoints})
        path_cells.update(cells)

    assert path_infos, "expected at least one path"
    assert label_cells, "expected endpoint labels"

    out = g.copy()
    for r, c in path_cells:
        out[r, c] = bg

    color_counts = Counter(int(v) for v in g.ravel())
    used_labels: set[Coord] = set()

    for info in path_infos:
        endpoints = list(info["endpoints"])
        labels: list[Coord] = []
        label_colors: list[int] = []

        for endpoint in endpoints:
            candidates = [
                (nr, nc)
                for nr, nc in _neighbors4(*endpoint)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in label_cells
            ]
            assert len(candidates) == 1, "each path endpoint must have exactly one label"
            label = candidates[0]
            assert label not in used_labels, "each label should belong to exactly one endpoint"
            used_labels.add(label)
            labels.append(label)
            label_colors.append(int(g[label]))

        start, end = endpoints
        start_color, end_color = label_colors
        assert start_color != end_color, "endpoint labels for one path must have different colors"
        assert color_counts[start_color] != color_counts[end_color], "corner color must be determined by a unique frequency winner"

        corner = (start[0], end[1])
        dominant_color = start_color if color_counts[start_color] > color_counts[end_color] else end_color

        sr, sc = start
        er, ec = end

        if sc != ec:
            step_c = 1 if ec > sc else -1
            for col in range(sc, ec + step_c, step_c):
                out[sr, col] = start_color
        else:
            out[sr, sc] = start_color

        if er != sr:
            step_r = 1 if er > sr else -1
            for row in range(sr, er + step_r, step_r):
                out[row, ec] = end_color

        out[corner] = dominant_color

    return out


# --------------- generator ---------------

GEN_RETRIES = 400
ZONE_RETRIES = 80
PAIR_RETRIES = 120

OUTER_MARGIN = 1
STACK_GAP_RANGE = (1, 2)
FULL_SPAN_RANGE = (14, 28)
TWO_ZONE_SPAN_RANGE = (7, 10)
THREE_ZONE_SPAN_RANGE = (6, 8)

MIN_PATH_LENGTH = 6
TURN_MODE_WEIGHT = 6
STRAIGHT_MODE_WEIGHT = 1
MIN_STRAIGHT_GAP = 4
MAX_DETOURS = 6
DETOUR_STALL_LIMIT = 30


def _in_zone(cell: Coord, zone: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, r1, c0, c1 = zone
    return r0 <= r <= r1 and c0 <= c <= c1


def _make_zones(rng: random.Random) -> tuple[int, int, list[tuple[int, int, int, int]]]:
    for _ in range(ZONE_RETRIES):
        path_count = rng.choice([2, 3])
        orientation = rng.choice(["rows", "cols"])
        span_lo, span_hi = TWO_ZONE_SPAN_RANGE if path_count == 2 else THREE_ZONE_SPAN_RANGE
        spans = [rng.randint(span_lo, span_hi) for _ in range(path_count)]
        gaps = [rng.randint(*STACK_GAP_RANGE) for _ in range(path_count - 1)]
        stacked_span = 2 * OUTER_MARGIN + sum(spans) + sum(gaps)
        other_span = rng.randint(*FULL_SPAN_RANGE)
        if stacked_span > 30 or other_span > 30:
            continue

        zones: list[tuple[int, int, int, int]] = []
        if orientation == "rows":
            h, w = stacked_span, other_span
            if w - 2 * OUTER_MARGIN < 12:
                continue
            row = OUTER_MARGIN
            for i, span in enumerate(spans):
                zones.append((row, row + span - 1, OUTER_MARGIN, w - OUTER_MARGIN - 1))
                row += span
                if i < len(gaps):
                    row += gaps[i]
        else:
            h, w = other_span, stacked_span
            if h - 2 * OUTER_MARGIN < 12:
                continue
            col = OUTER_MARGIN
            for i, span in enumerate(spans):
                zones.append((OUTER_MARGIN, h - OUTER_MARGIN - 1, col, col + span - 1))
                col += span
                if i < len(gaps):
                    col += gaps[i]

        return h, w, zones

    raise RuntimeError("failed to make zones")


def _label_pair_plan(path_count: int, label_colors: list[int], rng: random.Random) -> list[tuple[int, int]]:
    hub, side_a, side_b = rng.sample(label_colors, 3)
    if path_count == 2:
        pairs = [(hub, side_a), (hub, side_b)]
    else:
        if rng.random() < 0.5:
            pairs = [(hub, side_a), (hub, side_a), (hub, side_b)]
        else:
            pairs = [(hub, side_a), (hub, side_b), (hub, side_b)]
    rng.shuffle(pairs)
    return pairs


def _choose_endpoint_pair(zone: tuple[int, int, int, int], rng: random.Random) -> tuple[Coord, Coord] | None:
    r0, r1, c0, c1 = zone
    ir0, ir1 = r0 + 1, r1 - 1
    ic0, ic1 = c0 + 1, c1 - 1
    if ir0 > ir1 or ic0 > ic1:
        return None

    modes = ["turn", "same_row", "same_col"]
    weights = [TURN_MODE_WEIGHT, STRAIGHT_MODE_WEIGHT, STRAIGHT_MODE_WEIGHT]

    for _ in range(PAIR_RETRIES):
        mode = rng.choices(modes, weights=weights, k=1)[0]

        if mode == "turn":
            if ir1 - ir0 < 1 or ic1 - ic0 < 2:
                continue
            sr = rng.randint(ir0, ir1 - 1)
            er = rng.randint(sr + 1, ir1)
            sc = rng.randint(ic0, ic1)
            ec = rng.randint(ic0, ic1)
            if sc == ec:
                continue
        elif mode == "same_row":
            if ic1 - ic0 < MIN_STRAIGHT_GAP:
                continue
            row = rng.randint(ir0, ir1)
            sr = er = row
            sc = rng.randint(ic0, ic1 - MIN_STRAIGHT_GAP)
            ec = rng.randint(sc + MIN_STRAIGHT_GAP, ic1)
        else:
            if ir1 - ir0 < MIN_STRAIGHT_GAP:
                continue
            col = rng.randint(ic0, ic1)
            sc = ec = col
            sr = rng.randint(ir0, ir1 - MIN_STRAIGHT_GAP)
            er = rng.randint(sr + MIN_STRAIGHT_GAP, ir1)

        start = (sr, sc)
        end = (er, ec)
        if abs(er - sr) + abs(ec - sc) < MIN_PATH_LENGTH:
            continue
        return start, end

    return None


def _label_candidates(endpoint: Coord, base_set: set[Coord], zone: tuple[int, int, int, int]) -> list[Coord]:
    candidates: list[Coord] = []
    for cell in _neighbors4(*endpoint):
        if not _in_zone(cell, zone) or cell in base_set:
            continue
        touching = {nbr for nbr in _neighbors4(*cell) if nbr in base_set}
        if touching == {endpoint}:
            candidates.append(cell)
    return candidates


def _horizontal_snake(zone: tuple[int, int, int, int], rng: random.Random) -> list[Coord] | None:
    r0, r1, c0, c1 = zone
    max_rows = (r1 - r0 - 1) // 2
    if max_rows < 2 or c1 - c0 < MIN_STRAIGHT_GAP + 2:
        return None

    row_count = rng.randint(2, max_rows)
    top_max = r1 - 1 - 2 * (row_count - 1)
    if top_max < r0 + 1:
        return None
    top = rng.randint(r0 + 1, top_max)
    rows = [top + 2 * i for i in range(row_count)]

    left = rng.randint(c0 + 1, c1 - 1 - MIN_STRAIGHT_GAP)
    right = rng.randint(left + MIN_STRAIGHT_GAP, c1 - 1)

    path: list[Coord] = []
    for index, row in enumerate(rows):
        if index % 2 == 0:
            start_col = left if index == 0 else left + 1
            for col in range(start_col, right + 1):
                path.append((row, col))
            if index + 1 < len(rows):
                path.append((row + 1, right))
                path.append((row + 2, right))
        else:
            start_col = right - 1
            for col in range(start_col, left - 1, -1):
                path.append((row, col))
            if index + 1 < len(rows):
                path.append((row + 1, left))
                path.append((row + 2, left))
    return path


def _vertical_snake(zone: tuple[int, int, int, int], rng: random.Random) -> list[Coord] | None:
    r0, r1, c0, c1 = zone
    max_cols = (c1 - c0 - 1) // 2
    if max_cols < 2 or r1 - r0 < MIN_STRAIGHT_GAP + 2:
        return None

    col_count = rng.randint(2, max_cols)
    left_max = c1 - 1 - 2 * (col_count - 1)
    if left_max < c0 + 1:
        return None
    left = rng.randint(c0 + 1, left_max)
    cols = [left + 2 * i for i in range(col_count)]

    top = rng.randint(r0 + 1, r1 - 1 - MIN_STRAIGHT_GAP)
    bottom = rng.randint(top + MIN_STRAIGHT_GAP, r1 - 1)

    path: list[Coord] = []
    for index, col in enumerate(cols):
        if index % 2 == 0:
            start_row = top if index == 0 else top + 1
            for row in range(start_row, bottom + 1):
                path.append((row, col))
            if index + 1 < len(cols):
                path.append((bottom, col + 1))
                path.append((bottom, col + 2))
        else:
            start_row = bottom - 1
            for row in range(start_row, top - 1, -1):
                path.append((row, col))
            if index + 1 < len(cols):
                path.append((top, col + 1))
                path.append((top, col + 2))
    return path


def _build_snake_path(zone: tuple[int, int, int, int], rng: random.Random) -> list[Coord] | None:
    builders = [_horizontal_snake, _vertical_snake]
    rng.shuffle(builders)
    for builder in builders:
        path = builder(zone, rng)
        if path is None:
            continue
        endpoints = _path_endpoints(path)
        if endpoints == [path[0], path[-1]]:
            return path
    return None


def _can_insert_detour(
    path_set: set[Coord],
    a: Coord,
    b: Coord,
    p1: Coord,
    p2: Coord,
    zone: tuple[int, int, int, int],
    blocked: set[Coord],
    label_forbidden: set[Coord],
) -> bool:
    if p1 == p2:
        return False
    for cell in (p1, p2):
        if not _in_zone(cell, zone):
            return False
        if cell in blocked or cell in path_set or cell in label_forbidden:
            return False

    for cell, allowed in ((p1, {a, p2}), (p2, {p1, b})):
        for nbr in _neighbors4(*cell):
            if nbr in path_set and nbr not in allowed:
                return False

    return True


def _insert_detour_once(
    path: list[Coord],
    zone: tuple[int, int, int, int],
    blocked: set[Coord],
    label_forbidden: set[Coord],
    rng: random.Random,
) -> list[Coord] | None:
    edge_indices = list(range(len(path) - 1))
    rng.shuffle(edge_indices)
    path_set = set(path)

    for index in edge_indices:
        a = path[index]
        b = path[index + 1]
        options: list[tuple[Coord, Coord]] = []

        if a[0] == b[0]:
            shifts = [-1, 1]
            rng.shuffle(shifts)
            for dr in shifts:
                p1 = (a[0] + dr, a[1])
                p2 = (b[0] + dr, b[1])
                if _can_insert_detour(path_set, a, b, p1, p2, zone, blocked, label_forbidden):
                    options.append((p1, p2))
        else:
            shifts = [-1, 1]
            rng.shuffle(shifts)
            for dc in shifts:
                p1 = (a[0], a[1] + dc)
                p2 = (b[0], b[1] + dc)
                if _can_insert_detour(path_set, a, b, p1, p2, zone, blocked, label_forbidden):
                    options.append((p1, p2))

        if options:
            p1, p2 = rng.choice(options)
            return path[: index + 1] + [p1, p2] + path[index + 1 :]

    return None


def _twist_path(
    base_path: list[Coord],
    zone: tuple[int, int, int, int],
    labels: tuple[Coord, Coord],
    blocked: set[Coord],
    rng: random.Random,
) -> list[Coord] | None:
    path = list(base_path)
    label_forbidden: set[Coord] = set()
    for label in labels:
        for nbr in _neighbors4(*label):
            if _in_zone(nbr, zone):
                label_forbidden.add(nbr)
    label_forbidden -= {base_path[0], base_path[-1]}

    target_detours = rng.randint(1, MAX_DETOURS)
    inserted = 0
    stalls = 0

    while inserted < target_detours and stalls < DETOUR_STALL_LIMIT:
        updated = _insert_detour_once(path, zone, blocked, label_forbidden, rng)
        if updated is None:
            stalls += 1
            if inserted > 0:
                break
        else:
            path = updated
            inserted += 1
            stalls = 0

    if inserted == 0:
        return None
    return path


def _place_one_path(
    grid: np.ndarray,
    zone: tuple[int, int, int, int],
    path_color: int,
    label_pair: tuple[int, int],
    rng: random.Random,
) -> bool:
    for _ in range(PAIR_RETRIES):
        path = _build_snake_path(zone, rng)
        if path is None:
            return False
        start, end = path[0], path[-1]
        path_set = set(path)

        start_labels = _label_candidates(start, path_set, zone)
        end_labels = _label_candidates(end, path_set, zone)
        if not start_labels or not end_labels:
            continue

        rng.shuffle(start_labels)
        rng.shuffle(end_labels)

        for start_label in start_labels:
            for end_label in end_labels:
                if start_label == end_label or abs(start_label[0] - end_label[0]) + abs(start_label[1] - end_label[1]) <= 1:
                    continue
                if set(nbr for nbr in _neighbors4(*start_label) if nbr in path_set) != {start}:
                    continue
                if set(nbr for nbr in _neighbors4(*end_label) if nbr in path_set) != {end}:
                    continue

                start_color, end_color = label_pair
                if rng.random() < 0.5:
                    start_color, end_color = end_color, start_color

                for r, c in path:
                    assert int(grid[r, c]) == int(grid[0, 0]), "zones must stay disjoint"
                    grid[r, c] = path_color
                assert int(grid[start_label]) == int(grid[0, 0]) and int(grid[end_label]) == int(grid[0, 0]), "labels must be placed on background"
                grid[start_label] = start_color
                grid[end_label] = end_color
                return True

    return False


def _build_scene(rng: random.Random) -> np.ndarray | None:
    h, w, zones = _make_zones(rng)
    palette = list(range(10))
    bg = rng.choice(palette)
    remaining = [color for color in palette if color != bg]

    label_palette = rng.sample(remaining, 3)
    path_palette = [color for color in remaining if color not in label_palette]
    if len(path_palette) < len(zones):
        return None
    path_colors = rng.sample(path_palette, len(zones))
    label_pairs = _label_pair_plan(len(zones), label_palette, rng)

    grid = np.full((h, w), bg, dtype=int)
    for zone, path_color, label_pair in zip(zones, path_colors, label_pairs):
        if not _place_one_path(grid, zone, path_color, label_pair, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _build_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
