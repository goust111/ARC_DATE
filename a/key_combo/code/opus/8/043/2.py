from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

MAX_ATTEMPTS = 2000


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *separators, size]
    bands = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    assert all(start < stop for start, stop in bands)
    return bands


def _separator_lines(grid: np.ndarray) -> tuple[list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    rows = [
        r
        for r in range(g.shape[0])
        if set(map(int, g[r, :].tolist())) <= {BLUE, GREY} and np.any(g[r, :] == GREY)
    ]
    cols = [
        c
        for c in range(g.shape[1])
        if set(map(int, g[:, c].tolist())) <= {BLUE, GREY} and np.any(g[:, c] == GREY)
    ]
    assert len(rows) in (2, 3), f"expected 2 or 3 separator rows, got {rows}"
    assert len(cols) == 2, f"expected 2 separator cols, got {cols}"
    return rows, cols


def _trace_blue_path(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    points = [(int(r), int(c)) for r, c in np.argwhere(g == BLUE)]
    assert points, "expected a blue path"
    point_set = set(points)
    degrees = {}
    for r, c in points:
        degrees[(r, c)] = sum((nr, nc) in point_set for nr, nc in _neighbors4(r, c))
    endpoints = [pt for pt in points if degrees[pt] == 1]
    assert len(endpoints) == 2, f"expected exactly 2 endpoints, got {endpoints}"
    start = min(endpoints)

    order: list[tuple[int, int]] = []
    prev = None
    cur = start
    for _ in range(len(points)):
        order.append(cur)
        r, c = cur
        nxts = [(nr, nc) for nr, nc in _neighbors4(r, c) if (nr, nc) in point_set and (nr, nc) != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path must be a simple chain"
        prev, cur = cur, nxts[0]
    assert len(order) == len(points), "path traversal did not cover all blue cells"
    return order


def _region_of(cell: tuple[int, int], sep_rows: list[int], sep_cols: list[int]) -> tuple[int, int] | None:
    r, c = cell
    if r in sep_rows or c in sep_cols:
        return None
    rr = sum(sr < r for sr in sep_rows)
    cc = sum(sc < c for sc in sep_cols)
    return rr, cc


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color separator-defined regions by the odd/even order in which a snake path visits them.

    Input structure:
    - The background is black, while grey separator rows and columns partition the grid
      into a 3x3 or 4x3 table of rectangular regions.
    - A single blue 4-connected snake path has exactly two endpoints and may pass through
      one-cell gaps carved into the grey separator lines.
    - Inside any one region, the snake occupies at most one continuous segment.

    Rule:
    - Detect the grey separator rows and columns even when blue gap cells interrupt them.
    - Trace the blue snake from the top-left-most endpoint, and record the first time the
      traversal enters each region; those first entries define visit numbers 1, 2, 3, ....
    - Recolor every black background cell in odd-numbered visited regions red and in
      even-numbered visited regions green, while preserving the blue snake, grey separators,
      and untouched black regions.

    Key insight:
    - The fill color comes from the snake's topological visitation order through regions,
      not from the regions' spatial coordinates, so separator corridors between visits must
      be traced exactly.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    sep_rows, sep_cols = _separator_lines(g)
    path = _trace_blue_path(g)
    visit_order: dict[tuple[int, int], int] = {}
    for cell in path:
        region = _region_of(cell, sep_rows, sep_cols)
        if region is not None and region not in visit_order:
            visit_order[region] = len(visit_order) + 1

    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)
    for (rr, cc), index in visit_order.items():
        rs, re = row_bands[rr]
        cs, ce = col_bands[cc]
        fill_color = RED if index % 2 == 1 else GREEN
        region = out[rs:re, cs:ce]
        region[region == BLACK] = fill_color
    return out


def _choose_sizes(region_rows: int) -> tuple[list[int], list[int]]:
    if region_rows == 3:
        heights = [random.randint(4, 6) for _ in range(3)]
    else:
        heights = [random.randint(4, 5) for _ in range(4)]
    widths = [random.randint(4, 6) for _ in range(3)]
    return heights, widths


def _layout_boxes(heights: list[int], widths: list[int]) -> tuple[dict[tuple[int, int], tuple[int, int, int, int]], list[int], list[int]]:
    boxes: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    sep_rows: list[int] = []
    sep_cols: list[int] = []

    row_starts = []
    cur = 0
    for i, h in enumerate(heights):
        row_starts.append(cur)
        cur += h
        if i != len(heights) - 1:
            sep_rows.append(cur)
            cur += 1
    col_starts = []
    cur = 0
    for j, w in enumerate(widths):
        col_starts.append(cur)
        cur += w
        if j != len(widths) - 1:
            sep_cols.append(cur)
            cur += 1

    for rr, r0 in enumerate(row_starts):
        for cc, c0 in enumerate(col_starts):
            h = heights[rr]
            w = widths[cc]
            boxes[(rr, cc)] = (r0, r0 + h - 1, c0, c0 + w - 1)
    return boxes, sep_rows, sep_cols


def _separator_mask(shape: tuple[int, int], sep_rows: list[int], sep_cols: list[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[sep_rows, :] = True
    mask[:, sep_cols] = True
    return mask


def _gate_candidates(
    region: tuple[int, int],
    boxes: dict[tuple[int, int], tuple[int, int, int, int]],
    region_rows: int,
) -> list[dict[str, object]]:
    rr, cc = region
    r0, r1, c0, c1 = boxes[region]
    gates: list[dict[str, object]] = []

    if rr > 0:
        sep_r = r0 - 1
        for c in range(c0 + 1, c1):
            gates.append({"side": "N", "gate": (sep_r, c), "anchor": (r0, c)})
    if rr < region_rows - 1:
        sep_r = r1 + 1
        for c in range(c0 + 1, c1):
            gates.append({"side": "S", "gate": (sep_r, c), "anchor": (r1, c)})
    if cc > 0:
        sep_c = c0 - 1
        for r in range(r0 + 1, r1):
            gates.append({"side": "W", "gate": (r, sep_c), "anchor": (r, c0)})
    if cc < 2:
        sep_c = c1 + 1
        for r in range(r0 + 1, r1):
            gates.append({"side": "E", "gate": (r, sep_c), "anchor": (r, c1)})
    return gates


def _random_shortest_sep_path(
    start: tuple[int, int],
    goal: tuple[int, int],
    sep_mask: np.ndarray,
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    h, w = sep_mask.shape
    blocked = set(blocked)
    blocked.discard(start)
    blocked.discard(goal)
    q: deque[tuple[int, int]] = deque([start])
    dist = {start: 0}
    parents: dict[tuple[int, int], list[tuple[int, int]]] = {start: []}

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        nbrs = list(_neighbors4(*cur))
        random.shuffle(nbrs)
        for nr, nc in nbrs:
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not sep_mask[nr, nc] or nxt in blocked:
                continue
            nd = dist[cur] + 1
            if nxt not in dist:
                dist[nxt] = nd
                parents[nxt] = [cur]
                q.append(nxt)
            elif dist[nxt] == nd:
                parents[nxt].append(cur)
    if goal not in dist:
        return None

    path = [goal]
    cur = goal
    while cur != start:
        options = parents[cur]
        cur = random.choice(options)
        path.append(cur)
    path.reverse()
    return path


def _straight_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a[0] == b[0]:
        step = 1 if b[1] >= a[1] else -1
        return [(a[0], c) for c in range(a[1], b[1] + step, step)]
    step = 1 if b[0] >= a[0] else -1
    return [(r, a[1]) for r in range(a[0], b[0] + step, step)]


def _l_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    if a == b:
        return [a]
    if a[0] == b[0] or a[1] == b[1]:
        return _straight_path(a, b)
    pivots = [(a[0], b[1]), (b[0], a[1])]
    pivot = random.choice(pivots)
    first = _straight_path(a, pivot)
    second = _straight_path(pivot, b)
    return first + second[1:]


def _pen(anchor: tuple[int, int], side: str) -> tuple[int, int]:
    r, c = anchor
    if side == "N":
        return r + 1, c
    if side == "S":
        return r - 1, c
    if side == "W":
        return r, c + 1
    return r, c - 1


def _interior_cells(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = box
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _region_segment(
    box: tuple[int, int, int, int],
    entry_gate: dict[str, object] | None,
    exit_gate: dict[str, object] | None,
    endpoint: tuple[int, int] | None,
) -> list[tuple[int, int]]:
    if entry_gate is None:
        assert exit_gate is not None and endpoint is not None
        exit_anchor = tuple(exit_gate["anchor"])
        path = _l_path(endpoint, _pen(exit_anchor, str(exit_gate["side"])))
        return path + [exit_anchor]
    if exit_gate is None:
        assert endpoint is not None
        entry_anchor = tuple(entry_gate["anchor"])
        return [entry_anchor] + _l_path(_pen(entry_anchor, str(entry_gate["side"])), endpoint)
    entry_anchor = tuple(entry_gate["anchor"])
    exit_anchor = tuple(exit_gate["anchor"])
    middle = _l_path(_pen(entry_anchor, str(entry_gate["side"])), _pen(exit_anchor, str(exit_gate["side"])))
    return [entry_anchor] + middle + [exit_anchor]


def _choose_endpoint(
    box: tuple[int, int, int, int],
    forbidden: set[tuple[int, int]],
) -> tuple[int, int] | None:
    cells = [cell for cell in _interior_cells(box) if cell not in forbidden]
    if not cells:
        return None
    return random.choice(cells)


def _connected_components(cells: set[tuple[int, int]]) -> int:
    if not cells:
        return 0
    seen = set()
    count = 0
    for start in list(cells):
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            cur = stack.pop()
            for nxt in _neighbors4(*cur):
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return count


def _validate_semantics(grid: np.ndarray, expected_sequence: list[tuple[int, int]]) -> None:
    g = np.asarray(grid, dtype=int)
    sep_rows, sep_cols = _separator_lines(g)
    path = _trace_blue_path(g)
    sequence: list[tuple[int, int]] = []
    seen = set()
    for cell in path:
        region = _region_of(cell, sep_rows, sep_cols)
        if region is not None and region not in seen:
            seen.add(region)
            sequence.append(region)
    assert sequence == expected_sequence
    assert len(sequence) >= 6
    assert any(abs(a[0] - b[0]) + abs(a[1] - b[1]) > 1 for a, b in zip(sequence, sequence[1:]))
    start_parity = (sequence[0][0] + sequence[0][1]) % 2
    assert any(
        ((idx % 2) == 1) != (((rid[0] + rid[1]) % 2) == start_parity)
        for idx, rid in enumerate(sequence, start=1)
    )

    blue = {(int(r), int(c)) for r, c in np.argwhere(g == BLUE)}
    for rid in sequence:
        r0, r1 = _bands(g.shape[0], sep_rows)[rid[0]]
        c0, c1 = _bands(g.shape[1], sep_cols)[rid[1]]
        region_blue = {
            (r, c)
            for r in range(r0, r1)
            for c in range(c0, c1)
            if (r, c) in blue
        }
        assert _connected_components(region_blue) == 1


def _gate_key(gate: dict[str, object]) -> tuple[int, int]:
    return tuple(gate["gate"])


def _build_attempt() -> np.ndarray | None:
    region_rows = random.choice([3, 4])
    heights, widths = _choose_sizes(region_rows)
    boxes, sep_rows, sep_cols = _layout_boxes(heights, widths)
    h = sum(heights) + len(sep_rows)
    w = sum(widths) + len(sep_cols)
    assert max(h, w) <= 30

    total_regions = region_rows * 3
    upper = min(total_regions - 1, 7)
    visit_count = 6 if upper == 6 else random.choice([6, 6, 7])
    regions = list(boxes)
    gates = {region: _gate_candidates(region, boxes, region_rows) for region in boxes}
    sep_mask = _separator_mask((h, w), sep_rows, sep_cols)

    start_region = random.choice(regions)
    sequence = [start_region]
    entry_gate: dict[tuple[int, int], dict[str, object]] = {}
    exit_gate: dict[tuple[int, int], dict[str, object]] = {}
    corridors: list[list[tuple[int, int]]] = []
    nonadjacent_pair = False

    start_exit = random.choice(gates[start_region])
    exit_gate[start_region] = start_exit
    used_sep: set[tuple[int, int]] = {_gate_key(start_exit)}
    current_region = start_region
    current_gate = start_exit

    while len(sequence) < visit_count:
        remaining_regions = [region for region in regions if region not in sequence]
        final = len(sequence) == visit_count - 1
        nonadj = [region for region in remaining_regions if abs(region[0] - current_region[0]) + abs(region[1] - current_region[1]) > 1]
        adj = [region for region in remaining_regions if region not in nonadj]
        random.shuffle(nonadj)
        random.shuffle(adj)
        ordered_regions = nonadj + adj
        found_move = None
        for region in ordered_regions:
            entry_choices = random.sample(gates[region], k=min(4, len(gates[region])))
            for entry in entry_choices:
                if _gate_key(entry) in used_sep:
                    continue
                path = _random_shortest_sep_path(
                    _gate_key(current_gate),
                    _gate_key(entry),
                    sep_mask,
                    used_sep,
                )
                if path is None:
                    continue
                if final:
                    found_move = (region, entry, None, path)
                    break
                exit_choices = [
                    gate
                    for gate in random.sample(gates[region], k=min(5, len(gates[region])))
                    if _gate_key(gate) != _gate_key(entry) and _gate_key(gate) not in used_sep
                ]
                if exit_choices:
                    found_move = (region, entry, random.choice(exit_choices), path)
                    break
            if found_move is not None:
                break
        if found_move is None:
            return None
        region, entry, out_gate, path = found_move
        nonadjacent_pair = nonadjacent_pair or (
            abs(region[0] - current_region[0]) + abs(region[1] - current_region[1]) > 1
        )
        entry_gate[region] = entry
        corridors.append(path)
        used_sep.update(path)
        sequence.append(region)
        if out_gate is not None:
            exit_gate[region] = out_gate
            used_sep.add(_gate_key(out_gate))
            current_gate = out_gate
            current_region = region

    if not nonadjacent_pair:
        return None

    endpoints: dict[tuple[int, int], tuple[int, int]] = {}
    end_region = sequence[-1]
    start_forbidden = {_pen(tuple(exit_gate[start_region]["anchor"]), str(exit_gate[start_region]["side"]))}
    end_forbidden = {_pen(tuple(entry_gate[end_region]["anchor"]), str(entry_gate[end_region]["side"]))}
    endpoints[start_region] = _choose_endpoint(boxes[start_region], start_forbidden)
    endpoints[end_region] = _choose_endpoint(boxes[end_region], end_forbidden)
    if endpoints[start_region] is None or endpoints[end_region] is None:
        return None

    ordered_path: list[tuple[int, int]] = []
    start_segment = _region_segment(boxes[start_region], None, exit_gate[start_region], endpoints[start_region])
    ordered_path.extend(start_segment)
    for i, corridor in enumerate(corridors):
        if ordered_path[-1] == corridor[0]:
            return None
        ordered_path.extend(corridor)
        region = sequence[i + 1]
        segment = _region_segment(
            boxes[region],
            entry_gate[region],
            exit_gate.get(region),
            endpoints.get(region),
        )
        if ordered_path[-1] == segment[0]:
            return None
        ordered_path.extend(segment)

    if len(set(ordered_path)) != len(ordered_path):
        return None
    if any(abs(r1 - r0) + abs(c1 - c0) != 1 for (r0, c0), (r1, c1) in zip(ordered_path, ordered_path[1:])):
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    grid[sep_rows, :] = GREY
    grid[:, sep_cols] = GREY
    for r, c in ordered_path:
        grid[r, c] = BLUE

    try:
        _validate_semantics(grid, sequence)
    except AssertionError:
        return None

    out = main(grid)
    if not (np.any(out == RED) and np.any(out == GREEN)):
        return None
    if visit_count < total_regions and not np.any(out == BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
