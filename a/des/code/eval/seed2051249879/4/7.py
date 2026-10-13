from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


Coord = tuple[int, int]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _neighbors4(r: int, c: int, shape: tuple[int, int]) -> list[Coord]:
    h, w = shape
    out: list[Coord] = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components_of_color(grid: np.ndarray, color: int) -> list[list[Coord]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, grid.shape):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[Coord]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_rings(grid: np.ndarray, bg: int) -> tuple[list[dict], set[Coord], set[Coord]]:
    rings: list[dict] = []
    ring_cells: set[Coord] = set()
    marker_cells: set[Coord] = set()

    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for comp in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(comp)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
                continue

            expected_area = 2 * h + 2 * w - 4
            if len(comp) != expected_area:
                continue

            mask = np.zeros((h, w), dtype=bool)
            for r, c in comp:
                mask[r - r0, c - c0] = True

            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if not np.array_equal(mask, border):
                continue

            cr = r0 + h // 2
            cc = c0 + w // 2

            marker_candidates: list[Coord] = []
            if r0 > 0:
                for c in range(c0, c1 + 1):
                    if c != cc:
                        marker_candidates.append((r0 - 1, c))
            if c0 > 0:
                for r in range(r0, r1 + 1):
                    if r != cr:
                        marker_candidates.append((r, c0 - 1))

            marker_hits = [
                pos
                for pos in marker_candidates
                if int(grid[pos]) != bg and int(grid[pos]) != color
            ]
            if len(marker_hits) != 1:
                continue

            marker = marker_hits[0]
            ring = {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "center": (cr, cc),
                "marker_pos": marker,
                "marker_color": int(grid[marker]),
            }
            rings.append(ring)
            ring_cells.update(comp)
            marker_cells.add(marker)

    assert rings, "expected at least one ring"
    assert len({tuple(ring["center"]) for ring in rings}) == len(rings), "ring centers must be unique"
    return rings, ring_cells, marker_cells


def _path_cells_by_color(
    grid: np.ndarray,
    bg: int,
    ring_cells: set[Coord],
    marker_cells: set[Coord],
) -> dict[int, set[Coord]]:
    cells: dict[int, set[Coord]] = defaultdict(set)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            pos = (r, c)
            if pos in ring_cells or pos in marker_cells:
                continue
            color = int(grid[r, c])
            if color != bg:
                cells[color].add(pos)
    return cells


def _add_jump_edges(
    graph: dict[Coord, set[Coord]],
    cells: set[Coord],
    rings: list[dict],
) -> dict[Coord, int]:
    ring_markers: dict[Coord, int] = {}

    for ring in rings:
        center = tuple(ring["center"])
        if center not in cells:
            continue

        r0, c0, r1, c1 = ring["bbox"]
        cr, cc = center
        horizontal = [(cr, c0 - 1), (cr, c1 + 1)]
        vertical = [(r0 - 1, cc), (r1 + 1, cc)]

        has_horizontal = all(pos in cells for pos in horizontal)
        has_vertical = all(pos in cells for pos in vertical)
        assert has_horizontal ^ has_vertical, "ring traversal must be horizontal or vertical"

        jump_targets = horizontal if has_horizontal else vertical
        for pos in jump_targets:
            graph[center].add(pos)
            graph[pos].add(center)
        ring_markers[center] = int(ring["marker_color"])

    return ring_markers


def _graph_components(graph: dict[Coord, set[Coord]]) -> list[set[Coord]]:
    seen: set[Coord] = set()
    comps: list[set[Coord]] = []

    for node in graph:
        if node in seen:
            continue
        q = deque([node])
        seen.add(node)
        comp: set[Coord] = set()
        while q:
            cur = q.popleft()
            comp.add(cur)
            for nxt in graph[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        comps.append(comp)
    return comps


def _endpoint_key(pos: Coord, shape: tuple[int, int]) -> tuple[int, int]:
    r, c = pos
    h, w = shape
    keys: list[tuple[int, int]] = []
    if r == 0:
        keys.append((0, c))
    if c == 0:
        keys.append((1, r))
    if r == h - 1:
        keys.append((2, c))
    if c == w - 1:
        keys.append((3, r))
    assert keys, "path endpoint must touch the boundary"
    return min(keys)


def _ordered_path(nodes: set[Coord], graph: dict[Coord, set[Coord]], shape: tuple[int, int]) -> list[Coord]:
    endpoints = [node for node in nodes if len(graph[node]) == 1]
    assert len(endpoints) == 2, "each rope must have exactly two endpoints"
    start = min(endpoints, key=lambda pos: _endpoint_key(pos, shape))

    order: list[Coord] = []
    prev: Coord | None = None
    cur = start
    while True:
        order.append(cur)
        next_nodes = [node for node in graph[cur] if node != prev]
        if not next_nodes:
            break
        assert len(next_nodes) == 1, "rope graph must be a simple path"
        prev, cur = cur, next_nodes[0]

    assert len(order) == len(nodes), "ordered walk must cover the whole rope"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Treat each colored polyline as a rope, detect which grey rings it passes through,
    and recolor the visible rope segments according to the rings' marker colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    rings, ring_cells, marker_cells = _find_rings(g, bg)
    ropes = _path_cells_by_color(g, bg, ring_cells, marker_cells)
    assert ropes, "expected at least one rope"

    out = g.copy()
    for color, cells in ropes.items():
        graph = {pos: set() for pos in cells}
        for r, c in cells:
            for nr, nc in ((r + 1, c), (r, c + 1)):
                if (nr, nc) in cells:
                    graph[(r, c)].add((nr, nc))
                    graph[(nr, nc)].add((r, c))

        ring_markers = _add_jump_edges(graph, cells, rings)

        for comp in _graph_components(graph):
            order = _ordered_path(comp, graph, g.shape)
            current = int(color)
            for pos in order:
                if pos in ring_markers:
                    current = ring_markers[pos]
                out[pos] = current

    return out


def _distribute(total: int, slots: int) -> list[int]:
    out = [0] * slots
    for _ in range(total):
        out[random.randrange(slots)] += 1
    return out


def _sample_lanes(path_count: int) -> tuple[int, list[tuple[int, int]]]:
    heights = [random.randint(6, 8) for _ in range(path_count)]
    base_gaps = [2] + [2] * (path_count - 1) + [2]
    extra = random.randint(0, 4)
    gaps = [g + d for g, d in zip(base_gaps, _distribute(extra, len(base_gaps)), strict=False)]

    lanes: list[tuple[int, int]] = []
    cursor = gaps[0]
    for i, height in enumerate(heights):
        top = cursor
        bottom = top + height - 1
        lanes.append((top, bottom))
        cursor = bottom + 1
        if i + 1 < path_count:
            cursor += gaps[i + 1]

    height = sum(heights) + sum(gaps)
    return height, lanes


def _segment_lengths(total_steps: int, segments: int, min_len: int) -> list[int] | None:
    extra = total_steps - segments * min_len
    if extra < 0:
        return None
    lengths = [min_len] * segments
    for add in _distribute(extra, segments):
        lengths[lengths.index(min(lengths))] += add
    random.shuffle(lengths)
    return lengths


def _build_horizontal_path(lane: tuple[int, int], width: int) -> list[Coord] | None:
    top, bottom = lane
    usable_rows = list(range(top + 2, bottom - 1))
    if len(usable_rows) < 2:
        return None

    max_jogs = min(2, max(0, (width - 1) // 6 - 1))
    bend_count = random.randint(0, max_jogs)
    lengths = _segment_lengths(width - 1, bend_count + 1, 6)
    if lengths is None:
        return None

    rows = [random.choice(usable_rows)]
    for _ in range(bend_count):
        choices = [row for row in usable_rows if row != rows[-1]]
        rows.append(random.choice(choices))

    cells: list[Coord] = [(rows[0], 0)]
    c = 0
    for i, length in enumerate(lengths):
        end_c = c + length
        for cc in range(c + 1, end_c + 1):
            cells.append((rows[i], cc))
        if i + 1 < len(rows):
            step = 1 if rows[i + 1] > rows[i] else -1
            for rr in range(rows[i] + step, rows[i + 1] + step, step):
                cells.append((rr, end_c))
        c = end_c

    if len(cells) != len(set(cells)):
        return None
    return cells


def _is_ring_site(cells: list[Coord], idx: int, path_set: set[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    r, c = cells[idx]
    if not (2 <= r < h - 2 and 4 <= c < w - 4):
        return False

    window = cells[idx - 2 : idx + 3]
    if any(rr != r for rr, _cc in window):
        return False
    cols = [cc for _rr, cc in window]
    if cols != list(range(c - 2, c + 3)):
        return False

    for rr in range(r - 2, r + 3):
        for cc in range(c - 2, c + 3):
            if (rr, cc) in path_set and not (rr == r and abs(cc - c) <= 2):
                return False
    return True


def _choose_ring_centers(cells: list[Coord], shape: tuple[int, int]) -> list[Coord] | None:
    path_set = set(cells)
    candidates = [cells[i] for i in range(2, len(cells) - 2) if _is_ring_site(cells, i, path_set, shape)]
    if not candidates:
        return None

    max_rings = min(3, max(1, len(candidates) // 5))
    wanted = random.randint(1, max_rings)
    for _ in range(40):
        chosen: list[Coord] = []
        for cand in random.sample(candidates, len(candidates)):
            if all(max(abs(cand[0] - rr), abs(cand[1] - cc)) > 4 for rr, cc in chosen):
                chosen.append(cand)
            if len(chosen) == wanted:
                return sorted(chosen, key=lambda pos: pos[1])
    return None


def _marker_positions(center: Coord, side: str) -> list[Coord]:
    r, c = center
    if side == "above":
        return [(r - 2, c - 1), (r - 2, c + 1)]
    return [(r - 1, c - 2), (r + 1, c - 2)]


def _render_base_example() -> tuple[np.ndarray, np.ndarray] | None:
    path_count = random.randint(1, 3)
    height, lanes = _sample_lanes(path_count)
    width = random.randint(20, 30)

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    ring_color = random.choice(palette)
    palette.remove(ring_color)
    random.shuffle(palette)
    path_colors = palette[:path_count]
    marker_palette = palette[path_count : path_count + 3]
    if len(marker_palette) < 2:
        return None

    paths: list[dict] = []
    all_path_cells: set[Coord] = set()
    for lane, color in zip(lanes, path_colors, strict=False):
        cells = _build_horizontal_path(lane, width)
        if cells is None:
            return None
        if any(cell in all_path_cells for cell in cells):
            return None
        centers = _choose_ring_centers(cells, (height, width))
        if not centers:
            return None
        all_path_cells.update(cells)
        paths.append({"color": color, "cells": cells, "centers": centers})

    grid = np.full((height, width), bg, dtype=int)
    for path in paths:
        for r, c in path["cells"]:
            grid[r, c] = int(path["color"])

    ring_records: list[dict] = []
    used_markers: set[Coord] = set()
    hidden_cells: set[Coord] = set()
    for path in paths:
        path_set = set(path["cells"])
        for center in path["centers"]:
            r, c = center
            sides = ["above", "left"]
            random.shuffle(sides)
            marker_pos: Coord | None = None
            marker_side = ""
            for side in sides:
                candidates = [
                    pos
                    for pos in _marker_positions(center, side)
                    if pos not in path_set and pos not in used_markers
                ]
                if candidates:
                    marker_side = side
                    marker_pos = random.choice(candidates)
                    break
            if marker_pos is None:
                return None

            marker_color = random.choice(marker_palette)
            used_markers.add(marker_pos)
            ring_records.append(
                {
                    "center": center,
                    "marker_pos": marker_pos,
                    "marker_color": marker_color,
                    "path_color": int(path["color"]),
                    "side": marker_side,
                }
            )
            hidden_cells.add((r, c - 1))
            hidden_cells.add((r, c + 1))

    for record in ring_records:
        r, c = record["center"]
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if (rr, cc) != (r, c):
                    grid[rr, cc] = ring_color
        mr, mc = record["marker_pos"]
        grid[mr, mc] = int(record["marker_color"])

    expected = grid.copy()
    marker_by_center = {tuple(record["center"]): int(record["marker_color"]) for record in ring_records}
    for path in paths:
        current = int(path["color"])
        for pos in path["cells"]:
            if pos in marker_by_center:
                current = marker_by_center[pos]
            if pos not in hidden_cells:
                expected[pos] = current

    return grid, expected


def _generate_example() -> tuple[np.ndarray, np.ndarray] | None:
    sample = _render_base_example()
    if sample is None:
        return None
    inp, out = sample
    if random.random() < 0.5:
        inp = inp.T.copy()
        out = out.T.copy()
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(400):
        sample = _generate_example()
        if sample is None:
            continue
        inp, expected = sample
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate a valid rope-and-rings puzzle")
