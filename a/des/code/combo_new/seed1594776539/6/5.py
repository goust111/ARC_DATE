from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8, 9)

# Generator tuning.
SEGMENT_COUNT_CHOICES = (3, 4)
MIN_SEGMENT_LEN = 3
BASE_HEIGHT_RANGE = (4, 6)
BASE_WIDTH_RANGE = (4, 7)
TARGET_BUMP_RANGE = (1, 4)
MAX_BUMP_TRIES = 80
MAX_CYCLE_RETRIES = 120
MAX_GENERATION_ATTEMPTS = 400
MAX_CYCLE_SIDE = 12
WORK_CANVAS = 30
PLACEMENT_PAD = 2
PLACEMENT_TRIES = 500


Coord = tuple[int, int]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            q: deque[Coord] = deque([(int(r0), int(c0))])
            seen[r0, c0] = True
            comp: list[Coord] = []
            while q:
                r, c = q.popleft()
                comp.append((r, c))
                for dr, dc in DIR4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[Coord] | tuple[Coord, ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_nonblack(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BLACK)
    assert coords.size > 0, "expected at least one foreground pixel"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _connected(coords: set[Coord]) -> bool:
    start = next(iter(coords))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            nxt = (r + dr, c + dc)
            if nxt in coords and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _extract_segments(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    comps = _components(g != BLACK)
    assert comps, "expected at least one path segment"

    segments: list[dict] = []
    for comp in comps:
        comp_set = set(comp)
        degrees: dict[Coord, int] = {}
        for r, c in comp:
            degree = 0
            for dr, dc in DIR4:
                if (r + dr, c + dc) in comp_set:
                    degree += 1
            degrees[(r, c)] = degree

        assert all(degree in (1, 2) for degree in degrees.values()), "each segment must be a simple 4-connected polyline"
        endpoints = sorted(coord for coord, degree in degrees.items() if degree == 1)
        assert len(endpoints) == 2, "each segment must have exactly two endpoints"

        interior = [coord for coord in comp if degrees[coord] == 2]
        assert interior, "each segment must contain at least one non-endpoint body pixel"
        body_colors = {int(g[r, c]) for r, c in interior}
        assert len(body_colors) == 1, "all non-endpoint pixels in a segment must share one body color"
        body_color = int(next(iter(body_colors)))

        r0, c0, r1, c1 = _bbox(comp)
        locals_sorted = sorted((r - r0, c - c0) for r, c in comp)
        colors = {(r - r0, c - c0): int(g[r, c]) for r, c in comp}
        endpoint_info: list[dict] = []
        endpoint_lookup: dict[Coord, int] = {}
        for abs_coord in endpoints:
            neighbors: list[Coord] = []
            r, c = abs_coord
            for dr, dc in DIR4:
                nbr = (r + dr, c + dc)
                if nbr in comp_set:
                    neighbors.append(nbr)
            assert len(neighbors) == 1
            local = (r - r0, c - c0)
            inner = (neighbors[0][0] - r0, neighbors[0][1] - c0)
            endpoint_lookup[local] = len(endpoint_info)
            endpoint_info.append(
                {
                    "abs": abs_coord,
                    "local": local,
                    "marker": int(g[r, c]),
                    "inner_local": inner,
                }
            )

        segments.append(
            {
                "bbox": (r0, c0, r1, c1),
                "shape": (r1 - r0 + 1, c1 - c0 + 1),
                "cells": locals_sorted,
                "colors": colors,
                "body": body_color,
                "endpoints": endpoint_info,
                "endpoint_lookup": endpoint_lookup,
            }
        )

    return segments


def _build_marker_graph(segments: list[dict]) -> tuple[list[dict], list[list[int]]]:
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for seg_idx, seg in enumerate(segments):
        for end_idx, end in enumerate(seg["endpoints"]):
            by_color[int(end["marker"])].append((seg_idx, end_idx))

    edges: list[dict] = []
    for color in sorted(by_color):
        pair = by_color[color]
        assert len(pair) == 2, f"marker color {color} must appear exactly twice"
        (a, a_end), (b, b_end) = pair
        assert a != b, "paired endpoints must belong to different segments"
        edges.append(
            {
                "color": int(color),
                "a": int(a),
                "a_end": int(a_end),
                "b": int(b),
                "b_end": int(b_end),
            }
        )

    adj = [[] for _ in segments]
    for edge_idx, edge in enumerate(edges):
        adj[int(edge["a"])].append(edge_idx)
        adj[int(edge["b"])].append(edge_idx)

    assert all(len(neighbors) == 2 for neighbors in adj), "paired endpoints must form one cyclic segment graph"
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for edge_idx in adj[node]:
            edge = edges[edge_idx]
            other = int(edge["b"]) if int(edge["a"]) == node else int(edge["a"])
            if other not in seen:
                seen.add(other)
                stack.append(other)
    assert len(seen) == len(segments), "the segment graph must be connected"
    return edges, adj


def _anchor_segment(segments: list[dict]) -> int:
    best_seg = 0
    best_endpoint = None
    for seg_idx, seg in enumerate(segments):
        for end in seg["endpoints"]:
            coord = tuple(end["abs"])
            if best_endpoint is None or coord < best_endpoint:
                best_endpoint = coord
                best_seg = seg_idx
    return int(best_seg)


def _edge_step(edge: dict, current: int) -> tuple[int, int, int]:
    if int(edge["a"]) == current:
        return int(edge["a_end"]), int(edge["b"]), int(edge["b_end"])
    return int(edge["b_end"]), int(edge["a"]), int(edge["a_end"])


def _cycle_traversals(edges: list[dict], adj: list[list[int]], anchor: int) -> list[list[dict]]:
    traversals: list[list[dict]] = []
    signatures: set[tuple[tuple[int, int, int, int], ...]] = set()

    for first_edge in adj[anchor]:
        steps: list[dict] = []
        used_edges: set[int] = set()
        visited_nodes = [anchor]
        current = anchor
        edge_idx = int(first_edge)
        ok = True

        while True:
            if edge_idx in used_edges:
                ok = False
                break
            edge = edges[edge_idx]
            from_end, nxt, to_end = _edge_step(edge, current)
            steps.append(
                {
                    "edge": edge_idx,
                    "from": int(current),
                    "from_end": int(from_end),
                    "to": int(nxt),
                    "to_end": int(to_end),
                }
            )
            used_edges.add(edge_idx)
            current = int(nxt)
            if current == anchor:
                break
            visited_nodes.append(current)
            next_edges = [eid for eid in adj[current] if eid not in used_edges]
            if len(next_edges) != 1:
                ok = False
                break
            edge_idx = int(next_edges[0])

        if not ok:
            continue
        if len(used_edges) != len(edges):
            continue
        if len(set(visited_nodes)) != len(adj):
            continue
        signature = tuple((step["from"], step["from_end"], step["to"], step["to_end"]) for step in steps)
        if signature in signatures:
            continue
        signatures.add(signature)
        traversals.append(steps)

    assert traversals, "failed to derive a cyclic traversal order from marker pairings"
    return traversals


def _translated_lookup(segment: dict, pos: Coord) -> dict[Coord, Coord]:
    return {(pos[0] + r, pos[1] + c): (r, c) for r, c in segment["cells"]}


def _required_pair_keys(pairs: list[tuple[int, int, int, int]]) -> set[frozenset[tuple[int, int]]]:
    out: set[frozenset[tuple[int, int]]] = set()
    for a_seg, a_end, b_seg, b_end in pairs:
        out.add(frozenset({(int(a_seg), int(a_end)), (int(b_seg), int(b_end))}))
    return out


def _try_place_segment(
    segments: list[dict],
    seg_idx: int,
    pos: Coord,
    occupied: dict[Coord, tuple[int, Coord]],
    required_pairs: list[tuple[int, int, int, int]],
) -> dict[Coord, Coord] | None:
    segment = segments[seg_idx]
    placed = _translated_lookup(segment, pos)
    for abs_coord in placed:
        if abs_coord in occupied:
            return None

    required_keys = _required_pair_keys(required_pairs)
    seen_keys: set[frozenset[tuple[int, int]]] = set()

    for abs_coord, local in placed.items():
        for dr, dc in DIR4:
            nbr = (abs_coord[0] + dr, abs_coord[1] + dc)
            if nbr not in occupied:
                continue
            other_seg, other_local = occupied[nbr]
            this_end = segment["endpoint_lookup"].get(local)
            other_end = segments[other_seg]["endpoint_lookup"].get(other_local)
            if this_end is None or other_end is None:
                return None
            key = frozenset({(seg_idx, int(this_end)), (int(other_seg), int(other_end))})
            if key not in required_keys:
                return None
            seen_keys.add(key)

    if seen_keys != required_keys:
        return None
    return placed


def _final_arrangement_is_valid(
    segments: list[dict],
    edges: list[dict],
    positions: dict[int, Coord],
) -> bool:
    occupied: dict[Coord, tuple[int, Coord]] = {}
    for seg_idx, segment in enumerate(segments):
        translated = _translated_lookup(segment, positions[seg_idx])
        for abs_coord, local in translated.items():
            if abs_coord in occupied:
                return False
            occupied[abs_coord] = (seg_idx, local)

    valid_pairs = _required_pair_keys(
        [(edge["a"], edge["a_end"], edge["b"], edge["b_end"]) for edge in edges]
    )
    seen_pairs: set[frozenset[tuple[int, int]]] = set()

    for abs_coord, (seg_idx, local) in occupied.items():
        degree = 0
        for dr, dc in DIR4:
            nbr = (abs_coord[0] + dr, abs_coord[1] + dc)
            if nbr not in occupied:
                continue
            degree += 1
            other_seg, other_local = occupied[nbr]
            if other_seg == seg_idx:
                continue
            this_end = segments[seg_idx]["endpoint_lookup"].get(local)
            other_end = segments[other_seg]["endpoint_lookup"].get(other_local)
            if this_end is None or other_end is None:
                return False
            pair = frozenset({(seg_idx, int(this_end)), (int(other_seg), int(other_end))})
            if pair not in valid_pairs:
                return False
            seen_pairs.add(pair)
        if degree != 2:
            return False

    if seen_pairs != valid_pairs:
        return False
    return _connected(set(occupied))


def _render_output(segments: list[dict], positions: dict[int, Coord]) -> np.ndarray:
    all_cells: list[Coord] = []
    for seg_idx, segment in enumerate(segments):
        top, left = positions[seg_idx]
        all_cells.extend((top + r, left + c) for r, c in segment["cells"])

    r0, c0, r1, c1 = _bbox(all_cells)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for seg_idx, segment in enumerate(segments):
        top, left = positions[seg_idx]
        body = int(segment["body"])
        for r, c in segment["cells"]:
            out[top + r - r0, left + c - c0] = body
    return out


def _enumerate_outputs_for_traversal(
    segments: list[dict],
    edges: list[dict],
    steps: list[dict],
    limit: int,
) -> list[np.ndarray]:
    anchor = int(steps[0]["from"])
    positions: dict[int, Coord] = {anchor: (0, 0)}
    occupied = {abs_coord: (anchor, local) for abs_coord, local in _translated_lookup(segments[anchor], (0, 0)).items()}
    unique: dict[bytes, np.ndarray] = {}

    def record_solution() -> None:
        if len(unique) >= limit:
            return
        if not _final_arrangement_is_valid(segments, edges, positions):
            return
        out = _render_output(segments, positions)
        key = out.tobytes() + str(out.shape).encode()
        unique.setdefault(key, out)

    def rec(step_idx: int) -> None:
        if len(unique) >= limit:
            return
        if step_idx == len(steps) - 1:
            close = steps[step_idx]
            last_seg = int(close["from"])
            anchor_seg = int(close["to"])
            assert anchor_seg == anchor
            pos_last = positions[last_seg]
            pos_anchor = positions[anchor_seg]
            end_last = tuple(segments[last_seg]["endpoints"][int(close["from_end"])]["local"])
            end_anchor = tuple(segments[anchor_seg]["endpoints"][int(close["to_end"])]["local"])
            abs_last = (pos_last[0] + end_last[0], pos_last[1] + end_last[1])
            abs_anchor = (pos_anchor[0] + end_anchor[0], pos_anchor[1] + end_anchor[1])
            if _manhattan(abs_last, abs_anchor) == 1:
                record_solution()
            return

        step = steps[step_idx]
        from_seg = int(step["from"])
        to_seg = int(step["to"])
        assert from_seg in positions and to_seg not in positions

        from_local = tuple(segments[from_seg]["endpoints"][int(step["from_end"])]["local"])
        to_local = tuple(segments[to_seg]["endpoints"][int(step["to_end"])]["local"])
        pos_from = positions[from_seg]
        abs_from = (pos_from[0] + from_local[0], pos_from[1] + from_local[1])

        required = [(from_seg, int(step["from_end"]), to_seg, int(step["to_end"]))]
        if step_idx == len(steps) - 2:
            close = steps[-1]
            required.append(
                (
                    int(close["from"]),
                    int(close["from_end"]),
                    int(close["to"]),
                    int(close["to_end"]),
                )
            )

        for dr, dc in DIR4:
            pos_to = (abs_from[0] + dr - to_local[0], abs_from[1] + dc - to_local[1])
            placed = _try_place_segment(segments, to_seg, pos_to, occupied, required)
            if placed is None:
                continue
            positions[to_seg] = pos_to
            for abs_coord, local in placed.items():
                occupied[abs_coord] = (to_seg, local)
            rec(step_idx + 1)
            for abs_coord in placed:
                occupied.pop(abs_coord)
            positions.pop(to_seg)

    rec(0)
    return list(unique.values())


def _solve_outputs(grid: np.ndarray, limit: int = 2) -> list[np.ndarray]:
    segments = _extract_segments(np.asarray(grid, dtype=int))
    edges, adj = _build_marker_graph(segments)
    anchor = _anchor_segment(segments)
    traversals = _cycle_traversals(edges, adj, anchor)

    unique: dict[bytes, np.ndarray] = {}
    for steps in traversals:
        for out in _enumerate_outputs_for_traversal(segments, edges, steps, limit):
            key = out.tobytes() + str(out.shape).encode()
            unique.setdefault(key, out)
            if len(unique) >= limit:
                return list(unique.values())
    return list(unique.values())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the scattered colored path segments into one continuous loop by matching same-color endpoint markers.

    Input structure:
    - The background is black.
    - Several disconnected one-pixel-wide 4-connected path segments are scattered on the grid.
    - Each segment has one body color on its non-endpoint pixels, while its two endpoint pixels are replaced by marker colors.
    - Every marker color appears exactly twice, and the two matching occurrences lie on endpoints of two different segments.

    Rule:
    - Treat each connected non-black component as one path segment and identify its two degree-1 endpoint pixels.
    - Pair endpoints with the same marker color, fix the segment containing the top-leftmost endpoint, and translate the other segments so each matched endpoint pair becomes 4-neighbor adjacent.
    - Keep only assemblies that produce one non-overlapping continuous one-pixel-wide path, then replace every endpoint marker with that endpoint segment's body color.
    - Return the minimum bounding box of the assembled path on a black background.

    Key insight:
    - The endpoint colors are connection codes rather than paint: same-color endpoint pairs define which segments must touch, and the valid assembly is the one where those coded contacts form a single coherent path.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    solutions = _solve_outputs(g, limit=2)
    assert len(solutions) == 1, "the endpoint pairings must determine a unique assembled path"
    return solutions[0]


def _ordered_rectangle_cycle(height: int, width: int) -> list[Coord]:
    coords: list[Coord] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height):
        coords.append((r, width - 1))
    for c in range(width - 2, -1, -1):
        coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _normalized_cycle(cycle: list[Coord]) -> list[Coord]:
    r0, c0, _r1, _c1 = _bbox(cycle)
    return [(r - r0, c - c0) for r, c in cycle]


def _turn_count(cycle: list[Coord]) -> int:
    turns = 0
    n = len(cycle)
    for i in range(n):
        prev = cycle[i - 1]
        cur = cycle[i]
        nxt = cycle[(i + 1) % n]
        d1 = (cur[0] - prev[0], cur[1] - prev[1])
        d2 = (nxt[0] - cur[0], nxt[1] - cur[1])
        if d1 != d2:
            turns += 1
    return turns


def _is_ordered_cycle(cycle: list[Coord]) -> bool:
    if len(cycle) < 4 or len(set(cycle)) != len(cycle):
        return False
    n = len(cycle)
    for i in range(n):
        if _manhattan(cycle[i], cycle[(i + 1) % n]) != 1:
            return False
    coords = set(cycle)
    for coord in coords:
        degree = 0
        for dr, dc in DIR4:
            if (coord[0] + dr, coord[1] + dc) in coords:
                degree += 1
        if degree != 2:
            return False
    return _connected(coords)


def _insert_bump(cycle: list[Coord], rng: random.Random) -> list[Coord] | None:
    edge_indices = list(range(len(cycle)))
    rng.shuffle(edge_indices)
    coord_set = set(cycle)

    for idx in edge_indices:
        a = cycle[idx]
        b = cycle[(idx + 1) % len(cycle)]
        if a[0] == b[0]:
            deltas = [-1, 1]
            rng.shuffle(deltas)
            for delta in deltas:
                p1 = (a[0] + delta, a[1])
                p2 = (b[0] + delta, b[1])
                if p1 in coord_set or p2 in coord_set:
                    continue
                candidate = cycle[: idx + 1] + [p1, p2] + cycle[idx + 1 :]
                if _is_ordered_cycle(candidate):
                    return candidate
        else:
            deltas = [-1, 1]
            rng.shuffle(deltas)
            for delta in deltas:
                p1 = (a[0], a[1] + delta)
                p2 = (b[0], b[1] + delta)
                if p1 in coord_set or p2 in coord_set:
                    continue
                candidate = cycle[: idx + 1] + [p1, p2] + cycle[idx + 1 :]
                if _is_ordered_cycle(candidate):
                    return candidate
    return None


def _insert_specific_bump(cycle: list[Coord], a: Coord, b: Coord, p1: Coord, p2: Coord) -> list[Coord]:
    for idx in range(len(cycle)):
        if cycle[idx] == a and cycle[(idx + 1) % len(cycle)] == b:
            return cycle[: idx + 1] + [p1, p2] + cycle[idx + 1 :]
    raise AssertionError("failed to find the requested edge for bump insertion")


def _walk_cycle(start: Coord, runs: list[tuple[Coord, int]]) -> list[Coord]:
    r, c = start
    coords = [(r, c)]
    for (dr, dc), length in runs:
        for _ in range(length):
            r += dr
            c += dc
            coords.append((r, c))
    assert coords[-1] == start
    return coords[:-1]


def _make_cycle(rng: random.Random) -> list[Coord] | None:
    if rng.random() < 0.5:
        a = rng.randint(3, 6)
        b = rng.randint(1, 3)
        c = rng.randint(2, 4)
        d = rng.randint(2, 4)
        cycle = _walk_cycle(
            (0, 0),
            [
                ((0, 1), a),
                ((1, 0), b),
                ((0, 1), c),
                ((1, 0), d),
                ((0, -1), a + c),
                ((-1, 0), b + d),
            ],
        )
    else:
        a = rng.randint(3, 6)
        b = rng.randint(1, 3)
        c = rng.randint(2, 4)
        d = rng.randint(2, 4)
        cycle = _walk_cycle(
            (0, 0),
            [
                ((1, 0), a),
                ((0, 1), b),
                ((1, 0), c),
                ((0, 1), d),
                ((-1, 0), a + c),
                ((0, -1), b + d),
            ],
        )

    if rng.random() < 0.5:
        cycle = [(r, -c) for r, c in cycle]
    if rng.random() < 0.5:
        cycle = [(-r, c) for r, c in cycle]
    if rng.random() < 0.5:
        cycle = [(c, r) for r, c in cycle]
    cycle = _normalized_cycle(cycle)
    if len(cycle) < MIN_SEGMENT_LEN * min(SEGMENT_COUNT_CHOICES):
        return None
    if _turn_count(cycle) < 6:
        return None
    r0, c0, r1, c1 = _bbox(cycle)
    if max(r1 - r0 + 1, c1 - c0 + 1) > MAX_CYCLE_SIDE:
        return None
    assert _is_ordered_cycle(cycle)
    return cycle


def _sample_segment_lengths(total: int, count: int, rng: random.Random) -> list[int] | None:
    extra = total - count * MIN_SEGMENT_LEN
    if extra < 0:
        return None
    lengths = [MIN_SEGMENT_LEN] * count
    for _ in range(extra):
        lengths[rng.randrange(count)] += 1
    rng.shuffle(lengths)
    return lengths


def _partition_cycle(cycle: list[Coord], count: int, rng: random.Random) -> list[list[Coord]] | None:
    lengths = _sample_segment_lengths(len(cycle), count, rng)
    if lengths is None:
        return None
    rotated = cycle[:]
    if rng.random() < 0.5:
        rotated = list(reversed(rotated))
    shift = rng.randrange(len(rotated))
    rotated = rotated[shift:] + rotated[:shift]

    parts: list[list[Coord]] = []
    start = 0
    for length in lengths:
        parts.append(rotated[start : start + length])
        start += length
    assert start == len(rotated)
    return parts


def _render_expected(parts: list[list[Coord]], body_colors: list[int]) -> np.ndarray:
    all_cells = [coord for part in parts for coord in part]
    r0, c0, r1, c1 = _bbox(all_cells)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BLACK, dtype=int)
    for part, body in zip(parts, body_colors, strict=False):
        for r, c in part:
            out[r - r0, c - c0] = int(body)
    return out


def _build_segment_patch(
    coords: list[Coord],
    body_color: int,
    start_marker: int,
    end_marker: int,
) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    local_coords = [(r - r0, c - c0) for r, c in coords]
    colors = {local: int(body_color) for local in local_coords}
    colors[local_coords[0]] = int(start_marker)
    colors[local_coords[-1]] = int(end_marker)
    return {
        "shape": (r1 - r0 + 1, c1 - c0 + 1),
        "cells": local_coords,
        "colors": colors,
    }


def _reserve_coords(cells: list[Coord]) -> set[Coord]:
    reserved: set[Coord] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                reserved.add((r + dr, c + dc))
    return reserved


def _scatter_patches(patches: list[dict], rng: random.Random) -> np.ndarray | None:
    canvas = np.full((WORK_CANVAS, WORK_CANVAS), BLACK, dtype=int)
    reserved = np.zeros((WORK_CANVAS, WORK_CANVAS), dtype=bool)
    order = sorted(range(len(patches)), key=lambda idx: len(patches[idx]["cells"]), reverse=True)

    for idx in order:
        patch = patches[idx]
        h, w = patch["shape"]
        candidates = [
            (top, left)
            for top in range(PLACEMENT_PAD, WORK_CANVAS - h - PLACEMENT_PAD + 1)
            for left in range(PLACEMENT_PAD, WORK_CANVAS - w - PLACEMENT_PAD + 1)
        ]
        rng.shuffle(candidates)
        placed = False
        for top, left in candidates[:PLACEMENT_TRIES]:
            abs_cells = [(top + r, left + c) for r, c in patch["cells"]]
            reserve = _reserve_coords(abs_cells)
            if any(not (0 <= rr < WORK_CANVAS and 0 <= cc < WORK_CANVAS) for rr, cc in reserve):
                continue
            if any(reserved[rr, cc] for rr, cc in reserve):
                continue
            for rr, cc in reserve:
                reserved[rr, cc] = True
            for local, color in patch["colors"].items():
                canvas[top + local[0], left + local[1]] = int(color)
            placed = True
            break
        if not placed:
            return None

    coords = np.argwhere(canvas != BLACK)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_margin = rng.randint(0, min(3, int(r0)))
    left_margin = rng.randint(0, min(3, int(c0)))
    bottom_margin = rng.randint(0, min(3, WORK_CANVAS - 1 - int(r1)))
    right_margin = rng.randint(0, min(3, WORK_CANVAS - 1 - int(c1)))
    return canvas[
        int(r0) - top_margin : int(r1) + bottom_margin + 1,
        int(c0) - left_margin : int(c1) + right_margin + 1,
    ]


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        cycle = _make_cycle(rng)
        if cycle is None:
            continue

        max_segments = min(max(SEGMENT_COUNT_CHOICES), len(PALETTE) // 2, len(cycle) // MIN_SEGMENT_LEN)
        segment_choices = [count for count in SEGMENT_COUNT_CHOICES if count <= max_segments]
        if not segment_choices:
            continue
        segment_count = rng.choice(segment_choices)

        parts = _partition_cycle(cycle, segment_count, rng)
        if parts is None:
            continue

        colors = rng.sample(PALETTE, 2 * segment_count)
        body_colors = list(colors[:segment_count])
        marker_colors = list(colors[segment_count:])
        expected = _render_expected(parts, body_colors)

        patches: list[dict] = []
        for idx, part in enumerate(parts):
            patches.append(
                _build_segment_patch(
                    part,
                    body_colors[idx],
                    marker_colors[(idx - 1) % segment_count],
                    marker_colors[idx],
                )
            )

        scattered = _scatter_patches(patches, rng)
        if scattered is None:
            continue
        if max(scattered.shape) > 30:
            continue

        solutions = _solve_outputs(scattered, limit=2)
        if len(solutions) != 1:
            continue
        if not np.array_equal(solutions[0], expected):
            continue
        return scattered

    raise RuntimeError("failed to generate a valid endpoint-pair assembly puzzle")
