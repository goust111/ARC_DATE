from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
MAROON = 9
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8]

MIN_GRID = 14
MAX_GRID = 22
MIN_SEGMENTS = 3
MAX_SEGMENTS = 6
MIN_SEG_LEN = 3
MAX_SEG_LEN = 6
MAX_GEN_ATTEMPTS = 500

DIRS = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # R, D, L, U


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return a[0] + b[0], a[1] + b[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _marker_info(g: np.ndarray) -> tuple[set[tuple[int, int]], str]:
    coords = [tuple(map(int, p)) for p in np.argwhere(g == MAROON)]
    assert len(coords) == 4, "expected one 2x2 maroon start marker"
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    assert len(rows) == 2 and len(cols) == 2, "marker must be 2x2"
    assert rows[1] == rows[0] + 1 and cols[1] == cols[0] + 1
    marker = set(coords)
    expected = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    assert marker == expected
    h, w = g.shape
    if rows[0] == 0 and cols[0] == 0:
        corner = "tl"
    elif rows[0] == 0 and cols[1] == w - 1:
        corner = "tr"
    elif rows[1] == h - 1 and cols[0] == 0:
        corner = "bl"
    elif rows[1] == h - 1 and cols[1] == w - 1:
        corner = "br"
    else:
        raise AssertionError("marker must sit in a grid corner")
    return marker, corner


def _segment_endpoints(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]]:
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    assert len(rows) == 1 or len(cols) == 1, "segment must be a straight line"
    if len(rows) == 1:
        ordered = sorted(coords, key=lambda p: p[1])
    else:
        ordered = sorted(coords, key=lambda p: p[0])
    assert MIN_SEG_LEN <= len(ordered) <= MAX_SEG_LEN
    return ordered, ordered[0], ordered[-1]


def _parse_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    marker, _corner = _marker_info(g)

    fg_mask = (g != BG) & (g != MAROON)
    comps = _components(fg_mask)
    assert comps, "expected path content"

    segments: list[dict] = []
    gate_pixels: dict[tuple[int, int], int] = {}
    path_lookup: dict[tuple[int, int], int] = {}
    endpoint_lookup: dict[tuple[int, int], int] = {}

    for comp in comps:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "components must be monochrome"
        color = colors.pop()
        if len(comp) == 1:
            gate_pixels[comp[0]] = color
            continue
        ordered, end_a, end_b = _segment_endpoints(comp)
        sid = len(segments)
        segment = {
            "id": sid,
            "coords": tuple(ordered),
            "coord_set": set(ordered),
            "color": color,
            "endpoints": (end_a, end_b),
        }
        segments.append(segment)
        for cell in ordered:
            assert cell not in path_lookup
            path_lookup[cell] = sid
        endpoint_lookup[end_a] = sid
        endpoint_lookup[end_b] = sid

    assert segments, "expected at least one path segment"
    assert gate_pixels, "expected at least one gate"

    gates: list[dict] = []
    endpoint_to_gate: dict[tuple[int, int], int] = {}
    used_gate_pixels: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BG:
                continue
            cell = (r, c)
            nbrs = _neighbors4(r, c)
            gate_nbrs = [p for p in nbrs if p in gate_pixels]
            path_nbrs = [p for p in nbrs if p in path_lookup]
            if len(gate_nbrs) != 2 or len(path_nbrs) != 2:
                continue
            if len(gate_nbrs) + len(path_nbrs) != 4:
                continue
            gate_colors = {gate_pixels[p] for p in gate_nbrs}
            if len(gate_colors) != 1:
                continue
            if any(p not in endpoint_lookup for p in path_nbrs):
                continue
            sid_a = path_lookup[path_nbrs[0]]
            sid_b = path_lookup[path_nbrs[1]]
            if sid_a == sid_b:
                continue
            gid = len(gates)
            gates.append(
                {
                    "id": gid,
                    "gap": cell,
                    "color": gate_colors.pop(),
                    "gate_pixels": tuple(gate_nbrs),
                    "links": ((sid_a, path_nbrs[0]), (sid_b, path_nbrs[1])),
                }
            )
            for gp in gate_nbrs:
                assert gp not in used_gate_pixels
                used_gate_pixels.add(gp)
            for ep in path_nbrs:
                assert ep not in endpoint_to_gate
                endpoint_to_gate[ep] = gid

    assert used_gate_pixels == set(gate_pixels), "every gate pixel must belong to exactly one gate"
    assert len(gates) + 1 == len(segments), "segments and gates must form one chain"

    start_candidates: list[tuple[int, tuple[int, int]]] = []
    free_endpoints: list[tuple[int, tuple[int, int]]] = []
    for seg in segments:
        degree = 0
        for ep in seg["endpoints"]:
            touches_marker = any(nb in marker for nb in _neighbors4(*ep))
            touches_gate = ep in endpoint_to_gate
            if touches_marker:
                start_candidates.append((seg["id"], ep))
                degree += 1
            elif touches_gate:
                degree += 1
            else:
                free_endpoints.append((seg["id"], ep))
        assert degree in (1, 2)

    assert len(start_candidates) == 1, "expected one marker-adjacent endpoint"
    assert len(free_endpoints) == 1, "expected one terminal endpoint"

    return {
        "grid": g,
        "marker": marker,
        "segments": segments,
        "gates": gates,
        "path_lookup": path_lookup,
        "endpoint_to_gate": endpoint_to_gate,
        "start": start_candidates[0],
        "end": free_endpoints[0],
    }


def _segment_order_from(seg: dict, start_endpoint: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = list(seg["coords"])
    if ordered[0] == start_endpoint:
        return ordered
    assert ordered[-1] == start_endpoint
    ordered.reverse()
    return ordered


def main(grid: np.ndarray) -> np.ndarray:
    data = _parse_structure(grid)
    g = data["grid"]
    out = np.full_like(g, BG)
    for r, c in data["marker"]:
        out[r, c] = MAROON

    segments = data["segments"]
    gates = data["gates"]
    endpoint_to_gate = data["endpoint_to_gate"]

    current_sid, enter_endpoint = data["start"]
    current_color = int(segments[current_sid]["color"])
    visited_segments: set[int] = set()
    visited_gates: set[int] = set()

    while True:
        seg = segments[current_sid]
        assert current_sid not in visited_segments
        ordered = _segment_order_from(seg, enter_endpoint)
        for r, c in ordered:
            out[r, c] = current_color
        visited_segments.add(current_sid)
        exit_endpoint = ordered[-1]
        gate_id = endpoint_to_gate.get(exit_endpoint)
        if gate_id is None:
            break
        assert gate_id not in visited_gates
        gate = gates[gate_id]
        current_color = int(gate["color"])
        gr, gc = gate["gap"]
        out[gr, gc] = current_color
        visited_gates.add(gate_id)
        (sid_a, ep_a), (sid_b, ep_b) = gate["links"]
        if current_sid == sid_a:
            current_sid, enter_endpoint = sid_b, ep_b
        else:
            assert current_sid == sid_b
            current_sid, enter_endpoint = sid_a, ep_a

    assert len(visited_segments) == len(segments)
    assert len(visited_gates) == len(gates)
    return out


def _marker_cells(shape: tuple[int, int], corner: str) -> tuple[set[tuple[int, int]], list[tuple[tuple[int, int], int]]]:
    h, w = shape
    if corner == "tl":
        marker = {(0, 0), (0, 1), (1, 0), (1, 1)}
        options = [((1, 2), 0), ((2, 1), 1)]
    elif corner == "tr":
        marker = {(0, w - 2), (0, w - 1), (1, w - 2), (1, w - 1)}
        options = [((1, w - 3), 2), ((2, w - 2), 1)]
    elif corner == "bl":
        marker = {(h - 2, 0), (h - 2, 1), (h - 1, 0), (h - 1, 1)}
        options = [((h - 2, 2), 0), ((h - 3, 1), 3)]
    else:
        assert corner == "br"
        marker = {(h - 2, w - 2), (h - 2, w - 1), (h - 1, w - 2), (h - 1, w - 1)}
        options = [((h - 2, w - 3), 2), ((h - 3, w - 2), 3)]
    return marker, options


def _segment_cells(start: tuple[int, int], dir_idx: int, length: int) -> list[tuple[int, int]]:
    dr, dc = DIRS[dir_idx]
    return [(start[0] + dr * k, start[1] + dc * k) for k in range(length)]


def _can_place_segment(
    shape: tuple[int, int],
    coords: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    marker: set[tuple[int, int]],
    is_first: bool,
) -> bool:
    coord_set = set(coords)
    if len(coord_set) != len(coords):
        return False
    for cell in coords:
        if not _in_bounds(shape, cell) or cell in occupied:
            return False
    for idx, cell in enumerate(coords):
        allowed = set()
        if idx > 0:
            allowed.add(coords[idx - 1])
        if idx + 1 < len(coords):
            allowed.add(coords[idx + 1])
        if is_first and idx == 0:
            allowed |= marker
        for nb in _neighbors4(*cell):
            if nb in occupied and nb not in allowed:
                return False
    if is_first:
        touch_count = sum(1 for nb in _neighbors4(*coords[0]) if nb in marker)
        if touch_count != 1:
            return False
    return True


def _can_place_gate(
    shape: tuple[int, int],
    gap: tuple[int, int],
    gate_cells: tuple[tuple[int, int], tuple[int, int]],
    occupied: set[tuple[int, int]],
    prev_end: tuple[int, int],
) -> bool:
    if not _in_bounds(shape, gap) or gap in occupied:
        return False
    for nb in _neighbors4(*gap):
        if nb in occupied and nb != prev_end:
            return False
    for cell in gate_cells:
        if not _in_bounds(shape, cell) or cell in occupied:
            return False
        for nb in _neighbors4(*cell):
            if nb in occupied:
                return False
    return gate_cells[0] != gate_cells[1]


def _build_layout(shape: tuple[int, int], segment_count: int) -> tuple[set[tuple[int, int]], list[list[tuple[int, int]]], list[dict]] | None:
    marker, options = _marker_cells(shape, random.choice(["tl", "tr", "bl", "br"]))
    start, dir_idx = random.choice(options)
    occupied = set(marker)
    segments: list[list[tuple[int, int]]] = []
    gates: list[dict] = []

    def dfs(index: int, seg_start: tuple[int, int], current_dir: int) -> bool:
        lengths = list(range(MIN_SEG_LEN, MAX_SEG_LEN + 1))
        random.shuffle(lengths)
        for length in lengths:
            coords = _segment_cells(seg_start, current_dir, length)
            if not _can_place_segment(shape, coords, occupied, marker, is_first=index == 0):
                continue

            new_seg_cells = set(coords)
            segments.append(coords)
            occupied.update(new_seg_cells)

            if index == segment_count - 1:
                return True

            prev_end = coords[-1]
            gap = _add(prev_end, DIRS[current_dir])
            next_dirs = [current_dir, (current_dir + 1) % 4, (current_dir - 1) % 4]
            random.shuffle(next_dirs)
            for next_dir in next_dirs:
                next_start = _add(gap, DIRS[next_dir])
                neighbor_set = set(_neighbors4(*gap))
                gate_cells = tuple(sorted(neighbor_set - {prev_end, next_start}))
                assert len(gate_cells) == 2
                if not _can_place_gate(shape, gap, gate_cells, occupied, prev_end):
                    continue
                occupied.update(gate_cells)
                gates.append({"gap": gap, "gate_cells": gate_cells})
                if dfs(index + 1, next_start, next_dir):
                    return True
                gates.pop()
                occupied.difference_update(gate_cells)

            occupied.difference_update(new_seg_cells)
            segments.pop()
        return False

    if dfs(0, start, dir_idx):
        return marker, segments, gates
    return None


def _assign_colors(segments: list[list[tuple[int, int]]], gates: list[dict]) -> tuple[list[int], list[int]] | None:
    if len(segments) > len(PALETTE):
        return None
    segment_colors = random.sample(PALETTE, len(segments))
    gate_colors: list[int] = []
    used_gate_colors: set[int] = set()
    for idx in range(len(gates)):
        forbidden = {segment_colors[idx], segment_colors[idx + 1], *used_gate_colors}
        choices = [color for color in PALETTE if color not in forbidden]
        if not choices:
            return None
        gate_color = random.choice(choices)
        gate_colors.append(gate_color)
        used_gate_colors.add(gate_color)
    return segment_colors, gate_colors


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    segment_count = random.randint(MIN_SEGMENTS, MAX_SEGMENTS)
    layout = _build_layout((h, w), segment_count)
    if layout is None:
        return None
    marker, segments, gates = layout
    colors = _assign_colors(segments, gates)
    if colors is None:
        return None
    segment_colors, gate_colors = colors

    grid = np.full((h, w), BG, dtype=int)
    for r, c in marker:
        grid[r, c] = MAROON
    for coords, color in zip(segments, segment_colors):
        for r, c in coords:
            grid[r, c] = color
    for gate, color in zip(gates, gate_colors):
        for r, c in gate["gate_cells"]:
            grid[r, c] = color

    parsed = _parse_structure(grid)
    assert len(parsed["segments"]) == len(segments)
    assert len(parsed["gates"]) == len(gates)
    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid path-and-gate instance")
