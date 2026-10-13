from __future__ import annotations

import random
from collections import defaultdict

import numpy as np


BG = 0
GREY = 5
OUTPUT_COLORS = (2, 1, 3, 4)  # red, blue, green, yellow
INPUT_COLOR_POOL = (6, 7, 8, 9)
SHAPE_ORDER = ("dot", "hline2", "vline2", "square2")
SHAPE_RANK = {shape: idx for idx, shape in enumerate(SHAPE_ORDER)}
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
CARDINALS = ((0, 1), (0, -1), (1, 0), (-1, 0))
MAX_GENERATION_ATTEMPTS = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(coords)
    return comps


def _color_components(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if bool(seen[r, c]):
                continue
            color = int(grid[r, c])
            if color in {BG, GREY}:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIR4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not bool(seen[nr, nc]) and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append({"color": int(color), "coords": coords})
    return comps


def _shape_from_marker(coords: list[tuple[int, int]]) -> str:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    area = len(coords)
    height = max(rows) - min(rows) + 1
    width = max(cols) - min(cols) + 1
    if area == 1:
        return "dot"
    if area == 2 and height == 1 and width == 2:
        return "hline2"
    if area == 2 and height == 2 and width == 1:
        return "vline2"
    if area == 4 and height == 2 and width == 2:
        return "square2"
    raise AssertionError(f"unsupported grey marker shape: {sorted(coords)}")


def _segment_from_component(comp: dict[str, object], seg_id: int) -> dict[str, object]:
    coords = [(int(r), int(c)) for r, c in comp["coords"]]
    color = int(comp["color"])
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})
    if len(rows) == 1:
        row = int(rows[0])
        cols_sorted = sorted(c for _, c in coords)
        assert cols_sorted == list(range(cols_sorted[0], cols_sorted[-1] + 1))
        endpoints = [(row, int(cols_sorted[0])), (row, int(cols_sorted[-1]))]
        orientation = "h"
    elif len(cols) == 1:
        col = int(cols[0])
        rows_sorted = sorted(r for r, _ in coords)
        assert rows_sorted == list(range(rows_sorted[0], rows_sorted[-1] + 1))
        endpoints = [(int(rows_sorted[0]), col), (int(rows_sorted[-1]), col)]
        orientation = "v"
    else:
        raise AssertionError(f"segment is not a straight 1-pixel line: {sorted(coords)}")
    return {
        "id": int(seg_id),
        "color": int(color),
        "coords": coords,
        "cells": set(coords),
        "length": int(len(coords)),
        "orientation": orientation,
        "endpoints": [
            {"coord": endpoints[0], "shape": None},
            {"coord": endpoints[1], "shape": None},
        ],
    }


def _touches_endpoint(marker_coords: list[tuple[int, int]], endpoint: tuple[int, int]) -> bool:
    er, ec = endpoint
    return any(abs(er - r) + abs(ec - c) == 1 for r, c in marker_coords)


def _extract_segments_and_markers(grid: np.ndarray) -> list[dict[str, object]]:
    segments = [_segment_from_component(comp, seg_id) for seg_id, comp in enumerate(_color_components(grid))]
    markers = [_shape_from_marker(coords) for coords in _components_of_color(grid, GREY)]
    marker_coords = _components_of_color(grid, GREY)

    assert len(marker_coords) == 2 * len(segments), "each segment must have one grey marker at each endpoint"

    endpoint_refs = []
    for seg in segments:
        for ep_idx, endpoint in enumerate(seg["endpoints"]):
            endpoint_refs.append((int(seg["id"]), int(ep_idx), tuple(endpoint["coord"])))

    assignments = set()
    for coords, shape in zip(marker_coords, markers):
        matches = [(sid, ep_idx) for sid, ep_idx, endpoint in endpoint_refs if _touches_endpoint(coords, endpoint)]
        assert len(matches) == 1, f"grey marker must touch exactly one endpoint, got {matches} for {sorted(coords)}"
        sid, ep_idx = matches[0]
        assert (sid, ep_idx) not in assignments
        assignments.add((sid, ep_idx))
        segments[sid]["endpoints"][ep_idx]["shape"] = shape

    for seg in segments:
        for endpoint in seg["endpoints"]:
            assert endpoint["shape"] is not None, "every segment endpoint must receive a marker"
        assert seg["endpoints"][0]["shape"] != seg["endpoints"][1]["shape"], "segment endpoints must have different markers"

    return segments


def _ordered_chains(segments: list[dict[str, object]]) -> list[list[tuple[int, int]]]:
    endpoints: dict[tuple[int, int], dict[str, object]] = {}
    shape_to_refs: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for seg in segments:
        sid = int(seg["id"])
        for ep_idx, endpoint in enumerate(seg["endpoints"]):
            ref = (sid, int(ep_idx))
            endpoints[ref] = {"shape": endpoint["shape"], "coord": tuple(endpoint["coord"])}
            shape_to_refs[str(endpoint["shape"])].append(ref)

    paired: dict[tuple[int, int], tuple[int, int]] = {}
    terminals: list[tuple[int, int]] = []
    for refs in shape_to_refs.values():
        if len(refs) == 1:
            terminals.append(refs[0])
        elif len(refs) == 2:
            a, b = refs
            paired[a] = b
            paired[b] = a
        else:
            raise AssertionError("marker shapes must appear at most twice")

    def ref_key(ref: tuple[int, int]) -> tuple[int, int, int, int, int]:
        info = endpoints[ref]
        r, c = info["coord"]
        return (int(SHAPE_RANK[str(info["shape"])]), int(r), int(c), int(ref[0]), int(ref[1]))

    unvisited = {int(seg["id"]) for seg in segments}
    chains: list[list[tuple[int, int]]] = []
    preferred_starts = terminals if terminals else list(endpoints)

    while unvisited:
        candidates = [ref for ref in preferred_starts if int(ref[0]) in unvisited]
        if not candidates:
            candidates = [ref for ref in endpoints if int(ref[0]) in unvisited]
        current = min(candidates, key=ref_key)
        chain: list[tuple[int, int]] = []
        while True:
            sid, in_idx = int(current[0]), int(current[1])
            if sid not in unvisited:
                break
            chain.append((sid, in_idx))
            unvisited.remove(sid)
            out_ref = (sid, 1 - in_idx)
            next_ref = paired.get(out_ref)
            if next_ref is None or int(next_ref[0]) not in unvisited:
                break
            current = next_ref
        chains.append(chain)

    chains.sort(key=lambda chain: ref_key(chain[0]))
    return chains


def _canonical_l_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r1, c1 = start
    r2, c2 = end
    path = [(int(r1), int(c1))]
    if c1 != c2:
        step = 1 if c2 > c1 else -1
        for cc in range(int(c1 + step), int(c2 + step), int(step)):
            path.append((int(r1), int(cc)))
    if r1 != r2:
        step = 1 if r2 > r1 else -1
        for rr in range(int(r1 + step), int(r2 + step), int(step)):
            cell = (int(rr), int(c2))
            if cell != path[-1]:
                path.append(cell)
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect equal endpoint markers into a relay chain and recolor each relay leg cyclically.

    The implementation treats the chain direction as starting from the endpoint whose
    unmatched terminal marker has the smallest fixed shape rank:
    dot < horizontal-2 < vertical-2 < square.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    segments = _extract_segments_and_markers(g)
    chains = _ordered_chains(segments)

    out = np.full(g.shape, BG, dtype=int)
    color_index = 0
    for chain in chains:
        for idx, (sid, in_idx) in enumerate(chain):
            color = int(OUTPUT_COLORS[color_index % len(OUTPUT_COLORS)])
            segment = segments[sid]
            for r, c in segment["coords"]:
                out[int(r), int(c)] = color
            if idx + 1 < len(chain):
                next_sid, next_in_idx = chain[idx + 1]
                start = tuple(segment["endpoints"][1 - in_idx]["coord"])
                end = tuple(segments[next_sid]["endpoints"][next_in_idx]["coord"])
                for r, c in _canonical_l_path(start, end)[1:-1]:
                    out[int(r), int(c)] = color
            color_index += 1
    return out


def _line_cells(start: tuple[int, int], step: tuple[int, int], length: int) -> list[tuple[int, int]]:
    r0, c0 = start
    dr, dc = step
    return [(int(r0 + i * dr), int(c0 + i * dc)) for i in range(int(length))]


def _in_bounds(cells: list[tuple[int, int]] | set[tuple[int, int]], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _make_segment(seg_id: int, color: int, start: tuple[int, int], step: tuple[int, int], length: int) -> dict[str, object]:
    cells = _line_cells(start, step, length)
    return {
        "id": int(seg_id),
        "color": int(color),
        "start": tuple(start),
        "end": tuple(cells[-1]),
        "step": tuple(step),
        "length": int(length),
        "coords": cells,
        "cells": set(cells),
        "endpoints": [
            {"coord": tuple(start)},
            {"coord": tuple(cells[-1])},
        ],
    }


def _marker_variants(shape: str, outward: tuple[int, int]) -> list[list[tuple[int, int]]]:
    dr, dc = outward
    if shape == "dot":
        return [[(int(dr), int(dc))]]
    if shape == "hline2":
        if dr == 0:
            return [[(0, int(dc)), (0, int(2 * dc))]]
        return [
            [(int(dr), 0), (int(dr), 1)],
            [(int(dr), -1), (int(dr), 0)],
        ]
    if shape == "vline2":
        if dc == 0:
            return [[(int(dr), 0), (int(2 * dr), 0)]]
        return [
            [(0, int(dc)), (1, int(dc))],
            [(-1, int(dc)), (0, int(dc))],
        ]
    assert shape == "square2"
    if dr == 0 and dc == 1:
        return [
            [(0, 1), (0, 2), (1, 1), (1, 2)],
            [(-1, 1), (-1, 2), (0, 1), (0, 2)],
        ]
    if dr == 0 and dc == -1:
        return [
            [(0, -1), (0, -2), (1, -1), (1, -2)],
            [(-1, -1), (-1, -2), (0, -1), (0, -2)],
        ]
    if dr == 1 and dc == 0:
        return [
            [(1, 0), (2, 0), (1, 1), (2, 1)],
            [(1, -1), (2, -1), (1, 0), (2, 0)],
        ]
    return [
        [(-1, 0), (-2, 0), (-1, 1), (-2, 1)],
        [(-1, -1), (-2, -1), (-1, 0), (-2, 0)],
    ]


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(int(r + dr), int(c + dc)) for dr, dc in DIR4]


def _min_endpoint_distance(endpoint: tuple[int, int], cells: set[tuple[int, int]]) -> int:
    er, ec = endpoint
    return min(abs(er - r) + abs(ec - c) for r, c in cells)


def _place_markers(
    h: int,
    w: int,
    segments: list[dict[str, object]],
    endpoint_shapes: dict[tuple[int, int], str],
    connection_map: dict[tuple[int, int], set[tuple[int, int]]],
) -> dict[tuple[int, int], set[tuple[int, int]]] | None:
    all_segment_cells = set().union(*(seg["cells"] for seg in segments))
    endpoint_coords = {ref: tuple(segments[ref[0]]["endpoints"][ref[1]]["coord"]) for ref in endpoint_shapes}
    own_segment_cells = {(seg["id"], idx): set(seg["cells"]) for seg in segments for idx in (0, 1)}
    all_connection_cells = set()
    for cells in connection_map.values():
        all_connection_cells.update(cells)

    refs = list(endpoint_shapes)
    refs.sort(key=lambda ref: (-len(_marker_variants(endpoint_shapes[ref], _endpoint_outward(segments[ref[0]], ref[1]))[0]), -_shape_size(endpoint_shapes[ref]), ref))

    candidates: dict[tuple[int, int], list[set[tuple[int, int]]]] = {}
    for ref in refs:
        endpoint = endpoint_coords[ref]
        shape = endpoint_shapes[ref]
        outward = _endpoint_outward(segments[ref[0]], ref[1])
        incident = connection_map.get(ref, set())
        forbidden_connections = all_connection_cells - incident
        own_cells = own_segment_cells[(ref[0], ref[1])]
        other_segment_cells = all_segment_cells - own_cells

        variants = _marker_variants(shape, outward)
        random.shuffle(variants)
        good: list[set[tuple[int, int]]] = []
        for rels in variants:
            cells = {(int(endpoint[0] + dr), int(endpoint[1] + dc)) for dr, dc in rels}
            if not _in_bounds(cells, h, w):
                continue
            if cells & all_segment_cells:
                continue
            if cells & forbidden_connections:
                continue
            if any(_min_endpoint_distance(coord, cells) == 1 and other_ref != ref for other_ref, coord in endpoint_coords.items()):
                continue
            bad_touch = False
            for cell in cells:
                for neighbor in _neighbors4(cell):
                    if neighbor in other_segment_cells:
                        bad_touch = True
                        break
                if bad_touch:
                    break
            if bad_touch:
                continue
            good.append(cells)
        if not good:
            return None
        candidates[ref] = good

    placed: dict[tuple[int, int], set[tuple[int, int]]] = {}
    occupied: set[tuple[int, int]] = set()

    def search(idx: int) -> bool:
        if idx == len(refs):
            return True
        ref = refs[idx]
        options = list(candidates[ref])
        random.shuffle(options)
        for cells in options:
            if cells & occupied:
                continue
            if any(neighbor in occupied for cell in cells for neighbor in _neighbors4(cell)):
                continue
            placed[ref] = cells
            occupied.update(cells)
            if search(idx + 1):
                return True
            occupied.difference_update(cells)
            del placed[ref]
        return False

    return placed if search(0) else None


def _shape_size(shape: str) -> int:
    if shape == "dot":
        return 1
    if shape in {"hline2", "vline2"}:
        return 2
    return 4


def _endpoint_outward(segment: dict[str, object], endpoint_idx: int) -> tuple[int, int]:
    dr, dc = segment["step"]
    if endpoint_idx == 0:
        return (-int(dr), -int(dc))
    return (int(dr), int(dc))


def _construct_attempt() -> np.ndarray | None:
    n_segments = random.choice((2, 3))
    h = random.randint(18 + 2 * (n_segments - 2), 26)
    w = random.randint(18 + 2 * (n_segments - 2), 26)
    colors = random.sample(list(INPUT_COLOR_POOL), n_segments)
    lengths = random.sample(list(range(3, 9)), n_segments)
    shape_chain = random.sample(list(SHAPE_ORDER), n_segments + 1)

    segments: list[dict[str, object]] = []
    connections: list[set[tuple[int, int]]] = []
    reserved: set[tuple[int, int]] = set()

    for seg_id in range(n_segments):
        placed = False
        for _ in range(250):
            step = random.choice(CARDINALS)
            if seg_id == 0:
                start = (random.randint(2, h - 3), random.randint(2, w - 3))
            else:
                prev_end = segments[-1]["end"]
                dr = random.choice(tuple(range(-8, -1)) + tuple(range(2, 9)))
                dc = random.choice(tuple(range(-8, -1)) + tuple(range(2, 9)))
                start = (int(prev_end[0] + dr), int(prev_end[1] + dc))
                if start[0] == prev_end[0] or start[1] == prev_end[1]:
                    continue
            segment = _make_segment(seg_id, colors[seg_id], start, step, lengths[seg_id])
            if not _in_bounds(segment["coords"], h, w):
                continue
            if segment["cells"] & reserved:
                continue
            if seg_id > 0:
                path = _canonical_l_path(tuple(segments[-1]["end"]), tuple(segment["start"]))
                interior = set(path[1:-1])
                if not interior:
                    continue
                if interior & reserved:
                    continue
                if interior & segment["cells"]:
                    continue
            else:
                interior = set()
            segments.append(segment)
            reserved.update(segment["cells"])
            reserved.update(interior)
            if seg_id > 0:
                connections.append(interior)
            placed = True
            break
        if not placed:
            return None

    endpoint_shapes: dict[tuple[int, int], str] = {}
    for seg_id in range(n_segments):
        endpoint_shapes[(seg_id, 0)] = shape_chain[seg_id]
        endpoint_shapes[(seg_id, 1)] = shape_chain[seg_id + 1]

    connection_map: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for idx, cells in enumerate(connections):
        connection_map[(idx, 1)] = set(cells)
        connection_map[(idx + 1, 0)] = set(cells)

    markers = _place_markers(h, w, segments, endpoint_shapes, connection_map)
    if markers is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for segment in segments:
        for r, c in segment["coords"]:
            grid[int(r), int(c)] = int(segment["color"])
    for cells in markers.values():
        for r, c in cells:
            grid[int(r), int(c)] = GREY

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _construct_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid relay-chain puzzle")
