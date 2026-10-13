from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BLACK = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
MAX_GENERATION_ATTEMPTS = 400
PATH_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 160


@dataclass(frozen=True)
class Segment:
    cells: tuple[tuple[int, int], ...]
    orientation: str
    endpoints: tuple[tuple[int, int], tuple[int, int]]
    outward_dirs: tuple[tuple[int, int], tuple[int, int]]
    key: tuple[int, int]


@dataclass(frozen=True)
class Connection:
    a: int
    b: int
    ea: int
    eb: int
    gap: tuple[int, int]


@dataclass(frozen=True)
class PathSpec:
    color: int
    cells: tuple[tuple[int, int], ...]
    height: int
    width: int


def _in_bounds(shape: tuple[int, int], cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue

        stack = [(sr, sc)]
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in DIR4:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        comps.append(comp)

    return comps


def _build_segment(coords: list[tuple[int, int]]) -> Segment:
    rows = sorted({r for r, _ in coords})
    cols = sorted({c for _, c in coords})

    if len(rows) == 1:
        row = rows[0]
        span = list(range(cols[0], cols[-1] + 1))
        assert cols == span, "horizontal segment must be contiguous"
        cells = tuple((row, c) for c in span)
        assert 2 <= len(cells) <= 5
        return Segment(
            cells=cells,
            orientation="H",
            endpoints=(cells[0], cells[-1]),
            outward_dirs=((0, -1), (0, 1)),
            key=cells[0],
        )

    if len(cols) == 1:
        col = cols[0]
        span = list(range(rows[0], rows[-1] + 1))
        assert rows == span, "vertical segment must be contiguous"
        cells = tuple((r, col) for r in span)
        assert 2 <= len(cells) <= 5
        return Segment(
            cells=cells,
            orientation="V",
            endpoints=(cells[0], cells[-1]),
            outward_dirs=((-1, 0), (1, 0)),
            key=cells[0],
        )

    raise AssertionError("each component must be a straight segment")


def _extract_segments(grid: np.ndarray, color: int) -> list[Segment]:
    segments = [_build_segment(comp) for comp in _components_of_color(grid, color)]
    segments.sort(key=lambda seg: seg.key)
    return segments


def _maybe_connection(grid: np.ndarray, seg_a: Segment, seg_b: Segment, a_idx: int, b_idx: int) -> Connection | None:
    matches: list[Connection] = []
    for ea in range(2):
        ar, ac = seg_a.endpoints[ea]
        adr, adc = seg_a.outward_dirs[ea]
        gap_a = (ar + adr, ac + adc)
        if not _in_bounds(grid.shape, gap_a) or int(grid[gap_a]) != BLACK:
            continue

        for eb in range(2):
            br, bc = seg_b.endpoints[eb]
            bdr, bdc = seg_b.outward_dirs[eb]
            gap_b = (br + bdr, bc + bdc)
            if gap_a == gap_b and _in_bounds(grid.shape, gap_b) and int(grid[gap_b]) == BLACK:
                matches.append(Connection(a_idx, b_idx, ea, eb, gap_a))

    assert len(matches) <= 1, "segment pair must have at most one valid gap connection"
    return matches[0] if matches else None


def _other(edge: Connection, seg_idx: int) -> int:
    return edge.b if seg_idx == edge.a else edge.a


def _trace_color(grid: np.ndarray, color: int) -> dict[str, object]:
    segments = _extract_segments(grid, color)
    count = len(segments)
    assert 2 <= count <= 4, "each color must contain 2-4 segments"

    edges: list[Connection] = []
    adjacency: list[list[int]] = [[] for _ in segments]
    endpoint_use: dict[tuple[int, int], int] = {}

    for i in range(count):
        for j in range(i + 1, count):
            edge = _maybe_connection(grid, segments[i], segments[j], i, j)
            if edge is None:
                continue
            edge_idx = len(edges)
            edges.append(edge)
            adjacency[i].append(edge_idx)
            adjacency[j].append(edge_idx)
            endpoint_use[(i, edge.ea)] = endpoint_use.get((i, edge.ea), 0) + 1
            endpoint_use[(j, edge.eb)] = endpoint_use.get((j, edge.eb), 0) + 1

    assert all(v == 1 for v in endpoint_use.values()), "each segment endpoint can join at most one gap"
    degrees = [len(items) for items in adjacency]
    assert sum(degrees) == 2 * (count - 1), "path must have exactly count-1 connections"

    endpoint_segments = [idx for idx, degree in enumerate(degrees) if degree == 1]
    assert len(endpoint_segments) == 2, "segment graph must be a simple chain"

    start = min(range(count), key=lambda idx: segments[idx].key)
    assert degrees[start] == 1, "top-left-most segment must be a path endpoint"

    order: list[int] = []
    edge_order: list[int] = []
    prev = -1
    cur = start
    while True:
        order.append(cur)
        next_edges = [eid for eid in adjacency[cur] if _other(edges[eid], cur) != prev]
        if not next_edges:
            break
        assert len(next_edges) == 1, "path traversal must be unambiguous"
        eid = next_edges[0]
        edge_order.append(eid)
        prev, cur = cur, _other(edges[eid], cur)

    assert len(order) == count, "segment graph must be connected"

    gap_cells = [edges[eid].gap for eid in edge_order]
    end_seg_idx = order[-1]
    end_edge = edges[adjacency[end_seg_idx][0]]
    connected_endpoint = end_edge.ea if end_edge.a == end_seg_idx else end_edge.eb
    outer_endpoint = 1 - connected_endpoint
    ray_origin = segments[end_seg_idx].endpoints[outer_endpoint]
    ray_dir = segments[end_seg_idx].outward_dirs[outer_endpoint]

    ray_cells: list[tuple[int, int]] = []
    for step in range(1, count + 1):
        rr = ray_origin[0] + ray_dir[0] * step
        cc = ray_origin[1] + ray_dir[1] * step
        if _in_bounds(grid.shape, (rr, cc)):
            ray_cells.append((rr, cc))

    return {
        "segments": segments,
        "gap_cells": gap_cells,
        "ray_cells": ray_cells,
    }


def _output_cells_for_single_path(spec: PathSpec, top: int, left: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    grid = np.zeros(shape, dtype=int)
    cells = {(top + r, left + c) for r, c in spec.cells}
    for r, c in cells:
        grid[r, c] = spec.color

    info = _trace_color(grid, spec.color)
    out_cells = set(cells)
    out_cells.update(info["gap_cells"])
    out_cells.update(info["ray_cells"])
    return out_cells


def _expand8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIR8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
                out.add((rr, cc))
        out.add((r, c))
    return out


def _instance_is_valid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    colors = [int(v) for v in np.unique(grid) if int(v) != BLACK]
    if not colors:
        return False

    h, w = grid.shape
    for r, c in np.argwhere(grid != BLACK):
        r = int(r)
        c = int(c)
        color = int(grid[r, c])
        for dr, dc in DIR8:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w:
                other = int(grid[rr, cc])
                if other != BLACK and other != color:
                    return False

    for color in colors:
        try:
            info = _trace_color(grid, color)
        except AssertionError:
            return False
        if len(info["segments"]) < 2:
            return False

    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == grid.shape


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each color's broken line segments into one path, then extend a short ray from the path's far end.

    Input structure:
    - The background is black.
    - Each non-black color appears as 2-4 separate horizontal or vertical line segments, each 2-5 cells long.
    - Segments of the same color belong to one broken path: consecutive segments are separated by exactly one black gap,
      either straight along one row/column or around a diagonal corner between a horizontal and a vertical endpoint.
    - Different colors do not overlap or touch.

    Rule:
    - Group cells by color and split them into straight segments.
    - Reconstruct the unique segment chain by matching segment endpoints whose one-step outward gap cell is the same black cell.
    - Start from the top-left-most endpoint segment, fill every matched one-cell gap with that color, and keep the original segments.
    - Let N be the number of segments of that color; from the outer endpoint of the last segment, continue in that segment's forward direction for N cells, clipping at the grid edge.

    Key insight:
    - The hidden path order is encoded by which segment endpoints share the same one-cell outward gap, and that same endpoint geometry also determines the ray direction.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    out = grid.copy()
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BLACK):
        info = _trace_color(grid, color)
        for r, c in info["gap_cells"]:
            out[r, c] = color
        for r, c in info["ray_cells"]:
            out[r, c] = color
    return out


def _abstract_segments() -> list[list[tuple[int, int]]]:
    segment_count = random.randint(2, 4)
    orientation = random.choice(("H", "V"))
    row = 0
    col = 0
    segments: list[list[tuple[int, int]]] = []

    for idx in range(segment_count):
        length = random.randint(2, 5)
        if orientation == "H":
            segment = [(row, c) for c in range(col, col + length)]
        else:
            segment = [(r, col) for r in range(row, row + length)]
        segments.append(segment)

        if idx == segment_count - 1:
            break

        make_corner = random.random() < 0.4
        if orientation == "H":
            if make_corner:
                row += 1
                col += length
                orientation = "V"
            else:
                col += length + 1
        else:
            if make_corner:
                row += length
                col += 1
                orientation = "H"
            else:
                row += length + 1

    return segments


def _transform_segments(segments: list[list[tuple[int, int]]]) -> tuple[tuple[tuple[int, int], ...], int, int]:
    all_cells = [cell for seg in segments for cell in seg]
    height = max(r for r, _ in all_cells) + 1
    width = max(c for _, c in all_cells) + 1

    transpose = random.choice((False, True))
    if transpose:
        height, width = width, height

    flip_r = random.choice((False, True))
    flip_c = random.choice((False, True))
    transformed: list[tuple[int, int]] = []
    for r, c in all_cells:
        if transpose:
            r, c = c, r
        if flip_r:
            r = height - 1 - r
        if flip_c:
            c = width - 1 - c
        transformed.append((r, c))

    min_r = min(r for r, _ in transformed)
    min_c = min(c for _, c in transformed)
    normalized = tuple(sorted((r - min_r, c - min_c) for r, c in transformed))
    final_h = max(r for r, _ in normalized) + 1
    final_w = max(c for _, c in normalized) + 1
    return normalized, final_h, final_w


def _generate_path_spec(color: int) -> PathSpec | None:
    for _ in range(PATH_ATTEMPTS):
        cells, height, width = _transform_segments(_abstract_segments())
        local = np.zeros((height, width), dtype=int)
        for r, c in cells:
            local[r, c] = color
        if _instance_is_valid(local):
            return PathSpec(color=color, cells=cells, height=height, width=width)
    return None


def _place_paths(paths: list[PathSpec], height: int, width: int) -> np.ndarray | None:
    grid = np.zeros((height, width), dtype=int)
    reserved: set[tuple[int, int]] = set()

    ordered = sorted(paths, key=lambda spec: (spec.height * spec.width, len(spec.cells)), reverse=True)
    for spec in ordered:
        candidates = [(r0, c0) for r0 in range(height - spec.height + 1) for c0 in range(width - spec.width + 1)]
        random.shuffle(candidates)
        candidates = candidates[:PLACEMENT_ATTEMPTS]

        placed = False
        for top, left in candidates:
            input_cells = {(top + r, left + c) for r, c in spec.cells}
            if any(grid[r, c] != BLACK for r, c in input_cells):
                continue
            if input_cells & reserved:
                continue

            out_cells = _output_cells_for_single_path(spec, top, left, (height, width))
            expanded = _expand8(out_cells, (height, width))
            if expanded & reserved:
                continue

            for r, c in input_cells:
                grid[r, c] = spec.color
            reserved.update(expanded)
            placed = True
            break

        if not placed:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        path_count = random.choice((1, 2, 3))
        colors = random.sample(range(1, 10), path_count)

        paths: list[PathSpec] = []
        for color in colors:
            spec = _generate_path_spec(color)
            if spec is None:
                break
            paths.append(spec)
        if len(paths) != path_count:
            continue

        max_h = max(spec.height for spec in paths)
        max_w = max(spec.width for spec in paths)
        min_side = {1: 10, 2: 16, 3: 20}[path_count]
        low_h = max(min_side, max_h + 1)
        low_w = max(min_side, max_w + 1)
        if low_h > 30 or low_w > 30:
            continue

        height = random.randint(low_h, 30)
        width = random.randint(low_w, 30)
        grid = _place_paths(paths, height, width)
        if grid is None:
            continue
        if _instance_is_valid(grid):
            return grid

    raise RuntimeError("failed to generate a valid ARC instance")
