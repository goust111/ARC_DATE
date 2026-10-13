import random
from typing import Dict, List, Sequence, Tuple

import numpy as np


Point = Tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _line_neighbors(mask: np.ndarray, r: int, c: int) -> List[Point]:
    h, w = mask.shape
    out: List[Point] = []
    if r > 0 and mask[r - 1, c]:
        out.append((r - 1, c))
    if r + 1 < h and mask[r + 1, c]:
        out.append((r + 1, c))
    if c > 0 and mask[r, c - 1]:
        out.append((r, c - 1))
    if c + 1 < w and mask[r, c + 1]:
        out.append((r, c + 1))
    return out


def _is_turn(point: Point, neighbors: Sequence[Point]) -> bool:
    (r, c) = point
    (r1, c1), (r2, c2) = neighbors
    return not (r1 - r == -(r2 - r) and c1 - c == -(c2 - c))


def _edge_key(a: Point, b: Point) -> Tuple[Point, Point]:
    return (a, b) if a <= b else (b, a)


def _extract_scene(grid: np.ndarray) -> Dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    bg = _get_bg(g)
    legend = {
        2: int(g[0, 0]),
        3: int(g[0, w - 1]),
        4: int(g[h - 1, 0]),
        5: int(g[h - 1, w - 1]),
    }
    assert len(set(legend.values())) == 4, "corner legend colors must be distinct"
    assert bg not in legend.values(), "background must differ from the corner legend"

    fg_colors = set(map(int, np.unique(g))) - {bg}
    line_colors = fg_colors - set(legend.values())
    assert len(line_colors) == 1, f"expected one path color, got {line_colors}"
    line_color = int(next(iter(line_colors)))

    line_mask = g == line_color
    line_mask[0, 0] = False
    line_mask[0, w - 1] = False
    line_mask[h - 1, 0] = False
    line_mask[h - 1, w - 1] = False
    assert bool(line_mask.any()), "path network must be non-empty"

    nodes: Dict[Point, int] = {}
    boundary_points: set[Point] = set()
    for r, c in map(tuple, np.argwhere(line_mask)):
        point = (int(r), int(c))
        neighbors = _line_neighbors(line_mask, point[0], point[1])
        degree = len(neighbors)
        if degree == 1:
            assert point[0] in (0, h - 1) or point[1] in (0, w - 1), "dangling endpoints must touch the border"
            boundary_points.add(point)
        elif degree == 2:
            if _is_turn(point, neighbors):
                nodes[point] = 2
        elif degree in (3, 4):
            nodes[point] = degree
        elif degree == 0:
            raise AssertionError("isolated path pixels are not allowed")
        else:
            nodes[point] = 5

    assert nodes, "network must contain at least one junction"
    stop_points = set(nodes) | boundary_points

    segments: List[List[Point]] = []
    visited_edges: set[Tuple[Point, Point]] = set()
    for start in sorted(stop_points):
        for nxt in _line_neighbors(line_mask, start[0], start[1]):
            first_edge = _edge_key(start, nxt)
            if first_edge in visited_edges:
                continue
            visited_edges.add(first_edge)
            path = [start, nxt]
            prev = start
            cur = nxt
            while cur not in stop_points:
                neighbors = _line_neighbors(line_mask, cur[0], cur[1])
                assert len(neighbors) == 2, "non-junction interior pixels must be straight"
                assert prev in neighbors, "segment tracing lost the current run"
                nxt = neighbors[0] if neighbors[1] == prev else neighbors[1]
                visited_edges.add(_edge_key(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            segments.append(path)

    painted = np.zeros_like(line_mask, dtype=bool)
    for path in segments:
        for r, c in path:
            painted[r, c] = True
    assert bool(np.all(painted[line_mask])), "every path pixel must belong to a traced segment"

    return {
        "grid": g,
        "bg": bg,
        "legend": legend,
        "line_color": line_color,
        "line_mask": line_mask,
        "nodes": nodes,
        "boundary_points": boundary_points,
        "segments": segments,
    }


def _paint_segment(out: np.ndarray, path: Sequence[Point], color_a: int, color_b: int | None = None) -> None:
    if color_b is None or color_a == color_b:
        for r, c in path:
            out[r, c] = int(color_a)
        return

    n = len(path)
    for i, (r, c) in enumerate(path):
        out[r, c] = int(color_a if 2 * i <= n - 1 else color_b)


def main(grid: np.ndarray) -> np.ndarray:
    """Color each junction by legend-coded degree, then recolor every straight run from its endpoint junction colors.

    Input structure:
    - The background is the dominant color.
    - One other color forms a one-cell-thick orthogonal path network made of horizontal and vertical runs.
    - Path runs meet at turns, T-junctions, and cross-junctions; run endpoints otherwise lie on the grid border.
    - The four corner cells are single-pixel legend markers whose positions encode the colors for junction counts 2, 3, 4, and 5.

    Rule:
    - Read the corner legend in fixed order: top-left -> 2, top-right -> 3, bottom-left -> 4, bottom-right -> 5.
    - Detect every junction pixel where at least two straight runs meet: a turn counts as 2, a T-junction as 3, and a cross as 4.
    - Recolor each junction with the legend color for its count.
    - Split the network into maximal straight runs between junctions and/or border endpoints. A run between equal-colored junctions is painted uniformly; a run between different-colored junctions is split at the midpoint so each half takes the nearer junction's color; a run with only one junction endpoint takes that junction's color for its full length.

    Key insight:
    - The corner pixels are not decoration: they are an ordered lookup table for junction degree, and every grey-looking line must be mentally broken into straight runs whose colors are determined jointly by their endpoint junctions.
    """
    scene = _extract_scene(grid)
    g = scene["grid"]
    legend: Dict[int, int] = scene["legend"]  # type: ignore[assignment]
    line_mask: np.ndarray = scene["line_mask"]  # type: ignore[assignment]
    nodes: Dict[Point, int] = scene["nodes"]  # type: ignore[assignment]
    segments: List[List[Point]] = scene["segments"]  # type: ignore[assignment]

    out = g.copy()
    out[line_mask] = scene["bg"]  # type: ignore[index]

    for point, degree in nodes.items():
        out[point] = int(legend[min(int(degree), 5)])

    for path in segments:
        start = path[0]
        end = path[-1]
        start_is_node = start in nodes
        end_is_node = end in nodes

        if start_is_node and end_is_node:
            color_a = int(legend[min(int(nodes[start]), 5)])
            color_b = int(legend[min(int(nodes[end]), 5)])
            _paint_segment(out, path, color_a, color_b)
        elif start_is_node ^ end_is_node:
            node = start if start_is_node else end
            color = int(legend[min(int(nodes[node]), 5)])
            _paint_segment(out, path, color)
        else:
            raise AssertionError("every run must touch at least one junction")

    return out


# --------------- generator ---------------

GEN_RETRIES = 300
GRID_RANGE = (17, 29)
PALETTE = tuple(range(10))
CENTER_MARGIN = 6


def _draw_span(grid: np.ndarray, a: Point, b: Point, color: int) -> None:
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        grid[r0, lo : hi + 1] = color
        return
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        grid[lo : hi + 1, c0] = color
        return
    raise AssertionError("spans must be axis-aligned")


def _sample_vertical_branch(rng: random.Random, h: int, w: int, rh: int, cv: int, vertical_side: str, horizontal_side: str) -> Tuple[int, int, int]:
    if vertical_side == "up":
        attach_r = rng.randint(3, rh - 3)
        bend_r = rng.randint(1, attach_r - 2)
    else:
        attach_r = rng.randint(rh + 3, h - 4)
        bend_r = rng.randint(attach_r + 2, h - 2)

    if horizontal_side == "left":
        bend_c = rng.randint(3, cv - 3)
    else:
        bend_c = rng.randint(cv + 3, w - 4)

    return attach_r, bend_r, bend_c


def _sample_horizontal_branch(rng: random.Random, h: int, w: int, rh: int, cv: int, vertical_side: str, horizontal_side: str) -> Tuple[int, int, int]:
    if horizontal_side == "left":
        attach_c = rng.randint(3, cv - 3)
        bend_c = rng.randint(1, attach_c - 2)
    else:
        attach_c = rng.randint(cv + 3, w - 4)
        bend_c = rng.randint(attach_c + 2, w - 2)

    if vertical_side == "up":
        bend_r = rng.randint(3, rh - 3)
    else:
        bend_r = rng.randint(rh + 3, h - 4)

    return attach_c, bend_r, bend_c


def _segment_category_counts(scene: Dict[str, object]) -> Tuple[set[int], int, int, int]:
    nodes: Dict[Point, int] = scene["nodes"]  # type: ignore[assignment]
    segments: List[List[Point]] = scene["segments"]  # type: ignore[assignment]

    degrees = set(map(int, nodes.values()))
    same = 0
    different = 0
    border = 0
    for path in segments:
        start = path[0]
        end = path[-1]
        start_is_node = start in nodes
        end_is_node = end in nodes
        if start_is_node and end_is_node:
            if int(nodes[start]) == int(nodes[end]):
                same += 1
            else:
                different += 1
        elif start_is_node ^ end_is_node:
            border += 1
    return degrees, same, different, border


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_RANGE)
    w = rng.randint(*GRID_RANGE)
    bg, line_color, color2, color3, color4, color5 = rng.sample(list(PALETTE), 6)

    grid = np.full((h, w), bg, dtype=int)
    rh = rng.randint(CENTER_MARGIN, h - CENTER_MARGIN - 1)
    cv = rng.randint(CENTER_MARGIN, w - CENTER_MARGIN - 1)

    vertical_side = rng.choice(["up", "down"])
    horizontal_side = rng.choice(["left", "right"])
    opposite_vertical = "down" if vertical_side == "up" else "up"
    opposite_horizontal = "right" if horizontal_side == "left" else "left"

    try:
        attach_r, branch_bend_r, branch_bend_c = _sample_vertical_branch(
            rng, h, w, rh, cv, vertical_side, horizontal_side
        )
        attach_c, bend_r2, bend_c2 = _sample_horizontal_branch(
            rng, h, w, rh, cv, opposite_vertical, opposite_horizontal
        )
    except ValueError:
        return None

    _draw_span(grid, (rh, 0), (rh, w - 1), line_color)
    _draw_span(grid, (0, cv), (h - 1, cv), line_color)

    _draw_span(grid, (attach_r, cv), (attach_r, branch_bend_c), line_color)
    _draw_span(grid, (attach_r, branch_bend_c), (branch_bend_r, branch_bend_c), line_color)
    _draw_span(
        grid,
        (branch_bend_r, branch_bend_c),
        (branch_bend_r, 0 if horizontal_side == "left" else w - 1),
        line_color,
    )

    _draw_span(grid, (rh, attach_c), (bend_r2, attach_c), line_color)
    _draw_span(grid, (bend_r2, attach_c), (bend_r2, bend_c2), line_color)
    _draw_span(
        grid,
        (bend_r2, bend_c2),
        (0 if opposite_vertical == "up" else h - 1, bend_c2),
        line_color,
    )

    grid[0, 0] = color2
    grid[0, -1] = color3
    grid[-1, 0] = color4
    grid[-1, -1] = color5

    try:
        scene = _extract_scene(grid)
        degrees, same, different, border = _segment_category_counts(scene)
        if not {2, 3, 4}.issubset(degrees):
            return None
        if min(same, different, border) == 0:
            return None

        out = main(grid)
        if np.array_equal(out, grid):
            return None
        if int(np.count_nonzero(out == line_color)) != 0:
            return None
        network_colors = set(map(int, np.unique(out[scene["line_mask"]])))  # type: ignore[index]
        if not {color2, color3, color4}.issubset(network_colors):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
