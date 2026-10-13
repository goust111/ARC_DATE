from __future__ import annotations

import math
import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_key_combo_12_050_10",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
draw_line = _ARC_UTILS.draw_line
in_bounds = _ARC_UTILS.in_bounds


BG = Color.BLACK
VERTEX = Color.BLUE
SHADOW = Color.RED
GEN_RETRIES = 400


def _sorted_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(vertices, dtype=float)
    center = arr.mean(axis=0)
    return sorted(
        [(int(r), int(c)) for r, c in vertices],
        key=lambda rc: math.atan2(rc[0] - center[0], rc[1] - center[1]),
    )


def _step_between(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    if dr == 0:
        sdr = 0
    else:
        sdr = dr // abs(dr)
    if dc == 0:
        sdc = 0
    else:
        sdc = dc // abs(dc)
    if not (dr == 0 or dc == 0 or abs(dr) == abs(dc)):
        raise ValueError(f"edge {a}->{b} is not H/V/45-degree aligned")
    return sdr, sdc


def _line_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = _step_between(a, b)
    length = max(abs(int(b[0] - a[0])), abs(int(b[1] - a[1])))
    return [(int(a[0] + i * dr), int(a[1] + i * dc)) for i in range(length + 1)]


def _edge_outward_offset(
    start: tuple[int, int],
    end: tuple[int, int],
    centroid: tuple[float, float],
) -> tuple[int, int]:
    dr, dc = _step_between(start, end)
    cand = (-dc, dr)
    mid_r = (start[0] + end[0]) / 2.0
    mid_c = (start[1] + end[1]) / 2.0
    to_center = (centroid[0] - mid_r, centroid[1] - mid_c)
    dot = cand[0] * to_center[0] + cand[1] * to_center[1]
    if dot < 0:
        return int(cand[0]), int(cand[1])
    return int(-cand[0]), int(-cand[1])


def _build_edges(vertices: list[tuple[int, int]]) -> list[dict]:
    cycle = _sorted_vertices(vertices)
    centroid = tuple(np.mean(np.array(cycle, dtype=float), axis=0))
    edges: list[dict] = []
    for idx, start in enumerate(cycle):
        end = cycle[(idx + 1) % len(cycle)]
        points = _line_points(start, end)
        edges.append(
            {
                "start": start,
                "end": end,
                "points": points,
                "length": len(points) - 1,
                "offset": _edge_outward_offset(start, end, centroid),
            }
        )
    return edges


def _shadow_targets_for_edge(
    edge: dict,
    period: int,
    shape: tuple[int, int],
) -> list[tuple[int, tuple[int, int]]]:
    out: list[tuple[int, tuple[int, int]]] = []
    off_r, off_c = edge["offset"]
    for arc in range(int(period), int(edge["length"]), int(period)):
        base_r, base_c = edge["points"][arc]
        shadow_pos = (int(base_r + off_r), int(base_c + off_c))
        if in_bounds(shape, shadow_pos[0], shadow_pos[1]):
            out.append((int(arc), shadow_pos))
    return out


def _candidate_period_sets(
    edges: list[dict],
    shape: tuple[int, int],
) -> dict[int, set[tuple[int, int]]]:
    out: dict[int, set[tuple[int, int]]] = {}
    for period in (2, 3):
        cells: set[tuple[int, int]] = set()
        for edge in edges:
            cells.update(pos for _arc, pos in _shadow_targets_for_edge(edge, period, shape))
        out[period] = cells
    return out


def _infer_period(
    red_pixels: set[tuple[int, int]],
    edges: list[dict],
    shape: tuple[int, int],
) -> int:
    allowed = _candidate_period_sets(edges, shape)
    candidates = [period for period in (2, 3) if red_pixels.issubset(allowed[period])]
    if not candidates:
        raise AssertionError("no valid shadow period found")
    if len(candidates) == 1:
        return candidates[0]

    exclusive_scores: dict[int, int] = {}
    for period in candidates:
        other = 5 - period
        exclusive_scores[period] = len(red_pixels & (allowed[period] - allowed[other]))
    best = max(candidates, key=lambda period: (exclusive_scores[period], -period))
    return int(best)


def _paint_edges(grid: np.ndarray, edges: list[dict]) -> None:
    for edge in edges:
        start = edge["start"]
        end = edge["end"]
        draw_line(grid, start[0], start[1], end_r=end[0], end_c=end[1], color=VERTEX)


def _render_shadow_subset(
    shape: tuple[int, int],
    vertices: list[tuple[int, int]],
    edge_to_cells: dict[int, set[tuple[int, int]]],
    *,
    draw_full_outline: bool,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in vertices:
        grid[int(r), int(c)] = VERTEX
    if draw_full_outline:
        _paint_edges(grid, _build_edges(vertices))
    for cells in edge_to_cells.values():
        for r, c in cells:
            if int(grid[r, c]) == BG:
                grid[r, c] = SHADOW
    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the blue outline and complete the periodic outward shadow along every edge.

    Input structure:
    - The grid is black except for 4-6 blue pixels that mark polygon vertices and some red shadow pixels.
    - The blue vertices form a simple octilinear polygon when sorted by polar angle around their centroid.
    - Red pixels lie one outward offset away from some edge interiors, never on the vertices themselves.

    Rule:
    - Sort the blue vertices by polar angle and connect consecutive vertices cyclically with horizontal, vertical, or 45-degree blue segments.
    - For each edge, measure arc-length in pixel steps from its start vertex and identify the outward parallel offset direction.
    - Infer the shared period K in {2,3} from the existing red pixels, then place red pixels at every non-vertex multiple of K along every edge's outward offset, clipping out-of-bounds shadows.

    Key insight:
    - The periodic pattern lives along the reconstructed outline path rather than along rows or columns, so the outline must be rebuilt before the shadow cadence can be completed.
    """
    g = np.asarray(grid, dtype=int)
    vertices = [tuple(map(int, rc)) for rc in np.argwhere(g == VERTEX)]
    if len(vertices) < 3:
        return g.copy()

    edges = _build_edges(vertices)
    red_pixels = {tuple(map(int, rc)) for rc in np.argwhere(g == SHADOW)}
    period = _infer_period(red_pixels, edges, g.shape)

    out = np.full_like(g, BG)
    _paint_edges(out, edges)
    for edge in edges:
        for _arc, (r, c) in _shadow_targets_for_edge(edge, period, g.shape):
            if int(out[r, c]) == BG:
                out[r, c] = SHADOW
    return out


def _normalize_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in vertices)
    min_c = min(c for _, c in vertices)
    return [(int(r - min_r), int(c - min_c)) for r, c in vertices]


def _transform_vertices(
    vertices: list[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]]:
    pts = list(vertices)
    pts = _normalize_vertices(pts)
    for _ in range(rng.randrange(4)):
        h = max(r for r, _ in pts) + 1
        pts = [(int(c), int(h - 1 - r)) for r, c in pts]
        pts = _normalize_vertices(pts)
    if rng.random() < 0.5:
        w = max(c for _, c in pts) + 1
        pts = [(int(r), int(w - 1 - c)) for r, c in pts]
        pts = _normalize_vertices(pts)
    return _normalize_vertices(pts)


def _template_rectangle(rng: random.Random) -> list[tuple[int, int]]:
    h = rng.randint(4, 8)
    w = rng.randint(4, 8)
    return [(0, 0), (0, w), (h, w), (h, 0)]


def _template_diamond(rng: random.Random) -> list[tuple[int, int]]:
    arm = rng.randint(3, 5)
    return [(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)]


def _template_house(rng: random.Random) -> list[tuple[int, int]]:
    roof = rng.randint(3, 5)
    wall = rng.randint(3, 6)
    return [(roof, 0), (0, roof), (roof, 2 * roof), (roof + wall, 2 * roof), (roof + wall, 0)]


def _template_hexagon(rng: random.Random) -> list[tuple[int, int]]:
    scale = rng.randint(2, 4)
    return [(0, scale), (0, 2 * scale), (scale, 3 * scale), (2 * scale, 2 * scale), (2 * scale, scale), (scale, 0)]


def _template_cut_corner(rng: random.Random) -> list[tuple[int, int]]:
    cut = rng.randint(2, 4)
    h = rng.randint(4, 7)
    w = rng.randint(5, 8)
    return [(0, cut), (0, w), (h, w), (h, 0), (cut, 0)]


TEMPLATES = (
    _template_rectangle,
    _template_diamond,
    _template_house,
    _template_hexagon,
    _template_cut_corner,
)


def _place_vertices(
    local_vertices: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[list[tuple[int, int]], tuple[int, int], bool]:
    pts = _transform_vertices(local_vertices, rng)
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    touch_border = rng.random() < 0.3

    min_h = max(15, max_r + 2 + (0 if touch_border else 1))
    min_w = max(15, max_c + 2 + (0 if touch_border else 1))
    if min_h > 22 or min_w > 22:
        raise ValueError("template too large")

    h = rng.randint(min_h, 22)
    w = rng.randint(min_w, 22)

    if touch_border:
        sides = []
        if h >= max_r + 1:
            sides.extend(["top", "bottom"])
        if w >= max_c + 1:
            sides.extend(["left", "right"])
        side = rng.choice(sides)
        if side == "top":
            r_shift = 0
            c_shift = rng.randint(0, w - max_c - 1)
        elif side == "bottom":
            r_shift = h - max_r - 1
            c_shift = rng.randint(0, w - max_c - 1)
        elif side == "left":
            r_shift = rng.randint(0, h - max_r - 1)
            c_shift = 0
        else:
            r_shift = rng.randint(0, h - max_r - 1)
            c_shift = w - max_c - 1
    else:
        r_shift = rng.randint(1, h - max_r - 2)
        c_shift = rng.randint(1, w - max_c - 2)

    placed = [(int(r + r_shift), int(c + c_shift)) for r, c in pts]
    return placed, (h, w), touch_border


def _edge_shadow_lists(
    edges: list[dict],
    period: int,
    shape: tuple[int, int],
) -> list[list[tuple[int, tuple[int, int]]]]:
    return [_shadow_targets_for_edge(edge, period, shape) for edge in edges]


def _validate_full_output(
    vertices: list[tuple[int, int]],
    edges: list[dict],
    shadow_lists: list[list[tuple[int, tuple[int, int]]]],
) -> bool:
    blue_cells = {cell for edge in edges for cell in edge["points"]}
    red_cells: set[tuple[int, int]] = set()
    total = 0
    for cells in shadow_lists:
        for _arc, pos in cells:
            total += 1
            if pos in blue_cells:
                return False
            red_cells.add(pos)
    return len(red_cells) == total and len(vertices) == len(set(vertices))


def _period_candidates_from_input(
    grid: np.ndarray,
) -> list[int]:
    vertices = [tuple(map(int, rc)) for rc in np.argwhere(grid == VERTEX)]
    edges = _build_edges(vertices)
    red_pixels = {tuple(map(int, rc)) for rc in np.argwhere(grid == SHADOW)}
    allowed = _candidate_period_sets(edges, grid.shape)
    return [period for period in (2, 3) if red_pixels.issubset(allowed[period])]


def _build_input_subset(
    edges: list[dict],
    shadow_lists: list[list[tuple[int, tuple[int, int]]]],
    period: int,
    rng: random.Random,
) -> dict[int, set[tuple[int, int]]] | None:
    eligible = [idx for idx, cells in enumerate(shadow_lists) if cells]
    rich = [idx for idx, cells in enumerate(shadow_lists) if len(cells) >= 2]
    if len(eligible) < 3 or not rich:
        return None

    other = 5 - period
    evidence_edges = [
        idx
        for idx, cells in enumerate(shadow_lists)
        if any(arc % other != 0 for arc, _pos in cells)
    ]
    if not evidence_edges:
        return None

    evidence_edge = rng.choice(evidence_edges)
    partial_choices = [idx for idx in rich if idx != evidence_edge] or rich
    partial_edge = rng.choice(partial_choices)
    remaining_full = [idx for idx in eligible if idx not in {evidence_edge, partial_edge}]
    if not remaining_full:
        return None
    full_edge = rng.choice(remaining_full)
    remaining_empty = [idx for idx in eligible if idx not in {evidence_edge, partial_edge, full_edge}]
    if not remaining_empty:
        return None
    empty_edge = rng.choice(remaining_empty)

    observed: dict[int, set[tuple[int, int]]] = {idx: set() for idx in range(len(edges))}

    for idx, cells in enumerate(shadow_lists):
        pos_only = [pos for _arc, pos in cells]
        if not pos_only:
            observed[idx] = set()
            continue
        if idx == full_edge:
            observed[idx] = set(pos_only)
            continue
        if idx == empty_edge:
            observed[idx] = set()
            continue

        if idx == evidence_edge:
            keep = {pos for arc, pos in cells if arc % other != 0}
            if not keep:
                return None
            if len(keep) == len(pos_only) and len(pos_only) > 1:
                keep.remove(pos_only[-1])
            observed[idx] = set(keep)
            continue

        if idx == partial_edge:
            count = rng.randint(1, len(pos_only) - 1)
            observed[idx] = set(rng.sample(pos_only, count))
            continue

        mode = rng.choices(("full", "partial", "empty"), weights=(0.35, 0.4, 0.25), k=1)[0]
        if mode == "full":
            observed[idx] = set(pos_only)
        elif mode == "empty":
            observed[idx] = set()
        else:
            if len(pos_only) == 1:
                observed[idx] = {pos_only[0]}
            else:
                count = rng.randint(1, len(pos_only) - 1)
                observed[idx] = set(rng.sample(pos_only, count))

    observed_edges = sum(1 for cells in observed.values() if cells)
    if observed_edges < 2:
        return None
    if not any(0 < len(observed[idx]) < len(shadow_lists[idx]) for idx in range(len(edges)) if shadow_lists[idx]):
        return None
    if not any(len(observed[idx]) == 0 for idx in eligible):
        return None
    return observed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    period = rng.choice((2, 3))
    local_vertices = rng.choice(TEMPLATES)(rng)
    try:
        vertices, shape, _touch_border = _place_vertices(local_vertices, rng)
    except ValueError:
        return None

    edges = _build_edges(vertices)
    shadow_lists = _edge_shadow_lists(edges, period, shape)

    if not _validate_full_output(vertices, edges, shadow_lists):
        return None
    eligible = [cells for cells in shadow_lists if cells]
    if len(eligible) < 3:
        return None
    if not any(len(cells) >= 2 for cells in eligible):
        return None

    observed = _build_input_subset(edges, shadow_lists, period, rng)
    if observed is None:
        return None

    input_grid = _render_shadow_subset(shape, vertices, observed, draw_full_outline=False)
    output_grid = _render_shadow_subset(
        shape,
        vertices,
        {idx: {pos for _arc, pos in cells} for idx, cells in enumerate(shadow_lists)},
        draw_full_outline=True,
    )

    candidates = _period_candidates_from_input(input_grid)
    if candidates != [period]:
        return None
    if not np.array_equal(main(input_grid), output_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for seed in range(GEN_RETRIES):
        rng = random.Random(random.randrange(1 << 30) + seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid periodic-outline-shadow input")
