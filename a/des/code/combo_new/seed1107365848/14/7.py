from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG = 0
COLORS = tuple(range(1, 10))

GRID_MIN = 14
GRID_MAX = 30
GEN_RETRIES = 400
PLACEMENT_TRIES = 120
FAMILIES = ("rectangle", "bottom_notch", "left_notch")


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    comps.sort(key=lambda comp: (min(r for r, _ in comp), min(c for _, c in comp)))
    return comps


def _degrees(comp: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    deg: dict[tuple[int, int], int] = {}
    for r, c in comp:
        d = 0
        for dr, dc in DIR4:
            if (r + dr, c + dc) in comp:
                d += 1
        deg[(r, c)] = d
    return deg


def _is_simple_open_path(comp: list[tuple[int, int]]) -> bool:
    if len(comp) < 2:
        return False
    deg = _degrees(set(comp))
    endpoints = [pt for pt, d in deg.items() if d == 1]
    return len(endpoints) == 2 and all(d in (1, 2) for d in deg.values())


def _find_path_color(grid: np.ndarray, bg: int) -> tuple[int, list[list[tuple[int, int]]]]:
    candidates: list[tuple[int, list[list[tuple[int, int]]]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        comps = _components_of_color(grid, color)
        if not comps or any(len(comp) < 2 for comp in comps):
            continue
        if all(_is_simple_open_path(comp) for comp in comps):
            candidates.append((color, comps))
    assert len(candidates) == 1, "expected a unique path color"
    return candidates[0]


def _l_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    pts: list[tuple[int, int]] = []

    c_step = 1 if ec >= sc else -1
    for c in range(sc, ec + c_step, c_step):
        pts.append((sr, c))

    if er != sr:
        r_step = 1 if er >= sr else -1
        for r in range(sr + r_step, er + r_step, r_step):
            pts.append((r, ec))
    return pts


def _interior_cells(boundary: set[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in boundary)
    max_r = max(r for r, _ in boundary)
    min_c = min(c for _, c in boundary)
    max_c = max(c for _, c in boundary)

    h = max_r - min_r + 3
    w = max_c - min_c + 3
    blocked = np.zeros((h, w), dtype=bool)
    for r, c in boundary:
        blocked[r - min_r + 1, c - min_c + 1] = True

    seen = np.zeros((h, w), dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not seen[nr, nc] and not blocked[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    inside: list[tuple[int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if blocked[r, c] or seen[r, c]:
                continue
            inside.append((min_r + r - 1, min_c + c - 1))
    return inside


def main(grid: np.ndarray) -> np.ndarray:
    """
    Close each open path with a horizontal-then-vertical segment and fill its interior using the endpoint marker color.

    Input structure:
    - Background is black.
    - One foreground color forms several 4-connected, non-branching open paths; each path has exactly two endpoints.
    - Exactly one endpoint of each path has a single non-black, non-path marker pixel in its 4-neighborhood.
    - Paths are separated from one another so each marker belongs unambiguously to one path.

    Rule:
    - Identify the shared path color and each of its connected path components.
    - For every path, use the marker-adjacent endpoint as the start and connect it to the other endpoint with path-color pixels, moving horizontally first and then vertically.
    - Treat the original path plus this added L-shaped segment as a closed polygon and fill its interior with the marker's color, without recoloring the boundary.
    - Erase each marker pixel to black while leaving all path-color pixels in place.

    Key insight:
    - The marker identifies both the fill color and which endpoint is the start of the missing closure, so the gap must be sealed with a specific L-shaped route rather than by an arbitrary straight connection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _bg_color(g)
    path_color, path_components = _find_path_color(g, bg)

    out = g.copy()
    h, w = g.shape

    for comp in path_components:
        comp_set = set(comp)
        deg = _degrees(comp_set)
        endpoints = [pt for pt, d in deg.items() if d == 1]
        assert len(endpoints) == 2, "each path component must have two endpoints"

        start = None
        marker = None
        for ep in endpoints:
            neighbors = []
            for nr, nc in _neighbors4(ep[0], ep[1], h, w):
                v = int(g[nr, nc])
                if v not in (bg, path_color):
                    neighbors.append((nr, nc))
            if neighbors:
                assert start is None, "marker must identify a unique start endpoint"
                assert len(neighbors) == 1, "start endpoint must have exactly one marker"
                start = ep
                marker = neighbors[0]

        assert start is not None and marker is not None, "no marker-adjacent endpoint found"
        end = endpoints[0] if endpoints[1] == start else endpoints[1]
        marker_color = int(g[marker])

        closure = _l_path(start, end)
        boundary = comp_set | set(closure)
        fill_cells = _interior_cells(boundary)

        for r, c in closure:
            out[r, c] = path_color
        for r, c in fill_cells:
            if (r, c) != marker:
                out[r, c] = marker_color
        out[marker] = bg

    return out


def _segment_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    ar, ac = a
    br, bc = b
    if ar == br:
        step = 1 if bc >= ac else -1
        return [(ar, c) for c in range(ac, bc + step, step)]
    assert ac == bc, "segment must be axis-aligned"
    step = 1 if br >= ar else -1
    return [(r, ac) for r in range(ar, br + step, step)]


def _polyline(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for i in range(len(vertices) - 1):
        seg = _segment_points(vertices[i], vertices[i + 1])
        if pts:
            seg = seg[1:]
        pts.extend(seg)
    return pts


def _reflect_point(
    pt: tuple[int, int], height: int, width: int, flip_r: bool, flip_c: bool
) -> tuple[int, int]:
    r, c = pt
    if flip_r:
        r = height - 1 - r
    if flip_c:
        c = width - 1 - c
    return r, c


def _make_vertices(rng: random.Random, family: str) -> list[tuple[int, int]]:
    if family == "rectangle":
        height = rng.randint(4, 8)
        width = rng.randint(4, 8)
        return [(0, 0), (0, width - 1), (height - 1, width - 1), (height - 1, 0), (0, 0)]

    if family == "bottom_notch":
        height = rng.randint(6, 10)
        width = rng.randint(7, 11)
        x1 = rng.randint(1, width - 4)
        x2 = rng.randint(x1 + 2, width - 2)
        y = rng.randint(2, height - 3)
        return [
            (0, 0),
            (0, width - 1),
            (height - 1, width - 1),
            (height - 1, x2),
            (y, x2),
            (y, x1),
            (height - 1, x1),
            (height - 1, 0),
            (0, 0),
        ]

    assert family == "left_notch"
    height = rng.randint(7, 11)
    width = rng.randint(6, 10)
    y1 = rng.randint(1, height - 4)
    y2 = rng.randint(y1 + 2, height - 2)
    x = rng.randint(2, width - 3)
    return [
        (0, 0),
        (0, width - 1),
        (height - 1, width - 1),
        (height - 1, 0),
        (y2, 0),
        (y2, x),
        (y1, x),
        (y1, 0),
        (0, 0),
    ]


def _normalize_points(points: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    shifted = [(r - min_r, c - min_c) for r, c in points]
    height = max(r for r, _ in shifted) + 1
    width = max(c for _, c in shifted) + 1
    return shifted, height, width


def _make_object(rng: random.Random, marker_color: int) -> dict[str, object] | None:
    family = rng.choice(FAMILIES)
    vertices = _make_vertices(rng, family)
    full_boundary = _polyline(vertices)
    open_path = _polyline(vertices[2:])
    closure = _l_path(vertices[0], vertices[2])

    base_height = max(r for r, _ in full_boundary) + 1
    base_width = max(c for _, c in full_boundary) + 1

    marker = rng.choice([(-1, 0), (0, -1)])
    flip_r = rng.choice([False, True])
    flip_c = rng.choice([False, True])

    full_boundary = [_reflect_point(pt, base_height, base_width, flip_r, flip_c) for pt in full_boundary]
    open_path = [_reflect_point(pt, base_height, base_width, flip_r, flip_c) for pt in open_path]
    closure = [_reflect_point(pt, base_height, base_width, flip_r, flip_c) for pt in closure]
    marker = _reflect_point(marker, base_height, base_width, flip_r, flip_c)

    all_pts = full_boundary + [marker]
    normalized_all, obj_h, obj_w = _normalize_points(all_pts)
    shift_r = normalized_all[-1][0] - marker[0]
    shift_c = normalized_all[-1][1] - marker[1]

    def shift(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        return [(r + shift_r, c + shift_c) for r, c in points]

    full_boundary = shift(full_boundary)
    open_path = shift(open_path)
    closure = shift(closure)
    marker = (marker[0] + shift_r, marker[1] + shift_c)

    boundary_set = set(full_boundary)
    path_set = set(open_path)
    closure_set = set(closure)
    fill_cells = set(_interior_cells(boundary_set))
    if not fill_cells:
        return None
    if marker in boundary_set or marker in fill_cells:
        return None
    if closure_set - path_set - boundary_set:
        return None
    if (path_set | closure_set) != boundary_set:
        return None
    if not _is_simple_open_path(sorted(path_set)):
        return None

    deg = _degrees(path_set)
    endpoints = [pt for pt, d in deg.items() if d == 1]
    if len(endpoints) != 2:
        return None
    marker_adj = [pt for pt in endpoints if abs(pt[0] - marker[0]) + abs(pt[1] - marker[1]) == 1]
    if len(marker_adj) != 1:
        return None
    for pt in path_set:
        if pt != marker_adj[0] and abs(pt[0] - marker[0]) + abs(pt[1] - marker[1]) == 1:
            return None

    reserve = boundary_set | fill_cells | {marker}
    return {
        "path": path_set,
        "marker": marker,
        "marker_color": marker_color,
        "reserve": reserve,
        "height": obj_h,
        "width": obj_w,
    }


def _mark_forbidden(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                mask[rr, cc] = True


def _place_object(
    rng: random.Random,
    grid: np.ndarray,
    forbidden: np.ndarray,
    path_color: int,
    obj: dict[str, object],
) -> bool:
    h, w = grid.shape
    obj_h = int(obj["height"])
    obj_w = int(obj["width"])
    if obj_h > h or obj_w > w:
        return False

    path_cells = sorted(obj["path"])
    reserve_cells = sorted(obj["reserve"])
    marker_r, marker_c = obj["marker"]
    marker_color = int(obj["marker_color"])

    for _ in range(PLACEMENT_TRIES):
        top = rng.randint(0, h - obj_h)
        left = rng.randint(0, w - obj_w)
        placed_reserve = [(top + r, left + c) for r, c in reserve_cells]
        if any(forbidden[r, c] for r, c in placed_reserve):
            continue

        for r, c in path_cells:
            grid[top + r, left + c] = path_color
        grid[top + marker_r, left + marker_c] = marker_color
        _mark_forbidden(forbidden, set(placed_reserve))
        return True

    return False


def _validate_generated_input(grid: np.ndarray, n_objects: int) -> bool:
    bg = _bg_color(grid)
    path_color, comps = _find_path_color(grid, bg)
    if len(comps) != n_objects:
        return False
    h, w = grid.shape
    for comp in comps:
        deg = _degrees(set(comp))
        endpoints = [pt for pt, d in deg.items() if d == 1]
        if len(endpoints) != 2:
            return False
        marked = 0
        for ep in endpoints:
            neighbors = 0
            for nr, nc in _neighbors4(ep[0], ep[1], h, w):
                if int(grid[nr, nc]) not in (bg, path_color):
                    neighbors += 1
            if neighbors:
                if neighbors != 1:
                    return False
                marked += 1
        if marked != 1:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.choice([1, 2, 3])
    min_dim = max(GRID_MIN, 12 + 4 * (n_objects - 1))
    height = rng.randint(min_dim, GRID_MAX)
    width = rng.randint(min_dim, GRID_MAX)
    path_color = rng.choice(COLORS)
    marker_colors = rng.sample([c for c in COLORS if c != path_color], n_objects)

    objects: list[dict[str, object]] = []
    for marker_color in marker_colors:
        obj = _make_object(rng, marker_color)
        if obj is None:
            return None
        objects.append(obj)

    objects.sort(key=lambda obj: -len(obj["reserve"]))

    grid = np.full((height, width), BG, dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)
    for obj in objects:
        if not _place_object(rng, grid, forbidden, path_color, obj):
            return None

    if not _validate_generated_input(grid, n_objects):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
