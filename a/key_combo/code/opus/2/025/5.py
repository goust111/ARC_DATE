from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from arc_utils import Color, coords_where, get_bg, in_bounds
except ImportError:
    utils_dir = Path("/home/lixinyu/snap/new/solvers")
    if str(utils_dir) not in sys.path:
        sys.path.append(str(utils_dir))
    from arc_utils import Color, coords_where, get_bg, in_bounds


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


GRID_ROWS_RANGE = (18, 24)
GRID_COLS_RANGE = (18, 24)
MAX_GENERATION_ATTEMPTS = 5000
MIN_BBOX_GAP = 2
MAX_BBOX_GAP = 5
GREEN = Color.GREEN
MARKER_COLORS = (Color.BLUE, Color.RED)


@dataclass(frozen=True)
class Parallelogram:
    marker_color: int
    p: Coord
    q: Coord
    diag_vec: Coord
    straight_vec: Coord
    diag_len: int
    straight_len: int
    vertices: tuple[Coord, Coord, Coord, Coord]
    border_cells: frozenset[Coord]
    green_cells: frozenset[Coord]
    marker_cells: frozenset[Coord]
    bbox: BBox


@dataclass(frozen=True)
class RayTrace:
    color: int
    start: Coord
    direction: Coord
    path: tuple[Coord, ...]
    hit: Coord | None
    recolor_hit: bool


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _step(vec: Coord) -> Coord:
    return _sign(vec[0]), _sign(vec[1])


def _add(a: Coord, b: Coord) -> Coord:
    return a[0] + b[0], a[1] + b[1]


def _neg(a: Coord) -> Coord:
    return -a[0], -a[1]


def _segment_cells(a: Coord, b: Coord) -> list[Coord]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    length = max(abs(dr), abs(dc))
    assert length > 0
    assert abs(dr) in (0, length)
    assert abs(dc) in (0, length)
    sr = _sign(dr)
    sc = _sign(dc)
    return [(a[0] + i * sr, a[1] + i * sc) for i in range(length + 1)]


def _decompose_opposite_corners(p: Coord, q: Coord) -> tuple[Coord, Coord, int, int]:
    dr = q[0] - p[0]
    dc = q[1] - p[1]
    abs_dr = abs(dr)
    abs_dc = abs(dc)
    assert abs_dr > 0 and abs_dc > 0 and abs_dr != abs_dc

    diag_len = min(abs_dr, abs_dc)
    straight_len = abs(abs_dr - abs_dc)
    assert diag_len > 0 and straight_len > 0

    sr = _sign(dr)
    sc = _sign(dc)
    diag_vec = (sr * diag_len, sc * diag_len)
    if abs_dr > abs_dc:
        straight_vec = (sr * straight_len, 0)
    else:
        straight_vec = (0, sc * straight_len)
    return diag_vec, straight_vec, diag_len, straight_len


def _build_parallelogram(marker_color: int, p: Coord, q: Coord) -> Parallelogram:
    diag_vec, straight_vec, diag_len, straight_len = _decompose_opposite_corners(p, q)
    corner_diag = _add(p, diag_vec)
    corner_straight = _add(p, straight_vec)
    vertices = (p, corner_diag, q, corner_straight)

    border = set()
    for a, b in (
        (p, corner_diag),
        (p, corner_straight),
        (corner_diag, q),
        (corner_straight, q),
    ):
        border.update(_segment_cells(a, b))

    rows = [r for r, _ in vertices]
    cols = [c for _, c in vertices]
    markers = frozenset((p, q))
    return Parallelogram(
        marker_color=int(marker_color),
        p=p,
        q=q,
        diag_vec=diag_vec,
        straight_vec=straight_vec,
        diag_len=diag_len,
        straight_len=straight_len,
        vertices=vertices,
        border_cells=frozenset(border),
        green_cells=frozenset(border - set(markers)),
        marker_cells=markers,
        bbox=(min(rows), min(cols), max(rows), max(cols)),
    )


def _outward_normal(edge_vec: Coord, other_vec: Coord, *, flip: bool = False) -> Coord:
    er, ec = _step(edge_vec)
    normal = (ec, -er)
    if other_vec[0] * normal[0] + other_vec[1] * normal[1] > 0:
        normal = (-normal[0], -normal[1])
    if flip:
        normal = (-normal[0], -normal[1])
    return normal


def _marker_rays(shape: Parallelogram, *, use_short_edge: bool = True, flip_normals: bool = False) -> list[tuple[Coord, Coord]]:
    short_is_diag = shape.diag_len < shape.straight_len
    edge_vec = shape.diag_vec if short_is_diag == use_short_edge else shape.straight_vec
    other_vec = shape.straight_vec if short_is_diag == use_short_edge else shape.diag_vec
    return [
        (shape.p, _outward_normal(edge_vec, other_vec, flip=flip_normals)),
        (
            shape.q,
            _outward_normal(_neg(edge_vec), _neg(other_vec), flip=flip_normals),
        ),
    ]


def _draw_base_grid(shape: tuple[int, int], bg: int, parallelograms: list[Parallelogram]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for pg in parallelograms:
        for r, c in pg.green_cells:
            out[r, c] = GREEN
    for pg in parallelograms:
        out[pg.p] = pg.marker_color
        out[pg.q] = pg.marker_color
    return out


def _trace_one_ray(
    base: np.ndarray,
    own_shape: Parallelogram,
    other_shape: Parallelogram,
    start: Coord,
    direction: Coord,
    bg: int,
) -> RayTrace:
    path: list[Coord] = []
    r, c = start
    while True:
        r += direction[0]
        c += direction[1]
        if not in_bounds(base.shape, r, c):
            return RayTrace(own_shape.marker_color, start, direction, tuple(path), None, False)
        if int(base[r, c]) != int(bg):
            hit = (r, c)
            recolor = hit in other_shape.green_cells and int(base[r, c]) == GREEN
            return RayTrace(own_shape.marker_color, start, direction, tuple(path), hit, recolor)
        path.append((r, c))


def _collect_rays(
    base: np.ndarray,
    parallelograms: list[Parallelogram],
    bg: int,
    *,
    use_short_edge: bool = True,
    flip_normals: bool = False,
) -> list[RayTrace]:
    rays: list[RayTrace] = []
    for i, pg in enumerate(parallelograms):
        other = parallelograms[1 - i]
        for start, direction in _marker_rays(pg, use_short_edge=use_short_edge, flip_normals=flip_normals):
            rays.append(_trace_one_ray(base, pg, other, start, direction, bg))
    return rays


def _apply_rays(base: np.ndarray, rays: list[RayTrace]) -> np.ndarray:
    out = base.copy()
    for ray in rays:
        for r, c in ray.path:
            out[r, c] = ray.color
    for ray in rays:
        if ray.recolor_hit and ray.hit is not None:
            out[ray.hit] = ray.color
    return out


def _simulate(
    grid_shape: tuple[int, int],
    parallelograms: list[Parallelogram],
    bg: int = Color.BLACK,
    *,
    use_short_edge: bool = True,
    flip_normals: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[RayTrace]]:
    base = _draw_base_grid(grid_shape, bg, parallelograms)
    rays = _collect_rays(base, parallelograms, bg, use_short_edge=use_short_edge, flip_normals=flip_normals)
    return base, _apply_rays(base, rays), rays


def _bbox_gap(a: BBox, b: BBox) -> int:
    row_gap = max(b[0] - a[2] - 1, a[0] - b[2] - 1, 0)
    col_gap = max(b[1] - a[3] - 1, a[1] - b[3] - 1, 0)
    return max(row_gap, col_gap)


def _sample_shape(marker_color: int, rows: int, cols: int) -> Parallelogram:
    short_is_diag = bool(random.randrange(2))
    if short_is_diag:
        diag_len = random.randint(2, 4)
        straight_len = random.randint(diag_len + 1, 5)
    else:
        straight_len = random.randint(2, 4)
        diag_len = random.randint(straight_len + 1, 5)

    sr = random.choice((-1, 1))
    sc = random.choice((-1, 1))
    axis = random.choice(("horizontal", "vertical"))

    diag_vec = (sr * diag_len, sc * diag_len)
    if axis == "horizontal":
        straight_vec = (0, sc * straight_len)
    else:
        straight_vec = (sr * straight_len, 0)

    offsets = [(0, 0), diag_vec, straight_vec, _add(diag_vec, straight_vec)]
    min_r = min(r for r, _ in offsets)
    max_r = max(r for r, _ in offsets)
    min_c = min(c for _, c in offsets)
    max_c = max(c for _, c in offsets)
    p = (
        random.randint(-min_r, rows - 1 - max_r),
        random.randint(-min_c, cols - 1 - max_c),
    )
    q = _add(_add(p, diag_vec), straight_vec)
    return _build_parallelogram(marker_color, p, q)


def _candidate_input(rows: int, cols: int, parallelograms: list[Parallelogram]) -> np.ndarray:
    grid = np.zeros((rows, cols), dtype=int)
    for pg in parallelograms:
        grid[pg.p] = pg.marker_color
        grid[pg.q] = pg.marker_color
    return grid


def _ray_groups(rays: list[RayTrace]) -> tuple[list[RayTrace], list[RayTrace]]:
    return rays[:2], rays[2:]


def _valid_candidate(rows: int, cols: int, parallelograms: list[Parallelogram]) -> bool:
    blue, red = parallelograms
    if blue.border_cells & red.border_cells:
        return False
    if blue.marker_cells & red.border_cells:
        return False
    if red.marker_cells & blue.border_cells:
        return False
    gap = _bbox_gap(blue.bbox, red.bbox)
    if not (MIN_BBOX_GAP <= gap <= MAX_BBOX_GAP):
        return False

    base, out, rays = _simulate((rows, cols), parallelograms)
    blue_rays, red_rays = _ray_groups(rays)
    if not any(ray.recolor_hit for ray in blue_rays):
        return False
    if not any(ray.recolor_hit for ray in red_rays):
        return False
    if any(len(ray.path) == 0 for ray in rays):
        return False

    painted_regions: list[set[Coord]] = []
    for ray in rays:
        cells = set(ray.path)
        if ray.recolor_hit and ray.hit is not None:
            cells.add(ray.hit)
        painted_regions.append(cells)
    for i in range(len(painted_regions)):
        for j in range(i + 1, len(painted_regions)):
            if painted_regions[i] & painted_regions[j]:
                return False

    if np.array_equal(base, out):
        return False

    _, alt_long, _ = _simulate((rows, cols), parallelograms, use_short_edge=False)
    if np.array_equal(out, alt_long):
        return False

    _, alt_inward, _ = _simulate((rows, cols), parallelograms, flip_normals=True)
    if np.array_equal(out, alt_inward):
        return False

    inp = _candidate_input(rows, cols, parallelograms)
    if not np.array_equal(main(inp), out):
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw two marker-defined green parallelograms, then fire short-edge outward rays that repaint the other frame on impact.

    Input structure:
    - The grid has a black background and exactly four marker pixels: two blue cells and two red cells.
    - Each same-color pair marks opposite corners of one implied parallelogram.
    - For each pair, both row and column offsets are non-zero and have different magnitudes, so the diagonal-plus-axis decomposition is unique.

    Rule:
    - For each color pair, decompose the marker-to-marker offset into a 45-degree diagonal vector and the remaining axis-aligned vector.
    - Use those two vectors as adjacent sides, draw the green outline of the parallelogram, and keep the original markers visible.
    - From each marker, emit a ray in the outward normal direction of the shorter incident side; draw the ray in the marker color until the first non-black pixel, and recolor that hit only when it is a green border pixel of the other parallelogram.

    Key insight:
    - The markers do not define a direct segment; they define a whole parallelogram whose shorter side fixes each ray direction, so the final recolored border pixels depend on both shapes together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(get_bg(g))
    assert bg == Color.BLACK

    parallelograms: list[Parallelogram] = []
    for color in MARKER_COLORS:
        coords = coords_where(g == color)
        assert coords.shape == (2, 2)
        p = tuple(map(int, coords[0]))
        q = tuple(map(int, coords[1]))
        parallelograms.append(_build_parallelogram(color, p, q))

    base = _draw_base_grid(g.shape, bg, parallelograms)
    rays = _collect_rays(base, parallelograms, bg)
    return _apply_rays(base, rays)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rows = random.randint(*GRID_ROWS_RANGE)
        cols = random.randint(*GRID_COLS_RANGE)
        parallelograms = [_sample_shape(color, rows, cols) for color in MARKER_COLORS]
        if _valid_candidate(rows, cols, parallelograms):
            return _candidate_input(rows, cols, parallelograms)
    raise RuntimeError("failed to generate a valid puzzle instance")
