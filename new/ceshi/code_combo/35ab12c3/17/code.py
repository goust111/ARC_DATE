# Combo variant of 35ab12c3 (index 17)
# combo_spec: D5/2+D21/0
# Name: 阴影方向由形状朝向决定+环绕边界

# Design analysis for the fused variant
# ====================================
# a) Concrete behavior and trace
#    - Base task: same-color vertices define a polyline/polygon; the solver connects
#      aligned H/V/45deg neighbors into the main outline. In the original task, a
#      different-colored 8-neighbor next to one vertex supplied the shadow offset.
#    - D5/2 replaces that anchor rule: the shadow color is still visible in the input,
#      but only as one singleton marker strictly inside the shape bbox. The offset is
#      inferred geometrically from the shape itself: pick the unique longest drawable
#      edge, take a unit normal, then choose the normal pointing away from the other
#      vertices. Example: vertices (0,0),(0,8),(4,4) yield longest edge (0,0)->(0,8),
#      so the shadow direction is upward (-1,0), away from the third vertex below.
#    - D21/0 replaces clipping with toroidal continuation. If the same triangle is
#      placed with its top edge on row 0, shifting by (-1,0) moves that top edge to
#      row H-1 while the other shifted edges remain near the shape. The shadow can
#      therefore appear on the opposite border while still matching the shape's normal.
#    - Fused difference: the input no longer exposes the offset locally, and the
#      output may contain border-wrapped shadow cells. The only consistent explanation
#      across examples is "infer shadow direction from the shape; then wrap overflow".
#
# b) Complexity calibration
#    - If shadowed shapes used arbitrary symmetric polygons, "longest edge normal"
#      would often be ambiguous or visually weak. To keep the rule legible, every
#      shadowed shape is generated from a dedicated asymmetric triangle with one
#      unique longest edge and all remaining vertices on one side of it.
#    - Unshadowed shapes still come from the broader original mixture so the task does
#      not collapse into a single template. The shadow direction has only eight unit
#      possibilities, and repeated wrapped examples reinforce the same geometric idea.
#
# c) Implicit assumptions audit
#    - Original `seed_idxs` / neighbor-carried offsets assume a different-colored
#      pixel sits next to a vertex. This fails under D5/2, so the fused solver removes
#      all seed-neighbor logic and instead searches singleton colors inside a shape bbox.
#    - Original shadow drawing assumed translated endpoints must remain in-bounds; that
#      is the clipped behavior broken by D21/0. The fused solver replaces endpoint
#      bounds checks with per-cell modulo wrapping for the shadow layer only.
#    - Original `_clear_in_original(... allow=set())` assumed the input contains no
#      pixels on the shifted path. Here the singleton marker lies on the future shadow,
#      so shifted-path clearance must allow that shadow color.
#    - New assumption introduced by the fused solver: if a shape has a shadow, there is
#      exactly one singleton color strictly inside its bbox, the longest edge is unique,
#      and the chosen normal points away from the remaining vertices. The generator
#      constructs only such shadowed triangles and prevents foreign singleton pixels
#      from landing inside any shape bbox.
#
# d) Placement legality
#    - Because wrapped shadows may occupy the opposite canvas edge, the generator works
#      directly on the full canvas instead of pasting local patches. It tracks occupied
#      input/output cells and enforces toroidal 8-neighbor separation, so wrapped
#      shadows cannot collide with unrelated objects across the border.
#    - It also tracks every shape bbox and forbids unrelated input pixels from appearing
#      strictly inside any bbox; otherwise a distractor singleton could masquerade as a
#      shadow marker.
#
# e) Visual observability
#    - Every generated puzzle has at least one shadow and at least one true wrap event.
#      The generator rejects any sample whose output matches a "same geometry but
#      clipped" reference, so the toroidal behavior is visible in examples.
#    - It also rejects samples whose output matches a toroidal old-anchor reference.
#      This ensures the singleton marker is genuinely non-local and the geometry-driven
#      direction rule matters.
#
# f) Visual cleanliness
#    - Only one shadow marker is kept per shadowed shape, and that marker is chosen on
#      the actual shadow path, strictly inside the shape bbox, and outside every vertex
#      8-neighborhood. This avoids noisy clusters that could be mistaken for extra
#      vertices.
#    - Toroidal spacing checks keep wrapped shadows from touching other groups on the
#      far side of the board.
#
# g) Change inferability
#    - Colors keep their original semantic roles: one color per main shape, one unique
#      color per shadow, optional distractor color unchanged. No color-id ordering or
#      hidden palette rule drives the geometry.
#    - The only color-specific cue is the singleton shadow marker, and its role is
#      paired with a repeated structural cue: the corresponding output shadow is always
#      a one-cell normal offset of the main shape, sometimes continuing across a border.
#
# h) Validation targets
#    - Generated samples must satisfy:
#      1. `main(inp)` reproduces the intended output.
#      2. At least one shadow wraps across a canvas edge.
#      3. The output differs from a clipped-orientation ablation and from a toroidal
#         old-anchor ablation on the same input.
#      4. Every shadow marker is not 8-adjacent to any vertex of its shape.
#
# i) Solver assumptions matched by the generator
#    - Each shadow color appears exactly once in the input.
#    - That singleton lies on the true shadow-only path for its shape.
#    - No other singleton lies strictly inside any shape bbox.
#    - Shadowed shapes always have a unique longest edge, so the normal direction is
#      deterministic.

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, colors_present, get_bg  # noqa: E402


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


def _wrap_rc(h: int, w: int, r: int, c: int) -> tuple[int, int]:
    return r % h, c % w


def _neighbors8_toroidal(h: int, w: int, r: int, c: int):
    for dr, dc in DIR8:
        yield dr, dc, (r + dr) % h, (c + dc) % w


def _clear_in_original(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _clear_shifted_in_original_toroidal(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    off_r: int,
    off_c: int,
    bg: int,
    allow: set[int],
) -> bool:
    h, w = g.shape
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        wr, wc = _wrap_rc(h, w, rr + off_r, cc + off_c)
        v = int(g[wr, wc])
        if v != bg and v not in allow:
            return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _draw_shifted_line_bg_only_toroidal(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    off_r: int,
    off_c: int,
    col: int,
    bg: int,
) -> None:
    h, w = out.shape
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        wr, wc = _wrap_rc(h, w, rr + off_r, cc + off_c)
        if out[wr, wc] == bg:
            out[wr, wc] = col


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _strict_interior_marker_color(
    singletons: dict[int, tuple[int, int]],
    vertices: list[tuple[int, int]],
) -> int | None:
    arr = np.array(vertices, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    candidates = [col for col, (r, c) in singletons.items() if r0 < r < r1 and c0 < c < c1]
    if len(candidates) != 1:
        return None
    return int(candidates[0])


def _shape_shadow_offset(
    vertices: list[tuple[int, int]],
    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[int, int] | None:
    if not drawable_edges:
        return None
    lengths = [max(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in drawable_edges]
    max_len = max(lengths)
    if lengths.count(max_len) != 1:
        return None
    edge_idx = lengths.index(max_len)
    (r0, c0), (r1, c1) = drawable_edges[edge_idx]
    others = [p for p in vertices if p != (r0, c0) and p != (r1, c1)]
    if not others:
        return None

    sr, sc = _step_dir(r0, c0, r1, c1)
    perp_a = (-sc, sr)
    perp_b = (sc, -sr)
    mr = (r0 + r1) / 2.0
    mc = (c0 + c1) / 2.0
    orr = sum(r for r, _ in others) / len(others)
    occ = sum(c for _, c in others) / len(others)

    dot_a = (orr - mr) * perp_a[0] + (occ - mc) * perp_a[1]
    dot_b = (orr - mr) * perp_b[0] + (occ - mc) * perp_b[1]
    if dot_a == dot_b:
        return None
    return perp_a if dot_a < dot_b else perp_b


def _solve_orientation_shadow(grid: np.ndarray, *, wrap_shadow: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    all_colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    coords_by_color = {
        col: [tuple(map(int, p)) for p in np.argwhere(g == col)]
        for col in all_colors
    }
    singletons = {col: pts[0] for col, pts in coords_by_color.items() if len(pts) == 1}

    for col in all_colors:
        pts_rc = coords_by_color[col]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
        cycle = cycle[start:] + cycle[:start]

        edges = _shape_edges(cycle)
        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        shadow_col = _strict_interior_marker_color(singletons, cycle)
        if shadow_col is None:
            continue

        offset = _shape_shadow_offset(cycle, drawable_edges)
        if offset is None:
            continue
        drn, dcn = offset

        for (r0, c0), (r1, c1) in drawable_edges:
            if wrap_shadow:
                if not _clear_shifted_in_original_toroidal(
                    g, r0, c0, r1, c1, drn, dcn, bg, allow={shadow_col}
                ):
                    continue
                _draw_shifted_line_bg_only_toroidal(out, r0, c0, r1, c1, drn, dcn, shadow_col, bg)
            else:
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow={shadow_col}):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, shadow_col, bg)

    return out


def _solve_anchor_toroidal_reference(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        seed_idxs: list[int] = []
        for i, (r, c) in enumerate(cycle):
            for _drn, _dcn, rr, cc in _neighbors8_toroidal(h, w, r, c):
                v = int(g[rr, cc])
                if v != bg and v != col:
                    seed_idxs.append(i)
                    break

        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
        cycle = cycle[start:] + cycle[:start]

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn, rr, cc in _neighbors8_toroidal(h, w, sr0, sc0):
            v = int(g[rr, cc])
            if v != bg and v != col:
                carried.append((drn, dcn, v))

        edges = _shape_edges(cycle)
        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                if not _clear_shifted_in_original_toroidal(g, r0, c0, r1, c1, drn, dcn, bg, allow=set()):
                    continue
                _draw_shifted_line_bg_only_toroidal(out, r0, c0, r1, c1, drn, dcn, ncol, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect same-color vertices into straight outlines, then add a wrapped shadow whose
    direction is inferred from the shape rather than from a local anchor pixel.

    Input structure:
    - Background is the most common color.
    - Each shape color appears as 2+ vertex pixels whose intended consecutive edges are
      H/V/45deg-aligned when ordered cyclically.
    - A shadowed shape has exactly one singleton pixel of another color strictly inside
      its bbox; that singleton only declares the shadow color and lies on the future
      shadow path, not next to a vertex.
    - Optional distractor colors remain unchanged because they do not form drawable shapes.

    Rule:
    - For every multi-pixel color, order its vertices around the centroid and draw each
      clear H/V/45deg edge between consecutive vertices.
    - If the shape contains exactly one interior singleton color, infer the shadow
      offset from the unique longest drawable edge: take the unit normal pointing away
      from the other vertices.
    - Copy every drawable edge by that offset using toroidal wrapping, painting only on
      background cells so the main shape keeps priority over the shadow.

    Key insight:
    - The singleton shadow color no longer tells you where to shift. It only tells you
      which shadow color belongs to the shape. The displacement is encoded by the shape's
      own orientation, and any overflow continues from the opposite border.
    """
    return _solve_orientation_shadow(grid, wrap_shadow=True)


def _main_clipped_orientation_reference(grid: np.ndarray) -> np.ndarray:
    return _solve_orientation_shadow(grid, wrap_shadow=False)


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (3, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)
SHADOW_TRIANGLE_ARM_RANGE = (3, 7)

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 90
DISTRACTOR_PLACEMENT_RETRIES = 90
GENERATE_RETRIES = 500


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _build_line() -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*LINE_VERTICES_RANGE))
    remaining = LINE_TOTAL_LENGTH
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle() -> list[tuple[int, int]]:
    side = int(np.random.randint(*TRIANGLE_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle() -> list[tuple[int, int]]:
    hh = int(np.random.randint(*RECT_H_RANGE))
    ww = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, ww), (hh, ww), (hh, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    hh = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (hh, 0), (hh, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


def _build_shadow_triangle() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*SHADOW_TRIANGLE_ARM_RANGE))
    return _transform_coords([(0, 0), (0, 2 * arm), (arm, arm)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _shape_bounds(vertices: list[tuple[int, int]], shape_cells: set[tuple[int, int]]) -> tuple[int, int]:
    coords = list(vertices) + list(shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _globalize(coords, base_r: int, base_c: int) -> set[tuple[int, int]]:
    return {(int(r) + base_r, int(c) + base_c) for r, c in coords}


def _globalize_list(coords: list[tuple[int, int]], base_r: int, base_c: int) -> list[tuple[int, int]]:
    return [(int(r) + base_r, int(c) + base_c) for r, c in coords]


def _shift_toroidal(coords: set[tuple[int, int]], offset: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    dr, dc = offset
    return {((r + dr) % h, (c + dc) % w) for r, c in coords}


def _has_wrap(coords: set[tuple[int, int]], offset: tuple[int, int], h: int, w: int) -> bool:
    dr, dc = offset
    return any((r + dr) < 0 or (r + dr) >= h or (c + dc) < 0 or (c + dc) >= w for r, c in coords)


def _placement_wraps(base_r: int, base_c: int, shape_h: int, shape_w: int, offset: tuple[int, int], h: int, w: int) -> bool:
    dr, dc = offset
    conds = []
    if dr == -1:
        conds.append(base_r == 0)
    if dr == 1:
        conds.append(base_r == h - shape_h)
    if dc == -1:
        conds.append(base_c == 0)
    if dc == 1:
        conds.append(base_c == w - shape_w)
    return any(conds)


def _strictly_inside_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = bbox
    return r0 < r < r1 and c0 < c < c1


def _any_inside_bbox(cells: set[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    return any(_strictly_inside_bbox(cell, bbox) for cell in cells)


def _inside_any_bbox(cells: set[tuple[int, int]], bboxes: list[tuple[int, int, int, int]]) -> bool:
    return any(_any_inside_bbox(cells, bbox) for bbox in bboxes)


def _has_toroidal_contact(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]], h: int, w: int) -> bool:
    if not cells_a or not cells_b:
        return False
    for r, c in cells_a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if ((r + dr) % h, (c + dc) % w) in cells_b:
                    return True
    return False


def _select_shadow_marker(
    shadow_only: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
) -> tuple[int, int] | None:
    arr = np.array(vertices, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    cr = sum(r for r, _ in vertices) / len(vertices)
    cc = sum(c for _, c in vertices) / len(vertices)

    candidates: list[tuple[float, int, int]] = []
    for rr, cc2 in shadow_only:
        if not (r0 < rr < r1 and c0 < cc2 < c1):
            continue
        if any(max(abs(rr - vr), abs(cc2 - vc)) <= 1 for vr, vc in vertices):
            continue
        dist = (rr - cr) ** 2 + (cc2 - cc) ** 2
        candidates.append((dist, rr, cc2))
    if not candidates:
        return None
    _, rr, cc2 = min(candidates)
    return int(rr), int(cc2)


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    coords: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    while len(coords) < count:
        cand = (int(np.random.randint(h)), int(np.random.randint(w)))
        if cand in seen:
            continue
        coords.append(cand)
        seen.add(cand)
    for i in range(len(coords)):
        for j in range(i + 1, len(coords)):
            if _aligned(coords[j][0] - coords[i][0], coords[j][1] - coords[i][1]):
                return None
    return coords


def _make_distractor_patch(color: int) -> np.ndarray:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        return np.full((1, 1), color, dtype=int)
    for _ in range(DISTRACTOR_RETRIES):
        hh = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        ww = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(hh, ww, count)
        if coords is None:
            continue
        patch = np.full((hh, ww), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch
    return np.full((1, 1), color, dtype=int)


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(
        MAX_SHADOWS_CAP,
        shape_count - 1,
        len(COLORS) - shape_count - int(has_distractor),
    )
    if max_shadows < 2:
        return None
    shadow_count = int(np.random.randint(2, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _sample_canvas_size() -> tuple[int, int]:
    hh = int(np.random.randint(*CANVAS_H_RANGE))
    ww = int(np.clip(hh + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return hh, ww


def _try_place_shape(
    inp: np.ndarray,
    out: np.ndarray,
    occupied_in: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
    shape_bboxes: list[tuple[int, int, int, int]],
    shape_color: int,
    shadow_color: int | None,
    *,
    require_wrap: bool,
    forbid_wrap: bool = False,
) -> bool:
    h, w = inp.shape
    if shadow_color is None:
        vertices = _make_vertices()
        shadow_offset = None
    else:
        vertices = _build_shadow_triangle()
        shadow_offset = _shape_shadow_offset(vertices, _shape_edges(vertices))
        if shadow_offset is None:
            return False

    shape_cells = _shape_cells(vertices)
    shape_h, shape_w = _shape_bounds(vertices, shape_cells)
    if shape_h > h or shape_w > w:
        return False

    candidates = [(base_r, base_c) for base_r in range(h - shape_h + 1) for base_c in range(w - shape_w + 1)]
    if not candidates:
        return False
    order = np.random.permutation(len(candidates))

    for idx in order:
        base_r, base_c = candidates[int(idx)]
        if require_wrap and shadow_offset is not None:
            if not _placement_wraps(base_r, base_c, shape_h, shape_w, shadow_offset, h, w):
                continue

        global_vertices = _globalize_list(vertices, base_r, base_c)
        global_shape_cells = _globalize(shape_cells, base_r, base_c)
        bbox = bbox_coords(np.array(global_vertices, dtype=int))

        input_cells = set(global_vertices)
        output_cells = set(global_shape_cells)

        marker = None
        shadow_only: set[tuple[int, int]] = set()
        if shadow_color is not None and shadow_offset is not None:
            shadow_cells = _shift_toroidal(global_shape_cells, shadow_offset, h, w)
            shadow_only = shadow_cells - global_shape_cells
            if not shadow_only:
                continue
            if require_wrap and not _has_wrap(global_shape_cells, shadow_offset, h, w):
                continue
            if forbid_wrap and _has_wrap(global_shape_cells, shadow_offset, h, w):
                continue
            marker = _select_shadow_marker(shadow_only, global_vertices)
            if marker is None:
                continue
            input_cells.add(marker)
            output_cells |= shadow_only

        if _has_toroidal_contact(input_cells, occupied_in, h, w):
            continue
        if _has_toroidal_contact(output_cells, occupied_out, h, w):
            continue
        if _inside_any_bbox(input_cells, shape_bboxes):
            continue
        if _any_inside_bbox(occupied_in, bbox):
            continue

        for r, c in global_vertices:
            inp[r, c] = int(shape_color)
        for r, c in global_shape_cells:
            out[r, c] = int(shape_color)
        if marker is not None and shadow_color is not None:
            inp[marker] = int(shadow_color)
            for r, c in shadow_only:
                if out[r, c] == BG:
                    out[r, c] = int(shadow_color)

        occupied_in.update(input_cells)
        occupied_out.update(output_cells)
        shape_bboxes.append(tuple(map(int, bbox)))
        return True

    return False


def _try_place_distractor(
    inp: np.ndarray,
    out: np.ndarray,
    occupied_in: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
    shape_bboxes: list[tuple[int, int, int, int]],
    color: int,
) -> bool:
    h, w = inp.shape
    patch = _make_distractor_patch(color)
    ph, pw = patch.shape
    if ph > h or pw > w:
        return False

    local_cells = {tuple(map(int, p)) for p in np.argwhere(patch != BG)}
    candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    if not candidates:
        return False
    order = np.random.permutation(len(candidates))

    for idx in order[:DISTRACTOR_PLACEMENT_RETRIES]:
        base_r, base_c = candidates[int(idx)]
        cells = _globalize(local_cells, base_r, base_c)
        if _has_toroidal_contact(cells, occupied_in, h, w):
            continue
        if _has_toroidal_contact(cells, occupied_out, h, w):
            continue
        if _inside_any_bbox(cells, shape_bboxes):
            continue

        for r, c in cells:
            inp[r, c] = int(color)
            out[r, c] = int(color)
        occupied_in.update(cells)
        occupied_out.update(cells)
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)
    occupied_in: set[tuple[int, int]] = set()
    occupied_out: set[tuple[int, int]] = set()
    shape_bboxes: list[tuple[int, int, int, int]] = []

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    wrapped_shape_idx = int(np.random.choice(list(shadow_map.keys())))
    nonwrapped_shadow_choices = [idx for idx in shadow_map if idx != wrapped_shape_idx]
    if not nonwrapped_shadow_choices:
        return None
    nonwrapped_shadow_idx = int(np.random.choice(nonwrapped_shadow_choices))

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    order = [wrapped_shape_idx] + [i for i in range(shape_count) if i != wrapped_shape_idx]

    for idx in order:
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            if _try_place_shape(
                inp,
                out,
                occupied_in,
                occupied_out,
                shape_bboxes,
                int(shape_colors[idx]),
                shadow_map.get(idx),
                require_wrap=(idx == wrapped_shape_idx),
                forbid_wrap=(idx == nonwrapped_shadow_idx),
            ):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place_distractor(inp, out, occupied_in, occupied_out, shape_bboxes, int(distractor_color)):
            return None

    if int(get_bg(inp)) != BG or int(get_bg(out)) != BG:
        return None
    if len(colors_present(inp, exclude={BG})) != total_colors:
        return None
    if sum(1 for idx in range(shape_count) if idx not in shadow_map) < 1:
        return None
    if np.array_equal(_main_clipped_orientation_reference(inp), out):
        return None
    if np.array_equal(_solve_anchor_toroidal_reference(inp), out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
