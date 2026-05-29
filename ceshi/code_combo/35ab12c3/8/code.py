# Combo variant of 35ab12c3 (index 8)
# combo_spec: D9/1+D15/2
# Name: 干扰项作为障碍物+路径绕行
#
# Design analysis
# ---------------
# a) Concrete behavior + trace:
#    - Base task: same-color vertices are ordered into a polyline / polygon cycle.
#      Each aligned edge is drawn only if the direct H/V/45deg segment is clear in
#      the input; one singleton neighbor near the chosen start vertex is copied as a
#      parallel "shadow".
#    - Combo task: the optional distractor color is no longer inert noise. Its 1-3
#      pixels are obstacle cells that remain visible in the output. If an intended
#      direct segment hits such an obstacle, the solver tries a small bridge route
#      `A -> A+s -> B+s -> B` with a deterministic shift `s` (scale 1 or 2, non-
#      collinear with the edge). If no bridge is legal, the edge trims to the prefix
#      before the first blocker instead of crossing it.
#    - Example: shape endpoints at (4,2) and (4,8), obstacle at (4,5).
#      Original solver: whole edge is skipped because the straight row is blocked.
#      Combo solver: tries shifts like (-1,0), (+1,0), then diagonals. If row 3 is
#      clear, output becomes (4,2)->(3,2)->(3,8)->(4,8), while the obstacle pixel at
#      (4,5) stays in place. Any carried shadow follows this bent polyline in parallel.
# b) Complexity calibration:
#    - Routing is intentionally small and repetitive: direct path first, then bridge
#      candidates with shifts of length 1 or 2. This is visibly different from the
#      original task but much more learnable than unrestricted BFS.
#    - Generator always includes exactly one obstacle color-group, forces at least
#      one routed edge, and also keeps at least one visible shadow-bearing shape so
#      the fusion is not hidden behind a mostly-original sample.
#    - The obstacle-bearing shape is a simple 2-vertex line. Other shapes stay close
#      to the base task, which keeps the changed mechanism easy to isolate.
# c) Hidden assumptions reviewed:
#    - Original assumption "any foreign interior pixel vetoes the whole edge" is
#      replaced by `_find_path_in_original(...)`.
#    - Original assumption "any different-color neighbor near the start vertex is a
#      shadow anchor" breaks once obstacle groups exist. The combo solver only treats
#      singleton colors that are 8-neighboring exactly one vertex as carried shadows.
#    - Original straight-segment shadow copy is no longer sufficient; shadows must
#      shift the entire routed polyline, not just a single segment.
#    - New assumption introduced here: true shadow colors are singleton groups, and
#      obstacle pixels are never 8-neighboring any shape vertex. The generator enforces
#      both, so the solver's shadow/obstacle disambiguation is valid.
# d) Placement legality:
#    - Each patch is constructed locally, validated by running `main()` on the patch,
#      then globally placed with a one-cell buffer using `random_free_location(...)`.
#    - The blocked-line patch is padded before cropping so scale-2 bridge routes stay
#      in-bounds. Extra obstacle pixels are sampled away from vertices and away from
#      the chosen routed polyline so they do not accidentally destroy the intended route.
# e) Visual observability:
#    - Every generated grid contains one conspicuous bent line caused by an obstacle.
#      Human solvers can compare this against untouched shapes and infer that the
#      foreign-color group is acting as a blocker rather than a decorative distractor.
# f) Visual cleanliness:
#    - Only one obstacle color-group exists per grid, with at most 3 sparse pixels.
#      Other shapes are placed with output-level padding, so routed elbows do not mix
#      with unrelated objects.
# g) Inference friendliness:
#    - No hidden color ordering is used. Colors remain arbitrary distinct tokens; the
#      observable cues are structural: singleton adjacent pixel = shadow anchor, sparse
#      non-singleton foreign group on a line corridor = obstacle group, bent output
#      polyline = routed connection.

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
DETOUR_BASES = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
ROUTE_SHIFT_SCALES = (1, 2)


def _aligned_8(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    if dr == 0 and dc != 0:
        return True
    if dc == 0 and dr != 0:
        return True
    if dr != 0 and abs(dr) == abs(dc):
        return True
    return False


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


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


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _draw_polyline_bg_only(out: np.ndarray, points: list[tuple[int, int]], col: int, bg: int) -> None:
    for i in range(len(points) - 1):
        r0, c0 = points[i]
        r1, c1 = points[i + 1]
        for j, (rr, cc) in enumerate(_segment_cells(r0, c0, r1, c1)):
            if i > 0 and j == 0:
                continue
            if out[rr, cc] == bg:
                out[rr, cc] = col


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _cycle_edges(cycle: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _color_counts(g: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(g, return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _point_is_clear(g: np.ndarray, r: int, c: int, bg: int, allow: set[int]) -> bool:
    return int(g[r, c]) == bg or int(g[r, c]) in allow


def _polyline_clear(g: np.ndarray, points: list[tuple[int, int]], bg: int, allow: set[int]) -> bool:
    h, w = g.shape
    for rr, cc in points:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
    for rr, cc in points[1:-1]:
        if not _point_is_clear(g, rr, cc, bg, allow):
            return False
    for i in range(len(points) - 1):
        r0, c0 = points[i]
        r1, c1 = points[i + 1]
        if not _aligned_8(r0, c0, r1, c1):
            return False
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow):
            return False
    return True


def _first_blocker(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> tuple[int, tuple[int, int]] | None:
    cells = list(_segment_cells(r0, c0, r1, c1))
    for idx, (rr, cc) in enumerate(cells[1:-1], start=1):
        if not _point_is_clear(g, rr, cc, bg, allow):
            return idx, (rr, cc)
    return None


def _candidate_detour_shifts(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    drs, dcs = _step_dir(r0, c0, r1, c1)
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for scale in ROUTE_SHIFT_SCALES:
        for shift in [(-dcs * scale, drs * scale), (dcs * scale, -drs * scale)]:
            if shift != (0, 0) and shift not in seen:
                out.append(shift)
                seen.add(shift)
        for br, bc in DETOUR_BASES:
            shift = (br * scale, bc * scale)
            if drs * shift[1] - dcs * shift[0] == 0:
                continue
            if shift not in seen:
                out.append(shift)
                seen.add(shift)
    return out


def _find_path_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> list[tuple[int, int]] | None:
    if not _aligned_8(r0, c0, r1, c1):
        return None
    if _clear_in_original(g, r0, c0, r1, c1, bg, allow):
        return [(r0, c0), (r1, c1)]

    for drs, dcs in _candidate_detour_shifts(r0, c0, r1, c1):
        path = [(r0, c0), (r0 + drs, c0 + dcs), (r1 + drs, c1 + dcs), (r1, c1)]
        if _polyline_clear(g, path, bg, allow):
            return path

    block = _first_blocker(g, r0, c0, r1, c1, bg, allow)
    if block is None:
        return None
    cells = list(_segment_cells(r0, c0, r1, c1))
    prev = cells[block[0] - 1]
    if prev == (r0, c0):
        return None
    return [(r0, c0), prev]


def _shift_polyline(points: list[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for r, c in points]


def _is_shadow_anchor(
    g: np.ndarray,
    rr: int,
    cc: int,
    col: int,
    bg: int,
    vertices: list[tuple[int, int]],
    color_counts: dict[int, int],
) -> bool:
    v = int(g[rr, cc])
    if v == bg or v == col or color_counts.get(v, 0) != 1:
        return False
    hits = sum(1 for vr, vc in vertices if max(abs(rr - vr), abs(cc - vc)) == 1)
    return hits == 1


def _ordered_cycle_for_color(
    g: np.ndarray,
    col: int,
    bg: int,
    pts_rc: list[tuple[int, int]],
    color_counts: dict[int, int],
) -> list[tuple[int, int]]:
    cycle = _angle_cycle(pts_rc)
    if len(cycle) < 2:
        return cycle

    h, w = g.shape
    seed_idxs: list[int] = []
    for i, (r, c) in enumerate(cycle):
        for drn, dcn in DIR8:
            rr = r + drn
            cc = c + dcn
            if 0 <= rr < h and 0 <= cc < w and _is_shadow_anchor(g, rr, cc, col, bg, cycle, color_counts):
                seed_idxs.append(i)
                break

    if seed_idxs:
        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
    else:
        start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))
    return cycle[start:] + cycle[:start]


def _carried_neighbors(
    g: np.ndarray,
    col: int,
    bg: int,
    cycle: list[tuple[int, int]],
    color_counts: dict[int, int],
) -> list[tuple[int, int, int]]:
    h, w = g.shape
    carried: list[tuple[int, int, int]] = []
    sr0, sc0 = cycle[0]
    for drn, dcn in DIR8:
        rr = sr0 + drn
        cc = sc0 + dcn
        if 0 <= rr < h and 0 <= cc < w and _is_shadow_anchor(g, rr, cc, col, bg, cycle, color_counts):
            carried.append((drn, dcn, int(g[rr, cc])))
    return carried


def _route_stats_for_color(g: np.ndarray, col: int, bg: int) -> tuple[int, int, int]:
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return 0, 0, 0
    color_counts = _color_counts(g)
    cycle = _ordered_cycle_for_color(g, col, bg, pts_rc, color_counts)
    blocked = 0
    detoured = 0
    trimmed = 0
    for (r0, c0), (r1, c1) in _cycle_edges(cycle):
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
            continue
        blocked += 1
        path = _find_path_in_original(g, r0, c0, r1, c1, bg, allow={col})
        if path is None:
            continue
        if path[-1] == (r1, c1) and len(path) > 2:
            detoured += 1
        elif path[-1] != (r1, c1):
            trimmed += 1
    return blocked, detoured, trimmed


def _count_routed_edges(g: np.ndarray) -> int:
    bg = _bg_color(g)
    total = 0
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        _blocked, detoured, _trimmed = _route_stats_for_color(g, col, bg)
        total += detoured
    return total


def _has_visible_shadow_anchor(g: np.ndarray) -> bool:
    bg = _bg_color(g)
    color_counts = _color_counts(g)
    h, w = g.shape
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue
        for r, c in pts_rc:
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w and _is_shadow_anchor(g, rr, cc, col, bg, pts_rc, color_counts):
                    return True
    return False


# 1. Overview:
#    Connect each same-color vertex cycle as in the base task, but treat the distractor
#    color-group as a sparse obstacle set that forces route detours instead of deleting
#    the blocked edge outright.
# 2. Input structure:
#    Several shape colors appear as isolated vertices; some shapes have one singleton
#    neighboring shadow-anchor color; one extra sparse color-group (1-3 pixels) acts as
#    obstacles and remains unchanged in the output.
# 3. Rule:
#    Draw each aligned edge directly if clear; otherwise try a small bridge polyline
#    `A -> A+s -> B+s -> B` using deterministic non-collinear shifts `s`. If no route
#    works, trim at the last clear cell before the first blocker. Then shift the actual
#    drawn polyline by each carried shadow offset and paint that copy if it stays clear.
# 4. Key insight:
#    The foreign-color singleton next to a vertex is a shadow anchor, but the sparse
#    foreign-color group on a line corridor is an obstacle set; blocked edges bend around
#    it, and the shadow bends in parallel with the chosen route.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    out = g.copy()

    color_counts = _color_counts(g)
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _ordered_cycle_for_color(g, col, bg, pts_rc, color_counts)
        if len(cycle) < 2:
            continue

        carried = _carried_neighbors(g, col, bg, cycle, color_counts)
        drawable_paths: list[list[tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in _cycle_edges(cycle):
            path = _find_path_in_original(g, r0, c0, r1, c1, bg, allow={col})
            if path is not None:
                drawable_paths.append(path)

        for path in drawable_paths:
            _draw_polyline_bg_only(out, path, col, bg)

        for path in drawable_paths:
            r0, c0 = path[0]
            r1, c1 = path[-1]
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                shifted = _shift_polyline(path, drn, dcn)
                if not _polyline_clear(g, shifted, bg, allow=set()):
                    continue
                _draw_polyline_bg_only(out, shifted, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

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

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

BLOCKED_PATCH_RETRIES = 160
OBSTACLE_GROUP_RANGE = (1, 4)
OBSTACLE_EXTRA_RETRIES = 80
LOCAL_ROUTE_PAD = 4
BLOCKED_LINE_LEN_RANGE = (6, 11)
SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


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
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_diamond() -> list[tuple[int, int]]:
    arm = int(np.random.randint(*DIAMOND_ARM_RANGE))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _build_pentagon() -> list[tuple[int, int]]:
    roof = int(np.random.randint(*PENTAGON_ROOF_RANGE))
    wall = int(np.random.randint(*PENTAGON_WALL_RANGE))
    h = roof + wall
    return _transform_coords([(0, roof), (roof, 0), (h, 0), (h, 2 * roof), (roof, 2 * roof)])


def _build_hexagon() -> list[tuple[int, int]]:
    slant = int(np.random.randint(*HEXAGON_SLANT_RANGE))
    wall = int(np.random.randint(*HEXAGON_WALL_RANGE))
    mid = slant + wall
    bot = 2 * slant + wall
    return _transform_coords([(0, slant), (slant, 0), (mid, 0), (bot, slant), (mid, 2 * slant), (slant, 2 * slant)])


_SHAPE_BUILDERS = [_build_line, _build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return _cycle_edges(vertices)


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
    margin: int = 0,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0 - margin), int(c0 - margin), int(r1 + margin), int(c1 + margin)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    dr, dc = offset
    h, w = out.shape
    src = out == shape_color
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _crop_pair(inp: np.ndarray, out: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = np.argwhere((inp != BG) | (out != BG))
    r0, c0, r1, c1 = bbox_coords(coords)
    return inp[r0 : r1 + 1, c0 : c1 + 1].copy(), out[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_plain_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        _paint_shadow(out, shape_color, int(shadow_color), offset)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)
        nb_hits = sum(1 for rr, cc in neighbors8(anchor[0], anchor[1]) if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color)
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _polyline_cells(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for i in range(len(points) - 1):
        seg = list(_segment_cells(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1]))
        if i > 0:
            seg = seg[1:]
        cells.update(seg)
    return cells


def _sample_extra_obstacles(
    inp: np.ndarray,
    endpoints: list[tuple[int, int]],
    obstacle_cells: list[tuple[int, int]],
    routed_path: list[tuple[int, int]],
    needed_total: int,
) -> list[tuple[int, int]] | None:
    if needed_total <= len(obstacle_cells):
        return obstacle_cells

    h, w = inp.shape
    forbidden = _polyline_cells(routed_path)
    direct = _polyline_cells([endpoints[0], endpoints[1]])
    forbidden.update(direct)
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if inp[r, c] != BG:
                continue
            if (r, c) in forbidden:
                continue
            if any(max(abs(r - vr), abs(c - vc)) <= 1 for vr, vc in endpoints):
                continue
            candidates.append((r, c))

    np.random.shuffle(candidates)
    chosen = list(obstacle_cells)
    for r, c in candidates:
        if any(_aligned(r - rr, c - cc) for rr, cc in chosen):
            continue
        chosen.append((r, c))
        if len(chosen) == needed_total:
            return chosen
    return None


def _make_obstacle_line_patch(shape_color: int, obstacle_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    directions = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
    for _ in range(BLOCKED_PATCH_RETRIES):
        length = int(np.random.randint(*BLOCKED_LINE_LEN_RANGE))
        drs, dcs = directions[int(np.random.randint(len(directions)))]
        blocker_step = int(np.random.randint(2, length - 1))

        raw_pts = np.array(
            [
                (0, 0),
                (drs * length, dcs * length),
                (drs * blocker_step, dcs * blocker_step),
            ],
            dtype=int,
        )
        rmin, cmin = raw_pts.min(axis=0)
        rmax, cmax = raw_pts.max(axis=0)
        h = int(rmax - rmin + 1 + 2 * LOCAL_ROUTE_PAD)
        w = int(cmax - cmin + 1 + 2 * LOCAL_ROUTE_PAD)
        shift = np.array([LOCAL_ROUTE_PAD - rmin, LOCAL_ROUTE_PAD - cmin], dtype=int)
        pts = raw_pts + shift

        inp = np.full((h, w), BG, dtype=int)
        a = tuple(map(int, pts[0]))
        b = tuple(map(int, pts[1]))
        blocker = tuple(map(int, pts[2]))
        inp[a] = shape_color
        inp[b] = shape_color
        inp[blocker] = obstacle_color

        routed = _find_path_in_original(inp, a[0], a[1], b[0], b[1], BG, allow={shape_color})
        if routed is None or routed[-1] != b or len(routed) <= 2:
            continue

        obstacle_total = int(np.random.randint(*OBSTACLE_GROUP_RANGE))
        obstacle_cells = [blocker]
        if obstacle_total > 1:
            extra = _sample_extra_obstacles(inp, [a, b], obstacle_cells, routed, obstacle_total)
            if extra is None:
                continue
            obstacle_cells = extra

        for r, c in obstacle_cells[1:]:
            inp[r, c] = obstacle_color

        blocked, detoured, trimmed = _route_stats_for_color(inp, shape_color, BG)
        if blocked < 1 or detoured < 1 or trimmed != 0:
            continue

        out = main(inp)
        if np.array_equal(out, inp):
            continue
        if len(obstacle_cells) > 3:
            continue
        if any(max(abs(r - a[0]), abs(c - a[1])) <= 1 or max(abs(r - b[0]), abs(c - b[1])) <= 1 for r, c in obstacle_cells):
            continue
        ok_pairs = True
        for i in range(len(obstacle_cells)):
            for j in range(i + 1, len(obstacle_cells)):
                if _aligned(obstacle_cells[j][0] - obstacle_cells[i][0], obstacle_cells[j][1] - obstacle_cells[i][1]):
                    ok_pairs = False
        if not ok_pairs:
            continue
        return _crop_pair(inp, out)
    return None


def _sample_palette(shape_count: int) -> tuple[list[int], list[int], int] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count - 1, len(COLORS) - shape_count - 1)
    if max_shadows < 0:
        return None
    shadow_lo = 1 if max_shadows >= 1 else 0
    shadow_count = int(np.random.randint(shadow_lo, max_shadows + 1))
    total = shape_count + shadow_count + 1
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
    obstacle_color = int(palette[-1])
    return shape_colors, shadow_colors, obstacle_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(list(range(1, shape_count)), size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    try:
        loc = random_free_location(
            out,
            patch_out,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(out, patch_out, loc[0], loc[1], transparent=BG)
    paste(inp, patch_in, loc[0], loc[1], transparent=BG)
    return True


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    palette = _sample_palette(shape_count)
    if palette is None:
        return None
    shape_colors, shadow_colors, obstacle_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    total_colors = shape_count + len(shadow_colors) + 1

    blocked_patch = _make_obstacle_line_patch(int(shape_colors[0]), obstacle_color)
    if blocked_patch is None or not _try_place(inp, out, *blocked_patch):
        return None

    for idx, shape_color in enumerate(shape_colors[1:], start=1):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_plain_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if _count_routed_edges(inp) < 1:
        return None
    if not _has_visible_shadow_anchor(inp):
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
