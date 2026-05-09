# Variant of 35ab12c3
# Dimension: D21 — 边界处理 (meta)
# Change: 阴影超出网格边界时直接截断（不画超出部分），使用bounds check确保所有坐标在[0,H)x[0,W)内... → 环绕边界（toroidal）

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, colors_present, get_bg  # noqa: E402


# Design analysis for the D21 toroidal variant
# ============================================
# source_task: None.
#
# a) Concrete behavior and trace
#    - The base shape logic is unchanged: each non-bg color with >=2 pixels is treated
#      as a vertex set, ordered into an angle cycle, and straight H/V/45deg edges are
#      drawn between successive vertices if the original input is clear.
#    - Only the shadow layer changes. In the original task, if a carried neighbor offset
#      would move any shadow cell outside the grid, that part was dropped. Here the moved
#      cell wraps by modulo H/W.
#    - Small trace:
#        shape edge cells: (0,5),(1,5),(2,5), offset = (-1,0)
#        old shadow cells: (-1,5),(0,5),(1,5) -> keep only in-bounds -> (0,5),(1,5)
#        new shadow cells: (-1,5),(0,5),(1,5) -> modulo H -> (H-1,5),(0,5),(1,5)
#      So one extra shadow pixel appears on the opposite border.
#    - Input implication: the single shadow-anchor pixel may itself wrap across an edge,
#      so seed detection must also use toroidal 8-neighbors.
#
# b) Complexity calibration
#    - The variant would be visually invisible if all shadows stayed away from borders.
#      The generator therefore guarantees at least one shadow with offset dr=-1 whose
#      anchor sits on the top border, forcing a wrapped copy onto the bottom row.
#    - Other structure (shape count, colors, distractors, edge geometry) stays close to
#      the original to avoid introducing a second independent rule.
#
# c) Old implicit assumptions that break under D21
#    - Original seed-neighbor search assumed `0 <= rr < h and 0 <= cc < w`; a wrapped
#      anchor on the opposite side would be missed.
#    - Original carried-neighbor collection had the same bounded-neighbor assumption.
#    - Original shadow phase rejected shifted endpoints that left the grid; that was the
#      clipping rule itself and must be removed.
#    - Original shadow clearance checked a translated segment in ordinary coordinates.
#      Under D21, each shifted interior cell must be checked after modulo wrapping.
#    - Original `_draw_line_bg_only` cannot express toroidal copies because its target
#      coordinates are not wrapped.
#
# d) Placement legality
#    - The generator now places groups directly on the full canvas instead of pasting a
#      local patch, because a wrapped shadow may occupy both the main area and the far
#      edge simultaneously.
#    - Distinct groups are checked for overlap and 8-neighbor contact on the torus in
#      both input and output, preventing accidental cross-border interactions.
#
# e) Visual observability
#    - Every generated puzzle has at least one true wrap event, and the generator rejects
#      samples where the toroidal output matches the original clipped output.
#    - The wrapped anchor often appears on the opposite border in the input, giving the
#      solver a visible clue rather than a hidden metadata change.
#
# f) Visual cleanliness
#    - Inter-group toroidal padding is enforced, so wrapped shadows do not create noisy
#      border collisions with unrelated objects.
#    - Only one topological idea changes: shadow continuation across borders.
#
# g) Color inference
#    - Colors keep their original role: each group has a unique color, and a shadow color
#      is still read purely from adjacency to one anchor vertex. No color-id-dependent
#      rule is introduced.
#
# 9) New assumptions introduced by the toroidal solver
#    - If a color has a wrapped shadow-anchor, that anchor must still be the only
#      different-colored toroidal 8-neighbor of the chosen start vertex.
#    - No unrelated group may become a toroidal 8-neighbor of a shape vertex or shadow
#      path, otherwise the solver could carry the wrong offset/color.
#    - The generator guarantees both by toroidal separation checks and by validating the
#      final sample with `main(inp) == out`.


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


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


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

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

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

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


def _main_clipped_reference(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        seed_idxs: list[int] = []
        for i, (r, c) in enumerate(cycle):
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
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
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

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
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
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

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 80
DISTRACTOR_PLACEMENT_RETRIES = 80
GENERATE_RETRIES = 400


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


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
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _choose_shadow_candidates(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    *,
    require_top_wrap: bool,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for dr, dc in DIR8:
        if require_top_wrap and dr != -1:
            continue
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((anchor, (dr, dc)))
    return candidates


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
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch
    return np.full((1, 1), color, dtype=int)


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 1:
        return None
    shadow_count = int(np.random.randint(1, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _toroidal_neighbor_count(point: tuple[int, int], others: set[tuple[int, int]], h: int, w: int) -> int:
    count = 0
    r, c = point
    for dr, dc in DIR8:
        if ((r + dr) % h, (c + dc) % w) in others:
            count += 1
    return count


def _has_toroidal_contact(cells_a: set[tuple[int, int]], cells_b: set[tuple[int, int]], h: int, w: int) -> bool:
    if not cells_a or not cells_b:
        return False
    for r, c in cells_a:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if ((r + dr) % h, (c + dc) % w) in cells_b:
                    return True
    return False


def _shape_bounds(vertices: list[tuple[int, int]], shape_cells: set[tuple[int, int]]) -> tuple[int, int]:
    coords = list(vertices) + list(shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _globalize(coords: set[tuple[int, int]] | list[tuple[int, int]], base_r: int, base_c: int) -> set[tuple[int, int]]:
    return {(int(r) + base_r, int(c) + base_c) for r, c in coords}


def _globalize_list(coords: list[tuple[int, int]], base_r: int, base_c: int) -> list[tuple[int, int]]:
    return [(int(r) + base_r, int(c) + base_c) for r, c in coords]


def _shift_toroidal(coords: set[tuple[int, int]], offset: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    dr, dc = offset
    return {((r + dr) % h, (c + dc) % w) for r, c in coords}


def _placement_wraps(base_r: int, base_c: int, shape_h: int, shape_w: int, offset: tuple[int, int], h: int, w: int) -> bool:
    dr, dc = offset
    if dr == -1 and base_r == 0:
        return True
    if dr == 1 and base_r == h - shape_h:
        return True
    if dc == -1 and base_c == 0:
        return True
    if dc == 1 and base_c == w - shape_w:
        return True
    return False


def _try_place_shape(
    inp: np.ndarray,
    out: np.ndarray,
    occupied_in: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
    shape_color: int,
    shadow_color: int | None,
    *,
    require_wrap: bool,
) -> bool:
    h, w = inp.shape
    vertices = _make_vertices()
    shape_cells = _shape_cells(vertices)
    shape_h, shape_w = _shape_bounds(vertices, shape_cells)
    if shape_h > h or shape_w > w:
        return False

    shadow_options: list[tuple[tuple[int, int], tuple[int, int]] | None]
    if shadow_color is None:
        shadow_options = [None]
    else:
        shadow_options = _choose_shadow_candidates(vertices, shape_cells, require_top_wrap=require_wrap)
        if not shadow_options:
            return False
        order = np.random.permutation(len(shadow_options))
        shadow_options = [shadow_options[int(i)] for i in order]

    for shadow in shadow_options:
        r_vals = range(h - shape_h + 1)
        c_vals = range(w - shape_w + 1)
        candidates: list[tuple[int, int]] = []
        for base_r in r_vals:
            for base_c in c_vals:
                if require_wrap and shadow is not None:
                    if base_r != 0:
                        continue
                    if not _placement_wraps(base_r, base_c, shape_h, shape_w, shadow[1], h, w):
                        continue
                candidates.append((base_r, base_c))
        if not candidates:
            continue
        order = np.random.permutation(len(candidates))
        for idx in order:
            base_r, base_c = candidates[int(idx)]
            global_vertices = _globalize_list(vertices, base_r, base_c)
            vertex_set = set(global_vertices)
            global_shape_cells = _globalize(shape_cells, base_r, base_c)
            input_cells = set(vertex_set)
            output_cells = set(global_shape_cells)

            shadow_anchor = None
            shadow_only: set[tuple[int, int]] = set()
            if shadow is not None and shadow_color is not None:
                (anchor_r, anchor_c), offset = shadow
                shadow_anchor = _wrap_rc(h, w, anchor_r + base_r + offset[0], anchor_c + base_c + offset[1])
                if shadow_anchor in input_cells:
                    continue
                if _toroidal_neighbor_count(shadow_anchor, vertex_set, h, w) != 1:
                    continue
                shadow_cells = _shift_toroidal(global_shape_cells, offset, h, w)
                shadow_only = shadow_cells - global_shape_cells
                input_cells.add(shadow_anchor)
                output_cells |= shadow_only
                if require_wrap and not shadow_only:
                    continue

            if _has_toroidal_contact(input_cells, occupied_in, h, w):
                continue
            if _has_toroidal_contact(output_cells, occupied_out, h, w):
                continue

            for r, c in global_vertices:
                inp[r, c] = int(shape_color)
            for r, c in global_shape_cells:
                out[r, c] = int(shape_color)
            if shadow_anchor is not None and shadow_color is not None:
                inp[shadow_anchor] = int(shadow_color)
                for r, c in shadow_only:
                    if out[r, c] == BG:
                        out[r, c] = int(shadow_color)

            occupied_in.update(input_cells)
            occupied_out.update(output_cells)
            return True

    return False


def _try_place_distractor(
    inp: np.ndarray,
    out: np.ndarray,
    occupied_in: set[tuple[int, int]],
    occupied_out: set[tuple[int, int]],
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

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    wrapped_shape_idx = int(np.random.choice(list(shadow_map.keys())))

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            if _try_place_shape(
                inp,
                out,
                occupied_in,
                occupied_out,
                int(shape_color),
                shadow_map.get(idx),
                require_wrap=(idx == wrapped_shape_idx),
            ):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place_distractor(inp, out, occupied_in, occupied_out, int(distractor_color)):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if np.array_equal(_main_clipped_reference(inp), out):
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
