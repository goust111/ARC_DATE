# Variant of 35ab12c3
# Dimension: D12 — 网格形状/输出裁剪 (grid_shape)
# Change: same——输出网格与输入尺寸完全相同（15-22行 x 15-22列）... → 网格扩展以容纳阴影
#
# Design analysis for this variant
# --------------------------------
# a) Concrete behavior / trace:
#    The original task already knows how to infer a shadow offset from one
#    different-colored 8-neighbor next to the chosen start vertex, then copy the
#    whole connected shape along that offset. The old solver simply dropped any
#    shifted edge whose endpoint left the input grid. In this variant the logical
#    shape drawing rule is unchanged, but the output canvas is allowed to grow by
#    one row/col on any side needed by the shadow copy.
#    Small trace:
#      - Input contains a top-border polyline whose shadow color sits one cell
#        down-left of the start vertex.
#      - Old logic: the shadowed parallel edge would cross column -1, so that edge
#        is skipped and output stays HxW.
#      - New logic: draw on a padded workspace, keep in-bounds checks against the
#        original input for collision testing, then return the original HxW area
#        plus exactly the extra border columns/rows that received shadow pixels.
#    Difference: only the final canvas extent changes; connectivity, color roles,
#    vertex ordering, and shadow priority are preserved.
#
# b) Complexity calibration:
#    Expansion is limited to at most one cell per side because the carried shadow
#    offset is always an 8-neighbor delta. That keeps the rule visible but not
#    combinatorially messy. The generator still uses 2-4 shapes, unique colors,
#    optional distractors, and 1-cell spacing. To make D12 observable, the
#    generator now guarantees at least one shadow and at least one real spill.
#
# c) Implicit-assumption audit of the original main():
#    - `out = g.copy()` assumed output shape == input shape. Invalid now.
#    - Shadow endpoint bounds check assumed out-of-bounds shadow segments should be
#      discarded. Invalid now; they must be drawn on an expanded canvas.
#    - `_clear_in_original(...)` indexed only in-bounds segment cells. For spilled
#      shadow segments, some checked cells may sit outside the original grid; they
#      should be treated as clear background rather than causing an index error.
#    - Main-shape edge drawing itself still relies on original-grid coordinates and
#      remains valid because vertices never leave the input.
#
# d) Placement legality:
#    Generator placement is checked against a hidden `(H+2) x (W+2)` workspace
#    whose central `H x W` window is the input canvas. Each object's local output
#    patch may extend by one side-cell beyond its local input patch. We only place
#    objects if:
#      - the input patch stays fully inside the original canvas,
#      - the corresponding output patch stays inside the hidden padded workspace,
#      - non-background output cells keep a >=1-cell 8-neighborhood buffer from
#        previously placed objects.
#
# e) Visual observability:
#    Every generated sample has at least one shadowed shape whose inferred shadow
#    really crosses an input border, so the output size change is visible. The side
#    set varies across samples (top/bottom/left/right, sometimes two sides), which
#    is enough for a human to infer "expand wherever the shadow would continue".
#
# f) Cleanliness:
#    The generator avoids unintended extra spills by forcing non-required spill
#    sides away from the corresponding canvas border when possible. This keeps the
#    expanded border signal isolated and readable instead of creating crowded
#    multi-side growth accidentally.
#
# g) Color inference:
#    No new color mapping is introduced. Shape colors and shadow colors still act
#    exactly as in the base task, so D12 is carried entirely by geometry/cropping,
#    not by hidden color-id conventions.
#
# h) New solver assumptions introduced here:
#    - Any needed expansion comes only from shadow copies, so one-cell padding on
#      each side is sufficient.
#    - Input colors remain unique per group as in the original generator family.
#    - A shadow color is still represented in the input by a single anchor pixel,
#      so it never becomes a separate connectable multi-point shape.
#    The generator explicitly preserves these assumptions.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, paste
from arc_utils import get_bg as _bg_color

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CARDINAL_SIDES = ("top", "bottom", "left", "right")


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


def _clear_in_input(g: np.ndarray, r0: int, c0: int, r1: int, c1: int, bg: int, allow: set[int]) -> bool:
    h, w = g.shape
    first = True
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        if first:
            first = False
            continue
        if rr == r1 and cc == c1:
            break
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v not in allow:
                return False
    return True


def _draw_line_bg_only(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int, bg: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
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


def _workspace_crop_bounds(workspace: np.ndarray, h: int, w: int, bg: int) -> tuple[tuple[int, int, int, int], set[str]]:
    sides: set[str] = set()
    if np.any(workspace[0] != bg):
        sides.add("top")
    if np.any(workspace[h + 1] != bg):
        sides.add("bottom")
    if np.any(workspace[:, 0] != bg):
        sides.add("left")
    if np.any(workspace[:, w + 1] != bg):
        sides.add("right")

    r0 = 0 if "top" in sides else 1
    r1 = h + 2 if "bottom" in sides else h + 1
    c0 = 0 if "left" in sides else 1
    c1 = w + 2 if "right" in sides else w + 1
    return (r0, c0, r1, c1), sides


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, set[str]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    workspace = np.full((h + 2, w + 2), bg, dtype=int)
    workspace[1 : h + 1, 1 : w + 1] = g

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        seed_idxs = []
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
            if not _clear_in_input(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(workspace, r0 + 1, c0 + 1, r1 + 1, c1 + 1, col, bg)

        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = r0 + drn
                cc0 = c0 + dcn
                rr1 = r1 + drn
                cc1 = c1 + dcn
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_input(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                wr0 = rr0 + 1
                wc0 = cc0 + 1
                wr1 = rr1 + 1
                wc1 = cc1 + 1
                if not (0 <= wr0 < h + 2 and 0 <= wc0 < w + 2 and 0 <= wr1 < h + 2 and 0 <= wc1 < w + 2):
                    continue
                _draw_line_bg_only(workspace, wr0, wc0, wr1, wc1, ncol, bg)

    (r0, c0, r1, c1), sides = _workspace_crop_bounds(workspace, h, w, bg)
    return workspace[r0:r1, c0:c1], sides


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_meta(np.asarray(grid, dtype=int))[0]


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
MIN_SHADOWS = 1
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1

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

PATCH_BUILD_RETRIES = 80
GENERATE_RETRIES = 500


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


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    directions = _edge_dirs(vertices)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for anchor in vertices:
        for dr, dc in DIR8:
            if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
                continue
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in shape_cells:
                continue
            if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
                continue
            candidates.append((anchor, (dr, dc)))
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _build_shape_patch(shape_color: int, shadow_color: int | None, required_side: str | None):
    for _ in range(PATCH_BUILD_RETRIES):
        vertices = _make_vertices()
        shape_cells = _shape_cells(vertices)

        shadow = None
        if shadow_color is not None:
            shadow = _choose_shadow(vertices, shape_cells)
            if shadow is None:
                continue

        input_coords = list(vertices)
        if shadow is not None:
            (ar, ac), (dr, dc) = shadow
            shadow_anchor = (ar + dr, ac + dc)
            if shadow_anchor in shape_cells:
                continue
            input_coords.append(shadow_anchor)

        input_arr = np.array(input_coords, dtype=int)
        in_r0, in_c0, in_r1, in_c1 = bbox_coords(input_arr)
        in_h = int(in_r1 - in_r0 + 1)
        in_w = int(in_c1 - in_c0 + 1)

        out_cells = set(shape_cells)
        if shadow is not None:
            (_ar, _ac), (dr, dc) = shadow
            out_cells.update((r + dr, c + dc) for r, c in shape_cells)

        out_arr = np.array(list(out_cells), dtype=int)
        out_r0, out_c0, out_r1, out_c1 = bbox_coords(out_arr)
        extras = {
            "top": int(max(0, in_r0 - out_r0)),
            "bottom": int(max(0, out_r1 - in_r1)),
            "left": int(max(0, in_c0 - out_c0)),
            "right": int(max(0, out_c1 - in_c1)),
        }
        if required_side is not None and extras[required_side] != 1:
            continue

        out_h = in_h + extras["top"] + extras["bottom"]
        out_w = in_w + extras["left"] + extras["right"]

        inp = np.full((in_h, in_w), BG, dtype=int)
        out = np.full((out_h, out_w), BG, dtype=int)

        for r, c in vertices:
            inp[r - in_r0, c - in_c0] = int(shape_color)

        if shadow is not None:
            (ar, ac), (dr, dc) = shadow
            sa = (ar + dr - in_r0, ac + dc - in_c0)
            if not (0 <= sa[0] < in_h and 0 <= sa[1] < in_w) or inp[sa] != BG:
                continue
            inp[sa] = int(shadow_color)

        if shadow is not None:
            (_ar, _ac), (dr, dc) = shadow
            for r, c in shape_cells:
                rr = r + dr - in_r0 + extras["top"]
                cc = c + dc - in_c0 + extras["left"]
                if out[rr, cc] == BG:
                    out[rr, cc] = int(shadow_color)
        for r, c in shape_cells:
            rr = r - in_r0 + extras["top"]
            cc = c - in_c0 + extras["left"]
            out[rr, cc] = int(shape_color)

        if not np.array_equal(main(inp), out):
            continue

        return {
            "patch_in": inp,
            "patch_out": out,
            "extras": extras,
            "mask": out != BG,
            "required_side": required_side,
            "has_shadow": shadow is not None,
        }
    return None


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


def _make_distractor_patch(color: int):
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return {
            "patch_in": patch.copy(),
            "patch_out": patch.copy(),
            "extras": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "mask": patch != BG,
            "required_side": None,
            "has_shadow": False,
        }
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return {
            "patch_in": patch.copy(),
            "patch_out": patch.copy(),
            "extras": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "mask": patch != BG,
            "required_side": None,
            "has_shadow": False,
        }
    patch = np.full((1, 1), color, dtype=int)
    return {
        "patch_in": patch.copy(),
        "patch_out": patch.copy(),
        "extras": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        "mask": patch != BG,
        "required_side": None,
        "has_shadow": False,
    }


def _sample_palette(shape_count: int, has_distractor: bool, min_shadows: int) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < min_shadows:
        return None
    shadow_count = int(np.random.randint(min_shadows, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _choose_required_sides(shadow_count: int) -> list[str]:
    max_side_count = min(2, shadow_count)
    count = int(np.random.randint(1, max_side_count + 1))
    sides = list(np.random.choice(CARDINAL_SIDES, size=count, replace=False))
    return [str(s) for s in sides]


def _position_values(total: int, size: int, must_low: bool, must_high: bool, avoid_low: bool, avoid_high: bool) -> list[int]:
    if size > total:
        return []
    hi_edge = total - size
    if must_low and must_high:
        return [0] if hi_edge == 0 else []
    if must_low:
        return [0]
    if must_high:
        return [hi_edge]
    lo = 1 if avoid_low else 0
    hi = hi_edge - 1 if avoid_high else hi_edge
    if hi < lo:
        return []
    vals = list(range(lo, hi + 1))
    if vals:
        vals = list(np.random.permutation(vals))
    return [int(v) for v in vals]


def _can_stamp_with_padding(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    h, w = occ.shape
    for rr, cc in np.argwhere(mask):
        ar = int(r0 + rr)
        ac = int(c0 + cc)
        r1 = max(0, ar - padding)
        r2 = min(h, ar + padding + 1)
        c1 = max(0, ac - padding)
        c2 = min(w, ac + padding + 1)
        if np.any(occ[r1:r2, c1:c2]):
            return False
    return True


def _find_placement(inp: np.ndarray, out_ws: np.ndarray, obj: dict) -> tuple[int, int] | None:
    H, W = inp.shape
    in_h, in_w = obj["patch_in"].shape
    out_h, out_w = obj["patch_out"].shape
    extras = obj["extras"]
    req = obj["required_side"]

    r_vals = _position_values(
        H,
        in_h,
        must_low=req == "top",
        must_high=req == "bottom",
        avoid_low=extras["top"] == 1 and req != "top",
        avoid_high=extras["bottom"] == 1 and req != "bottom",
    )
    c_vals = _position_values(
        W,
        in_w,
        must_low=req == "left",
        must_high=req == "right",
        avoid_low=extras["left"] == 1 and req != "left",
        avoid_high=extras["right"] == 1 and req != "right",
    )
    if not r_vals or not c_vals:
        return None

    occ = out_ws != BG
    valid: list[tuple[int, int]] = []
    for r in r_vals:
        for c in c_vals:
            out_r = int(r - extras["top"] + 1)
            out_c = int(c - extras["left"] + 1)
            if out_r < 0 or out_c < 0 or out_r + out_h > H + 2 or out_c + out_w > W + 2:
                continue
            if not _can_stamp_with_padding(occ, obj["mask"], out_r, out_c, PLACEMENT_PADDING):
                continue
            valid.append((int(r), int(c)))
    if not valid:
        return None
    idx = int(np.random.randint(len(valid)))
    return valid[idx]


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _workspace_to_output(workspace: np.ndarray, h: int, w: int) -> tuple[np.ndarray, set[str]]:
    (r0, c0, r1, c1), sides = _workspace_crop_bounds(workspace, h, w, BG)
    return workspace[r0:r1, c0:c1].copy(), sides


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    inp = np.full((H, W), BG, dtype=int)
    out_ws = np.full((H + 2, W + 2), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor, min_shadows=MIN_SHADOWS)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette

    shadow_targets = sorted(np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist())
    shadow_map = {int(idx): int(col) for idx, col in zip(shadow_targets, shadow_colors)}
    required_sides = _choose_required_sides(len(shadow_colors))
    required_assign = {int(idx): side for idx, side in zip(shadow_targets[: len(required_sides)], required_sides)}

    objects: list[dict] = []
    for idx, shape_color in enumerate(shape_colors):
        obj = _build_shape_patch(int(shape_color), shadow_map.get(idx), required_assign.get(idx))
        if obj is None:
            return None
        objects.append(obj)

    if has_distractor and distractor_color is not None:
        objects.append(_make_distractor_patch(int(distractor_color)))

    objects.sort(key=lambda obj: (obj["required_side"] is None, -(obj["patch_out"].shape[0] * obj["patch_out"].shape[1])))

    for obj in objects:
        loc = _find_placement(inp, out_ws, obj)
        if loc is None:
            return None
        r, c = loc
        paste(inp, obj["patch_in"], r, c, transparent=BG)
        out_r = int(r - obj["extras"]["top"] + 1)
        out_c = int(c - obj["extras"]["left"] + 1)
        paste(out_ws, obj["patch_out"], out_r, out_c, transparent=BG)

    expected_out, expected_sides = _workspace_to_output(out_ws, H, W)
    actual_out, actual_sides = _solve_with_meta(inp)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    if get_bg(inp) != BG or get_bg(expected_out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if len(colors_present(expected_out, exclude=[BG])) != total_colors:
        return None
    if actual_sides != set(required_sides):
        return None
    if expected_sides != set(required_sides):
        return None
    if not np.array_equal(actual_out, expected_out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
