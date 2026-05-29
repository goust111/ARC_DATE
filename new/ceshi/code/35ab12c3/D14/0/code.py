# Variant of 35ab12c3
# Dimension: D14 — 阴影锚点检测规则 (src)
# Change: 8邻居检测——在起始顶点（最小坐标的有阴影邻居的顶点）的8邻居中寻找不同颜色非背景像素作为阴影锚点，偏移量=锚点位置-顶点位置
#         → 4邻居检测（仅上下左右）
#
# Design analysis for this variant
# =================================
# a) Concrete behavior and trace
# - Original task: a shape color contributes 2-7 vertex pixels in the input. The
#   solver orders them, draws H/V/45deg edges, and if the chosen start vertex has
#   a foreign non-background neighbor in its 8-neighborhood, it also copies the
#   filled shape along that offset in the neighbor's color.
# - D14 changes only the source-side detection rule: the start vertex may only
#   read foreign pixels from its 4-neighborhood. Diagonal foreign pixels are now
#   ignored for shadow anchoring and do not create copied outlines.
# - Small trace:
#   shape vertices: {(0,0), (0,4)}  color A
#   real shadow anchor: (1,0)       color B   -> orthogonal, valid in variant
#   diagonal marker:    (1,-1)      color C   -> diagonal only, must be ignored
#   Original 8-neighbor logic would carry both offsets (1,0) and (1,-1).
#   New 4-neighbor logic carries only (1,0), so output contains the B parallel
#   line but preserves C as a single pixel. This makes the D14 restriction visible.
#
# b) Complexity calibration
# - If generator merely restricts shadows to orthogonal offsets, many samples are
#   still solved identically by the original 8-neighbor solver, so the variant is
#   visually weak and behaviorally degenerate.
# - To make the difference observable, every generated grid contains:
#   1. at least one real orthogonal shadow, and
#   2. one diagonal single-pixel decoy adjacent to the same start vertex.
#   The decoy is preserved as-is by the new solver but would incorrectly become a
#   full parallel copy under the original rule.
# - Freedom is still limited: one decoy color, 1-3 true shadows total, and the
#   same shape vocabulary as the original task except that shadow-bearing shapes
#   are sampled from shadow-compatible families.
#
# c) Main() assumption audit
# - `seed_idxs`: in the original code, scanning `DIR8` assumes diagonal foreign
#   pixels may legitimately define the start vertex. Under D14 this is false, so
#   the scan must become `DIR4`.
# - `carried`: the original code copies every foreign 8-neighbor around the chosen
#   start vertex. Under D14, diagonal offsets must not be carried, or the solver
#   would still recreate the old behavior.
# - Edge drawing, cycle ordering, line alignment, collision checks, and phase
#   ordering do not depend on the anchor neighborhood type and stay unchanged.
# - Hidden harmless-old-pattern check: under the old rule, a diagonal distractor
#   near the start vertex was interpreted as an extra shadow. That was acceptable.
#   Under D14 it must remain untouched, so the solver must not scan or carry it.
#
# d) Placement legality
# - Real shadows are generated only with 4-neighbor offsets from the anchor.
# - Shadow-bearing shapes are sampled only from families that admit orthogonal
#   non-collinear offsets (line / diamond / hexagon), otherwise generation would
#   often fail or silently collapse to no-shadow cases.
# - The diagonal decoy is placed inside the local patch only if it is:
#   1. not on the shape,
#   2. not on the true shadow copy,
#   3. 8-adjacent to exactly one vertex,
#   4. not 4-adjacent to any vertex.
# - Global placement still uses a one-cell buffer, so different groups do not
#   accidentally create cross-group anchor cues.
#
# e) Visual observability
# - Humans can infer the variant from 2-4 examples because each sample shows both
#   positive and negative evidence:
#   1. orthogonally adjacent foreign pixels do create copied shadows;
#   2. diagonally adjacent foreign pixels do not.
# - No sorting or color transformation depends on color ids. Colors remain purely
#   symbolic and only structure/adjacency matter.
#
# f) Visual cleanliness
# - The decoy is only one extra pixel, attached locally to a demonstration shape,
#   so it is easy to read and does not create clutter.
# - Because the decoy cannot overlap the true shadow copy, the output remains
#   visually separable: one copied shadow plus one untouched diagonal dot.
#
# g) Color inference
# - Every non-background color still has a structural role: shape, real shadow, or
#   a single untouched decoy pixel. The decoy's behavior is inferable from its
#   diagonal placement, not from the palette value itself.
#
# h) New main() assumptions introduced by D14
# - If a shape is intended to have a shadow, its anchor pixel must be 4-adjacent to
#   the chosen start vertex; diagonal-only anchors are ignored by design.
# - If a diagonal foreign pixel is present near a shape, it must not also be 4-
#   adjacent to another vertex, or it could become a valid anchor accidentally.
# - Generator guarantee: all real shadows satisfy the 4-neighbor condition; the
#   mandatory decoy is diagonal-only and unique around its anchor.
#
# i) Generator-vs-solver contract
# - The first shape is a demo line that always has one orthogonal shadow and one
#   diagonal decoy. This guarantees the D14 behavior is exercised on every sample.
# - Additional shapes may or may not have shadows, but any shadowed shape is built
#   from a shadow-compatible family so `main(generate_input())` remains reliable.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, paste, random_free_location

DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]
DIAG4 = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


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

        seed_idxs = []
        for i, (r, c) in enumerate(cycle):
            for drn, dcn in DIR4:
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
        for drn, dcn in DIR4:
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
                if not _aligned_8(rr0, cc0, rr1, cc1):
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
MIN_SHADOWS = 1
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
SHADOWABLE_WEIGHTS = np.array([0.28, 0.14, 0.10], dtype=float)

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


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
_SHADOWABLE_BUILDERS = [_build_line, _build_diamond, _build_hexagon]


def _make_vertices(*, shadowable: bool = False, force_line: bool = False) -> list[tuple[int, int]]:
    if force_line:
        return _build_line()
    if shadowable:
        weights = SHADOWABLE_WEIGHTS / SHADOWABLE_WEIGHTS.sum()
        idx = int(np.random.choice(len(_SHADOWABLE_BUILDERS), p=weights))
        return _SHADOWABLE_BUILDERS[idx]()
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
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR4:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if abs(pos[0] - vr) + abs(pos[1] - vc) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _choose_diag_decoy(
    vertices: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    shadow_offset: tuple[int, int] | None,
) -> tuple[int, int] | None:
    anchor = min(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIAG4:
        if shadow_offset is not None:
            sdr, sdc = shadow_offset
            # Put the decoy on the side opposite the real orthogonal shadow so the
            # old 8-neighbor rule would draw an extra parallel copy instead of
            # having it blocked immediately by the true shadow anchor.
            if sdr != 0 and dr == sdr:
                continue
            if sdc != 0 and dc == sdc:
                continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in occupied:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        if any(abs(pos[0] - vr) + abs(pos[1] - vc) == 1 for vr, vc in vertices):
            continue
        candidates.append(pos)
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
    decoy: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        _anchor, (dr, dc) = shadow
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    if decoy is not None:
        coords.append(decoy)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


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


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
    decoy_color: int | None = None,
    *,
    force_line: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices(shadowable=shadow_color is not None, force_line=force_line)
    cells = _shape_cells(vertices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None

    occupied = set(cells)
    if shadow is not None:
        _anchor, (dr, dc) = shadow
        occupied.update((r + dr, c + dc) for r, c in cells)

    decoy = None
    if decoy_color is not None:
        decoy = _choose_diag_decoy(vertices, occupied, None if shadow is None else shadow[1])
        if decoy is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow, decoy)
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
        ortho_hits = 0
        for dr, dc in DIR4:
            rr = anchor[0] + dr
            cc = anchor[1] + dc
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color:
                ortho_hits += 1
        if ortho_hits != 1:
            return None

    if decoy is not None:
        dp = (decoy[0] - r0, decoy[1] - c0)
        if out[dp] != BG or inp[dp] != BG:
            return None
        out[dp] = int(decoy_color)
        inp[dp] = int(decoy_color)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _sample_palette(shape_count: int) -> tuple[list[int], list[int], int] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - 1)
    if max_shadows < MIN_SHADOWS:
        return None
    shadow_count = int(np.random.randint(MIN_SHADOWS, max_shadows + 1))
    total = shape_count + shadow_count + 1
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
    decoy_color = int(palette[-1])
    return shape_colors, shadow_colors, decoy_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    shadow_map: dict[int, int] = {0: int(shadow_colors[0])}
    remaining_colors = [int(c) for c in shadow_colors[1:]]
    if remaining_colors:
        targets = np.random.choice(np.arange(1, shape_count), size=len(remaining_colors), replace=False).tolist()
        for idx, col in zip(sorted(int(t) for t in targets), remaining_colors):
            shadow_map[idx] = col
    return shadow_map


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
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    palette = _sample_palette(shape_count)
    if palette is None:
        return None
    shape_colors, shadow_colors, decoy_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    total_colors = shape_count + len(shadow_colors) + 1

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(
                int(shape_color),
                shadow_map.get(idx),
                decoy_color if idx == 0 else None,
                force_line=idx == 0,
            )
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
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
