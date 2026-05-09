# Variant of 35ab12c3
# Dimension: D7 — 源对象——形状顶点来源 (src)
# Change: 同色像素组作为多边形/折线顶点，每组2-7个同色像素，形状类型包括折线、三角形、矩形、菱形、五边形、六边形... → 模板图案中的关键像素作为源
#
# Design analysis for the D7 variant
# a) Concrete behavior:
#    Each real shape color now appears in two disconnected same-color components:
#    a tiny 3x3 template and a large hollow square frame. The non-bg pixels inside
#    the 3x3 template are the canonical vertex pattern. The frame does not contain
#    the vertices directly; instead it provides the scale. If the frame side is
#    2*s+3, then the true vertices are placed on the frame's inner square at
#    offset (1 + template_coord * s). The output keeps the template and frame,
#    then draws the polygon/polyline implied by those scaled vertices. Optional
#    shadow pixels still sit 8-neighbor to one inferred start vertex and generate
#    a parallel copy of the shape as in the original task.
#    Small trace:
#    - Original: three red pixels already sit at the triangle vertices; solver
#      angle-sorts them and draws edges.
#    - Variant: the red input instead shows a 3x3 red triangle template plus a
#      larger red square frame. Solver reads the template pixels, computes the
#      scale from the frame, reconstructs the three large red vertices inside the
#      frame, then draws the same edges. The changed stage is "vertex discovery":
#      direct reading of same-color pixels becomes template lookup + scaling.
# b) Complexity calibration:
#    The template library is small and repeated across samples: segment, triangle,
#    rectangle, diamond, pentagon, hexagon. That keeps the rule inferable while
#    still visibly different from the base task. Scale is limited to 2-4 so the
#    enlargement is obvious but placement remains feasible on 15-22 grids.
# c) Old-solver assumptions that no longer hold:
#    - `np.argwhere(g == col)` used to return the actual vertices. Not true now:
#      same-color pixels include a template and a frame.
#    - `_angle_cycle` used to be applied directly to input pixels. Now it must be
#      applied only after reconstructing scaled vertices from template + frame.
#    - "start vertex has a foreign-color neighbor" still works, but only after the
#      inferred vertices are known. The raw input no longer contains those vertices.
#    - Shadow drawing used to shift every cell of the output shape color, but this
#      variant also has same-color frame/template cells; therefore generator output
#      must draw shadow lines edge-by-edge, not by shifting all shape-color cells.
# d) Placement legality:
#    Each shape patch is built locally from template + frame + optional shadow and
#    then pasted with >=1-cell padding. Shadow offsets are accepted only when the
#    shifted copy avoids the template and frame in the original input, so solver
#    line-clear checks stay valid.
# e) Visual observability:
#    The change is visible with the naked eye because the sparse direct vertices of
#    the base task disappear. Instead, each shape shows a tiny motif and a larger
#    guide frame of the same color. Across 2-4 examples, the repeated "template
#    pixels become enlarged interior vertices" mechanism is learnable.
# f) Visual cleanliness:
#    Real shapes are drawn on the frame's inner square, one cell away from the
#    border, so the frame remains legible. Templates are placed outside the frame
#    with a fixed gap, avoiding same-color touching and reducing clutter.
# g) Color inference:
#    No hidden color ordering is used. A shape is matched only by visible same-color
#    template/frame pairing. Shadow colors still act locally through adjacency.
# Post-change solver assumptions audited:
#    - For every true shape color, generator guarantees exactly one valid 3x3
#      template region (possibly disconnected inside that 3x3 box) and exactly
#      one valid hollow square frame component.
#    - Template pixels are chosen from a library whose angle-sorted scaled vertices
#      produce only H/V/45-degree edges, so the original line-drawing core remains
#      valid.
#    - Optional shadow anchors are placed adjacent to exactly one inferred vertex,
#      ensuring the start-vertex selection stays unambiguous.

import math
import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, label_mask, paste, random_free_location

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


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
TEMPLATE_SIZE = 3
FRAME_MARGIN = 1
TEMPLATE_GAP = 2

# ---- CFG: canvas and palette ----
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

# ---- CFG: structural counts ----
SHAPE_COUNT_RANGE = (2, 4)  # 2-3
MAX_SHADOWS_CAP = 2
DISTRACTOR_PROB = 0.45
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: shape geometry ----
SCALE_RANGE = (2, 5)  # 2-4

# ---- CFG: distractor geometry ----
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

# ---- CFG: retry budgets ----
SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


PATTERN_LIBRARY: list[tuple[str, list[tuple[int, int]]]] = [
    ("segment", [(0, 0), (2, 2)]),
    ("triangle", [(0, 0), (0, 2), (2, 2)]),
    ("rectangle", [(0, 0), (0, 2), (2, 2), (2, 0)]),
    ("diamond", [(0, 1), (1, 2), (2, 1), (1, 0)]),
    ("pentagon", [(0, 1), (1, 0), (2, 0), (2, 2), (1, 2)]),
    ("hexagon", [(0, 1), (1, 0), (2, 0), (2, 1), (1, 2), (0, 2)]),
]
PATTERN_WEIGHTS = np.array([0.18, 0.18, 0.18, 0.16, 0.15, 0.15], dtype=float)


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    cycle = _angle_cycle(vertices)
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step_dir(a[0], a[1], b[0], b[1]) for a, b in _shape_edges(vertices)]


def _frame_border(side: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(side):
        cells.add((0, c))
        cells.add((side - 1, c))
    for r in range(side):
        cells.add((r, 0))
        cells.add((r, side - 1))
    return cells


def _is_template_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    area = int(coords.shape[0])
    return h == TEMPLATE_SIZE and w == TEMPLATE_SIZE and 2 <= area <= 6


def _is_frame_component(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = bbox_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h != w or h < 7 or h % 2 == 0:
        return False
    expected = _frame_border(h)
    actual = {(int(r - r0), int(c - c0)) for r, c in coords}
    return actual == expected


def _extract_scaled_vertices(g: np.ndarray, col: int) -> list[tuple[int, int]] | None:
    mask = (g == int(col))
    lab, n = label_mask(mask, connectivity=8)
    comps = [np.argwhere(lab == k) for k in range(1, n + 1)]

    frames = [coords for coords in comps if _is_frame_component(coords)]
    if len(frames) != 1:
        return None

    frame = frames[0]
    frame_mask = np.zeros_like(mask, dtype=bool)
    for r, c in frame:
        frame_mask[int(r), int(c)] = True
    template = np.argwhere(mask & ~frame_mask)
    if template.size == 0 or not _is_template_component(template):
        return None

    tr0, tc0, _tr1, _tc1 = bbox_coords(template)
    fr0, fc0, fr1, fc1 = bbox_coords(frame)
    side = fr1 - fr0 + 1
    scale = (side - 3) // 2
    if 2 * scale + 3 != side:
        return None

    rel_template = [(int(r - tr0), int(c - tc0)) for r, c in template]
    inner_r0 = fr0 + FRAME_MARGIN
    inner_c0 = fc0 + FRAME_MARGIN
    verts = [(inner_r0 + r * scale, inner_c0 + c * scale) for r, c in rel_template]
    if len(verts) < 2:
        return None
    if any(not _aligned_8(a[0], a[1], b[0], b[1]) for a, b in _shape_edges(verts)):
        return None
    return verts


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = _extract_scaled_vertices(g, int(col))
        if pts_rc is None:
            continue

        cycle = _angle_cycle(pts_rc)

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


def _transform_template(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    rot = int(np.random.randint(4))
    for _ in range(rot):
        arr = np.stack([arr[:, 1], TEMPLATE_SIZE - 1 - arr[:, 0]], axis=1)
    if np.random.rand() < 0.5:
        arr[:, 1] = TEMPLATE_SIZE - 1 - arr[:, 1]
    pts = sorted({(int(r), int(c)) for r, c in arr})
    return pts


def _choose_pattern() -> tuple[str, list[tuple[int, int]]]:
    idx = int(np.random.choice(len(PATTERN_LIBRARY), p=PATTERN_WEIGHTS / PATTERN_WEIGHTS.sum()))
    name, pts = PATTERN_LIBRARY[idx]
    return name, _transform_template(pts)


def _template_origin(side: int) -> tuple[int, int]:
    mode = int(np.random.randint(4))
    mid = side // 2 - 1
    if mode == 0:  # above
        return -TEMPLATE_SIZE - TEMPLATE_GAP, mid
    if mode == 1:  # below
        return side + TEMPLATE_GAP, mid
    if mode == 2:  # left
        return mid, -TEMPLATE_SIZE - TEMPLATE_GAP
    return mid, side + TEMPLATE_GAP  # right


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    blocked_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in blocked_cells or pos in shape_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        shifted_shape = {(r + dr, c + dc) for r, c in shape_cells}
        if shifted_shape & blocked_cells:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


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
            if _aligned_8(coords[i][0], coords[i][1], coords[j][0], coords[j][1]):
                return None
    return coords


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy()


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count : shape_count + shadow_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _draw_edges(out: np.ndarray, vertices: list[tuple[int, int]], color: int) -> None:
    for a, b in _shape_edges(vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=color)


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    _name, template_rel = _choose_pattern()
    scale = int(np.random.randint(*SCALE_RANGE))
    side = 2 * scale + 3
    frame = _frame_border(side)
    temp_r0, temp_c0 = _template_origin(side)
    template_abs = {(temp_r0 + r, temp_c0 + c) for r, c in template_rel}

    vertices = [(FRAME_MARGIN + r * scale, FRAME_MARGIN + c * scale) for r, c in template_rel]
    shape_cells = _shape_cells(vertices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, shape_cells, frame | template_abs)
        if shadow is None:
            return None

    coords = list(frame | template_abs | shape_cells)
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    def shift(p: tuple[int, int]) -> tuple[int, int]:
        return p[0] - r0, p[1] - c0

    shifted_frame = [shift(p) for p in frame]
    shifted_template = [shift(p) for p in template_abs]
    shifted_vertices = [shift(p) for p in vertices]

    for r, c in shifted_frame:
        inp[r, c] = shape_color
        out[r, c] = shape_color
    for r, c in shifted_template:
        inp[r, c] = shape_color
        out[r, c] = shape_color

    _draw_edges(out, shifted_vertices, int(shape_color))

    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        shifted_anchor = shift((ar + dr, ac + dc))
        if inp[shifted_anchor] != BG:
            return None
        inp[shifted_anchor] = int(shadow_color)
        for a, b in _shape_edges(shifted_vertices):
            rr0 = a[0] + dr
            cc0 = a[1] + dc
            rr1 = b[0] + dr
            cc1 = b[1] + dc
            if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                return None
            _draw_line_bg_only(out, rr0, cc0, rr1, cc1, int(shadow_color), BG)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


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

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(int(distractor_color))):
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
