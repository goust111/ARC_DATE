# Variant of 35ab12c3
# Dimension: D12 — 网格形状/输出裁剪 (grid_shape)
# Change: same——输出网格与输入尺寸完全相同（15-22行 x 15-22列）... → 按形状分离输出

# Design analysis for the D12 variant
# ----------------------------------
# a) Concretization / behavior trace
#    - Original behavior: solve every drawable same-color group in-place on the full
#      canvas, so the output keeps the original 15-22 x 15-22 scene.
#    - Variant behavior: still solve each true shape-group exactly the same way, but
#      then isolate each solved shape-group into its own tight bbox crop and assemble
#      those crops into one strip with a 1-cell background gutter.
#    - Small trace:
#      input has three real shape groups A,B,C spread left-to-right and one distractor D.
#      Original pipeline:
#        vertices/shadow anchors -> draw A on the big canvas -> draw B -> draw C -> keep D.
#      Variant pipeline:
#        vertices/shadow anchors -> solve A/B/C exactly as before -> crop bbox(A), bbox(B),
#        bbox(C) -> concatenate [A | gap | B | gap | C]. D is ignored because it is not a
#        drawable shape-group.
#      The only conceptual change is the output reorganization stage; the line/shadow
#      geometry stays faithful to the source task.
# b) Complexity calibration
#    - To keep the strip readable, the generator still uses 2-4 shape groups and optional
#      shadows, but shape size ranges are slightly tightened so several crops can fit in a
#      15-22 sized input canvas while still producing a visibly long output strip.
#    - Orientation is not random noise: groups are placed sequentially along one dominant
#      axis in the input, and the output strip follows that visible axis. This gives the
#      solver a repeatable structural cue instead of an invisible latent flag.
# c) Audit of old assumptions in main()
#    - `out = g.copy()` used to be harmless because the final answer lived on the same
#      canvas. In the variant this would be wrong: distractors and unused empty space would
#      leak into the answer. The new main therefore solves groups on blank same-sized
#      canvases, then crops and reassembles.
#    - Iterating colors in numeric order used to be harmless because geometry stayed in
#      place. In the variant, output order must not depend on color ids. Groups are sorted
#      only by their visible positions along the dominant axis.
#    - Colors with 2-3 pixels can be distractors. The old solver safely left them in place
#      because no drawable edges were found. The variant must explicitly exclude such colors
#      from the strip, so only groups with at least one drawable edge are exported.
# d) Placement legality
#    - Generator places each shape patch using its fully solved bbox footprint, not just
#      its sparse input vertices. This guarantees that later completion/shadows cannot make
#      groups overlap.
#    - Sequential placement uses a 2-cell gap along the strip axis; this keeps different
#      groups from becoming 8-neighbors, which matters because the solver reads shadow
#      anchor neighbors from the input.
#    - Optional distractors are added only after shape placement with padded free-space
#      search against the solved footprint.
# e) Visual observability
#    - Across examples the human can see that shapes appear arranged mainly left-to-right
#      or top-to-bottom in the input, and the output is a packed strip in that same order.
#    - No ordering or transformation uses hidden properties such as color id or sampling
#      order.
# f) Visual cleanliness
#    - Each output crop is tight, and crops are separated by a 1-cell background gutter,
#      so different groups remain visually separable instead of touching into one cluttered
#      blob.
# g) Color inference
#    - Colors themselves do not change. The variant only changes where solved groups are
#      shown. Shadow colors remain paired with the same local shape as in the source task.
# h) New assumptions introduced by the variant
#    - `main()` now assumes at least one dominant placement axis can be inferred from the
#      true shape groups. The generator guarantees this by arranging them sequentially with
#      small orthogonal jitter and large primary-axis span.
#    - `main()` also assumes each exported group has a non-empty solved bbox. The generator
#      guarantees this by using the same per-group oracle used by the solver.

import math
import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    draw_line,
    get_bg,
    neighbors8,
    paste,
    random_free_location,
)
from arc_utils import get_bg as _bg_color

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


def _clear_in_original(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bg: int,
    allow: set[int],
) -> bool:
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


def _solve_shape_canvas(g: np.ndarray, col: int, bg: int) -> np.ndarray | None:
    h, w = g.shape
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return None

    cycle = _angle_cycle(pts_rc)
    if len(cycle) < 2:
        return None

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

    if not drawable_edges:
        return None

    out = np.full_like(g, bg)
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


def _extract_shape_groups(g: np.ndarray) -> tuple[list[dict[str, object]], int]:
    bg = _bg_color(g)
    groups: list[dict[str, object]] = []
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        solved = _solve_shape_canvas(g, col, bg)
        if solved is None:
            continue
        coords = np.argwhere(solved != bg)
        if coords.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(coords)
        crop = solved[r0 : r1 + 1, c0 : c1 + 1].copy()
        groups.append(
            {
                "color": int(col),
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "crop": crop,
            }
        )
    return groups, bg


def _infer_horizontal(groups: list[dict[str, object]], grid_shape: tuple[int, int]) -> bool:
    if len(groups) <= 1:
        return True
    row_centers = [float(info["center"][0]) for info in groups]
    col_centers = [float(info["center"][1]) for info in groups]
    row_span = max(row_centers) - min(row_centers)
    col_span = max(col_centers) - min(col_centers)
    if col_span != row_span:
        return col_span > row_span
    return grid_shape[1] >= grid_shape[0]


def _assemble_strip(crops: list[np.ndarray], bg: int, horizontal: bool) -> np.ndarray:
    if not crops:
        return np.full((1, 1), bg, dtype=int)
    if len(crops) == 1:
        return crops[0].copy()

    gap = 1
    if horizontal:
        out_h = max(int(crop.shape[0]) for crop in crops)
        out_w = sum(int(crop.shape[1]) for crop in crops) + gap * (len(crops) - 1)
        out = np.full((out_h, out_w), bg, dtype=int)
        c = 0
        for crop in crops:
            r = (out_h - crop.shape[0]) // 2
            paste(out, crop, r, c, transparent=bg)
            c += crop.shape[1] + gap
        return out

    out_h = sum(int(crop.shape[0]) for crop in crops) + gap * (len(crops) - 1)
    out_w = max(int(crop.shape[1]) for crop in crops)
    out = np.full((out_h, out_w), bg, dtype=int)
    r = 0
    for crop in crops:
        c = (out_w - crop.shape[1]) // 2
        paste(out, crop, r, c, transparent=bg)
        r += crop.shape[0] + gap
    return out


def _solve_full_canvas(g: np.ndarray) -> np.ndarray:
    bg = _bg_color(g)
    out = np.asarray(g, dtype=int).copy()
    groups, _ = _extract_shape_groups(g)
    for info in groups:
        crop = np.asarray(info["crop"], dtype=int)
        r0, c0, r1, c1 = info["bbox"]
        target = np.full_like(g, bg)
        target[r0 : r1 + 1, c0 : c1 + 1] = crop
        paste(out, target, 0, 0, transparent=bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    groups, bg = _extract_shape_groups(g)
    if not groups:
        return np.full((1, 1), bg, dtype=int)

    horizontal = _infer_horizontal(groups, g.shape)
    if horizontal:
        groups.sort(key=lambda info: (float(info["center"][1]), float(info["center"][0]), info["bbox"][1], info["bbox"][0]))
    else:
        groups.sort(key=lambda info: (float(info["center"][0]), float(info["center"][1]), info["bbox"][0], info["bbox"][1]))
    crops = [np.asarray(info["crop"], dtype=int) for info in groups]
    return _assemble_strip(crops, bg, horizontal)


# --------------- generator (D12 variant) ---------------

# ---- CFG: canvas and palette ----
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_MIN = 15
CANVAS_MAX = 22

# ---- CFG: structural counts ----
SHAPE_COUNT_RANGE = (2, 5)         # 2-4 shapes
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.4
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: variant layout ----
SEQUENCE_GAP = 2                   # keeps shape groups non-adjacent in 8-neighborhood
OUTER_MARGIN = 1
ORTH_JITTER = 2

# ---- CFG: shape geometry ----
# Slightly tightened from the source task so several cropped groups can fit in one
# 15-22 sized input canvas while still yielding a visibly recomposed output strip.
LINE_VERTICES_RANGE = (2, 6)       # 2-5 vertices
LINE_TOTAL_LENGTH = 9
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 6)
RECT_H_RANGE = (2, 7)
RECT_W_RANGE = (2, 7)
DIAMOND_ARM_RANGE = (2, 5)
PENTAGON_ROOF_RANGE = (2, 4)
PENTAGON_WALL_RANGE = (2, 6)
HEXAGON_SLANT_RANGE = (2, 4)
HEXAGON_WALL_RANGE = (2, 5)

# ---- CFG: shape mixture ----
SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

# ---- CFG: distractor geometry ----
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

# ---- CFG: retry budgets ----
SHAPE_BUILD_RETRIES = 60
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
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
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


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
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
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
        )
        if nb_hits != 1:
            return None

    solved = _solve_shape_canvas(inp, shape_color, BG)
    if solved is None or not np.array_equal(solved, out):
        return None
    return inp, out


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
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count : shape_count + shadow_count]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
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


def _layout_base_dims(shape_patches: list[tuple[np.ndarray, np.ndarray]], horizontal: bool) -> tuple[int, int]:
    heights = [int(patch_out.shape[0]) for _, patch_out in shape_patches]
    widths = [int(patch_out.shape[1]) for _, patch_out in shape_patches]
    if horizontal:
        need_h = max(heights) + 2 * OUTER_MARGIN + 2 * ORTH_JITTER
        need_w = sum(widths) + SEQUENCE_GAP * (len(widths) - 1) + 2 * OUTER_MARGIN
    else:
        need_h = sum(heights) + SEQUENCE_GAP * (len(heights) - 1) + 2 * OUTER_MARGIN
        need_w = max(widths) + 2 * OUTER_MARGIN + 2 * ORTH_JITTER
    return max(CANVAS_MIN, need_h), max(CANVAS_MIN, need_w)


def _choose_layout(shape_patches: list[tuple[np.ndarray, np.ndarray]]) -> tuple[bool, int, int] | None:
    options: list[tuple[bool, int, int]] = []
    for horizontal in (True, False):
        base_h, base_w = _layout_base_dims(shape_patches, horizontal)
        if base_h <= CANVAS_MAX and base_w <= CANVAS_MAX:
            H = int(np.random.randint(base_h, CANVAS_MAX + 1))
            W = int(np.random.randint(base_w, CANVAS_MAX + 1))
            options.append((horizontal, H, W))
    if not options:
        return None
    return options[int(np.random.randint(len(options)))]


def _place_shape_sequence(
    inp: np.ndarray,
    out: np.ndarray,
    shape_patches: list[tuple[np.ndarray, np.ndarray]],
    horizontal: bool,
) -> None:
    heights = [int(patch_out.shape[0]) for _, patch_out in shape_patches]
    widths = [int(patch_out.shape[1]) for _, patch_out in shape_patches]

    if horizontal:
        total_w = sum(widths) + SEQUENCE_GAP * (len(widths) - 1)
        start_c = int(np.random.randint(OUTER_MARGIN, inp.shape[1] - total_w - OUTER_MARGIN + 1))
        band_h = max(heights)
        base_band_r = OUTER_MARGIN + (inp.shape[0] - 2 * OUTER_MARGIN - band_h) // 2
        c = start_c
        for patch_in, patch_out in shape_patches:
            h_i, w_i = patch_out.shape
            base_r = base_band_r + (band_h - h_i) // 2
            lo = max(OUTER_MARGIN, base_r - ORTH_JITTER)
            hi = min(inp.shape[0] - OUTER_MARGIN - h_i, base_r + ORTH_JITTER)
            r = int(np.random.randint(lo, hi + 1))
            paste(out, patch_out, r, c, transparent=BG)
            paste(inp, patch_in, r, c, transparent=BG)
            c += w_i + SEQUENCE_GAP
        return

    total_h = sum(heights) + SEQUENCE_GAP * (len(heights) - 1)
    start_r = int(np.random.randint(OUTER_MARGIN, inp.shape[0] - total_h - OUTER_MARGIN + 1))
    band_w = max(widths)
    base_band_c = OUTER_MARGIN + (inp.shape[1] - 2 * OUTER_MARGIN - band_w) // 2
    r = start_r
    for patch_in, patch_out in shape_patches:
        h_i, w_i = patch_out.shape
        base_c = base_band_c + (band_w - w_i) // 2
        lo = max(OUTER_MARGIN, base_c - ORTH_JITTER)
        hi = min(inp.shape[1] - OUTER_MARGIN - w_i, base_c + ORTH_JITTER)
        c = int(np.random.randint(lo, hi + 1))
        paste(out, patch_out, r, c, transparent=BG)
        paste(inp, patch_in, r, c, transparent=BG)
        r += h_i + SEQUENCE_GAP


def _gen_attempt() -> np.ndarray | None:
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    shape_patches: list[tuple[np.ndarray, np.ndarray]] = []
    for idx, shape_color in enumerate(shape_colors):
        built = None
        for _ in range(SHAPE_BUILD_RETRIES):
            built = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if built is not None:
                break
        if built is None:
            return None
        shape_patches.append(built)

    layout = _choose_layout(shape_patches)
    if layout is None:
        return None
    horizontal, H, W = layout
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    _place_shape_sequence(inp, out, shape_patches, horizontal)

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    expected = _assemble_strip([patch_out for _, patch_out in shape_patches], BG, horizontal)
    actual = main(inp)
    if not np.array_equal(actual, expected):
        return None

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if actual.shape == inp.shape and np.array_equal(actual, _solve_full_canvas(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
