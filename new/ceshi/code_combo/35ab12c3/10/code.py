# Combo variant of 35ab12c3 (index 10)
# combo_spec: D6/1+D20/0
# Name: 重叠涂第三色+阴影颜色由映射决定
#
# Design analysis for this combo variant
# --------------------------------------
# a) Concrete behavior and trace
#    Base task: same-color vertices define one shape; an adjacent off-color pixel
#    near a chosen start vertex defines a parallel shadow offset, and the output
#    draws the shape plus the shifted copy in that off-color.
#    Fusion change:
#    - input shadow cues are no longer the real shadow colors; every cue pixel is
#      fixed color 4;
#    - output shadow color is inferred from the shape color by a fixed visible map
#      1->6, 2->7, 3->8;
#    - if a shifted shadow cell lands on the already drawn shape, that cell becomes
#      overlap color 5 instead of staying shape-colored.
#    Small trace: shape color 2 with vertices of an L-like triangle and a cue pixel
#    4 at offset (+1,+1). Old task would copy color 4 as the shadow and preserve 2
#    on collisions. This combo instead draws the shadow in 7 and paints overlap
#    cells in 5. The changed stages are therefore both "how shadow color is chosen"
#    and "what happens on shape/shadow conflicts".
#
# b) Complexity calibration
#    To keep the mapping inferable from 2-4 examples, shape colors are restricted to
#    {1,2,3} and every generated shape has a shadow, so each grid repeats the pairing
#    1->6, 2->7, 3->8. To keep the D6 change visible, every shadowed shape is forced
#    to have at least two overlap cells and at least two shadow-only cells. That
#    prevents degeneration into either "no overlap" or "overlap exists but mapping is
#    invisible".
#
# c) Implicit assumption audit
#    - `out = g.copy()` is no longer safe by itself because cue color 4 should not
#      remain in the output; main must erase cue pixels before drawing.
#    - seed detection cannot treat any foreign color as a shadow hint; only cue color
#      4 should trigger shadow logic, otherwise overlap color 5 or mapped shadow
#      colors could be misread on unusual inputs.
#    - carried shadow metadata is now only the offset; the actual shadow color must be
#      derived from the shape color by `SHADOW_COLOR_MAP`.
#    - the old bg-only shadow drawer hard-coded "shape wins"; it must be replaced with
#      a painter that converts shape/shadow collisions into overlap color 5.
#    - generator-side shadow search can no longer reject overlap-causing offsets; it
#      must actively prefer offsets that produce both overlap and exposed shadow area.
#
# d) Placement legality
#    Each object is generated as a local patch containing one polygonal shape, one cue
#    pixel, and its shifted shadow extent. The patch bounding box includes the shape,
#    the shifted copy, and the cue cell, so nothing clips. Global placement uses
#    `random_free_location(..., padding=1, connectivity=8)` on the full output patch,
#    which keeps different objects disjoint even after overlap color 5 is introduced.
#
# e) Visual observability
#    A human sees three layers in the output: shape color, mapped shadow color, and a
#    fixed grey overlap color 5. In the input, every shadow hint is the same yellow 4,
#    so the solver must infer that the cue supplies only direction, not color. Because
#    each grid has 2-3 shadowed objects, the mapping and the overlap rule both repeat
#    within a single example set.
#
# f) Visual cleanliness
#    The grid stays readable because cross-object intersections are still forbidden;
#    only intra-object shape/shadow crossings produce color 5. Restricting shapes to
#    simple convex polygons avoids tangled self-crossings and keeps the added overlap
#    layer local and legible.
#
# g) Change inferability
#    The color change is structural rather than arbitrary: yellow 4 always means "cue",
#    shape colors are limited to three reusable categories, and each category always
#    maps to one specific shadow color. Overlap color 5 has a consistent positional
#    meaning: it appears exactly where the mapped shadow crosses the base shape.
#
# h) New solver assumptions and generator guarantees
#    - Assumption: only colors 1,2,3 form shapes. Guarantee: generator samples shapes
#      exclusively from those colors.
#    - Assumption: only color 4 is a cue. Guarantee: input uses 4 only for one cue per
#      shape and never as a shape color.
#    - Assumption: each cue is 8-adjacent to exactly one vertex, so the start anchor is
#      unambiguous. Guarantee: `_choose_shadow` enforces that uniqueness.
#    - Assumption: the combo behavior matters on generated samples. Guarantee: every
#      shape has overlap_count>=2 and shadow_only_count>=2 under the fused solver.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, neighbors8, paste, random_free_location
from arc_utils import get_bg as _bg_color

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

BG = 0
CUE_COLOR = 4
OVERLAP_COLOR = 5
SHAPE_COLORS = [1, 2, 3]
SHADOW_COLOR_MAP = {1: 6, 2: 7, 3: 8}

# ---- CFG: canvas and counts ----
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)
SHAPE_COUNT_RANGE = (2, 4)  # 2-3 shapes
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: shape geometry ----
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)
SHAPE_WEIGHTS = np.array([0.24, 0.24, 0.18, 0.18, 0.16], dtype=float)

# ---- CFG: shadow visibility ----
MIN_OVERLAP_CELLS = 2
MIN_SHADOW_ONLY_CELLS = 2

# ---- CFG: retry budgets ----
SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 400


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


def _draw_shadow_line_with_overlap(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    shape_color: int,
    shadow_color: int,
    overlap_color: int,
    bg: int,
) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        cur = int(out[rr, cc])
        if cur == bg:
            out[rr, cc] = shadow_color
        elif cur == shape_color:
            out[rr, cc] = overlap_color


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


# Overview:
# Reconstruct polygonal traces from shape-color vertices, then add one parallel shadow
# per shape using a fixed cue color for offset detection, a fixed mapping for the true
# shadow color, and a dedicated overlap color where shape and shadow coincide.
# Input structure:
# Background is 0. Shape vertices use only colors 1, 2, 3. Each shape has exactly one
# adjacent cue pixel of color 4 that marks the shadow offset, and cue pixels are
# 8-adjacent to exactly one vertex. There are no distractors in this combo.
# Rule:
# For each shape color, connect its vertices in centroid-angle order, rotating the cycle
# so the cue-adjacent vertex is first. Draw the base shape in its own color. Copy the
# same edges by the cue offset; use shadow color 6/7/8 according to 1/2/3, and replace
# any shape/shadow collision cell with overlap color 5.
# Key insight:
# Color 4 supplies direction only; actual shadow color comes from the shape color, and
# color 5 marks where the shifted copy overlaps the original trace.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    out = g.copy()
    out[out == CUE_COLOR] = bg

    for col in SHAPE_COLORS:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        seed_idxs: list[int] = []
        for i, (r, c) in enumerate(cycle):
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == CUE_COLOR:
                    seed_idxs.append(i)
                    break

        if seed_idxs:
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

        cycle = cycle[start:] + cycle[:start]

        carried_offsets: list[tuple[int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) == CUE_COLOR:
                carried_offsets.append((drn, dcn))

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

        shadow_color = SHADOW_COLOR_MAP[col]
        for (r0, c0), (r1, c1) in drawable_edges:
            for drn, dcn in carried_offsets:
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
                _draw_shadow_line_with_overlap(out, rr0, cc0, rr1, cc1, col, shadow_color, OVERLAP_COLOR, bg)

    return out


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


_SHAPE_BUILDERS = [_build_triangle, _build_rectangle, _build_diamond, _build_pentagon, _build_hexagon]


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


def _shifted_cells(shape_cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in shape_cells}


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates: list[tuple[int, int, tuple[int, int], tuple[int, int]]] = []
    order = list(vertices)
    np.random.shuffle(order)
    for anchor in order:
        for dr, dc in DIR8:
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in shape_cells:
                continue
            cue_hits = sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1)
            if cue_hits != 1:
                continue
            shifted = _shifted_cells(shape_cells, (dr, dc))
            overlap_count = len(shape_cells & shifted)
            shadow_only_count = len(shifted - shape_cells)
            if overlap_count < MIN_OVERLAP_CELLS:
                continue
            if shadow_only_count < MIN_SHADOW_ONLY_CELLS:
                continue
            score = min(overlap_count, shadow_only_count)
            candidates.append((score, overlap_count, anchor, (dr, dc)))
    if not candidates:
        return None
    best_score = max(score for score, _ov, _anchor, _offset in candidates)
    strong = [(anchor, offset) for score, _ov, anchor, offset in candidates if score == best_score]
    return strong[int(np.random.randint(len(strong)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices
    (ar, ac), (dr, dc) = shadow
    coords.append((ar + dr, ac + dc))
    coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    cells = _shape_cells(vertices)
    shadow = _choose_shadow(vertices, cells)
    if shadow is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    (ar, ac), offset = shadow
    anchor = (ar - r0, ac - c0)
    cue = (anchor[0] + offset[0], anchor[1] + offset[1])
    if not (0 <= cue[0] < h and 0 <= cue[1] < w):
        return None
    if inp[cue] != BG:
        return None
    inp[cue] = CUE_COLOR

    cue_hits = sum(
        1
        for rr, cc in neighbors8(anchor[0], anchor[1])
        if 0 <= rr < h and 0 <= cc < w and int(inp[rr, cc]) == CUE_COLOR
    )
    if cue_hits != 1:
        return None

    out = main(inp)
    shadow_color = SHADOW_COLOR_MAP[shape_color]
    overlap_count = int(np.count_nonzero(out == OVERLAP_COLOR))
    shadow_only_count = int(np.count_nonzero(out == shadow_color))
    if overlap_count < MIN_OVERLAP_CELLS:
        return None
    if shadow_only_count < MIN_SHADOW_ONLY_CELLS:
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
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _sample_shape_colors(shape_count: int) -> list[int]:
    return list(np.random.permutation(SHAPE_COLORS)[:shape_count])


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    shape_colors = _sample_shape_colors(shape_count)

    for shape_color in shape_colors:
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    input_colors = set(colors_present(inp, exclude=[BG]))
    expected_input_colors = set(shape_colors) | {CUE_COLOR}
    if input_colors != expected_input_colors:
        return None
    output_colors = set(colors_present(out, exclude=[BG]))
    expected_output_colors = set(shape_colors) | {SHADOW_COLOR_MAP[col] for col in shape_colors} | {OVERLAP_COLOR}
    if output_colors != expected_output_colors:
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
