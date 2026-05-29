# Combo variant of 35ab12c3 (index 21)
# combo_spec: D7/3+D17/0
# Name: 模板图案源+单一形状极简
#
# Design analysis for the combo variant
# a) Concrete behavior:
#    The base task reads each same-color pixel set as the actual polygon/polyline
#    vertices. This combo removes direct vertices from the input. Instead, the
#    shape color appears as two visible same-color structures: a tiny 3x3 template
#    and one larger hollow square frame. The template's non-background pixels are
#    the canonical vertex pattern, and the frame side determines the scale. If the
#    frame side is 2*s+5, the true vertices are reconstructed at the frame's inner
#    square via (inner_top_left + template_coord * s). The output keeps the frame
#    and template, then draws the enlarged shape between the inferred vertices.
#    Small trace:
#    - Original: three red pixels already sit at triangle corners, so the solver
#      angle-sorts those three pixels and draws the triangle.
#    - Combo: the red input instead shows a red 3x3 triangle template and a red
#      hollow square. The solver first detects the template, infers scale from the
#      frame, reconstructs the large triangle vertices inside the frame, and only
#      then draws the edges. If a shadow anchor is present, it is adjacent to one
#      inferred vertex and generates a parallel copy exactly as in the base task.
# b) Complexity calibration:
#    D17/0 simplifies the scene to exactly one shape group and no distractors, but
#    the template/frame encoding still makes the task visibly different from the
#    original. The template library is small and repeated (segment, triangle,
#    rectangle, diamond, pentagon, hexagon), and the scale is restricted to 2-4.
#    That preserves learnability while avoiding an over-trivial "single line only"
#    dataset. Optional shadow is binary (present/absent), so the key free variables
#    stay low-cardinality.
# c) Implicit assumption audit:
#    - Old `np.argwhere(g == col)` no longer returns real vertices. The combo
#      solver must first separate the frame component from the 3x3 template.
#    - Old start-vertex detection via "different-color 8-neighbor" still works,
#      but only after the inferred large vertices are reconstructed.
#    - Old shadow generation could shift all shape-colored output cells. That would
#      wrongly shift the template and frame too, so the generator now paints shadow
#      edge-by-edge from inferred vertices instead.
#    - Old multi-color loop assumed many colors might be shapes or distractors.
#      Here only one color should decode as a valid template+frame shape; the other
#      visible color, when present, is just the shadow anchor. The generator
#      guarantees that uniqueness.
#    - `_clear_in_original` now relies on the template living outside the frame and
#      on shadow lines avoiding both template and frame cells. The generator checks
#      those placement constraints before accepting a sample.
# d) Placement legality:
#    A complete local patch is built first: frame, external template, optional
#    shadow anchor, and the intended output lines. The patch bbox includes the
#    shifted shadow copy, so later canvas placement only needs a simple in-bounds
#    paste. The generator rejects any shadow whose translated shape would overlap
#    the frame, the template, or leave the patch.
# e) Visual observability:
#    Across a few examples, humans can see that the output shape is much larger
#    than the input template, yet its corners mirror the 3x3 motif. Because there
#    is only one shape, attention is focused on "template -> scaled vertices" rather
#    than on multi-object matching. When a shadow exists, it is visibly attached to
#    one inferred vertex and traced in parallel.
# f) Visual cleanliness:
#    The frame and the drawn shape are separated by a two-cell inner margin, and
#    the template is placed outside the frame with a fixed gap. Removing all extra
#    shapes and distractors keeps the grid sparse and avoids ambiguous overlays.
# g) Change inferability:
#    No hidden color ordering is used. The shape color is identified only by its
#    visible template/frame pairing, and the shadow color is identified only by its
#    local adjacency to one inferred vertex. Since there are at most two active
#    colors besides background, the color logic remains structurally legible.
# Post-change solver assumptions audited:
#    - Exactly one input color forms one valid hollow odd-sized square frame and
#      one valid 3x3 template remainder.
#    - The template points come from a small library whose scaled angle-cycle edges
#      remain H/V/45-degree aligned.
#    - The optional shadow anchor is adjacent to exactly one inferred vertex and
#      uses an offset not collinear with any incident edge direction.
#    - No distractor or second shape color exists, so template/frame pairing stays
#      unambiguous and the single-shape reading remains valid.

import math
import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, label_mask, paste

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
FRAME_MARGIN = 2
TEMPLATE_GAP = 2

CANVAS_RANGE = (15, 23)
SCALE_RANGE = (2, 5)  # 2-4
SHADOW_PROB = 0.6
GENERATE_RETRIES = 300

PATTERN_LIBRARY: list[tuple[str, list[tuple[int, int]]]] = [
    ("segment", [(0, 0), (2, 2)]),
    ("triangle", [(0, 0), (0, 2), (2, 2)]),
    ("rectangle", [(0, 0), (0, 2), (2, 2), (2, 0)]),
    ("diamond", [(0, 1), (1, 2), (2, 1), (1, 0)]),
    ("pentagon", [(0, 1), (1, 0), (2, 0), (2, 2), (1, 2)]),
    ("hexagon", [(0, 1), (1, 0), (2, 0), (2, 1), (1, 2), (0, 2)]),
]
PATTERN_WEIGHTS = np.array([0.16, 0.19, 0.18, 0.16, 0.16, 0.15], dtype=float)


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
    if h != w or h < 2 * FRAME_MARGIN + 5 or h % 2 == 0:
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
    scale = (side - (2 * FRAME_MARGIN + 1)) // 2
    if 2 * scale + (2 * FRAME_MARGIN + 1) != side:
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


def _shape_candidates(g: np.ndarray, bg: int) -> list[tuple[int, list[tuple[int, int]]]]:
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    out: list[tuple[int, list[tuple[int, int]]]] = []
    for col in colors:
        verts = _extract_scaled_vertices(g, int(col))
        if verts is not None:
            out.append((int(col), verts))
    return out


# Overview:
# The input encodes one connect-the-vertices shape indirectly through a tiny
# template and a larger same-color frame, then the output draws the scaled shape.
# Input structure:
# One non-background color forms exactly one 3x3 template region plus one hollow
# odd-sized square frame; optionally a second color appears as a single shadow
# anchor pixel 8-neighbor to one inferred vertex.
# Rule:
# Infer the scale from the frame side, map the template pixels onto the frame's
# interior as true vertices, connect them with H/V/45-degree edges, and if the
# optional shadow anchor exists, copy every edge in parallel by that offset.
# Key insight:
# The visible same-color pixels are not the final vertices. The 3x3 template is
# the source of vertex layout, and the frame only tells you how far to scale it.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    for col, pts_rc in _shape_candidates(g, bg):
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
    if mode == 0:
        return -TEMPLATE_SIZE - TEMPLATE_GAP, mid
    if mode == 1:
        return side + TEMPLATE_GAP, mid
    if mode == 2:
        return mid, -TEMPLATE_SIZE - TEMPLATE_GAP
    return mid, side + TEMPLATE_GAP


def _shadow_candidates(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    blocked_cells: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    directions = _edge_dirs(vertices)
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for anchor in vertices:
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
            candidates.append((anchor, (dr, dc)))
    return candidates


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    blocked_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates = _shadow_candidates(vertices, shape_cells, blocked_cells)
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _draw_edges(out: np.ndarray, vertices: list[tuple[int, int]], color: int) -> None:
    for a, b in _shape_edges(vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=color)


def _make_shape_patch(shape_color: int) -> tuple[np.ndarray, np.ndarray, bool] | None:
    _name, template_rel = _choose_pattern()
    scale = int(np.random.randint(*SCALE_RANGE))
    side = 2 * scale + (2 * FRAME_MARGIN + 1)
    frame = _frame_border(side)
    temp_r0, temp_c0 = _template_origin(side)
    template_abs = {(temp_r0 + r, temp_c0 + c) for r, c in template_rel}

    vertices = [(FRAME_MARGIN + r * scale, FRAME_MARGIN + c * scale) for r, c in template_rel]
    shape_cells = _shape_cells(vertices)

    shadow_candidates = _shadow_candidates(vertices, shape_cells, frame | template_abs)
    use_shadow = bool(shadow_candidates) and np.random.rand() < SHADOW_PROB
    shadow_color = None
    shadow = None
    if use_shadow:
        shadow_choices = [c for c in COLORS if c != shape_color]
        shadow_color = int(np.random.choice(shadow_choices))
        shadow = shadow_candidates[int(np.random.randint(len(shadow_candidates)))]

    coords = list(frame | template_abs | shape_cells)
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)

    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

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

    if len(_shape_candidates(inp, BG)) != 1:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out, shadow is not None


def _sample_shape_color() -> int:
    return int(np.random.choice(COLORS))


def _sample_canvas_size(min_h: int, min_w: int) -> tuple[int, int] | None:
    low_h = max(CANVAS_RANGE[0], int(min_h))
    low_w = max(CANVAS_RANGE[0], int(min_w))
    high = CANVAS_RANGE[1] - 1
    if low_h > high or low_w > high:
        return None
    H = int(np.random.randint(low_h, high + 1))
    W = int(np.random.randint(low_w, high + 1))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    shape_color = _sample_shape_color()
    patch = _make_shape_patch(shape_color)
    if patch is None:
        return None
    patch_in, patch_out, has_shadow = patch
    ph, pw = patch_in.shape

    canvas_size = _sample_canvas_size(ph, pw)
    if canvas_size is None:
        return None
    H, W = canvas_size
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)

    r0 = int(np.random.randint(0, H - ph + 1))
    c0 = int(np.random.randint(0, W - pw + 1))
    paste(inp, patch_in, r0, c0, transparent=BG)
    paste(out, patch_out, r0, c0, transparent=BG)

    expected_colors = 1 + int(has_shadow)
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != expected_colors:
        return None
    if len(_shape_candidates(inp, BG)) != 1:
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
