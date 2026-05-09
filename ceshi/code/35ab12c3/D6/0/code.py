# Variant of 35ab12c3
# Dimension: D6 — 阴影优先级规则 (transform)
# Change: 形状颜色优先（shape color wins）——当阴影像素与形状像素重叠时，形状颜色覆盖阴影颜色... → 阴影颜色优先

import math

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the D6 variant
#
# a) Concrete behavior and trace
#    - Original task: each shape color defines vertices; the solver fills the
#      aligned edges between consecutive vertices. If the chosen start vertex has
#      an adjacent foreign-color pixel, that pixel encodes a shadow offset/color.
#      The solver then draws a parallel copy of every drawable edge at that offset.
#    - Original priority: both the main shape and the shadow are painted only into
#      background cells. This makes shape-shadow conflicts invisible because the
#      shape is drawn first and the shadow cannot overwrite it.
#    - Variant D6: keep the geometry, anchor detection, edge filtering, and input
#      encoding unchanged. Only flip the overlap priority: when a shifted shadow
#      edge lands on a cell already occupied by the main shape, the shadow color
#      must replace the shape color.
#    - Small trace:
#      shape vertices: (0,0), (0,4), (4,4), (4,0)
#      shadow anchor: a different-color pixel at (1,1), so offset=(+1,+1)
#      output shape border contains top row, right col, bottom row, left col.
#      shifted shadow border overlaps cells near the lower-right corner.
#      original code: those overlap cells stay in the shape color because the
#      shadow only paints background.
#      variant code: those same overlap cells are recolored to the shadow color,
#      so the shape edge looks "eaten away" by the shadow.
#
# b) Complexity calibration
#    - The rule itself changes only one precedence relation, so the generator must
#      force visible overlap. Otherwise many examples would look identical to the
#      original task.
#    - To keep the task inferable, each generated grid contains at least one shape
#      with a shadow, and every attached shadow is required to overlap the filled
#      shape in the final output.
#    - Shadowed shapes are sampled from polygonal builders that are much more
#      likely than open polylines to admit valid overlapping shadow placements
#      without introducing ambiguity in the input.
#
# c) Audit of original main() assumptions
#    - `out = g.copy()`: still valid. Input vertices and the single shadow-anchor
#      pixel remain as the seed of the output.
#    - `_clear_in_original(...)`: still valid. The variant does not change which
#      edges are drawable; it changes only how already-drawn shape cells interact
#      with later shadow painting.
#    - `_draw_line_bg_only` in phase 1: still valid. Main edges should keep using
#      normal shape drawing and do not need to overwrite unrelated objects.
#    - `_draw_line_bg_only` in phase 2: invalid under D6. This line assumes the
#      old "shape wins" invariant. It must be replaced by unconditional painting
#      so the shadow can overwrite shape cells.
#    - color iteration order: acceptable because generator placement preserves a
#      buffer between distinct objects, so cross-object overwrite is not relied on.
#
# d) Placement legality
#    - Each local patch is still built output-first, then reduced to input by
#      keeping only vertices plus one shadow-anchor pixel.
#    - Candidate shadow offsets are enumerated explicitly. A candidate is accepted
#      only if:
#        1. the shadow-anchor pixel is not placed on a shape cell,
#        2. that anchor pixel is 8-adjacent to exactly one vertex,
#        3. the new `main()` reproduces the local output exactly.
#    - Whole-patch placement still uses `random_free_location(..., padding=1)` on
#      the final output patch, so different groups remain separated.
#
# e) Visual observability
#    - The crucial observable is `main_variant(inp) != main_original(inp)` on the
#      same input. That only happens when a shadow and its base shape overlap.
#    - Therefore the generator rejects shadow placements with zero overlap and
#      rejects entire grids without at least one shadow-bearing shape.
#
# f) Visual cleanliness
#    - Overlap is restricted to one shape and its own shadow. Different objects are
#      still separated by a background buffer, so the output is not globally messy.
#    - The input remains sparse and legible: same-colored vertices plus a single
#      nearby shadow cue.
#
# g) Color inferability
#    - No hidden use of color ids is introduced.
#    - Each shadow color still appears as the nearby cue pixel in the input and as
#      the parallel copy in the output; D6 only changes which color remains visible
#      on overlap cells.
#
# 9) New solver assumptions and generator guarantees
#    - New assumption: if a shadow is present, drawing it after the main shape may
#      overwrite some cells already painted by that shape.
#      Generator guarantee: every shadowed patch is solver-verified and built from
#      a candidate whose overlap is intentional and local to that patch.
#    - New assumption: inputs meant to exhibit the variant should include at least
#      one visible shape-shadow conflict.
#      Generator guarantee: every generated grid has >=1 shadowed shape, and every
#      attached shadow has positive overlap with the filled shape.
# -----------------------------------------------------------------------------

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


def _draw_line_overwrite(out: np.ndarray, r0: int, c0: int, r1: int, c1: int, col: int) -> None:
    for rr, cc in _segment_cells(r0, c0, r1, c1):
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
    bg = _bg_color(g)

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
                _draw_line_overwrite(out, rr0, cc0, rr1, cc1, ncol)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MIN_SHADOWS_PER_GRID = 1
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
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
SHADOW_BUILDER_IDXS = [1, 2, 3, 4, 5]
SHADOW_SHAPE_WEIGHTS = np.array([0.18, 0.28, 0.22, 0.16, 0.16], dtype=float)
MIN_SHADOW_OVERLAP = 1

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

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


def _make_vertices(shadow_required: bool = False) -> list[tuple[int, int]]:
    if not shadow_required:
        weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
        idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
        return _SHAPE_BUILDERS[idx]()
    weights = SHADOW_SHAPE_WEIGHTS / SHADOW_SHAPE_WEIGHTS.sum()
    sub_idx = int(np.random.choice(len(SHADOW_BUILDER_IDXS), p=weights))
    return _SHAPE_BUILDERS[SHADOW_BUILDER_IDXS[sub_idx]]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _shift_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _shadow_overlap_count(shape_cells: set[tuple[int, int]], offset: tuple[int, int]) -> int:
    return len(shape_cells & _shift_cells(shape_cells, offset))


def _shadow_candidates(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for anchor in vertices:
        for dr, dc in DIR8:
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in shape_cells:
                continue
            if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
                continue
            overlap = _shadow_overlap_count(shape_cells, (dr, dc))
            if overlap < MIN_SHADOW_OVERLAP:
                continue
            candidates.append((anchor, (dr, dc), overlap))
    np.random.shuffle(candidates)
    candidates.sort(key=lambda item: item[2], reverse=True)
    return candidates


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


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int], overwrite: bool) -> None:
    dr, dc = offset
    h, w = out.shape
    src = out == shape_color
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted if overwrite else (shifted & (out == BG))
    out[mask] = shadow_color


def _build_base_patch(vertices: list[tuple[int, int]], shape_color: int) -> tuple[np.ndarray, np.ndarray]:
    cells = _shape_cells(vertices)
    r0, c0, r1, c1 = _patch_bbox(cells, vertices, None)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color
    return inp, out


def _build_shadow_patch(
    vertices: list[tuple[int, int]],
    shape_color: int,
    shadow_color: int,
    anchor_src: tuple[int, int],
    offset: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    cells = _shape_cells(vertices)
    r0, c0, r1, c1 = _patch_bbox(cells, vertices, (anchor_src, offset))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for a, b in _shape_edges(shifted_vertices):
        draw_line(out, a[0], a[1], end_r=b[0], end_c=b[1], color=shape_color)
    for r, c in shifted_vertices:
        inp[r, c] = shape_color

    anchor = (anchor_src[0] - r0, anchor_src[1] - c0)
    shadow_anchor = (anchor[0] + offset[0], anchor[1] + offset[1])
    if not (0 <= shadow_anchor[0] < h and 0 <= shadow_anchor[1] < w):
        return None
    if inp[shadow_anchor] != BG:
        return None

    _paint_shadow(out, shape_color, shadow_color, offset, overwrite=True)

    inp[shadow_anchor] = shadow_color
    nb_hits = sum(
        1
        for rr, cc in neighbors8(shadow_anchor[0], shadow_anchor[1])
        if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shape_color
    )
    if nb_hits != 1:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices(shadow_required=shadow_color is not None)

    if shadow_color is None:
        inp, out = _build_base_patch(vertices, shape_color)
        if not np.array_equal(main(inp), out):
            return None
        return inp, out

    cells = _shape_cells(vertices)
    candidates = _shadow_candidates(vertices, cells)
    if not candidates:
        return None

    for anchor, offset, _overlap in candidates:
        patch = _build_shadow_patch(vertices, shape_color, int(shadow_color), anchor, offset)
        if patch is not None:
            return patch
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
    if max_shadows < MIN_SHADOWS_PER_GRID:
        return None
    shadow_count = int(np.random.randint(MIN_SHADOWS_PER_GRID, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = palette[:shape_count]
    shadow_colors = palette[shape_count:shape_count + shadow_count]
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


def _sample_canvas_size() -> tuple[int, int]:
    h = int(np.random.randint(*CANVAS_H_RANGE))
    w = int(np.clip(h + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return h, w


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

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
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
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
