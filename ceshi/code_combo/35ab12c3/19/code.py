# Combo variant of 35ab12c3 (index 19)
# combo_spec: D4/3+D22/1
# Name: 仅涂顶点不涂边+顶点扩展为十字标记
#
# Design analysis for this combo
# ------------------------------
# a) Concrete behavior and trace
#    - Original task: same-color input vertices are ordered into a line/polygon,
#      every H/V/45deg edge is painted continuously, and one adjacent foreign-color
#      anchor may spawn a parallel shadow copy of those edges.
#    - Combo behavior: keep the same shape-recognition pipeline, but change
#      rendering completely. No edge interiors are painted at all. Every
#      recognized vertex center becomes a five-cell cross
#      {(0,0),(-1,0),(1,0),(0,-1),(0,1)} in the shape color.
#    - The original foreign-color shadow anchor is intentionally not amplified in
#      this combo. Under a one-cell 8-neighbor offset, any translated cross would
#      necessarily overlap the source cross and create unreadable conflicts. The
#      fusion prompt only modifies vertices, so non-vertex anchors simply remain
#      as unchanged input pixels if present.
#    - Trace example: input has color 4 at (5,5) and (5,9), plus color 7 at (4,5).
#      Original output would fill 4 along row 5 from col 5..9 and 7 along row 4.
#      This combo instead outputs a cross around (5,5) and a cross around (5,9);
#      the color-7 anchor stays a single unchanged pixel. The cells between the
#      vertices remain background. The visual change is therefore
#      "line drawing -> marker amplification", not a color remap or a topology
#      change.
#
# b) Complexity calibration
#    - A single retained center pixel would make many outputs look too close to the
#      input. The cross marker is large enough to be visible but still local.
#    - A 3x3 full square would create too much clutter and merge nearby vertices, so
#      the five-cell cross is the tightest clear marker.
#    - To avoid ambiguous merged markers, the generator raises the minimum vertex
#      spacing to 3 cells on line-based shapes and rejects any patch whose markers
#      overlap within a shape.
#
# c) Hidden assumptions audit
#    - `_aligned_8`, `_angle_cycle`, and `_clear_in_original` still matter: the task
#      is still about recognizing which sparse same-color pixels form a legitimate
#      original shape versus a distractor.
#    - The original `_draw_line_bg_only` stage no longer matches the combo semantics.
#      Rendering is replaced by cross stamping centered on vertices.
#    - `out = g.copy()` remains valid because the input's vertex centers and any
#      foreign anchor centers must remain visible; cross arms are only painted onto
#      original background cells, so original foreign pixels are never erased.
#    - Colors with fewer than 2 pixels are still skipped as primary shapes. This is
#      necessary so 1-pixel distractors and 1-pixel shadow anchors are not promoted
#      into false shapes.
#    - The combo adds a stricter recognition assumption: we require the full set of
#      expected drawable edges after angle ordering. Otherwise a random multi-pixel
#      distractor could receive markers despite not corresponding to a valid shape.
#
# d) Placement legality
#    - The generator is still output-first in spirit: it constructs a local patch
#      whose output is the union of vertex crosses, then derives the sparse input by
#      keeping only vertex centers plus optional unchanged distractors.
#    - Patch bounding boxes must include every cross arm, otherwise local examples
#      would clip the intended output near the patch boundary.
#    - Placement uses `random_free_location` against the final patch output, so two
#      objects cannot overlap after marker expansion even if their sparse inputs look
#      far apart. Explicit marker-overlap checks inside each shape patch enforce the
#      same rule locally.
#
# e) Visual observability
#    - Across 2-4 training pairs, a human sees the same sparse vertex-only input but
#      no continuous trails in the output. Instead, every meaningful vertex becomes a
#      cross while unrelated sparse clutter stays unchanged.
#    - No invisible ordering feature such as raw color id is used. The only cues are
#      geometric: aligned vertex sets stay aligned, and each output mark is local.
#
# f) Visual cleanliness
#    - The generator keeps at least one-cell object spacing on the final output and
#      rejects merged markers, so the grid stays readable even with several shapes.
#    - Because the combo removes all edge filling, outputs are actually sparser than
#      the original task; that reduces accidental cross-line tangles.
#
# g) Change inferability
#    - Colors are preserved, not remapped. Shape colors still label shape vertices.
#      The only changed semantic is where those colors are painted.
#
# New solver-side assumptions introduced by the combo
# --------------------------------------------------
# 1. Intended shapes have enough spacing that five-cell vertex crosses do not merge.
# 2. Every non-distractor multi-pixel color group satisfies the original edge
#    legality checks under angle-cycle ordering; the generator guarantees this.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste, random_free_location
from arc_utils import get_bg as _bg_color

MARKER_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _marker_cells(r: int, c: int):
    for dr, dc in MARKER_OFFSETS:
        yield r + dr, c + dc


def _paint_marker_bg_only(out: np.ndarray, src: np.ndarray, r: int, c: int, col: int, bg: int) -> None:
    h, w = out.shape
    for rr, cc in _marker_cells(r, c):
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        src_v = int(src[rr, cc])
        if src_v == bg or src_v == col:
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


# Overview:
#   Recognize the same sparse vertex sets as the original task, but render only
#   local five-cell crosses at vertices.
# Input structure:
#   Colors with >=2 pixels are candidate shapes made of H/V/45deg-aligned
#   vertices; distractors remain sparse and non-shape-like.
# Rule:
#   Do not paint any connecting edge cells. For each valid shape, stamp a cross on
#   every vertex center. Ignore the original shadow-edge mechanism in the output:
#   non-vertex foreign pixels stay as they are. Never overwrite original foreign
#   pixels from the input.
# Key insight:
#   Geometry detection is unchanged from the original; only the rendering stage is
#   inverted from "fill the trail" to "amplify the endpoints".
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

        expected_edges = 1 if len(cycle) == 2 else len(cycle)
        if len(drawable_edges) != expected_edges:
            continue

        for r, c in cycle:
            _paint_marker_bg_only(out, g, r, c, col, bg)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 0
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

LINE_VERTICES_RANGE = (2, 6)
LINE_TOTAL_LENGTH = 15
LINE_MIN_GAP = 3
TRIANGLE_SIDE_RANGE = (3, 8)
RECT_H_RANGE = (3, 9)
RECT_W_RANGE = (3, 9)
DIAMOND_ARM_RANGE = (3, 6)
PENTAGON_ROOF_RANGE = (3, 5)
PENTAGON_WALL_RANGE = (3, 7)
HEXAGON_SLANT_RANGE = (3, 5)
HEXAGON_WALL_RANGE = (3, 6)

SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


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


def _marker_cell_set(center: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    return {(r + dr, c + dc) for dr, dc in MARKER_OFFSETS}


def _markers_pairwise_disjoint(centers: list[tuple[int, int]]) -> bool:
    seen: set[tuple[int, int]] = set()
    for center in centers:
        cells = _marker_cell_set(center)
        if seen & cells:
            return False
        seen |= cells
    return True


def _paint_marker_strict(out: np.ndarray, center: tuple[int, int], color: int) -> bool:
    h, w = out.shape
    for rr, cc in _marker_cells(center[0], center[1]):
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if out[rr, cc] not in (BG, color):
            return False
    for rr, cc in _marker_cells(center[0], center[1]):
        out[rr, cc] = color
    return True


def _patch_bbox(
    vertices: list[tuple[int, int]],
    shadow_centers: list[tuple[int, int]],
) -> tuple[int, int, int, int]:
    coords = set(vertices) | set(shadow_centers)
    for center in list(coords):
        coords.update(_marker_cell_set(center))
    r0, c0, r1, c1 = bbox_coords(np.array(sorted(coords), dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    if shadow_color is not None:
        return None

    vertices = _make_vertices()

    if not _markers_pairwise_disjoint(vertices):
        return None

    r0, c0, r1, c1 = _patch_bbox(vertices, [])
    h, w = r1 - r0 + 1, c1 - c0 + 1
    out = np.full((h, w), BG, dtype=int)
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for r, c in shifted_vertices:
        inp[r, c] = shape_color
        if not _paint_marker_strict(out, (r, c), shape_color):
            return None

    if not np.array_equal(main(inp), out):
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
