# Combo variant of 35ab12c3 (index 28)
# combo_spec: D5/3+D12/2
# Name: 阴影为缩放副本+网格扩展
#
# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior / trace
#    - Base task: same-color vertices define a line/polygon; one foreign 8-neighbor next
#      to one vertex used to mean "translate the whole path by this offset" and the output
#      size stayed fixed.
#    - Here the foreign pixel is no longer copied as a parallel offset. It is a singleton
#      scaling center. If that singleton lies 1 or 2 cells away from exactly one vertex on
#      an H/V/45 ray, the distance encodes a scale:
#        distance 1 -> scale 2
#        distance 2 -> scale 3
#      The output draws the original shape plus a shadow-colored scaled copy of that shape
#      around the singleton center. The singleton itself is an input-only cue, not a forced
#      output pixel.
#    - Small trace:
#      base line vertices: (4,4) and (4,8)
#      shadow marker:      (3,3)  (distance 1 from vertex (4,4), so scale 2)
#      old rule: draw a parallel copy shifted by (-1,-1)
#      new rule: scale both endpoints about (3,3):
#        (4,4) -> (5,5)
#        (4,8) -> (5,13)
#      so the shadow is a longer same-orientation copy, not a unit translation.
#    - D12 is fused into the same geometric step: if the scaled copy reaches column 13 while
#      the input width is only 10, the output widens to include it instead of clipping it.
#
# b) Complexity calibration
#    - Scale choices are restricted to {2,3}. This keeps the hypothesis space small enough
#      for humans while still making "distance matters" visible across examples.
#    - Every generated grid has exactly one scaled shadow shape, so the fused behavior is
#      always exercised; other shapes stay plain to preserve contrast.
#    - Shadow-capable shapes are biased toward smaller lines / simple polygons. That keeps
#      the expanded canvas readable and avoids four-side spill chaos.
#
# c) Hidden-assumption audit of the original solver
#    - `out = g.copy()` is no longer valid: the shadow marker singleton would become an
#      orphan output pixel, while in this variant it is only a cue. Output must be rebuilt.
#    - Seed selection by foreign 8-neighbors is invalid: the cue may now be 1 or 2 cells
#      away and its direction does not define a translation.
#    - The old second drawing phase assumed one shared offset for all edges. That is replaced
#      by endpoint-wise integer scaling around the singleton center.
#    - Fixed-size output is invalid. The solver now computes all transformed endpoints first,
#      then crops a workspace to "original input window + actual spill on each side".
#
# d) Placement legality
#    - The generator still works output-first per object patch. A local patch stores:
#        * input bbox: vertices + one singleton scaling center
#        * output bbox: original input bbox plus whatever the scaled shadow spills
#    - Global placement uses the output patch mask, not only the input patch, so scaled
#      shadows cannot overlap neighboring objects even when they spill outside the input.
#    - At least one object is forced to spill beyond the input border, making D12 observable.
#
# e) Visual observability
#    - A singleton of a new color sits on a short H/V/45 ray from one vertex, and farther
#      singletons create visibly larger similar shadows. Because the shadow is larger rather
#      than merely shifted, the new D5 rule is distinguishable in 2-3 examples.
#    - Output expansion is coupled to that same shadow; humans can see that the canvas grows
#      exactly on the sides where the scaled copy continues.
#
# f) Cleanliness
#    - Generated shadow objects are limited to at most two spill sides locally, and all other
#      objects preferentially stay fully inside the input window. This keeps the crop signal
#      attributable to one scaled shadow instead of a crowded multi-object border.
#
# g) Color inference
#    - No hidden color-id mapping is introduced. Shape colors still identify which vertices
#      connect, and the singleton's own color becomes the shadow color of the scaled copy.
#      The only new cue is geometric: aligned distance to one vertex.
#
# h) New solver assumptions introduced here
#    - A shadow marker color appears as exactly one pixel.
#    - Its center is aligned to exactly one vertex of exactly one shape at distance 1 or 2.
#    - Distractor colors are generated with >=2 pixels and are not shape-valid under the
#      same H/V/45 cycle test, so rebuilding output from classified groups is safe.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
CARDINAL_SIDES = ("top", "bottom", "left", "right")
MARKER_DIST_TO_SCALE = {1: 2, 2: 3}


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


def _clear_in_input(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
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
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v not in allow:
                return False
    return True


def _clear_in_set(a: tuple[int, int], b: tuple[int, int], blocked: set[tuple[int, int]]) -> bool:
    first = True
    for cell in _segment_cells(a[0], a[1], b[0], b[1]):
        if first:
            first = False
            continue
        if cell == b:
            break
        if cell in blocked:
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


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(vertices) == 2:
        return [(vertices[0], vertices[1])]
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _scale_point_about(point: tuple[int, int], center: tuple[int, int], scale: int) -> tuple[int, int]:
    return (
        int(center[0] + scale * (point[0] - center[0])),
        int(center[1] + scale * (point[1] - center[1])),
    )


def _extract_shape(g: np.ndarray, color: int, bg: int):
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == color)]
    if len(pts_rc) < 2:
        return None
    cycle = _angle_cycle(pts_rc)
    edges = _shape_edges(cycle)
    drawable: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for a, b in edges:
        if not _aligned_8(a[0], a[1], b[0], b[1]):
            return None
        if not _clear_in_input(g, a[0], a[1], b[0], b[1], bg, allow={color}):
            return None
        drawable.append((a, b))
    return {"color": int(color), "vertices": cycle, "edges": drawable}


def _shadow_candidates_for_shape(
    g: np.ndarray,
    shape: dict,
    singleton_positions: dict[int, tuple[int, int]],
    bg: int,
) -> list[dict]:
    candidates: list[dict] = []
    vertices = shape["vertices"]
    for shadow_color, center in singleton_positions.items():
        hits: list[tuple[int, tuple[int, int]]] = []
        for vertex in vertices:
            if not _aligned_8(vertex[0], vertex[1], center[0], center[1]):
                continue
            dist = max(abs(center[0] - vertex[0]), abs(center[1] - vertex[1]))
            if dist not in MARKER_DIST_TO_SCALE:
                continue
            if not _clear_in_input(g, vertex[0], vertex[1], center[0], center[1], bg, allow=set()):
                continue
            hits.append((int(dist), vertex))
        if len(hits) != 1:
            continue

        dist, anchor = hits[0]
        scale = MARKER_DIST_TO_SCALE[int(dist)]
        shadow_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        valid = True
        for a, b in shape["edges"]:
            aa = _scale_point_about(a, center, scale)
            bb = _scale_point_about(b, center, scale)
            if not _aligned_8(aa[0], aa[1], bb[0], bb[1]):
                valid = False
                break
            if not _clear_in_input(g, aa[0], aa[1], bb[0], bb[1], bg, allow={shape["color"], shadow_color}):
                valid = False
                break
            shadow_edges.append((aa, bb))
        if not valid:
            continue

        candidates.append(
            {
                "shape_color": int(shape["color"]),
                "shadow_color": int(shadow_color),
                "center": (int(center[0]), int(center[1])),
                "anchor": (int(anchor[0]), int(anchor[1])),
                "distance": int(dist),
                "scale": int(scale),
                "edges": shadow_edges,
            }
        )
    return candidates


def _workspace_bounds_from_segments(
    h: int,
    w: int,
    shapes: list[dict],
    attached_shadows: dict[int, list[dict]],
) -> tuple[int, int, int, int]:
    min_r, min_c = 0, 0
    max_r, max_c = h - 1, w - 1
    for shape in shapes:
        for a, b in shape["edges"]:
            min_r = min(min_r, a[0], b[0])
            min_c = min(min_c, a[1], b[1])
            max_r = max(max_r, a[0], b[0])
            max_c = max(max_c, a[1], b[1])
    for shadow_list in attached_shadows.values():
        for shadow in shadow_list:
            for a, b in shadow["edges"]:
                min_r = min(min_r, a[0], b[0])
                min_c = min(min_c, a[1], b[1])
                max_r = max(max_r, a[0], b[0])
                max_c = max(max_c, a[1], b[1])
    return int(min_r), int(min_c), int(max_r), int(max_c)


def _solve_with_meta(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    colors = colors_present(g, exclude=[bg])
    shapes: list[dict] = []
    shape_colors: set[int] = set()
    for color in colors:
        shape = _extract_shape(g, int(color), bg)
        if shape is None:
            continue
        shapes.append(shape)
        shape_colors.add(int(color))

    singleton_positions = {
        int(color): tuple(map(int, np.argwhere(g == color)[0]))
        for color in colors
        if int(color) not in shape_colors and int(np.sum(g == color)) == 1
    }

    candidate_map: dict[int, list[tuple[int, dict]]] = {}
    for idx, shape in enumerate(shapes):
        for candidate in _shadow_candidates_for_shape(g, shape, singleton_positions, bg):
            candidate_map.setdefault(int(candidate["shadow_color"]), []).append((idx, candidate))

    attached_shadows: dict[int, list[dict]] = {idx: [] for idx in range(len(shapes))}
    consumed_shadow_colors: set[int] = set()
    for shadow_color, uses in candidate_map.items():
        if len(uses) != 1:
            continue
        idx, candidate = uses[0]
        attached_shadows[idx].append(candidate)
        consumed_shadow_colors.add(int(shadow_color))

    min_r, min_c, max_r, max_c = _workspace_bounds_from_segments(h, w, shapes, attached_shadows)
    row_shift = -min_r
    col_shift = -min_c
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)

    for color in colors:
        if int(color) in shape_colors or int(color) in consumed_shadow_colors:
            continue
        for rr, cc in np.argwhere(g == color):
            out[int(rr) + row_shift, int(cc) + col_shift] = int(color)

    for shape in shapes:
        for a, b in shape["edges"]:
            _draw_line_bg_only(
                out,
                a[0] + row_shift,
                a[1] + col_shift,
                b[0] + row_shift,
                b[1] + col_shift,
                int(shape["color"]),
                bg,
            )

    for shadow_list in attached_shadows.values():
        for shadow in shadow_list:
            for a, b in shadow["edges"]:
                _draw_line_bg_only(
                    out,
                    a[0] + row_shift,
                    a[1] + col_shift,
                    b[0] + row_shift,
                    b[1] + col_shift,
                    int(shadow["shadow_color"]),
                    bg,
                )

    meta = {
        "shadow_scales": [int(shadow["scale"]) for shadow_list in attached_shadows.values() for shadow in shadow_list],
        "shadow_distances": [int(shadow["distance"]) for shadow_list in attached_shadows.values() for shadow in shadow_list],
        "consumed_shadow_colors": consumed_shadow_colors,
        "spill": {
            "top": int(max(0, -min_r)),
            "bottom": int(max(0, max_r - (h - 1))),
            "left": int(max(0, -min_c)),
            "right": int(max(0, max_c - (w - 1))),
        },
    }
    return out, meta


# 1. Overview
#    Connect each valid same-color vertex set into its H/V/45 shape, then treat a singleton
#    foreign pixel aligned 1 or 2 cells away from one vertex as the center of a scaled shadow.
# 2. Input structure:
#    - Shape colors: 2+ pixels that form a valid line/polygon cycle under the original rule.
#    - Shadow colors: exactly one pixel, aligned to exactly one shape vertex at distance 1 or 2.
#    - Distractors: non-shape colors left unchanged.
# 3. Rule:
#    - Draw the main shape normally.
#    - If a singleton marker sits distance d from one vertex, draw a shadow-colored copy of the
#      whole shape scaled by factor d+1 about that singleton center.
#    - Return the original input window plus any extra border needed to contain the scaled copy.
# 4. Key insight:
#    The singleton no longer encodes a translation vector; its aligned distance to one vertex
#    encodes the magnification, and the final crop must preserve the full spilled shadow.
def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_meta(np.asarray(grid, dtype=int))[0]


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)  # 2-3 shapes
DISTRACTOR_PROB = 0.35
PLACEMENT_PADDING = 1
OUTPUT_PAD = 40

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
SHAPE_WEIGHTS = np.array([0.36, 0.24, 0.24, 0.16], dtype=float)

SHADOW_LINE_VERTICES_RANGE = (2, 6)
SHADOW_LINE_TOTAL_LENGTH = 9
SHADOW_TRIANGLE_SIDE_RANGE = (3, 7)
SHADOW_RECT_H_RANGE = (3, 7)
SHADOW_RECT_W_RANGE = (3, 7)
SHADOW_DIAMOND_ARM_RANGE = (2, 5)
SHADOW_SHAPE_WEIGHTS = np.array([0.42, 0.24, 0.22, 0.12], dtype=float)

DISTRACTOR_COUNT_RANGE = (2, 4)  # 2-3 pixels only; reserve singletons for shadows
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 60

MAX_LOCAL_SPILL_SIDES = 2
MIN_VISIBLE_SHADOW_CELLS = 5
PATCH_BUILD_RETRIES = 120
GENERATE_RETRIES = 600


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


def _build_line(vertices_range: tuple[int, int], total_length: int) -> list[tuple[int, int]]:
    n_vertices = int(np.random.randint(*vertices_range))
    remaining = int(total_length)
    pos = [0]
    for i in range(n_vertices - 1):
        steps_left = n_vertices - 2 - i
        gap_hi = remaining - LINE_MIN_GAP * steps_left
        gap = int(np.random.randint(LINE_MIN_GAP, gap_hi + 1))
        pos.append(pos[-1] + gap)
        remaining -= gap
    return _transform_coords([(0, x) for x in pos])


def _build_triangle(side_range: tuple[int, int]) -> list[tuple[int, int]]:
    side = int(np.random.randint(*side_range))
    return _transform_coords([(0, 0), (0, side), (side, side)])


def _build_rectangle(h_range: tuple[int, int], w_range: tuple[int, int]) -> list[tuple[int, int]]:
    hh = int(np.random.randint(*h_range))
    ww = int(np.random.randint(*w_range))
    return _transform_coords([(0, 0), (0, ww), (hh, ww), (hh, 0)])


def _build_diamond(arm_range: tuple[int, int]) -> list[tuple[int, int]]:
    arm = int(np.random.randint(*arm_range))
    return _transform_coords([(0, arm), (arm, 2 * arm), (2 * arm, arm), (arm, 0)])


def _make_vertices(*, shadowable: bool) -> list[tuple[int, int]]:
    if shadowable:
        weights = SHADOW_SHAPE_WEIGHTS / SHADOW_SHAPE_WEIGHTS.sum()
        idx = int(np.random.choice(4, p=weights))
        if idx == 0:
            return _build_line(SHADOW_LINE_VERTICES_RANGE, SHADOW_LINE_TOTAL_LENGTH)
        if idx == 1:
            return _build_triangle(SHADOW_TRIANGLE_SIDE_RANGE)
        if idx == 2:
            return _build_rectangle(SHADOW_RECT_H_RANGE, SHADOW_RECT_W_RANGE)
        return _build_diamond(SHADOW_DIAMOND_ARM_RANGE)

    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(4, p=weights))
    if idx == 0:
        return _build_line(LINE_VERTICES_RANGE, LINE_TOTAL_LENGTH)
    if idx == 1:
        return _build_triangle(TRIANGLE_SIDE_RANGE)
    if idx == 2:
        return _build_rectangle(RECT_H_RANGE, RECT_W_RANGE)
    return _build_diamond(DIAMOND_ARM_RANGE)


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


def _make_distractor_patch(color: int) -> dict:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    for _ in range(DISTRACTOR_RETRIES):
        hh = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        ww = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(hh, ww, count)
        if coords is None:
            continue
        patch = np.full((hh, ww), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = int(color)
        return {
            "patch_in": patch.copy(),
            "patch_out": patch.copy(),
            "extras": {"top": 0, "bottom": 0, "left": 0, "right": 0},
            "mask": patch != BG,
            "has_shadow": False,
        }
    patch = np.full((1, 2), int(color), dtype=int)
    return {
        "patch_in": patch.copy(),
        "patch_out": patch.copy(),
        "extras": {"top": 0, "bottom": 0, "left": 0, "right": 0},
        "mask": patch != BG,
        "has_shadow": False,
    }


def _shadow_spec(vertices: list[tuple[int, int]], shape_cells: set[tuple[int, int]]) -> dict | None:
    edges = _shape_edges(vertices)
    dirs = list(DIR8)
    np.random.shuffle(dirs)
    dists = list(MARKER_DIST_TO_SCALE.keys())
    np.random.shuffle(dists)
    anchors = list(vertices)
    np.random.shuffle(anchors)

    for anchor in anchors:
        for dist in dists:
            for dr, dc in dirs:
                center = (anchor[0] + dist * dr, anchor[1] + dist * dc)
                if center in shape_cells:
                    continue
                if not _clear_in_set(anchor, center, shape_cells - {anchor}):
                    continue

                unique_hits = []
                for vertex in vertices:
                    if not _aligned_8(vertex[0], vertex[1], center[0], center[1]):
                        continue
                    vd = max(abs(center[0] - vertex[0]), abs(center[1] - vertex[1]))
                    if vd not in MARKER_DIST_TO_SCALE:
                        continue
                    if not _clear_in_set(vertex, center, shape_cells - {vertex}):
                        continue
                    unique_hits.append(vertex)
                if unique_hits != [anchor]:
                    continue

                scale = MARKER_DIST_TO_SCALE[int(dist)]
                shadow_cells: set[tuple[int, int]] = set()
                shadow_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
                valid = True
                for a, b in edges:
                    aa = _scale_point_about(a, center, scale)
                    bb = _scale_point_about(b, center, scale)
                    if not _aligned_8(aa[0], aa[1], bb[0], bb[1]):
                        valid = False
                        break
                    shadow_edges.append((aa, bb))
                    shadow_cells.update(_gen_segment_cells(aa, bb))
                if not valid:
                    continue

                visible_cells = shadow_cells - shape_cells
                if len(visible_cells) < MIN_VISIBLE_SHADOW_CELLS:
                    continue

                input_coords = list(vertices) + [center]
                in_r0, in_c0, in_r1, in_c1 = bbox_coords(np.array(input_coords, dtype=int))
                out_coords = list(shape_cells | shadow_cells)
                out_r0, out_c0, out_r1, out_c1 = bbox_coords(np.array(out_coords, dtype=int))
                extras = {
                    "top": int(max(0, in_r0 - out_r0)),
                    "bottom": int(max(0, out_r1 - in_r1)),
                    "left": int(max(0, in_c0 - out_c0)),
                    "right": int(max(0, out_c1 - in_c1)),
                }
                spill_sides = sum(v > 0 for v in extras.values())
                if spill_sides == 0 or spill_sides > MAX_LOCAL_SPILL_SIDES:
                    continue

                return {
                    "center": center,
                    "anchor": anchor,
                    "distance": int(dist),
                    "scale": int(scale),
                    "edges": shadow_edges,
                    "cells": shadow_cells,
                    "extras": extras,
                }
    return None


def _build_shape_patch(shape_color: int, shadow_color: int | None) -> dict | None:
    for _ in range(PATCH_BUILD_RETRIES):
        vertices = _make_vertices(shadowable=shadow_color is not None)
        shape_cells = _shape_cells(vertices)
        shadow = None
        if shadow_color is not None:
            shadow = _shadow_spec(vertices, shape_cells)
            if shadow is None:
                continue

        input_coords = list(vertices)
        if shadow is not None:
            input_coords.append(shadow["center"])

        in_r0, in_c0, in_r1, in_c1 = bbox_coords(np.array(input_coords, dtype=int))
        in_h = int(in_r1 - in_r0 + 1)
        in_w = int(in_c1 - in_c0 + 1)

        extras = {"top": 0, "bottom": 0, "left": 0, "right": 0}
        output_cells = set(shape_cells)
        if shadow is not None:
            output_cells |= set(shadow["cells"])
            extras = dict(shadow["extras"])

        out_h = int(in_h + extras["top"] + extras["bottom"])
        out_w = int(in_w + extras["left"] + extras["right"])
        patch_in = np.full((in_h, in_w), BG, dtype=int)
        patch_out = np.full((out_h, out_w), BG, dtype=int)

        for r, c in vertices:
            patch_in[r - in_r0, c - in_c0] = int(shape_color)
        if shadow is not None:
            cr, cc = shadow["center"]
            if patch_in[cr - in_r0, cc - in_c0] != BG:
                continue
            patch_in[cr - in_r0, cc - in_c0] = int(shadow_color)

        if shadow is not None:
            for r, c in shadow["cells"]:
                rr = int(r - in_r0 + extras["top"])
                cc = int(c - in_c0 + extras["left"])
                patch_out[rr, cc] = int(shadow_color)
        for r, c in shape_cells:
            rr = int(r - in_r0 + extras["top"])
            cc = int(c - in_c0 + extras["left"])
            patch_out[rr, cc] = int(shape_color)

        if not np.array_equal(main(patch_in), patch_out):
            continue

        return {
            "patch_in": patch_in,
            "patch_out": patch_out,
            "extras": extras,
            "mask": patch_out != BG,
            "has_shadow": shadow is not None,
            "shadow_scale": None if shadow is None else int(shadow["scale"]),
            "shadow_distance": None if shadow is None else int(shadow["distance"]),
        }
    return None


def _sample_canvas_size() -> tuple[int, int]:
    hh = int(np.random.randint(*CANVAS_H_RANGE))
    ww = int(np.clip(hh + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return hh, ww


def _can_stamp_with_padding(occ: np.ndarray, mask: np.ndarray, r0: int, c0: int, padding: int) -> bool:
    hh, ww = occ.shape
    for rr, cc in np.argwhere(mask):
        ar = int(r0 + rr)
        ac = int(c0 + cc)
        r1 = max(0, ar - padding)
        r2 = min(hh, ar + padding + 1)
        c1 = max(0, ac - padding)
        c2 = min(ww, ac + padding + 1)
        if np.any(occ[r1:r2, c1:c2]):
            return False
    return True


def _spill_sides_for_placement(H: int, W: int, obj: dict, r: int, c: int) -> set[str]:
    in_h, in_w = obj["patch_in"].shape
    extras = obj["extras"]
    sides: set[str] = set()
    if int(r) - int(extras["top"]) < 0:
        sides.add("top")
    if int(c) - int(extras["left"]) < 0:
        sides.add("left")
    if int(r) + int(in_h) + int(extras["bottom"]) > H:
        sides.add("bottom")
    if int(c) + int(in_w) + int(extras["right"]) > W:
        sides.add("right")
    return sides


def _find_placement(inp: np.ndarray, out_ws: np.ndarray, obj: dict, *, require_spill: bool) -> tuple[int, int] | None:
    H, W = inp.shape
    in_h, in_w = obj["patch_in"].shape
    out_h, out_w = obj["patch_out"].shape
    extras = obj["extras"]
    occ = out_ws != BG

    candidates_with_spill: list[tuple[int, int]] = []
    candidates_without_spill: list[tuple[int, int]] = []

    row_values = np.random.permutation(H - in_h + 1)
    col_values = np.random.permutation(W - in_w + 1)
    for r in row_values:
        for c in col_values:
            out_r = int(r - extras["top"] + OUTPUT_PAD)
            out_c = int(c - extras["left"] + OUTPUT_PAD)
            if out_r < 0 or out_c < 0 or out_r + out_h > out_ws.shape[0] or out_c + out_w > out_ws.shape[1]:
                continue
            if not _can_stamp_with_padding(occ, obj["mask"], out_r, out_c, PLACEMENT_PADDING):
                continue
            spill_sides = _spill_sides_for_placement(H, W, obj, int(r), int(c))
            if spill_sides:
                candidates_with_spill.append((int(r), int(c)))
            else:
                candidates_without_spill.append((int(r), int(c)))

    if require_spill:
        if not candidates_with_spill:
            return None
        return candidates_with_spill[int(np.random.randint(len(candidates_with_spill)))]

    if candidates_without_spill:
        return candidates_without_spill[int(np.random.randint(len(candidates_without_spill)))]
    if candidates_with_spill:
        return candidates_with_spill[int(np.random.randint(len(candidates_with_spill)))]
    return None


def _workspace_to_output(workspace: np.ndarray, h: int, w: int) -> tuple[np.ndarray, set[str]]:
    coords = np.argwhere(workspace != BG)
    if coords.size == 0:
        return workspace[OUTPUT_PAD : OUTPUT_PAD + h, OUTPUT_PAD : OUTPUT_PAD + w].copy(), set()

    min_r, min_c = coords.min(axis=0)
    max_r, max_c = coords.max(axis=0)
    top = int(max(0, OUTPUT_PAD - int(min_r)))
    left = int(max(0, OUTPUT_PAD - int(min_c)))
    bottom = int(max(0, int(max_r) - (OUTPUT_PAD + h - 1)))
    right = int(max(0, int(max_c) - (OUTPUT_PAD + w - 1)))
    sides: set[str] = set()
    if top > 0:
        sides.add("top")
    if bottom > 0:
        sides.add("bottom")
    if left > 0:
        sides.add("left")
    if right > 0:
        sides.add("right")
    return (
        workspace[
            OUTPUT_PAD - top : OUTPUT_PAD + h + bottom,
            OUTPUT_PAD - left : OUTPUT_PAD + w + right,
        ].copy(),
        sides,
    )


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    inp = np.full((H, W), BG, dtype=int)
    out_ws = np.full((H + 2 * OUTPUT_PAD, W + 2 * OUTPUT_PAD), BG, dtype=int)

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    total_colors = shape_count + 1 + int(has_distractor)
    if total_colors > len(COLORS):
        return None

    palette = list(np.random.permutation(COLORS)[:total_colors])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_color = int(palette[shape_count])
    distractor_color = int(palette[-1]) if has_distractor else None

    objects: list[dict] = []
    shadow_idx = int(np.random.randint(shape_count))
    for idx, shape_color in enumerate(shape_colors):
        obj = _build_shape_patch(shape_color, shadow_color if idx == shadow_idx else None)
        if obj is None:
            return None
        objects.append(obj)

    if has_distractor and distractor_color is not None:
        objects.append(_make_distractor_patch(distractor_color))

    order = list(range(len(objects)))
    order.sort(key=lambda i: (not objects[i]["has_shadow"], -(objects[i]["patch_out"].shape[0] * objects[i]["patch_out"].shape[1])))

    spill_seen = False
    shadow_scale = None
    shadow_distance = None
    for idx in order:
        obj = objects[idx]
        require_spill = bool(obj["has_shadow"] and not spill_seen)
        loc = _find_placement(inp, out_ws, obj, require_spill=require_spill)
        if loc is None:
            return None
        r, c = loc
        paste(inp, obj["patch_in"], r, c, transparent=BG)
        out_r = int(r - obj["extras"]["top"] + OUTPUT_PAD)
        out_c = int(c - obj["extras"]["left"] + OUTPUT_PAD)
        paste(out_ws, obj["patch_out"], out_r, out_c, transparent=BG)
        spill_sides = _spill_sides_for_placement(H, W, obj, r, c)
        if obj["has_shadow"]:
            spill_seen = spill_seen or bool(spill_sides)
            shadow_scale = int(obj["shadow_scale"])
            shadow_distance = int(obj["shadow_distance"])

    expected_out, expected_sides = _workspace_to_output(out_ws, H, W)
    actual_out, meta = _solve_with_meta(inp)

    if not spill_seen:
        return None
    if shadow_scale not in {2, 3} or shadow_distance not in {1, 2}:
        return None
    if meta["shadow_scales"] != [shadow_scale]:
        return None
    if meta["shadow_distances"] != [shadow_distance]:
        return None
    if expected_sides != {side for side, amount in meta["spill"].items() if amount > 0}:
        return None
    if get_bg(inp) != BG or get_bg(expected_out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if len(colors_present(expected_out, exclude=[BG])) < shape_count + int(has_distractor) + 1:
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
