# Variant of 35ab12c3
# Dimension: D17 — 形状数量与密度 (meta)
# Change: 2-4个形状组，0-3个阴影，0-1个干扰组，在15-22的网格中保持>=1像素的缓冲间距... → 零缓冲间距（形状紧密相邻）

import math
import numpy as np

from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste

# Design analysis for the D17 variant (kept in-code because the generator relies on
# these assumptions directly).
#
# a) Concrete behavior:
#    Original instances keep a one-cell background moat between different groups, so
#    a shape vertex touching a different color almost always means "this is the shadow
#    anchor next to the start vertex". In the variant, two different shape groups are
#    intentionally allowed to be 8-neighbors, so a vertex may see foreign colors that
#    are unrelated to its own shadow. Example:
#      before: shape A vertex at (5,5), shadow anchor x at (5,6), no other group nearby
#      now:    shape A vertex at (5,5), shape B vertex at (4,6), real shadow x at (6,4)
#    The old solver would pick (4,6) as evidence that (5,5) is the seed and would
#    carry shape B's color as a fake shadow. The new solver therefore only treats a
#    neighboring color as a carried shadow if that color appears exactly once in the
#    whole input and its offset can project at least one translated edge for this
#    shape. Group adjacency still affects the input/output visually, but only color
#    information identifies ownership.
#
# b) Complexity calibration:
#    The variant changes density, not topology. Shapes, shadows, colors, and canvas
#    ranges stay the same. To make the density change observable, the generator forces
#    at least one pair of distinct shape colors to be 8-adjacent in the input. This is
#    enough to remove the clean visual separation without making every sample a tangled
#    mass of overlaps.
#
# c) Implicit-assumption audit for main():
#    - colors iteration is still safe because groups never overlap in the final output.
#    - _clear_in_original still assumes another non-bg pixel on a segment blocks that
#      segment. This remains correct under zero-buffer spacing: touching is allowed,
#      crossing through another group is not.
#    - the old seed selection "any different-color neighbor" is no longer safe.
#    - the old carried-neighbor collection "all different-color neighbors of the seed"
#      is no longer safe.
#    - drawing only on background is still safe because the generator forbids output
#      overlap between groups and preserves shape-before-shadow priority inside a group.
#
# d) Placement legality:
#    The generator now uses explicit candidate enumeration instead of padding=1 random
#    placement. A candidate is legal iff:
#    - its output cells do not overlap existing output cells;
#    - no singleton input marker (shadow anchor or 1-pixel distractor) is 8-adjacent
#      to any foreign input pixel, because that would create fake shadow evidence;
#    - shape vertices are allowed to become 8-neighbors of other shape vertices.
#
# e) Visual observability:
#    At least one shape-shape 8-neighbor contact is required. This guarantees that a
#    training set actually shows the new regime rather than occasionally falling back
#    to the original spaced layout.
#
# f) Visual cleanliness:
#    Only vertex-to-vertex touching is encouraged across groups. Singleton anchors and
#    distractors stay isolated, so the dense layout is readable instead of becoming a
#    cloud of ambiguous one-pixel markers.
#
# g) Color inference:
#    No color-id arithmetic or ordering is used for the transformation itself. Colors
#    only label groups, as in the source task; the new rule is purely spatial.
#
# h) New solver assumptions and generator guarantees:
#    - A real shadow anchor uses a color that appears exactly once in the whole input.
#    - Such singleton markers are kept 8-separated from foreign groups, so the solver
#      can safely interpret "singleton adjacent to one vertex" as shadow evidence.
#    - Shape-shape contact is therefore restricted to multi-pixel shape colors, which
#      is exactly the ambiguous boundary D17 wants humans to resolve by color.

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


def _aligned(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _step_dir(r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    dr = r1 - r0
    dc = c1 - c0
    sr = 0 if dr == 0 else (1 if dr > 0 else -1)
    sc = 0 if dc == 0 else (1 if dc > 0 else -1)
    return sr, sc


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _segment_cells(r0: int, c0: int, r1: int, c1: int):
    sr, sc = _step_dir(r0, c0, r1, c1)
    steps = max(abs(r1 - r0), abs(c1 - c0))
    for k in range(steps + 1):
        yield r0 + sr * k, c0 + sc * k


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


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


def _touches_exactly_one_vertex(pos: tuple[int, int], vertices: list[tuple[int, int]]) -> bool:
    rr, cc = pos
    hits = 0
    for vr, vc in vertices:
        if max(abs(vr - rr), abs(vc - cc)) == 1:
            hits += 1
            if hits > 1:
                return False
    return hits == 1


def _valid_projected_offset(
    g: np.ndarray,
    bg: int,
    h: int,
    w: int,
    edges: list[tuple[tuple[int, int], tuple[int, int]]],
    offset: tuple[int, int],
) -> bool:
    drn, dcn = offset
    for (r0, c0), (r1, c1) in edges:
        drs, dcs = _step_dir(r0, c0, r1, c1)
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
        if _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
            return True
    return False


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    color_counts = {col: int(np.sum(g == col)) for col in colors}

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

        carried_by_vertex: dict[int, list[tuple[int, int, int]]] = {}
        for i, (r, c) in enumerate(cycle):
            valid_neighbors: list[tuple[int, int, int]] = []
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                v = int(g[rr, cc])
                if v == bg or v == col or color_counts.get(v, 0) != 1:
                    continue
                if not _touches_exactly_one_vertex((rr, cc), cycle):
                    continue
                if not _valid_projected_offset(g, bg, h, w, drawable_edges, (drn, dcn)):
                    continue
                valid_neighbors.append((drn, dcn, v))
            if valid_neighbors:
                carried_by_vertex[i] = valid_neighbors

        if carried_by_vertex:
            start = min(carried_by_vertex, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

        cycle = cycle[start:] + cycle[:start]
        if drawable_edges:
            carried = carried_by_vertex.get(start, [])
        else:
            carried = []

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges = []
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

SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


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
    src = (out == shape_color)
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _coords_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, p)) for p in np.argwhere(mask)]


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> dict | None:
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
        nb_hits = sum(1 for rr, cc in neighbors8(anchor[0], anchor[1]) if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color)
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None

    touch_mask = (inp == shape_color)
    isolated_mask = (inp != BG) & ~touch_mask
    return {
        "kind": "shape",
        "inp": inp,
        "out": out,
        "touch_cells": _coords_from_mask(touch_mask),
        "isolated_cells": _coords_from_mask(isolated_mask),
        "out_cells": _coords_from_mask(out != BG),
    }


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
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
    else:
        patch = None
        for _ in range(DISTRACTOR_RETRIES):
            h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
            w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
            coords = _sample_distractor_coords(h, w, count)
            if coords is None:
                continue
            patch = np.full((h, w), BG, dtype=int)
            for r, c in coords:
                patch[r, c] = color
            break
        if patch is None:
            patch = np.full((1, 1), color, dtype=int)

    occ = _coords_from_mask(patch != BG)
    return {
        "kind": "distractor",
        "inp": patch.copy(),
        "out": patch.copy(),
        "touch_cells": [],
        "isolated_cells": occ,
        "out_cells": occ,
    }


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


def _translate(coords: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r + r0, c + c0) for r, c in coords]


def _touches_existing(coords: list[tuple[int, int]], existing: set[tuple[int, int]]) -> bool:
    if not existing or not coords:
        return False
    for r, c in coords:
        for dr, dc in DIR8:
            if (r + dr, c + dc) in existing:
                return True
    return False


def _candidate_locations(
    canvas_shape: tuple[int, int],
    patch: dict,
    state: dict[str, set[tuple[int, int]]],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]:
    H, W = canvas_shape
    h, w = patch["out"].shape
    out_occ = state["out_occ"]
    input_occ = state["input_occ"]
    isolated_occ = state["isolated_occ"]
    touch_occ = state["touch_occ"]

    out_local = patch["out_cells"]
    touch_local = patch["touch_cells"]
    isolated_local = patch["isolated_cells"]

    touch_candidates: list[tuple[int, int]] = []
    output_touch_candidates: list[tuple[int, int]] = []
    general_candidates: list[tuple[int, int]] = []

    for r0 in range(H - h + 1):
        for c0 in range(W - w + 1):
            out_cells = _translate(out_local, r0, c0)
            if any(cell in out_occ for cell in out_cells):
                continue

            isolated_cells = _translate(isolated_local, r0, c0)
            if any(_touches_existing([cell], input_occ) for cell in isolated_cells):
                continue

            touch_cells = _translate(touch_local, r0, c0)
            if isolated_occ and touch_cells:
                bad = False
                for cell in touch_cells:
                    if _touches_existing([cell], isolated_occ):
                        bad = True
                        break
                if bad:
                    continue

            shape_touch = _touches_existing(touch_cells, touch_occ)
            output_touch = _touches_existing(out_cells, out_occ)
            if patch["kind"] == "shape" and shape_touch:
                touch_candidates.append((r0, c0))
            elif patch["kind"] == "shape" and output_touch:
                output_touch_candidates.append((r0, c0))
            else:
                general_candidates.append((r0, c0))

    return touch_candidates, output_touch_candidates, general_candidates


def _choose_location(
    canvas_shape: tuple[int, int],
    patch: dict,
    state: dict[str, set[tuple[int, int]]],
    prefer_touch: bool,
) -> tuple[int, int] | None:
    touch_candidates, output_touch_candidates, general_candidates = _candidate_locations(canvas_shape, patch, state)
    pools: list[list[tuple[int, int]]] = []
    if prefer_touch:
        pools = [touch_candidates, output_touch_candidates, general_candidates]
    else:
        pools = [general_candidates, output_touch_candidates, touch_candidates]
    for pool in pools:
        if pool:
            return pool[int(np.random.randint(len(pool)))]
    return None


def _place_patch(
    inp: np.ndarray,
    out: np.ndarray,
    patch: dict,
    state: dict[str, set[tuple[int, int]]],
    prefer_touch: bool,
) -> bool:
    loc = _choose_location(out.shape, patch, state, prefer_touch=prefer_touch)
    if loc is None:
        return False

    r0, c0 = loc
    paste(out, patch["out"], r0, c0, transparent=BG)
    paste(inp, patch["inp"], r0, c0, transparent=BG)

    state["out_occ"].update(_translate(patch["out_cells"], r0, c0))
    state["input_occ"].update(_translate(patch["touch_cells"], r0, c0))
    state["input_occ"].update(_translate(patch["isolated_cells"], r0, c0))
    state["touch_occ"].update(_translate(patch["touch_cells"], r0, c0))
    state["isolated_occ"].update(_translate(patch["isolated_cells"], r0, c0))
    return True


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _shape_touch_count(inp: np.ndarray, shape_colors: list[int]) -> int:
    total = 0
    for i, c1 in enumerate(shape_colors):
        pts1 = [tuple(map(int, p)) for p in np.argwhere(inp == c1)]
        s1 = set(pts1)
        for c2 in shape_colors[i + 1:]:
            pts2 = set(tuple(map(int, p)) for p in np.argwhere(inp == c2))
            if not s1 or not pts2:
                continue
            if any((r + dr, c + dc) in pts2 for r, c in s1 for dr, dc in DIR8):
                total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)
    state = {
        "out_occ": set(),
        "input_occ": set(),
        "touch_occ": set(),
        "isolated_occ": set(),
    }

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    total_colors = shape_count + len(shadow_colors) + int(has_distractor)

    placed_shapes = 0
    for idx, shape_color in enumerate(shape_colors):
        placed = False
        prefer_touch = placed_shapes > 0
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            if _place_patch(inp, out, patch, state, prefer_touch=prefer_touch):
                placed = True
                placed_shapes += 1
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _place_patch(inp, out, _make_distractor_patch(distractor_color), state, prefer_touch=False):
            return None

    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if _shape_touch_count(inp, shape_colors) < 1:
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
