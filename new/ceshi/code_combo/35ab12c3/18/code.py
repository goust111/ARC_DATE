# Combo variant of 35ab12c3 (index 18)
# combo_spec: D1/3+D3/2
# Name: 颜色序列配对+行主序连接
#
# Design analysis for the combo
# a) Concrete behavior and trace:
#    - Original task: each non-background color formed its own same-color vertex group.
#      The solver angle-sorted that one-color group, usually closed it into a cycle for
#      3+ vertices, drew the main line in that same color, then optionally copied a
#      parallel shadow from one seed vertex's foreign-color neighbor.
#    - Combo: colors no longer define separate drawable objects one by one. Instead,
#      drawable colors are first sorted by visible palette id among colors that have at
#      least two pixels, then paired consecutively: `(c0,c1)`, `(c2,c3)`, ...
#      Each paired two-color set is treated as one object. Its vertices are merged,
#      sorted by row-major order `(r,c)`, and connected consecutively as an open
#      polyline. The segment from vertex `v_i` to `v_{i+1}` is painted in `v_i`'s color,
#      so the output is a genuinely cross-color polyline rather than two monochrome
#      outlines.
#    - Small trace:
#      suppose drawable colors are `[1,2,3,4]`, giving pairs `(1,2)` and `(3,4)`.
#      For pair `(1,2)`, the input vertices in row-major order are
#      `[(0,0,1), (0,4,2), (3,4,1), (3,8,2)]`. Original logic would split them by color:
#      color 1 uses `[(0,0),(3,4)]`, color 2 uses `[(0,4),(3,8)]`, and both groups act
#      independently. Combo merges them into one ordered path
#      `(0,0)->(0,4)->(3,4)->(3,8)`. The first segment is filled in color 1, the second
#      in color 2, the third in color 1; the color-2 endpoints stay color 2 because the
#      drawing only fills background cells. If a third-color marker sits next to the
#      first row-major vertex, it still produces a parallel shadow copy of this merged
#      path.
# b) Complexity calibration:
#    - Pairing by color value is not directly geometric, so the examples must make that
#      relation repetitive without hard-coding one specific palette. The generator
#      therefore samples four drawable colors per example, sorts them, and pairs them as
#      `(c0,c1)` and `(c2,c3)`, while reserving the remaining colors for optional
#      shadows or single-pixel distractors. Across examples the observer repeatedly sees
#      "the first two active colors" fuse into one path and "the next two" fuse into
#      another, even when the actual color ids change.
#    - If row-major order were applied to arbitrary scattered two-color points, many
#      consecutive pairs would be non-aligned and the effect would degenerate to sparse
#      dots. The generator therefore constructs explicit 4-5 vertex open-path templates
#      whose row-major order already is the intended path order.
#    - If one color occupied all early vertices and the other all late vertices, the
#      "cross-color" nature would be visually weak. The generator alternates the two
#      colors along the intended path order, making the merged-path interpretation
#      visibly necessary.
# c) Hidden-assumption audit against the original main():
#    - `for col in colors:` originally assumed each color is a full object. This fails
#      now because one drawable object spans two colors. The combo replaces the outer
#      loop with sequential pairs of drawable colors.
#    - `_angle_cycle(...)` assumed cyclic symmetry inside one-color groups. For an open
#      path across two colors, rotating the order changes the object. We replace it with
#      one stable visible ordering: merged row-major order.
#    - `cycle = cycle[start:] + cycle[:start]` was harmless only for closed cycles. It
#      would corrupt an open path, so shadow-anchor choice must be decoupled from path
#      order. The combo always uses the first row-major merged vertex as the only anchor
#      candidate for carried shadows.
#    - `_clear_in_original(..., allow={col})` assumed interior cells may only contain
#      the current color. After fusion, the intended path spans two colors, so the
#      interior check must allow both pair colors.
#    - The original carried-neighbor logic treated any non-`col` neighbor as a shadow
#      source. Here the partner color is part of the same object, not a shadow. The new
#      code therefore excludes both paired colors when collecting carried neighbors.
# d) Placement legality:
#    - Each local patch is built output-first in the sense that its bbox already covers
#      the intended path and optional shifted shadow copy, but the actual output is still
#      oracle-derived as `main(inp)` to guarantee consistency.
#    - Patch placement uses the solved output footprint plus a one-cell 8-connected
#      padding, so different two-color objects cannot touch and accidentally create new
#      cross-pair paths or shadow anchors.
#    - Optional distractors are single pixels only. This avoids creating extra colors
#      with count >= 2, which the new solver would otherwise legitimately treat as
#      drawable paired colors.
# e) Visual observability:
#    - Row-major ordering is fully geometric and human-visible.
#    - The color-value pairing itself is not geometric, so the generator makes it
#      inferable by varying the active palette across examples while always pairing the
#      sorted colors sequentially, and by alternating the two colors along each
#      polyline. A human repeatedly sees "sorted-neighbor colors" behave as one object.
#    - Comparing with the original task is visually easy: instead of two separate
#      monochrome shapes, each paired color set produces one intertwined multicolor open
#      path.
# f) Visual cleanliness:
#    - Only two paired objects are placed per grid, with at most one shadow marker each
#      and wide empty space between them. This keeps the cross-color polylines readable.
#    - Shadow markers use a third color and sit only near the first merged vertex, so
#      they remain local cues rather than creating clutter across the whole patch.
# g) New solver assumptions and generator guarantees:
#    - Assumption: drawable colors are exactly the colors with at least two pixels.
#      Guarantee: only the four active path colors satisfy that; shadows and distractors
#      are singletons.
#    - Assumption: within each paired two-color object, merged row-major consecutive
#      vertices are H/V/45-degree aligned with clear interiors. Guarantee: path
#      templates are built directly in that order and oracle-checked.
#    - Assumption: if a shadow exists, it is adjacent to the first merged row-major
#      vertex and uses a color outside the paired object. Guarantee: shadow offsets are
#      sampled around that first vertex only and validated before acceptance.
#    - Anti-degeneration guarantee: generation rejects any sample where the combo output
#      equals either the original angle-cycle solver or a same-color row-major baseline.

import math

import numpy as np

from arc_utils import bbox_coords, get_bg, paste, random_free_location

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
        if int(out[rr, cc]) == int(bg):
            out[rr, cc] = int(col)


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _drawable_colors(g: np.ndarray, bg: int) -> list[int]:
    colors = sorted(set(int(v) for v in np.unique(g)) - {int(bg)})
    return [col for col in colors if int(np.count_nonzero(g == col)) >= 2]


def _paired_drawable_colors(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    colors = _drawable_colors(g, bg)
    return [(colors[i], colors[i + 1]) for i in range(0, len(colors) - 1, 2)]


def _pair_points(g: np.ndarray, ca: int, cb: int) -> list[tuple[int, int, int]]:
    pts: list[tuple[int, int, int]] = []
    for r, c in np.argwhere((g == int(ca)) | (g == int(cb))):
        pts.append((int(r), int(c), int(g[r, c])))
    pts.sort(key=lambda item: (item[0], item[1], item[2]))
    return pts


def _carried_neighbors(
    g: np.ndarray,
    start: tuple[int, int],
    pair_colors: set[int],
    bg: int,
) -> list[tuple[int, int, int]]:
    h, w = g.shape
    r0, c0 = start
    carried: list[tuple[int, int, int]] = []
    for drn, dcn in DIR8:
        rr = r0 + drn
        cc = c0 + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != int(bg) and v not in pair_colors:
                carried.append((drn, dcn, v))
    return carried


def _combo_plan(
    g: np.ndarray,
    ca: int,
    cb: int,
    bg: int,
) -> tuple[list[tuple[int, int, int]], list[tuple[tuple[int, int], tuple[int, int], int]], list[tuple[int, int, int]]]:
    pts = _pair_points(g, int(ca), int(cb))
    pair_colors = {int(ca), int(cb)}
    carried = _carried_neighbors(g, (pts[0][0], pts[0][1]), pair_colors, int(bg)) if pts else []

    edges: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for i in range(len(pts) - 1):
        r0, c0, col0 = pts[i]
        r1, c1, _col1 = pts[i + 1]
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_in_original(g, r0, c0, r1, c1, int(bg), allow=pair_colors):
            continue
        edges.append(((r0, c0), (r1, c1), int(col0)))
    return pts, edges, carried


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {int(bg)})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == int(col))]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        seed_idxs = []
        for i, (r, c) in enumerate(cycle):
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != int(bg) and v != int(col):
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
                if v != int(bg) and v != int(col):
                    carried.append((drn, dcn, v))

        if len(cycle) == 2:
            edges = [(cycle[0], cycle[1])]
        else:
            edges = [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]

        drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for (r0, c0), (r1, c1) in edges:
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, int(bg), allow={int(col)}):
                continue
            drawable_edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, int(col), int(bg))

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
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, int(bg), allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, int(ncol), int(bg))

    return out


def _same_color_row_major_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {int(bg)})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == int(col))]
        if len(pts_rc) < 2:
            continue

        path = sorted(pts_rc)
        carried = _carried_neighbors(g, path[0], {int(col)}, int(bg))
        edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for i in range(len(path) - 1):
            r0, c0 = path[i]
            r1, c1 = path[i + 1]
            if not _aligned_8(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, int(bg), allow={int(col)}):
                continue
            edges.append(((r0, c0), (r1, c1)))

        for (r0, c0), (r1, c1) in edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, int(col), int(bg))

        for (r0, c0), (r1, c1) in edges:
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
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, int(bg), allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, int(ncol), int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build cross-color row-major polylines from sequentially paired drawable colors.

    Input structure:
    - Background is the most common color.
    - Drawable colors are exactly the non-background colors that appear at least twice.
    - After sorting those drawable colors by color value, consecutive colors form pairs:
      `(c0,c1)`, `(c2,c3)`, ...
    - Within each such two-color pair, the input contains 4-5 total vertex pixels, split
      across the two colors and intended to interleave along one visible path. Optional
      single-pixel third colors may sit next to the first merged vertex as shadow cues.

    Rule:
    - For each paired drawable-color set, merge all vertices of the two colors and sort
      the merged list by row-major order `(r,c)`.
    - Connect consecutive merged vertices only; never close the path. A segment is drawn
      only if its endpoints are H/V/45-degree aligned and its interior is clear of every
      color except the two paired colors.
    - Paint each main segment with the color of its start vertex, filling background
      cells only so the differently colored endpoints remain intact.
    - If the first merged vertex has adjacent non-pair colors, copy those colors along
      the same drawable edges with the original parallel-shadow rule.

    Key insight:
    - The object identity is no longer "all pixels of one color". Two colors become one
      object because they are consecutive in the drawable-color order, and the final path
      order is then determined purely by row-major geometry over the merged two-color
      vertices.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for ca, cb in _paired_drawable_colors(g, int(bg)):
        _pts, edges, carried = _combo_plan(g, int(ca), int(cb), int(bg))
        for (r0, c0), (r1, c1), seg_col in edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, int(seg_col), int(bg))

        for (r0, c0), (r1, c1), _seg_col in edges:
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
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, int(bg), allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, int(ncol), int(bg))

    return out


# --------------- generator ---------------

BG = 0
DRAWABLE_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

PAIR_OBJECT_COUNT = 2
SHADOW_COUNT_RANGE = (0, 3)  # 0-2 shadows across the two objects
DISTRACTOR_PROB = 0.35
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

LEN_SHORT = (2, 5)
LEN_LONG = (3, 7)
PATH_BUILD_RETRIES = 40
SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 400


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _make_path_template(diag_dc: int) -> list[tuple[int, int]]:
    a = int(np.random.randint(*LEN_LONG))
    b = int(np.random.randint(*LEN_SHORT))
    c = int(np.random.randint(*LEN_LONG))
    d = int(np.random.randint(*LEN_SHORT))
    e = int(np.random.randint(*LEN_LONG))
    builders = [
        [(0, 0), (0, a), (b, a), (b, a + c)],
        [(0, 0), (0, a), (b, a + diag_dc * b), (b, a + diag_dc * b + c)],
        [(0, 0), (a, 0), (a + b, diag_dc * b), (a + b, diag_dc * b + c)],
        [(0, 0), (0, a), (b, a), (b + d, a + diag_dc * d), (b + d, a + diag_dc * d + e)],
    ]
    return builders[int(np.random.randint(len(builders)))]


def _path_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def _path_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _path_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step_dir(a[0], a[1], b[0], b[1]) for a, b in _path_edges(vertices)]


def _build_vertices() -> list[tuple[int, int]] | None:
    for _ in range(PATH_BUILD_RETRIES):
        diag_dc = int(np.random.choice([-1, 1]))
        vertices = _normalize_coords(_make_path_template(diag_dc))
        vertices = sorted(vertices)
        cells = _path_cells(vertices)
        rows = {r for r, _ in vertices}
        cols = {c for _, c in vertices}
        dirs = set(_edge_dirs(vertices))
        r0, c0, r1, c1 = bbox_coords(np.array(sorted(cells), dtype=int))
        if len(vertices) < 4:
            continue
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len(dirs) < 2:
            continue
        if (r1 - r0 + 1) > 11 or (c1 - c0 + 1) > 11:
            continue
        return vertices
    return None


def _choose_shadow(
    vertices: list[tuple[int, int]],
    path_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = vertices[0]
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in path_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    path_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    shadow: tuple[tuple[int, int], tuple[int, int]] | None,
) -> tuple[int, int, int, int]:
    coords = list(path_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in path_cells)
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    return int(r0), int(c0), int(r1), int(c1)


def _make_pair_patch(pair_colors: tuple[int, int], shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _build_vertices()
    if vertices is None:
        return None
    path_cells = _path_cells(vertices)

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, path_cells)
        if shadow is None:
            return None

    r0, c0, r1, c1 = _patch_bbox(path_cells, vertices, shadow)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    for i, (r, c) in enumerate(shifted_vertices):
        inp[r, c] = int(pair_colors[i % 2])

    if shadow is not None:
        (ar, ac), offset = shadow
        anchor = (ar - r0, ac - c0)
        sa = (anchor[0] + offset[0], anchor[1] + offset[1])
        if not (0 <= sa[0] < h and 0 <= sa[1] < w):
            return None
        if inp[sa] != BG:
            return None
        inp[sa] = int(shadow_color)

    out = main(inp)
    if np.array_equal(out, inp):
        return None
    if np.array_equal(out, _legacy_main(inp)):
        return None
    if np.array_equal(out, _same_color_row_major_main(inp)):
        return None
    return inp, out


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    patch = np.full((1, 1), int(color), dtype=int)
    return patch.copy(), patch.copy()


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


def _sample_active_pairs() -> tuple[list[tuple[int, int]], list[int]]:
    active = sorted(int(v) for v in np.random.choice(DRAWABLE_COLOR_POOL, size=2 * PAIR_OBJECT_COUNT, replace=False))
    pairs = [(active[i], active[i + 1]) for i in range(0, len(active), 2)]
    remaining = [col for col in DRAWABLE_COLOR_POOL if col not in active]
    return pairs, remaining


def _sample_shadow_assignment(singleton_pool: list[int]) -> dict[int, int]:
    shadow_count = int(np.random.randint(*SHADOW_COUNT_RANGE))
    if shadow_count == 0:
        return {}
    chosen_shapes = np.random.choice(PAIR_OBJECT_COUNT, size=shadow_count, replace=False).tolist()
    chosen_colors = list(np.random.permutation(singleton_pool)[:shadow_count])
    return {int(idx): int(col) for idx, col in zip(sorted(chosen_shapes), chosen_colors)}


def _gen_attempt() -> np.ndarray | None:
    h, w = _sample_canvas_size()
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    active_pairs, singleton_pool = _sample_active_pairs()
    shadow_map = _sample_shadow_assignment(singleton_pool)
    used_singleton_colors = set(shadow_map.values())

    for idx, pair_colors in enumerate(active_pairs):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_pair_patch(pair_colors, shadow_map.get(idx))
            if patch is None:
                continue
            if _try_place(inp, out, *patch):
                placed = True
                break
        if not placed:
            return None

    if np.random.rand() < DISTRACTOR_PROB:
        remaining = [col for col in singleton_pool if col not in used_singleton_colors]
        if remaining:
            dcol = int(remaining[int(np.random.randint(len(remaining)))])
            if not _try_place(inp, out, *_make_distractor_patch(dcol)):
                return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(solved, _legacy_main(inp)):
        return None
    if np.array_equal(solved, _same_color_row_major_main(inp)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
