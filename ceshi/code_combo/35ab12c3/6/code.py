# Combo variant of 35ab12c3 (index 6)
# combo_spec: D3/2+D4/1+D5/0
# Name: 行主序连接+颜色交替轨迹+无阴影
#
# Design analysis for the combo
# a) Concrete behavior and trace:
#    - Original task: same-colored vertices were angle-sorted, 3+ vertices were closed
#      into a polygonal cycle, every main trail cell was painted solid with the shape
#      color, and an adjacent foreign-color pixel could spawn a full parallel shadow copy.
#    - Combo: same-colored vertices are sorted by visible row-major order and connected
#      consecutively as an open polyline only. The single adjacent foreign-color marker
#      next to the first row-major vertex does not create any offset copy; instead it
#      supplies the B color for the main trail's `A,B,A,B,...` striping.
#    - Small trace:
#      input vertices `A` at `(0,0),(0,4),(3,4),(3,8)` and one marker `B` at `(1,0)`.
#      Original would typically close an angle-cycle and might also copy `B` along a
#      shifted parallel path. Combo draws only
#      `(0,0)->(0,4)->(3,4)->(3,8)` and fills background cells on that route as
#      `A,B,A,B,...`, while the lone `B` marker itself stays where it is.
# b) Complexity calibration:
#    - Every generated shape has exactly one stripe marker, so the ABAB rule repeats
#      across multiple groups in the same grid.
#    - Shapes use 4-5 open-path vertices with at least two bends, which makes the D3
#      row-major change visible and avoids degenerating into the legacy closed-hull look.
#    - Consecutive segment lengths are sampled as even numbers so the far endpoint color
#      still visually agrees with the alternating phase.
# c) Hidden-assumption audit:
#    - Original `cycle = _angle_cycle(...)` assumed cyclic symmetry; invalid here because
#      rotating an open path changes the object. We replace it with `sorted(points_rc)`.
#    - Original wraparound edge construction explicitly closed polygons; removed.
#    - Original `carried` offsets and the entire second drawing phase assumed every
#      adjacent foreign pixel should propagate as a parallel copy; D5/0 invalidates that.
#    - D4/1 changes the monochrome line-fill assumption. A new step-indexed fill routine
#      paints only background cells and alternates shape color with the adjacent marker.
# d) Placement legality:
#    - Generator builds each shape as a local patch, computes its output via `main(inp)`,
#      and pastes using the output footprint with a one-cell 8-connected buffer.
#    - The stripe marker is chosen from cells that are 8-adjacent to exactly one vertex,
#      outside the shape path, and not collinear with the first outgoing segment.
#    - Distractor pixels remain pairwise non-aligned so they cannot accidentally form a
#      drawable row-major path.
# e) Visual observability:
#    - The row-major order depends only on visible coordinates.
#    - The second trail color is locally explained by a visibly adjacent single marker
#      near the first vertex of each shape; no hidden color-id ordering is used.
#    - Across 2-3 examples, humans can see both changes together: open ordered bends
#      instead of closed outlines, and stripe coloring on the main path without any
#      parallel shadow copy.
# f) Visual cleanliness:
#    - Only one marker per shape and no offset-shadow layers keeps the grids sparse.
#    - Shapes are separated by padding in output space, so the striped paths stay readable.
# g) Color-change inferability and new solver assumptions:
#    - Each shape uses exactly one local pair `(shape color, marker color)`; this keeps the
#      ABAB mapping simple and repeated.
#    - The solver now assumes that any intended striped shape has exactly one foreign-color
#      marker adjacent to its first row-major vertex. The generator guarantees that.
#    - Any color with fewer than two input cells is inert as a shape, so marker colors stay
#      as isolated clues rather than becoming independent objects.

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste, random_free_location

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
        if int(out[rr, cc]) == int(bg):
            out[rr, cc] = int(col)


def _periodic_fill_bg_only(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    palette: list[int],
    bg: int,
) -> None:
    if not palette:
        return
    for k, (rr, cc) in enumerate(_segment_cells(r0, c0, r1, c1)):
        if int(out[rr, cc]) == int(bg):
            out[rr, cc] = int(palette[k % len(palette)])


def _angle_cycle(points_rc: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cr = sum(r for r, _ in points_rc) / len(points_rc)
    cc = sum(c for _, c in points_rc) / len(points_rc)

    def key(p: tuple[int, int]):
        r, c = p
        ang = math.atan2(r - cr, c - cc)
        rad = (r - cr) ** 2 + (c - cc) ** 2
        return (ang, rad, r, c)

    return sorted(points_rc, key=key)


def _find_partner_color(g: np.ndarray, start: tuple[int, int], col: int, bg: int) -> int | None:
    h, w = g.shape
    r0, c0 = start
    for drn, dcn in DIR8:
        rr = r0 + drn
        cc = c0 + dcn
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != int(bg) and v != int(col):
                return v
    return None


def _shape_plan(
    g: np.ndarray,
    col: int,
    bg: int,
) -> tuple[list[tuple[int, int]], int | None, list[tuple[tuple[int, int], tuple[int, int]]]] | None:
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == int(col))]
    if len(pts_rc) < 2:
        return None

    path = sorted(pts_rc)
    partner = _find_partner_color(g, path[0], int(col), int(bg))

    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for i in range(len(path) - 1):
        r0, c0 = path[i]
        r1, c1 = path[i + 1]
        if not _aligned_8(r0, c0, r1, c1):
            continue
        if not _clear_in_original(g, r0, c0, r1, c1, int(bg), allow={int(col)}):
            continue
        drawable_edges.append(((r0, c0), (r1, c1)))

    return path, partner, drawable_edges


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


def _row_major_solid_main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {int(bg)})
    for col in colors:
        plan = _shape_plan(g, int(col), int(bg))
        if plan is None:
            continue
        _path, _partner, drawable_edges = plan
        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, int(col), int(bg))

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw open row-major polylines with two-color stripes and no parallel shadows.

    Input structure:
    - Background is the most common color.
    - A drawable shape color appears as 4-5 same-colored vertex pixels whose consecutive
      row-major pairs are horizontally, vertically, or 45-degree aligned.
    - Exactly one different-colored single pixel sits 8-neighbor to the first row-major
      vertex of each generated shape; optional distractor colors remain non-alignable.

    Rule:
    - For each color with at least two input pixels, sort its pixels by row-major order
      and connect consecutive pairs only; never close the path.
    - On each drawable segment, recolor only background trail cells with an alternating
      two-color palette `[shape_color, partner_color]`, anchored at the first endpoint.
    - If no adjacent partner exists, fall back to solid shape-color filling.
    - Do not copy the partner along any offset path; the partner pixel is only a local
      stripe cue and stays where it is.

    Key insight:
    - The adjacent odd-color marker supplies the B color for the ABAB stripe on the main
      row-major path, but unlike the original task it never creates a shadow copy.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {int(bg)})
    for col in colors:
        plan = _shape_plan(g, int(col), int(bg))
        if plan is None:
            continue
        _path, partner, drawable_edges = plan
        palette = [int(col)] if partner is None else [int(col), int(partner)]
        for (r0, c0), (r1, c1) in drawable_edges:
            _periodic_fill_bg_only(out, r0, c0, r1, c1, palette, int(bg))

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 4)  # 2-3 striped shapes; each shape consumes two colors
DISTRACTOR_PROB = 0.35
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

LEN_SHORT_RANGE = (2, 7)  # even values only
LEN_LONG_RANGE = (4, 9)   # even values only
PATH_BUILD_RETRIES = 50
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50
SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 300


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return _step_dir(a[0], a[1], b[0], b[1])


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    return list(_segment_cells(a[0], a[1], b[0], b[1]))


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    if np.random.rand() < 0.5:
        arr = arr[:, ::-1]
    if np.random.rand() < 0.5:
        arr[:, 0] *= -1
    if np.random.rand() < 0.5:
        arr[:, 1] *= -1
    return _normalize_coords([tuple(map(int, p)) for p in arr])


def _even_choices(lo: int, hi: int) -> list[int]:
    return [int(v) for v in range(int(lo), int(hi)) if v % 2 == 0]


def _even_rand(lo: int, hi: int) -> int:
    choices = _even_choices(int(lo), int(hi))
    if not choices:
        raise ValueError(f"no even values in range [{lo}, {hi})")
    return int(np.random.choice(choices))


def _make_path_template(diag_dc: int) -> list[tuple[int, int]]:
    a = _even_rand(*LEN_LONG_RANGE)
    b = _even_rand(*LEN_SHORT_RANGE)
    c = _even_rand(*LEN_LONG_RANGE)
    d = _even_rand(*LEN_SHORT_RANGE)
    e = _even_rand(*LEN_LONG_RANGE)
    templates = [
        [(0, 0), (0, a), (b, a), (b, a + c)],
        [(0, 0), (0, a), (b, a + diag_dc * b), (b, a + diag_dc * b + c)],
        [(0, 0), (a, 0), (a + b, diag_dc * b), (a + b, diag_dc * b + c)],
        [(0, 0), (0, a), (b, a), (b + d, a + diag_dc * d), (b + d, a + diag_dc * d + e)],
    ]
    return templates[int(np.random.randint(len(templates)))]


def _path_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _path_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _path_edges(vertices)]


def _build_vertices() -> list[tuple[int, int]] | None:
    for _ in range(PATH_BUILD_RETRIES):
        diag_dc = int(np.random.choice([-1, 1]))
        vertices = _transform_coords(_make_path_template(diag_dc))
        vertices = sorted(vertices)
        if len(vertices) < 4:
            continue
        if any(not _aligned_8(a[0], a[1], b[0], b[1]) for a, b in _path_edges(vertices)):
            continue
        cells = _shape_cells(vertices)
        rows = {r for r, _ in vertices}
        cols = {c for _, c in vertices}
        dirs = set(_edge_dirs(vertices))
        r0, c0, r1, c1 = bbox_coords(np.array(list(cells), dtype=int))
        if len(rows) < 2 or len(cols) < 2:
            continue
        if len(dirs) < 2:
            continue
        if (r1 - r0 + 1) > 11 or (c1 - c0 + 1) > 11:
            continue
        return vertices
    return None


def _choose_marker(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    anchor = vertices[0]
    first_dir = _step(vertices[0], vertices[1])
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if first_dir[0] * dc - first_dir[1] * dr == 0:
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells:
            continue
        hits = sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1)
        if hits != 1:
            continue
        candidates.append(pos)
    if not candidates:
        return None
    return candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    vertices: list[tuple[int, int]],
    marker: tuple[int, int],
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + vertices + [marker]
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


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
        patch = np.full((1, 1), int(color), dtype=int)
        return patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = int(color)
        return patch.copy(), patch.copy()
    patch = np.full((1, 1), int(color), dtype=int)
    return patch.copy(), patch.copy()


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = main(g)
    solid = _row_major_solid_main(g)
    legacy = _legacy_main(g)

    striped_shapes = 0
    partner_trail_cells = 0
    drawable_edges = 0

    colors = sorted(set(int(v) for v in np.unique(g)) - {int(bg)})
    for col in colors:
        plan = _shape_plan(g, int(col), int(bg))
        if plan is None:
            continue
        _path, partner, edges = plan
        if partner is None or not edges:
            continue
        drawable_edges += len(edges)
        used_partner = False
        for (r0, c0), (r1, c1) in edges:
            for rr, cc in _segment_cells(r0, c0, r1, c1):
                if int(g[rr, cc]) != int(bg):
                    continue
                if int(out[rr, cc]) == int(partner):
                    partner_trail_cells += 1
                    used_partner = True
        if used_partner:
            striped_shapes += 1

    return {
        "striped_shapes": int(striped_shapes),
        "partner_trail_cells": int(partner_trail_cells),
        "drawable_edges": int(drawable_edges),
        "diff_vs_solid": int(np.count_nonzero(out != solid)),
        "diff_vs_legacy": int(np.count_nonzero(out != legacy)),
    }


def _make_shape_patch(shape_color: int, stripe_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _build_vertices()
    if vertices is None:
        return None
    shape_cells = _shape_cells(vertices)
    marker = _choose_marker(vertices, shape_cells)
    if marker is None:
        return None

    r0, c0, r1, c1 = _patch_bbox(shape_cells, vertices, marker)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    shifted_marker = (marker[0] - r0, marker[1] - c0)

    for r, c in shifted_vertices:
        inp[r, c] = int(shape_color)
    if int(inp[shifted_marker]) != BG:
        return None
    inp[shifted_marker] = int(stripe_color)

    foreign_hits = 0
    ar, ac = shifted_vertices[0]
    for rr, cc in DIR8:
        nr = ar + rr
        nc = ac + cc
        if 0 <= nr < h and 0 <= nc < w and int(inp[nr, nc]) == int(stripe_color):
            foreign_hits += 1
    if foreign_hits != 1:
        return None

    out = main(inp)
    metrics = _variant_metrics(inp)
    if metrics["striped_shapes"] != 1:
        return None
    if metrics["partner_trail_cells"] < 2:
        return None
    if metrics["diff_vs_solid"] <= 0:
        return None
    if metrics["diff_vs_legacy"] <= 0:
        return None
    return inp, out


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    total = 2 * int(shape_count) + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    stripe_colors = [int(v) for v in palette[shape_count:2 * shape_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, stripe_colors, distractor_color


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

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    can_add_distractor = 2 * shape_count + 1 <= len(COLORS)
    has_distractor = bool(can_add_distractor and np.random.rand() < DISTRACTOR_PROB)

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, stripe_colors, distractor_color = palette

    for shape_color, stripe_color in zip(shape_colors, stripe_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), int(stripe_color))
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

    total_colors = 2 * shape_count + int(has_distractor)
    if int(get_bg(inp)) != BG or int(get_bg(out)) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if not np.array_equal(main(inp), out):
        return None

    metrics = _variant_metrics(inp)
    if metrics["striped_shapes"] != shape_count:
        return None
    if metrics["partner_trail_cells"] < shape_count * 2:
        return None
    if metrics["diff_vs_solid"] < shape_count * 2:
        return None
    if metrics["diff_vs_legacy"] < shape_count * 2:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
