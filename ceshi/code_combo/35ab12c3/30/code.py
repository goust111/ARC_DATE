# Combo variant of 35ab12c3 (index 30)
# combo_spec: D2/0+D21/1
# Name: 仅正交路径+边界反射
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior and trace:
#    - Original task: same-color vertices are angle-sorted into a cycle/polyline;
#      an edge is drawable when it is H/V/45deg aligned; a differently colored
#      8-neighbor of the chosen start vertex defines a translated parallel shadow,
#      and old code clipped that translated copy at the border.
#    - Fused task:
#      1. only H/V edges remain drawable;
#      2. the shadow is no longer a single shifted straight segment guarded by one
#         bounds check. Instead, every shadow cell is translated first and then
#         reflected per-axis back into the grid, so a border hit creates a bounce.
#    - Small witness:
#      vertices {(0,0),(0,4),(4,4)} with shadow anchor offset (+1,+1) in a 6x6 grid.
#      Original output would draw the top edge, the right edge, and the diagonal
#      closing edge; its shadow would be clipped once shifted cells leave the grid.
#      Fused output draws only the L-shape (top + right), and the shadow keeps
#      painting after the bottom/right hit because reflected cells bounce inward.
#      The missing diagonal closure shows D2/0; the inward-bouncing shadow shows
#      D21/1.
# b) Complexity calibration:
#    - Using only lines/rectangles would make D2/0 too subtle, so the generator
#      forces one corner-triplet witness shape whose diagonal closure would exist in
#      the original rule but is absent here.
#    - Using reflection without forcing a border event would collapse back to the
#      clipped task, so the same witness shape is guaranteed to own a shadow whose
#      reflected output differs from clipped output.
# c) Implicit-assumption audit:
#    - `_segment_cells` and step-sign logic still work because all retained edges are
#      still straight H/V segments.
#    - Every old `_aligned_8(...)` check that decided edge drawability must become an
#      orth-only predicate. Otherwise diagonal closures would survive.
#    - The old shadow branch assumed "translate endpoints, check bounds, draw one
#      straight line". Reflection breaks that assumption, so shadow rendering must be
#      per-cell on a reflected path; endpoint alignment is no longer meaningful.
#    - Seed detection still uses 8-neighborhood because shadow anchors may remain
#      diagonal from the chosen vertex.
# d) Placement legality:
#    - Reflection depends on the global border, so local output patches are not
#      enough. The generator places each candidate directly on the full canvas,
#      computes its true solver output footprint, and enforces a one-cell 8-neighbor
#      gap between footprints.
#    - Candidate acceptance requires in-bounds input pixels, non-overlap with
#      existing input pixels, and no padded overlap with existing output footprints.
# e) Visual observability:
#    - Every sample contains at least one shape where a diagonal closure would have
#      appeared under the original rule, but is visibly absent now.
#    - Every sample also contains at least one shadow with extra reflected cells near
#      a border compared with clipped behavior. Both differences are visible from
#      2-4 examples without relying on color ids.
# f) Visual cleanliness:
#    - Distinct groups are spaced using full output footprints rather than sparse
#      input markers, so reflected shadows do not tangle with unrelated objects.
#    - Shadow offsets remain non-collinear with every retained edge direction, which
#      avoids self-smearing parallel copies.
# g) Color inference:
#    - Colors remain one-color-per-group with separate shadow colors. No remapping or
#      hidden numeric ordering is introduced.
# h) New solver assumptions and generator guarantees:
#    - A shadowed shape must have exactly one vertex adjacent to the shadow-anchor
#      input pixel; the generator enforces this when sampling offsets.
#    - The orth-rule witness must remain visible after shadows are drawn; the
#      generator rejects candidates whose fused output is identical to a diagonal-
#      enabled reflective solver.

from __future__ import annotations

import math

import numpy as np

from arc_utils import colors_present, get_bg, neighbors8


DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _aligned_orth(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0)


def _aligned_hvd(r0: int, c0: int, r1: int, c1: int) -> bool:
    dr = r1 - r0
    dc = c1 - c0
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0) or (dr != 0 and abs(dr) == abs(dc))


def _aligned_orth_delta(dr: int, dc: int) -> bool:
    return (dr == 0 and dc != 0) or (dc == 0 and dr != 0)


def _aligned_diag45_delta(dr: int, dc: int) -> bool:
    return dr != 0 and abs(dr) == abs(dc)


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


def _draw_path_bg_only(out: np.ndarray, path: list[tuple[int, int]], col: int, bg: int) -> None:
    for rr, cc in path:
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


def _reflect_index(idx: int, size: int) -> int:
    if size <= 1:
        return 0
    period = 2 * (size - 1)
    x = idx % period
    if x > size - 1:
        x = period - x
    return int(x)


def _shadow_path_reflected(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    drn: int,
    dcn: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    return [(_reflect_index(rr + drn, h), _reflect_index(cc + dcn, w)) for rr, cc in _segment_cells(r0, c0, r1, c1)]


def _shadow_path_clipped(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    drn: int,
    dcn: int,
    h: int,
    w: int,
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    for rr, cc in _segment_cells(r0, c0, r1, c1):
        rr2 = rr + drn
        cc2 = cc + dcn
        if 0 <= rr2 < h and 0 <= cc2 < w:
            path.append((int(rr2), int(cc2)))
    return path


def _clear_path_in_original(g: np.ndarray, path: list[tuple[int, int]], bg: int, allow: set[int]) -> bool:
    if len(path) <= 2:
        return True
    for i in range(1, len(path) - 1):
        rr, cc = path[i]
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _main_common(grid: np.ndarray, *, orth_only: bool, reflect_shadow: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    out = g.copy()
    aligned = _aligned_orth if orth_only else _aligned_hvd

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
            if not aligned(r0, c0, r1, c1):
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
                if reflect_shadow:
                    path = _shadow_path_reflected(r0, c0, r1, c1, drn, dcn, h, w)
                    if not _clear_path_in_original(g, path, bg, allow=set()):
                        continue
                    _draw_path_bg_only(out, path, ncol, bg)
                else:
                    rr0 = r0 + drn
                    cc0 = c0 + dcn
                    rr1 = r1 + drn
                    cc1 = c1 + dcn
                    if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                        continue
                    if not aligned(rr0, cc0, rr1, cc1):
                        continue
                    if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                        continue
                    _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


def _main_diag_reflect(grid: np.ndarray) -> np.ndarray:
    return _main_common(grid, orth_only=False, reflect_shadow=True)


def _main_clipped_orth(grid: np.ndarray) -> np.ndarray:
    return _main_common(grid, orth_only=True, reflect_shadow=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
    Connect each same-color group using only orthogonal edges, then copy any carried
    neighbor-color shadow along the same edge set with border reflection.

    Input structure:
    The input contains 2-4 sparse same-color vertex groups plus optional distractor
    pixels. Some groups have one differently colored 8-neighbor next to exactly one
    vertex; that neighbor is the shadow anchor and defines the shadow offset.

    Rule:
    Sort each color's pixels by angle around the centroid and consider consecutive
    pairs in that cycle. Draw an edge only when the pair is horizontal or vertical
    and its interior crosses only background or same-color input cells. After all
    main edges are drawn, copy them for every carried neighbor offset that is not
    collinear with the edge. For the copied shadow, reflect any out-of-bounds row or
    column back into the grid instead of clipping it away.

    Key insight:
    The fused variant differs from the original in two visible ways at once:
    diagonal closures disappear, while shadows that hit a border bounce inward and
    keep drawing.
    """
    return _main_common(grid, orth_only=True, reflect_shadow=True)


BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]

CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.4

LINE_VERTICES_RANGE = (2, 8)
LINE_TOTAL_LENGTH = 13
LINE_MIN_GAP = 2
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
CORNER_SIDE_RANGE = (2, 8)

SHAPE_WEIGHTS = np.array([0.25, 0.30, 0.45], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 6)
DISTRACTOR_RETRIES = 80

SHAPE_GEOMETRY_RETRIES = 60
SHAPE_PLACEMENT_RETRIES = 150
DISTRACTOR_PLACEMENT_RETRIES = 100
GENERATE_RETRIES = 400


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


def _build_rectangle() -> list[tuple[int, int]]:
    h = int(np.random.randint(*RECT_H_RANGE))
    w = int(np.random.randint(*RECT_W_RANGE))
    return _transform_coords([(0, 0), (0, w), (h, w), (h, 0)])


def _build_corner_triplet() -> list[tuple[int, int]]:
    side = int(np.random.randint(*CORNER_SIDE_RANGE))
    return _transform_coords([(0, 0), (0, side), (side, side)])


_SHAPE_BUILDERS = [
    ("line", _build_line, False),
    ("rectangle", _build_rectangle, False),
    ("corner_triplet", _build_corner_triplet, True),
]


def _make_vertices(force_witness: bool = False) -> tuple[list[tuple[int, int]], bool]:
    if force_witness:
        idx = next(i for i, (_, _, witness) in enumerate(_SHAPE_BUILDERS) if witness)
    else:
        idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()))
    _, builder, witness = _SHAPE_BUILDERS[idx]
    return builder(), witness


def _cycle_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    cycle = _angle_cycle(vertices)
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _drawable_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(a, b) for a, b in _cycle_edges(vertices) if _aligned_orth(a[0], a[1], b[0], b[1])]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _drawable_edges(vertices):
        cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step_dir(a[0], a[1], b[0], b[1]) for a, b in _drawable_edges(vertices)]


def _diag_only_edge_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _cycle_edges(vertices):
        dr = b[0] - a[0]
        dc = b[1] - a[1]
        if _aligned_diag45_delta(dr, dc):
            cells.update(_segment_cells(a[0], a[1], b[0], b[1]))
    return cells


def _choose_shadow_offsets(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    preserve_witness: bool,
) -> list[tuple[int, int]]:
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    blocked_diag_cells = (_diag_only_edge_cells(vertices) - set(vertices)) if preserve_witness else set()
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        if any(ds_r * dc - ds_c * dr == 0 for ds_r, ds_c in directions):
            continue
        pos = (anchor[0] + dr, anchor[1] + dc)
        if pos in shape_cells or pos in blocked_diag_cells:
            continue
        if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
            continue
        candidates.append((int(dr), int(dc)))
    return candidates


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
            dr = coords[j][0] - coords[i][0]
            dc = coords[j][1] - coords[i][1]
            if _aligned_orth_delta(dr, dc):
                return None

    if count >= 2:
        has_diag_pair = False
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                dr = coords[j][0] - coords[i][0]
                dc = coords[j][1] - coords[i][1]
                if _aligned_diag45_delta(dr, dc):
                    has_diag_pair = True
                    break
            if has_diag_pair:
                break
        if not has_diag_pair:
            return None

    return coords


def _make_distractor_cells() -> set[tuple[int, int]]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        return {(0, 0)}
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is not None:
            return set(coords)
    return {(0, 0)}


def _sample_palette(shape_count: int, has_distractor: bool) -> tuple[list[int], list[int], int | None] | None:
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count - int(has_distractor))
    if max_shadows < 1:
        return None
    shadow_count = int(np.random.randint(1, max_shadows + 1))
    total = shape_count + shadow_count + int(has_distractor)
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count:shape_count + shadow_count]]
    distractor_color = int(palette[-1]) if has_distractor else None
    return shape_colors, shadow_colors, distractor_color


def _assign_shadows(shape_count: int, shadow_colors: list[int], forced_idx: int) -> dict[int, int]:
    if not shadow_colors:
        return {}
    available = [i for i in range(shape_count) if i != forced_idx]
    chosen = [forced_idx]
    extra = len(shadow_colors) - 1
    if extra > 0:
        chosen.extend(int(v) for v in np.random.choice(available, size=extra, replace=False).tolist())
    return {int(idx): int(col) for idx, col in zip(sorted(chosen), shadow_colors)}


def _shape_reflection_score(vertices: list[tuple[int, int]], offset: tuple[int, int]) -> int:
    anchor = min(vertices)
    cells = _shape_cells(vertices)
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    score = 0
    if offset[0] == 1:
        score += 2
    if offset[1] == 1:
        score += 2
    if offset[0] == -1 and anchor[0] > 0:
        score += 1
    if offset[1] == -1 and anchor[1] > 0:
        score += 1
    score += int(max_r > anchor[0]) + int(max_c > anchor[1])
    return score


def _candidate_locations(
    H: int,
    W: int,
    vertices: list[tuple[int, int]],
    offset: tuple[int, int] | None,
    require_reflection: bool,
) -> list[tuple[int, int]]:
    max_r = max(r for r, _ in vertices)
    max_c = max(c for _, c in vertices)
    anchor = min(vertices)
    base_r0 = 0
    base_r1 = H - 1 - max_r
    base_c0 = 0
    base_c1 = W - 1 - max_c
    if base_r1 < base_r0 or base_c1 < base_c0:
        return []

    cells = _shape_cells(vertices)
    max_sr = max(r for r, _ in cells)
    max_sc = max(c for _, c in cells)
    candidates: list[tuple[int, int]] = []

    def add(loc_r: int, loc_c: int) -> None:
        if not (base_r0 <= loc_r <= base_r1 and base_c0 <= loc_c <= base_c1):
            return
        if offset is not None:
            sr = loc_r + anchor[0] + offset[0]
            sc = loc_c + anchor[1] + offset[1]
            if not (0 <= sr < H and 0 <= sc < W):
                return
        candidates.append((int(loc_r), int(loc_c)))

    if require_reflection and offset is not None:
        fixed_r: int | None = None
        fixed_c: int | None = None
        if offset[0] == 1:
            fixed_r = H - 1 - max_sr
            for _ in range(24):
                add(fixed_r, int(np.random.randint(base_c0, base_c1 + 1)))
        if offset[1] == 1:
            fixed_c = W - 1 - max_sc
            for _ in range(24):
                add(int(np.random.randint(base_r0, base_r1 + 1)), fixed_c)
        if offset[0] == -1 and anchor[0] > 0:
            fixed_r = 0
            for _ in range(24):
                add(0, int(np.random.randint(base_c0, base_c1 + 1)))
        if offset[1] == -1 and anchor[1] > 0:
            fixed_c = 0
            for _ in range(24):
                add(int(np.random.randint(base_r0, base_r1 + 1)), 0)
        if fixed_r is not None and fixed_c is not None:
            for _ in range(12):
                add(fixed_r, fixed_c)

    for _ in range(80):
        add(int(np.random.randint(base_r0, base_r1 + 1)), int(np.random.randint(base_c0, base_c1 + 1)))

    seen = set()
    uniq: list[tuple[int, int]] = []
    for loc in candidates:
        if loc not in seen:
            seen.add(loc)
            uniq.append(loc)
    np.random.shuffle(uniq)
    return uniq


def _render_shape_candidate(
    H: int,
    W: int,
    vertices: list[tuple[int, int]],
    shape_color: int,
    shadow_color: int | None,
    offset: tuple[int, int] | None,
    loc: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((H, W), BG, dtype=int)
    shifted_vertices = [(r + loc[0], c + loc[1]) for r, c in vertices]
    for r, c in shifted_vertices:
        if not (0 <= r < H and 0 <= c < W):
            return None
        inp[r, c] = int(shape_color)

    if shadow_color is not None and offset is not None:
        anchor = min(shifted_vertices)
        sr = anchor[0] + offset[0]
        sc = anchor[1] + offset[1]
        if not (0 <= sr < H and 0 <= sc < W):
            return None
        if inp[sr, sc] != BG:
            return None
        inp[sr, sc] = int(shadow_color)
        nb_hits = sum(1 for rr, cc in neighbors8(sr, sc) if 0 <= rr < H and 0 <= cc < W and inp[rr, cc] == shape_color)
        if nb_hits != 1:
            return None

    out = main(inp)
    return inp, out


def _cells_of(grid: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, p)) for p in np.argwhere(grid != BG)}


def _touches_with_padding(cells: set[tuple[int, int]], occupied: set[tuple[int, int]], pad: int = 1) -> bool:
    if not cells or not occupied:
        return False
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _merge_patches(dst: np.ndarray, patch: np.ndarray) -> None:
    mask = patch != BG
    dst[mask] = patch[mask]


def _reflection_difference(inp: np.ndarray) -> int:
    return int(np.count_nonzero(main(inp) != _main_clipped_orth(inp)))


def _orth_difference_from_diag(inp: np.ndarray) -> int:
    return int(np.count_nonzero(_main_diag_reflect(inp) != main(inp)))


def _reflection_events(inp: np.ndarray) -> int:
    g = np.asarray(inp, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))
    total = 0

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
        if not seed_idxs:
            continue
        start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
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

        for (r0, c0), (r1, c1) in edges:
            if not _aligned_orth(r0, c0, r1, c1):
                continue
            if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
                continue
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, _ in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                for rr, cc in _segment_cells(r0, c0, r1, c1):
                    rr2 = rr + drn
                    cc2 = cc + dcn
                    if not (0 <= rr2 < h and 0 <= cc2 < w):
                        total += 1
    return total


def _place_shape(
    full_inp: np.ndarray,
    occupied_out: set[tuple[int, int]],
    H: int,
    W: int,
    shape_color: int,
    shadow_color: int | None,
    *,
    force_witness: bool,
    require_reflection: bool,
) -> tuple[np.ndarray, set[tuple[int, int]]] | None:
    for _ in range(SHAPE_GEOMETRY_RETRIES):
        vertices, _witness = _make_vertices(force_witness=force_witness)
        shape_cells = _shape_cells(vertices)
        if not shape_cells:
            continue

        if shadow_color is None:
            offsets = [None]
        else:
            offsets = _choose_shadow_offsets(vertices, shape_cells, preserve_witness=force_witness)
            if not offsets:
                continue
            offsets = sorted(offsets, key=lambda o: _shape_reflection_score(vertices, o), reverse=True)

        for offset in offsets:
            locs = _candidate_locations(H, W, vertices, offset, require_reflection=require_reflection)
            for loc in locs[:SHAPE_PLACEMENT_RETRIES]:
                rendered = _render_shape_candidate(H, W, vertices, shape_color, shadow_color, offset, loc)
                if rendered is None:
                    continue
                patch_in, patch_out = rendered
                candidate_cells = _cells_of(patch_out)
                if not candidate_cells:
                    continue
                if _touches_with_padding(candidate_cells, occupied_out, pad=1):
                    continue
                if np.any((full_inp != BG) & (patch_in != BG)):
                    continue
                if require_reflection and shadow_color is not None and _reflection_difference(patch_in) == 0:
                    continue
                if force_witness and _orth_difference_from_diag(patch_in) == 0:
                    continue
                return patch_in, candidate_cells
    return None


def _place_distractor(
    full_inp: np.ndarray,
    occupied_out: set[tuple[int, int]],
    H: int,
    W: int,
    color: int,
) -> np.ndarray | None:
    cells_local = _make_distractor_cells()
    max_r = max(r for r, _ in cells_local)
    max_c = max(c for _, c in cells_local)
    if max_r >= H or max_c >= W:
        return None

    for _ in range(DISTRACTOR_PLACEMENT_RETRIES):
        loc_r = int(np.random.randint(0, H - max_r))
        loc_c = int(np.random.randint(0, W - max_c))
        patch = np.full((H, W), BG, dtype=int)
        for r, c in cells_local:
            patch[loc_r + r, loc_c + c] = int(color)
        cells = _cells_of(patch)
        if _touches_with_padding(cells, occupied_out, pad=1):
            continue
        if np.any((full_inp != BG) & (patch != BG)):
            continue
        return patch
    return None


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    inp = np.full((H, W), BG, dtype=int)
    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    witness_idx = int(np.random.randint(shape_count))

    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors, forced_idx=witness_idx)

    occupied_out: set[tuple[int, int]] = set()
    for idx, shape_color in enumerate(shape_colors):
        placed = _place_shape(
            inp,
            occupied_out,
            H,
            W,
            int(shape_color),
            shadow_map.get(idx),
            force_witness=(idx == witness_idx),
            require_reflection=(idx == witness_idx),
        )
        if placed is None:
            return None
        patch_in, cells_out = placed
        _merge_patches(inp, patch_in)
        occupied_out |= cells_out

    if has_distractor and distractor_color is not None:
        patch = _place_distractor(inp, occupied_out, H, W, int(distractor_color))
        if patch is None:
            return None
        _merge_patches(inp, patch)
        occupied_out |= _cells_of(patch)

    out = main(inp)
    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    if get_bg(inp) != BG or get_bg(out) != BG:
        return None
    if len(colors_present(inp, exclude=[BG])) != total_colors:
        return None
    if _reflection_difference(inp) == 0 or _reflection_events(inp) == 0:
        return None
    if _orth_difference_from_diag(inp) == 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GENERATE_RETRIES} attempts")
