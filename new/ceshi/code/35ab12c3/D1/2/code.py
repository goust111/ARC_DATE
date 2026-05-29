# Variant of 35ab12c3
# Dimension: D1 — 匹配条件——同色配对规则 (match_cond)
# Change: color_match——同色像素配对（s1.color == s2.color），同一颜色的所有像素按极角排序后形成多... → 距离约束配对
#
# Design analysis
# source_task: none
#
# a) Concretization / behavior trace
# - Original behavior: if color 6 has vertices A,B,C of one latent elbow, the solver
#   sorts all three by angle and tries to draw every consecutive edge, so the output
#   becomes a two-segment polyline A-B-C. If a shadow color 7 is attached near A, the
#   whole polyline is copied in parallel by that offset.
# - Variant behavior: among all drawable same-color pairs, only the unique shortest
#   pair is connected. Example: A=(1,1), B=(1,4), C=(5,4). Original draws A-B and
#   B-C. Variant compares segment lengths 3 and 4, keeps only A-B, leaves C as an
#   isolated dot, and if 7 is adjacent to A then only the translated copy of A-B is
#   painted.
# - The changed step is therefore the match stage inside `main()`: the old global
#   cycle over all same-color points is replaced by a closest-pair search over
#   aligned / unobstructed pairs.
#
# b) Complexity calibration
# - If a color group had only two vertices, the variant would collapse to the old
#   behavior and the D1 change would be invisible. The generator therefore produces
#   shape groups with 3-4 vertices and at least two drawable aligned pairs, while
#   ensuring the shortest pair is unique.
# - Tie-heavy symmetric polygons would hide the new rule. The generator therefore
#   favors asymmetric open elbows / zigzags with distinct segment lengths.
#
# c) Implicit-assumption review against the original main()
# - `_angle_cycle(...)`: depended on "all same-color points belong to one cycle or
#   polyline". This is no longer valid; a single best pair must be chosen instead.
# - `seed_idxs` over every vertex in the cycle: previously any cycle vertex could be
#   the shadow anchor. Now only the two endpoints of the chosen pair matter.
# - `edges = [...] for the whole cycle`: invalid in the variant because drawing all
#   edges would revert to the original task.
# - The shadow logic itself remains usable once it is restricted to the chosen
#   segment; its safety still depends on the anchor being the only endpoint with a
#   foreign-color 8-neighbor, which the generator guarantees.
#
# d) Placement legality
# - Each local patch contains all same-color vertices, the chosen output segment, and
#   optionally one translated shadow segment. Patch placement still uses a 1-cell
#   padding on final output occupancy, so distinct groups do not become adjacent and
#   cannot confuse endpoint-neighbor detection.
# - The generator rejects patches where the shifted shadow segment would overlap any
#   original vertex or the main segment.
#
# e) Visual observability
# - Across examples, the user sees multiple same-color points but only the nearest
#   pair grows into a line. Remaining same-color points stay as dots, which is a
#   directly visible contrast with the original "connect everything" rule.
# - No color-id ordering is used. Distance and visible adjacency alone determine the
#   chosen segment and the shadow anchor.
#
# f) Visual cleanliness
# - Only one main segment and at most one parallel shadow segment are drawn per shape
#   color, so overlaps are much lower than in the original task.
# - Shadow offsets are forced non-collinear and non-intersecting with the main
#   segment to keep the local motif readable.
#
# g) Color-change inferability
# - Colors preserve the original semantics: one color names the point group, one
#   optional different color names its shadow anchor/trace, and distractor colors are
#   unchanged. There is no hidden color remapping.
#
# h) Generator-side guarantees for the new solver assumptions
# - Every shape color has a unique shortest drawable pair.
# - The chosen pair's shadow anchor, if present, is adjacent to exactly one endpoint
#   of that pair and to no other same-color vertex.
# - Each grid contains multiple nontrivial shape groups, so the variant does not
#   silently regress to the original two-point case.

import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import colors_present, draw_line, get_bg, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
STEP_DIRS = DIR8[:]


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


def _sorted_pair(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return (a, b) if a <= b else (b, a)


def _seg_len(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _has_foreign_neighbor(g: np.ndarray, pt: tuple[int, int], bg: int, col: int) -> bool:
    h, w = g.shape
    r, c = pt
    for dr, dc in DIR8:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            v = int(g[rr, cc])
            if v != bg and v != col:
                return True
    return False


def _closest_drawable_pair(
    g: np.ndarray,
    pts_rc: list[tuple[int, int]],
    bg: int,
    col: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates: list[tuple[int, int, int, int, int, tuple[int, int], tuple[int, int]]] = []
    for i in range(len(pts_rc)):
        for j in range(i + 1, len(pts_rc)):
            a = pts_rc[i]
            b = pts_rc[j]
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            if not _clear_in_original(g, a[0], a[1], b[0], b[1], bg, allow={col}):
                continue
            aa, bb = _sorted_pair(a, b)
            candidates.append((_seg_len(aa, bb), aa[0], aa[1], bb[0], bb[1], aa, bb))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[:5])
    return candidates[0][5], candidates[0][6]


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

        pair = _closest_drawable_pair(g, pts_rc, bg, col)
        if pair is None:
            continue

        a, b = pair
        seed_pts = [pt for pt in (a, b) if _has_foreign_neighbor(g, pt, bg, col)]
        start = min(seed_pts) if seed_pts else min(a, b)
        end = b if start == a else a

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = start
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        if _aligned_8(start[0], start[1], end[0], end[1]) and _clear_in_original(
            g, start[0], start[1], end[0], end[1], bg, allow={col}
        ):
            _draw_line_bg_only(out, start[0], start[1], end[0], end[1], col, bg)

            drs, dcs = _step_dir(start[0], start[1], end[0], end[1])
            for drn, dcn, ncol in carried:
                if drs * dcn - dcs * drn == 0:
                    continue
                rr0 = start[0] + drn
                cc0 = start[1] + dcn
                rr1 = end[0] + drn
                cc1 = end[1] + dcn
                if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
                    continue
                if not _aligned_8(rr0, cc0, rr1, cc1):
                    continue
                if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
                    continue
                _draw_line_bg_only(out, rr0, cc0, rr1, cc1, ncol, bg)

    return out


# --------------- generator ---------------

# ---- CFG: canvas and palette ----
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

# ---- CFG: structural counts ----
SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.5
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: shape geometry ----
SEG_LEN_RANGE = (2, 8)
EXTRA_SEG_LEN_BONUS = 2
ZIGZAG_PROB = 0.45
SHAPE_BUILD_RETRIES = 80

# ---- CFG: distractor geometry ----
DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

# ---- CFG: retry budgets ----
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


def _normalize_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    arr = np.array(coords, dtype=int)
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _cross(d1: tuple[int, int], d2: tuple[int, int]) -> int:
    return d1[0] * d2[1] - d1[1] * d2[0]


def _move(pt: tuple[int, int], direction: tuple[int, int], length: int) -> tuple[int, int]:
    return pt[0] + direction[0] * length, pt[1] + direction[1] * length


def _polyline_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def _sample_distinct_lengths(k: int) -> list[int]:
    vals = np.random.choice(np.arange(SEG_LEN_RANGE[0], SEG_LEN_RANGE[1]), size=k, replace=False)
    vals = [int(v) for v in vals]
    vals.sort()
    return vals


def _build_elbow3() -> list[tuple[int, int]]:
    d1 = STEP_DIRS[int(np.random.randint(len(STEP_DIRS)))]
    turn_dirs = [d for d in STEP_DIRS if _cross(d1, d) != 0]
    d2 = turn_dirs[int(np.random.randint(len(turn_dirs)))]
    l1, l2 = _sample_distinct_lengths(2)
    if np.random.rand() < 0.5:
        l1, l2 = l2, l1
    p0 = (0, 0)
    p1 = _move(p0, d1, l1)
    p2 = _move(p1, d2, l2 + EXTRA_SEG_LEN_BONUS)
    return _normalize_coords([p0, p1, p2])


def _build_zigzag4() -> list[tuple[int, int]] | None:
    d1 = STEP_DIRS[int(np.random.randint(len(STEP_DIRS)))]
    d2_choices = [d for d in STEP_DIRS if _cross(d1, d) != 0]
    d2 = d2_choices[int(np.random.randint(len(d2_choices)))]
    d3_choices = [d for d in STEP_DIRS if _cross(d2, d) != 0 and d != (-d2[0], -d2[1])]
    d3 = d3_choices[int(np.random.randint(len(d3_choices)))]
    l1, l2, l3 = _sample_distinct_lengths(3)
    lens = [l1, l2 + EXTRA_SEG_LEN_BONUS, l3 + EXTRA_SEG_LEN_BONUS + 1]
    if np.random.rand() < 0.5:
        lens[0], lens[1] = lens[1], lens[0]
    p0 = (0, 0)
    p1 = _move(p0, d1, lens[0])
    p2 = _move(p1, d2, lens[1])
    p3 = _move(p2, d3, lens[2])
    coords = [p0, p1, p2, p3]
    if len(set(coords)) != len(coords):
        return None
    edges = _polyline_edges(coords)
    edge_cells = [set(_gen_segment_cells(a, b)) for a, b in edges]
    for i in range(len(edge_cells)):
        for j in range(i + 1, len(edge_cells)):
            overlap = edge_cells[i] & edge_cells[j]
            allowed = set()
            if j == i + 1:
                allowed.add(edges[i][1])
            if overlap - allowed:
                return None
    return _normalize_coords(coords)


def _make_vertices() -> list[tuple[int, int]] | None:
    if np.random.rand() < ZIGZAG_PROB:
        return _build_zigzag4()
    return _build_elbow3()


def _candidate_pairs(points: list[tuple[int, int]]) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            a = points[i]
            b = points[j]
            if not _aligned_8(a[0], a[1], b[0], b[1]):
                continue
            aa, bb = _sorted_pair(a, b)
            pairs.append((_seg_len(aa, bb), aa, bb))
    pairs.sort(key=lambda item: (item[0], item[1], item[2]))
    return pairs


def _shifted_segment_disjoint(
    a: tuple[int, int],
    b: tuple[int, int],
    offset: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> bool:
    shifted = {(r + offset[0], c + offset[1]) for r, c in _gen_segment_cells(a, b)}
    return shifted.isdisjoint(blocked)


def _choose_shadow(
    vertices: list[tuple[int, int]],
    pair: tuple[tuple[int, int], tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    a, b = pair
    vertex_set = set(vertices)
    main_cells = set(_gen_segment_cells(a, b))
    for anchor, other in ((a, b), (b, a)):
        drs, dcs = _step(anchor, other)
        candidates: list[tuple[int, int]] = []
        for dr, dc in DIR8:
            if drs * dc - dcs * dr == 0:
                continue
            pos = (anchor[0] + dr, anchor[1] + dc)
            if pos in vertex_set:
                continue
            if sum(1 for vr, vc in vertices if max(abs(pos[0] - vr), abs(pos[1] - vc)) == 1) != 1:
                continue
            shifted = {(r + dr, c + dc) for r, c in _gen_segment_cells(a, b)}
            if shifted & main_cells:
                continue
            if shifted & vertex_set:
                continue
            candidates.append((dr, dc))
        if candidates:
            return anchor, candidates[int(np.random.randint(len(candidates)))]
    return None


def _patch_bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_BUILD_RETRIES):
        vertices = _make_vertices()
        if vertices is None or len(vertices) < 3:
            continue

        candidates = _candidate_pairs(vertices)
        if len(candidates) < 2:
            continue
        if candidates[0][0] == candidates[1][0]:
            continue

        intended_edges = {_sorted_pair(a, b) for a, b in _polyline_edges(vertices)}
        pair = (candidates[0][1], candidates[0][2])
        if _sorted_pair(*pair) not in intended_edges:
            continue

        shadow = None
        if shadow_color is not None:
            shadow = _choose_shadow(vertices, pair)
            if shadow is None:
                continue

        main_cells = list(_gen_segment_cells(pair[0], pair[1]))
        coords = list(vertices) + main_cells
        if shadow is not None:
            anchor, offset = shadow
            coords.append((anchor[0] + offset[0], anchor[1] + offset[1]))
            coords.extend((r + offset[0], c + offset[1]) for r, c in main_cells)
        r0, c0, r1, c1 = _patch_bbox(coords)
        h, w = r1 - r0 + 1, c1 - c0 + 1

        inp = np.full((h, w), BG, dtype=int)
        shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
        shifted_pair = ((pair[0][0] - r0, pair[0][1] - c0), (pair[1][0] - r0, pair[1][1] - c0))

        for r, c in shifted_vertices:
            inp[r, c] = shape_color

        if shadow is not None:
            anchor, offset = shadow
            sa = (anchor[0] + offset[0] - r0, anchor[1] + offset[1] - c0)
            if not (0 <= sa[0] < h and 0 <= sa[1] < w) or inp[sa] != BG:
                continue
            inp[sa] = int(shadow_color)

        out = inp.copy()
        draw_line(
            out,
            shifted_pair[0][0],
            shifted_pair[0][1],
            end_r=shifted_pair[1][0],
            end_c=shifted_pair[1][1],
            color=shape_color,
        )

        if shadow is not None:
            offset = shadow[1]
            _draw_line_bg_only(
                out,
                shifted_pair[0][0] + offset[0],
                shifted_pair[0][1] + offset[1],
                shifted_pair[1][0] + offset[0],
                shifted_pair[1][1] + offset[1],
                int(shadow_color),
                BG,
            )

        if not np.array_equal(main(inp), out):
            continue
        return inp, out
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
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

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
