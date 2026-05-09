# Combo variant of 35ab12c3 (index 12)
# combo_spec: D4/2+D5/1
# Name: 渐变长度轨迹+多层嵌套阴影

import math

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, paste, random_free_location

# -----------------------------------------------------------------------------
# Design analysis for the combo variant
#
# [a] Concrete behavior / behavior trace
# - Base task:
#   same-color input vertices are connected into an H/V/45-degree polygon/polyline,
#   and one adjacent foreign-color pixel can trigger one parallel shadow copy.
# - D4/2 changes "how each edge is painted":
#   edge 1 keeps all internal cells, edge 2 keeps 3/4, edge 3 keeps 1/2,
#   edge 4+ keeps 1/4; the endpoint still appears.
# - D5/1 changes "how shadows are triggered":
#   the anchor pixels form a 2-3 step color chain, and each step adds to the
#   cumulative offset of a new shadow layer.
# - Fused behavior:
#   if base vertices are A and anchor chain colors are B,C,(D), then output draws
#   A with fading edge prefixes, then draws the same fading edge prefixes for
#   offset(B), then for offset(B)+offset(C), then optionally for
#   offset(B)+offset(C)+offset(D). The result is nested contour copies whose
#   later edges are progressively shorter on every layer.
# - Small example:
#   A rectangle with 4 equal sides and a 3-color anchor chain might output:
#   base layer lengths = full / 3-4 / 1-2 / endpoint-only,
#   first shadow uses the same four edge truncations one cell up-right,
#   second shadow repeats them another step outward in a different direction.
#   Compared to the original task, both the trail density and the shadow depth
#   change together rather than independently.
#
# [b] Complexity calibration
# - To keep the fading visible, generator only uses closed 3-4 edge shapes with
#   side lengths >= 4.
# - To keep the nested-shadow rule obvious without exceeding the 7-color budget,
#   generator uses 1-2 shapes, each forced to have 2 or 3 shadow layers.
# - Shapes whose later edges would look longer again after truncation are rejected;
#   this keeps the "inside to outside, edge-by-edge decay" readable.
#
# [c] Implicit-assumption audit against original main()
# - Original `seed_idxs` only needed a start for edge order; now that order also
#   controls fade strength, so generator must expose a unique anchored start.
# - Original `carried` assumed all shadow anchors touch the base start vertex;
#   fused solver replaces this with chained anchor extraction and cumulative
#   offsets.
# - Original full-edge painting would erase intended holes; fused solver paints
#   only edge prefixes for both base and shadow layers.
# - Original `_clear_in_original(... allow=set())` still works, but generator must
#   prevent later anchor pixels from landing inside any translated edge interior.
#
# [d] Placement legality
# - Local patches are still built output-first: choose vertices and anchor chain,
#   validate with `main()`, then place by `random_free_location` against the full
#   local output patch so outer contours cannot overlap other objects.
# - Patch bounding boxes include every translated contour layer, so no layer can
#   spill outside the patch when pasted globally.
#
# [e] Visual observability
# - Humans can see the rule from structure alone:
#   same-color vertices define the inner contour, a nearby color chain defines
#   2-3 nested outer contours, and every contour fades in the same edge order.
# - No color-id sorting is used as a semantic rule; colors only label groups.
#
# [f] Visual cleanliness
# - Padding is raised to 2 because 2-3 outer contour layers expand the footprint.
# - Inner/outer layers may partially cover one another under the original
#   bg-only paint priority, but generator requires every new layer to contribute
#   fresh visible cells, so the nested structure stays readable instead of
#   collapsing into a single thick stroke.
#
# [g] Inference of changes
# - There is no hidden color remapping: every shadow color appears once in input
#   and becomes the matching contour color in output.
# - New solver assumptions introduced here:
#   1. the first shadow anchor uniquely identifies the start vertex,
#   2. the anchor colors form a simple 2-3 step chain,
#   3. cumulative offsets produce translated contours with visible new cells,
#   4. truncation is visually distinguishable on the sampled shapes.
# - Generator explicitly enforces all four assumptions.
# -----------------------------------------------------------------------------

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
TRAIL_NUMERATORS = (4, 3, 2, 1)
TRAIL_DENOMINATOR = 4


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


def _edge_ratio_numerator(edge_idx: int) -> int:
    return TRAIL_NUMERATORS[min(edge_idx, len(TRAIL_NUMERATORS) - 1)]


def _partial_internal_count(total_steps: int, edge_idx: int) -> int:
    internal = max(0, total_steps - 1)
    num = _edge_ratio_numerator(edge_idx)
    return int(math.floor(internal * num / TRAIL_DENOMINATOR))


def _draw_partial_edge_bg_only(
    out: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    col: int,
    bg: int,
    edge_idx: int,
) -> None:
    cells = list(_segment_cells(r0, c0, r1, c1))
    total_steps = len(cells) - 1
    keep_internal = _partial_internal_count(total_steps, edge_idx)
    for rr, cc in cells[1 : 1 + keep_internal]:
        if out[rr, cc] == bg:
            out[rr, cc] = col
    if total_steps >= 1:
        rr, cc = cells[-1]
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


def _ordered_drawable_edges(
    g: np.ndarray,
    col: int,
    bg: int,
) -> tuple[list[tuple[tuple[int, int], tuple[int, int]]], tuple[int, int] | None]:
    h, w = g.shape
    pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
    if len(pts_rc) < 2:
        return [], None

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
    return drawable_edges, cycle[0]


def _extract_shadow_chain(
    g: np.ndarray,
    start_rc: tuple[int, int],
    base_col: int,
    bg: int,
    max_layers: int = 3,
) -> list[tuple[int, int, int]]:
    """Read a simple 2-3 layer anchor chain from the original input."""
    h, w = g.shape
    chain: list[tuple[int, int, int]] = []
    used_colors = {base_col}
    used_positions = {start_rc}
    prev: tuple[int, int] | None = None
    cur = start_rc

    for _ in range(max_layers):
        candidates: list[tuple[int, int, int, int, int]] = []
        for drn, dcn in DIR8:
            rr = cur[0] + drn
            cc = cur[1] + dcn
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if prev is not None and (rr, cc) == prev:
                continue
            if (rr, cc) in used_positions:
                continue
            v = int(g[rr, cc])
            if v == bg or v in used_colors:
                continue
            candidates.append((rr, cc, drn, dcn, v))
        if not candidates:
            break
        rr, cc, drn, dcn, v = min(candidates, key=lambda x: (x[0], x[1], x[4]))
        chain.append((drn, dcn, v))
        used_colors.add(v)
        used_positions.add((rr, cc))
        prev, cur = cur, (rr, cc)

    return chain


def _draw_shifted_edges(
    out: np.ndarray,
    g: np.ndarray,
    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]],
    offset: tuple[int, int],
    col: int,
    bg: int,
    *,
    partial: bool,
) -> None:
    h, w = g.shape
    dr_off, dc_off = offset
    for edge_idx, ((r0, c0), (r1, c1)) in enumerate(drawable_edges):
        rr0 = r0 + dr_off
        cc0 = c0 + dc_off
        rr1 = r1 + dr_off
        cc1 = c1 + dc_off
        if not (0 <= rr0 < h and 0 <= cc0 < w and 0 <= rr1 < h and 0 <= cc1 < w):
            continue
        if not _aligned_8(rr0, cc0, rr1, cc1):
            continue
        if not _clear_in_original(g, rr0, cc0, rr1, cc1, bg, allow=set()):
            continue
        if partial:
            _draw_partial_edge_bg_only(out, rr0, cc0, rr1, cc1, col, bg, edge_idx)
        else:
            _draw_line_bg_only(out, rr0, cc0, rr1, cc1, col, bg)


def _solve_impl(grid: np.ndarray, *, partial: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        drawable_edges, start_rc = _ordered_drawable_edges(g, col, bg)
        if not drawable_edges or start_rc is None:
            continue

        for edge_idx, ((r0, c0), (r1, c1)) in enumerate(drawable_edges):
            if partial:
                _draw_partial_edge_bg_only(out, r0, c0, r1, c1, col, bg, edge_idx)
            else:
                _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        chain = _extract_shadow_chain(g, start_rc, col, bg, max_layers=3)
        cum_r = 0
        cum_c = 0
        for drn, dcn, ncol in chain:
            cum_r += drn
            cum_c += dcn
            _draw_shifted_edges(
                out,
                g,
                drawable_edges,
                (cum_r, cum_c),
                ncol,
                bg,
                partial=partial,
            )

    return out


def _main_full_reference(grid: np.ndarray) -> np.ndarray:
    """Reference version with chained shadows but continuous trails."""
    return _solve_impl(grid, partial=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
        Connect each same-color vertex set into a contour, but make later edges
        progressively shorter; then repeat the same fading contour for each layer
        of a chained shadow-anchor sequence.

    Input structure:
        One or two base colors appear as 3-4 polygon vertices. Near the chosen
        start vertex of each base shape there is a 2-3 pixel chain of distinct
        shadow colors, where each shadow pixel is 8-neighbor-adjacent to the
        previous one. Optional distractor pixels may also appear and stay unchanged.

    Rule:
        Order the base vertices cyclically from the anchored start vertex. Draw
        edge 1 fully, edge 2 with 3/4 of its internal cells, edge 3 with 1/2,
        edge 4+ with 1/4, always keeping the edge endpoint. Then read the shadow
        chain step-by-step; each step adds to the cumulative offset, and the same
        truncated edge pattern is drawn again in that shadow color at the shifted
        contour location.

    Key insight:
        The anchor pixels do not start independent shapes. They define a recursive
        offset chain, and every layer copies the same edge-order fade pattern, so
        the output becomes nested fading contours rather than solid parallel lines.
    """
    return _solve_impl(grid, partial=True)


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_PLANS = [
    (2, [2, 2]),
    (2, [2, 3]),
    (2, [3, 2]),
    (1, [2]),
    (1, [3]),
]
SHAPE_PLAN_WEIGHTS = np.array([0.24, 0.16, 0.16, 0.20, 0.24], dtype=float)
DISTRACTOR_PROB = 0.25
PLACEMENT_PADDING = 2
PLACEMENT_PAD_CONNECTIVITY = 8

TRIANGLE_SIDE_RANGE = (4, 8)
RECT_H_RANGE = (4, 8)
RECT_W_RANGE = (4, 8)
DIAMOND_ARM_RANGE = (4, 7)
SHAPE_WEIGHTS = np.array([0.34, 0.33, 0.33], dtype=float)

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 5)
DISTRACTOR_RETRIES = 50

CHAIN_RETRIES = 80
SHAPE_PLACEMENT_RETRIES = 60
GENERATE_RETRIES = 500


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


_SHAPE_BUILDERS = [_build_triangle, _build_rectangle, _build_diamond]


def _make_vertices() -> list[tuple[int, int]]:
    weights = SHAPE_WEIGHTS / SHAPE_WEIGHTS.sum()
    idx = int(np.random.choice(len(_SHAPE_BUILDERS), p=weights))
    return _SHAPE_BUILDERS[idx]()


def _shape_edges(vertices: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]


def _shape_cells(vertices: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices):
        cells.update(_gen_segment_cells(a, b))
    return cells


def _edge_dirs(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return [_step(a, b) for a, b in _shape_edges(vertices)]


def _translate_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _collinear_with_any(offset: tuple[int, int], directions: list[tuple[int, int]]) -> bool:
    return any(ds_r * offset[1] - ds_c * offset[0] == 0 for ds_r, ds_c in directions)


def _is_adjacent(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) == 1


def _edge_lengths(vertices: list[tuple[int, int]]) -> list[int]:
    return [max(abs(b[0] - a[0]), abs(b[1] - a[1])) for a, b in _shape_edges(vertices)]


def _has_visible_gradient(vertices: list[tuple[int, int]]) -> bool:
    lengths = _edge_lengths(vertices)
    if len(lengths) < 3:
        return False
    previous = None
    truncated = False
    for edge_idx, steps in enumerate(lengths):
        kept = _partial_internal_count(steps, edge_idx)
        if previous is not None and kept > previous:
            return False
        if kept < max(0, steps - 1):
            truncated = True
        previous = kept
    return truncated


def _choose_shadow_steps(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
    layer_count: int,
) -> list[tuple[int, int]] | None:
    """Choose a simple 2-3 step chain whose translated contours stay separable."""
    anchor = min(vertices)
    directions = _edge_dirs(vertices)

    for _ in range(CHAIN_RETRIES):
        used_dirs: set[tuple[int, int]] = set()
        existing_points = set(vertices)
        occupied = set(shape_cells)
        steps: list[tuple[int, int]] = []
        prev_pos = anchor
        cum = (0, 0)
        ok = True

        for _layer in range(layer_count):
            candidates: list[tuple[int, int]] = []
            for off in DIR8:
                if off in used_dirs:
                    continue
                new_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
                if new_pos in existing_points or new_pos in shape_cells:
                    continue
                if any(_is_adjacent(new_pos, pt) for pt in existing_points if pt != prev_pos):
                    continue
                new_cum = (cum[0] + off[0], cum[1] + off[1])
                if _collinear_with_any(new_cum, directions):
                    continue
                shifted = _translate_cells(shape_cells, new_cum)
                if not (shifted - occupied):
                    continue
                candidates.append(off)

            if not candidates:
                ok = False
                break

            off = candidates[int(np.random.randint(len(candidates)))]
            new_pos = (prev_pos[0] + off[0], prev_pos[1] + off[1])
            cum = (cum[0] + off[0], cum[1] + off[1])
            used_dirs.add(off)
            existing_points.add(new_pos)
            occupied |= _translate_cells(shape_cells, cum)
            steps.append(off)
            prev_pos = new_pos

        if ok:
            return steps

    return None


def _make_shape_patch(shape_color: int, shadow_colors: list[int]) -> tuple[np.ndarray, np.ndarray] | None:
    vertices = _make_vertices()
    if not _has_visible_gradient(vertices):
        return None

    shape_cells = _shape_cells(vertices)
    steps = _choose_shadow_steps(vertices, shape_cells, len(shadow_colors))
    if steps is None:
        return None

    anchor = min(vertices)
    all_cells = set(shape_cells)
    anchor_positions: list[tuple[int, int]] = []
    cum = (0, 0)
    pos = anchor
    for off in steps:
        cum = (cum[0] + off[0], cum[1] + off[1])
        pos = (pos[0] + off[0], pos[1] + off[1])
        anchor_positions.append(pos)
        all_cells |= _translate_cells(shape_cells, cum)

    r0, c0, r1, c1 = bbox_coords(np.array(sorted(all_cells), dtype=int))
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)

    for r, c in vertices:
        inp[r - r0, c - c0] = int(shape_color)
    for (r, c), col in zip(anchor_positions, shadow_colors):
        rr = r - r0
        cc = c - c0
        if not (0 <= rr < h and 0 <= cc < w) or inp[rr, cc] != BG:
            return None
        inp[rr, cc] = int(col)

    out = main(inp)
    if np.array_equal(out, _main_full_reference(inp)):
        return None

    local_anchor = (anchor[0] - r0, anchor[1] - c0)
    if len(_extract_shadow_chain(inp, local_anchor, int(shape_color), BG, max_layers=3)) != len(shadow_colors):
        return None
    if len(colors_present(inp, exclude=[BG])) != 1 + len(shadow_colors):
        return None
    if int(np.count_nonzero(out == int(shape_color))) <= len(vertices):
        return None
    for col in shadow_colors:
        if int(np.count_nonzero(out == int(col))) <= 1:
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


def _sample_canvas_size() -> tuple[int, int]:
    H = int(np.random.randint(*CANVAS_H_RANGE))
    W = int(np.clip(H + np.random.randint(*CANVAS_W_JITTER), *CANVAS_W_CLIP))
    return H, W


def _sample_shape_plan() -> tuple[list[int], bool]:
    weights = SHAPE_PLAN_WEIGHTS / SHAPE_PLAN_WEIGHTS.sum()
    idx = int(np.random.choice(len(SHAPE_PLANS), p=weights))
    shape_count, layer_counts = SHAPE_PLANS[idx]
    used_colors = shape_count + sum(layer_counts)
    has_distractor = used_colors < len(COLORS) and bool(np.random.rand() < DISTRACTOR_PROB)
    return list(layer_counts), has_distractor


def _sample_palette(
    layer_counts: list[int],
    has_distractor: bool,
) -> tuple[list[tuple[int, list[int]]], int | None] | None:
    total = len(layer_counts) + sum(layer_counts) + int(has_distractor)
    if total > len(COLORS):
        return None
    palette = list(np.random.permutation(COLORS)[:total])
    ptr = 0
    plans: list[tuple[int, list[int]]] = []
    for count in layer_counts:
        base = int(palette[ptr])
        ptr += 1
        shadows = [int(v) for v in palette[ptr : ptr + count]]
        ptr += count
        plans.append((base, shadows))
    distractor = int(palette[ptr]) if has_distractor else None
    return plans, distractor


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


def _gen_attempt() -> np.ndarray | None:
    H, W = _sample_canvas_size()
    out = np.full((H, W), BG, dtype=int)
    inp = np.full((H, W), BG, dtype=int)

    layer_counts, has_distractor = _sample_shape_plan()
    palette = _sample_palette(layer_counts, has_distractor)
    if palette is None:
        return None
    shape_plans, distractor_color = palette
    total_colors = len(layer_counts) + sum(layer_counts) + int(has_distractor)

    for shape_color, shadow_colors in shape_plans:
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_colors)
            if patch is None:
                continue
            if _try_place(inp, out, patch[0], patch[1]):
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
    if np.array_equal(main(inp), _main_full_reference(inp)):
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
