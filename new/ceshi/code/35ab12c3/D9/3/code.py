# Variant of 35ab12c3
# Dimension: D9 — 干扰项设计 (obstacle)
# Change: 可选的干扰项组（0-1组，1-3个像素），其像素不能H/V/对角线对齐，在输出中保持不变... → 多组干扰项且数量更多

import math
import numpy as np

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location


# Design analysis for the D9 variant
# ----------------------------------
# a) Concrete behavior and trace
#    - Input still mixes shape-colors and shadow-anchor colors, but distractors are now
#      always present as 2-3 distinct colors.
#    - Each distractor color forms a sparse set of 3-5 pixels. Within one distractor
#      group, no pair of pixels may be horizontally, vertically, or 45-degree aligned.
#      Therefore the solver must leave the whole group unchanged.
#    - Example trace:
#        Original: one distractor color could be a 2-pixel or 3-pixel set. The solver
#        loops over each color; for a distractor it usually finds no drawable edge and
#        the copied output stays unchanged.
#        Variant: a distractor can now have 5 pixels and there can be 3 such colors.
#        The solver still loops by color, but it must explicitly reject any color whose
#        angle-cycle is not a fully drawable shape. Otherwise a larger distractor set
#        would create more opportunities for accidental partial connections.
# b) Complexity calibration
#    - The variant keeps shape and shadow rules intact so the core concept remains
#      recognizable, but increases distractor count and size enough to be visually
#      obvious across 2-4 examples.
#    - Shape count stays in the original 2-4 range; shadow count also keeps the
#      original cap, but the palette budget may force crowded examples to use fewer
#      shadows. This localizes the change to D9 instead of simplifying the base task.
# c) Implicit-assumption audit of main()
#    - Original `len(pts_rc) < 2: continue` assumed non-shapes were tiny and harmless.
#      With 3-5 distractor pixels, this is no longer sufficient.
#    - Original `drawable_edges.append(...)` then drawing whatever survived assumed
#      partial edge sets were not meaningful. In the variant, partial edge sets on a
#      distractor would be a bug, so valid shapes now require every expected edge to be
#      aligned and clear in the original input.
#    - Seed selection and carried-neighbor logic still assume different-color neighbors
#      only come from shadows. The generator enforces >=1-cell 8-connected spacing
#      between groups, so distractors never touch shape vertices.
# d) Placement legality
#    - Shapes are placed from output patches exactly as before.
#    - Distractors use a filled-bbox placement mask, so no other object can occupy the
#      holes inside a sparse distractor patch; this keeps groups visually separate.
# e) Visual observability
#    - Humans can now see multiple large sparse color sets that remain unchanged while
#      only the meaningful aligned shape-vertex sets get connected.
# f) Visual cleanliness
#    - Filled-bbox placement for distractors and the existing padding between objects
#      prevent interleaving clutter.
# g) Color inference
#    - No color-id-based ordering or recoloring is introduced. Each group still keeps
#      its own color identity; only geometry distinguishes shape groups from distractors.
# h) New solver assumptions introduced here
#    - Every real shape color forms a cycle where all expected edges are aligned and
#      clear in the original grid.
#    - Every distractor color violates that condition.
#    - `generate_input()` guarantees both conditions explicitly.


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


def _shape_edges_from_cycle(cycle: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if len(cycle) == 2:
        return [(cycle[0], cycle[1])]
    return [(cycle[i], cycle[(i + 1) % len(cycle)]) for i in range(len(cycle))]


def _all_edges_drawable(g: np.ndarray, cycle: list[tuple[int, int]], col: int, bg: int):
    edges = _shape_edges_from_cycle(cycle)
    drawable_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (r0, c0), (r1, c1) in edges:
        if not _aligned_8(r0, c0, r1, c1):
            return None
        if not _clear_in_original(g, r0, c0, r1, c1, bg, allow={col}):
            return None
        drawable_edges.append(((r0, c0), (r1, c1)))
    return drawable_edges


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
        drawable_edges = _all_edges_drawable(g, cycle, col, bg)
        if drawable_edges is None:
            continue

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = cycle[0]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

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

# ---- CFG: canvas and palette ----
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

# ---- CFG: structural counts ----
SHAPE_COUNT_RANGE = (2, 5)               # 2-4 shapes
MAX_SHADOWS_CAP = 3
DISTRACTOR_GROUP_RANGE = (2, 4)          # 2-3 groups
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: shape geometry ----
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

# ---- CFG: shape mixture ----
SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

# ---- CFG: distractor geometry ----
DISTRACTOR_PIXEL_RANGE = (3, 6)          # 3-5 pixels per group
DISTRACTOR_BBOX_RANGE = (5, 9)           # 5-8 rows/cols
DISTRACTOR_RETRIES = 60

# ---- CFG: retry budgets ----
SHAPE_PLACEMENT_RETRIES = 50
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
    return _shape_edges_from_cycle(vertices)


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
    src = out == shape_color
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
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
        nb_hits = sum(
            1
            for rr, cc in neighbors8(anchor[0], anchor[1])
            if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color
        )
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _sample_distractor_coords(h: int, w: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(h) for c in range(w)]
    perm = np.random.permutation(len(cells))
    cells = [cells[int(i)] for i in perm]

    def backtrack(
        start: int,
        chosen: list[tuple[int, int]],
        rows: set[int],
        cols: set[int],
        diag_a: set[int],
        diag_b: set[int],
    ) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return chosen.copy()
        if len(cells) - start < count - len(chosen):
            return None
        for idx in range(start, len(cells)):
            r, c = cells[idx]
            if r in rows or c in cols or (r - c) in diag_a or (r + c) in diag_b:
                continue
            chosen.append((r, c))
            rows.add(r)
            cols.add(c)
            diag_a.add(r - c)
            diag_b.add(r + c)
            found = backtrack(idx + 1, chosen, rows, cols, diag_a, diag_b)
            if found is not None:
                return found
            chosen.pop()
            rows.remove(r)
            cols.remove(c)
            diag_a.remove(r - c)
            diag_b.remove(r + c)
        return None

    return backtrack(0, [], set(), set(), set(), set())


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    count = int(np.random.randint(*DISTRACTOR_PIXEL_RANGE))
    for _ in range(DISTRACTOR_RETRIES):
        low = max(count, DISTRACTOR_BBOX_RANGE[0])
        h = int(np.random.randint(low, DISTRACTOR_BBOX_RANGE[1]))
        w = int(np.random.randint(low, DISTRACTOR_BBOX_RANGE[1]))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        placement = np.ones((h, w), dtype=int)
        return patch.copy(), patch.copy(), placement
    return None


def _sample_palette(shape_count: int, distractor_count: int) -> tuple[list[int], list[int], list[int]] | None:
    remaining = len(COLORS) - shape_count - distractor_count
    if remaining < 0:
        return None
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, remaining)
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count + distractor_count
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count:shape_count + shadow_count]]
    distractor_colors = [int(v) for v in palette[shape_count + shadow_count:]]
    return shape_colors, shadow_colors, distractor_colors


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(
    inp: np.ndarray,
    out: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    placement_patch: np.ndarray | None = None,
) -> bool:
    search_patch = patch_out if placement_patch is None else placement_patch
    try:
        loc = random_free_location(
            out,
            search_patch,
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

    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    distractor_count = int(np.random.randint(*DISTRACTOR_GROUP_RANGE))

    palette = _sample_palette(shape_count, distractor_count)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_colors = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors) + len(distractor_colors)

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

    for distractor_color in distractor_colors:
        patch = _make_distractor_patch(int(distractor_color))
        if patch is None:
            return None
        if not _try_place(inp, out, patch[0], patch[1], placement_patch=patch[2]):
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
