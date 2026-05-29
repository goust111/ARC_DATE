# Variant of 35ab12c3
# Dimension: D9 — 干扰项设计 (obstacle)
# Change: 可选的干扰项组（0-1组，1-3个像素），其像素不能H/V/对角线对齐，在输出中保持不变... → 无干扰项
#
# Design analysis for the D9 variant
# a) Concrete behavior and trace
#    - Input now contains only SHAPE vertex colors and optional SHADOW anchor colors.
#      There is no color group whose pixels are meant to remain unchanged as a free
#      obstacle. A 1-pixel color is therefore only legal as a shadow anchor adjacent
#      to one shape vertex; a 2+-pixel color is a shape to be connected.
#    - Original flow: `out = g.copy()` meant any unprocessed color group survived into
#      the output. That was correct for distractors because `len(pts_rc) < 2` or
#      "no drawable edge" groups were supposed to remain visible.
#    - Variant flow: start from a blank output and redraw only shape edges plus shadow
#      copies. Example: if a color had three mutually non-aligned pixels, the original
#      solver would leave them there via `g.copy()`. The variant must not preserve such
#      a group, because the generator no longer produces it and every visible pixel must
#      belong to some connectable shape or shadow.
# b) Complexity calibration
#    - The task is simplified only along D9: distractor colors are removed.
#      Shape families, line directions, optional shadows, palette uniqueness, canvas
#      size, and spacing are unchanged. This keeps multiple colors and topologies in
#      each sample, so the rule is still non-trivial and visually recognizable.
# c) Hidden-assumption audit for `main`
#    - `out = g.copy()` relied on the old invariant that some colors may be valid even
#      if never redrawn. That invariant is false in the variant, so output must be
#      rebuilt from background.
#    - `if len(pts_rc) < 2: continue` is still valid: singletons are shadow anchors and
#      participate indirectly when their owning shape is processed.
#    - Neighbor scanning for `seed_idxs` / `carried` used to accept either a shadow
#      anchor or a distractor touching the start vertex. The generator now guarantees
#      such neighbors, when present, are only shadow anchors.
#    - `_clear_in_original(... allow=set())` for shadow paths remains valid because the
#      input retains only the anchor pixel of that shadow color, not an entire object.
# d) Placement legality
#    - Generator still uses output-first patches and `random_free_location(..., padding=1)`
#      so filled shapes plus their shadows do not overlap or touch other groups.
#      Removing distractors only deletes one optional patch type; it does not weaken the
#      spacing or bounds guarantees.
# e) Visual observability
#    - Across samples there are no untouched speckles or color groups that stay identical
#      from input to output. Every non-background color either becomes a connected shape
#      or acts as a shadow anchor for one.
# f) Visual cleanliness
#    - The no-distractor variant is cleaner than the source task because all non-bg
#      pixels now participate in the same two mechanisms. Padding and unique colors are
#      preserved to keep groups separable.
# g) Color inference
#    - No new color remapping is introduced. Colors remain object-local and unique.
# 9) New `main()` assumptions and generator guarantees
#    - Any singleton color is assumed to be a shadow anchor rather than an obstacle.
#      `_make_shape_patch()` is the only producer of singleton colors.
#    - Any 2+-pixel color is assumed to be a connectable shape with H/V/diag edges.
#      `_make_vertices()` plus oracle checking preserve this.
#    - Different groups are assumed not to overlap in their final filled output.
#      `_try_place()` with padding preserves that.

import numpy as np
import math

from arc_utils import get_bg as _bg_color
from arc_utils import bbox_coords, colors_present, draw_line, get_bg, neighbors8, paste, random_free_location

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
        # Skip last cell (endpoint).
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
    # Order points by polar angle around the centroid (ties by radius, then (r,c)).
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

    # D9 change: rebuild from background so unsupported leftover color groups do not
    # silently survive as "obstacles". Valid singleton colors are reintroduced only
    # through the shadow-copy phase of their owning shape.
    out = np.full_like(g, bg)

    colors = sorted(set(int(v) for v in np.unique(g)) - {bg})
    for col in colors:
        pts_rc = [tuple(map(int, p)) for p in np.argwhere(g == col)]
        if len(pts_rc) < 2:
            continue

        cycle = _angle_cycle(pts_rc)
        if len(cycle) < 2:
            continue

        # Choose a start vertex that has a non-bg, different-color neighbor in the original input.
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
            # Stable: pick the seed vertex with smallest (r,c).
            start = min(seed_idxs, key=lambda i: (cycle[i][0], cycle[i][1]))
        else:
            start = min(range(len(cycle)), key=lambda i: (cycle[i][0], cycle[i][1]))

        cycle = cycle[start:] + cycle[:start]

        # Carry neighbor offsets/colors from the chosen start vertex.
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

        # Phase 1: draw all main hull edges (avoids parallel lines blocking main lines).
        for (r0, c0), (r1, c1) in drawable_edges:
            _draw_line_bg_only(out, r0, c0, r1, c1, col, bg)

        # Phase 2: draw parallel edges for the carried neighbors.
        for (r0, c0), (r1, c1) in drawable_edges:
            drs, dcs = _step_dir(r0, c0, r1, c1)
            for drn, dcn, ncol in carried:
                # Skip collinear offsets (along the edge direction or directly opposite).
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

# --------------- generator (attempt 2 (Codex)) ---------------

# ---- CFG: canvas and palette ----
BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)          # randint hi exclusive: H in [15, 22]
CANVAS_W_JITTER = (-3, 4)          # W = clip(H + randint(lo, hi), 15, 22); hi exclusive
CANVAS_W_CLIP = (15, 22)

# ---- CFG: structural counts ----
SHAPE_COUNT_RANGE = (2, 5)         # randint hi exclusive: shape_count in [2, 4]
MAX_SHADOWS_CAP = 3                # cap on number of shadows attached
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

# ---- CFG: shape geometry ----
LINE_VERTICES_RANGE = (2, 8)       # n_vertices in [2, 7]
LINE_TOTAL_LENGTH = 13             # total span of a polyline
LINE_MIN_GAP = 2                   # min gap between consecutive polyline vertices
TRIANGLE_SIDE_RANGE = (2, 8)
RECT_H_RANGE = (2, 9)
RECT_W_RANGE = (2, 9)
DIAMOND_ARM_RANGE = (2, 6)
PENTAGON_ROOF_RANGE = (2, 5)
PENTAGON_WALL_RANGE = (2, 7)
HEXAGON_SLANT_RANGE = (2, 5)
HEXAGON_WALL_RANGE = (2, 6)

# ---- CFG: shape mixture ----
# builders list is aligned with SHAPE_WEIGHTS below
SHAPE_WEIGHTS = np.array([0.28, 0.18, 0.18, 0.14, 0.12, 0.10], dtype=float)

# ---- CFG: retry budgets ----
SHAPE_PLACEMENT_RETRIES = 40
GENERATE_RETRIES = 300


def _step(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    return (0 if dr == 0 else (1 if dr > 0 else -1), 0 if dc == 0 else (1 if dc > 0 else -1))


def _gen_segment_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    drs, dcs = _step(a, b)
    steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
    return [(a[0] + drs * k, a[1] + dcs * k) for k in range(steps + 1)]


def _transform_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Randomly transpose / flip axes, then normalize to non-negative indices."""
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
    """Random polyline: n vertices spaced along one axis, total span LINE_TOTAL_LENGTH."""
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
    """Pick anchor vertex and 8-neighbor offset for a shadow pixel.

    Returns (anchor_rc, offset_drdc) or None if no valid position exists.
    The offset must not be collinear with any edge direction (otherwise the shadow
    copy of a collinear edge overlaps the shape), and the shadow pixel must land
    on a cell that is not part of the shape and is 8-adjacent to exactly one vertex.
    """
    anchor = min(vertices)
    directions = _edge_dirs(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        # Skip offsets collinear with any edge (cross-product == 0).
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


def _patch_bbox(shape_cells: set[tuple[int, int]], vertices: list[tuple[int, int]],
                shadow: tuple[tuple[int, int], tuple[int, int]] | None) -> tuple[int, int, int, int]:
    """Bounding box covering the shape + (optionally) its shadow copy and shadow anchor."""
    coords = list(shape_cells) + vertices
    if shadow is not None:
        (ar, ac), (dr, dc) = shadow
        coords.append((ar + dr, ac + dc))
        coords.extend((r + dr, c + dc) for r, c in shape_cells)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_shadow(out: np.ndarray, shape_color: int, shadow_color: int, offset: tuple[int, int]) -> None:
    """Paint the shape-shaped shadow at `offset`, background cells only."""
    dr, dc = offset
    h, w = out.shape
    src = (out == shape_color)
    # Shift `src` by `offset` into a same-sized buffer (bounds via slicing).
    shifted = np.zeros_like(src)
    r_src = slice(max(0, -dr), h - max(0, dr))
    c_src = slice(max(0, -dc), w - max(0, dc))
    r_dst = slice(max(0, dr), h - max(0, -dr))
    c_dst = slice(max(0, dc), w - max(0, -dc))
    shifted[r_dst, c_dst] = src[r_src, c_src]
    mask = shifted & (out == BG)
    out[mask] = shadow_color


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray] | None:
    """Build a local (input, output) patch for one shape, optionally with one shadow.

    Returns None if geometry is infeasible or the solver oracle disagrees.
    """
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
        # Shadow anchor pixel must be 8-adjacent to exactly one vertex (so solver
        # can unambiguously pair it with the anchor).
        nb_hits = sum(1 for rr, cc in neighbors8(anchor[0], anchor[1])
                      if 0 <= rr < h and 0 <= cc < w and inp[rr, cc] == shadow_color)
        if nb_hits != 1:
            return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def _sample_palette(shape_count: int) -> tuple[list[int], list[int]] | None:
    """Return (shape_colors, shadow_colors). No distractor color exists in D9."""
    max_shadows = min(MAX_SHADOWS_CAP, shape_count, len(COLORS) - shape_count)
    if max_shadows < 0:
        return None
    shadow_count = int(np.random.randint(max_shadows + 1))
    total = shape_count + shadow_count
    palette = list(np.random.permutation(COLORS)[:total])
    shape_colors = [int(v) for v in palette[:shape_count]]
    shadow_colors = [int(v) for v in palette[shape_count:shape_count + shadow_count]]
    return shape_colors, shadow_colors


def _assign_shadows(shape_count: int, shadow_colors: list[int]) -> dict[int, int]:
    """Pick which shape indices receive a shadow, map shape_idx -> shadow_color."""
    if not shadow_colors:
        return {}
    targets = np.random.choice(shape_count, size=len(shadow_colors), replace=False).tolist()
    return {int(idx): int(col) for idx, col in zip(sorted(targets), shadow_colors)}


def _try_place(inp: np.ndarray, out: np.ndarray, patch_in: np.ndarray, patch_out: np.ndarray) -> bool:
    """Attempt to paste one shape patch into the canvas with padding."""
    try:
        loc = random_free_location(out, patch_out, transparent=BG,
                                    padding=PLACEMENT_PADDING,
                                    padding_connectivity=PLACEMENT_PAD_CONNECTIVITY)
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

    palette = _sample_palette(shape_count)
    if palette is None:
        return None
    shape_colors, shadow_colors = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)

    total_colors = shape_count + len(shadow_colors)

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
