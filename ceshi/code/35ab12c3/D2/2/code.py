# Variant of 35ab12c3
# Dimension: D2 — 路径方向类型 (path)
# Change: straight_ray + diagonal_path——顶点间通过水平、垂直或严格45度对角线连接，要求dr==0或... → 骑士步路径
#
# Design analysis for this variant
# a) Concrete behavior:
#    Each same-color shape is still represented in the input by vertices only, plus
#    one optional 8-neighbor shadow-anchor pixel of another color. The output no
#    longer connects consecutive vertices with straight H/V/diag segments. Instead,
#    every drawable edge must be a single chess-knight displacement: abs(dr),abs(dc)
#    is (2,1) or (1,2). The filled path is the unique canonical "2 then 1" L-shape,
#    but because edges are undirected the path orientation cannot depend on traversal
#    order. This variant therefore first picks the lexicographically smaller endpoint
#    as the canonical start, then moves along the longer axis for two cells, then
#    moves one cell along the shorter axis into the other endpoint. Example:
#    {(r,c),(r+2,c+1)} paints (r,c),(r+1,c),(r+2,c),(r+2,c+1). The shadow still copies
#    the whole filled shape
#    by one 8-neighbor offset, but now that copied geometry is also a bent knight path.
# b) Complexity calibration:
#    If every color used only a 2-vertex shape, the variant would be visually weak.
#    The generator therefore mixes open chains and closed cycles, and rejects grids
#    where every shape has only one edge. This keeps repeated L-turn evidence in each
#    sample without exploding the number of cases.
# c) Old implicit assumptions that must change:
#    - _aligned_8 / _segment_cells / _step_dir assumed straight rays; they are invalid
#      because a knight edge is bent and only legal for deltas (2,1)/(1,2).
#    - _clear_in_original previously scanned a straight interior. Here the interior is
#      exactly the two bend cells of the canonical L-path.
#    - The old "parallel shadow edge" check used edge-direction collinearity. Knight
#      paths have no single line direction, so the variant removes that assumption and
#      instead validates each shifted filled path cell directly.
#    - Angle-sorting vertices was only a proxy for reconstructing path/cycle order.
#      For knight geometry the robust invariant is the knight-adjacency graph, so the
#      solver reconstructs a single connected path/cycle from legal knight edges.
# d) Placement legality:
#    Generator-built shapes forbid non-consecutive knight-adjacent vertex pairs and
#    forbid path-cell overlaps except shared consecutive endpoints. Optional shadows
#    are chosen so the shifted filled shape is disjoint from the base shape, and all
#    patches are still placed with >=1-cell 8-neighborhood padding on the global grid.
# e) Visual observability:
#    The rule is directly visible in output: straight connections disappear and become
#    repeated L-shaped bends. Because the input still only shows sparse vertices, the
#    input/output relation remains inferential rather than explicit copying.
# f) Visual cleanliness:
#    Shadows are kept disjoint from their base shape inside the local patch, which
#    avoids messy partial overlap between two bent traces. Distinct groups are also
#    padded apart globally, so different colors do not tangle.
# g) Color inference:
#    Colors keep their original role: one color per shape, optional extra color for a
#    copied shadow, optional distractor color left unchanged. No sorting or mapping
#    depends on color identity beyond harmless processing order.
# 9) New solver assumptions and how generation guarantees them:
#    - Same-color shapes must induce exactly one connected knight graph with all
#      degrees in {1,2} and either 2 endpoints (open chain) or all degree 2 (cycle).
#      Generator enforces this by construction and validates with main().
#    - Shadow anchors must be 8-adjacent to exactly one vertex and the shifted knight
#      paths must be clear in the original input except for that single anchor cell.
#      Generator explicitly checks both conditions before accepting a patch.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, get_bg as _bg_color, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _is_knight_delta(dr: int, dc: int) -> bool:
    return (abs(dr), abs(dc)) in {(1, 2), (2, 1)}


def _is_knight_pair(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return _is_knight_delta(b[0] - a[0], b[1] - a[1])


def _knight_path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    if b < a:
        a, b = b, a
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    adr = abs(dr)
    adc = abs(dc)
    if (adr, adc) not in {(2, 1), (1, 2)}:
        return None

    sr = _sign(dr)
    sc = _sign(dc)
    cells = [(r0, c0)]
    if adr == 2:
        cells.append((r0 + sr, c0))
        cells.append((r0 + 2 * sr, c0))
    else:
        cells.append((r0, c0 + sc))
        cells.append((r0, c0 + 2 * sc))
    cells.append((r1, c1))
    return cells


def _clear_path_in_original(g: np.ndarray, cells: list[tuple[int, int]], bg: int, allow: set[int]) -> bool:
    for rr, cc in cells[1:-1]:
        v = int(g[rr, cc])
        if v != bg and v not in allow:
            return False
    return True


def _clear_shifted_path_in_original(
    g: np.ndarray,
    cells: list[tuple[int, int]],
    bg: int,
    anchor: tuple[int, int],
    anchor_color: int,
) -> bool:
    for rr, cc in cells:
        v = int(g[rr, cc])
        if (rr, cc) == anchor:
            if v != bg and v != anchor_color:
                return False
        elif v != bg:
            return False
    return True


def _draw_path_bg_only(out: np.ndarray, cells: list[tuple[int, int]], col: int, bg: int) -> None:
    for rr, cc in cells:
        if out[rr, cc] == bg:
            out[rr, cc] = col


def _shape_graph(
    g: np.ndarray,
    pts_rc: list[tuple[int, int]],
    bg: int,
) -> tuple[list[set[int]], dict[tuple[int, int], list[tuple[int, int]]]]:
    n = len(pts_rc)
    adj = [set() for _ in range(n)]
    edge_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for i in range(n):
        for j in range(i + 1, n):
            cells = _knight_path_cells(pts_rc[i], pts_rc[j])
            if cells is None:
                continue
            if not _clear_path_in_original(g, cells, bg, allow=set()):
                continue
            adj[i].add(j)
            adj[j].add(i)
            edge_cells[(i, j)] = cells
    return adj, edge_cells


def _is_single_shape_graph(adj: list[set[int]]) -> bool:
    if len(adj) < 2:
        return False
    degs = [len(nbs) for nbs in adj]
    if any(d == 0 or d > 2 for d in degs):
        return False

    seen = set()
    stack = [0]
    while stack:
        cur = stack.pop()
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(adj[cur] - seen)
    if len(seen) != len(adj):
        return False

    edge_count = sum(degs) // 2
    deg1 = sum(d == 1 for d in degs)
    if edge_count == len(adj) - 1 and deg1 == 2:
        return True
    if edge_count == len(adj) and deg1 == 0 and all(d == 2 for d in degs):
        return True
    return False


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

        seed_idxs: list[int] = []
        for i, (r, c) in enumerate(pts_rc):
            for drn, dcn in DIR8:
                rr = r + drn
                cc = c + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        seed_idxs.append(i)
                        break

        if seed_idxs:
            start_idx = min(seed_idxs, key=lambda i: pts_rc[i])
        else:
            start_idx = min(range(len(pts_rc)), key=lambda i: pts_rc[i])

        carried: list[tuple[int, int, int]] = []
        sr0, sc0 = pts_rc[start_idx]
        for drn, dcn in DIR8:
            rr = sr0 + drn
            cc = sc0 + dcn
            if 0 <= rr < h and 0 <= cc < w:
                v = int(g[rr, cc])
                if v != bg and v != col:
                    carried.append((drn, dcn, v))

        adj, edge_cells = _shape_graph(g, pts_rc, bg)
        if not _is_single_shape_graph(adj):
            continue

        drawable_paths = [
            cells
            for _, cells in sorted(
                edge_cells.items(),
                key=lambda kv: (pts_rc[kv[0][0]], pts_rc[kv[0][1]]),
            )
        ]

        for cells in drawable_paths:
            _draw_path_bg_only(out, cells, col, bg)

        for drn, dcn, ncol in carried:
            anchor = (sr0 + drn, sc0 + dcn)
            for cells in drawable_paths:
                shifted = [(rr + drn, cc + dcn) for rr, cc in cells]
                if not all(0 <= rr < h and 0 <= cc < w for rr, cc in shifted):
                    continue
                if not _clear_shifted_path_in_original(g, shifted, bg, anchor, ncol):
                    continue
                _draw_path_bg_only(out, shifted, ncol, bg)

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
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

CHAIN_VERTICES_RANGE = (2, 7)      # 2-6 vertices
CYCLE_VERTICES_RANGE = (4, 7)      # 4-6 vertices
SHAPE_MAX_SPAN = 10
SHAPE_BUILD_RETRIES = 180
CHAIN_WEIGHT = 0.7

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 6)
DISTRACTOR_RETRIES = 60

SHAPE_PLACEMENT_RETRIES = 50
GENERATE_RETRIES = 400


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
    arr -= arr.min(axis=0)
    return [tuple(map(int, p)) for p in arr]


def _shape_edges(vertices: list[tuple[int, int]], closed: bool) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    if closed:
        return [(vertices[i], vertices[(i + 1) % len(vertices)]) for i in range(len(vertices))]
    return [(vertices[i], vertices[i + 1]) for i in range(len(vertices) - 1)]


def _shape_cells(vertices: list[tuple[int, int]], closed: bool) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for a, b in _shape_edges(vertices, closed):
        path = _knight_path_cells(a, b)
        if path is None:
            return set()
        cells.update(path)
    return cells


def _ordered_shape_valid(vertices: list[tuple[int, int]], closed: bool) -> bool:
    n = len(vertices)
    if (closed and n < 4) or (not closed and n < 2):
        return False

    for i in range(n):
        for j in range(i + 1, n):
            consecutive = (j == i + 1) or (closed and i == 0 and j == n - 1)
            if _is_knight_pair(vertices[i], vertices[j]) != consecutive:
                return False

    occupied: set[tuple[int, int]] = set()
    edges = _shape_edges(vertices, closed)
    for idx, (a, b) in enumerate(edges):
        path = _knight_path_cells(a, b)
        if path is None:
            return False
        allowed = {a}
        if closed and idx == len(edges) - 1:
            allowed.add(b)
        if any(cell in occupied and cell not in allowed for cell in path):
            return False
        occupied.update(path)

    if not occupied:
        return False
    r0, c0, r1, c1 = bbox_coords(np.array(list(occupied), dtype=int))
    if (r1 - r0 + 1) > SHAPE_MAX_SPAN or (c1 - c0 + 1) > SHAPE_MAX_SPAN:
        return False
    return True


def _build_chain() -> tuple[list[tuple[int, int]], bool] | None:
    target = int(np.random.randint(*CHAIN_VERTICES_RANGE))
    for _ in range(SHAPE_BUILD_RETRIES):
        vertices = [(0, 0)]
        occupied = {(0, 0)}
        while len(vertices) < target:
            cur = vertices[-1]
            candidates: list[tuple[int, int]] = []
            for dr, dc in np.random.permutation(np.array(KNIGHT_MOVES, dtype=int)):
                nxt = (cur[0] + int(dr), cur[1] + int(dc))
                if nxt in vertices:
                    continue
                if any(_is_knight_pair(prev, nxt) for prev in vertices[:-1]):
                    continue
                path = _knight_path_cells(cur, nxt)
                assert path is not None
                if any(cell in occupied and cell != cur for cell in path):
                    continue
                candidates.append(nxt)
            if not candidates:
                break
            nxt = candidates[int(np.random.randint(len(candidates)))]
            path = _knight_path_cells(cur, nxt)
            assert path is not None
            vertices.append(nxt)
            occupied.update(path)
        if len(vertices) != target:
            continue
        vertices = _transform_coords(vertices)
        if _ordered_shape_valid(vertices, closed=False):
            return vertices, False
    return None


def _build_cycle() -> tuple[list[tuple[int, int]], bool] | None:
    target = int(np.random.randint(*CYCLE_VERTICES_RANGE))
    for _ in range(SHAPE_BUILD_RETRIES):
        vertices = [(0, 0)]
        occupied = {(0, 0)}
        while len(vertices) < target:
            cur = vertices[-1]
            first = vertices[0]
            need_last = len(vertices) == target - 1
            candidates: list[tuple[int, int]] = []
            for dr, dc in np.random.permutation(np.array(KNIGHT_MOVES, dtype=int)):
                nxt = (cur[0] + int(dr), cur[1] + int(dc))
                if nxt in vertices:
                    continue

                path = _knight_path_cells(cur, nxt)
                assert path is not None
                if any(cell in occupied and cell != cur for cell in path):
                    continue

                bad_extra = False
                for idx, prev in enumerate(vertices[:-1]):
                    is_cycle_neighbor = need_last and idx == 0 and _is_knight_pair(prev, nxt)
                    if _is_knight_pair(prev, nxt) and not is_cycle_neighbor:
                        bad_extra = True
                        break
                if bad_extra:
                    continue

                if need_last:
                    if not _is_knight_pair(nxt, first):
                        continue
                    close_path = _knight_path_cells(nxt, first)
                    assert close_path is not None
                    new_occupied = occupied | set(path)
                    if any(cell in new_occupied and cell not in {nxt, first} for cell in close_path):
                        continue
                else:
                    if len(vertices) > 1 and _is_knight_pair(nxt, first):
                        continue
                candidates.append(nxt)

            if not candidates:
                break
            nxt = candidates[int(np.random.randint(len(candidates)))]
            path = _knight_path_cells(cur, nxt)
            assert path is not None
            vertices.append(nxt)
            occupied.update(path)

        if len(vertices) != target:
            continue
        vertices = _transform_coords(vertices)
        if _ordered_shape_valid(vertices, closed=True):
            return vertices, True
    return None


def _make_vertices() -> tuple[list[tuple[int, int]], bool] | None:
    if np.random.rand() < CHAIN_WEIGHT:
        return _build_chain()
    return _build_cycle()


def _shift_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _choose_shadow(
    vertices: list[tuple[int, int]],
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = min(vertices)
    candidates: list[tuple[int, int]] = []
    for dr, dc in DIR8:
        shadow_anchor = (anchor[0] + dr, anchor[1] + dc)
        if shadow_anchor in shape_cells:
            continue
        if sum(max(abs(shadow_anchor[0] - vr), abs(shadow_anchor[1] - vc)) == 1 for vr, vc in vertices) != 1:
            continue
        shifted = _shift_cells(shape_cells, (dr, dc))
        if shifted & shape_cells:
            continue
        candidates.append((dr, dc))
    if not candidates:
        return None
    return anchor, candidates[int(np.random.randint(len(candidates)))]


def _patch_bbox(
    shape_cells: set[tuple[int, int]],
    shadow_offset: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells)
    if shadow_offset is not None:
        coords.extend(_shift_cells(shape_cells, shadow_offset))
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _paint_cells(out: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[r, c] = color


def _make_shape_patch(shape_color: int, shadow_color: int | None) -> tuple[np.ndarray, np.ndarray, dict[str, int]] | None:
    spec = _make_vertices()
    if spec is None:
        return None
    vertices, closed = spec
    cells = _shape_cells(vertices, closed)
    if not cells:
        return None

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, cells)
        if shadow is None:
            return None

    shadow_offset = shadow[1] if shadow is not None else None
    r0, c0, r1, c1 = _patch_bbox(cells, shadow_offset)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)
    out = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    shifted_cells = {(r - r0, c - c0) for r, c in cells}

    for r, c in shifted_vertices:
        inp[r, c] = shape_color
    _paint_cells(out, shifted_cells, shape_color)

    if shadow is not None and shadow_color is not None:
        anchor, offset = shadow
        anchor = (anchor[0] - r0, anchor[1] - c0)
        shadow_cells = _shift_cells(shifted_cells, offset)
        shadow_anchor = (anchor[0] + offset[0], anchor[1] + offset[1])
        if shadow_anchor in shifted_cells or not (0 <= shadow_anchor[0] < h and 0 <= shadow_anchor[1] < w):
            return None
        if sum(
            max(abs(shadow_anchor[0] - vr), abs(shadow_anchor[1] - vc)) == 1
            for vr, vc in shifted_vertices
        ) != 1:
            return None
        _paint_cells(out, shadow_cells, int(shadow_color))
        inp[shadow_anchor] = int(shadow_color)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out, {"edge_count": len(_shape_edges(vertices, closed)), "closed": int(closed)}


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
            if _is_knight_pair(coords[i], coords[j]):
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
    shadow_colors = palette[shape_count : shape_count + shadow_count]
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
    max_edge_count = 0

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx))
            if patch is None:
                continue
            patch_in, patch_out, meta = patch
            if _try_place(inp, out, patch_in, patch_out):
                max_edge_count = max(max_edge_count, int(meta["edge_count"]))
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, *_make_distractor_patch(distractor_color)):
            return None

    if max_edge_count < 2:
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
