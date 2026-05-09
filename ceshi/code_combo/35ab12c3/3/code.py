# Combo variant of 35ab12c3 (index 3)
# combo_spec: D2/2+D4/0
# Name: 骑士步路径+间歇跳跃轨迹
#
# Design analysis for this combo
# a) Concrete behavior and trace:
#    - Preserved core: the input still shows sparse same-color vertices, plus an
#      optional 8-neighbor shadow-anchor pixel of another color. Distractors are
#      unchanged.
#    - D2/2 change: straight H/V/diag edges are replaced by chess-knight edges.
#      Each consecutive vertex pair must differ by (2,1) or (1,2), and the drawn
#      geometry is the L-shaped 4-cell walk that first travels 2 cells along the
#      longer axis, then 1 cell along the shorter axis.
#    - D4/0 change in the fused setting: the trail is no longer painted cell-by-cell
#      continuously. Instead, after reconstructing the whole ordered knight walk of
#      a shape, the solver advances one global dash phase across all non-vertex path
#      cells of that shape:
#        * 2 vertices -> paint 1, skip 1
#        * 3+ vertices -> paint 2, skip 1
#      All true vertex cells are always kept painted, and the shifted shadow follows
#      the exact same dash phase on the shifted knight walk.
#    - Example: vertices at (2,2) -> (4,3) -> (5,5) form two knight edges. The
#      solid walk would be:
#        (2,2),(3,2),(4,2),(4,3),(4,4),(4,5),(5,5)
#      With 3 vertices, only non-vertex cells consume the `paint 2, skip 1` rhythm:
#        interior order = (3,2),(4,2),(4,4),(4,5)
#        output paints (3,2),(4,2), skips (4,4), paints (4,5).
#      If the start vertex has a shadow anchor at offset (+1,0), the shifted walk
#      gets the same keep/skip pattern.
# b) Complexity calibration:
#    - A knight edge is short; if every sample only had 2-vertex shapes, the fusion
#      would look like tiny isolated Ls and the `2+` dash mode would never appear.
#      The generator therefore guarantees at least one 2-vertex shape and at least
#      one 3+-vertex shape in every grid.
#    - The dash rule uses only two rhythms tied to a human-visible quantity
#      (vertex count), which keeps the inference burden low while still making the
#      output visibly different from the solid knight-path variant.
# c) Implicit-assumption audit against the original solver:
#    - `_aligned_8`, `_segment_cells`, and angle-sorted straight edges are invalid:
#      knight geometry is bent and must be recovered as a graph of legal knight
#      adjacencies.
#    - The original edge fill assumed each edge could be drawn independently with a
#      continuous trail. Here the dash phase depends on the whole ordered walk, so
#      drawing order and shape traversal must be reconstructed explicitly.
#    - Starting the dash counter from raw path-cell index would let forced vertex
#      paints consume skip slots and often erase the visible D4 effect on 3+-vertex
#      knight shapes. The fused rule instead advances the rhythm only on non-vertex
#      path cells.
#    - The original shadow logic used "parallel line" reasoning. Knight paths do not
#      have a single direction vector, so the combo validates shifted path cells
#      directly.
# d) Placement legality:
#    - The generator keeps the D2 graph guarantees: every shape is exactly one open
#      chain or one cycle in the knight graph; non-consecutive vertices are never
#      knight-adjacent; solid knight cells do not overlap except at shared
#      consecutive endpoints.
#    - Because output trails are dashed, placement cannot use only the visible output
#      cells. The generator maintains a solid reserve mask made from the full knight
#      paths (and full shifted shadow paths) so other objects cannot land inside the
#      skipped gaps.
# e) Visual observability:
#    - Humans can see both control signals from examples alone:
#        * geometry is unmistakably L-shaped knight motion instead of straight rays
#        * 2-vertex groups produce sparser dashed Ls than 3+-vertex groups
#    - The rule does not depend on color id, hidden ordering by palette value, or any
#      non-visual metadata.
# f) Visual cleanliness:
#    - Shadows are only accepted when the fully shifted solid knight shape is disjoint
#      from the base solid shape, which avoids unreadable overlap between bent traces.
#    - Global placement still enforces at least a 1-cell 8-neighbor buffer between
#      occupied reserves of different groups.
# g) Color / rule inference:
#    - Colors keep their original semantics: one color per base shape, optional extra
#      color for its shifted copy, optional distractor color left unchanged.
#    - The dash mode is keyed by visible vertex count, not by color. This keeps the
#      mapping learnable from 2-4 ARC examples.
# 9) New solver assumptions and how generation guarantees them:
#    - If a shadow exists, its anchor is attached to the traversal start vertex used
#      by the solver, so the same start point controls both shadow offset detection
#      and dash phase origin.
#    - Open chains are traversed from an endpoint; cycles are traversed from the
#      lexicographically smallest vertex unless a shadow marks that start explicitly.
#      The generator constructs shapes in ordered form and only accepts patches that
#      reproduce exactly under `main()`.

from __future__ import annotations

import numpy as np

from arc_utils import bbox_coords, colors_present, get_bg, get_bg as _bg_color, neighbors8, paste, random_free_location

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
KNIGHT_MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]


def _sign(x: int) -> int:
    return 0 if x == 0 else (1 if x > 0 else -1)


def _is_knight_delta(dr: int, dc: int) -> bool:
    return (abs(dr), abs(dc)) in {(1, 2), (2, 1)}


def _is_knight_pair(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return _is_knight_delta(b[0] - a[0], b[1] - a[1])


def _knight_path_cells(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]] | None:
    r0, c0 = a
    r1, c1 = b
    dr = r1 - r0
    dc = c1 - c0
    adr = abs(dr)
    adc = abs(dc)
    if (adr, adc) not in {(1, 2), (2, 1)}:
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


def _trail_pattern(vertex_count: int) -> tuple[int, int]:
    if vertex_count <= 2:
        return 1, 1
    return 2, 1


def _walk_chain(adj: list[set[int]], pts_rc: list[tuple[int, int]], start_idx: int | None) -> list[int]:
    endpoints = sorted([i for i, nbs in enumerate(adj) if len(nbs) == 1], key=lambda i: pts_rc[i])
    if len(endpoints) != 2:
        return []
    cur = start_idx if start_idx in endpoints else endpoints[0]
    order = [cur]
    prev = -1
    while True:
        nxts = [nb for nb in adj[cur] if nb != prev]
        if not nxts:
            break
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def _walk_cycle_from_neighbor(adj: list[set[int]], start: int, second: int) -> list[int]:
    order = [start, second]
    prev = start
    cur = second
    while True:
        nxts = [nb for nb in adj[cur] if nb != prev]
        if not nxts:
            return []
        nxt = nxts[0]
        if nxt == start:
            break
        order.append(nxt)
        prev, cur = cur, nxt
    return order


def _walk_cycle(adj: list[set[int]], pts_rc: list[tuple[int, int]], start_idx: int) -> list[int]:
    candidates = []
    for second in sorted(adj[start_idx], key=lambda i: pts_rc[i]):
        order = _walk_cycle_from_neighbor(adj, start_idx, second)
        if len(order) == len(adj):
            candidates.append(order)
    if not candidates:
        return []
    return min(candidates, key=lambda order: [pts_rc[i] for i in order])


def _ordered_shape_walk(
    adj: list[set[int]],
    pts_rc: list[tuple[int, int]],
    start_idx: int | None,
) -> tuple[list[int], bool]:
    deg1 = [i for i, nbs in enumerate(adj) if len(nbs) == 1]
    if deg1:
        return _walk_chain(adj, pts_rc, start_idx), False
    cycle_start = start_idx if start_idx is not None else min(range(len(pts_rc)), key=lambda i: pts_rc[i])
    return _walk_cycle(adj, pts_rc, cycle_start), True


def _ordered_edge_paths(
    order: list[int],
    pts_rc: list[tuple[int, int]],
    closed: bool,
) -> list[list[tuple[int, int]]]:
    if len(order) < 2:
        return []
    edges = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    if closed:
        edges.append((order[-1], order[0]))

    paths: list[list[tuple[int, int]]] = []
    for i, j in edges:
        cells = _knight_path_cells(pts_rc[i], pts_rc[j])
        if cells is None:
            return []
        paths.append(cells)
    return paths


def _walk_sequence(paths: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    seq: list[tuple[int, int]] = []
    for idx, cells in enumerate(paths):
        if idx == 0:
            seq.extend(cells)
        else:
            seq.extend(cells[1:])
    return seq


def _draw_dashed_walk_bg_only(
    out: np.ndarray,
    seq: list[tuple[int, int]],
    protected: set[tuple[int, int]],
    col: int,
    bg: int,
    paint_len: int,
    skip_len: int,
) -> None:
    cycle = paint_len + skip_len
    interior_idx = 0
    for rr, cc in seq:
        if (rr, cc) in protected:
            if out[rr, cc] == bg:
                out[rr, cc] = col
            continue
        if (interior_idx % cycle) < paint_len and out[rr, cc] == bg:
            out[rr, cc] = col
        interior_idx += 1


def _paint_cells(out: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        out[r, c] = color


"""
Overview:
Reconstruct each same-color group as one knight-graph path or cycle, then paint its
L-shaped walk with a dashed rhythm tied to the number of visible vertices.

Input structure:
- Background 0.
- Each shape color appears as 2+ sparse vertices that form exactly one open chain or
  one cycle under knight adjacency.
- An optional extra-color pixel sits 8-neighbor adjacent to the traversal start
  vertex of a shape; that pixel encodes one shifted shadow copy.
- Distractor colors have no valid multi-vertex knight shape and remain unchanged.

Rule:
- Connect consecutive same-color vertices by canonical knight L-paths.
- Paint the full ordered walk with a global dash phase over non-vertex path cells:
  2 vertices -> paint 1 skip 1, 3+ vertices -> paint 2 skip 1.
- Keep all original vertex cells painted.
- If a start vertex has an adjacent shadow-anchor pixel, shift the whole dashed walk
  by that offset in the shadow color whenever the shifted cells are clear.

Key insight:
The geometry is no longer straight-line alignment. The correct structure is a knight
adjacency graph, and the visible dash density is determined by how many vertices the
input group contains.
"""
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
        start_idx = min(seed_idxs, key=lambda i: pts_rc[i]) if seed_idxs else None

        carried: list[tuple[int, int, int]] = []
        if start_idx is not None:
            sr0, sc0 = pts_rc[start_idx]
            for drn, dcn in DIR8:
                rr = sr0 + drn
                cc = sc0 + dcn
                if 0 <= rr < h and 0 <= cc < w:
                    v = int(g[rr, cc])
                    if v != bg and v != col:
                        carried.append((drn, dcn, v))

        adj, _ = _shape_graph(g, pts_rc, bg)
        if not _is_single_shape_graph(adj):
            continue

        order, closed = _ordered_shape_walk(adj, pts_rc, start_idx)
        if len(order) != len(pts_rc):
            continue
        if not closed and start_idx is not None and order[0] != start_idx:
            continue

        edge_paths = _ordered_edge_paths(order, pts_rc, closed)
        if not edge_paths:
            continue

        seq = _walk_sequence(edge_paths)
        paint_len, skip_len = _trail_pattern(len(pts_rc))
        protected = set(pts_rc)
        _draw_dashed_walk_bg_only(out, seq, protected, col, bg, paint_len, skip_len)

        if start_idx is None:
            continue
        sr0, sc0 = pts_rc[start_idx]
        for drn, dcn, ncol in carried:
            shifted_seq = [(rr + drn, cc + dcn) for rr, cc in seq]
            shifted_vertices = {(r + drn, c + dcn) for r, c in pts_rc}
            if not all(0 <= rr < h and 0 <= cc < w for rr, cc in shifted_seq):
                continue
            if not _clear_shifted_path_in_original(g, shifted_seq, bg, (sr0 + drn, sc0 + dcn), ncol):
                continue
            _draw_dashed_walk_bg_only(out, shifted_seq, shifted_vertices, ncol, bg, paint_len, skip_len)

    return out


# --------------- generator ---------------

BG = 0
COLORS = [1, 2, 3, 4, 6, 7, 8]
CANVAS_H_RANGE = (15, 23)
CANVAS_W_JITTER = (-3, 4)
CANVAS_W_CLIP = (15, 22)

SHAPE_COUNT_RANGE = (2, 5)
MAX_SHADOWS_CAP = 3
DISTRACTOR_PROB = 0.45
PLACEMENT_PADDING = 1
PLACEMENT_PAD_CONNECTIVITY = 8

PAIR_WEIGHT = 0.35
CHAIN_VERTICES_RANGE = (3, 7)      # 3-6 vertices
CYCLE_VERTICES_RANGE = (4, 7)      # 4-6 vertices
SHAPE_MAX_SPAN = 10
SHAPE_BUILD_RETRIES = 180
EXTRA_KIND_WEIGHTS = (
    ("pair", 0.25),
    ("chain", 0.50),
    ("cycle", 0.25),
)
CYCLE_GRID_PROB = 0.55

DISTRACTOR_COUNT_RANGE = (1, 4)
DISTRACTOR_BBOX_RANGE = (2, 6)
DISTRACTOR_RETRIES = 60

SHAPE_PLACEMENT_RETRIES = 50
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


def _build_pair() -> tuple[list[tuple[int, int]], bool] | None:
    for _ in range(SHAPE_BUILD_RETRIES):
        vertices = _transform_coords([(0, 0), (2, 1)])
        if _ordered_shape_valid(vertices, closed=False):
            return vertices, False
    return None


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


def _make_vertices(kind: str) -> tuple[list[tuple[int, int]], bool] | None:
    if kind == "pair":
        return _build_pair()
    if kind == "chain":
        return _build_chain()
    if kind == "cycle":
        return _build_cycle()
    raise ValueError(f"unknown shape kind: {kind}")


def _start_vertex_index(vertices: list[tuple[int, int]], closed: bool) -> int:
    if closed:
        return min(range(len(vertices)), key=lambda i: vertices[i])
    endpoints = [0, len(vertices) - 1]
    return min(endpoints, key=lambda i: vertices[i])


def _shift_cells(cells: set[tuple[int, int]], offset: tuple[int, int]) -> set[tuple[int, int]]:
    dr, dc = offset
    return {(r + dr, c + dc) for r, c in cells}


def _input_shape_matches_topology(inp: np.ndarray, color: int, closed: bool) -> bool:
    pts = [tuple(map(int, p)) for p in np.argwhere(inp == color)]
    if len(pts) < 2:
        return False
    adj, _ = _shape_graph(inp, pts, BG)
    if not _is_single_shape_graph(adj):
        return False
    deg1 = sum(len(nbs) == 1 for nbs in adj)
    return deg1 == 0 if closed else deg1 == 2


def _choose_shadow(
    vertices: list[tuple[int, int]],
    closed: bool,
    shape_cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    anchor = vertices[_start_vertex_index(vertices, closed)]
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
    vertices: list[tuple[int, int]],
    shadow_anchor: tuple[int, int] | None,
) -> tuple[int, int, int, int]:
    coords = list(shape_cells) + list(vertices)
    if shadow_offset is not None:
        coords.extend(_shift_cells(shape_cells, shadow_offset))
    if shadow_anchor is not None:
        coords.append(shadow_anchor)
    r0, c0, r1, c1 = bbox_coords(np.array(coords, dtype=int))
    return int(r0), int(c0), int(r1), int(c1)


def _shape_kinds(shape_count: int) -> list[str]:
    if shape_count < 2:
        raise ValueError("shape_count must be at least 2")

    require_cycle = bool(np.random.rand() < CYCLE_GRID_PROB)
    kinds = ["pair", "cycle" if require_cycle and shape_count == 2 else "chain"]
    while len(kinds) < shape_count:
        if require_cycle and "cycle" not in kinds and (shape_count - len(kinds)) == 1:
            kinds.append("cycle")
            continue
        roll = float(np.random.rand())
        acc = 0.0
        chosen = EXTRA_KIND_WEIGHTS[-1][0]
        for name, weight in EXTRA_KIND_WEIGHTS:
            acc += weight
            if roll < acc:
                chosen = name
                break
        kinds.append(chosen)
    np.random.shuffle(kinds)
    return kinds


def _make_shape_patch(
    shape_color: int,
    shadow_color: int | None,
    *,
    kind: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]] | None:
    spec = _make_vertices(kind)
    if spec is None:
        return None
    vertices, closed = spec
    shape_cells = _shape_cells(vertices, closed)
    if not shape_cells:
        return None

    shadow = None
    if shadow_color is not None:
        shadow = _choose_shadow(vertices, closed, shape_cells)
        if shadow is None:
            return None

    shadow_offset = shadow[1] if shadow is not None else None
    shadow_anchor = (shadow[0][0] + shadow[1][0], shadow[0][1] + shadow[1][1]) if shadow is not None else None
    r0, c0, r1, c1 = _patch_bbox(shape_cells, shadow_offset, vertices, shadow_anchor)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    inp = np.full((h, w), BG, dtype=int)
    reserve = np.full((h, w), BG, dtype=int)

    shifted_vertices = [(r - r0, c - c0) for r, c in vertices]
    shifted_shape_cells = {(r - r0, c - c0) for r, c in shape_cells}
    for r, c in shifted_vertices:
        inp[r, c] = shape_color
    _paint_cells(reserve, shifted_shape_cells, shape_color)

    if shadow is not None and shadow_color is not None:
        anchor, offset = shadow
        shadow_cells = _shift_cells(shifted_shape_cells, offset)
        anchor_local = (anchor[0] - r0, anchor[1] - c0)
        shadow_anchor_local = (anchor_local[0] + offset[0], anchor_local[1] + offset[1])
        if not (0 <= shadow_anchor_local[0] < h and 0 <= shadow_anchor_local[1] < w):
            return None
        if inp[shadow_anchor_local] != BG:
            return None
        if sum(
            max(abs(shadow_anchor_local[0] - vr), abs(shadow_anchor_local[1] - vc)) == 1
            for vr, vc in shifted_vertices
        ) != 1:
            return None
        inp[shadow_anchor_local] = int(shadow_color)
        _paint_cells(reserve, shadow_cells, int(shadow_color))

    if not _input_shape_matches_topology(inp, shape_color, closed):
        return None
    out = main(inp)
    if np.array_equal(out, inp):
        return None
    if np.array_equal(out, reserve):
        return None
    return inp, out, reserve, {"kind": 0 if kind == "pair" else 1, "closed": int(closed), "vertices": len(vertices)}


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


def _make_distractor_patch(color: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(np.random.randint(*DISTRACTOR_COUNT_RANGE))
    if count == 1:
        patch = np.full((1, 1), color, dtype=int)
        return patch.copy(), patch.copy(), patch.copy()
    for _ in range(DISTRACTOR_RETRIES):
        h = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        w = int(np.random.randint(*DISTRACTOR_BBOX_RANGE))
        coords = _sample_distractor_coords(h, w, count)
        if coords is None:
            continue
        patch = np.full((h, w), BG, dtype=int)
        for r, c in coords:
            patch[r, c] = color
        return patch.copy(), patch.copy(), patch.copy()
    patch = np.full((1, 1), color, dtype=int)
    return patch.copy(), patch.copy(), patch.copy()


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


def _try_place(
    inp: np.ndarray,
    out: np.ndarray,
    occ: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
    reserve: np.ndarray,
) -> bool:
    try:
        loc = random_free_location(
            occ,
            reserve,
            transparent=BG,
            padding=PLACEMENT_PADDING,
            padding_connectivity=PLACEMENT_PAD_CONNECTIVITY,
        )
    except ValueError:
        return False
    paste(occ, reserve, loc[0], loc[1], transparent=BG)
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
    occ = np.full((h, w), BG, dtype=int)

    has_distractor = bool(np.random.rand() < DISTRACTOR_PROB)
    shape_count = int(np.random.randint(*SHAPE_COUNT_RANGE))
    palette = _sample_palette(shape_count, has_distractor)
    if palette is None:
        return None
    shape_colors, shadow_colors, distractor_color = palette
    shadow_map = _assign_shadows(shape_count, shadow_colors)
    kinds = _shape_kinds(shape_count)

    total_colors = shape_count + len(shadow_colors) + int(has_distractor)
    kind_seen = set()

    for idx, shape_color in enumerate(shape_colors):
        placed = False
        for _ in range(SHAPE_PLACEMENT_RETRIES):
            patch = _make_shape_patch(int(shape_color), shadow_map.get(idx), kind=kinds[idx])
            if patch is None:
                continue
            patch_in, patch_out, reserve, meta = patch
            if _try_place(inp, out, occ, patch_in, patch_out, reserve):
                kind_seen.add(int(meta["kind"]))
                placed = True
                break
        if not placed:
            return None

    if has_distractor and distractor_color is not None:
        if not _try_place(inp, out, occ, *_make_distractor_patch(distractor_color)):
            return None

    if kind_seen != {0, 1}:
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
