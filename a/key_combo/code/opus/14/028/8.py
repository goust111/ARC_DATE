from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
DIR_BY_COLOR = {
    3: (0, -1),   # left
    4: (1, 0),    # down
    6: (-1, 0),   # up
    8: (0, 1),    # right
}
DIR_COLORS = tuple(DIR_BY_COLOR)
TREE_GAP_CHOICES = (1, 2, 3, 4, 5, 6)
GRID_MIN = 18
GRID_MAX = 30
GEN_ATTEMPTS = 500
MARKER_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _interior_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = rect
    return [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]


def _point_in_rect_interior(pos: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = pos
    r0, c0, r1, c1 = rect
    return r0 < r < r1 and c0 < c < c1


def _rectangles_and_markers(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[tuple[int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    rects: list[tuple[int, int, int, int]] = []
    markers: list[tuple[int, int, int]] = []

    for comp in _color_components(g, WALL):
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 4, "expected hollow rectangles"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        frame = np.zeros((h, w), dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        assert np.array_equal(patch == WALL, frame), "grey component must be a 1-pixel hollow rectangle"

        inside_markers = []
        for rr in range(r0 + 1, r1):
            for cc in range(c0 + 1, c1):
                v = int(g[rr, cc])
                assert v == BG or v in DIR_BY_COLOR, "rectangle interiors may only contain background or one marker"
                if v in DIR_BY_COLOR:
                    inside_markers.append((rr, cc, v))
        assert len(inside_markers) <= 1, "each rectangle may contain at most one marker"
        rects.append((r0, c0, r1, c1))
        markers.extend(inside_markers)

    rects.sort()
    markers.sort()
    assert 3 <= len(rects) <= 5, "expected 3-5 rectangles"
    assert len(markers) >= 2, "expected at least two markers"
    return rects, markers


def _candidate_edges(grid: np.ndarray, rects: list[tuple[int, int, int, int]]) -> list[dict]:
    edges: list[dict] = []
    for i, a in enumerate(rects):
        for j in range(i + 1, len(rects)):
            b = rects[j]

            if a[3] < b[1] or b[3] < a[1]:
                if a[3] < b[1]:
                    left_i, right_i, left, right = i, j, a, b
                else:
                    left_i, right_i, left, right = j, i, b, a
                r0 = max(left[0] + 1, right[0] + 1)
                r1 = min(left[2] - 1, right[2] - 1)
                c0 = left[3] + 1
                c1 = right[1] - 1
                if r0 <= r1 and c0 <= c1 and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG):
                    edges.append(
                        {
                            "kind": "h",
                            "a": left_i,
                            "b": right_i,
                            "gap": c1 - c0 + 1,
                            "r0": r0,
                            "r1": r1,
                            "c0": c0,
                            "c1": c1,
                        }
                    )

            if a[2] < b[0] or b[2] < a[0]:
                if a[2] < b[0]:
                    upper_i, lower_i, upper, lower = i, j, a, b
                else:
                    upper_i, lower_i, upper, lower = j, i, b, a
                c0 = max(upper[1] + 1, lower[1] + 1)
                c1 = min(upper[3] - 1, lower[3] - 1)
                r0 = upper[2] + 1
                r1 = lower[0] - 1
                if c0 <= c1 and r0 <= r1 and np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == BG):
                    edges.append(
                        {
                            "kind": "v",
                            "a": upper_i,
                            "b": lower_i,
                            "gap": r1 - r0 + 1,
                            "r0": r0,
                            "r1": r1,
                            "c0": c0,
                            "c1": c1,
                        }
                    )
    edges.sort(key=lambda e: (e["gap"], e["kind"], e["r0"], e["c0"], e["a"], e["b"]))
    return edges


def _spanning_edges(edges: list[dict], n_rects: int) -> list[dict]:
    parent = list(range(n_rects))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    chosen: list[dict] = []
    for edge in edges:
        if find(edge["a"]) != find(edge["b"]):
            chosen.append(edge)
            union(edge["a"], edge["b"])
        if len(chosen) == n_rects - 1:
            break

    assert len(chosen) == n_rects - 1, "candidate corridor graph must connect all rectangles"
    root = find(0)
    assert all(find(i) == root for i in range(n_rects))
    return chosen


def _apply_edge(
    out: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    edge: dict,
) -> None:
    r0 = edge["r0"]
    r1 = edge["r1"]
    c0 = edge["c0"]
    c1 = edge["c1"]
    if edge["kind"] == "h":
        left = rects[edge["a"]]
        right = rects[edge["b"]]
        out[r0, c0 : c1 + 1] = WALL
        out[r1, c0 : c1 + 1] = WALL
        if r0 + 1 <= r1 - 1:
            out[r0 + 1 : r1, left[3]] = BG
            out[r0 + 1 : r1, right[1]] = BG
    else:
        upper = rects[edge["a"]]
        lower = rects[edge["b"]]
        out[r0 : r1 + 1, c0] = WALL
        out[r0 : r1 + 1, c1] = WALL
        if c0 + 1 <= c1 - 1:
            out[upper[2], c0 + 1 : c1] = BG
            out[lower[0], c0 + 1 : c1] = BG


def _build_network(grid: np.ndarray, rects: list[tuple[int, int, int, int]], edges: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[np.isin(out, DIR_COLORS)] = BG
    for edge in edges:
        _apply_edge(out, rects, edge)
    return out


def _edge_passable_cells(edge: dict, rects: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if edge["kind"] == "h":
        left = rects[edge["a"]]
        right = rects[edge["b"]]
        for r in range(edge["r0"] + 1, edge["r1"]):
            for c in range(left[3], right[1] + 1):
                cells.add((r, c))
    else:
        upper = rects[edge["a"]]
        lower = rects[edge["b"]]
        for r in range(upper[2], lower[0] + 1):
            for c in range(edge["c0"] + 1, edge["c1"]):
                cells.add((r, c))
    return cells


def _slide_on_grid(
    grid: np.ndarray,
    start: tuple[int, int],
    color: int,
) -> tuple[tuple[int, int], list[tuple[int, int]], str]:
    dr, dc = DIR_BY_COLOR[color]
    r, c = start
    path: list[tuple[int, int]] = []
    while True:
        nr = r + dr
        nc = c + dc
        if not _in_bounds(grid.shape, nr, nc):
            return (r, c), path, "boundary"
        if int(grid[nr, nc]) != BG:
            reason = "wall" if int(grid[nr, nc]) == WALL else "marker"
            return (r, c), path, reason
        r, c = nr, nc
        path.append((r, c))


def _slide_markers(
    network: np.ndarray,
    markers: list[tuple[int, int, int]],
    rects: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[dict]]:
    out = np.asarray(network, dtype=int).copy()
    ordered = sorted(markers)
    for r, c, color in ordered:
        out[r, c] = color

    details: list[dict] = []
    for r, c, color in ordered:
        assert int(out[r, c]) == color
        out[r, c] = BG
        final, path, reason = _slide_on_grid(out, (r, c), color)
        out[final] = color
        start_rect = next(i for i, rect in enumerate(rects) if _point_in_rect_interior((r, c), rect))
        final_rect = next((i for i, rect in enumerate(rects) if _point_in_rect_interior(final, rect)), None)
        details.append(
            {
                "start": (r, c),
                "color": color,
                "final": final,
                "path": path,
                "stop_reason": reason,
                "start_rect": start_rect,
                "final_rect": final_rect,
                "left_start_rect": any(not _point_in_rect_interior(p, rects[start_rect]) for p in path),
            }
        )
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build the MST corridor network between hollow rectangles, then slide each internal marker in its color-coded direction until blocked.

    Input structure:
    - Black background contains 3-5 grey hollow rectangular frames with one-cell-thick walls and black interiors.
    - Some rectangle interiors contain exactly one single-pixel marker: green means left, yellow means down, pink means up, teal means right.
    - Rectangles are separated by black gaps, and visible horizontal or vertical gaps may exist between overlapping rectangle interiors.

    Rule:
    - Detect all hollow grey rectangles and compute every visible horizontal or vertical gap whose full gap area is black and whose rectangle interiors overlap on the perpendicular axis.
    - Choose the minimum spanning tree of those gaps by gap length, and for each chosen gap draw the b9630600-style corridor walls while erasing the facing wall segment between the corridor boundaries so the room interiors become connected.
    - Erase each marker from its original cell, then process markers in top-to-bottom, left-to-right order: slide the marker one cell at a time in the direction encoded by its color through black passable cells of the connected room-and-corridor space, stopping before a wall, the grid boundary, or an earlier settled marker.

    Key insight:
    - The final position depends on two coupled steps: the MST decides which corridors actually exist, and each marker then moves only along the resulting connected black walkway in its own encoded direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    rects, markers = _rectangles_and_markers(g)
    edges = _candidate_edges(g, rects)
    chosen = _spanning_edges(edges, len(rects))
    network = _build_network(g, rects, chosen)
    out, _details = _slide_markers(network, markers, rects)
    return out


def _random_tree(n_rooms: int) -> list[int]:
    parents = [-1]
    for node in range(1, n_rooms):
        parents.append(random.randrange(node))
    return parents


def _rect_clear(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        if not (r1 + 1 < or0 or or1 + 1 < r0 or c1 + 1 < oc0 or oc1 + 1 < c0):
            return False
    return True


def _child_candidates(
    parent_rect: tuple[int, int, int, int],
    child_size: tuple[int, int],
    orientation: str,
    gap: int,
    existing: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    pr0, pc0, pr1, pc1 = parent_rect
    ch, cw = child_size
    pi_h = pr1 - pr0 - 1
    pi_w = pc1 - pc0 - 1
    ci_h = ch - 2
    ci_w = cw - 2
    candidates: set[tuple[int, int, int, int]] = set()

    if orientation == "h":
        max_overlap = min(pi_h, ci_h)
        if max_overlap < 3:
            return []
        for _ in range(140):
            side = random.choice((-1, 1))
            overlap = random.randint(3, max_overlap)
            start_parent = random.randint(pr0 + 1, pr1 - overlap)
            child_inner_offset = random.randint(0, ci_h - overlap)
            r0 = start_parent - 1 - child_inner_offset
            c0 = pc1 + gap + 1 if side > 0 else pc0 - gap - cw
            rect = (r0, c0, r0 + ch - 1, c0 + cw - 1)
            if _rect_clear(rect, existing):
                candidates.add(rect)
    else:
        max_overlap = min(pi_w, ci_w)
        if max_overlap < 3:
            return []
        for _ in range(140):
            side = random.choice((-1, 1))
            overlap = random.randint(3, max_overlap)
            start_parent = random.randint(pc0 + 1, pc1 - overlap)
            child_inner_offset = random.randint(0, ci_w - overlap)
            c0 = start_parent - 1 - child_inner_offset
            r0 = pr1 + gap + 1 if side > 0 else pr0 - gap - ch
            rect = (r0, c0, r0 + ch - 1, c0 + cw - 1)
            if _rect_clear(rect, existing):
                candidates.add(rect)

    out = list(candidates)
    random.shuffle(out)
    return out


def _child_candidates_fixed_side(
    parent_rect: tuple[int, int, int, int],
    child_size: tuple[int, int],
    orientation: str,
    side: int,
    gap: int,
    existing: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    pr0, pc0, pr1, pc1 = parent_rect
    ch, cw = child_size
    pi_h = pr1 - pr0 - 1
    pi_w = pc1 - pc0 - 1
    ci_h = ch - 2
    ci_w = cw - 2
    candidates: set[tuple[int, int, int, int]] = set()

    if orientation == "h":
        max_overlap = min(pi_h, ci_h)
        if max_overlap < 3:
            return []
        for _ in range(180):
            overlap = random.randint(3, max_overlap)
            start_parent = random.randint(pr0 + 1, pr1 - overlap)
            child_inner_offset = random.randint(0, ci_h - overlap)
            r0 = start_parent - 1 - child_inner_offset
            c0 = pc1 + gap + 1 if side > 0 else pc0 - gap - cw
            rect = (r0, c0, r0 + ch - 1, c0 + cw - 1)
            if _rect_clear(rect, existing):
                candidates.add(rect)
    else:
        max_overlap = min(pi_w, ci_w)
        if max_overlap < 3:
            return []
        for _ in range(180):
            overlap = random.randint(3, max_overlap)
            start_parent = random.randint(pc0 + 1, pc1 - overlap)
            child_inner_offset = random.randint(0, ci_w - overlap)
            c0 = start_parent - 1 - child_inner_offset
            r0 = pr1 + gap + 1 if side > 0 else pr0 - gap - ch
            rect = (r0, c0, r0 + ch - 1, c0 + cw - 1)
            if _rect_clear(rect, existing):
                candidates.add(rect)

    out = list(candidates)
    random.shuffle(out)
    return out


def _place_tree(
    sizes: list[tuple[int, int]],
    parents: list[int],
    orientations: list[str],
    gaps: list[int],
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int] | None] = [None] * len(sizes)
    h0, w0 = sizes[0]
    rects[0] = (0, 0, h0 - 1, w0 - 1)

    order = list(range(1, len(sizes)))

    def search(idx: int) -> bool:
        if idx == len(order):
            return True
        node = order[idx]
        parent = parents[node]
        assert rects[parent] is not None
        existing = [r for i, r in enumerate(rects) if i != node and r is not None]
        candidates = _child_candidates(rects[parent], sizes[node], orientations[node], gaps[node], existing)
        for rect in candidates[:70]:
            rects[node] = rect
            if search(idx + 1):
                return True
            rects[node] = None
        return False

    if not search(0):
        return None
    return [r for r in rects if r is not None]


def _shift_rects(rects: list[tuple[int, int, int, int]]) -> tuple[list[tuple[int, int, int, int]], tuple[int, int]] | None:
    min_r = min(r0 for r0, _, _, _ in rects)
    min_c = min(c0 for _, c0, _, _ in rects)
    max_r = max(r1 for _, _, r1, _ in rects)
    max_c = max(c1 for _, _, _, c1 in rects)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    min_h = max(GRID_MIN, span_h + 2)
    min_w = max(GRID_MIN, span_w + 2)
    max_h = min(GRID_MAX, span_h + 8)
    max_w = min(GRID_MAX, span_w + 8)
    if min_h > max_h or min_w > max_w:
        return None

    h = random.randint(min_h, max_h)
    w = random.randint(min_w, max_w)
    sr_lo = 1 - min_r
    sr_hi = h - 2 - max_r
    sc_lo = 1 - min_c
    sc_hi = w - 2 - max_c
    if sr_lo > sr_hi or sc_lo > sc_hi:
        return None

    shift_r = random.randint(sr_lo, sr_hi)
    shift_c = random.randint(sc_lo, sc_hi)
    shifted = [(r0 + shift_r, c0 + shift_c, r1 + shift_r, c1 + shift_c) for r0, c0, r1, c1 in rects]
    return shifted, (h, w)


def _temp_grid_from_rects(rects: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    min_r = min(r0 for r0, _, _, _ in rects)
    min_c = min(c0 for _, c0, _, _ in rects)
    shifted = [(r0 - min_r + 1, c0 - min_c + 1, r1 - min_r + 1, c1 - min_c + 1) for r0, c0, r1, c1 in rects]
    max_r = max(r1 for _, _, r1, _ in shifted)
    max_c = max(c1 for _, _, _, c1 in shifted)
    grid = _draw_frames((max_r + 2, max_c + 2), shifted)
    return grid, shifted


def _draw_frames(shape: tuple[int, int], rects: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0, c0 : c1 + 1] = WALL
        grid[r1, c0 : c1 + 1] = WALL
        grid[r0 : r1 + 1, c0] = WALL
        grid[r0 : r1 + 1, c1] = WALL
    return grid


def _reachable_passable_component(network: np.ndarray, rects: list[tuple[int, int, int, int]]) -> set[tuple[int, int]]:
    start = (rects[0][0] + 1, rects[0][1] + 1)
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(network.shape, nr, nc) and int(network[nr, nc]) == BG and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return seen


def _all_rect_interiors_connected(network: np.ndarray, rects: list[tuple[int, int, int, int]]) -> bool:
    seen = _reachable_passable_component(network, rects)
    for rect in rects:
        for cell in _interior_cells(rect):
            if cell not in seen:
                return False
    return True


def _describe_candidate(
    rects: list[tuple[int, int, int, int]],
    corridor_cells: set[tuple[int, int]],
    start: tuple[int, int],
    color: int,
    final: tuple[int, int],
    path: list[tuple[int, int]],
    reason: str,
) -> dict:
    rect_idx = next(i for i, rect in enumerate(rects) if _point_in_rect_interior(start, rect))
    final_rect_idx = next((i for i, rect in enumerate(rects) if _point_in_rect_interior(final, rect)), None)
    return {
        "start": start,
        "color": color,
        "final": final,
        "path": path,
        "reason": reason,
        "rect_idx": rect_idx,
        "final_rect_idx": final_rect_idx,
        "uses_corridor": any(cell in corridor_cells for cell in path),
        "leaves_room": any(not _point_in_rect_interior(cell, rects[rect_idx]) for cell in path),
    }


def _build_candidate_pool(
    network_mst: np.ndarray,
    network_all: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    chosen_edges: list[dict],
) -> list[dict]:
    mst_corridor_cells = set()
    for edge in chosen_edges:
        mst_corridor_cells |= _edge_passable_cells(edge, rects)

    pool: list[dict] = []
    for rect in rects:
        for start in _interior_cells(rect):
            for color in DIR_COLORS:
                final_mst, path_mst, reason_mst = _slide_on_grid(network_mst, start, color)
                final_all, path_all, reason_all = _slide_on_grid(network_all, start, color)
                cand = _describe_candidate(rects, mst_corridor_cells, start, color, final_mst, path_mst, reason_mst)
                cand["all_final"] = final_all
                cand["all_path"] = path_all
                cand["all_reason"] = reason_all
                cand["sensitive_to_extra_edge"] = final_all != final_mst or path_all != path_mst
                pool.append(cand)
    return pool


def _independent_finals(markers: list[tuple[int, int, int]], pool_by_key: dict[tuple[int, int, int], dict]) -> dict[tuple[int, int], tuple[int, int]]:
    return {(r, c): pool_by_key[(r, c, color)]["final"] for r, c, color in markers}


def _candidate_subset(cands: list[dict]) -> list[dict]:
    groups = [
        [c for c in cands if c["sensitive_to_extra_edge"] and c["uses_corridor"]],
        [c for c in cands if c["sensitive_to_extra_edge"] and not c["uses_corridor"]],
        [c for c in cands if c["uses_corridor"] and not c["sensitive_to_extra_edge"]],
        [c for c in cands if len(c["path"]) == 0],
        [c for c in cands if c["leaves_room"]],
        cands,
    ]
    out: list[dict] = []
    seen: set[tuple[int, int, int]] = set()
    for group in groups:
        items = group[:]
        random.shuffle(items)
        for cand in items[:3]:
            key = (cand["start"][0], cand["start"][1], cand["color"])
            if key not in seen:
                seen.add(key)
                out.append(cand)
    return out


def _choose_markers(
    base_grid: np.ndarray,
    rects: list[tuple[int, int, int, int]],
    candidate_edges: list[dict],
    chosen_edges: list[dict],
) -> list[tuple[int, int, int]] | None:
    network_mst = _build_network(base_grid, rects, chosen_edges)
    network_all = _build_network(base_grid, rects, candidate_edges)
    pool = _build_candidate_pool(network_mst, network_all, rects, chosen_edges)
    by_rect: dict[int, list[dict]] = {}
    for cand in pool:
        by_rect.setdefault(cand["rect_idx"], []).append(cand)

    sensitive = [cand for cand in pool if cand["sensitive_to_extra_edge"]]
    corridor = [cand for cand in pool if cand["uses_corridor"]]
    if not sensitive or not corridor:
        return None

    pool_by_key = {(cand["start"][0], cand["start"][1], cand["color"]): cand for cand in pool}
    require_marker_block = random.random() < 0.45
    reduced = {idx: _candidate_subset(cands) for idx, cands in by_rect.items()}
    corridor_cells = set()
    for edge in chosen_edges:
        corridor_cells |= _edge_passable_cells(edge, rects)

    rect_ids = list(range(len(rects)))
    random.shuffle(rect_ids)

    for need_marker_block in ([True, False] if require_marker_block else [False]):
        k_values = [3, 2] if need_marker_block else [2, 3]
        for k in k_values:
            if k > len(rect_ids):
                continue
            rect_subsets = list(itertools.combinations(rect_ids, k))
            random.shuffle(rect_subsets)
            for rect_subset in rect_subsets:
                candidate_lists = [reduced[idx] for idx in rect_subset]
                combos = list(itertools.product(*candidate_lists))
                random.shuffle(combos)
                for chosen in combos:
                    starts = [cand["start"] for cand in chosen]
                    if len(set(starts)) != len(starts):
                        continue
                    if len({cand["color"] for cand in chosen}) < 2:
                        continue
                    start_set = set(starts)
                    if any(any(cell in start_set - {cand["start"]} for cell in cand["path"]) for cand in chosen):
                        continue

                    markers = [(cand["start"][0], cand["start"][1], cand["color"]) for cand in chosen]
                    inp = base_grid.copy()
                    for r, c, color in markers:
                        inp[r, c] = color

                    out_mst, details = _slide_markers(network_mst, markers, rects)
                    out_all, _ = _slide_markers(network_all, markers, rects)
                    if np.array_equal(out_mst, out_all):
                        continue
                    if not any(detail["left_start_rect"] for detail in details):
                        continue
                    if not any(any(cell in corridor_cells for cell in detail["path"]) for detail in details):
                        continue
                    finals = [detail["final"] for detail in details]
                    if len(set(finals)) != len(finals):
                        continue
                    if not any(len(detail["path"]) > 0 for detail in details):
                        continue

                    independent = _independent_finals(markers, pool_by_key)
                    has_marker_block = any(detail["final"] != independent[detail["start"]] for detail in details)
                    if need_marker_block and not has_marker_block:
                        continue

                    if np.array_equal(main(inp), out_mst):
                        return markers
    return None


def _generate_layout() -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[dict], list[dict]] | None:
    n_rooms = random.choices([3, 4, 5], weights=[1, 4, 3], k=1)[0]
    sizes = [(random.randint(5, 7), random.randint(5, 7)) for _ in range(n_rooms)]
    h_side = random.choice((-1, 1))
    v_side = random.choice((-1, 1))
    gap_perm = list(TREE_GAP_CHOICES)
    random.shuffle(gap_perm)
    gaps = gap_perm[: max(4, n_rooms)]

    rect0 = (0, 0, sizes[0][0] - 1, sizes[0][1] - 1)
    rect1_candidates = _child_candidates_fixed_side(rect0, sizes[1], "h", h_side, gaps[0], [rect0])
    rect2_candidates = _child_candidates_fixed_side(rect0, sizes[2], "v", v_side, gaps[1], [rect0])
    if not rect1_candidates or not rect2_candidates:
        return None

    rects_unshifted: list[tuple[int, int, int, int]] | None = None
    for rect1 in rect1_candidates[:50]:
        for rect2 in rect2_candidates[:50]:
            if not _rect_clear(rect2, [rect0, rect1]):
                continue
            placed = [rect0, rect1, rect2]

            if n_rooms == 3:
                temp_grid, temp_rects = _temp_grid_from_rects(placed)
                edge_keys = {(edge["a"], edge["b"], edge["kind"]) for edge in _candidate_edges(temp_grid, temp_rects)}
                if len(edge_keys) >= 3:
                    rects_unshifted = placed
                    break
            else:
                rect3_parent = rect1
                rect3_candidates = _child_candidates_fixed_side(rect3_parent, sizes[3], "v", v_side, gaps[2], placed)
                for rect3 in rect3_candidates[:70]:
                    placed4 = placed + [rect3]
                    temp_grid, temp_rects = _temp_grid_from_rects(placed4)
                    temp_edges = _candidate_edges(temp_grid, temp_rects)
                    if len(temp_edges) < 4:
                        continue
                    edge_pairs = {(edge["a"], edge["b"]) for edge in temp_edges}
                    if not {(0, 1), (0, 2), (1, 3), (2, 3)} <= edge_pairs:
                        continue
                    if n_rooms == 4:
                        rects_unshifted = placed4
                        break

                    room4_orientation = random.choice(["h", "v"])
                    room4_side = random.choice((-1, 1))
                    room4_parent_idx = random.choice([2, 3])
                    room4_parent = placed4[room4_parent_idx]
                    rect4_candidates = _child_candidates_fixed_side(
                        room4_parent,
                        sizes[4],
                        room4_orientation,
                        room4_side,
                        gaps[3],
                        placed4,
                    )
                    for rect4 in rect4_candidates[:50]:
                        rects_unshifted = placed4 + [rect4]
                        break
                    if rects_unshifted is not None:
                        break
                if rects_unshifted is not None:
                    break
        if rects_unshifted is not None:
            break

    if rects_unshifted is None:
        return None
    shifted = _shift_rects(rects_unshifted)
    if shifted is None:
        return None
    rects, shape = shifted
    base_grid = _draw_frames(shape, rects)

    candidate_edges = _candidate_edges(base_grid, rects)
    if len(candidate_edges) < n_rooms:
        return None
    if len({edge["gap"] for edge in candidate_edges}) != len(candidate_edges):
        return None

    chosen_edges = _spanning_edges(candidate_edges, n_rooms)
    network = _build_network(base_grid, rects, chosen_edges)
    if not _all_rect_interiors_connected(network, rects):
        return None

    extra_edges = [edge for edge in candidate_edges if edge not in chosen_edges]
    if not extra_edges:
        return None
    return base_grid, rects, candidate_edges, chosen_edges


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        layout = _generate_layout()
        if layout is None:
            continue
        base_grid, rects, candidate_edges, chosen_edges = layout
        markers = _choose_markers(base_grid, rects, candidate_edges, chosen_edges)
        if markers is None:
            continue
        inp = base_grid.copy()
        for r, c, color in markers:
            inp[r, c] = color
        out = main(inp)
        assert inp.ndim == 2 and out.ndim == 2
        assert max(inp.shape) <= 30 and max(out.shape) <= 30
        return inp
    raise RuntimeError("failed to generate a valid corridor-slide puzzle")
