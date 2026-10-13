from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

Rect = tuple[int, int, int, int]


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                nbrs = _neighbors8(cr, cc) if conn8 else _neighbors4(cr, cc)
                for nr, nc in nbrs:
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> Rect:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _draw_frame(grid: np.ndarray, rect: Rect) -> None:
    r0, c0, r1, c1 = rect
    grid[r0 : r1 + 1, c0] = GREEN
    grid[r0 : r1 + 1, c1] = GREEN
    grid[r0, c0 : c1 + 1] = GREEN
    grid[r1, c0 : c1 + 1] = GREEN


def _rectangles(grid: np.ndarray) -> list[Rect]:
    g = np.asarray(grid, dtype=int)
    present = set(map(int, np.unique(g)))
    assert present <= {BLACK, GREEN}
    comps = _components(g == GREEN, conn8=False)
    assert 3 <= len(comps) <= 5
    rects: list[Rect] = []
    for comp in comps:
        rect = _bbox(comp)
        r0, c0, r1, c1 = rect
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5
        frame = np.zeros((h, w), dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.array_equal(patch == GREEN, frame)
        rects.append(rect)
    rects.sort()
    return rects


def _candidate_edges(grid: np.ndarray, rects: list[Rect]) -> list[dict]:
    g = np.asarray(grid, dtype=int)
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
                if r0 <= r1 and c0 <= c1 and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == BLACK):
                    edges.append(
                        {
                            "gap": c1 - c0 + 1,
                            "kind": "h",
                            "a": left_i,
                            "b": right_i,
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
                if c0 <= c1 and r0 <= r1 and np.all(g[r0 : r1 + 1, c0 : c1 + 1] == BLACK):
                    edges.append(
                        {
                            "gap": r1 - r0 + 1,
                            "kind": "v",
                            "a": upper_i,
                            "b": lower_i,
                            "r0": r0,
                            "r1": r1,
                            "c0": c0,
                            "c1": c1,
                        }
                    )
    return edges


def _spanning_edges(edges: list[dict], n_rects: int) -> list[dict]:
    parent = list(range(n_rects))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            return False
        parent[rb] = ra
        return True

    chosen: list[dict] = []
    order = sorted(edges, key=lambda e: (e["gap"], e["kind"], e["a"], e["b"], e["r0"], e["c0"]))
    for edge in order:
        if union(edge["a"], edge["b"]):
            chosen.append(edge)
            if len(chosen) == n_rects - 1:
                break
    assert len(chosen) == n_rects - 1
    root = find(0)
    assert all(find(i) == root for i in range(n_rects))
    return chosen


def _apply_edge(out: np.ndarray, edge: dict, rects: list[Rect]) -> None:
    r0 = int(edge["r0"])
    r1 = int(edge["r1"])
    c0 = int(edge["c0"])
    c1 = int(edge["c1"])
    if edge["kind"] == "h":
        left = rects[edge["a"]]
        right = rects[edge["b"]]
        out[r0, c0 : c1 + 1] = GREEN
        out[r1, c0 : c1 + 1] = GREEN
        if r0 + 1 <= r1 - 1:
            out[r0 + 1 : r1, left[3]] = BLACK
            out[r0 + 1 : r1, right[1]] = BLACK
    else:
        upper = rects[edge["a"]]
        lower = rects[edge["b"]]
        out[r0 : r1 + 1, c0] = GREEN
        out[r0 : r1 + 1, c1] = GREEN
        if c0 + 1 <= c1 - 1:
            out[upper[2], c0 + 1 : c1] = BLACK
            out[lower[0], c0 + 1 : c1] = BLACK


def _enclosed_black_mask(grid: np.ndarray) -> np.ndarray:
    black = grid == BLACK
    enclosed = np.zeros_like(black, dtype=bool)
    for comp in _components(black, conn8=False):
        if any(r == 0 or c == 0 or r == grid.shape[0] - 1 or c == grid.shape[1] - 1 for r, c in comp):
            continue
        for r, c in comp:
            enclosed[r, c] = True
    return enclosed


def _distance_field(grid: np.ndarray, enclosed: np.ndarray) -> np.ndarray:
    dist = np.full(grid.shape, -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in np.argwhere(enclosed):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == GREEN:
                dist[rr, cc] = 1
                q.append((rr, cc))
                break
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors8(r, c):
            if _in_bounds(grid.shape, nr, nc) and enclosed[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    assert np.all(dist[enclosed] >= 1)
    return dist


def _color_for_distance(dist: int) -> int:
    return (RED, BLUE, YELLOW)[(dist - 1) % 3]


def main(grid: np.ndarray) -> np.ndarray:
    """Connect green hollow rectangles by a minimum-gap corridor tree, then fill the unified enclosed interior with cyclic distance colors.

    Input structure:
    - The grid has black background and 3-5 disjoint green hollow rectangular frames with 1-pixel-thick walls.
    - Some frame pairs have a clear horizontal or vertical black gap whose perpendicular overlap lies inside both frame interiors.
    - The visible-gap graph is connected, so the frames can be joined into one corridor network.

    Rule:
    - Detect every visible horizontal or vertical gap between two frames and build the minimum spanning tree using gap length as edge weight.
    - For each chosen edge, extend the two boundary rows or columns of the interior overlap across the gap as green corridor walls, and erase the facing wall pixels strictly between those two boundaries to open the passage.
    - After all corridors are opened, treat every black pixel enclosed by the unified wall network as interior space and compute its shortest king-move distance to any green wall while staying inside that enclosed topology.
    - Recolor enclosed distance 1,2,3,... with the repeating cycle red, blue, yellow; keep all green wall pixels and the exterior black background unchanged.

    Key insight:
    - The fill is not room-by-room: corridor openings merge all rooms into one enclosed topology, so the distance field must propagate globally through the opened corridors before the cyclic coloring is applied.
    """
    g = np.asarray(grid, dtype=int)
    rects = _rectangles(g)
    edges = _candidate_edges(g, rects)
    chosen = _spanning_edges(edges, len(rects))

    out = g.copy()
    for edge in chosen:
        _apply_edge(out, edge, rects)

    enclosed = _enclosed_black_mask(out)
    dist = _distance_field(out, enclosed)
    for r, c in np.argwhere(enclosed):
        rr = int(r)
        cc = int(c)
        out[rr, cc] = _color_for_distance(int(dist[rr, cc]))
    return out


def _row_gap(a: Rect, b: Rect) -> int:
    if a[2] < b[0]:
        return b[0] - a[2] - 1
    if b[2] < a[0]:
        return a[0] - b[2] - 1
    return 0


def _col_gap(a: Rect, b: Rect) -> int:
    if a[3] < b[1]:
        return b[1] - a[3] - 1
    if b[3] < a[1]:
        return a[1] - b[3] - 1
    return 0


def _boxes_touch(a: Rect, b: Rect) -> bool:
    return _row_gap(a, b) == 0 and _col_gap(a, b) == 0


def _random_room_size(big: bool) -> tuple[int, int]:
    if big:
        return random.randint(9, 11), random.randint(9, 11)
    return random.randint(5, 9), random.randint(5, 9)


def _build_cycle_rects(h: int, w: int, count: int) -> list[Rect] | None:
    assert 3 <= count <= 4
    for _ in range(300):
        tl_h, tl_w = _random_room_size(big=True)
        tr_h, tr_w = _random_room_size(big=False)
        bl_h, bl_w = _random_room_size(big=False)
        br_h, br_w = _random_room_size(big=False)

        max_tl_r = max(1, h - (tl_h + tr_h + bl_h + br_h) // 2 - 6)
        max_tl_c = max(1, w - (tl_w + tr_w + bl_w + br_w) // 2 - 6)
        tl_r0 = random.randint(1, min(6, max_tl_r))
        tl_c0 = random.randint(1, min(6, max_tl_c))

        top_gap = random.randint(1, 5)
        left_gap = random.randint(1, 5)

        valid_delta_top = []
        for d in range(-4, 5):
            tr_r0 = tl_r0 + d
            overlap = min(tl_r0 + tl_h - 2, tr_r0 + tr_h - 2) - max(tl_r0 + 1, tr_r0 + 1) + 1
            if overlap >= 3:
                valid_delta_top.append(d)
        valid_delta_left = []
        for d in range(-4, 5):
            bl_c0 = tl_c0 + d
            overlap = min(tl_c0 + tl_w - 2, bl_c0 + bl_w - 2) - max(tl_c0 + 1, bl_c0 + 1) + 1
            if overlap >= 3:
                valid_delta_left.append(d)
        if not valid_delta_top or not valid_delta_left:
            continue

        tr_r0 = tl_r0 + random.choice(valid_delta_top)
        tr_c0 = tl_c0 + tl_w + top_gap + 1
        bl_r0 = tl_r0 + tl_h + left_gap + 1
        bl_c0 = tl_c0 + random.choice(valid_delta_left)

        valid_delta_bottom = []
        for d in range(-4, 5):
            br_r0 = bl_r0 + d
            overlap = min(bl_r0 + bl_h - 2, br_r0 + br_h - 2) - max(bl_r0 + 1, br_r0 + 1) + 1
            if overlap >= 3:
                valid_delta_bottom.append(d)
        valid_delta_right = []
        for d in range(-4, 5):
            br_c0 = tr_c0 + d
            overlap = min(tr_c0 + tr_w - 2, br_c0 + br_w - 2) - max(tr_c0 + 1, br_c0 + 1) + 1
            if overlap >= 3:
                valid_delta_right.append(d)
        if not valid_delta_bottom or not valid_delta_right:
            continue

        br_r0 = bl_r0 + random.choice(valid_delta_bottom)
        br_c0 = tr_c0 + random.choice(valid_delta_right)

        tl = (tl_r0, tl_c0, tl_r0 + tl_h - 1, tl_c0 + tl_w - 1)
        tr = (tr_r0, tr_c0, tr_r0 + tr_h - 1, tr_c0 + tr_w - 1)
        bl = (bl_r0, bl_c0, bl_r0 + bl_h - 1, bl_c0 + bl_w - 1)
        br = (br_r0, br_c0, br_r0 + br_h - 1, br_c0 + br_w - 1)
        rects = [tl, tr, bl] if count == 3 else [tl, tr, bl, br]

        if any(r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1 for r0, c0, r1, c1 in rects):
            continue
        if any(_boxes_touch(rects[i], rects[j]) for i in range(len(rects)) for j in range(i + 1, len(rects))):
            continue
        return rects
    return None


def _relative_positions(parent: Rect, size: tuple[int, int], side: str, gap: int, shape: tuple[int, int]) -> list[Rect]:
    h, w = size
    pr0, pc0, pr1, pc1 = parent
    out: list[Rect] = []
    if side in ("left", "right"):
        if side == "left":
            c0 = pc0 - gap - w
        else:
            c0 = pc1 + gap + 1
        c1 = c0 + w - 1
        for r0 in range(pr0 - h + 4, pr1 - 1):
            r1 = r0 + h - 1
            overlap = min(pr1 - 1, r1 - 1) - max(pr0 + 1, r0 + 1) + 1
            if overlap < 3:
                continue
            if r0 < 1 or c0 < 1 or r1 >= shape[0] - 1 or c1 >= shape[1] - 1:
                continue
            out.append((r0, c0, r1, c1))
    else:
        if side == "up":
            r0 = pr0 - gap - h
        else:
            r0 = pr1 + gap + 1
        r1 = r0 + h - 1
        for c0 in range(pc0 - w + 4, pc1 - 1):
            c1 = c0 + w - 1
            overlap = min(pc1 - 1, c1 - 1) - max(pc0 + 1, c0 + 1) + 1
            if overlap < 3:
                continue
            if r0 < 1 or c0 < 1 or r1 >= shape[0] - 1 or c1 >= shape[1] - 1:
                continue
            out.append((r0, c0, r1, c1))
    random.shuffle(out)
    return out


def _add_leaf(rects: list[Rect], shape: tuple[int, int]) -> list[Rect] | None:
    for _ in range(200):
        parent_idx = random.randrange(len(rects))
        parent = rects[parent_idx]
        size = _random_room_size(big=False)
        gap = random.randint(1, 6)
        sides = ["left", "right", "up", "down"]
        random.shuffle(sides)
        for side in sides:
            for cand in _relative_positions(parent, size, side, gap, shape):
                if any(_boxes_touch(cand, other) for other in rects):
                    continue
                return rects + [cand]
    return None


def _edge_signature(edge: dict) -> tuple:
    return edge["gap"], edge["kind"], edge["a"], edge["b"], edge["r0"], edge["r1"], edge["c0"], edge["c1"]


def _is_unique_mst(edges: list[dict], n_rects: int) -> bool:
    if len({edge["gap"] for edge in edges}) != len(edges):
        return False
    chosen = _spanning_edges(edges, n_rects)
    total = sum(edge["gap"] for edge in chosen)
    target = {_edge_signature(edge) for edge in chosen}
    found = 0

    parent = list(range(n_rects))

    def find(p: list[int], x: int) -> int:
        while p[x] != x:
            p[x] = p[p[x]]
            x = p[x]
        return x

    def search(idx: int, selected: list[int], p: list[int], total_gap: int) -> None:
        nonlocal found
        if found > 1:
            return
        if len(selected) == n_rects - 1:
            roots = {find(p[:], i) for i in range(n_rects)}
            if len(roots) == 1 and total_gap == total:
                found += 1
            return
        if idx == len(edges):
            return
        if total_gap > total:
            return
        if len(selected) + (len(edges) - idx) < n_rects - 1:
            return

        edge = edges[idx]
        a = find(p[:], edge["a"])
        b = find(p[:], edge["b"])
        if a != b:
            p2 = p[:]
            ra = find(p2, edge["a"])
            rb = find(p2, edge["b"])
            p2[rb] = ra
            search(idx + 1, selected + [idx], p2, total_gap + edge["gap"])
        search(idx + 1, selected, p[:], total_gap)

    search(0, [], parent, 0)
    return found == 1 and sum(edge["gap"] for edge in chosen) == total and len(target) == n_rects - 1


def _max_distance_after_fill(grid: np.ndarray) -> int:
    rects = _rectangles(grid)
    edges = _candidate_edges(grid, rects)
    chosen = _spanning_edges(edges, len(rects))
    out = grid.copy()
    for edge in chosen:
        _apply_edge(out, edge, rects)
    enclosed = _enclosed_black_mask(out)
    dist = _distance_field(out, enclosed)
    return int(dist[enclosed].max()) if np.any(enclosed) else 0


def _valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        rects = _rectangles(grid)
        edges = _candidate_edges(grid, rects)
        if len(edges) < len(rects) - 1:
            return False
        chosen = _spanning_edges(edges, len(rects))
        if len({edge["gap"] for edge in edges}) != len(edges):
            return False
        if len(rects) >= 4 and len(edges) < len(rects):
            return False
        if not _is_unique_mst(edges, len(rects)):
            return False
        if len({edge["gap"] for edge in chosen}) < 2:
            return False
        if _max_distance_after_fill(grid) < 4:
            return False
        out = main(grid)
        colors = set(map(int, np.unique(out)))
        if not {RED, BLUE, YELLOW}.issubset(colors):
            return False
        return True
    except AssertionError:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(23, 29)
    w = random.randint(23, 29)
    room_count = random.choices([3, 4, 5], weights=[1, 3, 3])[0]

    base_count = 3 if room_count == 3 else 4
    rects = _build_cycle_rects(h, w, base_count)
    if rects is None:
        return None
    if room_count == 5:
        rects = _add_leaf(rects, (h, w))
        if rects is None:
            return None

    grid = np.zeros((h, w), dtype=int)
    for rect in rects:
        _draw_frame(grid, rect)
    if not _valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(1200):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
