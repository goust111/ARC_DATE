from __future__ import annotations

import random
import itertools
from collections import deque

import numpy as np


BG = 0
FRAME = 1
GREEN = 3
MARKER_COLORS = (2, 3, 4, 6, 7)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    rects: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        h, w = patch.shape
        frame = np.zeros((h, w), dtype=bool)
        frame[0, :] = True
        frame[-1, :] = True
        frame[:, 0] = True
        frame[:, -1] = True
        if h < 4 or w < 4 or not np.array_equal(patch == FRAME, frame):
            raise ValueError("expected only hollow blue rectangles")
        rects.append((r0, c0, r1, c1))
    rects.sort()
    if not rects:
        raise ValueError("no rectangles found")
    return rects


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
                            "gap": c1 - c0 + 1,
                            "a": left_i,
                            "b": right_i,
                            "r0": r0,
                            "r1": r1,
                            "c0": c0,
                            "c1": c1,
                            "span": r1 - r0 + 1,
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
                            "gap": r1 - r0 + 1,
                            "a": upper_i,
                            "b": lower_i,
                            "r0": r0,
                            "r1": r1,
                            "c0": c0,
                            "c1": c1,
                            "span": c1 - c0 + 1,
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

    def union(a: int, b: int) -> None:
        ra = find(a)
        rb = find(b)
        if ra == rb:
            raise ValueError("cycle")
        parent[rb] = ra

    chosen: list[dict] = []
    for gap in sorted({edge["gap"] for edge in edges}):
        group = [edge for edge in edges if edge["gap"] == gap]
        eligible = [edge for edge in group if find(edge["a"]) != find(edge["b"])]
        temp_parent: dict[int, int] = {}

        def temp_find(x: int) -> int:
            temp_parent.setdefault(x, x)
            while temp_parent[x] != x:
                temp_parent[x] = temp_parent[temp_parent[x]]
                x = temp_parent[x]
            return x

        for edge in eligible:
            ra = find(edge["a"])
            rb = find(edge["b"])
            ta = temp_find(ra)
            tb = temp_find(rb)
            if ta == tb:
                raise ValueError("ambiguous equal-gap cycle")
            temp_parent[tb] = ta

        for edge in eligible:
            union(edge["a"], edge["b"])
            chosen.append(edge)
        if len(chosen) == n_rects - 1:
            break

    if len(chosen) != n_rects - 1:
        raise ValueError("rooms are not fully connectable")
    root = find(0)
    if any(find(i) != root for i in range(n_rects)):
        raise ValueError("spanning tree failed")
    return chosen


def _apply_edge(out: np.ndarray, edge: dict, rects: list[tuple[int, int, int, int]]) -> None:
    r0, r1, c0, c1 = edge["r0"], edge["r1"], edge["c0"], edge["c1"]
    if edge["kind"] == "h":
        left = rects[edge["a"]]
        right = rects[edge["b"]]
        out[r0, c0 : c1 + 1] = FRAME
        out[r1, c0 : c1 + 1] = FRAME
        if r0 + 1 <= r1 - 1:
            out[r0 + 1 : r1, left[3]] = BG
            out[r0 + 1 : r1, right[1]] = BG
    else:
        upper = rects[edge["a"]]
        lower = rects[edge["b"]]
        out[r0 : r1 + 1, c0] = FRAME
        out[r0 : r1 + 1, c1] = FRAME
        if c0 + 1 <= c1 - 1:
            out[upper[2], c0 + 1 : c1] = BG
            out[lower[0], c0 + 1 : c1] = BG


def _corridor_grid(grid: np.ndarray) -> tuple[list[tuple[int, int, int, int]], list[dict], np.ndarray]:
    rects = _find_rectangles(grid)
    edges = _candidate_edges(grid, rects)
    chosen = _spanning_edges(edges, len(rects))
    out = np.array(grid, copy=True)
    for edge in chosen:
        _apply_edge(out, edge, rects)
    return rects, chosen, out


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    passable = grid != FRAME
    outside = np.zeros_like(passable, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return _components(passable & ~outside)


def _region_info(grid: np.ndarray, rects: list[tuple[int, int, int, int]]) -> list[dict]:
    infos: list[dict] = []
    for coords in _enclosed_regions(grid):
        mask = np.zeros_like(grid, dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        members: list[int] = []
        for idx, (r0, c0, r1, c1) in enumerate(rects):
            if mask[r0 + 1 : r1, c0 + 1 : c1].any():
                members.append(idx)
        if not members:
            continue
        colors = sorted({int(grid[r, c]) for r, c in coords if int(grid[r, c]) not in (BG, FRAME)})
        infos.append(
            {
                "coords": coords,
                "members": members,
                "colors": colors,
            }
        )
    infos.sort(key=lambda info: _bbox(info["coords"]))
    return infos


def _build_expected_output(grid: np.ndarray) -> np.ndarray:
    rects, _chosen, corridor = _corridor_grid(grid)
    out = corridor.copy()
    for info in _region_info(corridor, rects):
        bottoms = [rects[idx][2] for idx in info["members"]]
        if bottoms.count(max(bottoms)) != 1:
            raise ValueError("lowest rectangle must be unique")
        lowest_idx = max(info["members"], key=lambda idx: rects[idx][2])
        r0, c0, r1, c1 = rects[lowest_idx]
        height = len(info["colors"])
        if height == 0:
            raise ValueError("each enclosed region needs marker colors")
        if r1 + 1 + height > out.shape[0]:
            raise ValueError("marker rectangle exceeds canvas")
        out[r1 + 1 : r1 + 1 + height, c0 : c1 + 1] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect blue hollow rooms, measure color diversity inside each merged interior, and draw a green marker below the region's lowest room.

    Input structure:
    - The grid has a black background, 3-5 disjoint blue hollow rectangles, and colored marker pixels inside the rectangles.
    - Visible room-to-room gap candidates are horizontal or vertical clear background strips whose facing room interiors overlap.
    - The interior markers use non-blue colors and stay inside the rooms while the area below the bottom rooms is left free for later markers.

    Rule:
    - Detect every blue hollow rectangle and choose the shortest visible gaps that form a spanning tree over the rectangles.
    - For each chosen horizontal gap, draw the two blue corridor boundary rows across the gap and erase the facing wall segment between them; do the transposed operation for chosen vertical gaps.
    - After the corridors are added, find every 4-connected enclosed non-blue region surrounded by blue, count how many distinct marker colors it contains, and draw a solid green rectangle directly below the lowest rectangle belonging to that region, with matching width and height equal to that color count.

    Key insight:
    - The corridor network changes which room interiors become one enclosed region, so the relevant color count is taken after connectivity changes, not room by room before the corridors are added.
    """
    return _build_expected_output(np.asarray(grid, dtype=int))


def _draw_frame(grid: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _mirror_h(rects: list[tuple[int, int, int, int]], width: int) -> list[tuple[int, int, int, int]]:
    return [(r0, width - 1 - c1, r1, width - 1 - c0) for r0, c0, r1, c1 in rects]


def _make_layout_three(split: bool) -> list[tuple[int, int, int, int]] | None:
    wl = random.randint(5, 7)
    wr = random.randint(5, 7)
    ht = random.randint(6, 7)
    hb = random.randint(6, 7)
    gv = random.randint(2, 3)
    gh = gv + random.randint(1, 2)

    pass_shift_max = min(wl - 5, 1)
    if pass_shift_max < 0:
        return None
    left_shift = random.randint(0, pass_shift_max)

    if split:
        deltas = [d for d in range(1, 5) if 1 <= (ht - 2 - d) <= 2]
    else:
        deltas = [d for d in range(1, 5) if (ht - 2 - d) >= 3]
    if not deltas:
        return None
    top_delta = random.choice(deltas)

    a = (0, 0, ht - 1, wl - 1)
    c = (ht + gv, left_shift, ht + gv + hb - 1, left_shift + wl - 1)
    b = (top_delta, wl + gh, top_delta + ht - 1, wl + gh + wr - 1)
    return [a, b, c]


def _make_layout_four(split: bool) -> list[tuple[int, int, int, int]] | None:
    wl = random.randint(6, 7)
    wr = random.randint(6, 7)
    ht = random.randint(6, 7)
    hb = random.randint(6, 7)
    gv = random.randint(2, 3)
    gh_top = gv + random.randint(1, 2)
    if gh_top >= 5:
        gh_bottom_extra = 1
    else:
        gh_bottom_extra = random.randint(1, min(2, 5 - gh_top))

    max_left_shift = wl - 5
    if max_left_shift < 0:
        return None
    left_shift = random.randint(0, max_left_shift)
    right_shift = left_shift + gh_bottom_extra
    if right_shift > wr - 5:
        return None

    if split:
        deltas = [d for d in range(1, 5) if 1 <= (ht - 2 - d) <= 2 and (hb - 2 - d) >= 1]
    else:
        deltas = [d for d in range(1, 5) if (ht - 2 - d) >= 3]
    if not deltas:
        return None
    top_delta = random.choice(deltas)

    a = (0, 0, ht - 1, wl - 1)
    b = (top_delta, wl + gh_top, top_delta + ht - 1, wl + gh_top + wr - 1)
    c = (ht + gv, left_shift, ht + gv + hb - 1, left_shift + wl - 1)
    d = (top_delta + ht + gv, wl + gh_top + right_shift, top_delta + ht + gv + hb - 1, wl + gh_top + right_shift + wr - 1)
    return [a, b, c, d]


def _make_layout_five(split: bool) -> list[tuple[int, int, int, int]] | None:
    wl = random.randint(6, 7)
    wm = random.randint(6, 7)
    wr = random.randint(5, 7)
    ht = random.randint(6, 7)
    hb = random.randint(6, 7)
    he = random.randint(6, 7)
    gv = random.randint(2, 3)
    gh_top = gv + random.randint(1, 2)
    gh_de = random.randint(2, min(4, gh_top))
    gh_bottom_extra = 1 if gh_top == 5 else random.randint(1, min(2, 5 - gh_top))

    max_left_shift = wl - 5
    if max_left_shift < 0:
        return None
    left_shift = random.randint(0, max_left_shift)
    mid_shift = left_shift + gh_bottom_extra
    if mid_shift > wm - 5:
        return None

    if split:
        deltas = [d for d in range(1, 5) if 1 <= (ht - 2 - d) <= 2 and (hb - 2 - d) >= 1]
    else:
        deltas = [d for d in range(1, 5) if (ht - 2 - d) >= 3]
    if not deltas:
        return None
    top_delta = random.choice(deltas)
    e_delta = 1
    if he - 2 - e_delta < 3:
        return None

    a = (0, 0, ht - 1, wl - 1)
    b = (top_delta, wl + gh_top, top_delta + ht - 1, wl + gh_top + wm - 1)
    c = (ht + gv, left_shift, ht + gv + hb - 1, left_shift + wl - 1)
    d = (top_delta + ht + gv, wl + gh_top + mid_shift, top_delta + ht + gv + hb - 1, wl + gh_top + mid_shift + wm - 1)
    e = (d[0] + e_delta, d[3] + 1 + gh_de, d[0] + e_delta + he - 1, d[3] + 1 + gh_de + wr - 1)
    return [a, b, c, d, e]


def _shift_rects(rects: list[tuple[int, int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rects]


def _sample_marker_sets(region_members: list[list[int]], n_rects: int) -> list[tuple[int, ...]] | None:
    marker_sets: list[tuple[int, ...] | None] = [None] * n_rects
    used: set[tuple[int, ...]] = set()
    target_region = max(region_members, key=len)
    union_colors = random.sample(MARKER_COLORS, k=min(5, max(2, len(target_region) + 1)))
    if len(target_region) == 1:
        union_colors = random.sample(MARKER_COLORS, k=1)
    for offset, idx in enumerate(target_region):
        size = 1 if len(target_region) > 1 else random.randint(1, 3)
        colors = tuple(sorted({union_colors[offset % len(union_colors)]} if size == 1 else random.sample(union_colors, k=size)))
        if len(target_region) > 1 and offset == len(target_region) - 1 and len(set().union(*[set(s) for s in marker_sets if s])) < len(union_colors):
            colors = tuple(sorted(set(colors) | {union_colors[-1]}))
            colors = colors[:3]
        while colors in used:
            colors = tuple(sorted(random.sample(union_colors, k=random.randint(1, min(3, len(union_colors))))))
        marker_sets[idx] = colors
        used.add(colors)

    all_combos = []
    for size in (1, 2, 3):
        all_combos.extend(tuple(combo) for combo in itertools.combinations(MARKER_COLORS, size))
    random.shuffle(all_combos)
    for idx in range(n_rects):
        if marker_sets[idx] is not None:
            continue
        for combo in all_combos:
            if combo not in used:
                marker_sets[idx] = combo
                used.add(combo)
                break
        if marker_sets[idx] is None:
            return None
    return [tuple(s) for s in marker_sets]


def _place_markers(grid: np.ndarray, rects: list[tuple[int, int, int, int]], marker_sets: list[tuple[int, ...]]) -> None:
    for rect, colors in zip(rects, marker_sets):
        r0, c0, r1, c1 = rect
        cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        random.shuffle(cells)
        for (r, c), color in zip(cells, colors):
            grid[r, c] = color


def _region_members_for_frames(rects: list[tuple[int, int, int, int]], shape: tuple[int, int]) -> list[list[int]] | None:
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        _draw_frame(grid, rect)
    try:
        solved_rects, _chosen, corridor = _corridor_grid(grid)
    except ValueError:
        return None
    if solved_rects != rects:
        return None
    infos = _region_info(corridor, rects)
    if not infos:
        return None
    return [info["members"] for info in infos]


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        rects, chosen, corridor = _corridor_grid(grid)
        n = len(rects)
        if not 3 <= n <= 5:
            return False
        edges = _candidate_edges(grid, rects)
        if n >= 4 and len(edges) <= n - 1:
            return False
        marker_sets = []
        for rect in rects:
            r0, c0, r1, c1 = rect
            colors = sorted({int(v) for v in grid[r0 + 1 : r1, c0 + 1 : c1].flat if int(v) not in (BG, FRAME)})
            if not 1 <= len(colors) <= 3:
                return False
            marker_sets.append(tuple(colors))
            if grid.shape[0] - 1 - r1 < 5:
                return False
        if len(set(marker_sets)) != len(marker_sets):
            return False

        infos = _region_info(corridor, rects)
        if not any(len(info["members"]) >= 2 for info in infos):
            return False
        changed_diversity = False
        for info in infos:
            bottoms = [rects[idx][2] for idx in info["members"]]
            if bottoms.count(max(bottoms)) != 1:
                return False
            if len(info["colors"]) == 0:
                return False
            if len(info["members"]) >= 2:
                if len(info["colors"]) > max(len(marker_sets[idx]) for idx in info["members"]):
                    changed_diversity = True
        if not changed_diversity:
            return False

        expected = _build_expected_output(grid)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        for edge in chosen:
            if edge["span"] >= 3:
                break
        else:
            return False
        return True
    except Exception:
        return False


def _generate_candidate() -> np.ndarray | None:
    family = random.choices(["three", "four", "five"], weights=[1, 4, 4], k=1)[0]
    split = random.random() < 0.5
    if family == "three":
        rects = _make_layout_three(split)
    elif family == "four":
        rects = _make_layout_four(split)
    else:
        rects = _make_layout_five(split)
    if rects is None:
        return None

    local_h = max(r1 for _, _, r1, _ in rects) + 1
    local_w = max(c1 for _, _, _, c1 in rects) + 1
    if local_h + 5 > 30 or local_w > 30:
        return None
    bottom_margin = random.randint(5, min(8, 30 - local_h))
    top_margin = random.randint(0, min(4, 30 - local_h - bottom_margin))
    left_margin = random.randint(0, min(4, 30 - local_w))
    right_margin = random.randint(0, min(4, 30 - local_w - left_margin))

    rects = _shift_rects(rects, top_margin, left_margin)
    h = local_h + top_margin + bottom_margin
    w = local_w + left_margin + right_margin
    if random.random() < 0.5:
        rects = _mirror_h(rects, w)

    region_members = _region_members_for_frames(rects, (h, w))
    if region_members is None:
        return None
    marker_sets = _sample_marker_sets(region_members, len(rects))
    if marker_sets is None:
        return None

    grid = np.full((h, w), BG, dtype=int)
    for rect in rects:
        _draw_frame(grid, rect)
    _place_markers(grid, rects, marker_sets)
    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
