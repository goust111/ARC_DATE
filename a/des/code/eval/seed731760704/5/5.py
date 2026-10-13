from __future__ import annotations

from collections import deque
import random

import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox(points: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = points.min(axis=0)
    r1, c1 = points.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _center(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = bbox
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _boxes_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _find_containers(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    colors = sorted(set(map(int, np.unique(g))) - {bg})
    containers = []
    for fill_color in colors:
        for comp in _components(g == fill_color):
            r0, c0, r1, c1 = _bbox(comp)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if len(comp) != area:
                continue
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            if np.any(g[r0 : r1 + 1, c0 : c1 + 1] != fill_color):
                continue

            br0, bc0, br1, bc1 = r0 - 1, c0 - 1, r1 + 1, c1 + 1
            border_color = int(g[br0, bc0])
            if border_color in (bg, fill_color):
                continue

            ok = True
            for rr in range(br0, br1 + 1):
                for cc in range(bc0, bc1 + 1):
                    if rr in (br0, br1) or cc in (bc0, bc1):
                        if int(g[rr, cc]) != border_color:
                            ok = False
                            break
                if not ok:
                    break
            if not ok:
                continue

            containers.append(
                {
                    "bbox": (br0, bc0, br1, bc1),
                    "fill_color": int(fill_color),
                    "border_color": border_color,
                }
            )

    containers.sort(key=lambda item: item["bbox"])
    return containers, bg


def _extract_structure(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    containers, bg = _find_containers(g)
    if not containers:
        return containers, [], bg, None

    h, w = g.shape
    border_color = int(containers[0]["border_color"])
    for container in containers:
        assert int(container["border_color"]) == border_color

    border_owner = np.full((h, w), -1, dtype=int)
    border_mask = np.zeros((h, w), dtype=bool)
    interior_mask = np.zeros((h, w), dtype=bool)
    for idx, container in enumerate(containers):
        r0, c0, r1, c1 = container["bbox"]
        border_mask[r0, c0 : c1 + 1] = True
        border_mask[r1, c0 : c1 + 1] = True
        border_mask[r0 : r1 + 1, c0] = True
        border_mask[r0 : r1 + 1, c1] = True
        border_owner[r0, c0 : c1 + 1] = idx
        border_owner[r1, c0 : c1 + 1] = idx
        border_owner[r0 : r1 + 1, c0] = idx
        border_owner[r0 : r1 + 1, c1] = idx
        interior_mask[r0 + 1 : r1, c0 + 1 : c1] = True

    traversable = (g != bg) & (~border_mask) & (~interior_mask)
    pipes = []
    for comp in _components(traversable):
        comp_set = {tuple(map(int, pt)) for pt in comp}
        valves = []
        touches: set[int] = set()

        for r, c in comp:
            r = int(r)
            c = int(c)
            if int(g[r, c]) != border_color:
                valves.append((r, c))
            degree = 0
            for nr, nc in _neighbors4(r, c, h, w):
                if (nr, nc) in comp_set:
                    degree += 1
                owner = int(border_owner[nr, nc])
                if owner >= 0:
                    touches.add(owner)
            assert degree <= 2

        assert len(valves) == 1
        assert len(touches) == 2
        pipes.append(
            {
                "cells": comp,
                "valve": valves[0],
                "endpoints": tuple(sorted(touches)),
            }
        )

    pipes.sort(key=lambda item: item["endpoints"])
    return containers, pipes, bg, border_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap container fill colors across pipes whose valve matches exactly one endpoint color.
    """
    g = np.asarray(grid, dtype=int)
    containers, pipes, _bg, _border = _extract_structure(g)
    out = g.copy()
    if not containers:
        return out

    original = [int(container["fill_color"]) for container in containers]
    updated = list(original)
    used: set[int] = set()

    for pipe in pipes:
        a, b = map(int, pipe["endpoints"])
        vr, vc = pipe["valve"]
        valve_color = int(g[vr, vc])
        match_a = valve_color == original[a]
        match_b = valve_color == original[b]
        if match_a ^ match_b:
            assert a not in used and b not in used
            used.add(a)
            used.add(b)
            updated[a] = original[b]
            updated[b] = original[a]

    for idx, container in enumerate(containers):
        r0, c0, r1, c1 = container["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = updated[idx]
    return out


# --------------- generator ---------------

BG = 0
FRAME = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
CONTAINER_COUNT_RANGE = (3, 5)
INTERIOR_H_RANGE = (1, 4)
INTERIOR_W_RANGE = (1, 5)
PLACEMENT_PAD = 2
PIPE_COUNT_RANGE = (2, 4)
MAX_GENERATION_ATTEMPTS = 400
ROUTE_ATTEMPTS_PER_PAIR = 24


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _place_containers(rng: random.Random, h: int, w: int):
    n = rng.randint(*CONTAINER_COUNT_RANGE)
    containers = []
    for _ in range(200):
        if len(containers) == n:
            return containers
        ih = rng.randint(*INTERIOR_H_RANGE)
        iw = rng.randint(*INTERIOR_W_RANGE)
        bh = ih + 2
        bw = iw + 2
        max_r0 = h - bh - 3
        max_c0 = w - bw - 3
        if max_r0 < 2 or max_c0 < 2:
            return None
        r0 = rng.randint(2, max_r0)
        c0 = rng.randint(2, max_c0)
        bbox = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        if any(_boxes_too_close(bbox, item["bbox"], PLACEMENT_PAD) for item in containers):
            continue
        containers.append({"bbox": bbox})
    return None


def _attachment_options(
    bbox: tuple[int, int, int, int],
    target: tuple[float, float],
    shape: tuple[int, int],
) -> list[tuple[float, tuple[int, int], tuple[int, int]]]:
    h, w = shape
    r0, c0, r1, c1 = bbox
    tr, tc = target
    options: list[tuple[float, tuple[int, int], tuple[int, int]]] = []

    if r0 - 1 >= 1:
        for c in range(c0 + 1, c1):
            score = abs(tc - c) + abs(tr - (r0 - 1)) + (0 if tr < r0 else 4)
            options.append((score, (r0, c), (r0 - 1, c)))
    if r1 + 1 < h - 1:
        for c in range(c0 + 1, c1):
            score = abs(tc - c) + abs(tr - (r1 + 1)) + (0 if tr > r1 else 4)
            options.append((score, (r1, c), (r1 + 1, c)))
    if c0 - 1 >= 1:
        for r in range(r0 + 1, r1):
            score = abs(tc - (c0 - 1)) + abs(tr - r) + (0 if tc < c0 else 4)
            options.append((score, (r, c0), (r, c0 - 1)))
    if c1 + 1 < w - 1:
        for r in range(r0 + 1, r1):
            score = abs(tc - (c1 + 1)) + abs(tr - r) + (0 if tc > c1 else 4)
            options.append((score, (r, c1), (r, c1 + 1)))

    return options


def _route_ok(r: int, c: int, occupied: np.ndarray) -> bool:
    if occupied[r, c]:
        return False
    h, w = occupied.shape
    for nr, nc in _neighbors4(r, c, h, w):
        if occupied[nr, nc]:
            return False
    return True


def _find_path(
    start: tuple[int, int], goal: tuple[int, int], occupied: np.ndarray
) -> list[tuple[int, int]] | None:
    if occupied[start] or occupied[goal]:
        return None
    h, w = occupied.shape
    q = deque([start])
    parent = {start: None}
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for nxt in _neighbors4(cur[0], cur[1], h, w):
            if nxt in parent:
                continue
            if nxt == goal:
                if occupied[nxt]:
                    continue
            elif not _route_ok(nxt[0], nxt[1], occupied):
                continue
            parent[nxt] = cur
            q.append(nxt)
    if goal not in parent:
        return None

    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _route_between(
    rng: random.Random,
    a_bbox: tuple[int, int, int, int],
    b_bbox: tuple[int, int, int, int],
    occupied: np.ndarray,
) -> list[tuple[int, int]] | None:
    a_opts = _attachment_options(a_bbox, _center(b_bbox), occupied.shape)
    b_opts = _attachment_options(b_bbox, _center(a_bbox), occupied.shape)
    if not a_opts or not b_opts:
        return None

    a_opts.sort(key=lambda item: item[0] + rng.random())
    b_opts.sort(key=lambda item: item[0] + rng.random())
    pairs = []
    for a_item in a_opts[:6]:
        for b_item in b_opts[:6]:
            pairs.append((a_item[0] + b_item[0] + rng.random(), a_item, b_item))
    pairs.sort(key=lambda item: item[0])

    for _, a_item, b_item in pairs[:ROUTE_ATTEMPTS_PER_PAIR]:
        _a_border, a_start = a_item[1], a_item[2]
        _b_border, b_start = b_item[1], b_item[2]
        path = _find_path(a_start, b_start, occupied)
        if path is not None and len(path) >= 3:
            return path
    return None


def _choose_edges_and_paths(
    rng: random.Random, containers: list[dict], shape: tuple[int, int]
):
    n = len(containers)
    occupied = np.zeros(shape, dtype=bool)
    for container in containers:
        r0, c0, r1, c1 = container["bbox"]
        occupied[r0 : r1 + 1, c0 : c1 + 1] = True

    all_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            ar, ac = _center(containers[i]["bbox"])
            br, bc = _center(containers[j]["bbox"])
            dist = abs(ar - br) + abs(ac - bc)
            all_pairs.append((dist + rng.random(), i, j))
    rng.shuffle(all_pairs)
    all_pairs.sort(key=lambda item: item[0])

    target = min(rng.randint(*PIPE_COUNT_RANGE), len(all_pairs))
    edges = []
    paths = []
    for _, i, j in all_pairs:
        if len(edges) == target:
            break
        path = _route_between(rng, containers[i]["bbox"], containers[j]["bbox"], occupied)
        if path is None:
            continue
        edge = (i, j) if rng.random() < 0.5 else (j, i)
        edges.append(edge)
        paths.append(path)
        for r, c in path:
            occupied[r, c] = True

    if len(edges) < 2:
        return None, None
    return edges, paths


def _matching_subsets(edges: list[tuple[int, int]]) -> list[tuple[int, ...]]:
    subsets: list[tuple[int, ...]] = []
    m = len(edges)
    for mask in range(1 << m):
        used: set[int] = set()
        subset = []
        ok = True
        for i in range(m):
            if not (mask & (1 << i)):
                continue
            a, b = edges[i]
            if a in used or b in used:
                ok = False
                break
            used.add(a)
            used.add(b)
            subset.append(i)
        if ok:
            subsets.append(tuple(subset))
    return subsets


def _assign_semantics(
    rng: random.Random, n_containers: int, edges: list[tuple[int, int]]
):
    matchings = _matching_subsets(edges)
    by_size: dict[int, list[tuple[int, ...]]] = {}
    for subset in matchings:
        by_size.setdefault(len(subset), []).append(subset)

    preferred_sizes = [size for size in by_size if 0 < size < len(edges)]
    sizes = preferred_sizes if preferred_sizes else sorted(by_size)

    for _ in range(200):
        active_size = rng.choice(sizes)
        active_subset = rng.choice(by_size[active_size])
        active_set = set(active_subset)
        kinds = []
        for idx in range(len(edges)):
            if idx in active_set:
                kinds.append(rng.choice(["a", "b"]))
            else:
                kinds.append(rng.choice(["none", "both"]))
        if len(set(kinds)) < 2:
            continue

        dsu = _DSU(n_containers)
        for kind, (a, b) in zip(kinds, edges):
            if kind == "both":
                dsu.union(a, b)

        valid = True
        for kind, (a, b) in zip(kinds, edges):
            if kind in ("a", "b") and dsu.find(a) == dsu.find(b):
                valid = False
                break
        if not valid:
            continue

        roots = sorted({dsu.find(i) for i in range(n_containers)})
        adjacency = {root: set() for root in roots}
        for kind, (a, b) in zip(kinds, edges):
            if kind not in ("a", "b"):
                continue
            ra = dsu.find(a)
            rb = dsu.find(b)
            adjacency[ra].add(rb)
            adjacency[rb].add(ra)

        order = roots[:]
        rng.shuffle(order)
        order.sort(key=lambda root: len(adjacency[root]), reverse=True)

        group_colors: dict[int, int] = {}
        success = True
        for root in order:
            forbidden = {group_colors[nbr] for nbr in adjacency[root] if nbr in group_colors}
            choices = [color for color in COLOR_POOL if color not in forbidden]
            if not choices:
                success = False
                break
            group_colors[root] = rng.choice(choices)
        if not success:
            continue

        container_colors = [group_colors[dsu.find(i)] for i in range(n_containers)]
        valve_colors = []
        for kind, (a, b) in zip(kinds, edges):
            ca = container_colors[a]
            cb = container_colors[b]
            if kind == "a":
                valve_colors.append(ca)
            elif kind == "b":
                valve_colors.append(cb)
            elif kind == "both":
                assert ca == cb
                valve_colors.append(ca)
            else:
                choices = [color for color in COLOR_POOL if color not in {ca, cb}]
                valve_colors.append(rng.choice(choices))
        return kinds, container_colors, valve_colors

    return None


def _build_grid(
    containers: list[dict],
    paths: list[list[tuple[int, int]]],
    container_colors: list[int],
    valve_colors: list[int],
    rng: random.Random,
    shape: tuple[int, int],
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for container, fill_color in zip(containers, container_colors):
        bbox = container["bbox"]
        _paint_frame(grid, bbox, FRAME)
        r0, c0, r1, c1 = bbox
        grid[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    for path, valve_color in zip(paths, valve_colors):
        for r, c in path:
            grid[r, c] = FRAME
        valve_idx = rng.randint(1, len(path) - 2)
        vr, vc = path[valve_idx]
        grid[vr, vc] = valve_color
    return grid


def _expected_output(
    grid: np.ndarray,
    containers: list[dict],
    edges: list[tuple[int, int]],
    kinds: list[str],
    container_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    updated = list(container_colors)
    used: set[int] = set()
    for kind, (a, b) in zip(kinds, edges):
        if kind in ("a", "b"):
            assert a not in used and b not in used
            used.add(a)
            used.add(b)
            updated[a] = container_colors[b]
            updated[b] = container_colors[a]
    for fill_color, container in zip(updated, containers):
        r0, c0, r1, c1 = container["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    containers = _place_containers(rng, h, w)
    if containers is None:
        return None

    edges, paths = _choose_edges_and_paths(rng, containers, (h, w))
    if edges is None or paths is None:
        return None

    assigned = _assign_semantics(rng, len(containers), edges)
    if assigned is None:
        return None
    kinds, container_colors, valve_colors = assigned

    grid = _build_grid(containers, paths, container_colors, valve_colors, rng, (h, w))
    expected = _expected_output(grid, containers, edges, kinds, container_colors)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
