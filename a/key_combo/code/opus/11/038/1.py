from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FG = 8
HOLE_TO_COLOR = {1: 1, 2: 3, 3: 2, 4: 4}
PATTERNS = {
    1: [((1,),)],
    2: [((1, 1),), ((1,), (1,))],
    3: [((1, 1, 1),), ((1,), (1,), (1,))],
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _coords_from_mask(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _hole_count(coords: np.ndarray) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    obj = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    obj[coords[:, 0] - r0, coords[:, 1] - c0] = True
    padded = np.pad(obj, 1, constant_values=False)
    holes = 0
    for bg_comp in _coords_from_mask(~padded):
        rr = bg_comp[:, 0]
        cc = bg_comp[:, 1]
        touches_outer = np.any(
            (rr == 0) | (rr == padded.shape[0] - 1) | (cc == 0) | (cc == padded.shape[1] - 1)
        )
        if not touches_outer:
            holes += 1
    return holes


def _bridge_edges(coords: np.ndarray) -> tuple[list[tuple[int, ...]], set[tuple[int, int]]]:
    pts = [tuple(map(int, rc)) for rc in coords]
    n = len(pts)
    index = {pt: i for i, pt in enumerate(pts)}
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, (r, c) in enumerate(pts):
        for nr, nc in _neighbors4(r, c):
            j = index.get((nr, nc))
            if j is not None:
                adj[i].append(j)

    tin = [-1] * n
    low = [0] * n
    timer = 0
    bridges: set[tuple[int, int]] = set()

    def dfs(v: int, parent: int) -> None:
        nonlocal timer
        tin[v] = low[v] = timer
        timer += 1
        for to in adj[v]:
            if to == parent:
                continue
            if tin[to] != -1:
                low[v] = min(low[v], tin[to])
                continue
            dfs(to, v)
            low[v] = min(low[v], low[to])
            if low[to] > tin[v]:
                bridges.add((min(v, to), max(v, to)))

    for v in range(n):
        if tin[v] == -1:
            dfs(v, -1)
    return [tuple(neis) for neis in adj], bridges


def _split_cluster_objects(coords: np.ndarray) -> list[np.ndarray]:
    if coords.shape[0] <= 1:
        return [coords]

    adj, bridges = _bridge_edges(coords)
    core_nodes: list[int] = []
    for i, neis in enumerate(adj):
        if any((min(i, j), max(i, j)) not in bridges for j in neis):
            core_nodes.append(i)
    if not core_nodes:
        return [coords]

    core_set = set(core_nodes)
    seen: set[int] = set()
    objects: list[np.ndarray] = []
    for start in core_nodes:
        if start in seen:
            continue
        stack = [start]
        seen.add(start)
        pts: list[tuple[int, int]] = []
        while stack:
            cur = stack.pop()
            pts.append(tuple(map(int, coords[cur])))
            for nxt in adj[cur]:
                if nxt in core_set and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        objects.append(np.array(pts, dtype=int))
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each bridge-connected teal cluster by the total number of holes across its member objects.

    Input structure:
    - A black background contains 3 to 5 teal 4-connected closed objects, each enclosing 1 to 3 black holes.
    - Some object pairs are joined by 1-pixel-wide straight teal bridges, while other objects remain isolated.
    - Object bboxes do not overlap, and bridges only connect object boundaries.

    Rule:
    - Find every 4-connected teal cluster, treating bridges as connectivity links between objects.
    - Within each cluster, separate the cyclic object cores from the bridge-only corridor pixels, then count the enclosed black holes of each object core.
    - Sum those hole counts per cluster and recolor the entire cluster, including bridge pixels: 1->blue, 2->green, 3->red, 4 or more->yellow; keep black background unchanged.

    Key insight:
    - The bridge layout determines which objects share one color decision, and that color depends on the cluster-level hole total rather than any single object's own hole count.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for cluster in _coords_from_mask(g == FG):
        objects = _split_cluster_objects(cluster)
        total_holes = sum(_hole_count(obj) for obj in objects)
        color = HOLE_TO_COLOR[4 if total_holes >= 4 else total_holes]
        out[cluster[:, 0], cluster[:, 1]] = color
    return out


def _object_sprite(hole_count: int) -> np.ndarray:
    for _ in range(200):
        pattern = random.choice(PATTERNS[hole_count])
        rows = len(pattern)
        cols = len(pattern[0])
        row_heights = [random.randint(1, 2) for _ in range(rows)]
        col_widths = [random.randint(1, 2) for _ in range(cols)]
        h = sum(row_heights) + rows + 1
        w = sum(col_widths) + cols + 1
        sprite = np.full((h, w), FG, dtype=int)
        rr = 1
        for r, cell_h in enumerate(row_heights):
            cc = 1
            for c, cell_w in enumerate(col_widths):
                if pattern[r][c] == 1:
                    sprite[rr : rr + cell_h, cc : cc + cell_w] = BG
                cc += cell_w + 1
            rr += cell_h + 1
        if random.random() < 0.5:
            sprite = np.fliplr(sprite)
        if random.random() < 0.5:
            sprite = np.flipud(sprite)
        sprite = np.rot90(sprite, random.randint(0, 3)).copy()
        comps = _coords_from_mask(sprite == FG)
        if len(comps) != 1:
            continue
        if _hole_count(comps[0]) != hole_count:
            continue
        area = int(np.count_nonzero(sprite == FG))
        if not (8 <= area <= 20):
            continue
        inferred = _split_cluster_objects(comps[0])
        if len(inferred) != 1 or inferred[0].shape[0] != comps[0].shape[0]:
            continue
        return sprite
    raise RuntimeError(f"failed to build object with {hole_count} holes")


def _compositions(total: int, parts: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(prefix: list[int], remaining: int, slots: int) -> None:
        if slots == 0:
            if remaining == 0:
                out.append(tuple(prefix))
            return
        lo = 1
        hi = min(3, remaining - (slots - 1))
        for value in range(lo, hi + 1):
            prefix.append(value)
            rec(prefix, remaining - value, slots - 1)
            prefix.pop()

    rec([], total, parts)
    return out


def _sample_cluster_sizes(n_objects: int) -> list[int]:
    options = {
        3: [[2, 1]],
        4: [[2, 1, 1], [3, 1], [2, 2]],
        5: [[2, 1, 1, 1], [3, 1, 1], [2, 2, 1], [4, 1]],
    }
    return list(random.choice(options[n_objects]))


def _sample_cluster_holes(cluster_sizes: list[int]) -> list[list[int]]:
    candidate_totals: list[tuple[int, ...]] = []
    options_per_cluster: list[list[tuple[int, ...]]] = []
    domains: list[list[int]] = []
    for size in cluster_sizes:
        if size == 1:
            dom = [1, 2, 3]
        elif size == 2:
            dom = [2, 3, 4]
        elif size == 3:
            dom = [3, 4]
        else:
            dom = [4]
        domains.append(dom)

    def choose_totals(i: int, chosen: list[int]) -> None:
        if i == len(cluster_sizes):
            if len(set(chosen)) >= 2:
                candidate_totals.append(tuple(chosen))
            return
        values = domains[i][:]
        random.shuffle(values)
        for value in values:
            chosen.append(value)
            choose_totals(i + 1, chosen)
            chosen.pop()

    choose_totals(0, [])
    totals = random.choice(candidate_totals)
    for size, total in zip(cluster_sizes, totals):
        comps = _compositions(total, size)
        random.shuffle(comps)
        options_per_cluster.append(comps)

    clusters: list[list[int]] = []
    for size, total, comps in zip(cluster_sizes, totals, options_per_cluster):
        chosen = list(random.choice(comps))
        random.shuffle(chosen)
        assert size == len(chosen) and sum(chosen) == total
        clusters.append(chosen)
    return clusters


def _assemble_cluster(object_holes: list[int]) -> tuple[np.ndarray, dict]:
    sprites = [_object_sprite(h) for h in object_holes]
    orientation = random.choice(["h", "v"]) if len(sprites) > 1 else "single"
    gaps = [random.randint(1, 4) for _ in range(max(0, len(sprites) - 1))]

    if orientation in ("single", "h"):
        anchors = [sprite.shape[0] // 2 for sprite in sprites]
        baseline = max(anchors)
        tops = [baseline - anchor for anchor in anchors]
        lefts: list[int] = []
        cur = 0
        for i, sprite in enumerate(sprites):
            lefts.append(cur)
            cur += sprite.shape[1]
            if i < len(gaps):
                cur += gaps[i]
        height = max(top + sprite.shape[0] for top, sprite in zip(tops, sprites))
        patch = np.zeros((height, cur), dtype=int)
        object_coords: list[np.ndarray] = []
        bridge_coords: list[np.ndarray] = []
        for top, left, sprite in zip(tops, lefts, sprites):
            rr, cc = np.where(sprite == FG)
            patch[top + rr, left + cc] = FG
            object_coords.append(np.column_stack((top + rr, left + cc)).astype(int))
        for i, gap in enumerate(gaps):
            prev_right = lefts[i] + sprites[i].shape[1] - 1
            next_left = lefts[i + 1]
            cols = list(range(prev_right + 1, next_left))
            if not cols or len(cols) != gap:
                raise RuntimeError("invalid horizontal bridge")
            for c in cols:
                patch[baseline, c] = FG
            bridge_coords.append(np.array([(baseline, c) for c in cols], dtype=int))
    else:
        anchors = [sprite.shape[1] // 2 for sprite in sprites]
        baseline = max(anchors)
        lefts = [baseline - anchor for anchor in anchors]
        tops: list[int] = []
        cur = 0
        for i, sprite in enumerate(sprites):
            tops.append(cur)
            cur += sprite.shape[0]
            if i < len(gaps):
                cur += gaps[i]
        width = max(left + sprite.shape[1] for left, sprite in zip(lefts, sprites))
        patch = np.zeros((cur, width), dtype=int)
        object_coords = []
        bridge_coords = []
        for top, left, sprite in zip(tops, lefts, sprites):
            rr, cc = np.where(sprite == FG)
            patch[top + rr, left + cc] = FG
            object_coords.append(np.column_stack((top + rr, left + cc)).astype(int))
        for i, gap in enumerate(gaps):
            prev_bottom = tops[i] + sprites[i].shape[0] - 1
            next_top = tops[i + 1]
            rows = list(range(prev_bottom + 1, next_top))
            if not rows or len(rows) != gap:
                raise RuntimeError("invalid vertical bridge")
            for r in rows:
                patch[r, baseline] = FG
            bridge_coords.append(np.array([(r, baseline) for r in rows], dtype=int))

    cluster_coords = np.argwhere(patch == FG).astype(int)
    meta = {
        "object_holes": list(object_holes),
        "object_coords": object_coords,
        "bridge_coords": bridge_coords,
        "cluster_coords": cluster_coords,
        "total_holes": int(sum(object_holes)),
    }
    return patch, meta


def _can_place(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int, pad: int = 2) -> bool:
    rr, cc = np.where(patch == FG)
    h, w = grid.shape
    for dr, dc in zip(rr, cc):
        gr = int(r0 + dr)
        gc = int(c0 + dc)
        if not (0 <= gr < h and 0 <= gc < w):
            return False
        r1 = max(0, gr - pad)
        r2 = min(h, gr + pad + 1)
        c1 = max(0, gc - pad)
        c2 = min(w, gc + pad + 1)
        if np.any(grid[r1:r2, c1:c2] == FG):
            return False
    return True


def _place_clusters(patches: list[np.ndarray], metas: list[dict]) -> tuple[np.ndarray, list[dict]] | None:
    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    for _ in range(120):
        min_h = max(15, max_h + 2)
        max_canvas_h = min(26, max(15, max_h + 10))
        min_w = max(15, max_w + 2)
        max_canvas_w = min(26, max(15, max_w + 10))
        if min_h > max_canvas_h or min_w > max_canvas_w:
            return None
        h = random.randint(min_h, max_canvas_h)
        w = random.randint(min_w, max_canvas_w)
        grid = np.zeros((h, w), dtype=int)
        placed: list[dict] = []
        ok = True
        order = list(range(len(patches)))
        random.shuffle(order)
        for idx in order:
            patch = patches[idx]
            meta = metas[idx]
            ph, pw = patch.shape
            placed_here = False
            for _ in range(120):
                r0 = random.randint(0, h - ph)
                c0 = random.randint(0, w - pw)
                if not _can_place(grid, patch, r0, c0):
                    continue
                rr, cc = np.where(patch == FG)
                grid[r0 + rr, c0 + cc] = FG
                placed_meta = {
                    "object_holes": list(meta["object_holes"]),
                    "object_coords": [coords + np.array([r0, c0]) for coords in meta["object_coords"]],
                    "bridge_coords": [coords + np.array([r0, c0]) for coords in meta["bridge_coords"]],
                    "cluster_coords": meta["cluster_coords"] + np.array([r0, c0]),
                    "total_holes": int(meta["total_holes"]),
                }
                placed.append(placed_meta)
                placed_here = True
                break
            if not placed_here:
                ok = False
                break
        if ok:
            return grid, placed
    return None


def _cluster_key(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset((int(r), int(c)) for r, c in coords)


def _validate_scene(grid: np.ndarray, metas: list[dict]) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) - {BG, FG}:
        return False
    if not (3 <= sum(len(meta["object_holes"]) for meta in metas) <= 5):
        return False
    totals = [int(meta["total_holes"]) for meta in metas]
    if not any(len(meta["object_holes"]) > 1 for meta in metas):
        return False
    if not any(len(meta["object_holes"]) == 1 for meta in metas):
        return False
    if any(total < 1 or total > 4 for total in totals):
        return False
    if len(set(totals)) < 2:
        return False

    clusters = _coords_from_mask(grid == FG)
    if len(clusters) != len(metas):
        return False
    inferred_by_cluster = {_cluster_key(cluster): cluster for cluster in clusters}

    for meta in metas:
        cluster_key = _cluster_key(meta["cluster_coords"])
        cluster = inferred_by_cluster.get(cluster_key)
        if cluster is None:
            return False
        objects = _split_cluster_objects(cluster)
        inferred_holes = sorted(_hole_count(obj) for obj in objects)
        if inferred_holes != sorted(meta["object_holes"]):
            return False
        if len(objects) > 1 and sum(coords.shape[0] for coords in meta["bridge_coords"]) == 0:
            return False
        for obj_coords, intended_holes in zip(meta["object_coords"], meta["object_holes"]):
            if _hole_count(obj_coords) != intended_holes:
                return False

    out = main(grid)
    if out.shape != grid.shape:
        return False
    if not np.all(out[grid == BG] == BG):
        return False
    for meta in metas:
        color = HOLE_TO_COLOR[4 if meta["total_holes"] >= 4 else meta["total_holes"]]
        coords = meta["cluster_coords"]
        if not np.all(out[coords[:, 0], coords[:, 1]] == color):
            return False
    if len(set(map(int, np.unique(out))) - {BG}) < 2:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        n_objects = random.randint(3, 5)
        cluster_sizes = _sample_cluster_sizes(n_objects)
        cluster_holes = _sample_cluster_holes(cluster_sizes)
        metas: list[dict] = []
        patches: list[np.ndarray] = []
        try:
            for holes in cluster_holes:
                patch, meta = _assemble_cluster(holes)
                patches.append(patch)
                metas.append(meta)
        except RuntimeError:
            continue

        placed = _place_clusters(patches, metas)
        if placed is None:
            continue
        grid, placed_metas = placed
        if _validate_scene(grid, placed_metas):
            return grid
    raise RuntimeError("failed to generate valid bridge-hole scene")
