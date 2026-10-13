from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BLACK = 0
RED = 2
GREEN = 3

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]

MAX_RETRIES = 400
MAX_RAY = 4

BASE_SHAPES = [
    [(0, 0), (1, 1), (2, 0)],  # 8-connected but not 4-connected
    [(0, 0), (1, 0), (1, 1)],
    [(0, 1), (1, 0), (1, 1), (2, 0)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)],
    [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)],
]


def _neighbors(coord: tuple[int, int], connectivity: int) -> Iterable[tuple[int, int]]:
    dirs = DIRS8 if connectivity == 8 else DIRS4
    r, c = coord
    for dr, dc in dirs:
        yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nr, nc in _neighbors(cur, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _anchor(coords: list[tuple[int, int]]) -> tuple[int, int]:
    max_row = max(r for r, _ in coords)
    return min((r, c) for r, c in coords if r == max_row)


def _analyze_grid(grid: np.ndarray) -> dict:
    fg_mask = np.asarray(grid == GREEN, dtype=bool)
    comps = _components(fg_mask, connectivity=8)
    singletons = [comp[0] for comp in comps if len(comp) == 1]
    cluster_comps = [comp for comp in comps if len(comp) > 1]
    infos = []
    for idx, comp in enumerate(cluster_comps):
        infos.append(
            {
                "id": idx,
                "coords": comp,
                "coord_set": set(comp),
                "bbox": _bbox(comp),
                "anchor": _anchor(comp),
            }
        )
    assignments = {info["id"]: [] for info in infos}
    singleton_meta = []
    for pix in singletons:
        r, c = pix
        scored = []
        for info in infos:
            dist = min(abs(r - rr) + abs(c - cc) for rr, cc in info["coords"])
            scored.append((dist, info["bbox"][1], info["bbox"][0], info["id"]))
        winner = min(scored)
        owner = winner[3]
        tie = sum(1 for item in scored if item[0] == winner[0]) > 1
        anchor_owner = min(
            (
                abs(r - info["anchor"][0]) + abs(c - info["anchor"][1]),
                info["bbox"][1],
                info["bbox"][0],
                info["id"],
            )
            for info in infos
        )[3]
        assignments[owner].append(pix)
        singleton_meta.append(
            {
                "coord": pix,
                "owner": owner,
                "tie": tie,
                "anchor_owner": anchor_owner,
            }
        )
    return {
        "clusters": infos,
        "singletons": singletons,
        "assignments": assignments,
        "singleton_meta": singleton_meta,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Erase isolated green noise, assign it to the nearest 8-connected cluster, and turn each cluster's reclaimed count into a downward red ray.

    Input structure:
    - A black background contains only green foreground pixels.
    - The green pixels form several 8-connected multi-pixel clusters plus some isolated single green pixels.
    - Clusters may be irregular and can rely on diagonal contact for 8-connectivity.

    Rule:
    - Label all green components using 8-connectivity.
    - Every size-1 component is an isolated pixel; assign it to the multi-pixel cluster whose nearest pixel has minimum Manhattan distance, breaking ties by smaller bbox left column and then smaller bbox top row.
    - Remove all isolated pixels, keep every original multi-pixel green cluster, and for each cluster draw a vertical red ray downward from the cluster's lowest pixel (leftmost if several) with length equal to the number of assigned isolated pixels, stopping early if a non-black cell or the bottom edge is reached.

    Key insight:
    - The removed singleton noise is not discarded: each erased pixel is recycled into ray length for one nearby cluster, so correct solving needs both 8-connected component detection and per-pixel nearest-cluster assignment.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    analysis = _analyze_grid(grid)
    out = grid.copy()
    for r, c in analysis["singletons"]:
        out[r, c] = BLACK

    for info in analysis["clusters"]:
        ray_len = len(analysis["assignments"][info["id"]])
        ar, ac = info["anchor"]
        for step in range(1, ray_len + 1):
            rr = ar + step
            if rr >= out.shape[0] or out[rr, ac] != BLACK:
                break
            out[rr, ac] = RED

    return out


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return sorted((r - min_r, c - min_c) for r, c in coords)


def _transform_shape(coords: list[tuple[int, int]], rot: int, flip: bool) -> list[tuple[int, int]]:
    pts = [(r, c) for r, c in coords]
    if flip:
        pts = [(r, -c) for r, c in pts]
    for _ in range(rot % 4):
        pts = [(c, -r) for r, c in pts]
    return _normalize(pts)


def _is_four_connected(coords: list[tuple[int, int]]) -> bool:
    coord_set = set(coords)
    stack = [coords[0]]
    seen = {coords[0]}
    while stack:
        cur = stack.pop()
        for nxt in _neighbors(cur, 4):
            if nxt in coord_set and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(coords)


def _pairwise_min_manhattan(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    return min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in a for r2, c2 in b)


def _reserved_ray_cells(clusters: list[list[tuple[int, int]]], h: int) -> set[tuple[int, int]]:
    reserved: set[tuple[int, int]] = set()
    for comp in clusters:
        ar, ac = _anchor(comp)
        for step in range(1, MAX_RAY + 1):
            rr = ar + step
            if rr < h:
                reserved.add((rr, ac))
    return reserved


def _cluster_candidates(
    h: int,
    w: int,
    cluster_count: int,
    rng: random.Random,
) -> list[list[tuple[int, int]]] | None:
    min_anchor_row = 2
    max_anchor_row = h - 2 - MAX_RAY
    if max_anchor_row < min_anchor_row:
        return None

    clusters: list[list[tuple[int, int]]] = []
    used_anchor_cols: set[int] = set()
    need_diag = True

    for idx in range(cluster_count):
        placed = False
        for _ in range(160):
            if need_diag:
                base = BASE_SHAPES[0]
            else:
                base = rng.choice(BASE_SHAPES)
            shape = _transform_shape(base, rng.randrange(4), bool(rng.randrange(2)))
            local_anchor = _anchor(shape)
            max_local_c = max(c for _, c in shape)
            anchor_col_low = local_anchor[1]
            anchor_col_high = w - 1 - (max_local_c - local_anchor[1])
            if anchor_col_low > anchor_col_high:
                continue
            anchor_row_low = max(min_anchor_row, local_anchor[0])
            anchor_row_high = max_anchor_row
            if anchor_row_low > anchor_row_high:
                continue
            anchor_c = rng.randint(anchor_col_low, anchor_col_high)
            if anchor_c in used_anchor_cols:
                continue
            anchor_r = rng.randint(anchor_row_low, anchor_row_high)
            dr = anchor_r - local_anchor[0]
            dc = anchor_c - local_anchor[1]
            placed_coords = sorted((r + dr, c + dc) for r, c in shape)
            if any(not (0 <= r < h and 0 <= c < w) for r, c in placed_coords):
                continue
            if any(_pairwise_min_manhattan(placed_coords, other) < 3 for other in clusters):
                continue
            clusters.append(placed_coords)
            used_anchor_cols.add(anchor_c)
            if not _is_four_connected(shape):
                need_diag = False
            placed = True
            break
        if not placed:
            return None

    if need_diag:
        return None
    return clusters


def _candidate_metadata(clusters: list[list[tuple[int, int]]], h: int, w: int) -> dict[tuple[int, int], dict]:
    cluster_cells = {cell for comp in clusters for cell in comp}
    forbidden = set(cluster_cells)
    for r, c in cluster_cells:
        for nr, nc in _neighbors((r, c), 8):
            if 0 <= nr < h and 0 <= nc < w:
                forbidden.add((nr, nc))
    forbidden |= _reserved_ray_cells(clusters, h)

    analysis_stub = {
        "clusters": [
            {"id": idx, "coords": comp, "bbox": _bbox(comp), "anchor": _anchor(comp)}
            for idx, comp in enumerate(clusters)
        ]
    }

    meta: dict[tuple[int, int], dict] = {}
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                continue
            scored = []
            for info in analysis_stub["clusters"]:
                dist = min(abs(r - rr) + abs(c - cc) for rr, cc in info["coords"])
                scored.append((dist, info["bbox"][1], info["bbox"][0], info["id"]))
            winner = min(scored)
            owner = winner[3]
            tie = sum(1 for item in scored if item[0] == winner[0]) > 1
            anchor_owner = min(
                (
                    abs(r - info["anchor"][0]) + abs(c - info["anchor"][1]),
                    info["bbox"][1],
                    info["bbox"][0],
                    info["id"],
                )
                for info in analysis_stub["clusters"]
            )[3]
            meta[(r, c)] = {
                "owner": owner,
                "tie": tie,
                "anchor_owner": anchor_owner,
            }
    return meta


def _touches_selected(cell: tuple[int, int], chosen: list[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in chosen:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def _choose_singletons(
    meta: dict[tuple[int, int], dict],
    cluster_count: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    cells = list(meta)
    tie_cells = [cell for cell in cells if meta[cell]["tie"]]
    mismatch_cells = [cell for cell in cells if meta[cell]["owner"] != meta[cell]["anchor_owner"]]
    if not tie_cells or not mismatch_cells:
        return None

    for _ in range(200):
        chosen: list[tuple[int, int]] = []
        counts = [0] * cluster_count

        both = [cell for cell in tie_cells if meta[cell]["owner"] != meta[cell]["anchor_owner"]]
        first_pool = both if both else tie_cells
        first = rng.choice(first_pool)
        chosen.append(first)
        counts[meta[first]["owner"]] += 1

        if meta[first]["owner"] == meta[first]["anchor_owner"]:
            options = [cell for cell in mismatch_cells if not _touches_selected(cell, chosen)]
            if not options:
                continue
            second = rng.choice(options)
            chosen.append(second)
            counts[meta[second]["owner"]] += 1

        target_total = rng.randint(max(cluster_count + 1, len(chosen) + 2), min(9, len(cells)))
        while len(chosen) < target_total:
            available = [
                cell
                for cell in cells
                if not _touches_selected(cell, chosen) and counts[meta[cell]["owner"]] < MAX_RAY
            ]
            if not available:
                break
            min_count = min(counts)
            preferred = [cell for cell in available if counts[meta[cell]["owner"]] == min_count]
            pool = preferred if preferred and rng.random() < 0.7 else available
            cell = rng.choice(pool)
            chosen.append(cell)
            counts[meta[cell]["owner"]] += 1

        if len(chosen) < max(5, cluster_count + 1):
            continue
        positive_counts = [count for count in counts if count > 0]
        if len(set(counts)) < 2 or len({meta[cell]["owner"] for cell in chosen}) < 2:
            continue
        if max(counts) > MAX_RAY or not positive_counts:
            continue
        if not any(meta[cell]["tie"] for cell in chosen):
            continue
        if not any(meta[cell]["owner"] != meta[cell]["anchor_owner"] for cell in chosen):
            continue
        return chosen

    return None


def _paint_grid(h: int, w: int, clusters: list[list[tuple[int, int]]], singletons: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    for comp in clusters:
        for r, c in comp:
            grid[r, c] = GREEN
    for r, c in singletons:
        grid[r, c] = GREEN
    return grid


def _semantic_checks(grid: np.ndarray) -> bool:
    analysis = _analyze_grid(grid)
    if not (3 <= len(analysis["clusters"]) <= 5):
        return False
    if len(analysis["singletons"]) < 5:
        return False
    if not any(not _is_four_connected(info["coords"]) for info in analysis["clusters"]):
        return False
    if not any(item["tie"] for item in analysis["singleton_meta"]):
        return False
    if not any(item["owner"] != item["anchor_owner"] for item in analysis["singleton_meta"]):
        return False

    anchors = [info["anchor"] for info in analysis["clusters"]]
    if len({c for _, c in anchors}) != len(anchors):
        return False

    counts = [len(analysis["assignments"][info["id"]]) for info in analysis["clusters"]]
    if max(counts, default=0) > MAX_RAY or len(set(counts)) < 2:
        return False

    cluster_cells = {cell for info in analysis["clusters"] for cell in info["coords"]}
    for info in analysis["clusters"]:
        ar, ac = info["anchor"]
        for step in range(1, MAX_RAY + 1):
            rr = ar + step
            if rr >= grid.shape[0]:
                return False
            if (rr, ac) in cluster_cells:
                return False
            if grid[rr, ac] != BLACK:
                return False

    out = main(grid)
    if np.count_nonzero(grid != BLACK) != np.count_nonzero(out != BLACK):
        return False
    if not np.any(out == RED):
        return False

    for info in analysis["clusters"]:
        expected_len = len(analysis["assignments"][info["id"]])
        ar, ac = info["anchor"]
        drawn = 0
        rr = ar + 1
        while rr < out.shape[0] and out[rr, ac] == RED:
            drawn += 1
            rr += 1
        if drawn != expected_len:
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        rng = random.Random()
        h = rng.randint(13, 18)
        w = rng.randint(13, 18)
        cluster_count = rng.randint(3, 5)
        clusters = _cluster_candidates(h, w, cluster_count, rng)
        if clusters is None:
            continue
        meta = _candidate_metadata(clusters, h, w)
        singletons = _choose_singletons(meta, cluster_count, rng)
        if singletons is None:
            continue
        grid = _paint_grid(h, w, clusters, singletons)
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid noise-recycling-ray input")
