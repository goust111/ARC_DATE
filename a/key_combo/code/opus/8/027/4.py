from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
SEED_COLORS = (BLUE, RED, GREEN, YELLOW)

Coord = tuple[int, int]


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def component_coords(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    h, w = mask.shape
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)] if connectivity == 4 else [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),            (0, 1),
        (1, -1),  (1, 0),   (1, 1),
    ]
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def mask_has_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    pad = np.zeros((h + 2, w + 2), dtype=bool)
    pad[1:-1, 1:-1] = mask
    seen = np.zeros_like(pad, dtype=bool)
    q = deque([(0, 0)])
    seen[0, 0] = True
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if in_bounds(pad.shape, nr, nc) and not pad[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return bool((~pad & ~seen).any())


def collect_boundary_edges(obstacle_mask: np.ndarray) -> list[dict]:
    h, w = obstacle_mask.shape
    raw_edges: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not obstacle_mask[r, c]:
                continue
            if r == 0 or not obstacle_mask[r - 1, c]:
                raw_edges.append(
                    {
                        "start": (r, c),
                        "end": (r, c + 1),
                        "outside": (r - 1, c),
                        "normal": (-1, 0),
                    }
                )
            if c == w - 1 or not obstacle_mask[r, c + 1]:
                raw_edges.append(
                    {
                        "start": (r, c + 1),
                        "end": (r + 1, c + 1),
                        "outside": (r, c + 1),
                        "normal": (0, 1),
                    }
                )
            if r == h - 1 or not obstacle_mask[r + 1, c]:
                raw_edges.append(
                    {
                        "start": (r + 1, c + 1),
                        "end": (r + 1, c),
                        "outside": (r + 1, c),
                        "normal": (1, 0),
                    }
                )
            if c == 0 or not obstacle_mask[r, c - 1]:
                raw_edges.append(
                    {
                        "start": (r + 1, c),
                        "end": (r, c),
                        "outside": (r, c - 1),
                        "normal": (0, -1),
                    }
                )

    assert raw_edges, "expected a non-empty obstacle"
    by_start = {edge["start"]: idx for idx, edge in enumerate(raw_edges)}
    assert len(by_start) == len(raw_edges), "outer boundary must be a simple directed cycle"

    ordered: list[dict] = []
    cur = by_start[min(by_start)]
    seen: set[int] = set()
    while cur not in seen:
        seen.add(cur)
        ordered.append(raw_edges[cur])
        nxt = raw_edges[cur]["end"]
        assert nxt in by_start, "boundary chain must close"
        cur = by_start[nxt]
    assert cur == by_start[min(by_start)]
    assert len(ordered) == len(raw_edges), "expected one outer boundary cycle"
    return ordered


def contour_ring(obstacle_mask: np.ndarray) -> list[dict]:
    ring: list[dict] = []
    for edge in collect_boundary_edges(obstacle_mask):
        outside = tuple(map(int, edge["outside"]))
        if ring and ring[-1]["cell"] == outside:
            continue
        ring.append(
            {
                "cell": outside,
                "normal": tuple(map(int, edge["normal"])),
            }
        )
    if ring and ring[0]["cell"] == ring[-1]["cell"]:
        ring.pop()
    assert ring, "expected a non-empty contour ring"
    cells = [item["cell"] for item in ring]
    assert len(cells) == len(set(cells)), "contour ring cells must be unique"
    return ring


def circular_distance(n: int, i: int, j: int) -> int:
    d = abs(i - j)
    return min(d, n - d)


def parse_puzzle(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BLACK, "expected black background"

    obstacle_components = component_coords(g == GREY, connectivity=8)
    assert len(obstacle_components) == 1, "expected one grey obstacle"
    obstacle_mask = np.zeros_like(g, dtype=bool)
    coords = obstacle_components[0]
    obstacle_mask[coords[:, 0], coords[:, 1]] = True
    assert not mask_has_hole(obstacle_mask), "obstacle must be hole-free"

    ring = contour_ring(obstacle_mask)
    for item in ring:
        r, c = item["cell"]
        assert in_bounds(g.shape, r, c), "contour ring must stay inside the grid"
        assert not obstacle_mask[r, c]

    seeds: list[dict] = []
    for color in SEED_COLORS:
        comps = component_coords(g == color, connectivity=4)
        assert len(comps) <= 1, "each seed color may appear at most once"
        if not comps:
            continue
        coord_set = {tuple(map(int, p)) for p in comps[0]}
        claim: list[int] = []
        for idx, item in enumerate(ring):
            r, c = item["cell"]
            if (r, c) in coord_set or any((nr, nc) in coord_set for nr, nc in neighbors4(r, c)):
                claim.append(idx)
        assert claim, "each seed must touch the contour ring"
        seeds.append(
            {
                "color": int(color),
                "coords": comps[0],
                "coord_set": coord_set,
                "area": int(len(comps[0])),
                "claim": tuple(sorted(set(claim))),
            }
        )

    assert 2 <= len(seeds) <= 4
    assert len({seed["color"] for seed in seeds}) == len(seeds)
    assert len({seed["area"] for seed in seeds}) == len(seeds), "seed areas must be distinct"
    assert len(ring) > 2 * len(seeds)
    return obstacle_mask, ring, seeds


def assign_ring_by_distance(ring: list[dict], seeds: list[dict]) -> dict[int, int]:
    n = len(ring)
    owner: dict[int, int] = {}
    for idx in range(n):
        best_key = None
        best_seed = None
        for s_idx, seed in enumerate(seeds):
            dist = min(circular_distance(n, idx, src) for src in seed["claim"])
            key = (dist, -seed["area"])
            if best_key is None or key < best_key:
                best_key = key
                best_seed = s_idx
        assert best_seed is not None
        owner[idx] = best_seed
    return owner


def simulate_round_growth(ring: list[dict], seeds: list[dict]) -> dict[int, int]:
    n = len(ring)
    owner: dict[int, int] = {}
    for idx in range(n):
        contenders = [s_idx for s_idx, seed in enumerate(seeds) if idx in seed["claim"]]
        if contenders:
            owner[idx] = max(contenders, key=lambda s_idx: seeds[s_idx]["area"])

    while len(owner) < n:
        proposals: dict[int, list[int]] = {}
        for idx, own in tuple(owner.items()):
            for nxt in ((idx - 1) % n, (idx + 1) % n):
                if nxt not in owner:
                    proposals.setdefault(nxt, []).append(own)
        assert proposals, "growth stalled before covering the contour ring"
        for idx, contenders in proposals.items():
            owner[idx] = max(set(contenders), key=lambda s_idx: seeds[s_idx]["area"])
    return owner


def contested_indices(ring: list[dict], seeds: list[dict]) -> set[int]:
    n = len(ring)
    contested: set[int] = set()
    for idx in range(n):
        dists = [
            min(circular_distance(n, idx, src) for src in seed["claim"])
            for seed in seeds
        ]
        best = min(dists)
        if sum(dist == best for dist in dists) >= 2:
            contested.add(idx)
    return contested


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the obstacle's outside contour ring by letting unequal-size seeds compete along that closed one-dimensional boundary order.

    Input structure:
    - A black background contains one large grey obstacle and several small seed components.
    - The grey obstacle is a single 8-connected, hole-free region.
    - Each seed is a small 4-connected component in a distinct color from blue/red/green/yellow and touches the obstacle's outside contour neighborhood.

    Rule:
    - Trace the obstacle's outside contour as an ordered closed ring of outside-adjacent cells around the grey region.
    - Each seed initially claims every contour-ring cell that is itself part of the seed or 4-adjacent to any seed pixel.
    - Grow every seed's claimed territory simultaneously in both ring directions.
    - When multiple seeds reach the same contour-ring cell in the same round, the seed with larger original area wins that cell.
    - Keep the grey obstacle and every non-ring cell unchanged.

    Key insight:
    - The competition happens on the contour's cyclic order rather than through 2D distance in the canvas, and equal-time collisions are resolved by seed area.
    """
    g = np.asarray(grid, dtype=int)
    obstacle_mask, ring, seeds = parse_puzzle(g)
    owner = assign_ring_by_distance(ring, seeds)

    out = g.copy()
    for idx, item in enumerate(ring):
        r, c = item["cell"]
        out[r, c] = seeds[owner[idx]]["color"]
    out[obstacle_mask] = GREY
    return out


def random_convex_patch() -> np.ndarray:
    for _ in range(400):
        h = random.randint(6, 8)
        width = random.randint(5, 7)
        left = random.randint(0, 2)
        rows: list[tuple[int, int]] = []
        cur_left = left
        cur_right = left + width - 1
        max_w = cur_right + random.randint(2, 4) + 1
        changed = False
        for _r in range(h):
            dl = random.choice([-1, 0, 0, 1])
            dr = random.choice([-1, 0, 0, 1])
            nxt_left = max(0, cur_left + dl)
            nxt_right = min(max_w - 1, cur_right + dr)
            if nxt_right - nxt_left + 1 < 5:
                nxt_right = nxt_left + 4
            if nxt_right >= max_w:
                shift = nxt_right - max_w + 1
                nxt_left -= shift
                nxt_right -= shift
            if nxt_left != cur_left or nxt_right != cur_right:
                changed = True
            rows.append((nxt_left, nxt_right))
            cur_left, cur_right = nxt_left, nxt_right

        min_left = min(a for a, _b in rows)
        rows = [(a - min_left, b - min_left) for a, b in rows]
        w = max(b for _a, b in rows) + 1
        patch = np.zeros((h, w), dtype=bool)
        for r, (a, b) in enumerate(rows):
            patch[r, a : b + 1] = True

        area = int(patch.sum())
        if not changed or not (20 <= area <= 40):
            continue
        if len(component_coords(patch, connectivity=4)) != 1 or mask_has_hole(patch):
            continue
        return patch
    raise RuntimeError("failed to generate a valid obstacle patch")


def embed_obstacle(patch: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = patch.shape
    top = random.randint(4, 6)
    left = random.randint(4, 6)
    bottom = random.randint(4, 6)
    right = random.randint(4, 6)
    grid = np.zeros((h + top + bottom, w + left + right), dtype=int)
    obstacle_mask = np.zeros_like(grid, dtype=bool)
    obstacle_mask[top : top + h, left : left + w] = patch
    grid[obstacle_mask] = GREY
    return grid, obstacle_mask


def anchor_candidates(grid: np.ndarray, obstacle_mask: np.ndarray, ring: list[dict]) -> list[tuple[int, Coord, Coord]]:
    ring_cells = {item["cell"] for item in ring}
    out: list[tuple[int, Coord, Coord]] = []
    for idx, item in enumerate(ring):
        r, c = item["cell"]
        dr, dc = item["normal"]
        ok = True
        for step in (1, 2):
            nr = r + dr * step
            nc = c + dc * step
            if not in_bounds(grid.shape, nr, nc) or obstacle_mask[nr, nc]:
                ok = False
                break
            if step >= 1 and (nr, nc) in ring_cells:
                ok = False
                break
        if ok:
            out.append((idx, (r, c), (dr, dc)))
    return out


def circular_gap_ok(indices: list[int], n: int, min_gap: int) -> bool:
    ordered = sorted(indices)
    for i, idx in enumerate(ordered):
        nxt = ordered[(i + 1) % len(ordered)]
        gap = (nxt - idx) % n
        if gap == 0 or gap < min_gap:
            return False
    return True


def place_seeds(grid: np.ndarray, anchors: list[tuple[int, Coord, Coord]], areas: list[int], colors: list[int]) -> np.ndarray | None:
    out = grid.copy()
    for area, color, (_idx, (r, c), (dr, dc)) in zip(areas, colors, anchors):
        cells = [(r + dr * step, c + dc * step) for step in range(area)]
        for rr, cc in cells:
            if not in_bounds(out.shape, rr, cc) or out[rr, cc] != BLACK:
                return None
        for rr, cc in cells:
            out[rr, cc] = color
    return out


def validate_instance(grid: np.ndarray) -> bool:
    try:
        obstacle_mask, ring, seeds = parse_puzzle(grid)
    except Exception:
        return False

    owner_direct = assign_ring_by_distance(ring, seeds)
    owner_rounds = simulate_round_growth(ring, seeds)
    if owner_direct != owner_rounds:
        return False
    if not contested_indices(ring, seeds):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    ring_cells = {item["cell"] for item in ring}
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if (r, c) not in ring_cells and not obstacle_mask[r, c] and out[r, c] != grid[r, c]:
                return False

    ring_counts = []
    for s_idx, seed in enumerate(seeds):
        owned = {ring[idx]["cell"] for idx, own in owner_direct.items() if own == s_idx}
        if not (owned - seed["coord_set"]):
            return False
        ring_counts.append(len(owned))
    if len(set(ring_counts)) == 1:
        return False

    return bool(out.ndim == 2 and max(out.shape) <= 30)


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid, obstacle_mask = embed_obstacle(random_convex_patch())
        try:
            ring = contour_ring(obstacle_mask)
        except Exception:
            continue
        if len(ring) < 18:
            continue

        candidates = anchor_candidates(grid, obstacle_mask, ring)
        if len(candidates) < 2:
            continue

        k = 3 if len(candidates) >= 3 and random.random() < 0.7 else 2
        colors = random.sample(list(SEED_COLORS), k)

        for _ in range(80):
            areas = random.sample([1, 2, 3], k)
            chosen = random.sample(candidates, k)
            indices = [idx for idx, _cell, _normal in chosen]
            if not circular_gap_ok(indices, len(ring), min_gap=4):
                continue
            puzzle = place_seeds(grid, chosen, areas, colors)
            if puzzle is not None and validate_instance(puzzle):
                return puzzle

    raise RuntimeError("failed to generate a valid contour-growth puzzle")
