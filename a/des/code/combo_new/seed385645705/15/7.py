from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
GEN_WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _flood_from_edges(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (1, w - 1):
            if 0 <= c < w and passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(1, w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _infer_wall_color(grid: np.ndarray) -> int:
    vals = [int(v) for v in grid[:, 1:].ravel() if int(v) != BG]
    assert vals, "expected at least one non-background wall pixel"
    return Counter(vals).most_common(1)[0][0]


def _legend_colors(grid: np.ndarray, wall_color: int) -> list[int]:
    colors: list[int] = []
    seen: set[int] = set()
    for r in range(grid.shape[0]):
        v = int(grid[r, 0])
        if v in (BG, wall_color) or v in seen:
            continue
        colors.append(v)
        seen.add(v)
    assert colors, "expected a non-empty left-edge legend"
    return colors


def _enclosed_regions(grid: np.ndarray, wall_color: int) -> list[list[tuple[int, int]]]:
    passable = grid != wall_color
    passable[:, 0] = False
    outside = _flood_from_edges(passable)
    enclosed = passable & ~outside
    return _components(enclosed)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow seed pixels to fill their enclosed soil regions in left-legend priority order.

    Input structure:
    - Background is black.
    - The leftmost column contains a vertical legend of distinct colors whose top-to-bottom order defines growth priority.
    - Elsewhere, gray wall pixels draw several disjoint closed soil regions with black interiors, and some interiors contain colored seed pixels using legend colors.

    Rule:
    - Read the legend from top to bottom to obtain the processing order.
    - Find each enclosed non-wall region that does not connect to the outer background.
    - If a region contains one or more seed colors, fill every black cell in that region with the seed color that appears latest in the legend; seed pixels themselves stay unchanged.
    - Leave the legend, wall pixels, and seedless regions unchanged.

    Key insight: The left-edge color strip is a scheduler, not decoration: it orders region growth, so multi-seed regions resolve to the lowest legend seed while preserving the visible seed pixels.
    """
    g = np.asarray(grid, dtype=int)
    wall_color = _infer_wall_color(g)
    legend = _legend_colors(g, wall_color)
    priority = {color: idx for idx, color in enumerate(legend)}

    out = g.copy()
    for comp in _enclosed_regions(g, wall_color):
        fill_color = None
        fill_idx = -1
        for r, c in comp:
            v = int(g[r, c])
            if v in priority and priority[v] > fill_idx:
                fill_idx = priority[v]
                fill_color = v
        if fill_color is None:
            continue
        for r, c in comp:
            if int(g[r, c]) == BG:
                out[r, c] = fill_color
    return out


def _grow_connected_mask(h: int, w: int, target: int, rng: random.Random) -> np.ndarray:
    cells = {(rng.randrange(h), rng.randrange(w))}
    frontier = set()
    for nr, nc in _neighbors4(*next(iter(cells))):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < target and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in _neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_region_patch(rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int]]]:
    inner_h = rng.randint(2, 6)
    inner_w = rng.randint(2, 6)
    if rng.random() < 0.35:
        inner = np.ones((inner_h, inner_w), dtype=bool)
    else:
        area_lo = max(4, (inner_h * inner_w) // 2)
        area_hi = inner_h * inner_w
        inner = _grow_connected_mask(inner_h, inner_w, rng.randint(area_lo, area_hi), rng)

    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    inner_mask = np.zeros_like(patch, dtype=bool)
    inner_mask[1 : inner_h + 1, 1 : inner_w + 1] = inner

    wall_mask = np.zeros_like(inner_mask)
    for r, c in np.argwhere(inner_mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if not inner_mask[nr, nc]:
                wall_mask[nr, nc] = True
    patch[wall_mask] = GEN_WALL

    interior = [(int(r), int(c)) for r, c in np.argwhere(inner_mask)]
    assert interior, "region interior must be non-empty"
    return patch, interior


def _spread_points(points: list[tuple[int, int]], count: int, rng: random.Random) -> list[tuple[int, int]]:
    if count == 1:
        return [rng.choice(points)]
    best = None
    best_score = -1
    tries = 24
    for _ in range(tries):
        cand = rng.sample(points, count)
        score = min(
            abs(r1 - r2) + abs(c1 - c2)
            for i, (r1, c1) in enumerate(cand)
            for r2, c2 in cand[i + 1 :]
        )
        if score > best_score:
            best_score = score
            best = cand
    assert best is not None
    return best


def _expected_output(
    inp: np.ndarray,
    legend: list[int],
    region_infos: list[dict[str, object]],
) -> np.ndarray:
    order = {color: idx for idx, color in enumerate(legend)}
    out = inp.copy()
    for info in region_infos:
        seeds: dict[tuple[int, int], int] = info["seeds"]  # type: ignore[assignment]
        if not seeds:
            continue
        fill_color = max(seeds.values(), key=lambda color: order[color])
        for r, c in info["interior"]:  # type: ignore[index]
            if (r, c) not in seeds:
                out[r, c] = fill_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 29)
    w = rng.randint(15, 29)
    if w < 10:
        return None

    legend_len = rng.randint(2, min(5, h - 4, len(COLOR_POOL)))
    legend = rng.sample(list(COLOR_POOL), legend_len)

    grid = np.full((h, w), BG, dtype=int)
    for r, color in enumerate(legend):
        grid[r, 0] = color

    blocked = np.zeros((h, w), dtype=bool)
    blocked[:, 0] = True

    region_count = rng.randint(3, 6)
    statuses = ["none", "single", "multi"]
    region_statuses = statuses[:]
    while len(region_statuses) < region_count:
        region_statuses.append(rng.choice(statuses))
    rng.shuffle(region_statuses)

    region_infos: list[dict[str, object]] = []
    for status in region_statuses:
        placed = False
        for _ in range(90):
            patch, local_interior = _make_region_patch(rng)
            ph, pw = patch.shape
            positions: list[tuple[int, int]] = []
            for r0 in range(1, h - ph):
                for c0 in range(2, w - pw):
                    if blocked[r0 : r0 + ph, c0 : c0 + pw].any():
                        continue
                    positions.append((r0, c0))
            if not positions:
                continue

            r0, c0 = rng.choice(positions)
            sub = patch != BG
            grid[r0 : r0 + ph, c0 : c0 + pw][sub] = patch[sub]
            blocked[
                max(0, r0 - 1) : min(h, r0 + ph + 1),
                max(0, c0 - 1) : min(w, c0 + pw + 1),
            ] = True

            interior = [(r0 + rr, c0 + cc) for rr, cc in local_interior]
            seeds: dict[tuple[int, int], int] = {}
            if status != "none":
                max_seed_count = min(3, legend_len, len(interior))
                seed_count = 1 if status == "single" else rng.randint(2, max_seed_count)
                cells = _spread_points(interior, seed_count, rng)
                colors = [rng.choice(legend)] if seed_count == 1 else rng.sample(legend, seed_count)
                for cell, color in zip(cells, colors):
                    seeds[cell] = color
                    grid[cell] = color

            region_infos.append({"interior": interior, "seeds": seeds, "status": status})
            placed = True
            break
        if not placed:
            return None

    expected = _expected_output(grid, legend, region_infos)

    seed_counts = Counter(info["status"] for info in region_infos)
    if seed_counts["none"] == 0 or seed_counts["single"] == 0 or seed_counts["multi"] == 0:
        return None

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
