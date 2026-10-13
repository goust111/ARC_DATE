from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
FENCE = 3
SOURCE_COLORS = (1, 2, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
SIDE_PRIORITY = {"top": 0, "left": 1, "bottom": 2, "right": 3}

GEN_RETRIES = 400
GRID_H_RANGE = (22, 32)
GRID_W_RANGE = (22, 32)
FENCE_COUNT_CHOICES = (1, 2, 3)
INTERIOR_H_RANGE = (3, 8)
INTERIOR_W_RANGE = (3, 8)
MAX_SOURCE_DIST = 3
PLACEMENT_ATTEMPTS = 200
BRANCH_TYPES = ("single", "same_edge_multi", "multi_edge")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _perimeter_positions(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            pts.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            pts.append((r, c0))
    return pts


def _find_fences(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    fences: list[tuple[int, int, int, int]] = []
    for comp in _components(grid == FENCE):
        rs = comp[:, 0]
        cs = comp[:, 1]
        r0, r1 = int(rs.min()), int(rs.max())
        c0, c1 = int(cs.min()), int(cs.max())
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        perim = set(_perimeter_positions(r0, c0, r1, c1))
        if not all((int(r), int(c)) in perim for r, c in comp):
            continue
        if any(int(grid[r, c]) != FENCE for r, c in ((r0, c0), (r0, c1), (r1, c0), (r1, c1))):
            continue

        border_vals = [int(grid[r, c]) for r, c in perim]
        if any(v not in (BLACK, FENCE) for v in border_vals):
            continue
        gaps = [(r, c) for r, c in perim if int(grid[r, c]) == BLACK]
        if not 1 <= len(gaps) <= 3:
            continue

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0 or not np.all(interior == BLACK):
            continue
        fences.append((r0, c0, r1, c1))

    fences.sort()
    return fences


def _gap_side(bbox: tuple[int, int, int, int], gap: tuple[int, int]) -> str:
    r0, c0, r1, c1 = bbox
    r, c = gap
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise ValueError("gap is not on the fence border")


def _scan_source(grid: np.ndarray, gap: tuple[int, int], side: str) -> int:
    h, w = grid.shape
    r, c = gap
    if side == "top":
        ray = ((rr, c) for rr in range(r - 1, -1, -1))
    elif side == "bottom":
        ray = ((rr, c) for rr in range(r + 1, h))
    elif side == "left":
        ray = ((r, cc) for cc in range(c - 1, -1, -1))
    else:
        ray = ((r, cc) for cc in range(c + 1, w))

    for rr, cc in ray:
        val = int(grid[rr, cc])
        if val not in (BLACK, FENCE):
            return val
    raise AssertionError("every gap must have a visible source on its outward ray")


def _depth_palette(colors: list[int], depth: int) -> list[int]:
    assert 1 <= len(colors) <= depth
    palette = [colors[0]] * depth
    for idx, band in enumerate(np.array_split(np.arange(depth), len(colors))):
        for d in band:
            palette[int(d)] = int(colors[idx])
    return palette


def _side_bands(
    bbox: tuple[int, int, int, int], gaps: list[tuple[int, int]], grid: np.ndarray
) -> dict[str, list[int]]:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    grouped: dict[str, list[tuple[int, int]]] = {}
    for gap in gaps:
        side = _gap_side(bbox, gap)
        order_key = int(gap[1] if side in ("top", "bottom") else gap[0])
        grouped.setdefault(side, []).append((order_key, _scan_source(grid, gap, side)))

    bands: dict[str, list[int]] = {}
    for side, items in grouped.items():
        items.sort()
        colors = [color for _key, color in items]
        depth = inner_h if side in ("top", "bottom") else inner_w
        bands[side] = _depth_palette(colors, depth)
    return bands


def _fill_fence(out: np.ndarray, bbox: tuple[int, int, int, int], bands: dict[str, list[int]]) -> None:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1

    for dr in range(inner_h):
        for dc in range(inner_w):
            candidates: list[tuple[int, str, int]] = []
            if "top" in bands:
                candidates.append((dr, "top", bands["top"][dr]))
            if "bottom" in bands:
                d = inner_h - 1 - dr
                candidates.append((d, "bottom", bands["bottom"][d]))
            if "left" in bands:
                candidates.append((dc, "left", bands["left"][dc]))
            if "right" in bands:
                d = inner_w - 1 - dc
                candidates.append((d, "right", bands["right"][d]))
            dist, side, color = min(candidates, key=lambda item: (item[0], SIDE_PRIORITY[item[1]]))
            _ = dist, side
            out[r0 + 1 + dr, c0 + 1 + dc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced rectangle by propagating colors inward from its gaps.

    Deterministic interpretation for multi-edge cases:
    - Each side paints inward layer by layer.
    - Multiple gaps on the same side split that side's depth into equal bands.
    - When different sides compete for a cell, the side with the smaller inward
      distance wins; ties break with top > left > bottom > right.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for bbox in _find_fences(g):
        gaps = [pos for pos in _perimeter_positions(*bbox) if int(g[pos]) == BLACK]
        bands = _side_bands(bbox, gaps, g)
        _fill_fence(out, bbox, bands)
    return out


def _branch_sequence(count: int, rng: random.Random) -> list[str]:
    shuffled = list(BRANCH_TYPES)
    rng.shuffle(shuffled)
    branches = []
    for idx in range(count):
        if idx < len(shuffled):
            branches.append(shuffled[idx])
        else:
            branches.append(rng.choice(BRANCH_TYPES))
    rng.shuffle(branches)
    return branches


def _gap_distribution(branch: str, inner_h: int, inner_w: int, rng: random.Random) -> dict[str, int]:
    if branch == "single":
        return {rng.choice(SIDES): 1}

    if branch == "same_edge_multi":
        side = rng.choice(SIDES)
        max_count = inner_w if side in ("top", "bottom") else inner_h
        count = 3 if max_count >= 3 and rng.random() < 0.5 else 2
        return {side: count}

    total = rng.choice((2, 3))
    if total == 2:
        a, b = rng.sample(SIDES, 2)
        return {a: 1, b: 1}

    if rng.random() < 0.5:
        sides = rng.sample(SIDES, 3)
        return {side: 1 for side in sides}

    repeated = rng.choice(SIDES)
    other = rng.choice([side for side in SIDES if side != repeated])
    max_count = inner_w if repeated in ("top", "bottom") else inner_h
    if max_count < 2:
        sides = rng.sample(SIDES, 3)
        return {side: 1 for side in sides}
    return {repeated: 2, other: 1}


def _choose_gap_positions(
    bbox: tuple[int, int, int, int], counts: dict[str, int], rng: random.Random
) -> list[tuple[str, tuple[int, int]]]:
    r0, c0, r1, c1 = bbox
    chosen: list[tuple[str, tuple[int, int]]] = []
    for side, count in counts.items():
        if side == "top":
            cols = sorted(rng.sample(list(range(c0 + 1, c1)), count))
            chosen.extend((side, (r0, c)) for c in cols)
        elif side == "bottom":
            cols = sorted(rng.sample(list(range(c0 + 1, c1)), count))
            chosen.extend((side, (r1, c)) for c in cols)
        elif side == "left":
            rows = sorted(rng.sample(list(range(r0 + 1, r1)), count))
            chosen.extend((side, (r, c0)) for r in rows)
        else:
            rows = sorted(rng.sample(list(range(r0 + 1, r1)), count))
            chosen.extend((side, (r, c1)) for r in rows)
    return chosen


def _source_coord(gap: tuple[int, int], side: str, dist: int) -> tuple[int, int]:
    r, c = gap
    if side == "top":
        return (r - dist, c)
    if side == "bottom":
        return (r + dist, c)
    if side == "left":
        return (r, c - dist)
    return (r, c + dist)


def _draw_fence(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    gaps: list[tuple[str, tuple[int, int]]],
    colors: list[int],
    dists: list[int],
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FENCE
    grid[r1, c0 : c1 + 1] = FENCE
    grid[r0 : r1 + 1, c0] = FENCE
    grid[r0 : r1 + 1, c1] = FENCE
    for (_side, gap) in gaps:
        grid[gap] = BLACK
    for idx, (side, gap) in enumerate(gaps):
        sr, sc = _source_coord(gap, side, dists[idx])
        grid[sr, sc] = int(colors[idx])


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    fence_count = rng.choice(FENCE_COUNT_CHOICES)
    branches = _branch_sequence(fence_count, rng)

    specs: list[tuple[str, int, int, dict[str, int]]] = []
    total_gaps = 0
    for branch in branches:
        inner_h = rng.randint(*INTERIOR_H_RANGE)
        inner_w = rng.randint(*INTERIOR_W_RANGE)
        counts = _gap_distribution(branch, inner_h, inner_w, rng)
        total_gaps += sum(counts.values())
        specs.append((branch, inner_h, inner_w, counts))
    if total_gaps > len(SOURCE_COLORS):
        return None

    color_pool = list(SOURCE_COLORS)
    rng.shuffle(color_pool)
    color_cursor = 0

    for _branch, inner_h, inner_w, counts in specs:
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            min_r0 = MAX_SOURCE_DIST
            min_c0 = MAX_SOURCE_DIST
            max_r0 = h - MAX_SOURCE_DIST - outer_h
            max_c0 = w - MAX_SOURCE_DIST - outer_w
            if max_r0 < min_r0 or max_c0 < min_c0:
                return None

            r0 = rng.randint(min_r0, max_r0)
            c0 = rng.randint(min_c0, max_c0)
            r1 = r0 + outer_h - 1
            c1 = c0 + outer_w - 1

            er0 = max(0, r0 - MAX_SOURCE_DIST)
            ec0 = max(0, c0 - MAX_SOURCE_DIST)
            er1 = min(h - 1, r1 + MAX_SOURCE_DIST)
            ec1 = min(w - 1, c1 + MAX_SOURCE_DIST)
            if occupied[er0 : er1 + 1, ec0 : ec1 + 1].any():
                continue

            bbox = (r0, c0, r1, c1)
            gaps = _choose_gap_positions(bbox, counts, rng)

            source_cells: set[tuple[int, int]] = set()
            dists: list[int] = []
            ok = True
            for side, gap in gaps:
                dist = rng.randint(1, MAX_SOURCE_DIST)
                sr, sc = _source_coord(gap, side, dist)
                if not (0 <= sr < h and 0 <= sc < w):
                    ok = False
                    break
                if occupied[sr, sc] or (sr, sc) in source_cells:
                    ok = False
                    break
                source_cells.add((sr, sc))
                dists.append(dist)
            if not ok:
                continue

            needed = len(gaps)
            colors = color_pool[color_cursor : color_cursor + needed]
            _draw_fence(grid, bbox, gaps, colors, dists)
            occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
            color_cursor += needed
            placed = True
            break

        if not placed:
            return None

    fences = _find_fences(grid)
    if len(fences) != len(specs):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
