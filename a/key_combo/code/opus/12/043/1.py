from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 500
ROW_RANGE = (13, 19)
COL_RANGE = (15, 19)
BOTTOM_BAND = 4


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


def trace_ray(shape: tuple[int, int], seed: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    r, c = seed
    dc = 1 if c < (w / 2.0) else -1
    path: list[tuple[int, int]] = []
    while True:
        path.append((r, c))
        nr = r - 1
        if nr < 0:
            break
        nc = c + dc
        if nc < 0 or nc >= w:
            dc = -dc
            nc = c + dc
        r, c = nr, nc
    return path


def extract_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    coords = np.argwhere(grid != BLACK)
    seeds = [(int(r), int(c), int(grid[int(r), int(c)])) for r, c in coords]
    seeds.sort(key=lambda item: (item[0], item[1], item[2]))
    return seeds


def analyze_instance(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    seeds = extract_seeds(g)
    assert 2 <= len(seeds) <= 3
    assert all(color != BLACK for _, _, color in seeds)
    assert len({color for _, _, color in seeds}) == len(seeds)
    assert len({r for r, _, _ in seeds}) == len(seeds)
    assert len({c for _, c, _ in seeds}) == len(seeds)

    ray_pixels: dict[tuple[int, int], int] = {}
    ray_paths: list[dict] = []
    bounce_counts: list[int] = []
    for r, c, color in seeds:
        path = trace_ray((h, w), (r, c))
        bounces = 0
        prev_dc = None
        for idx in range(1, len(path)):
            dc = path[idx][1] - path[idx - 1][1]
            if prev_dc is not None and dc != prev_dc:
                bounces += 1
            prev_dc = dc
        bounce_counts.append(bounces)
        for pr, pc in path:
            if (pr, pc) in ray_pixels:
                raise AssertionError("ray overlap detected")
            ray_pixels[(pr, pc)] = color
        ray_paths.append({"seed": (r, c), "color": color, "path": path, "bounces": bounces})

    barrier = np.zeros((h, w), dtype=bool)
    for r, c in ray_pixels:
        barrier[r, c] = True

    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict] = []
    for r in range(h):
        for c in range(w):
            if barrier[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            boundary_pixels: list[tuple[int, int, int]] = []
            boundary_set: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds(g.shape, nr, nc) and not barrier[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                for nr, nc in neighbors8(cr, cc):
                    if not in_bounds(g.shape, nr, nc) or not barrier[nr, nc]:
                        continue
                    if (nr, nc) not in boundary_set:
                        boundary_set.add((nr, nc))
                        boundary_pixels.append((nr, nc, ray_pixels[(nr, nc)]))

            counts = Counter(color for _, _, color in boundary_pixels)
            assert counts, "every region must touch at least one ray"
            max_count = max(counts.values())
            winners = [color for color, count in counts.items() if count == max_count]
            if len(winners) == 1:
                fill_color = winners[0]
                tie = False
            else:
                tie = True
                fill_color = min(
                    winners,
                    key=lambda color: min((pc, pr) for pr, pc, clr in boundary_pixels if clr == color),
                )
            regions.append(
                {
                    "cells": cells,
                    "boundary_pixels": boundary_pixels,
                    "counts": counts,
                    "fill_color": fill_color,
                    "tie": tie,
                    "adjacent_colors": sorted(counts),
                }
            )

    return {
        "shape": g.shape,
        "seeds": seeds,
        "ray_paths": ray_paths,
        "ray_pixels": ray_pixels,
        "regions": regions,
        "bounce_counts": bounce_counts,
    }


def render_output_from_analysis(grid: np.ndarray, analysis: dict) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out.fill(BLACK)
    for (r, c), color in analysis["ray_pixels"].items():
        out[r, c] = color
    for region in analysis["regions"]:
        for r, c in region["cells"]:
            out[r, c] = region["fill_color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend bouncing diagonal rays to partition the grid, then color each region by adjacent-ray majority.

    Input structure:
    - The grid is black except for 2 or 3 single colored seed pixels near the bottom.
    - Each seed pixel has a distinct color and determines one diagonal ray.
    - There are no pre-drawn walls; the reflected ray paths themselves become the barriers.

    Rule:
    - For each seed, fire a diagonal ray upward: seeds left of the vertical midpoint move up-right, and seeds on or right of it move up-left.
    - When the next horizontal step would leave the grid, reflect off that side wall by flipping only the column direction; stop once the next row would leave the top.
    - Treat every ray pixel as a barrier, flood-fill the remaining black cells into 4-connected regions, and for each region count adjacent ray pixels by color in the 8-neighborhood.
    - Fill the whole region with the most frequent adjacent ray color; if tied, choose the tied color whose adjacent ray pixel has the smallest column, then the smallest row.

    Key insight:
    - The same reflected rays serve two roles at once: they carve the background into compartments and also provide the color votes that label those compartments.
    """
    analysis = analyze_instance(np.asarray(grid, dtype=int))
    return render_output_from_analysis(grid, analysis)


def choose_columns(width: int, seed_count: int) -> list[int] | None:
    interior = list(range(1, width - 1))
    candidates: list[list[int]] = []
    if seed_count == 2:
        for i, c0 in enumerate(interior):
            for c1 in interior[i + 1 :]:
                cols = [c0, c1]
                if c1 - c0 < 3:
                    continue
                if not any(c < (width / 2.0) for c in cols):
                    continue
                if not any(c >= (width / 2.0) for c in cols):
                    continue
                candidates.append(cols)
    else:
        for i, c0 in enumerate(interior):
            for j, c1 in enumerate(interior[i + 1 :], start=i + 1):
                for c2 in interior[j + 1 :]:
                    cols = [c0, c1, c2]
                    if min(cols[k + 1] - cols[k] for k in range(2)) < 3:
                        continue
                    if not any(c < (width / 2.0) for c in cols):
                        continue
                    if not any(c >= (width / 2.0) for c in cols):
                        continue
                    if c0 == 1 or c2 == width - 2:
                        candidates.append(cols)
                    elif abs(c1 - (width // 2)) <= 2:
                        candidates.append(cols)
    if not candidates:
        return None
    return random.choice(candidates)


def choose_rows(height: int, seed_count: int) -> list[int]:
    band = list(range(height - 1, max(height - BOTTOM_BAND - 1, -1), -1))
    assert len(band) >= seed_count
    rows = random.sample(band, seed_count)
    rows.sort(reverse=True)
    return rows


def maybe_force_tie(target_tie: bool, analysis: dict) -> bool:
    has_tie = any(region["tie"] for region in analysis["regions"])
    has_strict_multicolor = any(
        (not region["tie"]) and len(region["adjacent_colors"]) >= 2 for region in analysis["regions"]
    )
    return has_tie if target_tie else has_strict_multicolor


def is_semantically_good(grid: np.ndarray, analysis: dict, target_tie: bool) -> bool:
    h, w = grid.shape
    seeds = analysis["seeds"]
    rows = [r for r, _, _ in seeds]
    cols = [c for _, c, _ in seeds]
    if any(r < h - BOTTOM_BAND for r in rows):
        return False
    if any(abs(a - b) < 3 for i, a in enumerate(cols) for b in cols[i + 1 :]):
        return False
    if len(analysis["regions"]) < 3:
        return False
    if sum(analysis["bounce_counts"]) < 1:
        return False
    if not any(len(region["adjacent_colors"]) >= 2 for region in analysis["regions"]):
        return False
    if not any(
        (not region["tie"]) and len(region["adjacent_colors"]) >= 2 for region in analysis["regions"]
        for region in analysis["regions"]
    ):
        return False
    if len({region["fill_color"] for region in analysis["regions"]}) < 2:
        return False
    if not maybe_force_tie(target_tie, analysis):
        return False
    if max(len(region["cells"]) for region in analysis["regions"]) > (h * w * 3) // 5:
        return False
    return True


def attempt_generate() -> np.ndarray | None:
    h = random.randint(*ROW_RANGE)
    w = random.randint(*COL_RANGE)
    seed_count = random.choice([2, 3, 3, 3])
    cols = choose_columns(w, seed_count)
    if cols is None:
        return None
    rows = choose_rows(h, seed_count)
    colors = random.sample(PALETTE, seed_count)

    items = list(zip(rows, cols, colors))
    random.shuffle(items)
    grid = np.zeros((h, w), dtype=int)
    for r, c, color in items:
        grid[r, c] = color

    try:
        analysis = analyze_instance(grid)
    except AssertionError:
        return None

    target_tie = random.random() < 0.45
    if not is_semantically_good(grid, analysis, target_tie):
        return None

    out = render_output_from_analysis(grid, analysis)
    if np.array_equal(grid, out):
        return None
    if not np.array_equal(main(grid), out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-partition instance")
