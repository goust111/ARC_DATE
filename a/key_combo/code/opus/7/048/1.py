from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BACKGROUND = 0
PALETTE = (1, 2, 3, 4)
SIZE_OPTIONS = (11, 13, 15, 17)
MAX_GENERATION_ATTEMPTS = 400


def _boundary_scan(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    order: list[tuple[int, int]] = []
    for c in range(w):
        order.append((0, c))
    for r in range(1, h):
        order.append((r, w - 1))
    for c in range(w - 2, -1, -1):
        order.append((h - 1, c))
    for r in range(h - 2, 0, -1):
        order.append((r, 0))
    return order


def _initial_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    if r == 0 and c == 0:
        return 1, 1
    if r == 0 and c == w - 1:
        return 1, -1
    if r == h - 1 and c == 0:
        return -1, 1
    if r == h - 1 and c == w - 1:
        return -1, -1
    if r == 0:
        return 1, 1
    if r == h - 1:
        return -1, 1
    if c == 0:
        return 1, 1
    if c == w - 1:
        return 1, -1
    raise AssertionError("start must lie on the boundary")


def _advance_with_reflection(
    shape: tuple[int, int], r: int, c: int, dr: int, dc: int
) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    nr = r + dr
    nc = c + dc
    if nr < 0 or nr >= h:
        dr = -dr
        nr = r + dr
    if nc < 0 or nc >= w:
        dc = -dc
        nc = c + dc
    return nr, nc, dr, dc


def _trace_rays(
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int]], dict[int, int], list[int]]:
    g = np.asarray(grid, dtype=int).copy()
    order = _boundary_scan(g.shape)
    starts = [pos for pos in order if int(g[pos]) != BACKGROUND]
    first_rank_by_color: dict[int, int] = {}
    ray_lengths: list[int] = []

    for idx, (sr, sc) in enumerate(starts):
        color = int(g[sr, sc])
        first_rank_by_color.setdefault(color, idx)
        r, c = sr, sc
        dr, dc = _initial_direction(g.shape, sr, sc)
        painted = 0

        while True:
            nr, nc, dr, dc = _advance_with_reflection(g.shape, r, c, dr, dc)
            if (nr, nc) == (sr, sc):
                break
            if int(g[nr, nc]) != BACKGROUND:
                break
            g[nr, nc] = color
            r, c = nr, nc
            painted += 1

        ray_lengths.append(painted)

    return g, starts, first_rank_by_color, ray_lengths


def _exterior_background(mask_grid: np.ndarray) -> np.ndarray:
    h, w = map(int, mask_grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r, c in _boundary_scan(mask_grid.shape):
        if int(mask_grid[r, c]) == BACKGROUND and not seen[r, c]:
            seen[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(mask_grid[nr, nc]) == BACKGROUND:
                seen[nr, nc] = True
                q.append((nr, nc))

    return seen


def _closed_regions(
    wall_grid: np.ndarray,
) -> list[dict[str, object]]:
    h, w = map(int, wall_grid.shape)
    exterior = _exterior_background(wall_grid)
    enclosed = (wall_grid == BACKGROUND) & ~exterior
    seen = np.zeros((h, w), dtype=bool)
    regions: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if not enclosed[r, c] or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            border_counts: Counter[int] = Counter()

            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if enclosed[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                    elif int(wall_grid[nr, nc]) != BACKGROUND:
                        border_counts[int(wall_grid[nr, nc])] += 1

            if border_counts:
                regions.append({"cells": cells, "border_counts": border_counts})

    return regions


def _choose_fill_color(border_counts: Counter[int], first_rank_by_color: dict[int, int]) -> int:
    return min(
        border_counts,
        key=lambda color: (-int(border_counts[color]), int(first_rank_by_color[color])),
    )


def _solve(grid: np.ndarray) -> tuple[np.ndarray, dict[str, object]]:
    ray_grid, starts, first_rank_by_color, ray_lengths = _trace_rays(grid)
    regions = _closed_regions(ray_grid)

    out = ray_grid.copy()
    analyzed_regions: list[dict[str, object]] = []
    for region in regions:
        border_counts = region["border_counts"]
        fill_color = _choose_fill_color(border_counts, first_rank_by_color)
        for r, c in region["cells"]:
            out[r, c] = fill_color

        max_count = max(int(v) for v in border_counts.values())
        tied = [color for color, count in border_counts.items() if int(count) == max_count]
        analyzed_regions.append(
            {
                "size": len(region["cells"]),
                "border_counts": dict(border_counts),
                "fill_color": fill_color,
                "tie": len(tied) > 1,
            }
        )

    meta = {
        "starts": starts,
        "ray_grid": ray_grid,
        "ray_lengths": ray_lengths,
        "regions": analyzed_regions,
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each boundary seed into a bouncing diagonal wall network, then fill every enclosed black region by the majority boundary color.

    Input structure:
    - The grid is square with a black background and 2-4 colored seed pixels placed on the outer boundary.
    - Seeds use 2-3 colors from the ARC palette {blue, red, green, yellow}, and the interior starts empty.
    - Each seed launches its own diagonal path; later paths may stop when they hit earlier painted path cells.

    Rule:
    - Scan the boundary clockwise from the top-left corner and process seed pixels in that order.
    - From each seed, move one step at a time along its prescribed diagonal direction, reflecting off the grid sides and painting only black cells; stop before re-entering the seed or stepping onto any non-black cell.
    - After all paths are drawn, treat the painted path cells as walls, find every black 4-connected region that cannot reach a black boundary cell, and fill each such region with the color that appears most often on its 4-neighbor wall boundary.
    - If a region's boundary color counts tie, choose the color whose seed appeared earlier in the clockwise boundary scan.

    Key insight:
    - The diagonal rays are not the goal object themselves; they create a colored wall network, and each enclosed background pocket inherits its fill color from the dominant wall color around that pocket.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == g.shape[1], "expected a square grid"
    assert int(np.min(g)) >= 0
    assert set(map(int, np.unique(g))).issubset({0, 1, 2, 3, 4}), "unexpected colors"

    interior = g[1:-1, 1:-1]
    assert not np.any(interior != BACKGROUND), "interior should be empty before ray tracing"

    starts = [pos for pos in _boundary_scan(g.shape) if int(g[pos]) != BACKGROUND]
    assert 2 <= len(starts) <= 4, "expected 2-4 boundary seeds"

    start_colors = [int(g[pos]) for pos in starts]
    counts = Counter(start_colors)
    assert 2 <= len(counts) <= 3, "expected 2-3 seed colors"
    assert all(1 <= count <= 2 for count in counts.values()), "each color must appear once or twice"

    out, _meta = _solve(g)
    return out


def _cyclic_gaps(indices: list[int], perimeter: int) -> list[int]:
    if not indices:
        return []
    extended = indices + [indices[0] + perimeter]
    return [extended[i + 1] - extended[i] for i in range(len(indices))]


def _sample_start_positions(n: int, num_starts: int, allow_corner: bool) -> list[tuple[int, int]] | None:
    boundary = _boundary_scan((n, n))
    perimeter = len(boundary)
    min_gap = 3 if n >= 13 else 2
    corner_set = {(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)}

    for _ in range(120):
        positions = random.sample(boundary, num_starts)
        corner_count = sum(pos in corner_set for pos in positions)
        if corner_count > 1:
            continue
        if not allow_corner and corner_count:
            continue

        indices = sorted(boundary.index(pos) for pos in positions)
        if min(_cyclic_gaps(indices, perimeter)) < min_gap:
            continue

        side_count = len(
            {
                "top"
                if r == 0
                else "bottom"
                if r == n - 1
                else "left"
                if c == 0
                else "right"
                for r, c in positions
            }
        )
        if side_count < 2:
            continue

        return positions

    return None


def _sample_color_multiset(num_starts: int) -> list[int]:
    if num_starts == 2:
        colors = random.sample(PALETTE, 2)
        return colors

    if num_starts == 3:
        if random.random() < 0.55:
            colors = random.sample(PALETTE, 3)
            return colors
        a, b = random.sample(PALETTE, 2)
        colors = [a, a, b]
        random.shuffle(colors)
        return colors

    if random.random() < 0.45:
        a, b = random.sample(PALETTE, 2)
        colors = [a, a, b, b]
    else:
        a, b, c = random.sample(PALETTE, 3)
        colors = [a, a, b, c]
    random.shuffle(colors)
    return colors


def _valid_input(grid: np.ndarray, require_tie: bool) -> bool:
    out, meta = _solve(grid)
    regions = meta["regions"]
    if not regions:
        return False

    region_sizes = [int(region["size"]) for region in regions]
    filled_total = int(sum(region_sizes))
    if filled_total < max(4, grid.shape[0] // 2):
        return False
    if max(region_sizes) < max(3, filled_total // 2):
        return False

    ray_lengths = meta["ray_lengths"]
    if min(ray_lengths) < 2:
        return False

    ray_grid = meta["ray_grid"]
    if np.count_nonzero(ray_grid) > int(ray_grid.size * 0.42):
        return False

    if np.array_equal(grid, out):
        return False

    tie_present = any(bool(region["tie"]) for region in regions)
    if require_tie and not tie_present:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n = random.choice(SIZE_OPTIONS)
        num_starts = random.choices((2, 3, 4), weights=(1, 3, 4), k=1)[0]
        allow_corner = random.random() < 0.25
        positions = _sample_start_positions(n, num_starts, allow_corner)
        if positions is None:
            continue

        colors = _sample_color_multiset(num_starts)
        grid = np.zeros((n, n), dtype=int)
        for (r, c), color in zip(positions, colors):
            grid[r, c] = color

        require_tie = random.random() < 0.2
        if _valid_input(grid, require_tie=require_tie):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 1")
