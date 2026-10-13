from __future__ import annotations

import random

import numpy as np


BG = 0
DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SCENARIOS = ("clear", "count_block", "tie_block")
MAX_GENERATION_ATTEMPTS = 500


def _seed_positions(grid: np.ndarray) -> dict[int, list[tuple[int, int]]]:
    seeds: dict[int, list[tuple[int, int]]] = {}
    for r, c in np.argwhere(grid != BG):
        color = int(grid[r, c])
        seeds.setdefault(color, []).append((int(r), int(c)))
    for coords in seeds.values():
        coords.sort()
    return seeds


def _color_order(seeds: dict[int, list[tuple[int, int]]]) -> list[int]:
    return sorted(seeds, key=lambda color: (len(seeds[color]), seeds[color][0][0], seeds[color][0][1]))


def _simulate(grid: np.ndarray, *, return_stats: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    height, width = g.shape
    seeds = _seed_positions(g)
    order = _color_order(seeds)
    out = g.copy()

    stats = {
        "counts": {color: len(coords) for color, coords in seeds.items()},
        "first_positions": {color: coords[0] for color, coords in seeds.items()},
        "order": order,
        "painted_cells": 0,
        "occupied_stops": 0,
        "boundary_stops": 0,
        "shared_step_hits": 0,
        "total_rays": sum(4 * len(coords) for coords in seeds.values()),
    }

    for color in order:
        grow_len = len(seeds[color])
        rays = [(r, c, dr, dc) for r, c in seeds[color] for dr, dc in DIAGONALS]
        active = [True] * len(rays)

        for step in range(1, grow_len + 1):
            proposals: dict[tuple[int, int], list[int]] = {}
            next_active = [False] * len(rays)

            for ray_idx, (r0, c0, dr, dc) in enumerate(rays):
                if not active[ray_idx]:
                    continue

                rr = int(r0 + dr * step)
                cc = int(c0 + dc * step)
                if not (0 <= rr < height and 0 <= cc < width):
                    stats["boundary_stops"] += 1
                    continue
                if int(out[rr, cc]) != BG:
                    stats["occupied_stops"] += 1
                    continue

                proposals.setdefault((rr, cc), []).append(ray_idx)
                next_active[ray_idx] = True

            for (rr, cc), proposers in proposals.items():
                out[rr, cc] = int(color)
                if len(proposers) > 1:
                    stats["shared_step_hits"] += 1
            stats["painted_cells"] += len(proposals)
            active = next_active

    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Grow each seed along four diagonals by its color count, with earlier color groups blocking later ones.

    Input structure:
    - The background is black.
    - Several colors appear only as isolated one-pixel seeds scattered on the grid.
    - Each color occurs between one and four times, and there are two to four seed colors.

    Rule:
    - Count how many seeds each color has in the full input.
    - Process colors from smaller seed-count to larger seed-count; if counts tie,
      use the color whose first seed appears earlier in row-major order.
    - For every seed of the current color, extend one step at a time in the four
      diagonal directions for exactly that color count many steps.
    - If the next diagonal cell is outside the grid or is already non-black
      because of a seed or previously grown path, stop only that ray and never overwrite.
    - Keep the original seed pixels unchanged.

    Key insight:
    - The diagonal length is a global property of each color class, and the
      output also depends on the inter-color processing order because earlier
      growth can occupy cells that later colors would otherwise reach.
    """
    return _simulate(grid)


def _lex_less(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a < b


def _lex_greater(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a > b


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _sample_counts(n_colors: int, scenario: str) -> tuple[list[int], int | None, int | None]:
    counts = [random.randint(1, 4) for _ in range(n_colors)]
    blocker_idx = blocked_idx = None

    if scenario == "count_block":
        blocker_idx, blocked_idx = random.sample(range(n_colors), 2)
        blocker_count = random.randint(1, 3)
        blocked_count = random.randint(blocker_count + 1, 4)
        counts[blocker_idx] = blocker_count
        counts[blocked_idx] = blocked_count
    elif scenario == "tie_block":
        blocker_idx, blocked_idx = random.sample(range(n_colors), 2)
        tie_count = random.randint(1, 4)
        counts[blocker_idx] = tie_count
        counts[blocked_idx] = tie_count

    if max(counts) == 1:
        counts[random.randrange(n_colors)] = 2

    return counts, blocker_idx, blocked_idx


def _grid_shape(total_seeds: int) -> tuple[int, int]:
    base = max(12, total_seeds + 8)
    top = min(30, base + 6)
    return random.randint(base, top), random.randint(base, top)


def _sample_cell(
    height: int,
    width: int,
    occupied: set[tuple[int, int]],
    *,
    min_gap: int,
    predicate=None,
) -> tuple[int, int] | None:
    for _ in range(250):
        if random.random() < 0.2:
            side = random.choice(("top", "bottom", "left", "right"))
            if side == "top":
                cell = (0, random.randint(0, width - 1))
            elif side == "bottom":
                cell = (height - 1, random.randint(0, width - 1))
            elif side == "left":
                cell = (random.randint(0, height - 1), 0)
            else:
                cell = (random.randint(0, height - 1), width - 1)
        else:
            cell = (random.randint(0, height - 1), random.randint(0, width - 1))

        if cell in occupied:
            continue
        if predicate is not None and not predicate(cell):
            continue
        if all(_chebyshev(cell, other) >= min_gap for other in occupied):
            return cell
    return None


def _motif_anchor(height: int, width: int, downward_only: bool) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if downward_only:
        dr, dc = random.choice([(1, 1), (1, -1)])
    else:
        dr, dc = random.choice(DIAGONALS)

    row_lo = 0 if dr > 0 else 2
    row_hi = height - 3 if dr > 0 else height - 1
    col_lo = 0 if dc > 0 else 2
    col_hi = width - 3 if dc > 0 else width - 1

    a = (random.randint(row_lo, row_hi), random.randint(col_lo, col_hi))
    middle = (a[0] + dr, a[1] + dc)
    b = (a[0] + 2 * dr, a[1] + 2 * dc)
    return a, middle, b


def _build_candidate() -> tuple[np.ndarray, dict[str, object]]:
    scenario = random.choice(SCENARIOS)
    n_colors = random.randint(2, 4)
    counts, blocker_idx, blocked_idx = _sample_counts(n_colors, scenario)
    total_seeds = sum(counts)
    height, width = _grid_shape(total_seeds)
    min_gap = 4 if total_seeds <= 9 else 3

    color_ids = random.sample(COLOR_POOL, n_colors)
    coords_by_idx: list[list[tuple[int, int]]] = [[] for _ in range(n_colors)]
    occupied: set[tuple[int, int]] = set()
    motif = None

    if scenario in {"count_block", "tie_block"}:
        assert blocker_idx is not None and blocked_idx is not None
        anchor, middle, target = _motif_anchor(height, width, scenario == "tie_block")
        coords_by_idx[blocker_idx].append(anchor)
        coords_by_idx[blocked_idx].append(target)
        occupied.add(anchor)
        occupied.add(target)
        motif = {
            "blocker_color": color_ids[blocker_idx],
            "blocked_color": color_ids[blocked_idx],
            "middle": middle,
            "anchor": anchor,
            "target": target,
        }

    for color_idx, count in enumerate(counts):
        while len(coords_by_idx[color_idx]) < count:
            predicate = None
            if scenario == "tie_block" and color_idx == blocked_idx and motif is not None:
                anchor = motif["anchor"]
                predicate = lambda cell, anchor=anchor: _lex_greater(cell, anchor)

            cell = _sample_cell(height, width, occupied, min_gap=min_gap, predicate=predicate)
            if cell is None:
                raise ValueError("placement failed")
            coords_by_idx[color_idx].append(cell)
            occupied.add(cell)

    grid = np.zeros((height, width), dtype=int)
    for color_idx, coords in enumerate(coords_by_idx):
        color = color_ids[color_idx]
        for r, c in coords:
            grid[r, c] = color

    meta = {
        "scenario": scenario,
        "motif": motif,
        "color_ids": color_ids,
        "counts": counts,
    }
    return grid, meta


def _valid_sample(grid: np.ndarray, meta: dict[str, object]) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    colors = [int(color) for color in np.unique(grid) if int(color) != BG]
    if not (2 <= len(colors) <= 4):
        return False

    if int(np.count_nonzero(grid)) < 3:
        return False

    out, stats = _simulate(grid, return_stats=True)
    if np.array_equal(out, grid):
        return False
    if stats["painted_cells"] < 4:
        return False
    if stats["occupied_stops"] > stats["total_rays"] // 4 + 2:
        return False

    scenario = meta["scenario"]
    motif = meta["motif"]

    if scenario == "clear":
        if stats["occupied_stops"] != 0:
            return False
    else:
        if stats["occupied_stops"] < 1 or motif is None:
            return False

        blocker = int(motif["blocker_color"])
        blocked = int(motif["blocked_color"])
        middle = tuple(motif["middle"])

        if int(grid[middle]) != BG or int(out[middle]) != blocker:
            return False

        if scenario == "count_block":
            if stats["counts"][blocker] >= stats["counts"][blocked]:
                return False
        else:
            if stats["counts"][blocker] != stats["counts"][blocked]:
                return False
            if not _lex_less(stats["first_positions"][blocker], stats["first_positions"][blocked]):
                return False
            if len(set(stats["counts"].values())) == len(stats["counts"]):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid, meta = _build_candidate()
        except ValueError:
            continue
        if _valid_sample(grid, meta):
            return grid
    raise RuntimeError("failed to generate valid diagonal-growth input")
