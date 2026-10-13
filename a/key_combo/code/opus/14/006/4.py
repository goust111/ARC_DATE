from __future__ import annotations

import random
from typing import Iterable

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SEED_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_seeds(grid: np.ndarray) -> list[tuple[int, int, int]]:
    pts = np.argwhere((grid != 0) & (grid != 5))
    seeds = [(int(r), int(c), int(grid[r, c])) for r, c in pts]
    seeds.sort(key=lambda item: (item[0], item[1], item[2]))
    return seeds


def _compute_output_and_info(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seeds = _find_seeds(g)
    seed_pos = {(r, c) for r, c, _color in seeds}
    arrivals: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}

    for sr, sc, color in seeds:
        for dr, dc in DIRS4:
            r, c = sr + dr, sc + dc
            dist = 1
            while _in_bounds((h, w), r, c):
                if g[r, c] == 5 or (r, c) in seed_pos:
                    break
                arrivals.setdefault((r, c), []).append((dist, sr, sc, color))
                r += dr
                c += dc
                dist += 1

    out = g.copy()
    multi_cells: list[tuple[int, int]] = []
    unique_shortest_multi = False
    row_priority_tie = False
    col_priority_tie = False

    for (r, c), claims in arrivals.items():
        if len(claims) > 1:
            multi_cells.append((r, c))
        claims_sorted = sorted(claims, key=lambda item: (item[0], item[1], item[2]))
        min_dist = claims_sorted[0][0]
        tied = [item for item in claims_sorted if item[0] == min_dist]
        if len(claims) > 1 and len(tied) == 1:
            unique_shortest_multi = True
        if len(tied) >= 2:
            rows = {item[1] for item in tied}
            cols = {item[2] for item in tied}
            if len(rows) >= 2:
                row_priority_tie = True
            elif len(cols) >= 2:
                col_priority_tie = True
        out[r, c] = claims_sorted[0][3]

    info = {
        "seeds": seeds,
        "arrivals": arrivals,
        "multi_cells": multi_cells,
        "multi_count": len(multi_cells),
        "unique_shortest_multi": unique_shortest_multi,
        "row_priority_tie": row_priority_tie,
        "col_priority_tie": col_priority_tie,
    }
    return out, info


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast four-direction rays from all colored seeds, stop them at obstacles, and color crossings by shortest arrival distance.

    Input structure:
    - A black background grid contains several colored seed pixels and a few grey obstacle bars.
    - Every non-black, non-grey pixel is a seed; each seed is a single cell located away from the border.
    - Grey cells form short 1x2 or 2x1 obstacle segments that block rays without being overwritten.

    Rule:
    - From each seed, emit rays upward, downward, leftward, and rightward using the seed's own color.
    - A ray advances cell by cell and stops before leaving the grid, before entering a grey obstacle cell, and before entering another seed cell.
    - For every non-seed cell reached by one or more rays, compare all arrivals by path length from the seed; the shortest arrival wins.
    - If multiple arrivals tie on path length, prefer the one whose seed has the smaller row, then the smaller column; keep original seeds and grey obstacles unchanged.

    Key insight:
    - The visible color at a crossing is determined jointly by obstacle-created shadows and by a lexicographic comparison of the shortest seed-to-cell ray distances.
    """
    out, _info = _compute_output_and_info(np.asarray(grid, dtype=int))
    return out


def _seed_clear_of_bar(seed: tuple[int, int], bar: Iterable[tuple[int, int]]) -> bool:
    sr, sc = seed
    for r, c in bar:
        if max(abs(sr - r), abs(sc - c)) <= 1:
            return False
    return True


def _bar_is_valid(
    shape: tuple[int, int],
    seeds: list[tuple[int, int]],
    bar: tuple[tuple[int, int], tuple[int, int]],
    existing: list[tuple[tuple[int, int], tuple[int, int]]],
) -> bool:
    cells = list(bar)
    if len(set(cells)) != 2:
        return False
    for r, c in cells:
        if not _in_bounds(shape, r, c):
            return False
    dr = abs(cells[0][0] - cells[1][0])
    dc = abs(cells[0][1] - cells[1][1])
    if (dr, dc) not in {(1, 0), (0, 1)}:
        return False
    seed_set = set(seeds)
    if any(cell in seed_set for cell in cells):
        return False
    if any(not _seed_clear_of_bar(seed, cells) for seed in seeds):
        return False
    occupied = set()
    for other in existing:
        occupied.update(other)
    if any(cell in occupied for cell in cells):
        return False
    for r, c in cells:
        for rr, cc in occupied:
            if abs(r - rr) + abs(c - cc) == 1:
                return False
    return True


def _add_bar(grid: np.ndarray, bar: tuple[tuple[int, int], tuple[int, int]]) -> None:
    for r, c in bar:
        grid[r, c] = 5


def _candidate_bars_from_arrivals(
    shape: tuple[int, int],
    seeds: list[tuple[int, int]],
    arrivals: dict[tuple[int, int], list[tuple[int, int, int, int]]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    bars: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in arrivals:
        for bar in [
            ((r, c), (r + 1, c)),
            ((r - 1, c), (r, c)),
            ((r, c), (r, c + 1)),
            ((r, c - 1), (r, c)),
        ]:
            ordered = tuple(sorted(bar))
            if _bar_is_valid(shape, seeds, ordered, []):
                bars.add(ordered)
    result = list(bars)
    random.shuffle(result)
    return result


def _random_bar_candidates(
    shape: tuple[int, int],
    seeds: list[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    h, w = shape
    bars: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            for bar in [((r, c), (r + 1, c)), ((r, c), (r, c + 1))]:
                ordered = tuple(sorted(bar))
                if _bar_is_valid(shape, seeds, ordered, []):
                    bars.append(ordered)
    random.shuffle(bars)
    return bars


def _grid_from_seeds(
    shape: tuple[int, int],
    seeds: list[tuple[int, int, int]],
    bars: list[tuple[tuple[int, int], tuple[int, int]]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r, c, color in seeds:
        grid[r, c] = color
    for bar in bars:
        _add_bar(grid, bar)
    return grid


def _sample_distinct_positions(h: int, w: int) -> list[tuple[int, int]] | None:
    rows = [r for r in range(2, h - 2)]
    cols = [c for c in range(2, w - 2)]
    for _ in range(80):
        rs = sorted(random.sample(rows, 3))
        cs = random.sample(cols, 3)
        pts = list(zip(rs, cs))
        if len({r for r, _c in pts}) < 3 or len({c for _r, c in pts}) < 3:
            continue
        if min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for i, a in enumerate(pts) for b in pts[i + 1 :]) < 4:
            continue
        return pts
    return None


def _sample_row_priority_positions(h: int, w: int) -> list[tuple[int, int]] | None:
    if h < 11 or w < 11:
        return None
    for _ in range(80):
        c = random.randint(3, w - 4)
        d = random.randint(2, min(3, (h - 5) // 2))
        rmid = random.randint(2 + d, h - 3 - d)
        r1, r2 = rmid - d, rmid + d
        side = random.choice([-1, 1])
        max_far = min(4, (w - 3 - c) if side == 1 else (c - 2))
        if max_far < d + 1:
            continue
        far = random.randint(d + 1, max_far)
        c3 = c + side * far
        if not (2 <= c3 <= w - 3):
            continue
        pts = [(r1, c), (r2, c), (rmid, c3)]
        if min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for i, a in enumerate(pts) for b in pts[i + 1 :]) < 4:
            continue
        return pts
    return None


def _sample_col_priority_positions(h: int, w: int) -> list[tuple[int, int]] | None:
    if h < 11 or w < 11:
        return None
    for _ in range(80):
        r = random.randint(3, h - 4)
        d = random.randint(2, min(3, (w - 5) // 2))
        cmid = random.randint(2 + d, w - 3 - d)
        c1, c2 = cmid - d, cmid + d
        side = random.choice([-1, 1])
        max_far = min(4, (h - 3 - r) if side == 1 else (r - 2))
        if max_far < d + 1:
            continue
        far = random.randint(d + 1, max_far)
        r3 = r + side * far
        if not (2 <= r3 <= h - 3):
            continue
        pts = [(r, c1), (r, c2), (r3, cmid)]
        if min(abs(a[0] - b[0]) + abs(a[1] - b[1]) for i, a in enumerate(pts) for b in pts[i + 1 :]) < 4:
            continue
        return pts
    return None


def _sample_seed_layout() -> tuple[list[tuple[int, int, int]], str, tuple[int, int]] | None:
    h = random.randint(11, 17)
    w = random.randint(11, 17)
    mode = random.choices(
        ["distinct", "row_priority", "col_priority"],
        weights=[4, 3, 3],
        k=1,
    )[0]

    if mode == "distinct":
        pts = _sample_distinct_positions(h, w)
    elif mode == "row_priority":
        pts = _sample_row_priority_positions(h, w)
    else:
        pts = _sample_col_priority_positions(h, w)
    if pts is None:
        return None

    colors = random.sample(SEED_COLORS, 3)
    seeds = [(r, c, color) for (r, c), color in zip(pts, colors)]
    return seeds, mode, (h, w)


def _obstacle_components_ok(grid: np.ndarray) -> bool:
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if grid[r, c] != 5 or (r, c) in seen:
                continue
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and grid[nr, nc] == 5 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(comp)
    if len(comps) not in {2, 3}:
        return False
    for comp in comps:
        if len(comp) != 2:
            return False
        dr = abs(comp[0][0] - comp[1][0])
        dc = abs(comp[0][1] - comp[1][1])
        if (dr, dc) not in {(1, 0), (0, 1)}:
            return False
    return True


def _passes_mode_requirements(mode: str, info: dict) -> bool:
    if info["multi_count"] < 2:
        return False
    if mode == "distinct":
        return bool(info["unique_shortest_multi"])
    if mode == "row_priority":
        return bool(info["row_priority_tie"])
    return bool(info["col_priority_tie"])


def _generate_attempt() -> np.ndarray | None:
    sample = _sample_seed_layout()
    if sample is None:
        return None
    seeds, mode, shape = sample
    base_grid = _grid_from_seeds(shape, seeds, [])
    base_out, base_info = _compute_output_and_info(base_grid)
    if not _passes_mode_requirements(mode, base_info):
        return None

    seed_rc = [(r, c) for r, c, _color in seeds]
    targeted = _candidate_bars_from_arrivals(shape, seed_rc, base_info["arrivals"])
    fallback = _random_bar_candidates(shape, seed_rc)
    all_candidates = targeted + [bar for bar in fallback if bar not in set(targeted)]
    target_count = random.choice([2, 3])
    random.shuffle(all_candidates)

    chosen: list[tuple[tuple[int, int], tuple[int, int]]] = []
    current_grid = base_grid.copy()
    current_info = base_info
    changed = False

    for idx in range(target_count):
        found = False
        random.shuffle(all_candidates)
        for bar in all_candidates:
            if not _bar_is_valid(shape, seed_rc, bar, chosen):
                continue
            trial_bars = chosen + [bar]
            trial_grid = _grid_from_seeds(shape, seeds, trial_bars)
            trial_out, trial_info = _compute_output_and_info(trial_grid)
            if trial_info["multi_count"] < 2:
                continue
            if idx == 0 and np.array_equal(trial_out, base_out):
                continue
            if idx == target_count - 1 and not _passes_mode_requirements(mode, trial_info):
                continue
            chosen = trial_bars
            current_grid = trial_grid
            current_info = trial_info
            if not np.array_equal(trial_out, base_out):
                changed = True
            found = True
            break
        if not found:
            return None

    if not changed:
        return None
    if not _obstacle_components_ok(current_grid):
        return None
    if not _passes_mode_requirements(mode, current_info):
        return None
    return current_grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid multi-source ray puzzle input")
