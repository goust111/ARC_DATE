from __future__ import annotations

import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 500
_RNG = random.Random()


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int)[grid != bg], return_counts=True)
    return int(values[np.argmax(counts)])


def _ray_length(grid: np.ndarray, wall: int, r: int, c: int, dr: int, dc: int) -> int:
    h, w = grid.shape
    length = 1
    rr, cc = r + dr, c + dc
    while 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != wall:
        length += 1
        rr += dr
        cc += dc
    return length


def _seed_span(grid: np.ndarray, wall: int, r: int, c: int) -> int:
    return min(_ray_length(grid, wall, r, c, dr, dc) for dr, dc in DIR4)


def _square_info(grid: np.ndarray, wall: int, r: int, c: int) -> tuple[int, list[tuple[int, int]], bool]:
    h, w = grid.shape
    dist = _seed_span(grid, wall, r, c)
    radius = dist - 1
    cells: list[tuple[int, int]] = []
    truncated = False
    for rr in range(max(0, r - radius), min(h, r + radius + 1)):
        for cc in range(max(0, c - radius), min(w, c + radius + 1)):
            if int(grid[rr, cc]) == wall:
                truncated = True
            else:
                cells.append((rr, cc))
    return dist, cells, truncated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed into the largest wall-limited centered square.

    Input structure:
    - Black background fills the grid.
    - One grey color forms scattered horizontal and vertical wall segments.
    - Other non-background pixels are isolated single-cell seeds of distinct colors.

    Rule:
    - For each seed, scan up, down, left, and right until a grey wall or the grid boundary stops the ray.
    - Let D be the shortest centered ray length, so the seed itself contributes length 1.
    - Repaint the seed as a solid square of side length 2D-1 centered on that seed.
    - Wall pixels are never overwritten; background and walls otherwise stay unchanged.

    Key insight:
    - The nearest blocking wall or edge sets the same allowable radius in every direction, so each seed becomes the biggest centered square that fits inside its local corridor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    non_bg = g[g != bg]
    assert non_bg.size > 0, "expected at least one wall or seed"
    wall = _wall_color(g, bg)

    out = g.copy()
    seed_positions = [tuple(map(int, pos)) for pos in np.argwhere((g != bg) & (g != wall))]
    for r, c in sorted(seed_positions):
        color = int(g[r, c])
        dist = _seed_span(g, wall, r, c)
        radius = dist - 1
        for rr in range(max(0, r - radius), min(g.shape[0], r + radius + 1)):
            for cc in range(max(0, c - radius), min(g.shape[1], c + radius + 1)):
                if int(g[rr, cc]) != wall:
                    out[rr, cc] = color

    return out


def _segment_cells(r0: int, c0: int, orient: str, length: int) -> list[tuple[int, int]]:
    if orient == "h":
        return [(r0, c0 + k) for k in range(length)]
    return [(r0 + k, c0) for k in range(length)]


def _can_place_wall(grid: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = grid.shape
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != BG:
            return False
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if (rr, cc) not in cell_set and int(grid[rr, cc]) == WALL:
                    return False
    return True


def _place_walls(grid: np.ndarray, rng: random.Random) -> bool:
    h, w = grid.shape
    target = rng.randint(3, 5 + int(min(h, w) >= 18))
    placed = 0
    for _ in range(900):
        if placed >= target:
            break
        orient = rng.choice(("h", "v"))
        max_len = w if orient == "h" else h
        if max_len < 3:
            return False
        length_cap = min(max_len, 9 if min(h, w) < 18 else 11)
        length = rng.randint(3, length_cap)
        if orient == "h":
            r0 = rng.randrange(h)
            c0 = rng.randrange(w - length + 1)
        else:
            r0 = rng.randrange(h - length + 1)
            c0 = rng.randrange(w)
        cells = _segment_cells(r0, c0, orient, length)
        if not _can_place_wall(grid, cells):
            continue
        for r, c in cells:
            grid[r, c] = WALL
        placed += 1
    return placed >= 3


def _seed_cell_is_clear(grid: np.ndarray, r: int, c: int) -> bool:
    if int(grid[r, c]) != BG:
        return False
    h, w = grid.shape
    for rr in range(max(0, r - 1), min(h, r + 2)):
        for cc in range(max(0, c - 1), min(w, c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) == WALL:
                return False
    return True


def _candidate_seeds(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    out: list[dict[str, object]] = []
    for r in range(h):
        for c in range(w):
            if not _seed_cell_is_clear(grid, r, c):
                continue
            dist, cells, truncated = _square_info(grid, WALL, r, c)
            out.append(
                {
                    "r": r,
                    "c": c,
                    "dist": dist,
                    "cells": cells,
                    "area": len(cells),
                    "truncated": truncated,
                    "edge": r in (0, h - 1) or c in (0, w - 1),
                }
            )
    return out


def _fits_candidate(
    candidate: dict[str, object],
    selected: list[dict[str, object]],
    occupied: set[tuple[int, int]],
) -> bool:
    r = int(candidate["r"])
    c = int(candidate["c"])
    for other in selected:
        if max(abs(r - int(other["r"])), abs(c - int(other["c"]))) <= 1:
            return False
    return all(cell not in occupied for cell in candidate["cells"])


def _claim_candidate(
    candidate: dict[str, object],
    selected: list[dict[str, object]],
    occupied: set[tuple[int, int]],
) -> None:
    selected.append(candidate)
    occupied.update(candidate["cells"])


def _pick_from_pool(
    pool: list[dict[str, object]],
    selected: list[dict[str, object]],
    occupied: set[tuple[int, int]],
    rng: random.Random,
) -> dict[str, object] | None:
    options = [cand for cand in pool if _fits_candidate(cand, selected, occupied)]
    if not options:
        return None
    used_distances = {int(cand["dist"]) for cand in selected}
    scored = []
    for cand in options:
        scored.append(
            (
                0 if int(cand["dist"]) not in used_distances else 1,
                0 if not bool(cand["edge"]) else 1,
                0 if not bool(cand["truncated"]) else 1,
                -int(cand["dist"]),
                -int(cand["area"]),
                rng.random(),
                cand,
            )
        )
    scored.sort(key=lambda item: item[:-1])
    top = [item[-1] for item in scored[: min(12, len(scored))]]
    return rng.choice(top)


def _preferred_subset(
    pool: list[dict[str, object]],
    want_edge: bool,
    want_truncated: bool,
) -> list[dict[str, object]]:
    preferred = list(pool)
    if not want_edge:
        non_edge = [cand for cand in preferred if not bool(cand["edge"])]
        if non_edge:
            preferred = non_edge
    if not want_truncated:
        clean = [cand for cand in preferred if not bool(cand["truncated"])]
        if clean:
            preferred = clean
    return preferred


def _select_seeds(
    candidates: list[dict[str, object]],
    target: int,
    want_edge: bool,
    want_truncated: bool,
    rng: random.Random,
) -> list[dict[str, object]] | None:
    edge_pool = [cand for cand in candidates if bool(cand["edge"])]
    truncated_pool = [cand for cand in candidates if bool(cand["truncated"])]
    clean_pool = [cand for cand in candidates if not bool(cand["truncated"])]
    non_edge_pool = [cand for cand in candidates if not bool(cand["edge"])]
    large_pool = [cand for cand in candidates if int(cand["dist"]) >= 3]
    want_edge = want_edge and bool(edge_pool)
    want_truncated = want_truncated and bool(truncated_pool)

    for _ in range(220):
        selected: list[dict[str, object]] = []
        occupied: set[tuple[int, int]] = set()

        if want_edge:
            cand = _pick_from_pool(edge_pool, selected, occupied, rng)
            if cand is None:
                continue
            _claim_candidate(cand, selected, occupied)

        if want_truncated and not any(bool(cand["truncated"]) for cand in selected):
            cand = _pick_from_pool(truncated_pool, selected, occupied, rng)
            if cand is None:
                continue
            _claim_candidate(cand, selected, occupied)

        if large_pool and not any(int(cand["dist"]) >= 3 for cand in selected):
            cand = _pick_from_pool(_preferred_subset(large_pool, want_edge, want_truncated), selected, occupied, rng)
            if cand is None:
                continue
            _claim_candidate(cand, selected, occupied)

        shuffled = _preferred_subset(candidates, want_edge, want_truncated)
        rng.shuffle(shuffled)
        while len(selected) < target:
            cand = _pick_from_pool(shuffled, selected, occupied, rng)
            if cand is None:
                break
            _claim_candidate(cand, selected, occupied)

        if len(selected) != target:
            continue
        if want_edge and not any(bool(cand["edge"]) for cand in selected):
            continue
        if want_truncated and not any(bool(cand["truncated"]) for cand in selected):
            continue
        if not want_edge and non_edge_pool and any(bool(cand["edge"]) for cand in selected):
            continue
        if not want_truncated and clean_pool and any(bool(cand["truncated"]) for cand in selected):
            continue
        if not any(int(cand["dist"]) >= 2 for cand in selected):
            continue
        return selected
    return None


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _validate_grid(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False

    bg = _dominant_color(g)
    non_bg = g[g != bg]
    if non_bg.size == 0:
        return False
    wall = _wall_color(g, bg)

    wall_comps = _components(g == wall)
    if len(wall_comps) < 3:
        return False
    for comp in wall_comps:
        if len(comp) < 3:
            return False
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) != 1 and len(cols) != 1:
            return False

    seeds = [tuple(map(int, pos)) for pos in np.argwhere((g != bg) & (g != wall))]
    if not (2 <= len(seeds) <= len(PALETTE)):
        return False
    if len({int(g[r, c]) for r, c in seeds}) != len(seeds):
        return False

    occupied: set[tuple[int, int]] = set()
    has_growth = False
    for r, c in seeds:
        for rr in range(max(0, r - 1), min(g.shape[0], r + 2)):
            for cc in range(max(0, c - 1), min(g.shape[1], c + 2)):
                if (rr, cc) == (r, c):
                    continue
                if int(g[rr, cc]) != bg:
                    return False
        dist, cells, _truncated = _square_info(g, wall, r, c)
        has_growth = has_growth or dist >= 2
        if any(cell in occupied for cell in cells):
            return False
        occupied.update(cells)

    return has_growth


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 24)
    w = rng.randint(12, 24)
    grid = np.full((h, w), BG, dtype=int)
    if not _place_walls(grid, rng):
        return None

    candidates = _candidate_seeds(grid)
    if len(candidates) < 2:
        return None

    target = rng.choice([2, 2, 3, 3, 4])
    want_edge = rng.random() < 0.5
    want_truncated = rng.random() < 0.5
    seeds = _select_seeds(candidates, target, want_edge, want_truncated, rng)
    if seeds is None:
        return None

    colors = list(PALETTE)
    rng.shuffle(colors)
    for color, seed in zip(colors, seeds):
        grid[int(seed["r"]), int(seed["c"])] = color

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(_RNG)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
