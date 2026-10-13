from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 8
YELLOW = 4
MAROON = 9
ORANGE = 7
REGION_COLORS = [0, 1, 2, 3, 5, 6]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (15, 22)
REGION_COUNT_RANGE = (2, 4)
REGION_BBOX_RANGE = (4, 7)
REGION_AREA_MIN = 8
MAX_COUNTER = 4
GEN_ATTEMPTS = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _same_color_component(
    grid: np.ndarray, start: tuple[int, int], color: int
) -> set[tuple[int, int]]:
    h, w = grid.shape
    q = deque([start])
    comp = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (
                _in_bounds(h, w, nr, nc)
                and (nr, nc) not in comp
                and int(grid[nr, nc]) == color
            ):
                comp.add((nr, nc))
                q.append((nr, nc))
    return comp


def _extract_sources(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sources: list[dict[str, object]] = []
    special = {BG, YELLOW, MAROON, ORANGE}

    for yr, yc in sorted(map(tuple, np.argwhere(g == YELLOW))):
        candidates = []
        for dr, dc in DIR4:
            rr, cc = yr + dr, yc + dc
            mr, mc = yr - dr, yc - dc
            if not (_in_bounds(h, w, rr, cc) and _in_bounds(h, w, mr, mc)):
                continue
            color = int(g[rr, cc])
            if color in special or int(g[mr, mc]) != MAROON:
                continue

            layers = 0
            tr, tc = mr, mc
            while _in_bounds(h, w, tr, tc) and int(g[tr, tc]) == MAROON:
                layers += 1
                tr -= dr
                tc -= dc

            region = _same_color_component(g, (rr, cc), color)
            candidates.append(
                {
                    "marker": (int(yr), int(yc)),
                    "color": color,
                    "layers": layers,
                    "region": region,
                }
            )

        if len(candidates) == 1:
            sources.append(candidates[0])

    sources.sort(key=lambda item: item["marker"])  # deterministic tie-break
    return sources


def _solve_with_fill(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    base = g.copy()
    base[(base == YELLOW) | (base == MAROON)] = BG

    best_dist = np.full((h, w), 10**9, dtype=int)
    best_order = np.full((h, w), 10**9, dtype=int)
    best_color = np.full((h, w), -1, dtype=int)

    for order, source in enumerate(_extract_sources(g)):
        region = set(source["region"])
        color = int(source["color"])
        limit = int(source["layers"])
        q = deque((r, c, 0) for r, c in region)
        seen = set(region)

        while q:
            r, c, dist = q.popleft()
            if dist >= limit:
                continue

            nd = dist + 1
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if not _in_bounds(h, w, nr, nc) or (nr, nc) in seen:
                    continue
                if int(base[nr, nc]) != BG:
                    continue

                seen.add((nr, nc))
                q.append((nr, nc, nd))
                if nd < best_dist[nr, nc] or (
                    nd == best_dist[nr, nc] and order < best_order[nr, nc]
                ):
                    best_dist[nr, nc] = nd
                    best_order[nr, nc] = order
                    best_color[nr, nc] = color

    out = base.copy()
    mask = best_color != -1
    out[mask] = best_color[mask]
    filled = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    return out, filled


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_fill(grid)[0]


def _random_region_mask(rng: random.Random) -> np.ndarray | None:
    bh = rng.randint(*REGION_BBOX_RANGE)
    bw = rng.randint(*REGION_BBOX_RANGE)
    area_max = bh * bw - 2
    if area_max < REGION_AREA_MIN:
        return None

    target = rng.randint(REGION_AREA_MIN, area_max)
    seed = (rng.randrange(bh), rng.randrange(bw))
    cells = {seed}

    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < bh and 0 <= nc < bw and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None

        weights = []
        for nr, nc in frontier:
            touch = sum((nr + dr, nc + dc) in cells for dr, dc in DIR4)
            weights.append(1 + touch)
        pick = frontier[rng.choices(range(len(frontier)), weights=weights, k=1)[0]]
        cells.add(pick)

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return None
    if len(cells) >= (r1 - r0 + 1) * (c1 - c0 + 1):
        return None

    mask = np.zeros((bh, bw), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _cells_clear(grid: np.ndarray, cells: set[tuple[int, int]], pad: int) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not _in_bounds(h, w, r, c) or int(grid[r, c]) != BG:
            return False
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if not _in_bounds(h, w, rr, cc) or (rr, cc) in cells:
                    continue
                if int(grid[rr, cc]) != BG:
                    return False
    return True


def _place_region(
    grid: np.ndarray, mask: np.ndarray, color: int, rng: random.Random
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    bh, bw = mask.shape
    row_lo = 2
    row_hi = h - bh - 2
    col_lo = 2
    col_hi = w - bw - 2
    if row_lo > row_hi or col_lo > col_hi:
        return None

    local_cells = {tuple(map(int, rc)) for rc in np.argwhere(mask)}
    for _ in range(80):
        top = rng.randint(row_lo, row_hi)
        left = rng.randint(col_lo, col_hi)
        cells = {(top + r, left + c) for r, c in local_cells}
        if not _cells_clear(grid, cells, pad=1):
            continue
        for r, c in cells:
            grid[r, c] = color
        return cells
    return None


def _marker_candidates(
    grid: np.ndarray, region: set[tuple[int, int]]
) -> list[dict[str, object]]:
    h, w = grid.shape
    candidates: list[dict[str, object]] = []
    region_set = set(region)

    for r, c in region_set:
        for dr, dc in DIR4:
            yr, yc = r + dr, c + dc
            if not _in_bounds(h, w, yr, yc) or int(grid[yr, yc]) != BG:
                continue

            region_neighbors = 0
            other_non_bg = 0
            for sr, sc in ((yr - 1, yc), (yr + 1, yc), (yr, yc - 1), (yr, yc + 1)):
                if not _in_bounds(h, w, sr, sc):
                    continue
                if (sr, sc) in region_set:
                    region_neighbors += 1
                elif int(grid[sr, sc]) != BG:
                    other_non_bg += 1
            if region_neighbors != 1 or other_non_bg != 0:
                continue

            line: list[tuple[int, int]] = []
            tr, tc = yr + dr, yc + dc
            while (
                len(line) < MAX_COUNTER
                and _in_bounds(h, w, tr, tc)
                and int(grid[tr, tc]) == BG
            ):
                line.append((tr, tc))
                tr += dr
                tc += dc
            if not line:
                continue

            candidates.append({"yellow": (yr, yc), "dir": (dr, dc), "line": line})

    return candidates


def _choose_counter_length(max_len: int, rng: random.Random) -> int:
    choices = list(range(1, min(MAX_COUNTER, max_len) + 1))
    short = [n for n in choices if n <= 2]
    long = [n for n in choices if n >= 3]
    if short and long:
        pool = short if rng.random() < 0.5 else long
        return rng.choice(pool)
    return rng.choice(choices)


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _filled_positions_without_barriers(grid: np.ndarray) -> set[tuple[int, int]]:
    probe = np.asarray(grid, dtype=int).copy()
    probe[probe == ORANGE] = BG
    return _solve_with_fill(probe)[1]


def _bbox(region: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in region]
    cols = [c for _, c in region]
    return min(rows), min(cols), max(rows), max(cols)


def _candidate_side_barriers(
    grid: np.ndarray,
    sources: list[dict[str, object]],
    infected: set[tuple[int, int]],
    rng: random.Random,
) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    out: list[set[tuple[int, int]]] = []
    infected_bg = {cell for cell in infected if int(grid[cell]) == BG}

    for source in sources:
        region = set(source["region"])
        layers = int(source["layers"])
        r0, c0, r1, c1 = _bbox(region)
        for dist in range(1, layers + 1):
            north = {(r0 - dist, c) for c in range(max(0, c0 - dist), min(w, c1 + dist + 1))}
            south = {(r1 + dist, c) for c in range(max(0, c0 - dist), min(w, c1 + dist + 1))}
            west = {(r, c0 - dist) for r in range(max(0, r0 - dist), min(h, r1 + dist + 1))}
            east = {(r, c1 + dist) for r in range(max(0, r0 - dist), min(h, r1 + dist + 1))}
            for cells in (north, south, west, east):
                if len(cells) < 2:
                    continue
                if not all(_in_bounds(h, w, r, c) for r, c in cells):
                    continue
                if all(cell in infected_bg for cell in cells):
                    out.append(cells)

    rng.shuffle(out)
    return out


def _random_connected_shape(
    allowed: set[tuple[int, int]],
    rng: random.Random,
    size_lo: int,
    size_hi: int,
) -> set[tuple[int, int]] | None:
    if len(allowed) < size_lo:
        return None

    starts = list(allowed)
    rng.shuffle(starts)
    for start in starts[:20]:
        target = rng.randint(size_lo, size_hi)
        cells = {start}
        frontier = [nbr for nbr in _neighbors4_of(start) if nbr in allowed]
        while len(cells) < target and frontier:
            pick = frontier.pop(rng.randrange(len(frontier)))
            if pick in cells:
                continue
            cells.add(pick)
            for nbr in _neighbors4_of(pick):
                if nbr in allowed and nbr not in cells and nbr not in frontier:
                    frontier.append(nbr)
        if len(cells) >= size_lo:
            return cells
    return None


def _neighbors4_of(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _find_blocking_barrier(
    grid: np.ndarray, rng: random.Random
) -> set[tuple[int, int]] | None:
    baseline = _filled_positions_without_barriers(grid)
    if not baseline:
        return None

    sources = _extract_sources(grid)
    candidates = _candidate_side_barriers(grid, sources, baseline, rng)
    best_cells: set[tuple[int, int]] | None = None
    best_score = (-1, -1)

    for cells in candidates[:24]:
        trial = grid.copy()
        _paint_cells(trial, cells, ORANGE)
        reduced = _solve_with_fill(trial)[1]
        removed = baseline - reduced
        direct = baseline & cells
        score = (len(removed) - len(direct), len(removed))
        if score > best_score:
            best_cells = cells
            best_score = score
        if score[0] > 0:
            return cells

    allowed = {cell for cell in baseline if int(grid[cell]) == BG}
    for attempt in range(40):
        cells = _random_connected_shape(allowed, rng, 2, 5)
        if cells is None:
            break
        trial = grid.copy()
        _paint_cells(trial, cells, ORANGE)
        reduced = _solve_with_fill(trial)[1]
        removed = baseline - reduced
        direct = baseline & cells
        score = (len(removed) - len(direct), len(removed))
        if score > best_score:
            best_cells = cells
            best_score = score
        if score[0] > 0 or (attempt >= 20 and score[1] > 0):
            return cells

    if best_cells is not None and best_score[1] > 0:
        return best_cells
    return None


def _add_distractor_barriers(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    rng: random.Random,
    count: int,
) -> int:
    h, w = grid.shape
    placed = 0
    blocked = set(forbidden)
    for _ in range(count):
        allowed = {
            (r, c)
            for r in range(h)
            for c in range(w)
            if int(grid[r, c]) == BG and (r, c) not in blocked
        }
        cells = _random_connected_shape(allowed, rng, 2, 5)
        if cells is None:
            break
        _paint_cells(grid, cells, ORANGE)
        placed += 1
        for r, c in cells:
            blocked.add((r, c))
            for nbr in _neighbors4_of((r, c)):
                blocked.add(nbr)
    return placed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BG, dtype=int)

    num_regions = rng.randint(*REGION_COUNT_RANGE)
    colors = rng.sample(REGION_COLORS, num_regions)
    regions: list[dict[str, object]] = []

    for color in colors:
        placed = None
        for _ in range(60):
            mask = _random_region_mask(rng)
            if mask is None:
                continue
            cells = _place_region(grid, mask, color, rng)
            if cells is not None:
                placed = cells
                break
        if placed is None:
            return None
        regions.append({"color": color, "cells": placed})

    markable = []
    for idx, region in enumerate(regions):
        candidates = _marker_candidates(grid, set(region["cells"]))
        if candidates:
            markable.append(idx)
    if not markable:
        return None

    rng.shuffle(markable)
    source_count = rng.randint(1, min(3, len(markable)))
    chosen_sources = markable[:source_count]

    for idx in chosen_sources:
        candidates = _marker_candidates(grid, set(regions[idx]["cells"]))
        if not candidates:
            return None
        picked = rng.choice(candidates)
        layers = _choose_counter_length(len(picked["line"]), rng)
        yr, yc = picked["yellow"]
        grid[yr, yc] = YELLOW
        for r, c in picked["line"][:layers]:
            grid[r, c] = MAROON

    sources = _extract_sources(grid)
    if len(sources) != source_count:
        return None

    fill_without_barriers = _filled_positions_without_barriers(grid)
    if len(fill_without_barriers) < source_count * 4:
        return None

    barrier_mode = "blocking" if rng.random() < 0.5 else "distractor"
    if barrier_mode == "blocking":
        cells = _find_blocking_barrier(grid, rng)
        if cells is None:
            return None
        _paint_cells(grid, cells, ORANGE)
        _add_distractor_barriers(grid, fill_without_barriers, rng, rng.randint(0, 2))
    else:
        count = rng.randint(1, 3)
        if _add_distractor_barriers(grid, fill_without_barriers, rng, count) == 0:
            return None

    if len(_extract_sources(grid)) != source_count:
        return None

    out, filled = _solve_with_fill(grid)
    if not filled or out.shape != grid.shape:
        return None

    no_barrier_fill = _filled_positions_without_barriers(grid)
    if barrier_mode == "blocking":
        with_barrier_fill = filled
        barrier_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == ORANGE)}
        removed = no_barrier_fill - with_barrier_fill
        direct = barrier_cells & no_barrier_fill
        if len(removed) <= len(direct):
            return None
    else:
        if filled != no_barrier_fill:
            return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid infection-spread input")
