from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SEED_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, TEAL, MAROON)
CONTAINER_WIDTH = 3
CONTAINER_HEIGHT = 4
MAX_GEN_ATTEMPTS = 400


AREA_TEMPLATES = {
    1: [
        {"cells": [(0, 0), (0, 1)], "branching": False},
        {"cells": [(0, 0), (1, 0)], "branching": False},
    ],
    2: [
        {"cells": [(0, 0), (0, 1), (0, 2)], "branching": False},
        {"cells": [(0, 0), (0, 1), (1, 1)], "branching": False},
        {"cells": [(0, 0), (0, 1), (1, 0)], "branching": True},
    ],
    3: [
        {"cells": [(0, 0), (0, 1), (0, 2), (1, 2)], "branching": False},
        {"cells": [(0, 0), (0, 1), (1, 1), (2, 1)], "branching": False},
        {"cells": [(0, 0), (0, 1), (1, 0), (1, 1)], "branching": True},
    ],
    4: [
        {"cells": [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)], "branching": False},
        {"cells": [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)], "branching": False},
        {"cells": [(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)], "branching": True},
    ],
}


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _transform_cells(cells: list[tuple[int, int]], rotation: int, mirror: bool) -> list[tuple[int, int]]:
    out = []
    for r, c in cells:
        if mirror:
            c = -c
        for _ in range(rotation % 4):
            r, c = c, -r
        out.append((r, c))
    return out


def _orange_components(grid: np.ndarray, mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(grid.shape, nr, nc) and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _find_separator_row(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    assert len(rows) == 1, "expected exactly one full-width grey separator row"
    return rows[0]


def _find_seeds(grid: np.ndarray, separator_row: int) -> list[tuple[int, int, int]]:
    seeds = []
    for r in range(separator_row):
        for c in range(grid.shape[1]):
            value = int(grid[r, c])
            if value not in (BLACK, ORANGE, GREY):
                seeds.append((r, c, value))
    assert 3 <= len(seeds) <= 4, "expected 3 or 4 single-pixel seeds"
    assert len({c for _, c, _ in seeds}) == len(seeds), "seeds must occupy distinct columns"
    assert len({value for _, _, value in seeds}) == len(seeds), "seed colors must be distinct"
    return sorted(seeds, key=lambda item: (item[1], item[0]))


def _find_containers(grid: np.ndarray, separator_row: int) -> list[tuple[int, int, int, int]]:
    lower_mask = np.zeros_like(grid, dtype=bool)
    lower_mask[separator_row + 1 :, :] = grid[separator_row + 1 :, :] == ORANGE
    comps = _orange_components(grid, lower_mask)
    containers = []
    for comp in comps:
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height == CONTAINER_HEIGHT and width == CONTAINER_WIDTH and len(comp) == height * width:
            containers.append((r0, c0, r1, c1))
    assert 3 <= len(containers) <= 4, "expected 3 or 4 lower containers"
    containers.sort(key=lambda box: box[1])
    return containers


def _reachable_area(grid: np.ndarray, seed: tuple[int, int], separator_row: int) -> int:
    q = deque([seed])
    seen = {seed}
    area = 0
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < separator_row and 0 <= nc < grid.shape[1]):
                continue
            if (nr, nc) in seen or int(grid[nr, nc]) != BLACK:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))
            area += 1
    return area


def _direct_black_neighbor_count(grid: np.ndarray, seed: tuple[int, int], separator_row: int) -> int:
    r, c = seed
    total = 0
    for nr, nc in neighbors4(r, c):
        if 0 <= nr < separator_row and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == BLACK:
            total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert each seed's flood-reachable black-cell area into stripe height inside its aligned orange container.

    Input structure:
    - The grid has a black upper field, a full-width grey separator row, and 3 or 4 orange 3x4 containers below.
    - The upper field contains distinct single-pixel colored seeds plus orange blocker pixels that form small enclosed black regions.
    - Each seed column falls inside exactly one lower container's column span.

    Rule:
    - For every seed, run a 4-neighbor flood from the seed through black cells only, stopping at orange blockers, the grey separator, and the grid edge.
    - Count the reached black cells, excluding the seed itself, to obtain area A.
    - Find the unique lower orange container whose column range contains that seed column.
    - Recolor the top A full rows of that container with the seed color and leave the remaining container rows orange; keep the upper field and separator unchanged.

    Key insight:
    - The stripe height is determined by hidden 4-connected reachability inside the blocker maze, not by the seed color or by merely counting the seed's immediate empty neighbors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_row = _find_separator_row(g)
    seeds = _find_seeds(g, separator_row)
    containers = _find_containers(g, separator_row)
    assert len(seeds) == len(containers), "seed count must match container count"

    out = g.copy()
    matched_containers: set[int] = set()
    for r, c, color in seeds:
        area = _reachable_area(g, (r, c), separator_row)
        assert 0 <= area <= CONTAINER_HEIGHT, "reachable area must fit container height"
        matches = [i for i, (_, c0, _, c1) in enumerate(containers) if c0 <= c <= c1]
        assert len(matches) == 1, "each seed must align to exactly one container"
        idx = matches[0]
        matched_containers.add(idx)
        r0, c0, r1, c1 = containers[idx]
        if area > 0:
            out[r0 : r0 + area, c0 : c1 + 1] = color
        if area < CONTAINER_HEIGHT:
            out[r0 + area : r1 + 1, c0 : c1 + 1] = ORANGE

    assert len(matched_containers) == len(containers), "every container must be used"
    return out


def _split_extra(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _container_columns(width: int, count: int) -> list[int]:
    base = count * CONTAINER_WIDTH + (count - 1)
    extra = width - base
    assert extra >= 0
    slack = _split_extra(extra, count + 1)
    cols = []
    cur = slack[0]
    for i in range(count):
        cols.append(cur)
        cur += CONTAINER_WIDTH
        if i < count - 1:
            cur += 1 + slack[i + 1]
    return cols


def _region_blockers(region: set[tuple[int, int]], upper_h: int, width: int) -> set[tuple[int, int]]:
    blockers: set[tuple[int, int]] = set()
    for r, c in region:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < upper_h and 0 <= nc < width and (nr, nc) not in region:
                blockers.add((nr, nc))
    return blockers


def _place_region(
    upper_h: int,
    width: int,
    seed_col: int,
    area: int,
    reserved: set[tuple[int, int]],
    require_branching: bool,
) -> tuple[tuple[int, int], set[tuple[int, int]], set[tuple[int, int]], bool] | None:
    templates = AREA_TEMPLATES[area][:]
    random.shuffle(templates)
    if require_branching:
        templates.sort(key=lambda item: not item["branching"])

    for template in templates:
        for _ in range(24):
            rotation = random.randrange(4)
            mirror = bool(random.randrange(2))
            rel_cells = _transform_cells(template["cells"], rotation, mirror)
            min_r = min(r for r, _ in rel_cells)
            max_r = max(r for r, _ in rel_cells)
            min_c = min(c for _, c in rel_cells)
            max_c = max(c for _, c in rel_cells)
            low = max(1, -min_r)
            high = min(upper_h - 2, upper_h - 1 - max_r)
            if low > high:
                continue
            seed_row = random.randint(low, high)
            region = {(seed_row + dr, seed_col + dc) for dr, dc in rel_cells}
            if min(c for _, c in region) < 0 or max(c for _, c in region) >= width:
                continue
            blockers = _region_blockers(region, upper_h, width)
            footprint = region | blockers
            if footprint & reserved:
                continue
            return (seed_row, seed_col), region, blockers, bool(template["branching"])
    return None


def _add_distractors(grid: np.ndarray, upper_h: int, reserved: set[tuple[int, int]]) -> None:
    distractor_count = random.randint(2, 6)
    for _ in range(distractor_count):
        for _ in range(12):
            r = random.randrange(upper_h)
            c = random.randrange(grid.shape[1])
            if (r, c) in reserved or int(grid[r, c]) != BLACK:
                continue
            shape = random.choice(
                [
                    [(0, 0)],
                    [(0, 0), (0, 1)],
                    [(0, 0), (1, 0)],
                    [(0, 0), (1, 0), (1, 1)],
                ]
            )
            cells = {(r + dr, c + dc) for dr, dc in shape}
            if any(not (0 <= rr < upper_h and 0 <= cc < grid.shape[1]) for rr, cc in cells):
                continue
            if cells & reserved:
                continue
            for rr, cc in cells:
                grid[rr, cc] = ORANGE
            reserved |= cells
            break


def _semantic_checks(grid: np.ndarray) -> None:
    separator_row = _find_separator_row(grid)
    seeds = _find_seeds(grid, separator_row)
    containers = _find_containers(grid, separator_row)
    assert len(seeds) == len(containers)

    areas = []
    direct_neighbors = []
    for r, c, _ in seeds:
        area = _reachable_area(grid, (r, c), separator_row)
        areas.append(area)
        direct_neighbors.append(_direct_black_neighbor_count(grid, (r, c), separator_row))
    assert all(1 <= area <= CONTAINER_HEIGHT for area in areas)
    assert len(set(areas)) == len(areas), "areas must be distinct"

    found_counterexample = False
    for i in range(len(areas)):
        for j in range(i + 1, len(areas)):
            if direct_neighbors[i] == direct_neighbors[j] and areas[i] != areas[j]:
                found_counterexample = True
    assert found_counterexample, "need seeds with same local emptiness but different flood areas"

    out = main(grid)
    used_colors = {(c0, c1): None for _, c0, _, c1 in containers}
    for r, c, color in seeds:
        matches = [box for box in containers if box[1] <= c <= box[3]]
        assert len(matches) == 1
        r0, c0, r1, c1 = matches[0]
        area = _reachable_area(grid, (r, c), separator_row)
        if area:
            assert np.all(out[r0 : r0 + area, c0 : c1 + 1] == color)
        if area < CONTAINER_HEIGHT:
            assert np.all(out[r0 + area : r1 + 1, c0 : c1 + 1] == ORANGE)
        used_colors[(c0, c1)] = color
    assert all(color is not None for color in used_colors.values())


def _gen_attempt() -> np.ndarray | None:
    seed_count = random.choice([3, 4])
    width = random.randint(15 if seed_count == 4 else 13, 22)
    if width < seed_count * CONTAINER_WIDTH + (seed_count - 1):
        return None

    upper_h = random.randint(7, 10)
    top_pad = random.randint(0, 1)
    bottom_pad = random.randint(0, 2)
    separator_row = upper_h
    container_top = separator_row + 1 + top_pad
    height = container_top + CONTAINER_HEIGHT + bottom_pad
    if height > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[separator_row, :] = GREY

    container_cols = _container_columns(width, seed_count)
    containers = []
    for c0 in container_cols:
        grid[container_top : container_top + CONTAINER_HEIGHT, c0 : c0 + CONTAINER_WIDTH] = ORANGE
        containers.append((container_top, c0, container_top + CONTAINER_HEIGHT - 1, c0 + CONTAINER_WIDTH - 1))

    colors = random.sample(SEED_COLORS, seed_count)
    areas = random.sample([1, 2, 3, 4], seed_count)
    random.shuffle(areas)

    branching_index = None
    if random.random() < 0.5:
        branching_candidates = [i for i, area in enumerate(areas) if area >= 2]
        if branching_candidates:
            branching_index = random.choice(branching_candidates)

    reserved: set[tuple[int, int]] = set()
    seed_infos = []
    for i, ((_, c0, _, c1), color, area) in enumerate(zip(containers, colors, areas)):
        seed_col = random.randint(c0, c1)
        placed = _place_region(
            upper_h=upper_h,
            width=width,
            seed_col=seed_col,
            area=area,
            reserved=reserved,
            require_branching=(i == branching_index),
        )
        if placed is None:
            return None
        seed_pos, region, blockers, used_branching = placed
        if i == branching_index and not used_branching:
            return None
        reserved |= region | blockers
        seed_infos.append((seed_pos, color, area, region, blockers))

    for seed_pos, color, _area, region, blockers in seed_infos:
        for r, c in blockers:
            grid[r, c] = ORANGE
        sr, sc = seed_pos
        for r, c in region:
            if (r, c) != seed_pos:
                grid[r, c] = BLACK
        grid[sr, sc] = color

    _add_distractors(grid, upper_h, reserved)

    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_ATTEMPTS} attempts")
