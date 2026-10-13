from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

SOIL_POOL = (2, 3, 4, 8)
SEED_POOL = (1, 5, 6, 7, 9)
LEGEND_CORNERS = ("tl", "tr", "bl", "br")

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (18, 26)
REGION_COUNT_RANGE = (2, 4)
REGION_HEIGHT_RANGE = (4, 8)
REGION_WIDTH_RANGE = (4, 9)
REGION_FILL_RATIO = (0.35, 0.72)
REGION_PLACEMENT_PAD = 3
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in NEIGHBORS4]


def _component_from(
    grid: np.ndarray,
    start: tuple[int, int],
    visited: np.ndarray,
    allowed: np.ndarray,
) -> list[tuple[int, int]]:
    q = deque([start])
    visited[start] = True
    color = int(grid[start])
    cells: list[tuple[int, int]] = []
    height, width = grid.shape

    while q:
        r, c = q.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and not visited[nr, nc]:
                if allowed[nr, nc] and int(grid[nr, nc]) == color:
                    visited[nr, nc] = True
                    q.append((nr, nc))
    return cells


def _monochrome_components(grid: np.ndarray, mask: np.ndarray) -> list[dict[str, object]]:
    visited = np.zeros_like(mask, dtype=bool)
    comps: list[dict[str, object]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if visited[r, c]:
            continue
        cells = _component_from(grid, (r, c), visited, mask)
        comps.append({"color": int(grid[r, c]), "cells": cells})
    return comps


def _scan_corner_legend(
    grid: np.ndarray, top: bool, left: bool
) -> tuple[dict[int, int], np.ndarray] | None:
    height, width = grid.shape
    if height < 2:
        return None

    seed_row = 0 if top else height - 2
    soil_row = seed_row + 1
    order = list(range(width)) if left else list(range(width - 1, -1, -1))

    entries: list[tuple[int, int, int]] = []
    mask = np.zeros_like(grid, dtype=bool)
    idx = 0
    while idx < width:
        col = order[idx]
        seed_color = int(grid[seed_row, col])
        soil_color = int(grid[soil_row, col])
        if seed_color == BG or soil_color == BG:
            break

        entries.append((col, seed_color, soil_color))
        mask[seed_row, col] = True
        mask[soil_row, col] = True
        idx += 1
        if idx >= width:
            break

        sep_col = order[idx]
        if int(grid[seed_row, sep_col]) == BG and int(grid[soil_row, sep_col]) == BG:
            idx += 1
            continue
        break

    if len(entries) < 2:
        return None

    mapping: dict[int, int] = {}
    for _col, seed_color, soil_color in entries:
        if seed_color in mapping:
            return None
        mapping[seed_color] = soil_color
    return mapping, mask


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], np.ndarray]:
    best: tuple[int, dict[int, int], np.ndarray] | None = None
    for top, left in ((True, True), (True, False), (False, True), (False, False)):
        candidate = _scan_corner_legend(grid, top=top, left=left)
        if candidate is None:
            continue
        mapping, mask = candidate
        score = len(mapping)
        if best is None or score > best[0]:
            best = (score, mapping, mask)

    if best is None:
        raise AssertionError("legend not found")
    return best[1], best[2]


def _color_distance_map(
    region_set: set[tuple[int, int]], seed_positions: list[tuple[int, int]]
) -> dict[tuple[int, int], int]:
    dist: dict[tuple[int, int], int] = {}
    q: deque[tuple[int, int]] = deque()

    for sr, sc in seed_positions:
        for nr, nc in _neighbors4(sr, sc):
            if (nr, nc) in region_set and (nr, nc) not in dist:
                dist[(nr, nc)] = 1
                q.append((nr, nc))

    while q:
        r, c = q.popleft()
        base = dist[(r, c)]
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in region_set and (nr, nc) not in dist:
                dist[(nr, nc)] = base + 1
                q.append((nr, nc))
    return dist


def _region_assignments(
    region_cells: list[tuple[int, int]], valid_seeds: list[tuple[int, int, int]]
) -> tuple[dict[tuple[int, int], int], set[tuple[int, int]], dict[int, int]]:
    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, color in valid_seeds:
        by_color.setdefault(int(color), []).append((int(r), int(c)))

    region_set = set(region_cells)
    dist_maps = {
        color: _color_distance_map(region_set, positions) for color, positions in by_color.items()
    }
    assignments: dict[tuple[int, int], int] = {}
    tie_cells: set[tuple[int, int]] = set()
    painted_counts = {color: 0 for color in by_color}

    for cell in region_cells:
        best_dist: int | None = None
        best_color: int | None = None
        tied = False
        for color, dist_map in dist_maps.items():
            dist = dist_map.get(cell)
            if dist is None:
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_color = color
                tied = False
            elif dist == best_dist:
                tied = True
        if best_color is not None and not tied:
            assignments[cell] = best_color
            painted_counts[best_color] += 1
        elif tied:
            tie_cells.add(cell)
    return assignments, tie_cells, painted_counts


def _parsed_transform(
    grid: np.ndarray, mapping: dict[int, int], legend_mask: np.ndarray
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    nonlegend_nonbg = (g != BG) & (~legend_mask)
    comps = _monochrome_components(g, nonlegend_nonbg)

    soil_colors = set(mapping.values())
    soil_comps: list[dict[str, object]] = []
    seed_pixels: dict[tuple[int, int], int] = {}

    for comp in comps:
        color = int(comp["color"])
        cells = comp["cells"]
        if color in soil_colors or len(cells) > 1:
            soil_comps.append({"color": color, "cells": cells})
        else:
            for r, c in cells:
                seed_pixels[(r, c)] = color

    out = g.copy()
    out[legend_mask] = BG
    for r, c in seed_pixels:
        out[r, c] = BG

    live_seeds: dict[tuple[int, int], int] = {}
    for comp in soil_comps:
        soil_color = int(comp["color"])
        region_cells = comp["cells"]
        valid: list[tuple[int, int, int]] = []
        seen_positions: set[tuple[int, int]] = set()
        for r, c in region_cells:
            for nr, nc in _neighbors4(r, c):
                pos = (nr, nc)
                if pos in seen_positions or pos not in seed_pixels:
                    continue
                seen_positions.add(pos)
                seed_color = seed_pixels[pos]
                if mapping.get(seed_color) == soil_color:
                    valid.append((nr, nc, seed_color))
                    live_seeds[pos] = seed_color
        if not valid:
            continue

        assignments, _tie_cells, _painted = _region_assignments(region_cells, valid)
        for (r, c), color in assignments.items():
            out[r, c] = color

    for (r, c), color in live_seeds.items():
        out[r, c] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clear the legend, remove invalid seeds, and let matching seeds grow through
    adjacent soil regions by nearest-seed-color competition.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_mask = _find_legend(g)
    return _parsed_transform(g, mapping, legend_mask)


def _legend_rows(height: int, corner: str) -> tuple[int, int]:
    return (0, 1) if corner in {"tl", "tr"} else (height - 2, height - 1)


def _place_legend(
    grid: np.ndarray, corner: str, pairs: list[tuple[int, int]]
) -> set[tuple[int, int]]:
    height, width = grid.shape
    seed_row, soil_row = _legend_rows(height, corner)
    left = corner in {"tl", "bl"}
    legend_cells: set[tuple[int, int]] = set()
    for idx, (seed_color, soil_color) in enumerate(pairs):
        col = 2 * idx if left else width - 1 - 2 * idx
        grid[seed_row, col] = seed_color
        grid[soil_row, col] = soil_color
        legend_cells.add((seed_row, col))
        legend_cells.add((soil_row, col))
    return legend_cells


def _sample_mapping() -> tuple[dict[int, int], dict[int, list[int]]]:
    patterns = ([1, 1], [2, 1], [2, 2], [1, 1, 1], [2, 1, 1])
    counts = list(random.choice(patterns))
    soil_colors = random.sample(SOIL_POOL, len(counts))
    seed_colors = random.sample(SEED_POOL, sum(counts))
    random.shuffle(seed_colors)

    mapping: dict[int, int] = {}
    soil_to_seeds: dict[int, list[int]] = {}
    offset = 0
    for soil_color, count in zip(soil_colors, counts, strict=False):
        soil_to_seeds[soil_color] = seed_colors[offset : offset + count]
        for seed_color in soil_to_seeds[soil_color]:
            mapping[seed_color] = soil_color
        offset += count
    return mapping, soil_to_seeds


def _random_blob(height: int, width: int) -> list[tuple[int, int]] | None:
    total = height * width
    low = max(6, int(total * REGION_FILL_RATIO[0]))
    high = max(low, int(total * REGION_FILL_RATIO[1]))
    target = random.randint(low, high)

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    if max(rows) - min(rows) < 2 or max(cols) - min(cols) < 2:
        return None
    return list(cells)


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _placement_bounds(height: int, width: int, corner: str) -> tuple[int, int, int, int]:
    top_margin = 4 if corner in {"tl", "tr"} else 1
    bottom_margin = 4 if corner in {"bl", "br"} else 1
    return top_margin, height - bottom_margin, 1, width - 1


def _can_place_bbox(
    blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int
) -> bool:
    height, width = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height - 1, r1 + pad)
    cc1 = min(width - 1, c1 + pad)
    return not bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _mark_bbox(blocked: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int) -> None:
    height, width = blocked.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(height - 1, r1 + pad)
    cc1 = min(width - 1, c1 + pad)
    blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_regions(
    grid: np.ndarray, corner: str, region_plans: list[tuple[int, str]]
) -> list[dict[str, object]] | None:
    height, width = grid.shape
    top_min, bottom_max, left_min, right_max = _placement_bounds(height, width, corner)
    blocked = np.zeros_like(grid, dtype=bool)
    if corner in {"tl", "tr"}:
        blocked[0:4, :] = True
    else:
        blocked[height - 4 : height, :] = True
    blocked[:, 0] = True
    blocked[:, width - 1] = True

    regions: list[dict[str, object]] = []
    for soil_color, scenario in region_plans:
        placed = False
        for _ in range(120):
            box_h = random.randint(*REGION_HEIGHT_RANGE)
            box_w = random.randint(*REGION_WIDTH_RANGE)
            if box_h >= bottom_max - top_min or box_w >= right_max - left_min:
                continue

            local_cells = _random_blob(box_h, box_w)
            if local_cells is None:
                continue

            max_r0 = bottom_max - box_h
            max_c0 = right_max - box_w
            if top_min > max_r0 or left_min > max_c0:
                continue
            r0 = random.randint(top_min, max_r0)
            c0 = random.randint(left_min, max_c0)
            global_cells = [(r0 + r, c0 + c) for r, c in local_cells]
            gr0, gc0, gr1, gc1 = _bbox(global_cells)
            if not _can_place_bbox(blocked, gr0, gc0, gr1, gc1, REGION_PLACEMENT_PAD):
                continue

            for r, c in global_cells:
                grid[r, c] = soil_color
            _mark_bbox(blocked, gr0, gc0, gr1, gc1, REGION_PLACEMENT_PAD)
            regions.append(
                {"color": soil_color, "cells": global_cells, "scenario": scenario, "seeds": []}
            )
            placed = True
            break
        if not placed:
            return None
    return regions


def _perimeter_cells(grid: np.ndarray, region_cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    region_set = set(region_cells)
    candidates: set[tuple[int, int]] = set()
    height, width = grid.shape
    for r, c in region_cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in region_set:
                if int(grid[nr, nc]) == BG:
                    candidates.add((nr, nc))
    return sorted(candidates)


def _pick_far_positions(
    candidates: list[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    if len(candidates) < count:
        return None
    if count == 1:
        return [random.choice(candidates)]

    best: tuple[int, list[tuple[int, int]]] | None = None
    tries = min(120, len(candidates) * len(candidates))
    for _ in range(tries):
        picked = random.sample(candidates, count)
        score = 0
        for i in range(count):
            for j in range(i + 1, count):
                score += abs(picked[i][0] - picked[j][0]) + abs(picked[i][1] - picked[j][1])
        if best is None or score > best[0]:
            best = (score, picked)
    return None if best is None else best[1]


def _invalid_seed_choices(mapping: dict[int, int], soil_color: int) -> list[int]:
    choices = [color for color in SEED_POOL if mapping.get(color) != soil_color]
    return choices


def _scenario_valid(
    region: dict[str, object], seeds: list[tuple[int, int, int]], mapping: dict[int, int]
) -> bool:
    valid = [seed for seed in seeds if mapping.get(seed[2]) == int(region["color"])]
    if not valid:
        return True
    assignments, tie_cells, painted = _region_assignments(region["cells"], valid)
    _ = assignments
    distinct_colors = {seed[2] for seed in valid}
    if len(distinct_colors) <= 1:
        return True
    if tie_cells:
        return False
    return min(painted.values()) >= 3


def _assign_region_seeds(
    grid: np.ndarray,
    region: dict[str, object],
    mapping: dict[int, int],
    soil_to_seeds: dict[int, list[int]],
) -> bool:
    soil_color = int(region["color"])
    scenario = str(region["scenario"])
    candidates = _perimeter_cells(grid, region["cells"])
    valid_colors = list(soil_to_seeds.get(soil_color, []))
    invalid_colors = _invalid_seed_choices(mapping, soil_color)
    if not candidates:
        return False

    for _ in range(80):
        seeds: list[tuple[int, int, int]] = []
        if scenario == "split_pair":
            if len(valid_colors) < 2:
                return False
            positions = _pick_far_positions(candidates, 2)
            if positions is None:
                return False
            colors = random.sample(valid_colors, 2)
            seeds = [
                (positions[0][0], positions[0][1], colors[0]),
                (positions[1][0], positions[1][1], colors[1]),
            ]
        elif scenario == "same_color_pair":
            if not valid_colors:
                return False
            positions = _pick_far_positions(candidates, 2)
            if positions is None:
                return False
            color = random.choice(valid_colors)
            seeds = [(positions[0][0], positions[0][1], color), (positions[1][0], positions[1][1], color)]
        elif scenario == "valid_plus_invalid":
            if not valid_colors or not invalid_colors or len(candidates) < 2:
                return False
            positions = random.sample(candidates, 2)
            seeds = [
                (positions[0][0], positions[0][1], random.choice(valid_colors)),
                (positions[1][0], positions[1][1], random.choice(invalid_colors)),
            ]
        elif scenario == "invalid_only":
            if not invalid_colors:
                return False
            count = 1 if len(candidates) == 1 else random.choice((1, 2))
            positions = random.sample(candidates, count)
            seeds = [(r, c, random.choice(invalid_colors)) for r, c in positions]
        else:
            if not valid_colors:
                return False
            pos = random.choice(candidates)
            seeds = [(pos[0], pos[1], random.choice(valid_colors))]

        if len({(r, c) for r, c, _color in seeds}) != len(seeds):
            continue
        if not _scenario_valid(region, seeds, mapping):
            continue
        for r, c, color in seeds:
            if int(grid[r, c]) != BG:
                return False
            grid[r, c] = color
        region["seeds"] = seeds
        return True
    return False


def _scene_expected(
    grid: np.ndarray,
    legend_cells: set[tuple[int, int]],
    mapping: dict[int, int],
    regions: list[dict[str, object]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for r, c in legend_cells:
        out[r, c] = BG

    live_seeds: dict[tuple[int, int], int] = {}
    for region in regions:
        for r, c, _color in region["seeds"]:
            out[r, c] = BG

    for region in regions:
        valid = [
            (r, c, color)
            for r, c, color in region["seeds"]
            if mapping.get(int(color)) == int(region["color"])
        ]
        for r, c, color in valid:
            live_seeds[(r, c)] = color
        if not valid:
            continue
        assignments, _tie_cells, _painted = _region_assignments(region["cells"], valid)
        for (r, c), color in assignments.items():
            out[r, c] = color

    for (r, c), color in live_seeds.items():
        out[r, c] = color
    return out


def _build_region_plans(soil_to_seeds: dict[int, list[int]], region_count: int) -> list[tuple[int, str]]:
    soils = list(soil_to_seeds)
    multi_soils = [soil for soil, seeds in soil_to_seeds.items() if len(seeds) >= 2]
    plans: list[tuple[int, str]] = []

    if multi_soils:
        if random.random() < 0.5:
            plans.append((random.choice(multi_soils), "split_pair"))
        else:
            plans.append((random.choice(multi_soils), "same_color_pair"))

    invalid_scenario = random.choice(("invalid_only", "valid_plus_invalid"))
    plans.append((random.choice(soils), invalid_scenario))

    scenario_choices = ["single_valid", "valid_plus_invalid", "invalid_only"]
    if multi_soils:
        scenario_choices.extend(["split_pair", "same_color_pair"])

    while len(plans) < region_count:
        soil_color = random.choice(soils)
        scenario = random.choice(scenario_choices)
        if scenario == "split_pair" and len(soil_to_seeds[soil_color]) < 2:
            scenario = "single_valid"
        if scenario == "same_color_pair" and len(soil_to_seeds[soil_color]) < 1:
            scenario = "single_valid"
        plans.append((soil_color, scenario))

    random.shuffle(plans)
    return plans[:region_count]


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    mapping, soil_to_seeds = _sample_mapping()
    pairs = list(mapping.items())
    random.shuffle(pairs)

    legend_width = 2 * len(pairs) - 1
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(max(GRID_WIDTH_RANGE[0], legend_width + 4), GRID_WIDTH_RANGE[1])
    corner = random.choice(LEGEND_CORNERS)

    grid = np.zeros((height, width), dtype=int)
    legend_cells = _place_legend(grid, corner, pairs)

    region_count = random.randint(*REGION_COUNT_RANGE)
    plans = _build_region_plans(soil_to_seeds, region_count)
    regions = _place_regions(grid, corner, plans)
    if regions is None:
        return None

    for region in regions:
        if not _assign_region_seeds(grid, region, mapping, soil_to_seeds):
            return None

    expected = _scene_expected(grid, legend_cells, mapping, regions)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        scene = _sample_scene()
        if scene is None:
            continue
        grid, _expected = scene
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
