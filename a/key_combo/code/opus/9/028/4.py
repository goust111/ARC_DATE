from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4)

GRID_H_RANGE = (15, 22)
GRID_W_RANGE = (15, 22)
TARGET_REGION_OPTIONS = (4, 5, 6)
MIN_REGION_SIDE = 3
MIN_REGION_AREA = 9
GEN_RETRIES = 600


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _nonwall_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == WALL:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            region: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                region.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) != WALL:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            regions.append(region)
    return regions


def _region_seed_info(grid: np.ndarray, region: list[tuple[int, int]]) -> list[tuple[int, int, int]]:
    seeds: list[tuple[int, int, int]] = []
    for r, c in region:
        value = int(grid[r, c])
        if value in SEED_COLORS:
            seeds.append((r, c, value))
        else:
            assert value == BG, "non-wall region cells must be background or seed colors"
    return seeds


def _seed_bg_neighbors(
    grid: np.ndarray,
    region_set: set[tuple[int, int]],
    seed: tuple[int, int, int],
) -> int:
    h, w = grid.shape
    r, c, _color = seed
    total = 0
    for nr, nc in _neighbors4(r, c, h, w):
        if (nr, nc) in region_set and int(grid[nr, nc]) == BG:
            total += 1
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each walled region according to how many seed pixels it contains.

    Input structure:
    - The background is black (0), and gray (5) wall segments split the grid
      into several 4-connected non-wall regions bounded by walls and/or edges.
    - Every non-wall region contains 0, 1, or 2 single-pixel seeds colored from
      {blue(1), red(2), green(3), yellow(4)}.
    - In a 2-seed region, the two seeds have different colors.

    Rule:
    - Identify each 4-connected non-wall region independently.
    - If a region has 0 seeds, leave it unchanged.
    - If a region has 1 seed, flood-fill that region's background cells with the
      seed color.
    - If a region has 2 seeds, compare how many 4-neighbor cells inside the same
      region are background for each seed; the seed with the strictly larger
      count wins, and its color fills the entire region, covering both seeds.
    - Wall cells always stay gray.

    Key insight:
    - The seed count is not the final answer by itself; it selects which fill
      algorithm applies to that region, and only the 2-seed case needs an extra
      local-neighborhood comparison before filling.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    out = g.copy()
    for region in _nonwall_regions(g):
        seeds = _region_seed_info(g, region)
        assert len(seeds) in (0, 1, 2), "each region must contain 0, 1, or 2 seeds"

        if len(seeds) == 0:
            continue

        if len(seeds) == 1:
            fill_color = int(seeds[0][2])
            for r, c in region:
                if int(g[r, c]) == BG:
                    out[r, c] = fill_color
            continue

        region_set = set(region)
        scores = [_seed_bg_neighbors(g, region_set, seed) for seed in seeds]
        assert scores[0] != scores[1], "2-seed regions must have a unique winner"
        fill_color = int(seeds[0][2] if scores[0] > scores[1] else seeds[1][2])
        for r, c in region:
            out[r, c] = fill_color

    return out


def _rect_height(region: dict) -> int:
    return int(region["r1"] - region["r0"] + 1)


def _rect_width(region: dict) -> int:
    return int(region["c1"] - region["c0"] + 1)


def _split_positions(region: dict, orientation: str, allow_any: bool) -> list[int]:
    r0, r1 = int(region["r0"]), int(region["r1"])
    c0, c1 = int(region["c0"]), int(region["c1"])
    h = _rect_height(region)
    w = _rect_width(region)

    if orientation == "V":
        if not allow_any and not (region["top_wall"] or region["bottom_wall"]):
            return []
        positions = []
        for col in range(c0 + MIN_REGION_SIDE, c1 - MIN_REGION_SIDE + 1):
            left_w = col - c0
            right_w = c1 - col
            if left_w < MIN_REGION_SIDE or right_w < MIN_REGION_SIDE:
                continue
            if h * left_w < MIN_REGION_AREA or h * right_w < MIN_REGION_AREA:
                continue
            positions.append(col)
        return positions

    if not allow_any and not (region["left_wall"] or region["right_wall"]):
        return []
    positions = []
    for row in range(r0 + MIN_REGION_SIDE, r1 - MIN_REGION_SIDE + 1):
        top_h = row - r0
        bot_h = r1 - row
        if top_h < MIN_REGION_SIDE or bot_h < MIN_REGION_SIDE:
            continue
        if top_h * w < MIN_REGION_AREA or bot_h * w < MIN_REGION_AREA:
            continue
        positions.append(row)
    return positions


def _split_region(region: dict, orientation: str, pos: int):
    r0, r1 = int(region["r0"]), int(region["r1"])
    c0, c1 = int(region["c0"]), int(region["c1"])

    if orientation == "V":
        wall_cells = [(r, pos) for r in range(r0, r1 + 1)]
        left = {
            "r0": r0,
            "r1": r1,
            "c0": c0,
            "c1": pos - 1,
            "top_wall": bool(region["top_wall"]),
            "bottom_wall": bool(region["bottom_wall"]),
            "left_wall": bool(region["left_wall"]),
            "right_wall": True,
        }
        right = {
            "r0": r0,
            "r1": r1,
            "c0": pos + 1,
            "c1": c1,
            "top_wall": bool(region["top_wall"]),
            "bottom_wall": bool(region["bottom_wall"]),
            "left_wall": True,
            "right_wall": bool(region["right_wall"]),
        }
        return left, right, wall_cells

    wall_cells = [(pos, c) for c in range(c0, c1 + 1)]
    top = {
        "r0": r0,
        "r1": pos - 1,
        "c0": c0,
        "c1": c1,
        "top_wall": bool(region["top_wall"]),
        "bottom_wall": True,
        "left_wall": bool(region["left_wall"]),
        "right_wall": bool(region["right_wall"]),
    }
    bottom = {
        "r0": pos + 1,
        "r1": r1,
        "c0": c0,
        "c1": c1,
        "top_wall": True,
        "bottom_wall": bool(region["bottom_wall"]),
        "left_wall": bool(region["left_wall"]),
        "right_wall": bool(region["right_wall"]),
    }
    return top, bottom, wall_cells


def _build_partition(h: int, w: int, target_regions: int):
    leaves = [{
        "r0": 0,
        "r1": h - 1,
        "c0": 0,
        "c1": w - 1,
        "top_wall": False,
        "bottom_wall": False,
        "left_wall": False,
        "right_wall": False,
    }]
    wall_cells: list[tuple[int, int]] = []
    first_split = True

    while len(leaves) < target_regions:
        candidates = []
        for idx, region in enumerate(leaves):
            for orientation in ("V", "H"):
                positions = _split_positions(region, orientation, allow_any=first_split)
                if positions:
                    candidates.append((idx, orientation, positions))
        if not candidates:
            return None

        idx, orientation, positions = random.choice(candidates)
        pos = random.choice(positions)
        region = leaves.pop(idx)
        child_a, child_b, new_walls = _split_region(region, orientation, pos)
        leaves.extend([child_a, child_b])
        wall_cells.extend(new_walls)
        first_split = False

    return leaves, wall_cells


def _region_cells(region: dict) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(int(region["r0"]), int(region["r1"]) + 1)
        for c in range(int(region["c0"]), int(region["c1"]) + 1)
    ]


def _rect_degree(region: dict, cell: tuple[int, int]) -> int:
    r, c = cell
    degree = 0
    if r > int(region["r0"]):
        degree += 1
    if r < int(region["r1"]):
        degree += 1
    if c > int(region["c0"]):
        degree += 1
    if c < int(region["c1"]):
        degree += 1
    return degree


def _choose_seed_cells(region: dict, seed_count: int):
    cells = _region_cells(region)
    if seed_count == 0:
        return []

    if seed_count == 1:
        annotated = [(cell, _rect_degree(region, cell)) for cell in cells]
        max_degree = max(degree for _cell, degree in annotated)
        choices = [cell for cell, degree in annotated if degree == max_degree]
        if random.random() < 0.5:
            choices = cells
        return [random.choice(choices)]

    annotated = [(cell, _rect_degree(region, cell)) for cell in cells]
    pairs = []
    for i in range(len(annotated)):
        cell_a, deg_a = annotated[i]
        for j in range(i + 1, len(annotated)):
            cell_b, deg_b = annotated[j]
            if abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1]) < 2:
                continue
            if deg_a == deg_b:
                continue
            pairs.append((cell_a, deg_a, cell_b, deg_b))
    assert pairs, "expected at least one valid 2-seed placement"
    max_gap = max(abs(deg_a - deg_b) for _a, deg_a, _b, deg_b in pairs)
    strong_pairs = [pair for pair in pairs if abs(pair[1] - pair[3]) == max_gap]
    cell_a, _deg_a, cell_b, _deg_b = random.choice(strong_pairs if random.random() < 0.7 else pairs)
    return [cell_a, cell_b]


def _assign_seed_counts(n_regions: int) -> list[int]:
    counts = [0, 1, 2]
    while len(counts) < n_regions:
        counts.append(random.choice((0, 1, 1, 2, 2)))
    random.shuffle(counts)
    return counts


def _expected_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for region in _nonwall_regions(grid):
        seeds = _region_seed_info(grid, region)
        if len(seeds) == 1:
            color = int(seeds[0][2])
            for r, c in region:
                if int(grid[r, c]) == BG:
                    out[r, c] = color
        elif len(seeds) == 2:
            region_set = set(region)
            first = _seed_bg_neighbors(grid, region_set, seeds[0])
            second = _seed_bg_neighbors(grid, region_set, seeds[1])
            if first == second:
                raise AssertionError("double-seed region has tied neighborhood counts")
            color = int(seeds[0][2] if first > second else seeds[1][2])
            for r, c in region:
                out[r, c] = color
    return out


def _validate_semantics(grid: np.ndarray) -> None:
    h, w = grid.shape
    assert h <= 30 and w <= 30
    assert set(map(int, np.unique(grid))).issubset({BG, WALL, *SEED_COLORS})

    regions = _nonwall_regions(grid)
    assert 4 <= len(regions) <= 6, "must create 4-6 regions"

    seed_counts = []
    resolved_colors = []
    saw_single = False
    saw_double = False
    for region in regions:
        bg_cells = sum(int(grid[r, c]) == BG for r, c in region)
        assert bg_cells >= 4, "every region must keep at least four background cells in the input"
        seeds = _region_seed_info(grid, region)
        seed_counts.append(len(seeds))
        assert len(seeds) in (0, 1, 2)
        if len(seeds) == 1:
            saw_single = True
            resolved_colors.append(int(seeds[0][2]))
        elif len(seeds) == 2:
            saw_double = True
            assert seeds[0][2] != seeds[1][2], "double-seed colors must differ"
            region_set = set(region)
            scores = [_seed_bg_neighbors(grid, region_set, seed) for seed in seeds]
            assert scores[0] != scores[1], "double-seed regions need a unique winner"
            resolved_colors.append(int(seeds[0][2] if scores[0] > scores[1] else seeds[1][2]))
    assert 0 in seed_counts and 1 in seed_counts and 2 in seed_counts
    assert saw_single and saw_double
    assert len(set(resolved_colors)) >= 2, "filled regions should not collapse to a single output color"


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    target_regions = random.choice(TARGET_REGION_OPTIONS)
    built = _build_partition(h, w, target_regions)
    if built is None:
        return None
    regions, wall_cells = built

    grid = np.zeros((h, w), dtype=int)
    for r, c in wall_cells:
        grid[r, c] = WALL

    counts = _assign_seed_counts(len(regions))
    random.shuffle(regions)

    for region, seed_count in zip(regions, counts):
        seed_cells = _choose_seed_cells(region, seed_count)
        if seed_count == 1:
            color = random.choice(SEED_COLORS)
            r, c = seed_cells[0]
            grid[r, c] = color
        elif seed_count == 2:
            colors = random.sample(list(SEED_COLORS), 2)
            for (r, c), color in zip(seed_cells, colors):
                grid[r, c] = color

    try:
        _validate_semantics(grid)
        expected = _expected_output(grid)
        actual = main(grid)
    except AssertionError:
        return None

    if np.array_equal(actual, expected) and not np.array_equal(actual, grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
