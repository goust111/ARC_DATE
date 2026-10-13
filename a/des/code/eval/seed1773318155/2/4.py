import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
LEGEND_ROWS = 3
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
SHAPE_RETRIES = 80
PLACEMENT_RETRIES = 120
REGION_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _parse_legend(grid: np.ndarray) -> dict[int, str]:
    top = np.asarray(grid[:LEGEND_ROWS], dtype=int)
    mapping: dict[int, str] = {}

    for c in range(top.shape[1]):
        col = top[:, c]
        rows = np.flatnonzero(col != BLACK)
        if rows.size == 0:
            continue
        color = int(col[rows[0]])
        if not np.all(col[rows] == color):
            raise AssertionError("legend column must be monochrome")
        if rows[-1] - rows[0] + 1 != rows.size:
            raise AssertionError("legend column must be contiguous")
        if rows.size not in (2, 3):
            raise AssertionError("legend height must be 2 or 3")

        direction = "H" if rows.size == 2 else "V"
        prev = mapping.get(color)
        if prev is not None and prev != direction:
            raise AssertionError("legend color has inconsistent direction")
        mapping[color] = direction

    if not mapping:
        raise AssertionError("legend must contain at least one color")
    return mapping


def _component_from_seed(walkable: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = walkable.shape
    q = deque([start])
    seen = {start}
    comp = [start]

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
                comp.append((nr, nc))
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the legend block heights to decide each seed color's stripe direction.

    Height 2 means horizontal stripes and height 3 means vertical stripes.
    Each seed expands only inside its own grey-bounded region.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    mapping = _parse_legend(g)

    walkable = np.zeros_like(g, dtype=bool)
    walkable[LEGEND_ROWS:, :] = g[LEGEND_ROWS:, :] != GREY

    seeds = np.argwhere((g[LEGEND_ROWS:, :] != BLACK) & (g[LEGEND_ROWS:, :] != GREY))
    seen_cells: set[tuple[int, int]] = set()

    for sr0, sc in seeds:
        sr = int(sr0 + LEGEND_ROWS)
        sc = int(sc)
        color = int(g[sr, sc])
        direction = mapping[color]
        region = _component_from_seed(walkable, (sr, sc))
        region_set = set(region)
        if seen_cells & region_set:
            raise AssertionError("each region must contain exactly one seed")
        seen_cells |= region_set

        if direction == "H":
            parity = sr % 2
            for r, c in region:
                if r % 2 == parity:
                    out[r, c] = color
        else:
            parity = sc % 2
            for r, c in region:
                if c % 2 == parity:
                    out[r, c] = color

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _shell(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    shell: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in cells:
                shell.add((nr, nc))
    return shell


def _bbox_size(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _stripe_capacity(cells: set[tuple[int, int]], direction: str) -> int:
    if direction == "H":
        return max(sum(1 for r, _ in cells if r % 2 == parity) for parity in (0, 1))
    return max(sum(1 for _, c in cells if c % 2 == parity) for parity in (0, 1))


def _grow_region_shape(rng: random.Random, direction: str) -> tuple[set[tuple[int, int]], set[tuple[int, int]]] | None:
    for _ in range(SHAPE_RETRIES):
        box_h = rng.randint(4, 8)
        box_w = rng.randint(4, 9)
        rect_h = rng.randint(2, min(4, box_h))
        rect_w = rng.randint(2, min(4, box_w))
        r0 = rng.randint(0, box_h - rect_h)
        c0 = rng.randint(0, box_w - rect_w)

        cells = {(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)}
        target_min = max(len(cells) + 2, 10)
        target_max = min(box_h * box_w, 30)
        if target_min > target_max:
            continue
        target = rng.randint(target_min, target_max)

        while len(cells) < target:
            choices: dict[tuple[int, int], int] = {}
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells:
                        adj = sum((ar, ac) in cells for ar, ac in _neighbors4(nr, nc))
                        weight = 1 + 3 * adj
                        choices[(nr, nc)] = max(weight, choices.get((nr, nc), 0))
            if not choices:
                break
            pts = list(choices)
            weights = [choices[pt] for pt in pts]
            cells.add(rng.choices(pts, weights=weights, k=1)[0])

        cells = _normalize(cells)
        box_h2, box_w2 = _bbox_size(cells)
        area = len(cells)
        density = area / float(box_h2 * box_w2)

        if box_h2 < 3 or box_w2 < 3 or area < 10 or density < 0.45:
            continue
        if direction == "H" and box_h2 < 4:
            continue
        if direction == "V" and box_w2 < 4:
            continue
        if _stripe_capacity(cells, direction) < 4:
            continue

        shell = _shell(cells)
        union = cells | shell
        min_r = min(r for r, _ in union)
        min_c = min(c for _, c in union)
        cells = {(r - min_r, c - min_c) for r, c in cells}
        shell = {(r - min_r, c - min_c) for r, c in shell}
        if not shell:
            continue
        return cells, shell

    return None


def _expand8(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if LEGEND_ROWS <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _pick_seed(rng: random.Random, cells: set[tuple[int, int]], direction: str) -> tuple[int, int] | None:
    groups: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    for cell in cells:
        key = cell[0] % 2 if direction == "H" else cell[1] % 2
        groups[key].append(cell)

    viable = [pts for pts in groups.values() if len(pts) >= 4]
    if not viable:
        return None
    group = max(viable, key=len) if rng.random() < 0.75 else rng.choice(viable)
    return rng.choice(group)


def _place_region(
    grid: np.ndarray,
    blocked: np.ndarray,
    rng: random.Random,
    color: int,
    direction: str,
):
    shape = _grow_region_shape(rng, direction)
    if shape is None:
        return None
    local_cells, local_shell = shape
    total = local_cells | local_shell
    total_h, total_w = _bbox_size(total)
    h, w = grid.shape

    row_min = LEGEND_ROWS + 1
    row_max = h - total_h - 1
    col_min = 1
    col_max = w - total_w - 1
    if row_min > row_max or col_min > col_max:
        return None

    for _ in range(PLACEMENT_RETRIES):
        base_r = rng.randint(row_min, row_max)
        base_c = rng.randint(col_min, col_max)
        placed_cells = {(base_r + r, base_c + c) for r, c in local_cells}
        placed_shell = {(base_r + r, base_c + c) for r, c in local_shell}
        footprint = placed_cells | placed_shell
        padded = _expand8(footprint, h, w)

        if any(blocked[r, c] for r, c in padded):
            continue

        seed = _pick_seed(rng, placed_cells, direction)
        if seed is None:
            continue

        for r, c in placed_shell:
            grid[r, c] = GREY
        grid[seed] = color
        for r, c in padded:
            blocked[r, c] = True

        return {
            "cells": placed_cells,
            "seed": seed,
            "color": color,
            "direction": direction,
        }

    return None


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp = [(r, c)]
            while q:
                cr, cc = q.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
                        comp.append((nr, nc))
            comps.append(comp)
    return comps


def _validate_scene(grid: np.ndarray, expected_regions: int) -> bool:
    try:
        mapping = _parse_legend(grid)
    except AssertionError:
        return False

    walkable = np.zeros_like(grid, dtype=bool)
    walkable[LEGEND_ROWS:, :] = grid[LEGEND_ROWS:, :] != GREY
    comps = _components(walkable)

    seeded = 0
    empty = 0
    seen_dirs = set()
    for comp in comps:
        colors = [(r, c) for r, c in comp if grid[r, c] not in (BLACK, GREY)]
        if len(colors) == 0:
            empty += 1
            continue
        if len(colors) != 1:
            return False

        seeded += 1
        sr, sc = colors[0]
        direction = mapping.get(int(grid[sr, sc]))
        if direction is None:
            return False
        seen_dirs.add(direction)
        if _stripe_capacity(set(comp), direction) < 4:
            return False

    if seeded != expected_regions or empty != 1 or seen_dirs != {"H", "V"}:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(16, 26)
    w = rng.randint(18, 30)
    max_colors = min(5, len(REGION_COLORS), (w + 1) // 2)
    if max_colors < 2:
        return None

    n_colors = rng.randint(2, max_colors)
    legend_colors = rng.sample(REGION_COLORS, n_colors)
    rng.shuffle(legend_colors)

    directions = ["H", "V"] + [rng.choice(("H", "V")) for _ in range(n_colors - 2)]
    rng.shuffle(directions)
    color_to_dir = dict(zip(legend_colors, directions))

    legend_width = 2 * n_colors - 1
    legend_start = rng.randint(0, w - legend_width)

    grid = np.zeros((h, w), dtype=int)
    for i, color in enumerate(legend_colors):
        c = legend_start + 2 * i
        height = 2 if color_to_dir[color] == "H" else 3
        grid[:height, c] = color

    area = h * w
    max_regions = 5 if area >= 520 else 4 if area >= 380 else 3
    n_regions = rng.randint(2, max_regions)

    horiz_colors = [color for color in legend_colors if color_to_dir[color] == "H"]
    vert_colors = [color for color in legend_colors if color_to_dir[color] == "V"]
    if not horiz_colors or not vert_colors:
        return None

    region_colors = [rng.choice(horiz_colors), rng.choice(vert_colors)]
    while len(region_colors) < n_regions:
        region_colors.append(rng.choice(legend_colors))
    rng.shuffle(region_colors)

    blocked = np.zeros_like(grid, dtype=bool)
    placed = []
    for color in region_colors:
        direction = color_to_dir[color]
        region = _place_region(grid, blocked, rng, color, direction)
        if region is None:
            return None
        placed.append(region)

    if not _validate_scene(grid, len(placed)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-region puzzle input")
