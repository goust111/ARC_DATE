import random

import numpy as np


BG_COLOR = 0
WALL_COLOR = 9
SEED_COLORS = tuple(color for color in range(1, 10) if color != WALL_COLOR)
SIDES = ("top", "bottom", "left", "right")
GENERATION_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 150
OUTER_SIZE_RANGE = (5, 8)
GRID_SIZE_RANGE = (14, 30)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(coords)
    return parts


def _perimeter_cells(r0: int, r1: int, c0: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells |= {(r1, c) for c in range(c0, c1 + 1)}
    cells |= {(r, c0) for r in range(r0 + 1, r1)}
    cells |= {(r, c1) for r in range(r0 + 1, r1)}
    return cells


def _line_cells(gap: tuple[int, int], direction: tuple[int, int], length: int) -> list[tuple[int, int]]:
    gr, gc = gap
    dr, dc = direction
    return [(gr + dr * step, gc + dc * step) for step in range(1, length + 1)]


def _parse_walls(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    infos: list[dict] = []
    for comp in _components(g == WALL_COLOR):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        comp_set = set(comp)
        perimeter = _perimeter_cells(r0, r1, c0, c1)
        if not comp_set <= perimeter:
            continue

        missing = sorted(perimeter - comp_set)
        if len(missing) != 1:
            continue
        gap = missing[0]
        gr, gc = gap

        if gr == r0 and c0 < gc < c1:
            side = "top"
            direction = (-1, 0)
        elif gr == r1 and c0 < gc < c1:
            side = "bottom"
            direction = (1, 0)
        elif gc == c0 and r0 < gr < r1:
            side = "left"
            direction = (0, -1)
        elif gc == c1 and r0 < gr < r1:
            side = "right"
            direction = (0, 1)
        else:
            continue

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        inner_mask = (inner != BG_COLOR) & (inner != WALL_COLOR)
        seed_rel = np.argwhere(inner_mask)
        seed_count = int(seed_rel.shape[0])
        if not 1 <= seed_count <= 4:
            continue
        seed_colors = {int(inner[rr, cc]) for rr, cc in seed_rel}
        if len(seed_colors) != 1:
            continue
        seed_color = seed_colors.pop()
        seed_cells = [(r0 + 1 + int(rr), c0 + 1 + int(cc)) for rr, cc in seed_rel]
        infos.append(
            {
                "bbox": (r0, r1, c0, c1),
                "gap": gap,
                "side": side,
                "direction": direction,
                "seed_cells": seed_cells,
                "seed_count": seed_count,
                "seed_color": seed_color,
            }
        )

    infos.sort(key=lambda info: (info["bbox"][0], info["bbox"][2]))
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count the colored seeds inside each broken wall and grow an equally long line outward from its gap.

    Input structure:
    - The background is black.
    - One or more maroon one-pixel-thick rectangular walls appear on the grid.
    - Each wall is almost closed: exactly one border pixel is missing on one side,
      creating a single one-cell gap on the top, bottom, left, or right edge.
    - Inside each wall sit 1-4 single-pixel seeds of one shared non-maroon color.

    Rule:
    - For each wall, locate its unique missing border cell and identify which side
      of the rectangle contains that gap.
    - Count the interior seed pixels and read their shared color.
    - Erase those interior seed pixels to black.
    - Starting from the first cell just outside the gap, draw a straight outward
      segment whose direction is the wall normal at the gap and whose length equals
      the number of interior seeds.
    - Keep every maroon wall pixel unchanged.

    Key insight:
    - The gap orientation determines the growth direction, while the number of
      enclosed seeds determines the segment length; the seed positions themselves
      do not control the line geometry.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for info in _parse_walls(g):
        for r, c in info["seed_cells"]:
            out[r, c] = BG_COLOR
        for r, c in _line_cells(info["gap"], info["direction"], info["seed_count"]):
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = info["seed_color"]
    return out


def _spaced_seed_cells(inner_h: int, inner_w: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    patterns = []
    for row_offset in (0, 1):
        for col_offset in (0, 1):
            coords = [
                (rr, cc)
                for rr in range(row_offset, inner_h, 2)
                for cc in range(col_offset, inner_w, 2)
            ]
            if len(coords) >= count:
                patterns.append(coords)
    if not patterns:
        return None
    candidates = rng.choice(patterns).copy()
    rng.shuffle(candidates)
    return candidates[:count]


def _sample_wall_plan(rng: random.Random, h: int, w: int, seed_color: int) -> dict | None:
    seed_count = rng.randint(1, 4)
    side = rng.choice(SIDES)
    for _ in range(40):
        outer_h = rng.randint(*OUTER_SIZE_RANGE)
        outer_w = rng.randint(*OUTER_SIZE_RANGE)
        inner_h, inner_w = outer_h - 2, outer_w - 2
        seeds_rel = _spaced_seed_cells(inner_h, inner_w, seed_count, rng)
        if seeds_rel is None:
            continue

        if side == "top":
            r0_min, r0_max = seed_count, h - outer_h
            c0_min, c0_max = 0, w - outer_w
        elif side == "bottom":
            r0_min, r0_max = 0, h - outer_h - seed_count
            c0_min, c0_max = 0, w - outer_w
        elif side == "left":
            r0_min, r0_max = 0, h - outer_h
            c0_min, c0_max = seed_count, w - outer_w
        else:
            r0_min, r0_max = 0, h - outer_h
            c0_min, c0_max = 0, w - outer_w - seed_count

        if r0_min > r0_max or c0_min > c0_max:
            continue

        r0 = rng.randint(r0_min, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        if side in {"top", "bottom"}:
            gap = (r0 if side == "top" else r1, rng.randint(c0 + 1, c1 - 1))
            direction = (-1, 0) if side == "top" else (1, 0)
        else:
            gap = (rng.randint(r0 + 1, r1 - 1), c0 if side == "left" else c1)
            direction = (0, -1) if side == "left" else (0, 1)

        seed_cells = [(r0 + 1 + rr, c0 + 1 + cc) for rr, cc in seeds_rel]
        return {
            "bbox": (r0, r1, c0, c1),
            "gap": gap,
            "side": side,
            "direction": direction,
            "seed_cells": seed_cells,
            "seed_count": seed_count,
            "seed_color": seed_color,
        }
    return None


def _core_cells(plan: dict) -> set[tuple[int, int]]:
    r0, r1, c0, c1 = plan["bbox"]
    cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
    cells.update(_line_cells(plan["gap"], plan["direction"], plan["seed_count"]))
    return cells


def _expanded_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(max(0, r - pad), min(h - 1, r + pad) + 1):
            for nc in range(max(0, c - pad), min(w - 1, c + pad) + 1):
                expanded.add((nr, nc))
    return expanded


def _can_place(plan: dict, reserved: np.ndarray) -> bool:
    h, w = reserved.shape
    return all(not reserved[r, c] for r, c in _expanded_cells(_core_cells(plan), h, w, pad=1))


def _draw_plan(grid: np.ndarray, plan: dict) -> None:
    r0, r1, c0, c1 = plan["bbox"]
    wall = _perimeter_cells(r0, r1, c0, c1)
    wall.remove(plan["gap"])
    for r, c in wall:
        grid[r, c] = WALL_COLOR
    for r, c in plan["seed_cells"]:
        grid[r, c] = plan["seed_color"]


def _reserve_plan(reserved: np.ndarray, plan: dict) -> None:
    h, w = reserved.shape
    for r, c in _expanded_cells(_core_cells(plan), h, w, pad=1):
        reserved[r, c] = True


def _matches_plan(info: dict, plan: dict) -> bool:
    same_bbox = info["bbox"] == plan["bbox"]
    same_gap = info["gap"] == plan["gap"]
    same_color = info["seed_color"] == plan["seed_color"]
    same_count = info["seed_count"] == plan["seed_count"]
    same_seeds = set(info["seed_cells"]) == set(plan["seed_cells"])
    return same_bbox and same_gap and same_color and same_count and same_seeds


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_SIZE_RANGE)
    w = rng.randint(*GRID_SIZE_RANGE)
    grid = np.full((h, w), BG_COLOR, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    max_walls = min(4, max(1, (h * w) // 120))
    num_walls = rng.randint(1, max_walls)
    seed_colors = rng.sample(SEED_COLORS, k=num_walls)
    plans: list[dict] = []

    for seed_color in seed_colors:
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            plan = _sample_wall_plan(rng, h, w, seed_color)
            if plan is None or not _can_place(plan, reserved):
                continue
            _draw_plan(grid, plan)
            _reserve_plan(reserved, plan)
            plans.append(plan)
            placed = True
            break
        if not placed:
            return None

    infos = _parse_walls(grid)
    if len(infos) != len(plans):
        return None

    used = [False] * len(plans)
    for info in infos:
        matched = False
        for idx, plan in enumerate(plans):
            if not used[idx] and _matches_plan(info, plan):
                used[idx] = True
                matched = True
                break
        if not matched:
            return None

    out = main(grid)
    for info in infos:
        for r, c in info["seed_cells"]:
            if out[r, c] != BG_COLOR:
                return None
        for r, c in _line_cells(info["gap"], info["direction"], info["seed_count"]):
            if not (0 <= r < h and 0 <= c < w):
                return None
            if out[r, c] != info["seed_color"]:
                return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
