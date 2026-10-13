from __future__ import annotations

import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(10))

GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 30)
FENCE_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
FENCE_H_RANGE = (5, 9)
FENCE_W_RANGE = (5, 10)
MAX_SEED_COUNT = 6
PLACEMENT_RETRIES = 250
GENERATE_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, radius: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _analyze_fence(
    grid: np.ndarray,
    color: int,
    coords: list[tuple[int, int]],
    bg: int,
) -> dict[str, object] | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return None

    comp = set(coords)
    border = _border_cells(r0, c0, r1, c1)
    if comp - border:
        return None

    missing = border - comp
    if len(missing) != 1:
        return None
    gap = next(iter(missing))
    gr, gc = gap
    if int(grid[gr, gc]) != bg:
        return None
    if (gr in (r0, r1)) and (gc in (c0, c1)):
        return None

    if gr == r0:
        direction = (-1, 0)
    elif gr == r1:
        direction = (1, 0)
    elif gc == c0:
        direction = (0, -1)
    else:
        direction = (0, 1)

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if np.any(interior == color):
        return None

    core = grid[r0 + 2 : r1 - 1, c0 + 2 : c1 - 1]
    if core.size == 0:
        return None

    seed_mask = (interior != bg) & (interior != color)
    if not np.any(seed_mask):
        return None

    allowed = np.zeros_like(interior, dtype=bool)
    allowed[1:-1, 1:-1] = True
    if np.any(seed_mask & ~allowed):
        return None

    seed_colors = np.unique(interior[seed_mask])
    if seed_colors.size != 1:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "direction": direction,
        "seed_color": int(seed_colors[0]),
        "seed_count": int(np.count_nonzero(seed_mask)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    For each rectangular fence with one border gap, count the interior seeds and
    emit a ray of that length from the first exterior cell beyond the gap.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _get_bg(g)
    out = g.copy()

    for color in map(int, np.unique(g)):
        if color == bg:
            continue
        for comp in _components(g == color):
            info = _analyze_fence(g, color, comp, bg)
            if info is None:
                continue
            gr, gc = info["gap"]
            dr, dc = info["direction"]
            seed_color = int(info["seed_color"])
            seed_count = int(info["seed_count"])
            for step in range(1, seed_count + 1):
                rr = gr + dr * step
                cc = gc + dc * step
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = seed_color
    return out


def _reserve_bbox(r0: int, c0: int, r1: int, c1: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(max(0, r0 - 1), min(h, r1 + 2)):
        for c in range(max(0, c0 - 1), min(w, c1 + 2)):
            cells.add((r, c))
    return cells


def _place_one_fence(
    rng: random.Random,
    h: int,
    w: int,
    reserved: set[tuple[int, int]],
    seed_color: int,
) -> dict[str, object] | None:
    for _ in range(PLACEMENT_RETRIES):
        box_h = rng.randint(*FENCE_H_RANGE)
        box_w = rng.randint(*FENCE_W_RANGE)
        core_h = box_h - 4
        core_w = box_w - 4
        if core_h <= 0 or core_w <= 0:
            continue

        seed_count = rng.randint(1, min(MAX_SEED_COUNT, core_h * core_w))
        side = rng.choice(SIDES)

        if side == "top":
            r0_min, r0_max = seed_count + 1, h - box_h - 2
            c0_min, c0_max = 1, w - box_w - 2
        elif side == "bottom":
            r0_min, r0_max = 1, h - box_h - seed_count - 2
            c0_min, c0_max = 1, w - box_w - 2
        elif side == "left":
            r0_min, r0_max = 1, h - box_h - 2
            c0_min, c0_max = seed_count + 1, w - box_w - 2
        else:
            r0_min, r0_max = 1, h - box_h - 2
            c0_min, c0_max = 1, w - box_w - seed_count - 2

        if r0_min > r0_max or c0_min > c0_max:
            continue

        r0 = rng.randint(r0_min, r0_max)
        c0 = rng.randint(c0_min, c0_max)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1

        if side == "top":
            gap = (r0, rng.randint(c0 + 1, c1 - 1))
            direction = (-1, 0)
        elif side == "bottom":
            gap = (r1, rng.randint(c0 + 1, c1 - 1))
            direction = (1, 0)
        elif side == "left":
            gap = (rng.randint(r0 + 1, r1 - 1), c0)
            direction = (0, -1)
        else:
            gap = (rng.randint(r0 + 1, r1 - 1), c1)
            direction = (0, 1)

        gr, gc = gap
        dr, dc = direction
        ray_cells = {(gr + dr * step, gc + dc * step) for step in range(1, seed_count + 1)}
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in ray_cells):
            continue

        border = _border_cells(r0, c0, r1, c1)
        border.remove(gap)
        core_cells = [(r, c) for r in range(r0 + 2, r1 - 1) for c in range(c0 + 2, c1 - 1)]
        if len(core_cells) < seed_count:
            continue
        seed_cells = set(rng.sample(core_cells, seed_count))

        reserve = _reserve_bbox(r0, c0, r1, c1, h, w) | _expand_cells(ray_cells, h, w, radius=1)
        if reserve & reserved:
            continue

        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "direction": direction,
            "seed_color": seed_color,
            "seed_cells": seed_cells,
            "ray_cells": ray_cells,
            "reserve": reserve,
        }
    return None


def _render_case(
    h: int,
    w: int,
    bg: int,
    fence_color: int,
    fences: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), bg, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    for item in fences:
        r0, c0, r1, c1 = item["bbox"]
        gap = item["gap"]
        seed_color = int(item["seed_color"])
        border = _border_cells(r0, c0, r1, c1)
        border.remove(gap)
        for r, c in border:
            inp[r, c] = fence_color
            out[r, c] = fence_color
        for r, c in item["seed_cells"]:
            inp[r, c] = seed_color
            out[r, c] = seed_color
        for r, c in item["ray_cells"]:
            out[r, c] = seed_color
    return inp, out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)

    bg = rng.choice(PALETTE)
    remaining = [color for color in PALETTE if color != bg]
    fence_color = rng.choice(remaining)
    remaining.remove(fence_color)

    max_fences = min(len(remaining), max(FENCE_COUNT_CHOICES))
    count_choices = tuple(choice for choice in FENCE_COUNT_CHOICES if choice <= max_fences)
    if not count_choices:
        return None
    fence_count = rng.choice(count_choices)
    seed_colors = rng.sample(remaining, fence_count)

    reserved: set[tuple[int, int]] = set()
    fences: list[dict[str, object]] = []
    for seed_color in seed_colors:
        placed = _place_one_fence(rng, h, w, reserved, seed_color)
        if placed is None:
            return None
        reserved |= placed["reserve"]
        fences.append(placed)

    inp, expected = _render_case(h, w, bg, fence_color, fences)
    if _get_bg(inp) != bg:
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
