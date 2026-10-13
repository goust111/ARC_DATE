from __future__ import annotations

import random

import numpy as np


RED = 2
PINK = 6
DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
OBSTACLE_COLORS = (1, 3, 4, 5, 7, 8, 9)
BACKGROUND_CHOICES = (0, 1, 3, 4, 5, 7, 8, 9)
GENERATE_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _pink_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    red_cells = np.argwhere(grid == RED)
    pink_cells = np.argwhere(grid == PINK)
    assert len(red_cells) == 1, "expected one red marker"
    assert len(pink_cells) == 4, "expected a four-cell pink trail"

    red = tuple(map(int, red_cells[0]))
    pink = [tuple(map(int, rc)) for rc in pink_cells]
    pink_set = set(pink)

    endpoints = []
    for r, c in pink:
        deg = 0
        for dr, dc in DIRS8:
            if (r + dr, c + dc) in pink_set:
                deg += 1
        if deg == 1:
            endpoints.append((r, c))
    assert len(endpoints) == 2, "pink trail must be a straight 8-connected segment"

    adjacent = [p for p in endpoints if max(abs(p[0] - red[0]), abs(p[1] - red[1])) == 1]
    assert len(adjacent) == 1, "exactly one trail endpoint must touch red"
    near = adjacent[0]
    dr = near[0] - red[0]
    dc = near[1] - red[1]
    assert (dr, dc) in DIRS8, "trail direction must be 8-neighbor"

    expected = {(red[0] + k * dr, red[1] + k * dc) for k in range(1, 5)}
    assert expected == pink_set, "pink trail must be four consecutive cells"
    return red, (dr, dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase the first border/obstacle pixel hit by the ray opposite the pink trail.

    Input structure:
    - One unique red marker.
    - Four pink cells forming a straight 8-connected trail from the red marker outward.
    - In the opposite direction, after at least one background cell, the first non-background
      cell on that ray is the pixel to erase.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    red, trail_dir = _pink_direction(g)
    dr, dc = -trail_dir[0], -trail_dir[1]

    out = g.copy()
    r, c = red
    seen_bg = False
    while True:
        r += dr
        c += dc
        assert _in_bounds(g.shape, r, c), "reverse ray must hit a target before leaving the grid"
        if int(g[r, c]) == bg:
            seen_bg = True
            continue
        assert seen_bg, "target must be behind at least one background cell"
        out[r, c] = bg
        return out


def _boundary_positions(h: int, w: int, dr: int, dc: int) -> list[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    if dr == -1:
        for c in range(w):
            pts.add((h - 1, c))
    elif dr == 1:
        for c in range(w):
            pts.add((0, c))
    if dc == -1:
        for r in range(h):
            pts.add((r, w - 1))
    elif dc == 1:
        for r in range(h):
            pts.add((r, 0))
    return list(pts)


def _ray_cells_until_edge(
    start: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]
) -> list[tuple[int, int]]:
    r, c = start
    dr, dc = step
    cells = []
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, r, c):
            return cells
        cells.append((r, c))


def _place_decoys(
    grid: np.ndarray,
    forbidden: set[tuple[int, int]],
    bg: int,
    target_color: int,
) -> None:
    h, w = grid.shape
    decoy_count = random.randint(0, 3)
    colors = [c for c in OBSTACLE_COLORS if c not in {bg, target_color}]
    for _ in range(decoy_count):
        if random.random() < 0.5:
            r = random.randrange(h)
            c = random.randrange(w)
            if (r, c) in forbidden or int(grid[r, c]) != bg:
                continue
            grid[r, c] = random.choice(colors)
            forbidden.add((r, c))
            continue

        length = random.randint(2, 4)
        dr, dc = random.choice(DIRS8)
        if dr == 0 and dc == 0:
            continue
        r0 = random.randrange(h)
        c0 = random.randrange(w)
        cells = [(r0 + k * dr, c0 + k * dc) for k in range(length)]
        if not all(_in_bounds(grid.shape, r, c) for r, c in cells):
            continue
        if any((r, c) in forbidden or int(grid[r, c]) != bg for r, c in cells):
            continue
        color = random.choice(colors)
        for r, c in cells:
            grid[r, c] = color
            forbidden.add((r, c))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(9, 16)
    w = random.randint(9, 16)
    bg = random.choice(BACKGROUND_CHOICES)
    grid = np.full((h, w), bg, dtype=int)

    trail_dr, trail_dc = random.choice(DIRS8)
    rev_dr, rev_dc = -trail_dr, -trail_dc
    target_mode = random.choice(("border", "obstacle"))
    target_color = random.choice([c for c in OBSTACLE_COLORS if c != bg])

    if target_mode == "border":
        target = random.choice(_boundary_positions(h, w, trail_dr, trail_dc))
        ray_before_target = _ray_cells_until_edge(target, (trail_dr, trail_dc), (h, w))
        if len(ray_before_target) < 2:
            return None
        gap = random.randint(1, len(ray_before_target) - 1)
        red = ray_before_target[gap]
    else:
        boundary = random.choice(_boundary_positions(h, w, trail_dr, trail_dc))
        full_ray = _ray_cells_until_edge(boundary, (trail_dr, trail_dc), (h, w))
        if len(full_ray) < 4:
            return None
        obstacle_dist = random.randint(2, len(full_ray) - 2)
        target = full_ray[obstacle_dist - 1]
        remaining = full_ray[obstacle_dist:]
        if len(remaining) < 5:
            return None
        red = random.choice(remaining)

    trail_cells = [(red[0] + k * trail_dr, red[1] + k * trail_dc) for k in range(1, 5)]
    if not all(_in_bounds((h, w), r, c) for r, c in trail_cells):
        return None

    critical = {red, target, *trail_cells}
    if len(critical) != 6:
        return None

    reverse_ray = _ray_cells_until_edge(red, (rev_dr, rev_dc), (h, w))
    if target not in reverse_ray:
        return None
    target_idx = reverse_ray.index(target)
    if target_idx < 1:
        return None
    if any(cell in critical - {target} for cell in reverse_ray[: target_idx + 1]):
        return None

    grid[red] = RED
    for cell in trail_cells:
        grid[cell] = PINK
    grid[target] = target_color

    forbidden = set(critical)
    for cell in reverse_ray[:target_idx]:
        forbidden.add(cell)
    _place_decoys(grid, forbidden, bg, target_color)

    try:
        out = main(grid)
    except Exception:
        return None

    expected = grid.copy()
    expected[target] = bg
    if not np.array_equal(out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
