import random

import numpy as np


def _repeat_2x2(sprite: np.ndarray) -> np.ndarray:
    return np.tile(sprite, (2, 2))


def main(grid: np.ndarray) -> np.ndarray:
    """Build a four-quadrant rotational tiling of the input square.

    Input structure
    - The input is a square grid treated as one complete sprite.

    Rule
    - Each output quadrant is a 2x2 repetition of a rotated copy of the whole input.
    - The quadrant rotations are upper-left 180 degrees, upper-right 90 degrees
      clockwise, lower-left 90 degrees counterclockwise, and lower-right unchanged.

    Key insight
    - The output is a 4x4 grid of original-size blocks, not a per-pixel scale-up.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h > 0

    upper_left = _repeat_2x2(np.rot90(g, 2))
    upper_right = _repeat_2x2(np.rot90(g, -1))
    lower_left = _repeat_2x2(np.rot90(g, 1))
    lower_right = _repeat_2x2(g)
    return np.block([[upper_left, upper_right], [lower_left, lower_right]])


# --------------- generator ---------------

_ALL_COLORS = tuple(range(10))
_MIN_SIDE = 2
_MAX_SIDE = 7
_MAX_NON_BG_COLORS = 4
_RETRY_LIMIT = 300


def _expected_output(grid: np.ndarray) -> np.ndarray:
    return np.block(
        [
            [_repeat_2x2(np.rot90(grid, 2)), _repeat_2x2(np.rot90(grid, -1))],
            [_repeat_2x2(np.rot90(grid, 1)), _repeat_2x2(grid)],
        ]
    )


def _empty_sprite(side: int, bg: int) -> np.ndarray:
    return np.full((side, side), bg, dtype=int)


def _sample_palette(bg: int, rng: random.Random) -> list[int]:
    colors = [c for c in _ALL_COLORS if c != bg]
    rng.shuffle(colors)
    count = rng.randint(1, _MAX_NON_BG_COLORS)
    return colors[:count]


def _make_scatter(side: int, bg: int, colors: list[int], rng: random.Random) -> np.ndarray:
    grid = _empty_sprite(side, bg)
    cells = [(r, c) for r in range(side) for c in range(side)]
    count = rng.randint(2, min(len(cells), max(2, side + 3)))
    positions = rng.sample(cells, count)
    assignments = [colors[i % len(colors)] for i in range(count)]
    rng.shuffle(assignments)
    for (r, c), color in zip(positions, assignments):
        grid[r, c] = color
    return grid


def _make_rectangles(side: int, bg: int, colors: list[int], rng: random.Random) -> np.ndarray:
    grid = _empty_sprite(side, bg)
    occupied = np.zeros((side, side), dtype=bool)
    target = rng.randint(2, min(4, side + 1))
    placed = 0

    for _ in range(target * 12):
        if placed >= target:
            break
        height = rng.randint(1, side)
        width = rng.randint(1, side)
        if side > 2 and height == side and width == side:
            continue
        r0 = rng.randint(0, side - height)
        c0 = rng.randint(0, side - width)
        if occupied[r0 : r0 + height, c0 : c0 + width].any():
            continue

        grid[r0 : r0 + height, c0 : c0 + width] = colors[placed % len(colors)]
        occupied[r0 : r0 + height, c0 : c0 + width] = True
        placed += 1

    if placed == 0:
        return _make_scatter(side, bg, colors, rng)
    return grid


def _grow_connected_mask(side: int, target: int, rng: random.Random) -> np.ndarray:
    cells = {(rng.randrange(side), rng.randrange(side))}
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while len(cells) < target:
        frontier = set()
        for r, c in cells:
            for dr, dc in neighbors:
                nr, nc = r + dr, c + dc
                if 0 <= nr < side and 0 <= nc < side and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(tuple(frontier)))

    mask = np.zeros((side, side), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_blob(side: int, bg: int, colors: list[int], rng: random.Random) -> np.ndarray:
    grid = _empty_sprite(side, bg)
    max_area = side * side if side <= 2 else side * side - 1
    min_area = min(max_area, max(2, side - 1))
    target = rng.randint(min_area, max_area)
    mask = _grow_connected_mask(side, target, rng)

    mode = rng.choice(("sum", "row", "col", "checker"))
    color_count = len(colors)
    for r, c in np.argwhere(mask):
        if color_count == 1:
            idx = 0
        elif mode == "sum":
            idx = (int(r) + int(c)) % color_count
        elif mode == "row":
            idx = int(r) % color_count
        elif mode == "col":
            idx = int(c) % color_count
        else:
            idx = ((int(r) // 2) + (int(c) // 2)) % color_count
        grid[int(r), int(c)] = colors[idx]
    return grid


def _make_bands(side: int, bg: int, colors: list[int], rng: random.Random) -> np.ndarray:
    grid = _empty_sprite(side, bg)
    rows = rng.sample(range(side), rng.randint(1, side))
    cols = rng.sample(range(side), rng.randint(1, side))

    for i, r in enumerate(rows):
        grid[r, :] = colors[i % len(colors)]
    for i, c in enumerate(cols):
        grid[:, c] = colors[(i + 1) % len(colors)] if len(colors) > 1 else colors[0]

    if rng.random() < 0.5:
        diag_color = colors[-1]
        if rng.random() < 0.5:
            for i in range(side):
                grid[i, i] = diag_color
        else:
            for i in range(side):
                grid[i, side - 1 - i] = diag_color
    return grid


def _make_mirror(side: int, bg: int, colors: list[int], rng: random.Random) -> np.ndarray:
    grid = _empty_sprite(side, bg)
    half = (side + 1) // 2
    positions = [(r, c) for r in range(side) for c in range(half)]
    count = rng.randint(2, min(len(positions), max(2, side + half)))
    chosen = rng.sample(positions, count)
    symmetry = rng.choice(("vertical", "horizontal", "both"))

    for i, (r, c) in enumerate(chosen):
        color = colors[i % len(colors)]
        mirrors = {(r, c)}
        if symmetry in ("vertical", "both"):
            mirrors.add((r, side - 1 - c))
        if symmetry in ("horizontal", "both"):
            mirrors.add((side - 1 - r, c))
        if symmetry == "both":
            mirrors.add((side - 1 - r, side - 1 - c))
        for rr, cc in mirrors:
            grid[rr, cc] = color
    return grid


def _is_rotationally_unique(grid: np.ndarray) -> bool:
    return not any(np.array_equal(grid, np.rot90(grid, k)) for k in (1, 2, 3))


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    side = rng.randint(_MIN_SIDE, _MAX_SIDE)
    bg = rng.choice(_ALL_COLORS)
    colors = _sample_palette(bg, rng)
    mode = rng.choice(("scatter", "rectangles", "blob", "bands", "mirror"))

    if mode == "scatter":
        grid = _make_scatter(side, bg, colors, rng)
    elif mode == "rectangles":
        grid = _make_rectangles(side, bg, colors, rng)
    elif mode == "blob":
        grid = _make_blob(side, bg, colors, rng)
    elif mode == "bands":
        grid = _make_bands(side, bg, colors, rng)
    else:
        grid = _make_mirror(side, bg, colors, rng)

    distinct = np.unique(grid)
    if distinct.size < 2:
        return None

    if mode != "mirror" and rng.random() < 0.75 and not _is_rotationally_unique(grid):
        return None

    expected = _expected_output(grid)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
