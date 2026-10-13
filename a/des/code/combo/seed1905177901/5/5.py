import random

import numpy as np


BG_COLOR = 8
GRID_SIDES = (7, 9, 11, 13)
LAYOUTS = ("row_phase", "col_phase")
GEN_RETRIES = 300
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
OPPOSITE = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _quadrant_slices(side: int) -> dict[str, tuple[slice, slice]]:
    half = side // 2
    return {
        "tl": (slice(0, half), slice(0, half)),
        "tr": (slice(0, half), slice(half + 1, side)),
        "bl": (slice(half + 1, side), slice(0, half)),
        "br": (slice(half + 1, side), slice(half + 1, side)),
    }


def _swap_colors(arr: np.ndarray, color_a: int, color_b: int) -> np.ndarray:
    out = arr.copy()
    mask_a = out == color_a
    mask_b = out == color_b
    out[mask_a] = color_b
    out[mask_b] = color_a
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the missing quadrant of an odd square using 180-degree symmetry with color inversion.

    The grid is split into four corner quadrants around a background center row and column.
    Exactly one quadrant is entirely background. Copy the opposite quadrant through the
    center (a 180-degree rotation) and swap the two foreground colors while pasting it into
    the missing quadrant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1

    bg = _background_color(g)
    fg_colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert len(fg_colors) == 2
    color_a, color_b = fg_colors

    slices = _quadrant_slices(h)
    quadrants = {name: g[row_slice, col_slice] for name, (row_slice, col_slice) in slices.items()}
    missing = [name for name, quad in quadrants.items() if np.all(quad == bg)]
    assert len(missing) == 1
    missing_name = missing[0]
    source_name = OPPOSITE[missing_name]

    source = np.rot90(quadrants[source_name], 2)
    filled = _swap_colors(source, color_a, color_b)

    out = g.copy()
    row_slice, col_slice = slices[missing_name]
    out[row_slice, col_slice] = filled
    return out


def _neighbors(cell: tuple[int, int], side: int) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in NEIGHBORS4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < side and 0 <= cc < side:
            out.append((rr, cc))
    return out


def _random_connected_mask(side: int) -> np.ndarray:
    area = side * side
    target_lo = max(3, side)
    target_hi = min(area - 1, max(target_lo, (2 * area) // 3))
    target = random.randint(target_lo, target_hi)

    start = (random.randrange(side), random.randrange(side))
    cells = {start}
    frontier = set(_neighbors(start, side))
    while len(cells) < target:
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        if pick in cells:
            continue
        cells.add(pick)
        for nxt in _neighbors(pick, side):
            if nxt not in cells:
                frontier.add(nxt)

    mask = np.zeros((side, side), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _connected_subset(mask: np.ndarray, target: int) -> set[tuple[int, int]]:
    coords = {tuple(coord) for coord in np.argwhere(mask)}
    seed = random.choice(tuple(coords))
    subset = {seed}
    frontier = {nbr for nbr in _neighbors(seed, mask.shape[0]) if nbr in coords}
    while len(subset) < target:
        pick = random.choice(tuple(frontier))
        frontier.remove(pick)
        if pick in subset:
            continue
        subset.add(pick)
        for nxt in _neighbors(pick, mask.shape[0]):
            if nxt in coords and nxt not in subset:
                frontier.add(nxt)
    return subset


def _paint_mask(mask: np.ndarray, colors: tuple[int, int]) -> np.ndarray:
    color_a, color_b = colors
    coords = [tuple(coord) for coord in np.argwhere(mask)]
    assert len(coords) >= 2

    lo = max(1, len(coords) // 4)
    hi = min(len(coords) - 1, len(coords) - len(coords) // 4)
    if lo > hi:
        lo, hi = 1, len(coords) - 1
    cluster_size = random.randint(lo, hi)
    cluster = _connected_subset(mask, cluster_size)

    if random.random() < 0.5:
        first, second = color_a, color_b
    else:
        first, second = color_b, color_a

    patch = np.full(mask.shape, BG_COLOR, dtype=int)
    for r, c in coords:
        patch[r, c] = first if (r, c) in cluster else second
    return patch


def _build_solution(base: np.ndarray, colors: tuple[int, int], layout: str) -> np.ndarray:
    side = base.shape[0] * 2 + 1
    grid = np.full((side, side), BG_COLOR, dtype=int)
    slices = _quadrant_slices(side)

    rot0 = base
    rot_cw = np.rot90(base, -1)
    rot_180 = np.rot90(base, 2)
    rot_ccw = np.rot90(base, 1)
    swapped_180 = _swap_colors(rot_180, colors[0], colors[1])

    if layout == "row_phase":
        quads = {
            "tl": rot0,
            "tr": rot_cw,
            "br": swapped_180,
            "bl": _swap_colors(rot_ccw, colors[0], colors[1]),
        }
    else:
        quads = {
            "tl": rot0,
            "tr": _swap_colors(rot_cw, colors[0], colors[1]),
            "br": swapped_180,
            "bl": rot_ccw,
        }

    for name, quad in quads.items():
        row_slice, col_slice = slices[name]
        grid[row_slice, col_slice] = quad
    return grid


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(GRID_SIDES)
    half = side // 2
    colors = tuple(random.sample([c for c in range(10) if c != BG_COLOR], 2))
    layout = random.choice(LAYOUTS)

    base_mask = _random_connected_mask(half)
    if np.count_nonzero(base_mask) < 2:
        return None
    base = _paint_mask(base_mask, colors)

    solution = _build_solution(base, colors, layout)
    missing_name = random.choice(tuple(OPPOSITE))
    row_slice, col_slice = _quadrant_slices(side)[missing_name]
    puzzle = solution.copy()
    puzzle[row_slice, col_slice] = BG_COLOR

    if len([name for name, (rs, cs) in _quadrant_slices(side).items() if np.all(puzzle[rs, cs] == BG_COLOR)]) != 1:
        return None
    if len(set(int(color) for color in np.unique(puzzle) if int(color) != BG_COLOR)) != 2:
        return None
    if not np.array_equal(main(puzzle), solution):
        return None
    return puzzle


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
