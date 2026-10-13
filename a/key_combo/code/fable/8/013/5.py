import random

import numpy as np


GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(np.bincount(np.asarray(grid, dtype=int).ravel()).argmax())


def _detect_solid_square(mask: np.ndarray) -> tuple[int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h != w or coords.shape[0] != h * w:
        return None
    return int(r0), int(c0), h


def _rotate_top_left(r: int, c: int, side: int, shape: tuple[int, int], k: int) -> tuple[int, int]:
    h, w = shape
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return w - (c + side), r
    if k == 2:
        return h - (r + side), w - (c + side)
    return c, h - (r + side)


def _parse_canonical_candidate(
    grid: np.ndarray,
    bg: int,
    mover_color: int,
    river_color: int,
) -> tuple[int, int, tuple[int, int, int], tuple[int, int, int]] | None:
    start = _detect_solid_square(grid == mover_color)
    if start is None:
        return None

    sr, sc, side = start
    traversable = (grid == mover_color) | (grid == river_color)

    if sr > 0 and bool(np.any(traversable[:sr, sc : sc + side])):
        return None
    if sc + side >= grid.shape[1] or not bool(traversable[sr, sc + side]):
        return None

    widths: list[int] = []
    c = sc
    while c < grid.shape[1] and bool(traversable[sr, c]):
        height = 0
        while sr + height < grid.shape[0] and bool(traversable[sr + height, c]):
            height += 1
        if height <= 0:
            return None
        if bool(np.any(traversable[:sr, c])) or bool(np.any(traversable[sr + height :, c])):
            return None
        widths.append(height)
        c += 1

    if len(widths) < side + 1:
        return None
    if min(widths[:side]) < side:
        return None

    final_side = min(widths)
    end_col = sc + len(widths) - 1
    end = (sr, end_col - final_side + 1, final_side)
    return mover_color, river_color, start, end


def _solve_canonical(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]] | None:
    bg = _get_bg(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    if len(colors) != 2:
        return None

    candidates = []
    for mover_color in colors:
        river_color = colors[0] if colors[1] == mover_color else colors[1]
        parsed = _parse_canonical_candidate(grid, bg, mover_color, river_color)
        if parsed is not None:
            candidates.append(parsed)

    if len(candidates) != 1:
        return None

    mover_color, river_color, start, end = candidates[0]
    out = np.array(grid, dtype=int, copy=True)
    sr, sc, side = start
    er, ec, final_side = end
    out[sr : sr + side, sc : sc + side] = river_color
    out[er : er + final_side, ec : ec + final_side] = mover_color
    return out, start, end


def _generate_profile(start_side: int) -> list[int]:
    segment_count = random.randint(3, 5)
    final_side = random.randint(max(2, start_side - 3), start_side)
    widths = [start_side]
    for _ in range(segment_count - 2):
        widths.append(random.randint(final_side, start_side))
    widths.append(final_side)

    if final_side < start_side and min(widths) == start_side:
        widths[random.randint(1, len(widths) - 2)] = random.randint(final_side, start_side - 1)
        widths[-1] = final_side

    return widths


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    river_color, mover_color = random.sample(range(1, 10), 2)

    start_side = random.randint(4, 7)
    widths = _generate_profile(start_side)
    lengths = [random.randint(2, 5) for _ in widths]
    lengths[0] = random.randint(start_side + 1, start_side + 5)

    spur_len = random.randint(0, 3)
    spur_side = random.randint(1, start_side - 1) if spur_len else 0

    total_cols = sum(lengths)
    max_side = max(widths + ([spur_side] if spur_len else []))
    height = max(12, max_side + 4)
    width = max(12, 2 + spur_len + total_cols + 2)
    if height > 30 or width > 30:
        return None

    sr = 2
    sc = 2 + spur_len
    grid = np.full((height, width), bg, dtype=int)

    for i in range(spur_len):
        col = sc - spur_len + i
        grid[sr : sr + spur_side, col] = river_color

    col = sc
    for band_side, band_len in zip(widths, lengths):
        for _ in range(band_len):
            grid[sr : sr + band_side, col] = river_color
            col += 1

    grid[sr : sr + start_side, sc : sc + start_side] = mover_color

    rot = random.randint(0, 3)
    rotated = np.rot90(grid, rot)

    solved = main(rotated)
    if solved.shape != rotated.shape:
        return None
    if max(rotated.shape) > 30:
        return None
    return rotated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the solid square down the corridor and shrink it to the smallest forward width before parking it at the dead end.

    Input structure:
    - The background is the dominant color.
    - One river color forms a single corridor with changing width; the corridor may also have a narrower spur behind the start.
    - One distinct solid square sits fully on the corridor at the starting end.

    Rule:
    - Move the square forward along the corridor toward its closed end.
    - Whenever the forward corridor becomes narrower than the current square side, shrink the square to that width.
    - Wider later stretches do not enlarge it again; the final side length is the smallest forward corridor width.
    - Repaint the starting square back to the river color and place the resized square flush against the corridor's dead-end side.

    Key insight:
    - The narrow parts are shaving mouths, not blockers: the result is determined by the minimum forward channel width, while any narrower spur behind the start is irrelevant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    for k in range(4):
        rotated = np.rot90(g, k)
        solved = _solve_canonical(rotated)
        if solved is None:
            continue
        out, _start, _end = solved
        return np.rot90(out, -k)

    raise AssertionError("input does not match the expected corridor family")


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
