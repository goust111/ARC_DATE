from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7)
MIN_ROOM_SIDE = 2
MAX_ROOM_SIDE = 6
MIN_ROOM_COUNT = 2
MAX_ROOM_COUNT = 4
MAX_GENERATION_ATTEMPTS = 300


def _interior_slices(length: int, line_indices: list[int]) -> list[slice]:
    boundaries = [-1] + sorted(int(i) for i in line_indices) + [int(length)]
    slices: list[slice] = []
    for a, b in zip(boundaries, boundaries[1:]):
        start = int(a + 1)
        stop = int(b)
        if start < stop:
            slices.append(slice(start, stop))
    return slices


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-bounded room with the seed-anchored pattern selected by that room's aspect ratio.

    Input structure:
    - Black background is cut by complete grey horizontal and vertical grid lines into rectangular rooms.
    - Each room interior contains exactly one colored seed pixel, and no other non-black cells.
    - The seed colors are non-black and non-grey; grey separator lines remain outside the room interiors.

    Rule:
    - Measure each room's interior width W and height H, excluding grey separator lines.
    - If W > H, fill every other full row of the room with the seed color, anchored on the seed row.
    - If H > W, fill every other full column of the room with the seed color, anchored on the seed column.
    - If H = W, fill the seed-colored parity class of the checkerboard pattern anchored at the seed cell.
    - All unfilled room cells are black and all grey separator lines are preserved.

    Key insight:
    - The room aspect ratio chooses horizontal stripes, vertical stripes, or checkerboard, while the seed position fixes the phase of that pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    grey_rows = [r for r in range(height) if bool(np.all(g[r, :] == GREY))]
    grey_cols = [c for c in range(width) if bool(np.all(g[:, c] == GREY))]
    row_slices = _interior_slices(height, grey_rows)
    col_slices = _interior_slices(width, grey_cols)
    assert row_slices and col_slices, "expected at least one rectangular room"

    out = g.copy()
    for rs in row_slices:
        for cs in col_slices:
            room = g[rs, cs]
            assert not np.any(room == GREY), "grey pixels must be separator lines only"
            seed_mask = (room != BLACK) & (room != GREY)
            seeds = np.argwhere(seed_mask)
            assert len(seeds) == 1, "each room must contain exactly one colored seed"

            local_r, local_c = map(int, seeds[0])
            seed_r = int(rs.start + local_r)
            seed_c = int(cs.start + local_c)
            color = int(room[local_r, local_c])

            room_h = int(rs.stop - rs.start)
            room_w = int(cs.stop - cs.start)
            out[rs, cs] = BLACK

            if room_w > room_h:
                for r in range(rs.start, rs.stop):
                    if (r - seed_r) % 2 == 0:
                        out[r, cs] = color
            elif room_h > room_w:
                for c in range(cs.start, cs.stop):
                    if (c - seed_c) % 2 == 0:
                        out[rs, c] = color
            else:
                rr, cc = np.indices((room_h, room_w))
                parity = (rr + cc - local_r - local_c) % 2 == 0
                patch = np.full((room_h, room_w), BLACK, dtype=int)
                patch[parity] = color
                out[rs, cs] = patch

    return out


def _room_sizes(count: int, required_a: int, required_b: int) -> list[int]:
    sizes = [int(required_a), int(required_b)]
    while len(sizes) < count:
        sizes.append(random.randint(MIN_ROOM_SIDE, MAX_ROOM_SIDE))
    random.shuffle(sizes)
    return sizes


def _boundaries(sizes: list[int]) -> list[int]:
    pos = 0
    result = [0]
    for size in sizes:
        pos += int(size) + 1
        result.append(pos)
    return result


def _paint_grid_lines(grid: np.ndarray, row_bounds: list[int], col_bounds: list[int]) -> None:
    for r in row_bounds:
        grid[int(r), :] = GREY
    for c in col_bounds:
        grid[:, int(c)] = GREY


def _room_intervals(bounds: list[int]) -> list[tuple[int, int]]:
    return [(int(a + 1), int(b)) for a, b in zip(bounds, bounds[1:]) if int(a + 1) < int(b)]


def _has_all_branches(row_sizes: list[int], col_sizes: list[int]) -> bool:
    branches = set()
    for h in row_sizes:
        for w in col_sizes:
            if w > h:
                branches.add("wide")
            elif h > w:
                branches.add("tall")
            else:
                branches.add("square")
    return branches == {"wide", "tall", "square"}


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(MIN_ROOM_COUNT, MAX_ROOM_COUNT)
    col_count = random.randint(MIN_ROOM_COUNT, MAX_ROOM_COUNT)
    a, b = random.sample(range(MIN_ROOM_SIDE, MAX_ROOM_SIDE + 1), 2)

    row_sizes = _room_sizes(row_count, a, b)
    col_sizes = _room_sizes(col_count, a, b)
    if not _has_all_branches(row_sizes, col_sizes):
        return None

    row_bounds = _boundaries(row_sizes)
    col_bounds = _boundaries(col_sizes)
    height = int(row_bounds[-1] + 1)
    width = int(col_bounds[-1] + 1)
    if max(height, width) > 30:
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    _paint_grid_lines(grid, row_bounds, col_bounds)

    row_intervals = _room_intervals(row_bounds)
    col_intervals = _room_intervals(col_bounds)
    for r0, r1 in row_intervals:
        for c0, c1 in col_intervals:
            sr = random.randint(r0, r1 - 1)
            sc = random.randint(c0, c1 - 1)
            grid[sr, sc] = random.choice(SEED_COLORS)

    try:
        out = main(grid)
    except AssertionError:
        return None
    if not isinstance(out, np.ndarray) or out.shape != grid.shape or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid seed807260302/1/4 input")
