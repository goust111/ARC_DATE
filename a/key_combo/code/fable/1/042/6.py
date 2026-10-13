from __future__ import annotations

import random

import numpy as np


class Color:
    BLACK = 0
    RED = 2
    GREEN = 3
    YELLOW = 4
    PINK = 6


MIRRORS = {Color.GREEN, Color.PINK}
GEN_RETRY_LIMIT = 300
REVISIT_PROBABILITY = 0.8


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _entry_direction(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    starts = np.argwhere(grid == Color.RED)
    assert starts.shape[0] == 1, "expected exactly one red entry pixel"
    r, c = map(int, starts[0])
    h, w = map(int, grid.shape)
    on_sides = [r == 0, r == h - 1, c == 0, c == w - 1]
    assert sum(on_sides) == 1, "entry pixel must lie on exactly one edge"
    if r == 0:
        return (r, c), (1, 0)
    if r == h - 1:
        return (r, c), (-1, 0)
    if c == 0:
        return (r, c), (0, 1)
    return (r, c), (0, -1)


def _reflect(color: int, dr: int, dc: int) -> tuple[int, int]:
    if color == Color.GREEN:
        return -dc, -dr
    assert color == Color.PINK, "unexpected mirror color"
    return dc, dr


def _mirror_for_turn(prev_dir: tuple[int, int], next_dir: tuple[int, int]) -> int:
    dr, dc = map(int, prev_dir)
    nr, nc = map(int, next_dir)
    if _reflect(Color.GREEN, dr, dc) == (nr, nc):
        return Color.GREEN
    if _reflect(Color.PINK, dr, dc) == (nr, nc):
        return Color.PINK
    raise AssertionError("turn must be a 90-degree mirror reflection")


def _swap_mirror_colors(grid: np.ndarray) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    slash = out == Color.GREEN
    backslash = out == Color.PINK
    out[slash] = Color.PINK
    out[backslash] = Color.GREEN
    return out


def _rot_coord(coord: tuple[int, int], shape: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = map(int, coord)
    h, w = map(int, shape)
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return w - 1 - c, r
    if k == 2:
        return h - 1 - r, w - 1 - c
    return c, h - 1 - r


def _rotate_scene(
    grid: np.ndarray, mirrors: set[tuple[int, int]], revisited: set[tuple[int, int]], k: int
) -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    out = np.rot90(np.asarray(grid, dtype=int), int(k))
    if int(k) % 2 == 1:
        out = _swap_mirror_colors(out)
    rotated_mirrors = {_rot_coord(coord, grid.shape, int(k)) for coord in mirrors}
    rotated_revisited = {_rot_coord(coord, grid.shape, int(k)) for coord in revisited}
    return out, rotated_mirrors, rotated_revisited


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    valid_colors = set(map(int, np.unique(g)))
    assert valid_colors <= {Color.BLACK, Color.RED, Color.GREEN, Color.PINK}, "unexpected input colors"

    out = g.copy()
    (r, c), (dr, dc) = _entry_direction(g)
    mirror_count = int(np.count_nonzero((g == Color.GREEN) | (g == Color.PINK)))
    max_steps = max(g.shape) * (mirror_count + 2)

    consumed: set[tuple[int, int]] = set()
    mirror_hits: dict[tuple[int, int], int] = {}
    yellow_steps = 0

    for _ in range(max_steps):
        nr = r + dr
        nc = c + dc
        if not _in_bounds(g.shape, nr, nc):
            break

        cell = int(g[nr, nc])
        if cell == Color.BLACK:
            out[nr, nc] = Color.YELLOW
            yellow_steps += 1
        elif cell in MIRRORS:
            pos = (int(nr), int(nc))
            mirror_hits[pos] = mirror_hits.get(pos, 0) + 1
            if pos not in consumed:
                consumed.add(pos)
                dr, dc = _reflect(cell, dr, dc)

        r, c = int(nr), int(nc)
    else:
        raise AssertionError("light path did not exit within the expected segment bound")

    revisited = {coord for coord, count in mirror_hits.items() if count >= 2}
    stats = {
        "used_mirrors": consumed,
        "revisited_mirrors": revisited,
        "mirror_hits": mirror_hits,
        "mirror_colors_hit": {int(g[r0, c0]) for r0, c0 in consumed},
        "yellow_count": yellow_steps,
    }
    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace a border-entering light ray through one-shot diagonal mirrors and paint only the traversed background cells yellow.

    Input structure:
    - The grid has black background, exactly one red entry pixel on a non-corner edge cell, and optional single-pixel mirrors.
    - Green mirror pixels represent "/" reflectors and pink mirror pixels represent "\\" reflectors.
    - The red entry pixel stays on the border and the ray enters perpendicular to that border.

    Rule:
    - Follow the ray cell by cell from the red entry into the grid.
    - Every visited black background cell becomes yellow.
    - On the first hit of a green or pink mirror, turn the ray by the mirror's reflection rule and mark that mirror as consumed.
    - Later hits on the same consumed mirror do not turn the ray; the mirror color stays unchanged, and the ray simply passes through until it exits the grid.

    Key insight:
    - Mirror state matters: the same mirror can be hit twice, but only the first contact reflects, so the correct solution is a stateful ray simulation rather than a static bounce map.
    """
    return _simulate(np.asarray(grid, dtype=int))[0]


def _build_revisit_scene() -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    vertical_sign = random.choice((-1, 1))
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    loop_height = random.randint(2, 5)
    side_gap = random.randint(2, 4)
    right_pad = random.randint(2, 4)

    if vertical_sign == 1:
        start_row = top_pad
        far_row = start_row + loop_height
        h = far_row + bottom_pad + 1
    else:
        far_row = top_pad
        start_row = far_row + loop_height
        h = start_row + bottom_pad + 1

    first_col = random.randint(2, 5)
    detour_col = first_col + side_gap
    w = detour_col + right_pad + 1

    grid = np.full((h, w), Color.BLACK, dtype=int)
    grid[start_row, 0] = Color.RED

    dir_right = (0, 1)
    dir_vertical = (vertical_sign, 0)
    dir_reverse_vertical = (-vertical_sign, 0)
    dir_detour = (0, 1)
    dir_home = (0, -1)

    a = (start_row, first_col)
    b = (far_row, first_col)
    c = (far_row, detour_col)
    d = (start_row, detour_col)

    mirrors: dict[tuple[int, int], int] = {
        a: _mirror_for_turn(dir_right, dir_vertical),
        b: _mirror_for_turn(dir_vertical, dir_detour),
        c: _mirror_for_turn(dir_detour, dir_reverse_vertical),
        d: _mirror_for_turn(dir_reverse_vertical, dir_home),
    }

    for (r, c0), color in mirrors.items():
        grid[r, c0] = color

    return grid, set(mirrors), {a}


def _build_chain_scene() -> tuple[np.ndarray, set[tuple[int, int]], set[tuple[int, int]]]:
    vertical_legs = random.randint(2, 3)
    sign = random.choice((-1, 1))

    levels = [0]
    directions: list[tuple[int, int]] = []
    for _ in range(vertical_legs):
        step = random.randint(2, 4)
        levels.append(levels[-1] + sign * step)
        directions.append((sign, 0))
        sign *= -1

    min_level = min(levels)
    max_level = max(levels)
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    row_shift = top_pad - min_level
    rows = [level + row_shift for level in levels]
    h = max_level + row_shift + bottom_pad + 1

    cols = []
    col = random.randint(2, 4)
    for _ in range(vertical_legs):
        cols.append(col)
        col += random.randint(3, 5)

    add_tail = random.random() < 0.5
    tail_col = None
    if add_tail:
        tail_col = col
        w = tail_col + random.randint(1, 3) + 1
    else:
        w = cols[-1] + random.randint(2, 4) + 1

    grid = np.full((h, w), Color.BLACK, dtype=int)
    grid[rows[0], 0] = Color.RED

    mirrors: dict[tuple[int, int], int] = {}
    dir_right = (0, 1)
    for i, col in enumerate(cols):
        start_row = rows[i]
        end_row = rows[i + 1]
        dir_vertical = directions[i]
        mirrors[(start_row, col)] = _mirror_for_turn(dir_right, dir_vertical)
        mirrors[(end_row, col)] = _mirror_for_turn(dir_vertical, dir_right)

    if tail_col is not None:
        tail_options = []
        last_row = rows[-1]
        if last_row >= 2:
            tail_options.append((-1, 0))
        if last_row <= h - 3:
            tail_options.append((1, 0))
        tail_dir = random.choice(tail_options)
        mirrors[(last_row, tail_col)] = _mirror_for_turn(dir_right, tail_dir)

    for (r, c), color in mirrors.items():
        grid[r, c] = color

    return grid, set(mirrors), set()


def _gen_attempt() -> np.ndarray | None:
    if random.random() < REVISIT_PROBABILITY:
        grid, mirrors, revisited = _build_revisit_scene()
        family = "revisit"
    else:
        grid, mirrors, revisited = _build_chain_scene()
        family = "chain"

    rotation = random.randrange(4)
    grid, mirrors, revisited = _rotate_scene(grid, mirrors, revisited, rotation)

    out, stats = _simulate(grid)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if stats["used_mirrors"] != mirrors:
        return None
    if stats["revisited_mirrors"] != revisited:
        return None
    if family == "revisit" and not revisited:
        return None
    if family == "chain" and revisited:
        return None
    if stats["mirror_colors_hit"] != {Color.GREEN, Color.PINK}:
        return None
    if stats["yellow_count"] < 5:
        return None
    if stats["yellow_count"] > int(grid.size * 0.45):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid one-shot mirror scene")
