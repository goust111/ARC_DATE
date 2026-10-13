from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, scale_sprite


WORKING_BORDER_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
MAX_GENERATION_ATTEMPTS = 400


def _center_offset(scale: int) -> int:
    scale = int(scale)
    assert scale >= 1
    return (scale - 1) // 2


def _block_center(r: int, c: int, scale: int) -> tuple[int, int]:
    off = _center_offset(scale)
    return int(r) * scale + off, int(c) * scale + off


def _border_scale_factor(grid: np.ndarray) -> int:
    border = set(map(int, grid[-1, :])) | set(map(int, grid[:, -1]))
    colors = border - {Color.BLACK, Color.GREY}
    assert colors <= set(WORKING_BORDER_COLORS), "border band may only use blue/red/green/yellow"
    assert len(colors) in (2, 3), "border band must encode scale 2 or 3"
    return len(colors)


def _wall_specs(grid: np.ndarray) -> list[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    walls: list[tuple[int, int]] = []
    for c in range(w - 1):
        col = g[:, c]
        black_rows = np.flatnonzero(col == Color.BLACK)
        grey_rows = np.flatnonzero(col == Color.GREY)
        other_rows = np.flatnonzero((col != Color.BLACK) & (col != Color.GREY))
        if black_rows.size == 1 and grey_rows.size == h - 1 and other_rows.size == 0:
            walls.append((int(c), int(black_rows[0])))

    assert len(walls) in (2, 3), "expected 2 or 3 vertical walls"
    wall_cols = [c for c, _r in walls]
    gap_rows = [r for _c, r in walls]
    assert wall_cols == sorted(wall_cols), "wall columns must be ordered left-to-right"
    assert len(set(gap_rows)) == len(gap_rows), "door gaps must be on distinct rows"
    return walls


def _marker_pixels(grid: np.ndarray, walls: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    coords = np.argwhere(g == Color.PINK)
    assert coords.shape == (2, 2), "expected exactly two magenta marker pixels"

    wall_cols = {c for c, _r in walls}
    eligible_cols = [c for c in range(g.shape[1] - 1) if c not in wall_cols]
    assert eligible_cols, "expected corridor columns left of the right border band"

    markers = sorted((tuple(map(int, p)) for p in coords), key=lambda p: p[1])
    left, right = markers
    assert left[1] == eligible_cols[0], "left marker must sit in the leftmost non-wall corridor column"
    assert right[1] == eligible_cols[-1], "right marker must sit in the rightmost non-wall corridor column"
    assert left[0] != right[0], "marker rows must differ"
    return left, right


def _paint_axis_segment(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    *,
    allowed: set[int],
    color: int,
) -> None:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    assert r0 == r1 or c0 == c1, "segments must be orthogonal"
    if r0 == r1:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            assert int(out[r0, c]) in allowed, "path may only cover background, doors, or existing path"
            out[r0, c] = int(color)
    else:
        for r in range(min(r0, r1), max(r0, r1) + 1):
            assert int(out[r, c0]) in allowed, "path may only cover background, doors, or existing path"
            out[r, c0] = int(color)


def _solve_scaled_route(grid: np.ndarray, scale: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    walls = _wall_specs(g)
    left_marker, right_marker = _marker_pixels(g, walls)

    out = scale_sprite(g, scale)
    allowed = {Color.BLACK, Color.PINK}
    current = _block_center(*left_marker, scale)

    for wall_index, (wall_c, gap_r) in enumerate(walls):
        door_center = _block_center(gap_r, wall_c, scale)
        wall_left = wall_c * scale
        wall_right = wall_left + scale - 1
        entry = (current[0], wall_left - 1)
        turn = (door_center[0], wall_left - 1)
        exit_cell = (door_center[0], wall_right + 1)

        _paint_axis_segment(out, current, entry, allowed=allowed, color=Color.PINK)
        _paint_axis_segment(out, entry, turn, allowed=allowed, color=Color.PINK)
        _paint_axis_segment(out, turn, door_center, allowed=allowed, color=Color.PINK)
        _paint_axis_segment(out, door_center, exit_cell, allowed=allowed, color=Color.PINK)

        current = exit_cell
        if wall_index < len(walls) - 1:
            next_wall_left = walls[wall_index + 1][0] * scale
            assert current[1] < next_wall_left - 1, "corridor width must stay positive after scaling"

    target = _block_center(*right_marker, scale)
    _paint_axis_segment(out, current, (current[0], target[1]), allowed=allowed, color=Color.PINK)
    _paint_axis_segment(out, (current[0], target[1]), target, allowed=allowed, color=Color.PINK)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Scale the wall maze by the border color count, then thread a magenta route through every enlarged door center.

    Input structure:
    - The background is black.
    - Two or three vertical grey full-column walls each contain exactly one black one-pixel gap.
    - Two magenta marker pixels sit in the leftmost and rightmost corridor columns.
    - The bottom row and rightmost column contain blue/red/green/yellow border pixels whose distinct-color count is 2 or 3.

    Rule:
    - Count the distinct non-black, non-grey colors appearing on the bottom row or rightmost column; call that count N.
    - Magnify the whole grid by factor N so every input pixel becomes an N x N block.
    - Treat each enlarged black gap inside a grey wall as an N x N door block, and use its block-center pixel as the mandatory crossing point.
    - Starting from the left marker block center, visit the doors from left to right: in each corridor move horizontally to the cell just before the next wall block, move vertically to the door-center row, then cross the door block through its center; after the last door, connect to the right marker block center.
    - Paint every route cell magenta without overwriting grey wall cells.

    Key insight:
    - The original one-pixel gaps have no meaningful geometric center, but after scale-dependent pixel expansion each gap becomes a door block with a usable center waypoint, so the scaling step creates the navigation structure.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    scale = _border_scale_factor(g)
    return _solve_scaled_route(g, scale)


def _sample_spaced_positions(pool: list[int], count: int, min_gap: int) -> list[int] | None:
    if count == 0:
        return []
    if len(pool) < count:
        return None
    for _ in range(80):
        picks = sorted(random.sample(pool, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _paint_border_band(grid: np.ndarray, colors: tuple[int, ...], wall_cols: set[int]) -> None:
    h, w = grid.shape
    usable_bottom = [c for c in range(w - 1) if c not in wall_cols]
    usable_right = list(range(h - 1))
    assert usable_bottom and usable_right

    bottom_count = random.randint(2, min(len(usable_bottom), max(2, len(colors) + 1)))
    right_count = random.randint(2, min(len(usable_right), max(2, len(colors) + 1)))
    bottom_slots = sorted(random.sample(usable_bottom, bottom_count))
    right_slots = sorted(random.sample(usable_right, right_count))

    for idx, c in enumerate(bottom_slots):
        grid[h - 1, c] = int(colors[idx % len(colors)])
    for idx, r in enumerate(right_slots):
        grid[r, w - 1] = int(colors[(idx + 1) % len(colors)])

    grid[h - 1, w - 1] = int(colors[-1])
    assert _border_scale_factor(grid) == len(colors)


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> None:
    scale = _border_scale_factor(inp)
    scaled_inp = scale_sprite(inp, scale)
    walls = _wall_specs(inp)
    left_marker, right_marker = _marker_pixels(inp, walls)

    assert out.shape == (inp.shape[0] * scale, inp.shape[1] * scale)
    assert np.all(out[scaled_inp == Color.GREY] == Color.GREY), "route may not overwrite grey wall cells"
    assert np.count_nonzero(out == Color.PINK) > 2 * scale * scale, "route must add pixels beyond the markers"

    for wall_c, gap_r in walls:
        center = _block_center(gap_r, wall_c, scale)
        assert int(out[center]) == Color.PINK, "every door center must be crossed"

    assert int(out[_block_center(*left_marker, scale)]) == Color.PINK
    assert int(out[_block_center(*right_marker, scale)]) == Color.PINK


def _gen_attempt() -> np.ndarray | None:
    scale = random.choice((2, 3))
    n_walls = random.choice((2, 3))

    h = random.randint(max(6, n_walls + 3), 10)
    min_w = 7 if n_walls == 2 else 9
    w = random.randint(min_w, 10)
    if max(h * scale, w * scale) > 30:
        return None

    wall_cols = _sample_spaced_positions(list(range(2, w - 2)), n_walls, min_gap=2)
    if wall_cols is None:
        return None
    if wall_cols[-1] >= w - 2:
        return None

    gap_rows = random.sample(list(range(h - 1)), n_walls)
    left_row, right_row = random.sample(list(range(h - 1)), 2)

    grid = np.zeros((h, w), dtype=int)
    for c, gap_r in zip(wall_cols, gap_rows):
        grid[:, c] = Color.GREY
        grid[gap_r, c] = Color.BLACK

    grid[left_row, 0] = Color.PINK
    grid[right_row, w - 2] = Color.PINK

    border_colors = tuple(random.sample(WORKING_BORDER_COLORS, scale))
    _paint_border_band(grid, border_colors, set(wall_cols))

    try:
        walls = _wall_specs(grid)
        markers = _marker_pixels(grid, walls)
        if markers[0][0] == gap_rows[0] and markers[1][0] == gap_rows[-1] and len(set(gap_rows)) == len(gap_rows):
            # Prefer non-aligned endpoints so the vertical corridor step stays visible.
            return None
        out = main(grid)
        _semantic_checks(grid, out)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enlarged-door-route input")
