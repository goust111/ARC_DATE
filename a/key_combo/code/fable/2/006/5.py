from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(ARC_UTILS_DIR))

from arc_utils import Color, get_bg


CORNER_COORDS = {
    "tl": lambda n: (0, 0),
    "tr": lambda n: (0, n - 1),
    "br": lambda n: (n - 1, n - 1),
    "bl": lambda n: (n - 1, 0),
}

ROW_MAJOR_KEYS = {
    "tl": lambda rc: (rc[0], rc[1]),
    "tr": lambda rc: (rc[0], -rc[1]),
    "br": lambda rc: (-rc[0], -rc[1]),
    "bl": lambda rc: (-rc[0], rc[1]),
}

COL_MAJOR_KEYS = {
    "tl": lambda rc: (rc[1], rc[0]),
    "tr": lambda rc: (-rc[1], rc[0]),
    "br": lambda rc: (-rc[1], -rc[0]),
    "bl": lambda rc: (rc[1], -rc[0]),
}

GEN_SIZE_RANGE = (7, 15)
GEN_MARKER_COUNT_RANGE = (5, 11)
GEN_MAX_ATTEMPTS = 500


def _ring_coords_tl_clockwise(n: int, layer: int) -> list[tuple[int, int]]:
    side = n - 2 * layer
    if side == 1:
        return [(layer, layer)]

    r0 = c0 = layer
    r1 = c1 = n - 1 - layer
    ring: list[tuple[int, int]] = []
    ring.extend((r0, c) for c in range(c0, c1 + 1))
    ring.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    ring.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    ring.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return ring


def _ring_start_coord(n: int, layer: int, start_corner: str) -> tuple[int, int]:
    if start_corner == "tl":
        return layer, layer
    if start_corner == "tr":
        return layer, n - 1 - layer
    if start_corner == "br":
        return n - 1 - layer, n - 1 - layer
    if start_corner == "bl":
        return n - 1 - layer, layer
    raise AssertionError(f"unknown corner: {start_corner}")


def _spiral_coords(n: int, start_corner: str, *, clockwise: bool = True) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for layer in range((n + 1) // 2):
        ring = _ring_coords_tl_clockwise(n, layer)
        start = _ring_start_coord(n, layer, start_corner)
        start_idx = ring.index(start)
        ring = ring[start_idx:] + ring[:start_idx]
        if not clockwise and len(ring) > 1:
            ring = [ring[0]] + list(reversed(ring[1:]))
        coords.extend(ring)
    return coords


def _detect_start_corner(g: np.ndarray, bg: int) -> tuple[str, int]:
    n = g.shape[0]
    colored = []
    for name, coord_fn in CORNER_COORDS.items():
        r, c = coord_fn(n)
        color = int(g[r, c])
        if color != bg:
            colored.append((name, color))

    assert len(colored) == 1, "expected exactly one colored corner"
    start_corner, start_color = colored[0]
    assert int(np.count_nonzero(g == start_color)) == 1, "corner color must be unique in the grid"
    return start_corner, start_color


def _layer_of(n: int, coord: tuple[int, int]) -> int:
    r, c = coord
    return min(r, c, n - 1 - r, n - 1 - c)


def _tl_ring_offset(n: int, layer: int, coord: tuple[int, int]) -> int:
    r, c = coord
    side = n - 2 * layer
    r0 = c0 = layer
    r1 = c1 = n - 1 - layer
    assert r in (r0, r1) or c in (c0, c1)
    if side == 1:
        return 0
    if r == r0:
        return c - c0
    if c == c1:
        return (side - 1) + (r - r0)
    if r == r1:
        return 2 * (side - 1) + (c1 - c)
    return 3 * (side - 1) + (r1 - r)


def _spiral_index(n: int, start_corner: str, coord: tuple[int, int], *, clockwise: bool = True) -> int:
    layer = _layer_of(n, coord)
    side = n - 2 * layer
    base = n * n - side * side
    if side == 1:
        return base

    tl_offset = _tl_ring_offset(n, layer, coord)
    start_offset = {
        "tl": 0,
        "tr": side - 1,
        "br": 2 * (side - 1),
        "bl": 3 * (side - 1),
    }[start_corner]
    perimeter = 4 * side - 4
    if clockwise:
        return base + ((tl_offset - start_offset) % perimeter)
    return base + ((start_offset - tl_offset) % perimeter)


def _color_sequence_from_order(
    positions: list[tuple[int, int]],
    color_by_pos: dict[tuple[int, int], int],
) -> tuple[int, ...]:
    return tuple(int(color_by_pos[pos]) for pos in positions)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the scattered colored pixels in the hidden corner-anchored clockwise spiral order and flatten them into one row.

    Input structure:
    - The grid is square with a black background and scattered non-black pixels.
    - Exactly one of the four corners is colored, and that corner color appears nowhere else.
    - All other colored cells are ordinary markers that may repeat colors.

    Rule:
    - Use the unique colored corner as the spiral start.
    - Traverse the whole square by clockwise outer ring, then next inner ring, continuing inward until the center.
    - Whenever the traversal visits a colored cell, append its color to the output sequence.
    - Return a 1xN strip whose left-to-right colors match that encounter order.

    Key insight:
    - The ordering cue is an implicit spiral path over the full canvas; both ring depth and clockwise position within each ring matter.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h == w, "expected a square grid"

    bg = int(get_bg(g))
    start_corner, _start_color = _detect_start_corner(g, bg)

    colors: list[int] = []
    for r, c in _spiral_coords(h, start_corner, clockwise=True):
        value = int(g[r, c])
        if value != bg:
            colors.append(value)

    assert colors, "expected at least one colored marker"
    return np.asarray([colors], dtype=int)


def _is_isolated(coord: tuple[int, int], chosen: set[tuple[int, int]]) -> bool:
    r, c = coord
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen)


def _choose_coord(
    candidates: list[tuple[int, int]],
    chosen: set[tuple[int, int]],
) -> tuple[int, int] | None:
    random.shuffle(candidates)
    for coord in candidates:
        if coord not in chosen and _is_isolated(coord, chosen):
            return coord
    return None


def _marker_positions(n: int, start_corner: str, total_markers: int) -> list[tuple[int, int]] | None:
    start = CORNER_COORDS[start_corner](n)
    chosen = {start}
    positions = [start]

    outer_candidates = [
        coord
        for coord in _ring_coords_tl_clockwise(n, 0)
        if coord != start and coord not in {CORNER_COORDS[name](n) for name in CORNER_COORDS}
    ]
    outer_pick = _choose_coord(outer_candidates, chosen)
    if outer_pick is None:
        return None
    chosen.add(outer_pick)
    positions.append(outer_pick)

    inner_layers = list(range(1, (n + 1) // 2))
    random.shuffle(inner_layers)
    placed_inner = False
    for layer in inner_layers:
        layer_cells = _ring_coords_tl_clockwise(n, layer)
        first = _choose_coord(layer_cells[:], chosen)
        if first is None:
            continue
        temp_chosen = chosen | {first}
        second = _choose_coord(layer_cells[:], temp_chosen)
        if second is None:
            continue
        chosen.update({first, second})
        positions.extend([first, second])
        placed_inner = True
        break
    if not placed_inner:
        return None

    center = (n // 2, n // 2)
    if n % 2 == 1 and len(positions) < total_markers and random.random() < 0.4 and _is_isolated(center, chosen):
        chosen.add(center)
        positions.append(center)

    candidates = [
        (r, c)
        for r in range(n)
        for c in range(n)
        if (r, c) not in {CORNER_COORDS[name](n) for name in CORNER_COORDS}
    ]
    while len(positions) < total_markers:
        pick = _choose_coord(candidates[:], chosen)
        if pick is None:
            return None
        chosen.add(pick)
        positions.append(pick)

    return positions


def _assign_colors(
    n: int,
    positions: list[tuple[int, int]],
    start_corner: str,
) -> dict[tuple[int, int], int]:
    ordered = sorted(positions, key=lambda coord: _spiral_index(n, start_corner, coord))
    start = CORNER_COORDS[start_corner](n)
    assert ordered[0] == start

    palette = list(range(1, 10))
    start_color = random.choice(palette)
    remaining_palette = [color for color in palette if color != start_color]

    color_by_pos = {start: start_color}
    while True:
        pool_size = random.randint(2, min(5, len(remaining_palette)))
        pool = random.sample(remaining_palette, pool_size)
        others = [random.choice(pool) for _ in range(len(ordered) - 1)]
        if len(set(others)) < 2:
            continue
        if max(others.count(color) for color in pool) < 2:
            continue
        for pos, color in zip(ordered[1:], others):
            color_by_pos[pos] = int(color)
        return color_by_pos


def _ordered_output(
    n: int,
    start_corner: str,
    color_by_pos: dict[tuple[int, int], int],
    *,
    clockwise: bool = True,
) -> tuple[int, ...]:
    positions = sorted(
        color_by_pos,
        key=lambda coord: _spiral_index(n, start_corner, coord, clockwise=clockwise),
    )
    return _color_sequence_from_order(positions, color_by_pos)


def _semantic_checks(
    n: int,
    start_corner: str,
    positions: list[tuple[int, int]],
    color_by_pos: dict[tuple[int, int], int],
) -> bool:
    layers = [_layer_of(n, pos) for pos in positions]
    if len(set(layers)) < 2:
        return False
    if max(layers.count(layer) for layer in set(layers)) < 2:
        return False

    spiral_seq = _ordered_output(n, start_corner, color_by_pos, clockwise=True)
    ccw_seq = _ordered_output(n, start_corner, color_by_pos, clockwise=False)
    if spiral_seq == ccw_seq:
        return False

    row_major_seq = _color_sequence_from_order(
        sorted(positions, key=ROW_MAJOR_KEYS[start_corner]),
        color_by_pos,
    )
    col_major_seq = _color_sequence_from_order(
        sorted(positions, key=COL_MAJOR_KEYS[start_corner]),
        color_by_pos,
    )
    if spiral_seq in {row_major_seq, col_major_seq}:
        return False

    return True


def _build_grid(
    n: int,
    positions: list[tuple[int, int]],
    color_by_pos: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.full((n, n), Color.BLACK, dtype=int)
    for pos in positions:
        grid[pos] = int(color_by_pos[pos])
    return grid


def _expected_output_from_formula(grid: np.ndarray) -> np.ndarray:
    bg = int(get_bg(grid))
    n = grid.shape[0]
    start_corner, _ = _detect_start_corner(grid, bg)
    coords = [tuple(map(int, coord)) for coord in np.argwhere(grid != bg)]
    coords.sort(key=lambda coord: _spiral_index(n, start_corner, coord, clockwise=True))
    colors = [int(grid[coord]) for coord in coords]
    return np.asarray([colors], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*GEN_SIZE_RANGE)
    start_corner = random.choice(list(CORNER_COORDS))
    total_markers = random.randint(GEN_MARKER_COUNT_RANGE[0], min(GEN_MARKER_COUNT_RANGE[1], n + 2))

    positions = _marker_positions(n, start_corner, total_markers)
    if positions is None:
        return None

    color_by_pos = _assign_colors(n, positions, start_corner)
    if not _semantic_checks(n, start_corner, positions, color_by_pos):
        return None

    grid = _build_grid(n, positions, color_by_pos)
    expected = _expected_output_from_formula(grid)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid invisible-spiral input")
