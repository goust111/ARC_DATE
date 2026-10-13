from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _find_source_and_period(grid: np.ndarray) -> tuple[tuple[int, int], int]:
    source_coords = np.argwhere(grid == YELLOW)
    assert len(source_coords) == 1, "expected exactly one yellow source pixel"
    sr, sc = map(int, source_coords[0])

    length = 0
    rr = sr + 1
    while rr < grid.shape[0] and int(grid[rr, sc]) == BLUE:
        length += 1
        rr += 1
    assert 2 <= length <= 4, "scale bar length must be 2..4"
    assert int(grid[sr + 1, sc]) == BLUE, "scale bar must start directly below the source"
    return (sr, sc), length


def _distance_map(grid: np.ndarray, source: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = source
    dist[sr, sc] = 0
    q = deque([(sr, sc)])

    def passable(r: int, c: int) -> bool:
        if (r, c) == (sr, sc):
            return True
        return int(grid[r, c]) == BLACK

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if dist[nr, nc] != -1 or not passable(nr, nc):
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist


def _solve_with_period(grid: np.ndarray, period: int) -> np.ndarray:
    source, _ = _find_source_and_period(grid)
    dist = _distance_map(grid, source)
    out = np.array(grid, dtype=int, copy=True)
    ripple_mask = (grid == BLACK) & (dist > 0) & (dist % int(period) == 0)
    out[ripple_mask] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw red ripple rings on background cells whose wall-avoiding geodesic distance from the source is a positive multiple of the blue scale length.

    Input structure:
    - A black background grid.
    - Exactly one yellow source pixel, with a vertical blue scale bar directly below it of length 2 to 4.
    - Grey wall pixels forming rooms, concavities, or corridors.

    Rule:
    - Let L be the scale-bar length.
    - Starting from the yellow source, compute 4-connected shortest-path distances through background cells only; grey walls and the blue scale bar are not traversable.
    - Recolor each reachable black cell red iff its distance is a positive multiple of L.
    - Leave the source, the scale bar, the walls, and unreachable background cells unchanged.

    Key insight:
    - The visible rings may look like ordinary source-centered Manhattan diamonds, but both the period and the layering are constrained: the bar sets the spacing, and wall-avoiding geodesic distance rephases the rings inside rooms and corridors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _source, period = _find_source_and_period(g)
    return _solve_with_period(g, period)


def _rect_is_clear(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(grid.shape[0] - 1, r1 + pad)
    cc1 = min(grid.shape[1] - 1, c1 + pad)
    return bool(np.all(grid[rr0 : rr1 + 1, cc0 : cc1 + 1] == BLACK))


def _draw_room(
    grid: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    *,
    door_side: str | None = None,
    door_cells: list[tuple[int, int]] | None = None,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]], set[tuple[int, int]]]:
    wall = set()
    for c in range(c0, c1 + 1):
        wall.add((r0, c))
        wall.add((r1, c))
    for r in range(r0, r1 + 1):
        wall.add((r, c0))
        wall.add((r, c1))

    if door_side is not None and door_cells is None:
        if door_side == "bottom":
            door_cells = [(r1, c0 + 1)]
        elif door_side == "left":
            door_cells = [(r0 + 1, c0)]
        elif door_side == "right":
            door_cells = [(r0 + 1, c1)]
        else:
            raise ValueError("unsupported door side")

    doors = set(door_cells or [])
    wall -= doors
    for r, c in wall:
        grid[r, c] = GREY

    interior = {(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)}
    return wall, interior, doors


def _draw_l_wall(
    grid: np.ndarray,
    top: int,
    left: int,
    v_len: int,
    h_len: int,
    horizontal_dir: int,
) -> set[tuple[int, int]]:
    coords = {(top + k, left) for k in range(v_len)}
    end_r = top + v_len - 1
    if horizontal_dir > 0:
        coords |= {(end_r, left + k) for k in range(h_len)}
    else:
        coords |= {(end_r, left - k) for k in range(h_len)}
    for r, c in coords:
        grid[r, c] = GREY
    return coords


def _manhattan_mask(shape: tuple[int, int], source: tuple[int, int], period: int) -> np.ndarray:
    sr, sc = source
    rows = np.arange(shape[0])[:, None]
    cols = np.arange(shape[1])[None, :]
    dist = np.abs(rows - sr) + np.abs(cols - sc)
    return (dist > 0) & (dist % int(period) == 0)


def _try_place_closed_room(grid: np.ndarray, rng: np.random.Generator) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    for _ in range(80):
        room_h = int(rng.integers(5, 8))
        room_w = int(rng.integers(5, 8))
        r0 = int(rng.integers(max(1, h // 2), h - room_h - 1))
        c0 = int(rng.integers(1, w - room_w - 1))
        r1 = r0 + room_h - 1
        c1 = c0 + room_w - 1
        if not _rect_is_clear(grid, r0, c0, r1, c1, pad=1):
            continue
        _wall, interior, _doors = _draw_room(grid, r0, c0, r1, c1)
        return interior
    return None


def _try_place_extra_wall(grid: np.ndarray, rng: np.random.Generator) -> bool:
    h, w = grid.shape
    for _ in range(80):
        v_len = int(rng.integers(4, max(5, h // 2)))
        h_len = int(rng.integers(4, max(5, w // 3)))
        top = int(rng.integers(1, h - v_len - 1))
        direction = int(rng.choice([-1, 1]))
        if direction > 0:
            left = int(rng.integers(1, w - h_len - 1))
            c_min = left
            c_max = left + h_len - 1
        else:
            left = int(rng.integers(h_len, w - 1))
            c_min = left - h_len + 1
            c_max = left
        r1 = top + v_len - 1
        if not _rect_is_clear(grid, top, c_min, r1, c_max, pad=1):
            continue
        _draw_l_wall(grid, top, left, v_len, h_len, direction)
        return True
    return False


def _sample_detour_layout(
    h: int,
    w: int,
    source: tuple[int, int],
    period: int,
    rng: np.random.Generator,
) -> tuple[int, int, int, int, list[tuple[int, int]]]:
    sr, sc = source
    room_h = int(rng.integers(7, min(11, h - sr - period - 2)))
    room_w = int(rng.integers(7, min(12, w - 2)))
    r0_low = sr + period + 3
    r0_high = h - room_h - 3
    r0 = int(rng.integers(r0_low, r0_high + 1))

    c0_low = max(1, sc - room_w + 2)
    c0_high = min(sc - 1, w - room_w - 2)
    if c0_low > c0_high:
        c0 = max(1, min(w - room_w - 2, sc - room_w // 2))
    else:
        c0 = int(rng.integers(c0_low, c0_high + 1))

    r1 = r0 + room_h - 1
    c1 = c0 + room_w - 1

    side = str(rng.choice(["bottom", "left", "right"]))
    if side == "bottom":
        options = [c for c in range(c0 + 1, c1) if abs(c - sc) >= 2]
        if not options:
            options = list(range(c0 + 1, c1))
        door_c = int(rng.choice(options))
        doors = [(r1, door_c)]
    elif side == "left":
        options = [r for r in range(r0 + 1, r1) if r >= r0 + room_h // 2]
        door_r = int(rng.choice(options))
        doors = [(door_r, c0)]
    else:
        options = [r for r in range(r0 + 1, r1) if r >= r0 + room_h // 2]
        door_r = int(rng.choice(options))
        doors = [(door_r, c1)]

    return r0, c0, r1, c1, doors


def _semantic_checks(
    grid: np.ndarray,
    detour_interior: set[tuple[int, int]],
    closed_interior: set[tuple[int, int]],
) -> bool:
    try:
        source, period = _find_source_and_period(grid)
        out = main(grid)
    except Exception:
        return False

    dist = _distance_map(grid, source)
    true_red = out == RED
    detour_mask = np.zeros_like(grid, dtype=bool)
    closed_mask = np.zeros_like(grid, dtype=bool)
    for r, c in detour_interior:
        detour_mask[r, c] = True
    for r, c in closed_interior:
        closed_mask[r, c] = True

    if int(true_red.sum()) < 12:
        return False
    if not np.any((grid == BLACK) & (dist == -1)):
        return False
    if not np.all(dist[closed_mask] == -1):
        return False
    if np.any(true_red & closed_mask):
        return False
    if not np.any((grid == BLACK) & (dist > 0) & ~true_red):
        return False

    alt_periods = {2, 3, 4} - {period}
    for alt in alt_periods:
        alt_red = _solve_with_period(grid, alt) == RED
        if np.array_equal(alt_red, true_red):
            return False

    naive_red = (grid == BLACK) & _manhattan_mask(grid.shape, source, period)
    if np.array_equal(naive_red, true_red):
        return False
    if not np.any((naive_red ^ true_red) & detour_mask):
        return False
    if not np.any(true_red & detour_mask):
        return False
    if not np.any(true_red & (dist >= 2 * period)):
        return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(19, 29))
    w = int(rng.integers(19, 29))
    grid = np.full((h, w), BLACK, dtype=int)

    period = int(rng.integers(2, 5))
    source_r = int(rng.integers(1, 4))
    source_c = int(rng.integers(5, w - 5))
    grid[source_r, source_c] = YELLOW
    for k in range(1, period + 1):
        grid[source_r + k, source_c] = BLUE

    try:
        r0, c0, r1, c1, doors = _sample_detour_layout(h, w, (source_r, source_c), period, rng)
    except ValueError:
        return None
    if not _rect_is_clear(grid, r0, c0, r1, c1, pad=1):
        return None
    _wall, detour_interior, _doors = _draw_room(grid, r0, c0, r1, c1, door_cells=doors)

    closed_interior = _try_place_closed_room(grid, rng)
    if closed_interior is None:
        return None

    extra_components = int(rng.integers(0, 3))
    for _ in range(extra_components):
        _try_place_extra_wall(grid, rng)

    if not _semantic_checks(grid, detour_interior, closed_interior):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(500):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ripple-wall input")
