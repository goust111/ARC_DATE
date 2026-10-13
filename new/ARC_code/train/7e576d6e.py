from __future__ import annotations

import numpy as np

from arc_utils import components_mask, get_bg
import random


def _color_counts(g: np.ndarray) -> dict[int, int]:
    vals, counts = np.unique(g, return_counts=True)
    return {int(v): int(n) for v, n in zip(vals, counts)}


def _roles(g: np.ndarray) -> tuple[int, int, int, int]:
    bg = get_bg(g)
    counts = _color_counts(g)
    colors = [c for c in counts if c != bg]
    assert len(colors) == 3, "expected wall, door, and marker colors"

    marker_colors = [c for c in colors if counts[c] == 2]
    assert len(marker_colors) == 1, "expected exactly two marker pixels"
    path_color = marker_colors[0]

    remaining = [c for c in colors if c != path_color]
    areas = {c: counts[c] for c in remaining}
    door_area = min(areas.values())
    wall_area = max(areas.values())
    door_colors = [c for c in remaining if counts[c] == door_area]
    wall_colors = [c for c in remaining if counts[c] == wall_area]
    assert len(door_colors) == 1 and len(wall_colors) == 1
    assert door_area < wall_area, "door color must be the smaller non-marker object set"
    return int(bg), int(path_color), int(door_colors[0]), int(wall_colors[0])


def _door_centers(g: np.ndarray, door_color: int) -> tuple[str, list[tuple[int, int]]]:
    comps = components_mask(g == int(door_color), connectivity=4)
    assert comps, "expected door segments"

    orientations: list[str] = []
    centers: list[tuple[int, int]] = []
    for coords in comps:
        assert coords.shape[0] == 3, "each door segment is three pixels long"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        if height == 3 and width == 1:
            orientations.append("vertical")
            centers.append((int(r0 + 1), int(c0)))
        elif height == 1 and width == 3:
            orientations.append("horizontal")
            centers.append((int(r0), int(c0 + 1)))
        else:
            raise AssertionError("door segments must be straight length-3 bars")

    assert len(set(orientations)) == 1, "all doors must share an orientation"
    return orientations[0], centers


def _paint_segment(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    color: int,
    allowed: set[int],
) -> None:
    r0, c0 = map(int, start)
    r1, c1 = map(int, end)
    assert r0 == r1 or c0 == c1, "path segments must be orthogonal"
    height, width = out.shape
    assert 0 <= r0 < height and 0 <= r1 < height
    assert 0 <= c0 < width and 0 <= c1 < width

    if r0 == r1:
        for c in range(min(c0, c1), max(c0, c1) + 1):
            assert int(out[r0, c]) in allowed, "path may only cover bg, doors, or markers"
            out[r0, c] = int(color)
    else:
        for r in range(min(r0, r1), max(r0, r1) + 1):
            assert int(out[r, c0]) in allowed, "path may only cover bg, doors, or markers"
            out[r, c0] = int(color)


def _markers(g: np.ndarray, path_color: int) -> list[tuple[int, int]]:
    coords = np.argwhere(g == int(path_color))
    assert coords.shape[0] == 2, "expected two marker pixels"
    return [tuple(map(int, coord)) for coord in coords]


def _solve_vertical_walls(
    g: np.ndarray,
    centers: list[tuple[int, int]],
    markers: list[tuple[int, int]],
    bg: int,
    path_color: int,
    door_color: int,
    wall_color: int,
) -> np.ndarray:
    out = g.copy()
    centers = sorted(centers, key=lambda p: p[1])
    wall_cols = sorted(set(map(int, np.argwhere(g == int(wall_color))[:, 1])))
    assert len(wall_cols) == len(centers), "each vertical wall must have one door"
    assert [c for _r, c in centers] == wall_cols
    for _r, c in centers:
        assert np.all(np.isin(g[:, c], [wall_color, door_color]))

    start, end = sorted(markers, key=lambda p: p[1])
    assert start[1] < centers[0][1] and end[1] > centers[-1][1]

    allowed = {int(bg), int(path_color), int(door_color)}
    cur = start
    for door_r, door_c in centers:
        approach = (cur[0], door_c - 1)
        _paint_segment(out, cur, approach, path_color, allowed)
        _paint_segment(out, approach, (door_r, door_c - 1), path_color, allowed)
        _paint_segment(out, (door_r, door_c - 1), (door_r, door_c), path_color, allowed)
        cur = (door_r, door_c)

    _paint_segment(out, cur, (cur[0], end[1]), path_color, allowed)
    _paint_segment(out, (cur[0], end[1]), end, path_color, allowed)
    return out


def _solve_horizontal_walls(
    g: np.ndarray,
    centers: list[tuple[int, int]],
    markers: list[tuple[int, int]],
    bg: int,
    path_color: int,
    door_color: int,
    wall_color: int,
) -> np.ndarray:
    out = g.copy()
    centers = sorted(centers, key=lambda p: p[0])
    wall_rows = sorted(set(map(int, np.argwhere(g == int(wall_color))[:, 0])))
    assert len(wall_rows) == len(centers), "each horizontal wall must have one door"
    assert [r for r, _c in centers] == wall_rows
    for r, _c in centers:
        assert np.all(np.isin(g[r, :], [wall_color, door_color]))

    start, end = sorted(markers, key=lambda p: p[0])
    assert start[0] < centers[0][0] and end[0] > centers[-1][0]

    allowed = {int(bg), int(path_color), int(door_color)}
    cur = start
    for door_r, door_c in centers:
        approach = (door_r - 1, cur[1])
        _paint_segment(out, cur, approach, path_color, allowed)
        _paint_segment(out, approach, (door_r - 1, door_c), path_color, allowed)
        _paint_segment(out, (door_r - 1, door_c), (door_r, door_c), path_color, allowed)
        cur = (door_r, door_c)

    _paint_segment(out, cur, (end[0], cur[1]), path_color, allowed)
    _paint_segment(out, (end[0], cur[1]), end, path_color, allowed)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw the marker-colored route through the centers of all doors in the parallel wall maze.

    Input structure:
    - The grid has a background, parallel full-span walls, one three-pixel door
      segment in each wall, and two marker pixels sharing a color.
    - Walls are either all vertical or all horizontal, and the markers are on
      opposite outer sides of the wall set.

    Rule:
    - Keep the scene unchanged except for the route.
    - Visit the door centers in wall order from the upper/left marker side to
      the lower/right marker side.
    - In each corridor, draw an orthogonal path to the cell just before the next
      wall, move along that wall to the door center, cross the center, and
      continue to the next corridor; after the last door, connect to the other
      marker.

    Key insight:
    - The three-pixel door segments mark the only crossing points through a set
      of parallel walls, so the output is the corridor path threaded through
      those centers.
    """
    g = np.asarray(grid, dtype=int)
    bg, path_color, door_color, wall_color = _roles(g)
    orientation, centers = _door_centers(g, door_color)
    markers = _markers(g, path_color)

    if orientation == "vertical":
        return _solve_vertical_walls(g, centers, markers, bg, path_color, door_color, wall_color)
    assert orientation == "horizontal"
    return _solve_horizontal_walls(g, centers, markers, bg, path_color, door_color, wall_color)

# --------------- generator (att1) ---------------

# Canvas and color role sampling.
GRID_H = 30
GRID_W = 30
COLOR_POOL = range(10)
N_ROLE_COLORS = 4

# Parallel wall and door geometry.
ORIENTATIONS = ("vertical", "horizontal")
WALL_COUNT_RANGE = (3, 4)
WALL_POS_LO = 4
WALL_POS_HI_MARGIN = 5
WALL_MIN_GAP = 2
DOOR_CENTER_LO = 3
DOOR_CENTER_HI_MARGIN = 4

# Marker placement and retry budget.
MARKER_EDGE_RANGE = (1, 3)
MARKER_FREE_LO = 1
MARKER_FREE_HI_MARGIN = 2
GEN_RETRIES = 300


def _place_positions(n: int, lo: int, hi: int, min_gap: int, rng: random.Random) -> list[int] | None:
    """Place n positions in [lo..hi] with pairwise gap >= min_gap between consecutive sorted positions."""
    needed = n + (n - 1) * min_gap
    span = hi - lo + 1
    if span < needed:
        return None
    compressed_span = span - (n - 1) * min_gap
    picks = sorted(rng.sample(range(compressed_span), n))
    return [lo + picks[i] + i * min_gap for i in range(n)]


def _sample_wall_layout(orientation: str, n_walls: int, rng: random.Random):
    """Return wall positions and per-wall door centers for the chosen orientation."""
    wall_axis, door_axis = (GRID_W, GRID_H) if orientation == "vertical" else (GRID_H, GRID_W)
    walls = _place_positions(
        n_walls,
        lo=WALL_POS_LO,
        hi=wall_axis - WALL_POS_HI_MARGIN,
        min_gap=WALL_MIN_GAP,
        rng=rng,
    )
    if walls is None:
        return None
    doors = [rng.randint(DOOR_CENTER_LO, door_axis - DOOR_CENTER_HI_MARGIN) for _ in range(n_walls)]
    return walls, doors


def _sample_markers(orientation: str, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample start/end markers on opposite outer corridors."""
    edge_lo, edge_hi = MARKER_EDGE_RANGE
    if orientation == "vertical":
        start_col = rng.randint(edge_lo, edge_hi)
        start_row = rng.randint(MARKER_FREE_LO, GRID_H - MARKER_FREE_HI_MARGIN)
        end_col = rng.randint(GRID_W - edge_hi - 1, GRID_W - edge_lo - 1)
        end_row = rng.randint(MARKER_FREE_LO, GRID_H - MARKER_FREE_HI_MARGIN)
    else:
        start_row = rng.randint(edge_lo, edge_hi)
        start_col = rng.randint(MARKER_FREE_LO, GRID_W - MARKER_FREE_HI_MARGIN)
        end_row = rng.randint(GRID_H - edge_hi - 1, GRID_H - edge_lo - 1)
        end_col = rng.randint(MARKER_FREE_LO, GRID_W - MARKER_FREE_HI_MARGIN)
    return (start_row, start_col), (end_row, end_col)


def _render_grid(
    orientation, bg, wall_color, door_color, marker_color, walls, doors, start, end
) -> np.ndarray:
    """Render walls, doors, and the two markers into a fresh input grid."""
    grid = np.full((GRID_H, GRID_W), bg, dtype=int)
    if orientation == "vertical":
        for wall_col, door_row in zip(walls, doors):
            grid[:, wall_col] = wall_color
            grid[door_row - 1 : door_row + 2, wall_col] = door_color
    else:
        for wall_row, door_col in zip(walls, doors):
            grid[wall_row, :] = wall_color
            grid[wall_row, door_col - 1 : door_col + 2] = door_color

    grid[start] = marker_color
    grid[end] = marker_color
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg, wall_color, door_color, marker_color = rng.sample(COLOR_POOL, N_ROLE_COLORS)
    orientation = rng.choice(ORIENTATIONS)
    n_walls = rng.randint(*WALL_COUNT_RANGE)

    layout = _sample_wall_layout(orientation, n_walls, rng)
    if layout is None:
        return None
    walls, doors = layout
    start, end = _sample_markers(orientation, rng)
    grid = _render_grid(orientation, bg, wall_color, door_color, marker_color, walls, doors, start, end)

    try:
        main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
