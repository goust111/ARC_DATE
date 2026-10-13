from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

TIE = -1
UNREACHED = -2
WALL = -3

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_HEIGHT_RANGE = (12, 20)
GEN_WIDTH_RANGE = (12, 20)
GEN_ATTEMPTS = 400
MIN_SEED_GAP = 4
MIN_TERRITORY = 8
MIN_TIE_CELLS = 3
MIN_WALL_EFFECT = 8
MIN_CONNECTIVITY_EFFECT = 3


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _parse_scene(grid: np.ndarray) -> tuple[np.ndarray, list[int], list[tuple[int, int]], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"
    assert max(g.shape) <= 30, "ARC grids must stay within 30x30"

    colors = sorted(int(v) for v in np.unique(g))
    assert set(colors).issubset(set(range(10))), "expected ARC palette ids 0..9"
    seed_colors = [color for color in colors if color not in (BLACK, GREY)]
    assert 2 <= len(seed_colors) <= 3, "expected two or three distinct seed pixels"

    seed_positions: list[tuple[int, int]] = []
    for color in seed_colors:
        coords = np.argwhere(g == color)
        assert len(coords) == 1, "each seed color must appear exactly once"
        seed_positions.append(tuple(map(int, coords[0])))

    wall_mask = g == GREY
    assert np.any(wall_mask), "expected at least one grey wall cell"
    return g, seed_colors, seed_positions, wall_mask


def _distance_fields(
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    seed_positions: list[tuple[int, int]],
    *,
    connectivity: int,
) -> np.ndarray:
    dirs = DIRS8 if connectivity == 8 else DIRS4
    height, width = map(int, shape)
    inf = height * width + 100
    dist = np.full((len(seed_positions), height, width), inf, dtype=int)

    for idx, (r0, c0) in enumerate(seed_positions):
        queue: deque[tuple[int, int]] = deque([(int(r0), int(c0))])
        dist[idx, r0, c0] = 0
        while queue:
            r, c = queue.popleft()
            nxt = int(dist[idx, r, c]) + 1
            for dr, dc in dirs:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < height and 0 <= cc < width and not wall_mask[rr, cc] and nxt < dist[idx, rr, cc]:
                    dist[idx, rr, cc] = nxt
                    queue.append((rr, cc))
    return dist


def _winner_grid(
    shape: tuple[int, int],
    wall_mask: np.ndarray,
    seed_positions: list[tuple[int, int]],
    *,
    connectivity: int,
) -> np.ndarray:
    dist = _distance_fields(shape, wall_mask, seed_positions, connectivity=connectivity)
    inf = int(dist.max())
    winners = np.full(shape, TIE, dtype=int)
    winners[wall_mask] = WALL

    for r in range(shape[0]):
        for c in range(shape[1]):
            if wall_mask[r, c]:
                continue
            values = dist[:, r, c]
            best = int(values.min())
            if best >= inf:
                winners[r, c] = UNREACHED
                continue
            hits = np.flatnonzero(values == best)
            winners[r, c] = int(hits[0]) if len(hits) == 1 else TIE
    return winners


def _nowall_chebyshev_winners(shape: tuple[int, int], seed_positions: list[tuple[int, int]]) -> np.ndarray:
    winners = np.full(shape, TIE, dtype=int)
    for r in range(shape[0]):
        for c in range(shape[1]):
            values = [_chebyshev((r, c), seed) for seed in seed_positions]
            best = min(values)
            hits = [idx for idx, value in enumerate(values) if value == best]
            winners[r, c] = hits[0] if len(hits) == 1 else TIE
    return winners


def _render_output(
    grid: np.ndarray,
    seed_colors: list[int],
    seed_positions: list[tuple[int, int]],
    wall_mask: np.ndarray,
    winners: np.ndarray,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    background = out == BLACK
    for idx, color in enumerate(seed_colors):
        out[background & (winners == idx)] = int(color)
    out[wall_mask] = GREY
    for color, (r, c) in zip(seed_colors, seed_positions):
        out[r, c] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Let colored seed pixels claim the background by 8-neighbor spread while grey walls block the routes.

    Input structure:
    - A black background contains two or three singleton seed pixels, each in a different non-grey color.
    - Grey cells form one-pixel-thick wall segments that stay fixed and cannot be crossed.

    Rule:
    - All seeds expand simultaneously through the background using 8-neighbor steps.
    - For each background cell, compare the seeds' shortest path lengths measured in that wall-blocked 8-neighbor graph.
    - If exactly one seed arrives first, paint that cell with that seed's color; if multiple seeds arrive in the same round, leave it black.
    - Preserve every original seed pixel and every grey wall cell.

    Key insight:
    - The partition is a geodesic Chebyshev Voronoi diagram on the walkable cells, so walls bend the front, reroute ownership, and expose black tie seams where arrival times match.
    """
    g, seed_colors, seed_positions, wall_mask = _parse_scene(grid)
    winners = _winner_grid(g.shape, wall_mask, seed_positions, connectivity=8)
    return _render_output(g, seed_colors, seed_positions, wall_mask, winners)


def _off_center_choices(length: int) -> list[int]:
    return [idx for idx in range(2, length - 2) if idx <= length // 3 or idx >= length - 1 - length // 3]


def _interior_open_cells(
    wall_mask: np.ndarray, row_range: range, col_range: range
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in row_range:
        for c in col_range:
            if not wall_mask[r, c]:
                cells.append((r, c))
    return cells


def _make_wall_template(height: int, width: int) -> tuple[np.ndarray, dict[str, list[tuple[int, int]]]]:
    wall_mask = np.zeros((height, width), dtype=bool)
    vertical = random.random() < 0.5
    layout = random.choice(("single", "double"))

    if vertical:
        if layout == "single":
            wall_col = random.randint(4, width - 5)
            gate_row = random.choice(_off_center_choices(height))
            for r in range(1, height - 1):
                if r != gate_row:
                    wall_mask[r, wall_col] = True
            zones = {
                "a": _interior_open_cells(wall_mask, range(1, height - 1), range(1, wall_col - 1)),
                "b": _interior_open_cells(wall_mask, range(1, height - 1), range(wall_col + 2, width - 1)),
            }
            return wall_mask, zones

        left_col = random.randint(3, width // 2 - 2)
        right_col = random.randint(left_col + 3, min(width - 4, left_col + 5))
        upper_gate = random.choice(_off_center_choices(height)[: max(1, len(_off_center_choices(height)) // 2)])
        lower_gate = random.choice(_off_center_choices(height)[max(1, len(_off_center_choices(height)) // 2) :])
        for r in range(1, height - 1):
            if r != upper_gate:
                wall_mask[r, left_col] = True
            if r != lower_gate:
                wall_mask[r, right_col] = True
        zones = {
            "a": _interior_open_cells(wall_mask, range(1, height - 1), range(1, left_col - 1)),
            "b": _interior_open_cells(wall_mask, range(1, height - 1), range(right_col + 2, width - 1)),
            "middle": _interior_open_cells(wall_mask, range(1, height - 1), range(left_col + 1, right_col)),
        }
        return wall_mask, zones

    if layout == "single":
        wall_row = random.randint(4, height - 5)
        gate_col = random.choice(_off_center_choices(width))
        for c in range(1, width - 1):
            if c != gate_col:
                wall_mask[wall_row, c] = True
        zones = {
            "a": _interior_open_cells(wall_mask, range(1, wall_row - 1), range(1, width - 1)),
            "b": _interior_open_cells(wall_mask, range(wall_row + 2, height - 1), range(1, width - 1)),
        }
        return wall_mask, zones

    top_row = random.randint(3, height // 2 - 2)
    bottom_row = random.randint(top_row + 3, min(height - 4, top_row + 5))
    left_gate = random.choice(_off_center_choices(width)[: max(1, len(_off_center_choices(width)) // 2)])
    right_gate = random.choice(_off_center_choices(width)[max(1, len(_off_center_choices(width)) // 2) :])
    for c in range(1, width - 1):
        if c != left_gate:
            wall_mask[top_row, c] = True
        if c != right_gate:
            wall_mask[bottom_row, c] = True
    zones = {
        "a": _interior_open_cells(wall_mask, range(1, top_row - 1), range(1, width - 1)),
        "b": _interior_open_cells(wall_mask, range(bottom_row + 2, height - 1), range(1, width - 1)),
        "middle": _interior_open_cells(wall_mask, range(top_row + 1, bottom_row), range(1, width - 1)),
    }
    return wall_mask, zones


def _pick_seed_positions(zones: dict[str, list[tuple[int, int]]]) -> list[tuple[int, int]] | None:
    usable = [name for name, cells in zones.items() if len(cells) >= 8]
    if len(usable) < 2:
        return None

    use_three = ("middle" in usable and random.random() < 0.45) or random.random() < 0.35
    chosen = ["a", "b"]
    if use_three:
        chosen.append(random.choice(usable))
    random.shuffle(chosen)

    seeds: list[tuple[int, int]] = []
    for zone_name in chosen:
        options = zones[zone_name][:]
        random.shuffle(options)
        picked = None
        for cell in options:
            if all(_chebyshev(cell, seed) >= MIN_SEED_GAP for seed in seeds):
                picked = cell
                break
        if picked is None:
            return None
        seeds.append(picked)
    return seeds


def _build_input(height: int, width: int, wall_mask: np.ndarray, seed_positions: list[tuple[int, int]]) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    grid[wall_mask] = GREY
    colors = random.sample(SEED_COLORS, len(seed_positions))
    for color, (r, c) in zip(colors, seed_positions):
        grid[r, c] = int(color)
    return grid


def _semantic_ok(grid: np.ndarray) -> bool:
    try:
        g, seed_colors, seed_positions, wall_mask = _parse_scene(grid)
        actual = main(g)
    except Exception:
        return False

    winners8 = _winner_grid(g.shape, wall_mask, seed_positions, connectivity=8)
    if np.any(winners8 == UNREACHED):
        return False

    expected = _render_output(g, seed_colors, seed_positions, wall_mask, winners8)
    if not np.array_equal(actual, expected):
        return False

    background = g == BLACK
    tie_cells = int(np.count_nonzero(background & (winners8 == TIE)))
    if tie_cells < MIN_TIE_CELLS:
        return False

    territory_sizes = [int(np.count_nonzero(winners8 == idx)) - 1 for idx in range(len(seed_positions))]
    if min(territory_sizes) < MIN_TERRITORY:
        return False

    no_wall = _nowall_chebyshev_winners(g.shape, seed_positions)
    wall_effect = int(np.count_nonzero(background & (winners8 != no_wall)))
    if wall_effect < MIN_WALL_EFFECT:
        return False

    winners4 = _winner_grid(g.shape, wall_mask, seed_positions, connectivity=4)
    connectivity_effect = int(np.count_nonzero(background & (winners8 != winners4)))
    if connectivity_effect < MIN_CONNECTIVITY_EFFECT:
        return False

    if np.array_equal(actual, g):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    width = random.randint(*GEN_WIDTH_RANGE)
    wall_mask, zones = _make_wall_template(height, width)
    seed_positions = _pick_seed_positions(zones)
    if seed_positions is None:
        return None
    grid = _build_input(height, width, wall_mask, seed_positions)
    return grid if _semantic_ok(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wall-blocked geodesic Chebyshev territory input")
