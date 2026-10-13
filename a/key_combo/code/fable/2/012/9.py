from __future__ import annotations

from collections import deque
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    GREY = 5
    GRAY = 5


Coord = tuple[int, int]


MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
OBJECT_SHAPES = (
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
)

GEN_HEIGHT_RANGE = (11, 20)
GEN_LANE_COUNT_RANGE = (3, 5)
GEN_LANE_WIDTH_RANGE = (3, 5)
GEN_MARKER_COUNT_RANGE = (3, 5)
GEN_MAX_ATTEMPTS = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= r < h and 0 <= c < w


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[Coord] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _distance_map(grid: np.ndarray, starts: list[Coord]) -> np.ndarray:
    dist = np.full(grid.shape, -1, dtype=int)
    q = deque()
    for r, c in starts:
        dist[r, c] = 0
        q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if dist[nr, nc] != -1 or int(grid[nr, nc]) == Color.BLUE:
                continue
            dist[nr, nc] = dist[r, c] + 1
            q.append((nr, nc))
    return dist


def _extract_structure(grid: np.ndarray) -> tuple[np.ndarray, list[Coord], list[tuple[int, Coord]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey_comps = _components_mask(g == Color.GREY)
    assert len(grey_comps) == 1, "expected exactly one gray 4-connected object"
    object_coords = [tuple(map(int, rc)) for rc in grey_comps[0]]

    marker_info: list[tuple[int, Coord]] = []
    for color in sorted(set(map(int, np.unique(g))) - {Color.BLACK, Color.BLUE, Color.GREY}):
        comps = _components_mask(g == color)
        assert len(comps) == 1, "each marker color must appear in one component"
        assert comps[0].shape[0] == 1, "each marker must be a singleton pixel"
        marker_info.append((int(color), tuple(map(int, comps[0][0]))))
    assert len(marker_info) >= 2, "expected at least two marker pixels"
    return g, object_coords, marker_info


def _winner_from_distances(
    grid: np.ndarray, object_coords: list[Coord], marker_info: list[tuple[int, Coord]]
) -> tuple[int, dict[int, int]]:
    dist = _distance_map(grid, object_coords)
    marker_dists: dict[int, int] = {}
    for color, (r, c) in marker_info:
        steps = int(dist[r, c])
        assert steps >= 0, "every marker must reach the gray object without crossing blue"
        marker_dists[color] = steps

    best = min(marker_dists.values())
    winners = [color for color, steps in marker_dists.items() if steps == best]
    assert len(winners) == 1, "expected a unique geodesically nearest marker color"
    return winners[0], marker_dists


def _euclid_sq_to_object(cell: Coord, object_coords: list[Coord]) -> int:
    r, c = cell
    return min((r - orow) * (r - orow) + (c - ocol) * (c - ocol) for orow, ocol in object_coords)


def _expected_output(grid: np.ndarray) -> np.ndarray:
    g, object_coords, marker_info = _extract_structure(grid)
    winner_color, _ = _winner_from_distances(g, object_coords, marker_info)
    out = g.copy()
    for r, c in object_coords:
        out[r, c] = winner_color
    for _, (r, c) in marker_info:
        out[r, c] = Color.BLACK
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the gray object with the marker color that is closest by walkable path, not straight-line distance.

    Input structure:
    - A black-background grid contains blue wall pixels, one gray 4-connected object, and several isolated marker pixels of distinct colors.
    - Every marker can reach the gray object through 4-neighbor steps if blue cells are treated as blocked.

    Rule:
    - For each marker color, measure the shortest 4-neighbor path length from that marker pixel to any gray-object pixel while forbidding traversal through blue walls.
    - Choose the uniquely smallest such path length and recolor the entire gray object with that marker's color.
    - Erase all marker pixels to black while leaving the blue walls and all other cells unchanged.

    Key insight:
    - The deciding notion of "nearest" is maze distance through the open cells, so a marker that looks closer across a wall can still lose to a farther-looking marker with a shorter walkable route.
    """
    return _expected_output(grid)


def _lane_geometry(widths: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    lane_bounds: list[tuple[int, int]] = []
    wall_cols: list[int] = []
    col = 0
    for idx, width in enumerate(widths):
        lane_bounds.append((col, col + width - 1))
        col += width
        if idx != len(widths) - 1:
            wall_cols.append(col)
            col += 1
    return lane_bounds, wall_cols, col


def _shape_coords(shape: np.ndarray, top: int, left: int) -> list[Coord]:
    coords = np.argwhere(shape)
    return [(top + int(r), left + int(c)) for r, c in coords]


def _paint_coords(grid: np.ndarray, coords: list[Coord], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _choose_gap_row(height: int, near_top: bool) -> int:
    if near_top:
        return random.randint(1, min(3, height - 2))
    return random.randint(max(1, height - 4), height - 2)


def _lane_index(col: int, lane_bounds: list[tuple[int, int]]) -> int:
    for idx, (c0, c1) in enumerate(lane_bounds):
        if c0 <= col <= c1:
            return idx
    return -1


def _far_row_candidates(height: int, shape_h: int, gap_row: int) -> list[int]:
    candidates = []
    for top in range(1, height - shape_h):
        center = top + (shape_h - 1) / 2.0
        if abs(center - gap_row) >= max(3, height // 4):
            candidates.append(top)
    return candidates or list(range(1, height - shape_h))


def _chebyshev_far(a: Coord, b: Coord) -> bool:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) >= 2


def _sample_case() -> np.ndarray | None:
    height = random.randint(*GEN_HEIGHT_RANGE)
    lane_count = random.randint(*GEN_LANE_COUNT_RANGE)
    widths = [random.randint(*GEN_LANE_WIDTH_RANGE) for _ in range(lane_count)]
    lane_bounds, wall_cols, width = _lane_geometry(widths)

    grid = np.full((height, width), Color.BLACK, dtype=int)

    top_first = random.choice((True, False))
    gap_rows: list[int] = []
    for idx, wall_col in enumerate(wall_cols):
        gap_row = _choose_gap_row(height, near_top=top_first if idx % 2 == 0 else not top_first)
        gap_rows.append(gap_row)
        grid[:, wall_col] = Color.BLUE
        grid[gap_row, wall_col] = Color.BLACK

    wall_idx = random.randrange(len(wall_cols))
    side = random.choice(("left", "right"))
    object_lane = wall_idx if side == "left" else wall_idx + 1
    shape_choices = [shape for shape in OBJECT_SHAPES if shape.shape[1] <= widths[object_lane]]
    if not shape_choices:
        return None
    shape = random.choice(shape_choices)
    shape_h, shape_w = map(int, shape.shape)

    wall_col = wall_cols[wall_idx]
    object_top_choices = _far_row_candidates(height, shape_h, gap_rows[wall_idx])
    if not object_top_choices:
        return None
    object_top = random.choice(object_top_choices)
    object_left = wall_col - shape_w if side == "left" else wall_col + 1
    object_coords = _shape_coords(shape, object_top, object_left)
    _paint_coords(grid, object_coords, Color.GREY)

    dist = _distance_map(grid, object_coords)
    boundary_col = wall_col - 1 if side == "left" else wall_col + 1
    decoy_col = wall_col + 1 if side == "left" else wall_col - 1
    boundary_rows = sorted({r for r, c in object_coords if c == boundary_col})
    decoy_options = [
        (r, decoy_col)
        for r in boundary_rows
        if in_bounds(grid.shape, r, decoy_col) and int(grid[r, decoy_col]) == Color.BLACK and int(dist[r, decoy_col]) >= 0
    ]
    if not decoy_options:
        return None
    decoy = max(decoy_options, key=lambda rc: int(dist[rc]))
    decoy_dist = int(dist[decoy])
    decoy_euclid = _euclid_sq_to_object(decoy, object_coords)
    if decoy_euclid >= 9 or decoy_dist < 8:
        return None

    lane_c0, lane_c1 = lane_bounds[object_lane]
    winner_candidates: list[tuple[int, int, Coord]] = []
    for r in range(height):
        for c in range(lane_c0, lane_c1 + 1):
            if int(grid[r, c]) != Color.BLACK:
                continue
            steps = int(dist[r, c])
            if steps < 4 or steps > min(decoy_dist - 3, 10):
                continue
            euclid_sq = _euclid_sq_to_object((r, c), object_coords)
            if euclid_sq <= decoy_euclid:
                continue
            winner_candidates.append((steps, euclid_sq, (r, c)))
    if not winner_candidates:
        return None
    winner_candidates.sort(key=lambda item: (item[0], item[1]))
    smallest = winner_candidates[0][0]
    near_best = [item for item in winner_candidates if item[0] <= smallest + 1]
    winner_dist, _, winner = random.choice(near_best)
    if not _chebyshev_far(winner, decoy):
        return None

    target_markers = random.randint(*GEN_MARKER_COUNT_RANGE)
    need_extra = target_markers - 2
    occupied = {winner, decoy, *object_coords}
    other_lane_pool: list[Coord] = []
    same_lane_pool: list[Coord] = []
    for r in range(height):
        for c in range(width):
            if int(grid[r, c]) != Color.BLACK:
                continue
            cell = (r, c)
            if cell in occupied:
                continue
            if int(dist[cell]) < winner_dist + 2:
                continue
            if not all(_chebyshev_far(cell, ref) for ref in (winner, decoy)):
                continue
            lane_idx = _lane_index(c, lane_bounds)
            if lane_idx != object_lane:
                other_lane_pool.append(cell)
            else:
                same_lane_pool.append(cell)
    random.shuffle(other_lane_pool)
    random.shuffle(same_lane_pool)

    extras: list[Coord] = []
    if need_extra and other_lane_pool:
        extras.append(other_lane_pool.pop())
    combined_pool = other_lane_pool + same_lane_pool
    random.shuffle(combined_pool)
    for cell in combined_pool:
        if len(extras) == need_extra:
            break
        if all(_chebyshev_far(cell, ref) for ref in extras):
            extras.append(cell)
    if len(extras) != need_extra:
        return None

    marker_cells = [winner, decoy, *extras]
    marker_colors = random.sample(MARKER_COLORS, len(marker_cells))
    for color, cell in zip(marker_colors, marker_cells):
        grid[cell] = color

    transformed = np.rot90(grid, random.randrange(4))
    if random.random() < 0.5:
        transformed = np.fliplr(transformed)
    if random.random() < 0.5:
        transformed = np.flipud(transformed)
    return transformed.copy()


def _semantic_summary(grid: np.ndarray) -> dict[str, object]:
    g, object_coords, marker_info = _extract_structure(grid)
    winner_color, marker_dists = _winner_from_distances(g, object_coords, marker_info)
    euclid_sq = {color: _euclid_sq_to_object(coord, object_coords) for color, coord in marker_info}
    winner_euclid = euclid_sq[winner_color]
    counterexamples = [
        color
        for color, _ in marker_info
        if color != winner_color and euclid_sq[color] < winner_euclid and marker_dists[color] > marker_dists[winner_color]
    ]
    assert counterexamples, "generator must include a straight-line decoy marker"
    return {
        "output": _expected_output(g),
        "winner_color": winner_color,
        "marker_dists": marker_dists,
        "euclid_sq": euclid_sq,
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _sample_case()
        if grid is None:
            continue
        try:
            summary = _semantic_summary(grid)
        except AssertionError:
            continue
        if np.array_equal(main(grid), summary["output"]):
            return grid
    raise RuntimeError("failed to generate a valid geodesic-nearest ARC input")
