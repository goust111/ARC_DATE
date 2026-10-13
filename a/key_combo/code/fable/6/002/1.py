from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


CENTER_COLOR = 2
OBSTACLE_COLOR = 5
ROLE_COLORS = {CENTER_COLOR, OBSTACLE_COLOR}
COLOR_POOL = tuple(range(10))
FREE_COLORS = tuple(c for c in COLOR_POOL if c not in ROLE_COLORS)

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
CENTER_MARGIN = 4
OBSTACLE_COUNT_CHOICES = (0, 1, 2, 3)
OBSTACLE_COUNT_WEIGHTS = (0.28, 0.24, 0.30, 0.18)
OBSTACLE_HEIGHT_RANGE = (2, 5)
OBSTACLE_WIDTH_RANGE = (2, 6)
OBSTACLE_RADIUS_RANGE = (2, 7)
MARKER_COUNT_RANGE = (2, 4)
MIN_LAYER_SIZE = 6
MIN_MARKER_SEPARATION = 2
MAX_GENERATION_ATTEMPTS = 400

_DIR8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def get_bg(grid: np.ndarray) -> int:
    """Background color = most common value in the grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Return True iff (r,c) is within a grid of shape (H,W)."""
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors8(r: int, c: int):
    """8-neighborhood of (r,c)."""
    for dr, dc in _DIR8:
        yield int(r + dr), int(c + dc)


def _distance_field(shape: tuple[int, int], start: tuple[int, int], blocked: np.ndarray) -> np.ndarray:
    """8-connected shortest-path distances through all non-blocked cells."""
    h, w = map(int, shape)
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = map(int, start)
    if blocked[sr, sc]:
        return dist
    dist[sr, sc] = 0
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        nd = int(dist[r, c] + 1)
        for nr, nc in neighbors8(r, c):
            if not in_bounds((h, w), nr, nc):
                continue
            if blocked[nr, nc] or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = nd
            q.append((nr, nc))
    return dist


def _plain_distance_field(shape: tuple[int, int], center: tuple[int, int]) -> np.ndarray:
    """Obstacle-free 8-connected distance = Chebyshev distance to the center."""
    h, w = map(int, shape)
    rr, cc = np.indices((h, w))
    cr, c0 = map(int, center)
    return np.maximum(np.abs(rr - cr), np.abs(cc - c0)).astype(int)


def _extract_markers(grid: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    mask = (grid != int(bg)) & (grid != CENTER_COLOR) & (grid != OBSTACLE_COLOR)
    return [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(mask)]


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    center_coords = np.argwhere(g == CENTER_COLOR)
    assert center_coords.shape[0] == 1, "expected exactly one center pixel"
    center = tuple(map(int, center_coords[0]))

    blocked = g == OBSTACLE_COLOR
    dist = _distance_field(g.shape, center, blocked)
    assert int(dist[center]) == 0, "center must be reachable"

    by_distance: dict[int, int] = {}
    for r, c, color in _extract_markers(g, bg):
        d = int(dist[r, c])
        assert d > 0, "markers must lie on reachable non-center cells"
        if d in by_distance:
            assert by_distance[d] == color, "markers on the same distance contour must agree"
        else:
            by_distance[d] = int(color)

    out = np.full_like(g, bg)
    out[blocked] = OBSTACLE_COLOR
    out[center] = CENTER_COLOR
    for d, color in sorted(by_distance.items()):
        out[dist == int(d)] = int(color)
    out[blocked] = OBSTACLE_COLOR
    out[center] = CENTER_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color every obstacle-aware expansion contour with the marker that sits on the same geodesic layer.

    Input structure:
    - One dominant background color fills most cells.
    - A single red center pixel is the flood source, and gray solid rectangles are impassable obstacles.
    - Several other non-background, non-gray single pixels act as colored markers.

    Rule:
    - Run an 8-connected breadth-first expansion from the red center through every non-gray cell, assigning each reachable cell its obstacle-aware step distance.
    - For each marker, read the distance of its own cell and repaint every cell at exactly that distance with the marker's color.
    - Keep the red center and all gray obstacle cells unchanged; reachable cells on larger unclaimed distances stay background.

    Key insight: A marker does not specify a geometric radius around itself; it claims the whole equal-distance contour of the center's wave after the wave bends around obstacles.
    """
    return _solve(grid)


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _bbox_point_chebyshev(rect: tuple[int, int, int, int], point: tuple[int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    pr, pc = map(int, point)
    dr = 0 if r0 <= pr <= r1 else min(abs(pr - r0), abs(pr - r1))
    dc = 0 if c0 <= pc <= c1 else min(abs(pc - c0), abs(pc - c1))
    return max(dr, dc)


def _rect_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, rect)
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _place_obstacles(shape: tuple[int, int], center: tuple[int, int], count: int) -> np.ndarray | None:
    h, w = map(int, shape)
    blocked = np.zeros((h, w), dtype=bool)
    directions = list(_DIR8)
    random.shuffle(directions)

    for idx in range(int(count)):
        placed = False
        for _ in range(160):
            oh = random.randint(*OBSTACLE_HEIGHT_RANGE)
            ow = random.randint(*OBSTACLE_WIDTH_RANGE)
            dr, dc = directions[idx % len(directions)] if random.random() < 0.7 else random.choice(_DIR8)
            radius = random.randint(*OBSTACLE_RADIUS_RANGE)
            mid_r = center[0] + dr * radius + random.randint(-1, 1)
            mid_c = center[1] + dc * radius + random.randint(-1, 1)
            if dr == 0:
                mid_r += random.randint(-2, 2)
            if dc == 0:
                mid_c += random.randint(-2, 2)

            r0 = int(mid_r - oh // 2)
            c0 = int(mid_c - ow // 2)
            r1 = int(r0 + oh - 1)
            c1 = int(c0 + ow - 1)
            if r0 < 1 or c0 < 1 or r1 >= h - 1 or c1 >= w - 1:
                continue

            rect = (r0, c0, r1, c1)
            if _bbox_point_chebyshev(rect, center) < 2:
                continue
            if blocked[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)].any():
                continue

            blocked[r0 : r1 + 1, c0 : c1 + 1] = True
            placed = True
            break
        if not placed:
            return None

    return blocked


def _layer_catalog(
    dist: np.ndarray, plain: np.ndarray, blocked: np.ndarray
) -> tuple[list[int], list[int], dict[int, int], dict[int, list[tuple[int, int]]]]:
    reachable = dist >= 0
    max_d = int(dist[reachable].max())
    usable: list[int] = []
    impacted: list[int] = []
    changed_sizes: dict[int, int] = {}
    cells_by_layer: dict[int, list[tuple[int, int]]] = {}

    for d in range(2, max_d):
        mask = dist == d
        size = int(mask.sum())
        if size < MIN_LAYER_SIZE:
            continue
        usable.append(d)
        cells_by_layer[d] = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        plain_mask = (plain == d) & (~blocked)
        changed = int(np.count_nonzero(mask ^ plain_mask))
        changed_sizes[d] = changed
        if changed > 0:
            impacted.append(d)

    return usable, impacted, changed_sizes, cells_by_layer


def _choose_distances(usable: list[int], impacted: list[int], changed_sizes: dict[int, int]) -> list[int]:
    if len(usable) < 2:
        return []
    count = random.randint(MARKER_COUNT_RANGE[0], min(MARKER_COUNT_RANGE[1], len(usable)))
    chosen: list[int] = []

    if impacted:
        forced = max(impacted, key=lambda d: (changed_sizes[d], -abs(d - np.median(usable))))
        chosen.append(int(forced))

    while len(chosen) < count:
        remaining = [d for d in usable if d not in chosen]
        if not remaining:
            break
        if not chosen:
            target = np.median(remaining)
            pick = min(remaining, key=lambda d: abs(d - target))
        else:
            pick = max(
                remaining,
                key=lambda d: (
                    min(abs(d - c) for c in chosen),
                    min(d - usable[0], usable[-1] - d),
                ),
            )
        chosen.append(int(pick))

    return sorted(chosen)


def _pick_marker_positions(
    distances: list[int],
    cells_by_layer: dict[int, list[tuple[int, int]]],
    plain: np.ndarray,
    blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    h, w = map(int, plain.shape)
    chosen: list[tuple[int, int]] = []

    for d in distances:
        layer_cells = list(cells_by_layer[d])
        plain_mask = (plain == d) & (~blocked)
        changed = {cell for cell in layer_cells if not plain_mask[cell]}

        interior = [cell for cell in layer_cells if 0 < cell[0] < h - 1 and 0 < cell[1] < w - 1]
        if interior:
            layer_cells = interior
        random.shuffle(layer_cells)

        def score(cell: tuple[int, int]) -> tuple[int, int, int]:
            min_sep = min((_chebyshev(cell, other) for other in chosen), default=99)
            edge_clearance = min(cell[0], cell[1], h - 1 - cell[0], w - 1 - cell[1])
            return (1 if cell in changed else 0, min_sep, edge_clearance)

        layer_cells.sort(key=score, reverse=True)
        pick = None
        for cell in layer_cells:
            if all(_chebyshev(cell, other) >= MIN_MARKER_SEPARATION for other in chosen):
                pick = cell
                break
        if pick is None:
            pick = layer_cells[0] if layer_cells else None
        if pick is None:
            return None
        chosen.append(pick)

    for i, a in enumerate(chosen):
        for b in chosen[i + 1 :]:
            if _chebyshev(a, b) < MIN_MARKER_SEPARATION:
                return None
    return chosen


def _semantic_ok(
    grid: np.ndarray,
    selected_distances: list[int],
    obstacle_count: int,
    dist: np.ndarray,
    plain: np.ndarray,
) -> bool:
    bg = int(get_bg(grid))
    markers = _extract_markers(grid, bg)
    if len(markers) != len(selected_distances):
        return False
    if len(set(selected_distances)) != len(selected_distances):
        return False
    if int(np.count_nonzero(grid == CENTER_COLOR)) != 1:
        return False
    if obstacle_count == 0 and int(np.count_nonzero(grid == OBSTACLE_COLOR)) != 0:
        return False

    reachable = dist >= 0
    if not reachable.any():
        return False
    if max(selected_distances) >= int(dist[reachable].max()):
        return False

    out = _solve(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out == OBSTACLE_COLOR)) != int(np.count_nonzero(grid == OBSTACLE_COLOR)):
        return False
    if int(np.count_nonzero(out == CENTER_COLOR)) != 1:
        return False

    if obstacle_count > 0:
        changed_layers = [d for d in selected_distances if np.any((dist == d) ^ ((plain == d) & (grid != OBSTACLE_COLOR)))]
        if not changed_layers:
            return False
        no_obstacle_out = np.full_like(grid, bg)
        no_obstacle_out[grid == CENTER_COLOR] = CENTER_COLOR
        no_obstacle_out[grid == OBSTACLE_COLOR] = OBSTACLE_COLOR
        for r, c, color in markers:
            no_obstacle_out[(plain == plain[r, c]) & (grid != OBSTACLE_COLOR)] = color
        no_obstacle_out[grid == CENTER_COLOR] = CENTER_COLOR
        no_obstacle_out[grid == OBSTACLE_COLOR] = OBSTACLE_COLOR
        if np.array_equal(out, no_obstacle_out):
            return False

    return True


def _sample_case() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(FREE_COLORS)

    center = (
        random.randint(CENTER_MARGIN, h - CENTER_MARGIN - 1),
        random.randint(CENTER_MARGIN, w - CENTER_MARGIN - 1),
    )

    obstacle_count = random.choices(OBSTACLE_COUNT_CHOICES, weights=OBSTACLE_COUNT_WEIGHTS, k=1)[0]
    blocked = np.zeros((h, w), dtype=bool)
    if obstacle_count:
        placed = _place_obstacles((h, w), center, obstacle_count)
        if placed is None:
            return None
        blocked = placed

    dist = _distance_field((h, w), center, blocked)
    if int(dist[center]) != 0:
        return None
    plain = _plain_distance_field((h, w), center)

    usable, impacted, changed_sizes, cells_by_layer = _layer_catalog(dist, plain, blocked)
    if len(usable) < 2:
        return None
    if obstacle_count > 0 and not impacted:
        return None

    selected_distances = _choose_distances(usable, impacted, changed_sizes)
    if len(selected_distances) < 2:
        return None

    marker_positions = _pick_marker_positions(selected_distances, cells_by_layer, plain, blocked)
    if marker_positions is None:
        return None

    marker_colors = random.sample([c for c in FREE_COLORS if c != bg], len(selected_distances))

    grid = np.full((h, w), bg, dtype=int)
    grid[blocked] = OBSTACLE_COLOR
    grid[center] = CENTER_COLOR
    for (r, c), color in zip(marker_positions, marker_colors):
        grid[r, c] = int(color)

    if get_bg(grid) != bg:
        return None
    if not _semantic_ok(grid, selected_distances, obstacle_count, dist, plain):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _sample_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a non-degenerate geodesic contour instance")
