import random
from dataclasses import dataclass

import numpy as np


BG = 0
GREEN = 3
PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)
SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 500


@dataclass(frozen=True)
class Arrow:
    edge: str
    axis: tuple[int, int]
    step: tuple[int, int]
    order: int
    coords: frozenset[tuple[int, int]]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _trace(shape: tuple[int, int], axis: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = axis
    dr, dc = step
    cells = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _make_arrow(edge: str, pos: int, height: int, width: int) -> Arrow:
    if edge == "top":
        coords = frozenset({(0, pos - 1), (1, pos), (0, pos + 1)})
        return Arrow(edge, (1, pos), (1, 0), pos, coords)
    if edge == "right":
        coords = frozenset({(pos - 1, width - 1), (pos, width - 2), (pos + 1, width - 1)})
        return Arrow(edge, (pos, width - 2), (0, -1), pos, coords)
    if edge == "bottom":
        coords = frozenset({(height - 1, pos - 1), (height - 2, pos), (height - 1, pos + 1)})
        return Arrow(edge, (height - 2, pos), (-1, 0), -pos, coords)
    if edge == "left":
        coords = frozenset({(pos - 1, 0), (pos, 1), (pos + 1, 0)})
        return Arrow(edge, (pos, 1), (0, 1), -pos, coords)
    raise ValueError(f"unknown edge: {edge}")


def _candidate_arrows(g: np.ndarray) -> list[Arrow]:
    height, width = g.shape
    arrows: list[Arrow] = []

    for c in range(1, width - 1):
        coords = ((0, c - 1), (1, c), (0, c + 1))
        if all(int(g[p]) == GREEN for p in coords):
            arrows.append(_make_arrow("top", c, height, width))

    for r in range(1, height - 1):
        coords = ((r - 1, width - 1), (r, width - 2), (r + 1, width - 1))
        if all(int(g[p]) == GREEN for p in coords):
            arrows.append(_make_arrow("right", r, height, width))

    for c in range(1, width - 1):
        coords = ((height - 1, c - 1), (height - 2, c), (height - 1, c + 1))
        if all(int(g[p]) == GREEN for p in coords):
            arrows.append(_make_arrow("bottom", c, height, width))

    for r in range(1, height - 1):
        coords = ((r - 1, 0), (r, 1), (r + 1, 0))
        if all(int(g[p]) == GREEN for p in coords):
            arrows.append(_make_arrow("left", r, height, width))

    return arrows


def _find_arrows(g: np.ndarray) -> list[Arrow]:
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 3 and width >= 3, "grid is too small for edge arrows"

    green_cells = {tuple(map(int, p)) for p in np.argwhere(g == GREEN)}
    assert green_cells, "expected at least one green edge arrow"

    covered: set[tuple[int, int]] = set()
    arrows: list[Arrow] = []
    for arrow in _candidate_arrows(g):
        assert not (set(arrow.coords) & covered), "green arrow markers must not overlap"
        covered.update(arrow.coords)
        arrows.append(arrow)

    assert green_cells == covered, "every green pixel must belong to a three-pixel edge arrow"
    return arrows


def _ordered(arrows: list[Arrow]) -> list[Arrow]:
    by_edge = {edge: [] for edge in SIDES}
    for arrow in arrows:
        by_edge[arrow.edge].append(arrow)

    result: list[Arrow] = []
    for edge in SIDES:
        result.extend(sorted(by_edge[edge], key=lambda a: a.order))
    return result


def _draw_rays(g: np.ndarray, arrows: list[Arrow]) -> np.ndarray:
    out = g.copy()
    shape = tuple(map(int, g.shape))

    for arrow in _ordered(arrows):
        color = GREEN
        for i, (r, c) in enumerate(_trace(shape, arrow.axis, arrow.step)):
            if i and int(g[r, c]) not in (BG, GREEN):
                color = int(g[r, c])
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cast color-absorbing rays from three-pixel green arrows on the grid edge.

    Input structure:
    - The background is black, and every green foreground marker is one
      three-pixel arrow attached to the top, right, bottom, or left edge.
    - Each arrow has one inner axis pixel and two edge wing pixels, so its
      geometry points inward from that edge.
    - The interior may contain isolated non-black, non-green colored obstacle
      pixels.

    Rule:
    - Each arrow emits a one-pixel-wide ray from its axis pixel in the inward
      direction, initially green.
    - When the next input cell on the ray is a colored obstacle, paint that cell
      with the obstacle's original color and continue the ray in that absorbed
      color.
    - Draw arrows in edge order: top left-to-right, right top-to-bottom, bottom
      right-to-left, and left bottom-to-top. Later rays overwrite earlier rays
      at crossings.

    Key insight:
    - The edge-arrow shape encodes both the ray start and direction, while
      obstacle pixels act as visible color pickups that change the ray after
      they are hit.
    """
    g = np.asarray(grid, dtype=int)
    arrows = _find_arrows(g)
    return _draw_rays(g, arrows)


def _line_key(arrow: Arrow) -> tuple[str, int]:
    if arrow.edge in ("top", "bottom"):
        return "vertical", arrow.axis[1]
    return "horizontal", arrow.axis[0]


def _candidate_positions(edge: str, height: int, width: int) -> list[int]:
    if edge in ("top", "bottom"):
        return list(range(2, width - 2))
    return list(range(2, height - 2))


def _path_cover(arrows: list[Arrow], shape: tuple[int, int]) -> dict[tuple[int, int], list[int]]:
    cover: dict[tuple[int, int], list[int]] = {}
    for i, arrow in enumerate(_ordered(arrows)):
        for cell in _trace(shape, arrow.axis, arrow.step):
            cover.setdefault(cell, []).append(i)
    return cover


def _place_arrows(height: int, width: int, count: int) -> list[Arrow] | None:
    sides = [random.choice(("top", "bottom")), random.choice(("left", "right"))]
    sides.extend(random.choice(SIDES) for _ in range(count - 2))
    random.shuffle(sides)

    arrows: list[Arrow] = []
    occupied: set[tuple[int, int]] = set()
    used_lines = {"vertical": set(), "horizontal": set()}

    for edge in sides:
        positions = _candidate_positions(edge, height, width)
        random.shuffle(positions)
        placed = False
        for pos in positions:
            arrow = _make_arrow(edge, pos, height, width)
            category, line = _line_key(arrow)
            if line in used_lines[category] or set(arrow.coords) & occupied:
                continue
            arrows.append(arrow)
            occupied.update(arrow.coords)
            used_lines[category].add(line)
            placed = True
            break
        if not placed:
            return None

    all_arrow_cells = set().union(*(set(a.coords) for a in arrows))
    shape = (height, width)
    for arrow in arrows:
        for cell in _trace(shape, arrow.axis, arrow.step):
            if cell in all_arrow_cells and cell not in arrow.coords:
                return None

    cover = _path_cover(arrows, shape)
    if not any(len(indices) > 1 for indices in cover.values()):
        return None
    return arrows


def _far_from(cells: set[tuple[int, int]], cell: tuple[int, int], dist: int) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > dist for rr, cc in cells)


def _obstacle_ok(
    cell: tuple[int, int],
    shape: tuple[int, int],
    arrows: list[Arrow],
    crossings: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
) -> bool:
    height, width = shape
    r, c = cell
    if not (0 < r < height - 1 and 0 < c < width - 1):
        return False
    arrow_cells = set().union(*(set(a.coords) for a in arrows))
    if cell in arrow_cells or cell in crossings or cell in obstacles:
        return False
    if not _far_from(arrow_cells, cell, 1):
        return False
    if not _far_from(set(obstacles), cell, 1):
        return False
    return True


def _force_visible_crossing_obstacle(
    arrows: list[Arrow],
    shape: tuple[int, int],
    crossings: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
) -> bool:
    ordered = _ordered(arrows)
    path_by_index = [_trace(shape, arrow.axis, arrow.step) for arrow in ordered]
    options = []
    for crossing in crossings:
        ray_indices = [i for i, path in enumerate(path_by_index) if crossing in path]
        if len(ray_indices) < 2:
            continue
        later = max(ray_indices)
        path = path_by_index[later]
        cross_i = path.index(crossing)
        before = [
            cell
            for cell in path[1:cross_i]
            if _obstacle_ok(cell, shape, arrows, crossings, obstacles)
        ]
        if before:
            options.append(random.choice(before))

    if not options:
        return False

    obstacles[random.choice(options)] = random.choice(PALETTE)
    return True


def _add_random_obstacles(
    arrows: list[Arrow],
    shape: tuple[int, int],
    crossings: set[tuple[int, int]],
    obstacles: dict[tuple[int, int], int],
) -> None:
    candidates = []
    for arrow in arrows:
        for cell in _trace(shape, arrow.axis, arrow.step)[1:-1]:
            if _obstacle_ok(cell, shape, arrows, crossings, obstacles):
                candidates.append(cell)
    random.shuffle(candidates)

    target = random.randint(2, min(7, max(2, len(candidates) // 4 + 2)))
    for cell in candidates:
        if len(obstacles) >= target:
            break
        if _obstacle_ok(cell, shape, arrows, crossings, obstacles):
            obstacles[cell] = random.choice(PALETTE)


def _render_input(height: int, width: int, arrows: list[Arrow], obstacles: dict[tuple[int, int], int]) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for arrow in arrows:
        for r, c in arrow.coords:
            grid[r, c] = GREEN
    for (r, c), color in obstacles.items():
        grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(9, 22)
    width = random.randint(9, 22)
    count = random.randint(2, 6)

    arrows = _place_arrows(height, width, count)
    if arrows is None:
        return None

    shape = (height, width)
    cover = _path_cover(arrows, shape)
    crossings = {cell for cell, indices in cover.items() if len(indices) > 1}

    obstacles: dict[tuple[int, int], int] = {}
    if not _force_visible_crossing_obstacle(arrows, shape, crossings, obstacles):
        return None
    _add_random_obstacles(arrows, shape, crossings, obstacles)

    if len(obstacles) < 2:
        return None

    grid = _render_input(height, width, arrows, obstacles)
    expected = _draw_rays(grid, arrows)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None

    visible_crossings = [cell for cell in crossings if cell not in obstacles and int(expected[cell]) != GREEN]
    if not visible_crossings:
        return None

    obstacle_rays = set()
    ordered = _ordered(arrows)
    for i, arrow in enumerate(ordered):
        path = set(_trace(shape, arrow.axis, arrow.step))
        if path & set(obstacles):
            obstacle_rays.add(i)
    if len(obstacle_rays) == len(ordered):
        return None

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid absorbing-arrow-ray input")
