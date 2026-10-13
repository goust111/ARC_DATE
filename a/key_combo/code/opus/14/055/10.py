from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5

DIR_E = (0, 1)
DIR_S = (1, 0)
DIR_SE = (1, 1)
DIR_SW = (1, -1)

HORIZONTAL_TEMPLATES = (
    ((-1, 0), (0, 0), (1, 0)),
    ((-1, 0), (0, 0), (1, 0), (1, 1)),
    ((-1, 1), (0, 0), (0, 1), (1, 1)),
    ((-1, 0), (0, 0), (0, 1), (1, 1)),
    ((-1, 0), (0, 0), (1, 0), (1, 1), (1, 2)),
    ((-1, 1), (0, 0), (0, 1), (0, 2), (1, 1)),
    ((-1, 0), (-1, 1), (0, 0), (0, 1)),
)
DIAGONAL_SE_TEMPLATES = (
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (1, 2)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen: set[tuple[int, int]] = set()
    comps: list[list[tuple[int, int]]] = []
    for r, c in np.argwhere(mask):
        cell = (int(r), int(c))
        if cell in seen:
            continue
        stack = [cell]
        seen.add(cell)
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _marker_pair(grid: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], int]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    assert len(markers) == 2, "expected exactly two red markers"
    start, end = sorted(markers)
    dr = end[0] - start[0]
    dc = end[1] - start[1]
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc), "markers must share a row, column, or 45-degree diagonal"
    step = (
        0 if dr == 0 else dr // abs(dr),
        0 if dc == 0 else dc // abs(dc),
    )
    steps = max(abs(dr), abs(dc))
    assert steps >= 1
    return start, end, step, steps


def _segment(start: tuple[int, int], step: tuple[int, int], steps: int) -> list[tuple[int, int]]:
    return [(start[0] + k * step[0], start[1] + k * step[1]) for k in range(steps + 1)]


def _boundary_edges(obstacle: set[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    edges = []
    for r, c in obstacle:
        edges.extend(
            [
                ((r, c), (r, c + 1)),
                ((r, c + 1), (r + 1, c + 1)),
                ((r + 1, c + 1), (r + 1, c)),
                ((r + 1, c), (r, c)),
            ]
        )

    counts: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    for a, b in edges:
        key = tuple(sorted((a, b)))
        counts[key] = counts.get(key, 0) + 1

    kept = [(a, b) for a, b in edges if counts[tuple(sorted((a, b)))] == 1]
    outgoing: dict[tuple[int, int], tuple[int, int]] = {a: b for a, b in kept}
    start = min(outgoing)
    ordered: list[tuple[tuple[int, int], tuple[int, int]]] = []
    cur = start
    while True:
        nxt = outgoing[cur]
        ordered.append((cur, nxt))
        cur = nxt
        if cur == start:
            break
        assert len(ordered) <= len(kept)
    return ordered


def _outside_cell(edge: tuple[tuple[int, int], tuple[int, int]]) -> tuple[int, int]:
    (r1, c1), (r2, c2) = edge
    if r1 == r2 and c2 == c1 + 1:
        return (r1 - 1, c1)
    if c1 == c2 and r2 == r1 + 1:
        return (r1, c1)
    if r1 == r2 and c2 == c1 - 1:
        return (r1, c2)
    assert c1 == c2 and r2 == r1 - 1
    return (r2, c1 - 1)


def _convex_corner(
    vertex: tuple[int, int],
    d1: tuple[int, int],
    d2: tuple[int, int],
) -> tuple[int, int] | None:
    r, c = vertex
    mapping = {
        ((0, 1), (1, 0)): (r - 1, c),
        ((1, 0), (0, -1)): (r, c),
        ((0, -1), (-1, 0)): (r, c - 1),
        ((-1, 0), (0, 1)): (r - 1, c - 1),
    }
    return mapping.get((d1, d2))


def _obstacle_ring(obstacle: set[tuple[int, int]]) -> list[tuple[int, int]]:
    edges = _boundary_edges(obstacle)
    ring: list[tuple[int, int]] = []
    for idx, edge in enumerate(edges):
        ring.append(_outside_cell(edge))
        nxt = edges[(idx + 1) % len(edges)]
        d1 = (edge[1][0] - edge[0][0], edge[1][1] - edge[0][1])
        d2 = (nxt[1][0] - nxt[0][0], nxt[1][1] - nxt[0][1])
        corner = _convex_corner(edge[1], d1, d2)
        if corner is not None:
            ring.append(corner)

    compact: list[tuple[int, int]] = []
    for cell in ring:
        if not compact or cell != compact[-1]:
            compact.append(cell)
    if compact and compact[0] == compact[-1]:
        compact.pop()
    return compact


def _route_cells(
    grid: np.ndarray,
    *,
    clockwise: bool = True,
) -> list[tuple[int, int]]:
    start, end, step, steps = _marker_pair(grid)
    obstacle_comps = _components(grid == GREY)
    assert len(obstacle_comps) == 1, "expected a single grey obstacle component"
    obstacle = set(obstacle_comps[0])
    assert 3 <= len(obstacle) <= 5, "obstacle must have 3-5 pixels"

    line = _segment(start, step, steps)
    line_index = {cell: idx for idx, cell in enumerate(line)}
    line_hits = [idx for idx, cell in enumerate(line) if cell in obstacle]
    assert line_hits, "obstacle must intersect the straight marker segment"
    assert all(b == a + 1 for a, b in zip(line_hits, line_hits[1:])), "line-obstacle intersection must be contiguous"

    ring = _obstacle_ring(obstacle)
    h, w = grid.shape
    assert all(0 <= r < h and 0 <= c < w for r, c in ring), "reroute ring leaves the grid"
    if not clockwise:
        ring = [ring[0], *reversed(ring[1:])]

    path: list[tuple[int, int]] = []
    cur = start
    while cur != end:
        nxt = (cur[0] + step[0], cur[1] + step[1])
        if nxt in obstacle:
            entry = cur
            assert entry in ring, "entry point must lie on the obstacle ring"
            entry_idx = ring.index(entry)
            exit_cell = None
            for offset in range(1, len(ring) + 1):
                cell = ring[(entry_idx + offset) % len(ring)]
                path.append(cell)
                k = line_index.get(cell)
                if k is None or k <= line_index[entry]:
                    continue
                forward = (cell[0] + step[0], cell[1] + step[1])
                if 0 <= forward[0] < h and 0 <= forward[1] < w and forward not in obstacle:
                    exit_cell = cell
                    break
            assert exit_cell is not None, "failed to find a valid clockwise exit"
            cur = exit_cell
        else:
            cur = nxt
            path.append(cur)

    return path


def _expected_output(grid: np.ndarray) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    start, end, _step, _steps = _marker_pair(out)
    for r, c in _route_cells(out):
        if (r, c) == start or (r, c) == end:
            continue
        out[r, c] = GREEN if int(grid[r, c]) == BLUE else RED
    out[start] = RED
    out[end] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a marker-to-marker line that detours clockwise around the grey obstacle and recolors crossed cells by their original color.

    Input structure:
    - The grid uses black background, scattered blue foreground pixels, exactly two red marker pixels, and one small 4-connected grey obstacle component.
    - The lexicographically first red marker (topmost, then leftmost) and the other marker share a row, a column, or a 45-degree diagonal.
    - The grey obstacle lies on the straight marker segment and fully blocks it, while still leaving a one-pixel outer ring inside the grid.

    Rule:
    - Start from the lexicographically first red marker and walk toward the other marker along the straight queen-move direction.
    - While the next straight step is clear, keep moving straight. When the next step would enter the grey obstacle, switch to the obstacle's outer ring and follow that ring clockwise until you re-enter the original straight segment farther ahead with a clear forward step, then resume the straight walk.
    - Recolor every visited path cell except the two markers: original blue cells become green, and original black cells become red. The grey obstacle and the two red markers remain unchanged.

    Key insight:
    - The detour path is defined by the clockwise obstacle boundary, but the final color of each detour pixel depends on that pixel's original color, so route finding and recoloring must be solved together.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, BLUE, RED, GREY}, "unexpected colors in the input"
    return _expected_output(g)


def _transpose_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((c, r) for r, c in shape)


def _mirror_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((r, -c) for r, c in shape)


def _template_bank(direction: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], ...]:
    if direction == DIR_E:
        return HORIZONTAL_TEMPLATES
    if direction == DIR_S:
        return tuple(_transpose_shape(shape) for shape in HORIZONTAL_TEMPLATES)
    if direction == DIR_SE:
        return DIAGONAL_SE_TEMPLATES
    assert direction == DIR_SW
    return tuple(_mirror_shape(shape) for shape in DIAGONAL_SE_TEMPLATES)


def _choose_layout() -> tuple[
    int,
    int,
    tuple[int, int],
    tuple[int, int],
    tuple[int, int],
    tuple[tuple[int, int], ...],
    int,
]:
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    direction = random.choice((DIR_E, DIR_S, DIR_SE, DIR_SW))
    max_steps = {
        DIR_E: w - 5,
        DIR_S: h - 5,
        DIR_SE: min(h, w) - 5,
        DIR_SW: min(h, w) - 5,
    }[direction]
    steps = random.randint(6, max(6, min(11, max_steps)))
    shape = random.choice(_template_bank(direction))

    if direction == DIR_E:
        start = (random.randint(2, h - 3), random.randint(2, w - steps - 3))
    elif direction == DIR_S:
        start = (random.randint(2, h - steps - 3), random.randint(2, w - 3))
    elif direction == DIR_SE:
        start = (random.randint(2, h - steps - 3), random.randint(2, w - steps - 3))
    else:
        start = (random.randint(2, h - steps - 3), random.randint(steps + 2, w - 3))

    end = (start[0] + steps * direction[0], start[1] + steps * direction[1])
    blocked_start = random.randint(2, steps - 3)
    pivot = (start[0] + blocked_start * direction[0], start[1] + blocked_start * direction[1])
    return h, w, direction, start, end, shape, pivot[0] * 100 + pivot[1]


def _obstacle_from_template(
    shape: tuple[tuple[int, int], ...],
    pivot_code: int,
) -> set[tuple[int, int]]:
    pivot = (pivot_code // 100, pivot_code % 100)
    return {(pivot[0] + r, pivot[1] + c) for r, c in shape}


def _unique_ordered(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    seen: set[tuple[int, int]] = set()
    out: list[tuple[int, int]] = []
    for cell in cells:
        if cell not in seen:
            seen.add(cell)
            out.append(cell)
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        start, end, step, steps = _marker_pair(grid)
        obstacle_comps = _components(grid == GREY)
        if len(obstacle_comps) != 1 or not (3 <= len(obstacle_comps[0]) <= 5):
            return False

        line = _segment(start, step, steps)
        if sum(cell in set(obstacle_comps[0]) for cell in line) == 0:
            return False

        clockwise_route = _unique_ordered(_route_cells(grid, clockwise=True))
        counter_route = _unique_ordered(_route_cells(grid, clockwise=False))
        if clockwise_route == counter_route:
            return False

        routed = [cell for cell in clockwise_route if cell not in {start, end}]
        if len(routed) < 4:
            return False
        if not any(int(grid[r, c]) == BLUE for r, c in routed):
            return False
        if not any(int(grid[r, c]) == BLACK for r, c in routed):
            return False

        straight = [cell for cell in line[1:-1] if cell not in set(obstacle_comps[0])]
        if routed == straight:
            return False

        out = main(grid)
        expected = _expected_output(grid)
        if not np.array_equal(out, expected):
            return False
    except Exception:
        return False
    return True


def _attempt() -> np.ndarray | None:
    h, w, direction, start, end, shape, pivot_code = _choose_layout()
    obstacle = _obstacle_from_template(shape, pivot_code)

    if start in obstacle or end in obstacle:
        return None
    if any(not (0 <= r < h and 0 <= c < w) for r, c in obstacle):
        return None

    grid = np.zeros((h, w), dtype=int)
    grid[start] = RED
    grid[end] = RED
    for r, c in obstacle:
        grid[r, c] = GREY

    try:
        route = _unique_ordered(_route_cells(grid, clockwise=True))
    except Exception:
        return None

    routed = [cell for cell in route if cell not in {start, end}]
    available = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == BLACK]
    if len(routed) < 4 or len(available) < 10:
        return None

    path_blue = random.randint(1, max(1, min(len(routed) - 1, len(routed) // 2)))
    for r, c in random.sample(routed, path_blue):
        grid[r, c] = BLUE

    target_blue = random.randint(int(np.ceil(0.15 * h * w)), int(np.floor(0.25 * h * w)))
    current_blue = int(np.count_nonzero(grid == BLUE))
    off_path = [(r, c) for r, c in available if (r, c) not in routed]
    if current_blue > target_blue or len(off_path) < target_blue - current_blue:
        return None
    for r, c in random.sample(off_path, target_blue - current_blue):
        grid[r, c] = BLUE

    if not _validate_instance(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid obstacle-reroute input")
