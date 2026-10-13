import random

import numpy as np


BG = 0
GEN_RETRIES = 400


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _rotate_left(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return -dc, dr


def _rotate_right(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return dc, -dr


def _boundary_sides(r: int, c: int, shape: tuple[int, int]) -> set[str]:
    h, w = shape
    sides = set()
    if r == 0:
        sides.add("top")
    if r == h - 1:
        sides.add("bottom")
    if c == 0:
        sides.add("left")
    if c == w - 1:
        sides.add("right")
    return sides


def _components_by_color(grid: np.ndarray) -> list[tuple[int, list[tuple[int, int]]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, list[tuple[int, int]]]] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, grid.shape):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append((color, cells))
    return components


def _find_path_component(grid: np.ndarray) -> tuple[int, set[tuple[int, int]], list[tuple[int, int]]]:
    candidates = []
    for color, cells in _components_by_color(grid):
        if len(cells) < 5:
            continue
        cell_set = set(cells)
        degrees = {
            cell: sum((nbr in cell_set) for nbr in _neighbors4(cell[0], cell[1], grid.shape))
            for cell in cells
        }
        if any(deg not in (1, 2) for deg in degrees.values()):
            continue
        endpoints = [cell for cell, deg in degrees.items() if deg == 1]
        if len(endpoints) != 2:
            continue
        boundary_cells = [cell for cell in cells if _boundary_sides(cell[0], cell[1], grid.shape)]
        if set(boundary_cells) != set(endpoints):
            continue
        endpoint_sides = _boundary_sides(endpoints[0][0], endpoints[0][1], grid.shape) | _boundary_sides(
            endpoints[1][0], endpoints[1][1], grid.shape
        )
        if len(endpoint_sides) < 2:
            continue
        candidates.append((color, cell_set, endpoints))
    assert len(candidates) == 1, "expected exactly one edge-to-edge path component"
    return candidates[0]


def _trace_path(path_cells: set[tuple[int, int]], start: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    ordered = [start]
    prev = None
    cur = start
    while True:
        nxts = [nbr for nbr in _neighbors4(cur[0], cur[1], shape) if nbr in path_cells and nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path should not branch"
        prev, cur = cur, nxts[0]
        ordered.append(cur)
    return ordered


def _template_patch(grid: np.ndarray, center: tuple[int, int]) -> np.ndarray | None:
    r, c = center
    if r - 1 < 0 or c - 1 < 0 or r + 1 >= grid.shape[0] or c + 1 >= grid.shape[1]:
        return None
    return grid[r - 1 : r + 2, c - 1 : c + 2].copy()


def _extract_start_templates(
    grid: np.ndarray,
    path_cells: set[tuple[int, int]],
    endpoints: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], np.ndarray, np.ndarray]:
    matches = []
    for endpoint in endpoints:
        neighbors = [nbr for nbr in _neighbors4(endpoint[0], endpoint[1], grid.shape) if nbr in path_cells]
        assert len(neighbors) == 1, "endpoint must have exactly one path neighbor"
        nr, nc = neighbors[0]
        direction = (nr - endpoint[0], nc - endpoint[1])
        left_dir = _rotate_left(direction)
        right_dir = _rotate_right(direction)
        left_center = (endpoint[0] + direction[0] + 2 * left_dir[0], endpoint[1] + direction[1] + 2 * left_dir[1])
        right_center = (
            endpoint[0] + direction[0] + 2 * right_dir[0],
            endpoint[1] + direction[1] + 2 * right_dir[1],
        )
        left_patch = _template_patch(grid, left_center)
        right_patch = _template_patch(grid, right_center)
        if left_patch is None or right_patch is None:
            continue
        if np.any(left_patch != BG) and np.any(right_patch != BG):
            matches.append((endpoint, direction, left_patch, right_patch))
    assert len(matches) == 1, "expected exactly one template-bearing start endpoint"
    return matches[0]


def _stamp_template(out: np.ndarray, template: np.ndarray, center: tuple[int, int]) -> None:
    cr, cc = center
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            color = int(template[dr + 1, dc + 1])
            if color == BG:
                continue
            rr, cc2 = cr + dr, cc + dc
            if 0 <= rr < out.shape[0] and 0 <= cc2 < out.shape[1] and int(out[rr, cc2]) == BG:
                out[rr, cc2] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the unique edge-to-edge orthogonal path and stamp the left or right
    3x3 start template at each 90-degree turn, chosen by the turn direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _path_color, path_cells, endpoints = _find_path_component(g)
    start, _start_direction, left_template, right_template = _extract_start_templates(g, path_cells, endpoints)
    path = _trace_path(path_cells, start, g.shape)

    out = g.copy()
    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        cur = path[i]
        nxt = path[i + 1]
        prev_dir = (cur[0] - prev[0], cur[1] - prev[1])
        next_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
        if prev_dir == next_dir:
            continue
        if next_dir == _rotate_left(prev_dir):
            template = left_template
        else:
            assert next_dir == _rotate_right(prev_dir), "expected a 90-degree turn"
            template = right_template
        _stamp_template(out, template, cur)
    return out


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _random_template(color: int) -> np.ndarray:
    while True:
        mask = np.zeros((3, 3), dtype=bool)
        count = random.randint(3, 6)
        cells = random.sample([(r, c) for r in range(3) for c in range(3)], count)
        for r, c in cells:
            mask[r, c] = True
        if not np.any(mask & ~np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=bool)):
            continue
        patch = np.zeros((3, 3), dtype=int)
        patch[mask] = color
        return patch


def _distribute(total: int, minima: list[int]) -> list[int]:
    parts = minima[:]
    extra = total - sum(parts)
    for _ in range(extra):
        parts[random.randrange(len(parts))] += 1
    return parts


def _build_canonical_path(h: int, w: int) -> list[tuple[int, int]] | None:
    max_horizontal = min(4, (h - 4) // 3)
    if max_horizontal < 1 or w < 8:
        return None
    k = random.randint(1, max_horizontal)
    vertical_lengths = _distribute(h - 1, [4] + [3] * (k - 1) + [2])

    cols = [random.randint(3, w - 4)]
    for _ in range(k):
        cur = cols[-1]
        candidates = [c for c in range(1, w - 1) if abs(c - cur) >= 3]
        if not candidates:
            return None
        cols.append(random.choice(candidates))

    path = [(0, cols[0])]
    r, c = path[0]
    for i, vlen in enumerate(vertical_lengths):
        for _ in range(vlen):
            r += 1
            path.append((r, c))
        if i == k:
            break
        target_c = cols[i + 1]
        step = 1 if target_c > c else -1
        for _ in range(abs(target_c - c)):
            c += step
            path.append((r, c))
    if path[-1][0] != h - 1:
        return None
    return path


def _render_input(path: list[tuple[int, int]], shape: tuple[int, int], path_color: int, left_template: np.ndarray, right_template: np.ndarray) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for r, c in path:
        grid[r, c] = path_color

    start = path[0]
    direction = (path[1][0] - start[0], path[1][1] - start[1])
    left_dir = _rotate_left(direction)
    right_dir = _rotate_right(direction)
    left_center = (start[0] + direction[0] + 2 * left_dir[0], start[1] + direction[1] + 2 * left_dir[1])
    right_center = (start[0] + direction[0] + 2 * right_dir[0], start[1] + direction[1] + 2 * right_dir[1])
    _paste(grid, left_template, left_center[0] - 1, left_center[1] - 1)
    _paste(grid, right_template, right_center[0] - 1, right_center[1] - 1)
    return grid


def _expected_output(inp: np.ndarray, path: list[tuple[int, int]], left_template: np.ndarray, right_template: np.ndarray) -> np.ndarray:
    out = inp.copy()
    for i in range(1, len(path) - 1):
        prev = path[i - 1]
        cur = path[i]
        nxt = path[i + 1]
        prev_dir = (cur[0] - prev[0], cur[1] - prev[1])
        next_dir = (nxt[0] - cur[0], nxt[1] - cur[1])
        if prev_dir == next_dir:
            continue
        template = left_template if next_dir == _rotate_left(prev_dir) else right_template
        _stamp_template(out, template, cur)
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 21)
    w = random.randint(10, 21)
    path = _build_canonical_path(h, w)
    if path is None:
        return None

    path_color, left_color, right_color = random.sample(range(1, 10), 3)
    left_template = _random_template(left_color)
    right_template = _random_template(right_color)

    inp = _render_input(path, (h, w), path_color, left_template, right_template)
    out = _expected_output(inp, path, left_template, right_template)
    if np.array_equal(inp, out):
        return None

    k_rot = random.randint(0, 3)
    inp = np.rot90(inp, k_rot).copy()
    out = np.rot90(out, k_rot).copy()

    try:
        if np.array_equal(main(inp), out):
            return inp
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
