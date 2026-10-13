from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]


def _neighbors4(r: int, c: int):
    """Yield neighbors using the shared ARC convention: coordinates are (row, col)."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[Coord]]:
    """Small self-contained equivalent of arc_utils.components_mask (4-connectivity)."""
    H, W = mask.shape
    unseen = {tuple(map(int, p)) for p in np.argwhere(mask)}
    result: list[list[Coord]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        comp = []
        while queue:
            cell = queue.popleft()
            comp.append(cell)
            for nr, nc in _neighbors4(*cell):
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) in unseen:
                    unseen.remove((nr, nc))
                    queue.append((nr, nc))
        result.append(comp)
    return result


def _solid_rectangles(mask: np.ndarray) -> list[list[Coord]] | None:
    """Return the rectangular components, or None if any component is not solid."""
    comps = _components(mask)
    if not comps:
        return None
    for comp in comps:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return None
        if len(comp) < 4 or r0 == 0:
            return None
    return comps


def _ordered_anchored_path(mask: np.ndarray) -> list[Coord] | None:
    """Order an induced, unbranched 4-connected path from its unique top endpoint."""
    cells = {tuple(map(int, p)) for p in np.argwhere(mask)}
    if len(cells) < 3:
        return None
    degrees = {
        cell: sum(neighbor in cells for neighbor in _neighbors4(*cell))
        for cell in cells
    }
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    if len(endpoints) != 2 or any(degree not in (1, 2) for degree in degrees.values()):
        return None
    anchors = [cell for cell in endpoints if cell[0] == 0]
    if len(anchors) != 1:
        return None

    path = []
    previous = None
    current = anchors[0]
    while True:
        path.append(current)
        nxt = [
            cell
            for cell in _neighbors4(*current)
            if cell in cells and cell != previous
        ]
        if not nxt:
            break
        if len(nxt) != 1:
            return None
        previous, current = current, nxt[0]
    return path if len(path) == len(cells) else None


def _infer_roles(grid: np.ndarray) -> tuple[int, int, list[Coord], list[list[Coord]]]:
    """Infer background, obstacle color, ordered chain, and obstacle components uniquely."""
    bg = int(Counter(map(int, grid.ravel())).most_common(1)[0][0])
    foreground_colors = sorted(set(map(int, np.unique(grid))) - {bg})
    candidates = []
    for obstacle in foreground_colors:
        obstacle_comps = _solid_rectangles(grid == obstacle)
        if obstacle_comps is None:
            continue
        chain_mask = (grid != bg) & (grid != obstacle)
        path = _ordered_anchored_path(chain_mask)
        if path is None:
            continue
        chain_colors = {int(grid[r, c]) for r, c in path}
        if len(chain_colors) < 2:
            continue
        candidates.append((int(obstacle), path, obstacle_comps))

    assert len(candidates) == 1, "chain and obstacle roles must be structurally unique"
    obstacle, path, obstacle_comps = candidates[0]
    return bg, obstacle, path, obstacle_comps


def _component_lookup(
    shape: tuple[int, int], components: list[list[Coord]]
) -> tuple[dict[Coord, int], list[tuple[int, int, int, int]]]:
    lookup: dict[Coord, int] = {}
    boxes = []
    for i, comp in enumerate(components):
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        boxes.append((min(rs), min(cs), max(rs), max(cs)))
        for cell in comp:
            lookup[cell] = i
    return lookup, boxes


def _drop_coordinates(
    shape: tuple[int, int],
    anchor: Coord,
    obstacle_components: list[list[Coord]],
    length: int,
) -> tuple[list[Coord], list[str]]:
    """Lay a chain down, detouring over a hit rectangle toward its nearer side."""
    H, W = shape
    lookup, boxes = _component_lookup(shape, obstacle_components)
    obstacle_cells = set(lookup)
    route = [anchor]
    bypass_modes: list[str] = []

    while len(route) < length:
        r, c = route[-1]
        below = (r + 1, c)
        assert below[0] < H, "chain is too long to fit below its anchor"

        if below in lookup:
            r0, c0, _r1, c1 = boxes[lookup[below]]
            assert r + 1 == r0, "a falling chain must encounter an obstacle at its top"
            left_distance = c - c0
            right_distance = c1 - c
            if left_distance <= right_distance:
                step = -1
                target_c = c0 - 1
                mode = "tie" if left_distance == right_distance else "left"
            else:
                step = 1
                target_c = c1 + 1
                mode = "right"
            assert 0 <= target_c < W, "an obstacle must leave room around its chosen side"
            bypass_modes.append(mode)

            while c != target_c and len(route) < length:
                c += step
                cell = (r, c)
                assert cell not in obstacle_cells, "another obstacle blocks the top-edge detour"
                assert cell not in route, "rerouted chain must remain one pixel wide"
                route.append(cell)
        else:
            assert below not in route, "rerouted chain must not cross itself"
            route.append(below)

    return route, bypass_modes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rehang a curled multicolor chain from its top anchor, dropping it around solid obstacles while preserving length and color order.

    Input structure:
    - The most common color is background. One multicolor foreground object is a one-pixel-wide, 4-connected, branch-free path with exactly two endpoints and exactly one endpoint on the top edge.
    - A dedicated remaining color forms one or more separated solid rectangular obstacles below the anchor; removing that color uniquely exposes the anchored path.

    Rule:
    - Read colors by traversing the original chain from its top-edge endpoint, then erase all original chain cells while leaving every obstacle fixed.
    - Starting at the same anchor, place the colors in order. Prefer one cell downward; when that cell belongs to an obstacle, move horizontally across the row above its top toward the nearer side boundary (choose left on a tie), pass one cell beyond that boundary, and resume downward.
    - Stop after exactly the original number of chain cells has been placed.

    Key insight:
    - The curled input geometry only stores an ordered color string; the output geometry is determined independently by gravity and each encountered obstacle's bounds.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "grid must be a nonempty 2-D array"
    bg, obstacle, original_path, obstacle_comps = _infer_roles(g)
    sequence = [int(g[r, c]) for r, c in original_path]
    route, _ = _drop_coordinates(g.shape, original_path[0], obstacle_comps, len(sequence))

    out = g.copy()
    for r, c in original_path:
        out[r, c] = bg
    for (r, c), color in zip(route, sequence):
        assert int(out[r, c]) != obstacle, "the rerouted chain must not cover an obstacle"
        out[r, c] = color
    return out


# ------------------------------ generator ------------------------------

_GENERATION_COUNTER = 0
_MODE_OFFSET = random.randrange(3)
_MODES = ("left", "tie", "right")
_MAX_GENERATION_ATTEMPTS = 300


def _rectangle_for_mode(column: int, width: int, mode: str) -> tuple[int, int]:
    """Choose horizontal bounds containing column with the requested nearer side."""
    if mode == "tie":
        half = random.randint(1, min(3, (width - 3) // 2))
        return column - half, column + half
    near = random.randint(1, 2)
    far = near + random.randint(1, 3)
    if mode == "left":
        return column - near, column + far
    return column - far, column + near


def _separated(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = rect
    for a0, b0, a1, b1 in others:
        if not (r1 + 1 < a0 or a1 + 1 < r0 or c1 + 1 < b0 or b1 + 1 < c0):
            return False
    return True


def _make_obstacles(
    H: int, W: int, anchor_c: int, primary_mode: str, force_second: bool
) -> tuple[list[tuple[int, int, int, int]], int] | None:
    primary_top = random.randint(7, min(11, H - 13))
    primary_h = random.randint(2, 4)
    c0, c1 = _rectangle_for_mode(anchor_c, W, primary_mode)
    if c0 < 1 or c1 >= W - 1:
        return None
    rectangles = [(primary_top, c0, primary_top + primary_h - 1, c1)]

    first_exit = c0 - 1 if primary_mode in ("left", "tie") else c1 + 1
    if force_second:
        second_top = rectangles[0][2] + random.randint(4, 6)
        if second_top + 5 >= H:
            return None
        second_mode = random.choice(_MODES)
        s0, s1 = _rectangle_for_mode(first_exit, W, second_mode)
        second_h = random.randint(2, 4)
        second = (second_top, s0, second_top + second_h - 1, s1)
        if s0 < 1 or s1 >= W - 1 or not _separated(second, rectangles):
            return None
        rectangles.append(second)

    target_count = random.randint(max(2, len(rectangles)), 4)
    for _ in range(100):
        if len(rectangles) >= target_count:
            break
        rh = random.randint(2, 4)
        rw = random.randint(2, 6)
        r0 = random.randint(primary_top + 2, H - rh - 2)
        c0 = random.randint(1, W - rw - 1)
        rect = (r0, c0, r0 + rh - 1, c0 + rw - 1)
        if _separated(rect, rectangles):
            rectangles.append(rect)
    if len(rectangles) < target_count:
        return None
    return rectangles, primary_top


def _mask_from_rectangles(
    shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r0, c0, r1, c1 in rectangles:
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _full_drop_route(
    shape: tuple[int, int], anchor: Coord, components: list[list[Coord]]
) -> tuple[list[Coord], list[str]] | None:
    """Produce the complete route through the bottom row for generator planning."""
    H, _W = shape
    lookup, boxes = _component_lookup(shape, components)
    obstacle_cells = set(lookup)
    route = [anchor]
    modes = []
    while route[-1][0] < H - 1:
        r, c = route[-1]
        below = (r + 1, c)
        if below in lookup:
            r0, c0, _r1, c1 = boxes[lookup[below]]
            if r + 1 != r0:
                return None
            dl, dr = c - c0, c1 - c
            if dl <= dr:
                target = c0 - 1
                step = -1
                modes.append("tie" if dl == dr else "left")
            else:
                target = c1 + 1
                step = 1
                modes.append("right")
            if not 0 <= target < shape[1]:
                return None
            while c != target:
                c += step
                if (r, c) in obstacle_cells or (r, c) in route:
                    return None
                route.append((r, c))
        else:
            route.append(below)
    return route, modes


def _curled_path(length: int, anchor: Coord, max_row: int, W: int) -> list[Coord] | None:
    """Grow an induced random path in the upper band, anchored only at row zero."""
    if max_row < 4 or length < 10:
        return None
    first = (1, anchor[1])

    for _ in range(30):
        path = [anchor, first]
        used = set(path)
        node_budget = [6000]

        def search() -> bool:
            node_budget[0] -= 1
            if node_budget[0] <= 0:
                return False
            if len(path) == length:
                directions = [
                    (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
                    for i in range(len(path) - 1)
                ]
                turns = sum(a != b for a, b in zip(directions, directions[1:]))
                cols = [c for _, c in path]
                rows = [r for r, _ in path]
                return turns >= 5 and max(cols) - min(cols) >= 4 and max(rows) >= 4

            r, c = path[-1]
            candidates = []
            for nr, nc in _neighbors4(r, c):
                cell = (nr, nc)
                if not (1 <= nr <= max_row and 1 <= nc < W - 1):
                    continue
                if cell in used:
                    continue
                occupied_neighbors = sum(n in used for n in _neighbors4(nr, nc))
                if occupied_neighbors != 1:
                    continue
                candidates.append(cell)
            random.shuffle(candidates)

            if len(path) >= 3:
                pr, pc = path[-2]
                old_dir = (r - pr, c - pc)
                candidates.sort(
                    key=lambda q: (
                        (q[0] - r, q[1] - c) == old_dir,
                        random.random(),
                    )
                )
            for cell in candidates:
                path.append(cell)
                used.add(cell)
                if search():
                    return True
                used.remove(cell)
                path.pop()
            return False

        if search():
            return path
    return None


def _color_sequence(length: int, colors: list[int]) -> list[int]:
    """Build a visibly varied color string with short runs and every color represented."""
    sequence = list(colors)
    random.shuffle(sequence)
    while len(sequence) < length:
        choices = [color for color in colors if color != sequence[-1]]
        color = random.choice(choices)
        run = random.randint(1, 3)
        sequence.extend([color] * min(run, length - len(sequence)))
    if sequence == sequence[::-1]:
        sequence[-1] = next(color for color in colors if color != sequence[0])
    return sequence


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    bg: int,
    obstacle: int,
    input_path: list[Coord],
    route: list[Coord],
    sequence: list[int],
    primary_mode: str,
) -> bool:
    """Check puzzle semantics, anti-shortcuts, role uniqueness, and the solver oracle."""
    try:
        got_bg, got_obstacle, got_path, comps = _infer_roles(grid)
        if got_bg != bg or got_obstacle != obstacle or got_path != input_path:
            return False
        planned_route, modes = _drop_coordinates(grid.shape, got_path[0], comps, len(got_path))
        if planned_route != route or not modes or modes[0] != primary_mode:
            return False
        if route == input_path or not any(a[1] != b[1] for a, b in zip(route, route[1:])):
            return False
        if len(set(sequence)) < 3 or sequence == sequence[::-1]:
            return False
        if [int(expected[r, c]) for r, c in route] != sequence:
            return False
        if not np.array_equal(grid == obstacle, expected == obstacle):
            return False
        if Counter(sequence) != Counter(int(expected[r, c]) for r, c in route):
            return False
        if len(_components(grid == obstacle)) < 2:
            return False
        # At least two downward cells after the first bypass make both the bypass
        # and the subsequent return to gravity visibly necessary.
        first_horizontal = next(
            i for i, (a, b) in enumerate(zip(route, route[1:])) if a[0] == b[0]
        )
        if sum(route[i + 1][0] > route[i][0] for i in range(first_horizontal + 1, len(route) - 1)) < 2:
            return False
        return bool(np.array_equal(main(grid), expected))
    except (AssertionError, StopIteration, ValueError):
        return False


def _generation_attempt(primary_mode: str, force_second: bool) -> np.ndarray | None:
    H = random.randint(24, 30)
    W = random.randint(18, 30)
    anchor_c = random.randint(6, W - 7)
    obstacle_plan = _make_obstacles(H, W, anchor_c, primary_mode, force_second)
    if obstacle_plan is None:
        return None
    rectangles, primary_top = obstacle_plan
    obstacle_mask = _mask_from_rectangles((H, W), rectangles)
    obstacle_components = _components(obstacle_mask)

    full = _full_drop_route((H, W), (0, anchor_c), obstacle_components)
    if full is None:
        return None
    full_route, full_modes = full
    required_hits = 2 if force_second else 1
    if len(full_modes) < required_hits or full_modes[0] != primary_mode:
        return None

    intentional_bottom = rectangles[required_hits - 1][2]
    clear_indices = [i for i, (r, _c) in enumerate(full_route) if r >= intentional_bottom + 2]
    if not clear_indices:
        return None
    minimum_length = clear_indices[0] + 1
    maximum_length = min(len(full_route) - 1, 38)
    if minimum_length > maximum_length:
        return None
    length = random.randint(minimum_length, maximum_length)
    route = full_route[:length]

    input_path = _curled_path(length, (0, anchor_c), primary_top - 2, W)
    if input_path is None or input_path == route:
        return None

    palette = list(range(10))
    random.shuffle(palette)
    bg, obstacle = palette[:2]
    chain_colors = palette[2 : 2 + random.randint(3, 5)]
    sequence = _color_sequence(length, chain_colors)

    grid = np.full((H, W), bg, dtype=int)
    grid[obstacle_mask] = obstacle
    for (r, c), color in zip(input_path, sequence):
        if obstacle_mask[r, c]:
            return None
        grid[r, c] = color

    expected = grid.copy()
    for r, c in input_path:
        expected[r, c] = bg
    for (r, c), color in zip(route, sequence):
        if obstacle_mask[r, c]:
            return None
        expected[r, c] = color

    if int(Counter(map(int, grid.ravel())).most_common(1)[0][0]) != bg:
        return None
    if not _validate_generated(
        grid,
        expected,
        bg,
        obstacle,
        input_path,
        route,
        sequence,
        primary_mode,
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a diverse valid sample; each three calls cover left, tie, and right bypasses."""
    global _GENERATION_COUNTER
    mode = _MODES[(_MODE_OFFSET + _GENERATION_COUNTER) % len(_MODES)]
    force_second = (_GENERATION_COUNTER % 2 == 0)
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt(mode, force_second)
        if grid is not None:
            _GENERATION_COUNTER += 1
            return grid
    raise RuntimeError("failed to generate a valid hanging-chain input")
