from __future__ import annotations

import random

import numpy as np


ORTHOGONAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400
GRID_SIDE_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 4)
RULER_LENGTH_RANGE = (2, 6)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    r0, c0, _, _ = _bbox(coords)
    return sorted((r - r0, c - c0) for r, c in coords)


def _degree_map(coords: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    coord_set = set(coords)
    return {
        (r, c): sum((nr, nc) in coord_set for nr, nc in _neighbors4(r, c))
        for r, c in coords
    }


def _is_connected(coords: list[tuple[int, int]]) -> bool:
    if not coords:
        return False
    coord_set = set(coords)
    stack = [coords[0]]
    seen = {coords[0]}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in coord_set and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(coord_set)


def _find_monochrome_components(
    grid: np.ndarray, bg: int
) -> list[dict[str, int | list[tuple[int, int]] | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = sorted(coords)
            comps.append({"color": color, "coords": coords, "bbox": _bbox(coords)})
    return comps


def _is_ruler_component(coords: list[tuple[int, int]]) -> bool:
    if len(coords) < 2:
        return False
    rows = {r for r, _ in coords}
    cols = {c for _, c in coords}
    if len(rows) == 1:
        ordered = sorted(c for _, c in coords)
        return ordered == list(range(ordered[0], ordered[-1] + 1))
    if len(cols) == 1:
        ordered = sorted(r for r, _ in coords)
        return ordered == list(range(ordered[0], ordered[-1] + 1))
    return False


def _tail_info(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]]:
    degree = _degree_map(coords)
    tails = [cell for cell, deg in degree.items() if deg == 1]
    assert len(tails) == 1, "each object must have exactly one degree-1 tail pixel"

    tail = tails[0]
    coord_set = set(coords)
    neighbors = [(nr, nc) for nr, nc in _neighbors4(*tail) if (nr, nc) in coord_set]
    assert len(neighbors) == 1, "tail pixel must touch the body once"
    body_neighbor = neighbors[0]
    direction = (tail[0] - body_neighbor[0], tail[1] - body_neighbor[1])
    assert direction in ORTHOGONAL_DIRS, "tail direction must be orthogonal"

    r0, c0, r1, c1 = _bbox(coords)
    if direction == (-1, 0):
        assert tail[0] == r0
    elif direction == (1, 0):
        assert tail[0] == r1
    elif direction == (0, -1):
        assert tail[1] == c0
    else:
        assert tail[1] == c1
    return tail, direction


def _project_component(
    coords: list[tuple[int, int]],
    tail: tuple[int, int],
    direction: tuple[int, int],
    distance: int,
) -> list[tuple[int, int]]:
    tail_r, tail_c = tail
    new_tail = (tail_r + distance * direction[0], tail_c + distance * direction[1])
    projected = []
    for r, c in coords:
        dr = r - tail_r
        dc = c - tail_c
        if direction[0] == 0:
            mirrored = (dr, -dc)
        else:
            mirrored = (-dr, dc)
        projected.append((new_tail[0] + mirrored[0], new_tail[1] + mirrored[1]))
    return projected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project a mirrored copy of each tailed object outward by the shared ruler length.

    Input structure:
    - The dominant color is the background.
    - Several monochrome 4-connected foreground objects are present, and each object has exactly one degree-1 tail pixel on a bbox edge.
    - One additional monochrome component is a unique horizontal or vertical straight ruler bar whose color is different from every object color.

    Rule:
    - Measure the ruler bar length D.
    - For each non-ruler object, locate its unique tail pixel and the direction in which that tail sticks out from the body.
    - Place a mirrored copy of the whole object so that the copy's tail lands D cells farther along that same direction.
    - Use a left-right flip for left/right tails, and an up-down flip for up/down tails.
    - Keep the original objects and the ruler unchanged; any projected copy pixels that fall outside the grid are clipped.

    Key insight:
    - The ruler does not move anything directly; it encodes one shared projection distance, while each object's own tail determines both the mirror axis and the outward placement direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape

    bg = _mode_color(g)
    components = _find_monochrome_components(g, bg)
    assert components, "expected foreground components"

    ruler_candidates = [comp for comp in components if _is_ruler_component(comp["coords"])]
    assert len(ruler_candidates) == 1, "expected exactly one ruler bar"
    ruler = ruler_candidates[0]
    distance = len(ruler["coords"])

    objects = [comp for comp in components if comp is not ruler]
    assert objects, "expected at least one non-ruler object"

    out = g.copy()
    original_bg = g == bg
    ordered_objects = sorted(objects, key=lambda comp: tuple(comp["bbox"]))

    for comp in ordered_objects:
        coords = comp["coords"]
        color = int(comp["color"])
        tail, direction = _tail_info(coords)
        projected = _project_component(coords, tail, direction, distance)
        for rr, cc in projected:
            if 0 <= rr < h and 0 <= cc < w and original_bg[rr, cc]:
                out[rr, cc] = color

    return out


def _sample_body() -> list[tuple[int, int]] | None:
    for _ in range(200):
        canvas_h = random.randint(4, 7)
        canvas_w = random.randint(4, 7)
        mask = np.zeros((canvas_h, canvas_w), dtype=bool)
        rect_count = random.choice((1, 2, 2, 3))
        for _rect in range(rect_count):
            rh = random.randint(2, min(4, canvas_h))
            rw = random.randint(2, min(4, canvas_w))
            r0 = random.randint(0, canvas_h - rh)
            c0 = random.randint(0, canvas_w - rw)
            mask[r0 : r0 + rh, c0 : c0 + rw] = True

        coords = [tuple(map(int, cell)) for cell in np.argwhere(mask)]
        if not (4 <= len(coords) <= 14):
            continue
        coords = _normalize(coords)
        if not _is_connected(coords):
            continue
        if min(_degree_map(coords).values()) < 2:
            continue
        return coords
    return None


def _sample_object_shape(
    direction: tuple[int, int]
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    for _ in range(300):
        body = _sample_body()
        if body is None:
            continue

        r0, c0, r1, c1 = _bbox(body)
        if direction == (-1, 0):
            candidates = [(r, c) for r, c in body if r == r0]
        elif direction == (1, 0):
            candidates = [(r, c) for r, c in body if r == r1]
        elif direction == (0, -1):
            candidates = [(r, c) for r, c in body if c == c0]
        else:
            candidates = [(r, c) for r, c in body if c == c1]

        random.shuffle(candidates)
        body_set = set(body)
        for attach_r, attach_c in candidates:
            tail = (attach_r + direction[0], attach_c + direction[1])
            if tail in body_set:
                continue
            coords = sorted(body + [tail])
            tails = [cell for cell, deg in _degree_map(coords).items() if deg == 1]
            if len(tails) != 1 or tails[0] != tail:
                continue
            normalized = _normalize(coords)
            shift_r = min(r for r, _ in coords)
            shift_c = min(c for _, c in coords)
            tail_local = (tail[0] - shift_r, tail[1] - shift_c)
            return normalized, tail_local
    raise RuntimeError("failed to sample a valid tailed object")


def _can_place(
    cells: list[tuple[int, int]], blocked: np.ndarray
) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _translate(
    coords: list[tuple[int, int]], top: int, left: int
) -> list[tuple[int, int]]:
    return [(r + top, c + left) for r, c in coords]


def _sample_directions(count: int) -> list[tuple[int, int]]:
    dirs = ORTHOGONAL_DIRS[:]
    random.shuffle(dirs)
    picked = dirs[: min(count, len(dirs))]
    while len(picked) < count:
        picked.append(random.choice(ORTHOGONAL_DIRS))
    random.shuffle(picked)
    return picked


def _place_ruler(
    grid: np.ndarray, blocked: np.ndarray, color: int, distance: int
) -> None:
    h, w = grid.shape
    for _ in range(200):
        horizontal = bool(random.getrandbits(1))
        if horizontal:
            row = random.randint(1, h - 2)
            start = random.randint(1, w - distance - 1)
            cells = [(row, start + offset) for offset in range(distance)]
        else:
            col = random.randint(1, w - 2)
            start = random.randint(1, h - distance - 1)
            cells = [(start + offset, col) for offset in range(distance)]
        if not _can_place(cells, blocked):
            continue
        for r, c in cells:
            grid[r, c] = color
        _mark_blocked(blocked, cells)
        return
    raise RuntimeError("failed to place ruler")


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    distance = random.randint(*RULER_LENGTH_RANGE)
    min_side = max(GRID_SIDE_RANGE[0], 10 + 2 * object_count + distance)
    if min_side > GRID_SIDE_RANGE[1]:
        return None

    h = random.randint(min_side, GRID_SIDE_RANGE[1])
    w = random.randint(min_side, GRID_SIDE_RANGE[1])
    palette = random.sample(range(10), object_count + 2)
    bg = palette[0]
    ruler_color = palette[1]
    object_colors = palette[2:]

    grid = np.full((h, w), bg, dtype=int)
    blocked_output = np.zeros((h, w), dtype=bool)
    _place_ruler(grid, blocked_output, ruler_color, distance)

    directions = _sample_directions(object_count)
    for color, direction in zip(object_colors, directions):
        coords_local, tail_local = _sample_object_shape(direction)
        _, _, r1, c1 = _bbox(coords_local)
        obj_h = r1 + 1
        obj_w = c1 + 1

        placed = False
        for _ in range(250):
            top = random.randint(0, h - obj_h)
            left = random.randint(0, w - obj_w)
            coords_abs = _translate(coords_local, top, left)
            tail_abs = (tail_local[0] + top, tail_local[1] + left)
            projected = _project_component(coords_abs, tail_abs, direction, distance)

            if not _can_place(coords_abs, blocked_output):
                continue
            if not all(0 <= r < h and 0 <= c < w for r, c in projected):
                continue

            temp_blocked = blocked_output.copy()
            _mark_blocked(temp_blocked, coords_abs)
            if not _can_place(projected, temp_blocked):
                continue

            for r, c in coords_abs:
                grid[r, c] = color
            blocked_output = temp_blocked
            _mark_blocked(blocked_output, projected)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-tail projection input")
