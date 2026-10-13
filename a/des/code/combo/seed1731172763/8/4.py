from __future__ import annotations

import random
from collections import defaultdict, deque

import numpy as np


BG = 0
RECOLOR_CYCLE = (2, 7, 4, 3, 1)  # red, orange, yellow, green, blue
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ARRANGEMENT_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))

MIN_GRID = 12
MAX_GRID = 30
TARGET_COUNT_SMALL = (3, 5)
TARGET_COUNT_WRAP = 6
TARGET_SHAPE_AREA = (3, 7)
TARGET_SHAPE_SIDE = (2, 4)
DISTRACTOR_COUNT_RANGE = (1, 3)
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _shape_key(coords: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(coords)
    rel = sorted((int(r - r0), int(c - c0)) for r, c in coords)
    return tuple(rel)


def _principal_axis(centers: np.ndarray) -> np.ndarray:
    if len(centers) == 2:
        axis = centers[1] - centers[0]
    else:
        centered = centers - centers.mean(axis=0, keepdims=True)
        if np.allclose(centered, 0):
            axis = np.array([1.0, 0.0], dtype=float)
        else:
            _u, _s, vh = np.linalg.svd(centered, full_matrices=False)
            axis = vh[0]

    if np.allclose(axis, 0):
        span = centers.max(axis=0) - centers.min(axis=0)
        axis = np.array([1.0, 0.0], dtype=float) if span[0] >= span[1] else np.array([0.0, 1.0], dtype=float)

    axis = axis.astype(float)
    if abs(float(axis[1])) > abs(float(axis[0])):
        if axis[1] < 0:
            axis = -axis
    else:
        if axis[0] < 0:
            axis = -axis
    return axis


def _ordered_group(group: list[dict[str, object]]) -> list[dict[str, object]]:
    centers = np.asarray([obj["center"] for obj in group], dtype=float)
    axis = _principal_axis(centers)
    projections = centers @ axis
    order = sorted(
        range(len(group)),
        key=lambda i: (float(projections[i]), float(centers[i, 0]), float(centers[i, 1])),
    )
    return [group[i] for i in order]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each repeated-shape sprite group by its order along the group's main arrangement axis.

    Input structure:
    - Black background.
    - Several 4-connected foreground objects.
    - At least one shape appears multiple times with different original colors.

    Rule:
    - Group objects by identical shape after translation normalization.
    - For each repeated-shape group, sort members by bbox-center position along the
      group's main arrangement direction.
    - Recolor them using the cycle red, orange, yellow, green, blue, repeating as needed.
    - Objects whose shapes are not repeated keep their original colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    objects = []
    for coords in _components(g != BG):
        r0, c0, r1, c1 = _bbox(coords)
        colors = g[coords[:, 0], coords[:, 1]]
        objects.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "center": ((r0 + r1) / 2.0, (c0 + c1) / 2.0),
                "shape": _shape_key(coords),
                "color": int(colors[0]),
            }
        )

    groups: dict[tuple[tuple[int, int], ...], list[dict[str, object]]] = defaultdict(list)
    for obj in objects:
        groups[obj["shape"]].append(obj)

    for group in groups.values():
        if len(group) < 2:
            continue
        ordered = _ordered_group(group)
        for i, obj in enumerate(ordered):
            new_color = RECOLOR_CYCLE[i % len(RECOLOR_CYCLE)]
            coords = obj["coords"]
            out[coords[:, 0], coords[:, 1]] = new_color

    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _c in shape]
    cols = [c for _r, c in shape]
    return max(rows) + 1, max(cols) + 1


def _random_connected_shape(
    min_area: int = TARGET_SHAPE_AREA[0],
    max_area: int = TARGET_SHAPE_AREA[1],
    side_min: int = TARGET_SHAPE_SIDE[0],
    side_max: int = TARGET_SHAPE_SIDE[1],
) -> tuple[tuple[int, int], ...]:
    for _ in range(200):
        h = random.randint(side_min, side_max)
        w = random.randint(side_min, side_max)
        area_hi = min(max_area, h * w)
        if area_hi < min_area:
            continue
        area = random.randint(min_area, area_hi)

        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set()

        def add_frontier(cell: tuple[int, int]) -> None:
            rr, cc = cell
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))

        add_frontier(start)
        while int(mask.sum()) < area and frontier:
            cell = random.choice(tuple(frontier))
            frontier.remove(cell)
            mask[cell] = True
            add_frontier(cell)
        if int(mask.sum()) != area:
            continue

        rows, cols = np.where(mask)
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape == (1, 1):
            continue
        if int(cropped.sum()) == cropped.size and cropped.shape[0] >= 3 and cropped.shape[1] >= 3:
            continue
        rel = tuple(sorted((int(r), int(c)) for r, c in zip(*np.where(cropped))))
        return rel
    raise RuntimeError("failed to sample connected shape")


def _coords_at(shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        blocked[r0:r1, c0:c1] = True


def _can_place(blocked: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in coords:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _all_top_lefts(blocked: np.ndarray, shape: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    sh, sw = _shape_dims(shape)
    h, w = blocked.shape
    valid = []
    for top in range(0, h - sh + 1):
        for left in range(0, w - sw + 1):
            coords = _coords_at(shape, top, left)
            if _can_place(blocked, coords):
                valid.append((top, left))
    return valid


def _paint_object(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _step_bounds(shape: tuple[tuple[int, int], ...], direction: tuple[int, int]) -> tuple[int, int]:
    sh, sw = _shape_dims(shape)
    dr, dc = direction
    if dr == 0:
        min_step = sw + 1
    elif dc == 0:
        min_step = sh + 1
    else:
        min_step = max(sh, sw) + 1
    return min_step, min_step + 2


def _target_count() -> int:
    wrap = random.choice([False, True])
    if wrap:
        return TARGET_COUNT_WRAP
    return random.randint(*TARGET_COUNT_SMALL)


def _choose_canvas(
    shape: tuple[tuple[int, int], ...], count: int, direction: tuple[int, int], step: int
) -> tuple[int, int, int, int] | None:
    sh, sw = _shape_dims(shape)
    dr, dc = direction
    row_shift = (count - 1) * dr * step
    col_shift = (count - 1) * dc * step
    row_min = min(0, row_shift)
    row_max = max(0, row_shift)
    col_min = min(0, col_shift)
    col_max = max(0, col_shift)

    span_h = sh + (row_max - row_min)
    span_w = sw + (col_max - col_min)

    min_h = max(MIN_GRID, span_h + 2)
    min_w = max(MIN_GRID, span_w + 2)
    if min_h > MAX_GRID or min_w > MAX_GRID:
        return None

    h = random.randint(min_h, MAX_GRID)
    w = random.randint(min_w, MAX_GRID)
    top_low = 1 - row_min
    top_high = h - 1 - sh - row_max
    left_low = 1 - col_min
    left_high = w - 1 - sw - col_max
    if top_low > top_high or left_low > left_high:
        return None

    top = random.randint(top_low, top_high)
    left = random.randint(left_low, left_high)
    return h, w, top, left


def _build_expected(
    grid: np.ndarray, target_objects: list[list[tuple[int, int]]]
) -> np.ndarray:
    out = grid.copy()
    for i, coords in enumerate(target_objects):
        color = RECOLOR_CYCLE[i % len(RECOLOR_CYCLE)]
        for r, c in coords:
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    target_shape = _random_connected_shape()
    direction = random.choice(ARRANGEMENT_DIRECTIONS)
    count = _target_count()
    min_step, max_step = _step_bounds(target_shape, direction)
    step = random.randint(min_step, max_step)
    canvas = _choose_canvas(target_shape, count, direction, step)
    if canvas is None:
        return None
    h, w, start_r, start_c = canvas

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    target_colors = random.sample([1, 2, 3, 4, 5, 6, 7, 8, 9], count)
    target_objects: list[list[tuple[int, int]]] = []

    for i in range(count):
        top = start_r + i * direction[0] * step
        left = start_c + i * direction[1] * step
        coords = _coords_at(target_shape, top, left)
        if not _can_place(blocked, coords):
            return None
        _paint_object(grid, coords, target_colors[i])
        _mark_blocked(blocked, coords)
        target_objects.append(coords)

    if random.choice([False, True]):
        distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
        used_shapes = {target_shape}
        for _ in range(distractor_count):
            for _attempt in range(120):
                distractor_shape = _random_connected_shape(min_area=2, max_area=6, side_min=1, side_max=4)
                if distractor_shape in used_shapes:
                    continue
                valid_positions = _all_top_lefts(blocked, distractor_shape)
                if not valid_positions:
                    continue
                top, left = random.choice(valid_positions)
                coords = _coords_at(distractor_shape, top, left)
                color = random.randint(1, 9)
                _paint_object(grid, coords, color)
                _mark_blocked(blocked, coords)
                used_shapes.add(distractor_shape)
                break
            else:
                return None

    expected = _build_expected(grid, target_objects)

    objects = _components(grid != BG)
    grouped = defaultdict(int)
    for coords in objects:
        grouped[_shape_key(coords)] += 1
    if grouped[target_shape] != count:
        return None
    if any(freq > 1 for shape, freq in grouped.items() if shape != target_shape):
        return None

    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
