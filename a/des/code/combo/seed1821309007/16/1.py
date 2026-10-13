import math
from collections import deque

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_RNG = np.random.default_rng()
MAX_ATTEMPTS = 400


def _component_from(grid: np.ndarray, start: tuple[int, int], seen: np.ndarray) -> dict:
    h, w = grid.shape
    color = int(grid[start])
    q = deque([start])
    seen[start] = True
    coords = []
    while q:
        r, c = q.popleft()
        coords.append((r, c))
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                seen[nr, nc] = True
                q.append((nr, nc))

    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    return {
        "color": color,
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "center": (center_r, center_c),
    }


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    seen = np.zeros(g.shape, dtype=bool)
    objs = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] == BG or seen[r, c]:
                continue
            objs.append(_component_from(g, (r, c), seen))
    return objs


def _layer_of_point(point: tuple[float, float], shape: tuple[int, int]) -> int:
    center_r = shape[0] // 2
    center_c = shape[1] // 2
    cheb = max(abs(point[0] - center_r), abs(point[1] - center_c))
    return max(0, int(math.ceil((cheb - 1.0) / 2.0)))


def _clockwise_key(point: tuple[float, float], shape: tuple[int, int]) -> tuple[float, float, float]:
    center_r = shape[0] // 2
    center_c = shape[1] // 2
    dr = point[0] - center_r
    dc = point[1] - center_c
    angle = math.atan2(dr, dc)
    clockwise = (angle + 2.0 * math.pi) % (2.0 * math.pi)
    radius = dr * dr + dc * dc
    return clockwise, radius, point[0] + point[1] * 1e-3


def main(grid: np.ndarray) -> np.ndarray:
    """Group objects by square distance band from the center and rotate colors within each band."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _find_objects(g)

    by_layer: dict[int, list[dict]] = {}
    for obj in objects:
        layer = _layer_of_point(obj["center"], g.shape)
        by_layer.setdefault(layer, []).append(obj)

    for layer_objects in by_layer.values():
        if len(layer_objects) <= 1:
            continue
        ordered = sorted(layer_objects, key=lambda obj: _clockwise_key(obj["center"], g.shape))
        rotated_colors = [ordered[(i + 1) % len(ordered)]["color"] for i in range(len(ordered))]
        for obj, new_color in zip(ordered, rotated_colors):
            for r, c in obj["coords"]:
                out[r, c] = new_color

    return out


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _random_shape() -> tuple[tuple[int, int], ...]:
    target = int(_RNG.integers(2, 6))
    cells = {(0, 0)}
    frontier = [(0, 0)]
    while len(cells) < target:
        base_r, base_c = frontier[int(_RNG.integers(len(frontier)))]
        dr, dc = DIRS4[int(_RNG.integers(4))]
        nxt = (base_r + dr, base_c + dc)
        if nxt in cells:
            continue
        cells.add(nxt)
        frontier.append(nxt)
    return _normalize_shape(cells)


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shift_shape(shape: tuple[tuple[int, int], ...], top: int, left: int) -> list[tuple[int, int]]:
    return [(top + r, left + c) for r, c in shape]


def _paintable(occupied: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[rr, cc]:
                    return False
    return True


def _mark_occupied(occupied: np.ndarray, coords: list[tuple[int, int]]) -> None:
    for r, c in coords:
        occupied[r, c] = True


def _candidate_positions(
    shape: tuple[tuple[int, int], ...],
    grid_shape: tuple[int, int],
    layer: int,
    occupied: np.ndarray,
) -> list[tuple[int, int, float]]:
    h, w = grid_shape
    sh, sw = _shape_dims(shape)
    out = []
    for top in range(h - sh + 1):
        for left in range(w - sw + 1):
            coords = _shift_shape(shape, top, left)
            if not _paintable(occupied, coords):
                continue
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            center = ((min(rows) + max(rows)) / 2.0, (min(cols) + max(cols)) / 2.0)
            if _layer_of_point(center, grid_shape) != layer:
                continue
            angle = _clockwise_key(center, grid_shape)[0]
            out.append((top, left, angle))
    return out


def _choose_layer_counts(max_layer: int) -> list[tuple[int, int]]:
    layer_count = int(_RNG.integers(2, min(5, max_layer + 2)))
    layers = sorted(_RNG.choice(np.arange(max_layer + 1), size=layer_count, replace=False).tolist())
    counts = []
    must_have_singleton = bool(_RNG.integers(2))
    singleton_used = False
    multi_used = False
    for layer in layers:
        if must_have_singleton and not singleton_used:
            count = 1
            singleton_used = True
        else:
            count = int(_RNG.integers(1, 5))
        if count >= 2:
            multi_used = True
        counts.append([layer, count])
    if not multi_used:
        idx = int(_RNG.integers(len(counts)))
        counts[idx][1] = int(_RNG.integers(2, 5))
    return [(layer, count) for layer, count in counts]


def _assign_colors(layer_specs: list[tuple[int, int]]) -> list[list[int]]:
    color_orders = []
    for _, count in layer_specs:
        if count == 1:
            color_orders.append([int(_RNG.choice(COLOR_POOL))])
            continue
        perm = _RNG.permutation(COLOR_POOL).tolist()
        colors = perm[:count]
        if len(set(colors)) == 1:
            colors[0] = perm[count]
        color_orders.append([int(c) for c in colors])
    return color_orders


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.choice(np.arange(15, 26, 2)))
    w = int(_RNG.choice(np.arange(15, 26, 2)))
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    max_layer = _layer_of_point((0.0, 0.0), grid.shape)
    if max_layer < 1:
        return None
    layer_specs = _choose_layer_counts(max_layer)
    color_specs = _assign_colors(layer_specs)

    placements = []
    for (layer, count), colors in zip(layer_specs, color_specs):
        chosen_angles = []
        for color in colors:
            shape = _random_shape()
            candidates = _candidate_positions(shape, grid.shape, layer, occupied)
            if not candidates:
                return None
            _RNG.shuffle(candidates)
            selected = None
            for top, left, angle in candidates:
                if count >= 3 and any(abs(((angle - prev + math.pi) % (2.0 * math.pi)) - math.pi) < 0.35 for prev in chosen_angles):
                    continue
                selected = (top, left, angle)
                break
            if selected is None:
                selected = candidates[0]
            top, left, angle = selected
            coords = _shift_shape(shape, top, left)
            _mark_occupied(occupied, coords)
            chosen_angles.append(angle)
            placements.append((coords, color))

    for coords, color in placements:
        for r, c in coords:
            grid[r, c] = color

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if len(_find_objects(grid)) != len(placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
