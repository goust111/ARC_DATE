from __future__ import annotations

import random
from collections import Counter

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRECTION_CYCLE = ("up", "down", "left", "right")

GEN_ATTEMPTS = 400
GRID_H_RANGE = (14, 24)
GRID_W_RANGE = (14, 24)
MAX_CONTAINERS = 4
VERTICAL_WIDTH_OPTIONS = (5, 7, 9)
VERTICAL_HEIGHT_RANGE = (4, 8)
HORIZONTAL_HEIGHT_OPTIONS = (5, 7, 9)
HORIZONTAL_WIDTH_RANGE = (4, 8)
MARKER_COUNT_MAX = 6

_GEN_CALLS = 0


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_container(
    coords: list[tuple[int, int]],
) -> tuple[str, tuple[int, int, int, int]] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    local = {(r - r0, c - c0) for r, c in coords}
    if any(r not in (0, h - 1) and c not in (0, w - 1) for r, c in local):
        return None

    top_full = all((0, c) in local for c in range(w))
    bottom_full = all((h - 1, c) in local for c in range(w))
    left_full = all((r, 0) in local for r in range(h))
    right_full = all((r, w - 1) in local for r in range(h))

    top_gap = all((0, c) not in local for c in range(1, w - 1))
    bottom_gap = all((h - 1, c) not in local for c in range(1, w - 1))
    left_gap = all((r, 0) not in local for r in range(1, h - 1))
    right_gap = all((r, w - 1) not in local for r in range(1, h - 1))

    matches: list[str] = []
    if bottom_full and left_full and right_full and top_gap:
        matches.append("up")
    if top_full and left_full and right_full and bottom_gap:
        matches.append("down")
    if top_full and bottom_full and right_full and left_gap:
        matches.append("left")
    if top_full and bottom_full and left_full and right_gap:
        matches.append("right")
    if len(matches) != 1:
        return None

    direction = matches[0]
    if direction in ("up", "down"):
        if (w - 2) <= 0 or (w - 2) % 2 == 0:
            return None
        expected_size = 2 * h + w - 2
    else:
        if (h - 2) <= 0 or (h - 2) % 2 == 0:
            return None
        expected_size = h + 2 * w - 2
    if len(local) != expected_size:
        return None
    return direction, (r0, c0, r1, c1)


def _opening_start_and_delta(
    bbox: tuple[int, int, int, int], direction: str
) -> tuple[tuple[int, int], tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return (r0, (c0 + c1) // 2), (-1, 0)
    if direction == "down":
        return (r1, (c0 + c1) // 2), (1, 0)
    if direction == "left":
        return ((r0 + r1) // 2, c0), (0, -1)
    return ((r0 + r1) // 2, c1), (0, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the count of marker pixels inside each open container into an outward ray.

    Each container is a monochrome 4-connected U/C shape that occupies exactly three
    sides of its bounding box. The open side determines the ray direction, the number
    of interior marker pixels determines the ray length, and the marker color determines
    the ray color. Containers and markers remain unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)

    out = g.copy()
    containers: list[tuple[int, str, tuple[int, int, int, int]]] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_mask(g == color):
            classified = _classify_container(coords)
            if classified is None:
                continue
            direction, bbox = classified
            containers.append((color, direction, bbox))

    containers.sort(key=lambda item: (item[2][0], item[2][1], item[1], item[0]))

    for color, direction, bbox in containers:
        r0, c0, r1, c1 = bbox
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        marker_mask = (interior != bg) & (interior != color)
        if not marker_mask.any():
            continue
        marker_colors = np.unique(interior[marker_mask])
        assert marker_colors.size == 1, "each container interior must use one marker color"
        marker_color = int(marker_colors[0])
        ray_length = int(marker_mask.sum())

        (sr, sc), (dr, dc) = _opening_start_and_delta(bbox, direction)
        for step in range(ray_length):
            rr = sr + dr * step
            cc = sc + dc * step
            assert 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1], "ray must stay in bounds"
            out[rr, cc] = marker_color

    return out


def _sample_geometry(direction: str) -> tuple[int, int]:
    if direction in ("up", "down"):
        return random.randint(*VERTICAL_HEIGHT_RANGE), random.choice(VERTICAL_WIDTH_OPTIONS)
    return random.choice(HORIZONTAL_HEIGHT_OPTIONS), random.randint(*HORIZONTAL_WIDTH_RANGE)


def _container_border_local(h: int, w: int, direction: str) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    if direction == "up":
        for r in range(h):
            cells.add((r, 0))
            cells.add((r, w - 1))
        for c in range(w):
            cells.add((h - 1, c))
    elif direction == "down":
        for r in range(h):
            cells.add((r, 0))
            cells.add((r, w - 1))
        for c in range(w):
            cells.add((0, c))
    elif direction == "left":
        for c in range(w):
            cells.add((0, c))
            cells.add((h - 1, c))
        for r in range(h):
            cells.add((r, w - 1))
    else:
        for c in range(w):
            cells.add((0, c))
            cells.add((h - 1, c))
        for r in range(h):
            cells.add((r, 0))
    return cells


def _checkerboard_candidates(h: int, w: int, parity: int) -> list[tuple[int, int]]:
    cells = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r + c) % 2 == parity
    ]
    return cells


def _ray_cells_from_bbox(
    bbox: tuple[int, int, int, int], direction: str, length: int
) -> list[tuple[int, int]]:
    (sr, sc), (dr, dc) = _opening_start_and_delta(bbox, direction)
    return [(sr + dr * step, sc + dc * step) for step in range(length)]


def _expand_cells(
    cells: set[tuple[int, int]] | list[tuple[int, int]],
    h: int,
    w: int,
    radius: int = 1,
) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(max(0, r - radius), min(h, r + radius + 1)):
            for cc in range(max(0, c - radius), min(w, c + radius + 1)):
                expanded.add((rr, cc))
    return expanded


def _sample_directions(call_index: int, n_containers: int) -> list[str]:
    required = DIRECTION_CYCLE[call_index % len(DIRECTION_CYCLE)]
    dirs = [required]
    pool = list(DIRECTION_CYCLE)
    random.shuffle(pool)
    for direction in pool:
        if direction not in dirs:
            dirs.append(direction)
        if len(dirs) == n_containers:
            break
    while len(dirs) < n_containers:
        dirs.append(random.choice(DIRECTION_CYCLE))
    random.shuffle(dirs)
    return dirs


def _marker_count_weights(limit: int) -> list[int]:
    weights = []
    for n in range(1, limit + 1):
        weights.append(3 if n in (2, 3, 4) else 1)
    return weights


def _try_place_container(
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    direction: str,
    container_color: int,
    marker_color: int,
) -> dict[str, object] | None:
    h_grid, w_grid = grid.shape
    for _ in range(120):
        h, w = _sample_geometry(direction)
        if h > h_grid or w > w_grid:
            continue

        positions = [(r0, c0) for r0 in range(h_grid - h + 1) for c0 in range(w_grid - w + 1)]
        random.shuffle(positions)
        border_local = _container_border_local(h, w, direction)

        for r0, c0 in positions:
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            bbox_cells = {(r, c) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}

            border_abs = {(r0 + r, c0 + c) for r, c in border_local}
            if reserved & bbox_cells:
                continue

            if direction == "up":
                ray_cap = r0 + 1
            elif direction == "down":
                ray_cap = h_grid - (r0 + h - 1)
            elif direction == "left":
                ray_cap = c0 + 1
            else:
                ray_cap = w_grid - (c0 + w - 1)

            parity = random.randrange(2)
            candidates = _checkerboard_candidates(h, w, parity)
            if not candidates:
                candidates = _checkerboard_candidates(h, w, 1 - parity)
            max_markers = min(len(candidates), ray_cap, MARKER_COUNT_MAX)
            if max_markers < 1:
                continue

            marker_count = random.choices(
                list(range(1, max_markers + 1)),
                weights=_marker_count_weights(max_markers),
                k=1,
            )[0]
            marker_local = random.sample(candidates, marker_count)
            marker_abs = {(r0 + r, c0 + c) for r, c in marker_local}
            ray_cells = _ray_cells_from_bbox(bbox, direction, marker_count)

            ray_clearance = _expand_cells(ray_cells, h_grid, w_grid, radius=1)
            bbox_clearance = _expand_cells(bbox_cells, h_grid, w_grid, radius=1)
            clearance = bbox_clearance | ray_clearance
            if reserved & clearance:
                continue
            if any(cell in border_abs or cell in marker_abs for cell in ray_cells[1:]):
                continue

            for r, c in border_abs:
                grid[r, c] = container_color
            for r, c in marker_abs:
                grid[r, c] = marker_color

            reserved |= clearance
            return {
                "direction": direction,
                "bbox": bbox,
                "ray": ray_cells,
                "marker_color": marker_color,
                "container_color": container_color,
                "marker_count": marker_count,
            }
    return None


def _expected_output(grid: np.ndarray, placements: list[dict[str, object]]) -> np.ndarray:
    out = grid.copy()
    for placement in placements:
        color = int(placement["marker_color"])
        for r, c in placement["ray"]:
            out[r, c] = color
    return out


def _count_container_like_objects(grid: np.ndarray) -> int:
    bg = _get_bg(grid)
    total = 0
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_mask(grid == color):
            if _classify_container(coords) is not None:
                total += 1
    return total


def _gen_attempt(call_index: int) -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), random.randrange(10), dtype=int)
    bg = int(grid[0, 0])

    max_count = min(MAX_CONTAINERS, max(1, (h * w) // 95))
    n_containers = random.randint(1, max_count)
    if 1 + 2 * n_containers > 10:
        return None

    palette = random.sample(range(10), 1 + 2 * n_containers)
    bg = palette[0]
    container_colors = palette[1 : 1 + n_containers]
    marker_colors = palette[1 + n_containers :]
    grid[:, :] = bg

    directions = _sample_directions(call_index, n_containers)
    reserved: set[tuple[int, int]] = set()
    placements: list[dict[str, object]] = []

    for direction, container_color, marker_color in zip(directions, container_colors, marker_colors):
        placement = _try_place_container(grid, reserved, direction, container_color, marker_color)
        if placement is None:
            return None
        placements.append(placement)

    if _get_bg(grid) != bg:
        return None
    if _count_container_like_objects(grid) != n_containers:
        return None

    total_markers = sum(int(placement["marker_count"]) for placement in placements)
    if total_markers < 2:
        return None

    expected = _expected_output(grid, placements)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    changed = int(np.sum(actual != grid))
    if changed != total_markers:
        return None

    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALLS
    call_index = _GEN_CALLS
    _GEN_CALLS += 1
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(call_index)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
