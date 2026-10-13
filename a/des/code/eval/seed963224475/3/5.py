from __future__ import annotations

import random

import numpy as np


BG = 8
RED = 2
PALETTE = [0, 1, 3, 4, 5, 6, 7, 9]
DIRS = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

ARROW_PATTERNS = {
    "up": {(0, 1), (1, 0), (1, 2)},
    "down": {(0, 0), (0, 2), (1, 1)},
    "left": {(0, 1), (1, 0), (2, 1)},
    "right": {(0, 0), (1, 1), (2, 0)},
}

GRID_H_RANGE = (16, 22)
GRID_W_RANGE = (16, 22)
OBJECT_COUNT_CHOICES = (2, 3, 4)
OUTER_SIZE_RANGE = (4, 7)
MAX_GENERATE_ATTEMPTS = 300
MAX_PLACEMENT_ATTEMPTS = 250


def _dominant_color(grid: np.ndarray) -> int:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(np.argmax(counts))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, *, connectivity: int) -> list[list[tuple[int, int]]]:
    steps = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    height, width = mask.shape
    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and bool(mask[nr, nc])
                        and not bool(seen[nr, nc])
                    ):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _bbox(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return frozenset((r - r0, c - c0) for r, c in cells)


def _shift(cells: list[tuple[int, int]] | set[tuple[int, int]], dr: int, dc: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c + dc)) for r, c in cells]


def _expanded(
    cells: list[tuple[int, int]] | set[tuple[int, int]],
    shape: tuple[int, int],
    radius: int = 1,
) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if _in_bounds(shape, nr, nc):
                    out.add((int(nr), int(nc)))
    return out


def _container_pattern(direction: str, height: int, width: int) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    if direction in {"up", "down"} and (height < 2 or width < 3):
        return None
    if direction in {"left", "right"} and (height < 3 or width < 2):
        return None

    wall: set[tuple[int, int]] = set()
    interior: list[tuple[int, int]] = []

    if direction == "up":
        for r in range(height):
            wall.add((r, 0))
            wall.add((r, width - 1))
        for c in range(width):
            wall.add((height - 1, c))
        interior = [(r, c) for r in range(height - 1) for c in range(1, width - 1)]
    elif direction == "down":
        for r in range(height):
            wall.add((r, 0))
            wall.add((r, width - 1))
        for c in range(width):
            wall.add((0, c))
        interior = [(r, c) for r in range(1, height) for c in range(1, width - 1)]
    elif direction == "left":
        for c in range(width):
            wall.add((0, c))
            wall.add((height - 1, c))
        for r in range(height):
            wall.add((r, width - 1))
        interior = [(r, c) for r in range(1, height - 1) for c in range(width - 1)]
    else:
        for c in range(width):
            wall.add((0, c))
            wall.add((height - 1, c))
        for r in range(height):
            wall.add((r, 0))
        interior = [(r, c) for r in range(1, height - 1) for c in range(1, width)]

    if not interior:
        return None
    return wall, interior


def _recognize_container(cells: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(cells)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    cell_set = set(cells)

    for direction in DIRS:
        pat = _container_pattern(direction, height, width)
        if pat is None:
            continue
        wall_local, interior_local = pat
        wall_abs = set(_shift(wall_local, r0, c0))
        if wall_abs != cell_set:
            continue
        interior_abs = _shift(interior_local, r0, c0)
        return {
            "opening": direction,
            "bbox": (r0, c0, r1, c1),
            "wall_cells": cells,
            "interior": interior_abs,
            "mouth_width": len(_mouth_lanes(direction, (r0, c0, r1, c1))),
        }
    return None


def _mouth_lanes(direction: str, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if direction == "up":
        return [(int(r0 - 1), int(c)) for c in range(c0 + 1, c1)]
    if direction == "down":
        return [(int(r1 + 1), int(c)) for c in range(c0 + 1, c1)]
    if direction == "left":
        return [(int(r), int(c0 - 1)) for r in range(r0 + 1, r1)]
    return [(int(r), int(c1 + 1)) for r in range(r0 + 1, r1)]


def _pack_targets(cells: list[tuple[int, int]], count: int, gravity: str) -> list[tuple[int, int]]:
    if gravity == "up":
        ordered = sorted(cells, key=lambda p: (p[0], p[1]))
    elif gravity == "down":
        ordered = sorted(cells, key=lambda p: (-p[0], p[1]))
    elif gravity == "left":
        ordered = sorted(cells, key=lambda p: (p[1], p[0]))
    else:
        ordered = sorted(cells, key=lambda p: (-p[1], p[0]))
    assert 0 < count <= len(ordered)
    return ordered[:count]


def _leak_targets(
    bbox: tuple[int, int, int, int],
    direction: str,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    dr, dc = DIR_VECTORS[direction]
    lanes = _mouth_lanes(direction, bbox)
    active = [True] * len(lanes)
    out: list[tuple[int, int]] = []
    depth = 0
    while any(active):
        for idx, (sr, sc) in enumerate(lanes):
            if not active[idx]:
                continue
            rr = sr + dr * depth
            cc = sc + dc * depth
            if not _in_bounds(shape, rr, cc):
                active[idx] = False
                continue
            cell = (int(rr), int(cc))
            if cell in blocked:
                active[idx] = False
                continue
            out.append(cell)
        depth += 1
    return out


def _find_gravity(grid: np.ndarray) -> str:
    candidates: list[tuple[int, int, str]] = []
    for comp in _components(grid == RED, connectivity=8):
        if len(comp) != 3:
            continue
        norm = _normalize(comp)
        for direction, pattern in ARROW_PATTERNS.items():
            if norm == frozenset(pattern):
                _r0, c0, r1, _c1 = _bbox(comp)
                candidates.append((int(-r1), int(c0), direction))
                break
    assert candidates, "expected one red arrow near the lower-left corner"
    candidates.sort()
    return candidates[0][2]


def _extract_containers(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in {bg, RED}:
            continue
        for comp in _components(grid == color, connectivity=4):
            info = _recognize_container(comp)
            if info is None:
                continue
            interior = info["interior"]
            liquid_cells = [(r, c) for r, c in interior if int(grid[r, c]) != bg]
            if not liquid_cells:
                continue
            liquid_colors = {int(grid[r, c]) for r, c in liquid_cells}
            if color in liquid_colors or RED in liquid_colors:
                continue
            assert len(liquid_colors) == 1, "each container should contain one liquid color"
            liquid_color = liquid_colors.pop()
            for r, c in interior:
                assert int(grid[r, c]) in {bg, liquid_color}
            info["wall_color"] = color
            info["liquid_color"] = liquid_color
            info["liquid_cells"] = liquid_cells
            containers.append(info)
    assert containers, "expected at least one valid container"
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate liquid inside U/C containers under the gravity arrow in the lower-left corner.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    gravity = _find_gravity(g)
    containers = _extract_containers(g, bg)

    all_liquids = {
        (int(r), int(c))
        for container in containers
        for r, c in container["liquid_cells"]
    }
    blocked = {
        (int(r), int(c))
        for r, c in np.argwhere(g != bg)
        if (int(r), int(c)) not in all_liquids
    }

    out = g.copy()
    for r, c in all_liquids:
        out[r, c] = bg

    for container in sorted(containers, key=lambda item: tuple(item["bbox"])):
        liquid_cells = container["liquid_cells"]
        count = len(liquid_cells)
        if container["opening"] == gravity:
            targets = _leak_targets(container["bbox"], gravity, g.shape, blocked)
            assert len(targets) >= count, "leak path must have enough room"
            targets = targets[:count]
        else:
            targets = _pack_targets(container["interior"], count, gravity)
        for r, c in targets:
            assert (r, c) not in blocked
            blocked.add((int(r), int(c)))
            out[r, c] = int(container["liquid_color"])

    return out


def _arrow_cells(shape: tuple[int, int], direction: str) -> set[tuple[int, int]]:
    pattern = ARROW_PATTERNS[direction]
    ph = max(r for r, _ in pattern) + 1
    r0 = shape[0] - ph
    return {(int(r0 + r), int(c)) for r, c in pattern}


def _object_count(height: int, width: int) -> int:
    max_count = 4 if min(height, width) >= 20 else 3
    choices = [count for count in OBJECT_COUNT_CHOICES if count <= max_count]
    return random.choice(choices)


def _choose_branches(count: int, gravity: str) -> list[str]:
    leak_count = count // 2
    if count % 2 == 1 and random.random() < 0.5:
        leak_count += 1
    leak_count = max(1, min(count - 1, leak_count))
    directions = [gravity] * leak_count
    other_dirs = [direction for direction in DIRS if direction != gravity]
    directions.extend(random.choice(other_dirs) for _ in range(count - leak_count))
    random.shuffle(directions)
    return directions


def _choose_container_size(_direction: str) -> tuple[int, int]:
    return (
        random.randint(*OUTER_SIZE_RANGE),
        random.randint(*OUTER_SIZE_RANGE),
    )


def _corridor_set(corridor: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in corridor}


def _sample_positions(
    shape: tuple[int, int],
    direction: str,
    reserved: set[tuple[int, int]],
) -> dict[str, object] | None:
    height, width = _choose_container_size(direction)
    pat = _container_pattern(direction, height, width)
    if pat is None:
        return None
    wall_local, interior_local = pat
    grid_h, grid_w = shape
    max_r0 = grid_h - height - 1
    max_c0 = grid_w - width - 1
    if max_r0 < 1 or max_c0 < 1:
        return None

    for _ in range(MAX_PLACEMENT_ATTEMPTS):
        r0 = random.randint(1, max_r0)
        c0 = random.randint(1, max_c0)
        wall_abs = _shift(wall_local, r0, c0)
        interior_abs = _shift(interior_local, r0, c0)
        object_zone = set(wall_abs) | set(interior_abs)
        gap_zone = _expanded(object_zone, shape, radius=1)
        bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
        corridor = _leak_targets(bbox, direction, shape, set())
        if not corridor:
            continue
        if gap_zone & reserved:
            continue
        corridor_cells = _corridor_set(corridor)
        if corridor_cells & reserved:
            continue
        return {
            "opening": direction,
            "bbox": bbox,
            "wall_cells": wall_abs,
            "interior": interior_abs,
            "corridor": corridor,
            "reserve": gap_zone | corridor_cells,
        }
    return None


def _pick_liquid_count(area: int, capacity: int) -> int:
    upper = min(area - 1, capacity)
    assert upper >= 2
    lower = 2 if upper >= 2 else 1
    return random.randint(lower, upper)


def _sample_unsorted_liquid(interior: list[tuple[int, int]], count: int, gravity: str) -> list[tuple[int, int]]:
    settled = set(_pack_targets(interior, count, gravity))
    for _ in range(100):
        sample = random.sample(interior, count)
        if set(sample) != settled:
            return sample
    # Fallback: force one pixel away from the settled configuration.
    sample = list(settled)
    extras = [cell for cell in interior if cell not in settled]
    assert extras
    sample[-1] = random.choice(extras)
    return sample


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    gravity = random.choice(DIRS)
    container_count = _object_count(height, width)
    directions = _choose_branches(container_count, gravity)

    needed_colors = container_count * 2
    if needed_colors > len(PALETTE):
        return None
    color_pool = random.sample(PALETTE, needed_colors)
    wall_colors = color_pool[:container_count]
    liquid_colors = color_pool[container_count:]

    grid = np.full((height, width), BG, dtype=int)
    arrow = _arrow_cells(grid.shape, gravity)
    for r, c in arrow:
        grid[r, c] = RED
    reserved = _expanded(arrow, grid.shape, radius=1)

    container_specs: list[dict[str, object]] = []
    place_order = sorted(range(container_count), key=lambda idx: directions[idx] != gravity)
    for idx in place_order:
        spec = _sample_positions(grid.shape, directions[idx], reserved)
        if spec is None:
            return None
        spec["wall_color"] = wall_colors[idx]
        spec["liquid_color"] = liquid_colors[idx]
        reserved |= set(spec["reserve"])
        container_specs.append(spec)

    random.shuffle(container_specs)
    for spec in container_specs:
        for r, c in spec["wall_cells"]:
            grid[r, c] = int(spec["wall_color"])

        interior = spec["interior"]
        area = len(interior)
        if spec["opening"] == gravity:
            capacity = len(spec["corridor"])
            if min(area - 1, capacity) < 2:
                return None
            count = _pick_liquid_count(area, capacity)
            liquid_cells = random.sample(interior, count)
        else:
            if area < 3:
                return None
            count = _pick_liquid_count(area, area - 1)
            liquid_cells = _sample_unsorted_liquid(interior, count, gravity)

        for r, c in liquid_cells:
            grid[r, c] = int(spec["liquid_color"])

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-flow puzzle")
