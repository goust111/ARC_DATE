import random
from collections import deque

import numpy as np


BACKGROUND = 0
DRIVER_COLOR = 4
MARK_COLOR = 9
BASE_COLORS = (1, 2, 3, 5, 6, 7, 8)
MAX_GENERATION_ATTEMPTS = 300

CARDINALS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}
TOOTH_OFFSETS = {
    "up": (-2, 0),
    "down": (2, 0),
    "left": (0, -2),
    "right": (0, 2),
}
GEAR_OFFSETS = {
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
} | set(TOOTH_OFFSETS.values())
NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LATTICE_SHAPES = ((3, 3), (3, 4), (4, 3), (4, 4))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                rr, cc = queue.popleft()
                comp.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _rotate_offset(offset: tuple[int, int], direction: int) -> tuple[int, int]:
    dr, dc = offset
    if direction > 0:
        return (dc, -dr)
    if direction < 0:
        return (-dc, dr)
    return offset


def _extract_gear(grid: np.ndarray, coords: list[tuple[int, int]], gear_id: int) -> dict:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    assert len(coords) == 13
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    assert r1 - r0 == 4 and c1 - c0 == 4

    center = (r0 + 2, c0 + 2)
    rel_to_value: dict[tuple[int, int], int] = {}
    for r, c in coords:
        rel = (r - center[0], c - center[1])
        rel_to_value[rel] = int(grid[r, c])
    assert set(rel_to_value) == GEAR_OFFSETS

    tooth_dirs = [name for name, off in TOOTH_OFFSETS.items() if rel_to_value[off] == MARK_COLOR]
    assert len(tooth_dirs) == 1
    marked_dir = tooth_dirs[0]

    driver = rel_to_value[(0, 0)] == DRIVER_COLOR
    base_colors = {
        value
        for value in rel_to_value.values()
        if value not in (BACKGROUND, DRIVER_COLOR, MARK_COLOR)
    }
    assert len(base_colors) == 1
    base_color = next(iter(base_colors))
    assert base_color not in (BACKGROUND, DRIVER_COLOR, MARK_COLOR)

    for off, value in rel_to_value.items():
        if off == TOOTH_OFFSETS[marked_dir]:
            assert value == MARK_COLOR
        elif off == (0, 0) and driver:
            assert value == DRIVER_COLOR
        else:
            assert value == base_color

    return {
        "id": gear_id,
        "coords": coords,
        "center": center,
        "pixels": rel_to_value,
        "marked_dir": marked_dir,
        "driver": driver,
        "base_color": base_color,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each gear by propagating clockwise or counterclockwise turns through the meshing graph.

    Input structure:
    - The background is black.
    - Each foreground object is a 13-pixel gear: a solid 3x3 block plus one tooth cell on each side, all in one base color except for one maroon tooth that marks orientation.
    - Exactly one gear has a yellow center cell inside its 3x3 block; that gear is the driver.
    - Two gears mesh when opposing tooth cells face each other with exactly one black cell between them.

    Rule:
    - Find every gear and identify the unique driver gear whose center is yellow.
    - Build the gear-to-gear meshing graph from tooth pairs separated by one background cell.
    - Rotate the driver 90 degrees clockwise, then alternate rotation direction across each meshing edge so neighbors rotate 90 degrees counterclockwise, their neighbors clockwise, and so on.
    - Gears not connected to the driver keep their original orientation, and the output is formed by repainting every gear in its rotated state.

    Key insight:
    - The puzzle is solved by separating object recognition from graph reasoning: first recover each gear center and marked tooth, then propagate turn parity along the meshing links.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    components = _find_components(g != BACKGROUND)
    assert components
    gears = [_extract_gear(g, comp, gear_id) for gear_id, comp in enumerate(components)]

    driver_ids = [gear["id"] for gear in gears if gear["driver"]]
    assert len(driver_ids) == 1
    driver_id = driver_ids[0]

    tooth_lookup: dict[tuple[int, int], tuple[int, str]] = {}
    for gear in gears:
        cr, cc = gear["center"]
        for name, (dr, dc) in TOOTH_OFFSETS.items():
            pos = (cr + dr, cc + dc)
            assert pos not in tooth_lookup
            tooth_lookup[pos] = (gear["id"], name)

    adjacency = {gear["id"]: set() for gear in gears}
    for pos, (gear_id, side) in tooth_lookup.items():
        step_r, step_c = CARDINALS[side]
        gap = (pos[0] + step_r, pos[1] + step_c)
        partner = (pos[0] + 2 * step_r, pos[1] + 2 * step_c)
        if not _in_bounds(g.shape, gap[0], gap[1]) or not _in_bounds(g.shape, partner[0], partner[1]):
            continue
        if int(g[gap]) != BACKGROUND:
            continue
        hit = tooth_lookup.get(partner)
        if hit is None or hit[0] == gear_id:
            continue
        other_id, other_side = hit
        assert other_side == OPPOSITE[side]
        adjacency[gear_id].add(other_id)
        adjacency[other_id].add(gear_id)

    rotations = {driver_id: 1}
    queue = deque([driver_id])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            expected = -rotations[current]
            if neighbor in rotations:
                assert rotations[neighbor] == expected
                continue
            rotations[neighbor] = expected
            queue.append(neighbor)

    out = g.copy()
    for gear in gears:
        for r, c in gear["coords"]:
            out[r, c] = BACKGROUND

    for gear in gears:
        direction = rotations.get(gear["id"], 0)
        cr, cc = gear["center"]
        for offset, value in gear["pixels"].items():
            dr, dc = _rotate_offset(offset, direction)
            out[cr + dr, cc + dc] = value

    return out


def _lattice_neighbors(cell: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = cell
    result = []
    for dr, dc in NEIGHBORS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append((nr, nc))
    return result


def _grow_tree(size: int, rows: int, cols: int, rng: random.Random) -> set[tuple[int, int]] | None:
    start = (rng.randrange(rows), rng.randrange(cols))
    nodes = {start}
    while len(nodes) < size:
        candidates = []
        for cell in nodes:
            for neighbor in _lattice_neighbors(cell, rows, cols):
                if neighbor in nodes:
                    continue
                occupied_neighbors = sum(
                    adjacent in nodes for adjacent in _lattice_neighbors(neighbor, rows, cols)
                )
                if occupied_neighbors == 1:
                    candidates.append(neighbor)
        if not candidates:
            return None
        nodes.add(rng.choice(candidates))
    return nodes


def _choose_isolates(
    occupied: set[tuple[int, int]],
    count: int,
    rows: int,
    cols: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    chosen: list[tuple[int, int]] = []
    blocked = set(occupied)
    blocked_neighbors = {
        neighbor
        for cell in occupied
        for neighbor in _lattice_neighbors(cell, rows, cols)
    }
    candidates = [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in blocked and (r, c) not in blocked_neighbors
    ]
    rng.shuffle(candidates)
    for cell in candidates:
        if any(neighbor in chosen for neighbor in _lattice_neighbors(cell, rows, cols)):
            continue
        chosen.append(cell)
        if len(chosen) == count:
            break
    return chosen


def _split_extra(extra: int, rng: random.Random) -> tuple[int, int]:
    left = rng.randint(1, extra - 1)
    return left, extra - left


def _paint_gear(
    grid: np.ndarray,
    center: tuple[int, int],
    color: int,
    marked_dir: str,
    driver: bool,
) -> None:
    cr, cc = center
    for dr, dc in GEAR_OFFSETS:
        grid[cr + dr, cc + dc] = color
    tooth_dr, tooth_dc = TOOTH_OFFSETS[marked_dir]
    grid[cr + tooth_dr, cc + tooth_dc] = MARK_COLOR
    if driver:
        grid[cr, cc] = DRIVER_COLOR


def _build_attempt(rng: random.Random) -> np.ndarray | None:
    lattice_rows, lattice_cols = rng.choice(LATTICE_SHAPES)
    connected_size = rng.choice((1, 2, 3, 4, 5))
    connected_size = min(connected_size, lattice_rows * lattice_cols, len(BASE_COLORS))
    tree = _grow_tree(connected_size, lattice_rows, lattice_cols, rng)
    if tree is None:
        return None

    max_isolates = min(2, len(BASE_COLORS) - len(tree))
    feasible_isolates = []
    for isolate_count in range(max_isolates + 1):
        if len(tree) + isolate_count < 2:
            continue
        isolate_cells = _choose_isolates(tree, isolate_count, lattice_rows, lattice_cols, rng)
        if len(isolate_cells) == isolate_count:
            feasible_isolates.append((isolate_count, isolate_cells))
    if not feasible_isolates:
        return None

    isolate_count, isolates = rng.choice(feasible_isolates)
    del isolate_count
    gears = sorted(tree) + sorted(isolates)
    driver_cell = rng.choice(sorted(tree))

    min_r = min(r for r, _ in gears)
    max_r = max(r for r, _ in gears)
    min_c = min(c for _, c in gears)
    max_c = max(c for _, c in gears)
    span_h = 5 + 6 * (max_r - min_r)
    span_w = 5 + 6 * (max_c - min_c)
    extra_h = rng.randint(2, 30 - span_h)
    extra_w = rng.randint(2, 30 - span_w)
    top_margin, bottom_margin = _split_extra(extra_h, rng)
    left_margin, right_margin = _split_extra(extra_w, rng)
    del bottom_margin, right_margin

    height = span_h + extra_h
    width = span_w + extra_w
    grid = np.full((height, width), BACKGROUND, dtype=int)

    colors = rng.sample(BASE_COLORS, len(gears))
    gear_to_color = {cell: colors[idx] for idx, cell in enumerate(gears)}
    gear_to_marker = {cell: rng.choice(tuple(TOOTH_OFFSETS)) for cell in gears}

    for cell in gears:
        lattice_r, lattice_c = cell
        center = (
            top_margin + 2 + 6 * (lattice_r - min_r),
            left_margin + 2 + 6 * (lattice_c - min_c),
        )
        _paint_gear(
            grid,
            center,
            gear_to_color[cell],
            gear_to_marker[cell],
            driver=(cell == driver_cell),
        )

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = random.Random()
        grid = _build_attempt(rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if out.shape != grid.shape:
            continue
        return grid
    raise RuntimeError("failed to generate a valid gear-rotation input")
