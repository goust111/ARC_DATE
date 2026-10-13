import random
from collections import deque

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ORIENTATIONS = [
    {"missing_offset": (0, 0), "dir": (-1, -1)},
    {"missing_offset": (0, 1), "dir": (-1, 1)},
    {"missing_offset": (1, 0), "dir": (1, -1)},
    {"missing_offset": (1, 1), "dir": (1, 1)},
]

GRID_H_RANGE = (12, 24)
GRID_W_RANGE = (12, 24)
L_COUNT_RANGE = (2, 3)
DISTRACTOR_COUNT_RANGE = (1, 2)
GEN_RETRY_LIMIT = 800
PLACEMENT_RETRY_LIMIT = 120


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors8(cell: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = cell
    return {(r + dr, c + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)}


def _components_same_color(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            visited[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr, nc = cr + dr, cc + dc
                    if not _in_bounds(grid.shape, nr, nc) or visited[nr, nc]:
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append({"color": color, "coords": coords})
    return components


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], dict[tuple[int, int], int]]:
    bg = _get_bg(grid)
    l_shapes = []
    dots = {}
    for comp in _components_same_color(grid, bg):
        color = comp["color"]
        coords = comp["coords"]
        if coords.shape[0] == 1:
            dots[(int(coords[0, 0]), int(coords[0, 1]))] = color
            continue
        r0, c0, r1, c1 = _bbox(coords)
        if coords.shape[0] != 3 or r1 - r0 != 1 or c1 - c0 != 1:
            raise AssertionError("foreground objects must be 3-cell L shapes or singleton dots")
        present = {(int(r), int(c)) for r, c in coords}
        box_cells = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        missing = list(box_cells - present)
        if len(missing) != 1:
            raise AssertionError("each L must have exactly one missing corner")
        mr, mc = missing[0]
        direction = (-1 if mr == r0 else 1, -1 if mc == c0 else 1)
        l_shapes.append(
            {
                "color": color,
                "present": present,
                "box": box_cells,
                "missing": (mr, mc),
                "dir": direction,
            }
        )
    if not l_shapes:
        raise AssertionError("expected at least one L shape")
    l_shapes.sort(key=lambda spec: (spec["missing"][0], spec["missing"][1], spec["color"]))
    return bg, l_shapes, dots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each corner-missing L and let its missing corner point to the nearest diagonal target dot.

    Input structure:
    - The background is the dominant color, typically black.
    - Foreground objects are monochrome 3-cell L shapes, each occupying three cells of a 2x2 box.
    - Additional singleton pixels are scattered target or distractor dots.

    Rule:
    - For each L, fill its missing 2x2 corner with the L's own color.
    - The missing corner encodes a diagonal direction; scan outward from that corner along the same diagonal.
    - If the ray meets one or more dots, use the nearest dot as the target: paint every cell from one step beyond the filled corner up to the cell before the target with the target's original color, then recolor the target dot to the L's color.
    - If the ray meets no dot, only fill the missing corner. Dots on no L ray stay unchanged.

    Key insight:
    - Each absent corner is both a completion clue and a hidden arrow, and the visible color flow is reciprocal: the line takes the dot's color while the dot takes the L's color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    bg, l_shapes, dots = _parse_scene(g)
    out = g.copy()
    claimed = set()

    for spec in l_shapes:
        l_color = int(spec["color"])
        mr, mc = spec["missing"]
        dr, dc = spec["dir"]
        assert int(g[mr, mc]) == bg
        out[mr, mc] = l_color

        r, c = mr + dr, mc + dc
        target = None
        while _in_bounds(g.shape, r, c):
            if (r, c) in dots:
                target = (r, c)
                break
            r += dr
            c += dc

        if target is None:
            continue
        if target in claimed:
            raise AssertionError("each target dot must be claimed by at most one L")
        target_color = int(dots[target])

        r, c = mr + dr, mc + dc
        while (r, c) != target:
            assert int(g[r, c]) == bg
            out[r, c] = target_color
            r += dr
            c += dc
        out[target] = l_color
        claimed.add(target)

    return out


def _ray_from_missing(shape: tuple[int, int], missing: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = missing[0] + dr, missing[1] + dc
    cells = []
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _box_cells(top: int, left: int) -> set[tuple[int, int]]:
    return {(top + dr, left + dc) for dr in (0, 1) for dc in (0, 1)}


def _box_moat(shape: tuple[int, int], box_cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    moat = set()
    for r, c in box_cells:
        for nr, nc in _neighbors8((r, c)):
            if _in_bounds(shape, nr, nc):
                moat.add((nr, nc))
    return moat


def _sample_l_specs(shape: tuple[int, int], count: int) -> list[dict] | None:
    h, w = shape
    specs = []
    reserved_boxes = set()
    occupied = set()
    ray_union = set()
    for _ in range(count):
        placed = False
        for _ in range(PLACEMENT_RETRY_LIMIT):
            top = random.randint(1, h - 3)
            left = random.randint(1, w - 3)
            orient = random.choice(ORIENTATIONS)
            box = _box_cells(top, left)
            if _box_moat(shape, box) & occupied:
                continue
            missing = (top + orient["missing_offset"][0], left + orient["missing_offset"][1])
            ray = _ray_from_missing(shape, missing, orient["dir"])
            if box & ray_union:
                continue
            if any(cell in reserved_boxes for cell in ray):
                continue
            if any(cell in ray_union for cell in ray):
                continue

            present = box - {missing}
            specs.append(
                {
                    "present": present,
                    "box": box,
                    "missing": missing,
                    "dir": orient["dir"],
                    "ray": ray,
                }
            )
            occupied |= present
            reserved_boxes |= box
            ray_union |= set(ray)
            placed = True
            break
        if not placed:
            return None
    return specs


def _choose_target_steps(l_specs: list[dict]) -> tuple[set[int], dict[int, int]] | None:
    targetable = [i for i, spec in enumerate(l_specs) if len(spec["ray"]) >= 2]
    if not targetable:
        return None

    max_target_count = len(targetable)
    target_count = random.randint(1, max_target_count)
    if len(l_specs) > 1 and target_count == len(l_specs) and random.random() < 0.7:
        target_count -= 1
    target_indexes = set(random.sample(targetable, target_count))
    target_steps = {}
    for idx in target_indexes:
        step = random.randint(2, len(l_specs[idx]["ray"]))
        target_steps[idx] = step
    return target_indexes, target_steps


def _choose_far_dot(l_specs: list[dict], target_steps: dict[int, int]) -> tuple[int, int] | None:
    if random.random() >= 0.45:
        return None
    candidates = []
    for idx, step in target_steps.items():
        max_step = len(l_specs[idx]["ray"])
        if max_step >= step + 2:
            candidates.append(idx)
    if not candidates:
        return None
    idx = random.choice(candidates)
    far_step = random.randint(target_steps[idx] + 2, len(l_specs[idx]["ray"]))
    return idx, far_step


def _nonray_candidates(
    shape: tuple[int, int],
    forbidden: set[tuple[int, int]],
    ray_union: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    cells = []
    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden or (r, c) in ray_union:
                continue
            cells.append((r, c))
    random.shuffle(cells)
    return cells


def _try_generate() -> np.ndarray | None:
    shape = (random.randint(*GRID_H_RANGE), random.randint(*GRID_W_RANGE))
    l_count = random.randint(*L_COUNT_RANGE)
    l_specs = _sample_l_specs(shape, l_count)
    if l_specs is None:
        return None

    picked = _choose_target_steps(l_specs)
    if picked is None:
        return None
    target_indexes, target_steps = picked

    ray_union = set()
    for spec in l_specs:
        ray_union |= set(spec["ray"])

    box_cells = set().union(*(spec["box"] for spec in l_specs))
    forbidden_for_dots = set()
    for cell in box_cells:
        forbidden_for_dots |= {nbr for nbr in _neighbors8(cell) if _in_bounds(shape, *nbr)}

    target_cells = {}
    for idx, step in target_steps.items():
        target = l_specs[idx]["ray"][step - 1]
        if target in forbidden_for_dots:
            return None
        target_cells[idx] = target
        forbidden_for_dots |= {nbr for nbr in _neighbors8(target) if _in_bounds(shape, *nbr)}

    far_info = _choose_far_dot(l_specs, target_steps)
    far_cells = {}
    if far_info is not None:
        idx, far_step = far_info
        far_cell = l_specs[idx]["ray"][far_step - 1]
        if far_cell in forbidden_for_dots:
            return None
        far_cells[idx] = far_cell
        forbidden_for_dots |= {nbr for nbr in _neighbors8(far_cell) if _in_bounds(shape, *nbr)}

    distractor_count = random.randint(*DISTRACTOR_COUNT_RANGE)
    distractor_cells = []
    for _ in range(distractor_count):
        candidates = _nonray_candidates(shape, forbidden_for_dots, ray_union)
        if not candidates:
            return None
        cell = candidates[0]
        distractor_cells.append(cell)
        forbidden_for_dots |= {nbr for nbr in _neighbors8(cell) if _in_bounds(shape, *nbr)}

    total_dot_count = len(target_cells) + len(far_cells) + len(distractor_cells)
    if l_count + total_dot_count > len(PALETTE):
        return None

    colors = random.sample(PALETTE, l_count + total_dot_count)
    l_colors = colors[:l_count]
    dot_colors = iter(colors[l_count:])

    grid = np.full(shape, BG, dtype=int)
    for idx, spec in enumerate(l_specs):
        for r, c in spec["present"]:
            grid[r, c] = l_colors[idx]
        spec["color"] = l_colors[idx]

    for idx, cell in target_cells.items():
        grid[cell] = next(dot_colors)
        l_specs[idx]["target"] = cell

    for idx, cell in far_cells.items():
        grid[cell] = next(dot_colors)

    for cell in distractor_cells:
        grid[cell] = next(dot_colors)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if not any(idx not in target_indexes for idx in range(l_count)) and random.random() < 0.55:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _try_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 4")
