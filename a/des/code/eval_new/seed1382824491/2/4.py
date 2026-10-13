from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
DIRS = {
    UP: (-1, 0),
    RIGHT: (0, 1),
    DOWN: (1, 0),
    LEFT: (0, -1),
}

# A 2x2 tower side midpoint lies between cells, so we launch from the
# adjacent tower corner in a rotation-preserving cyclic order.
LAUNCH_ANCHOR_OFFSETS = {
    UP: (0, 0),
    RIGHT: (0, 1),
    DOWN: (1, 1),
    LEFT: (1, 0),
}

NEXT_COLOR = {
    RED: GREEN,
    GREEN: BLUE,
    BLUE: RED,
}

CYCLE_COLORS = (RED, GREEN, BLUE)
NONCYCLE_COLORS = (MAGENTA, ORANGE, CYAN)
TOWER_COLOR_POOL = list(CYCLE_COLORS + NONCYCLE_COLORS)

MODULE_SIZE = 8
GEN_ATTEMPTS = 500
EXTRA_OBJECTS_MAX = 2

TOWER_SHAPE = ((0, 0), (0, 1), (1, 0), (1, 1))

MODULE_TEMPLATES = {
    "split": {
        "towers": [("cycle", (3, 1))],
        "prisms": [(3, 6)],
        "absorbers": [],
    },
    "chain": {
        "towers": [("cycle", (3, 1))],
        "prisms": [(3, 6), (1, 6)],
        "absorbers": [(6, 6)],
    },
    "nonsplit": {
        "towers": [("noncycle", (3, 1))],
        "prisms": [(3, 6)],
        "absorbers": [],
    },
    "intersect": {
        "towers": [("any", (3, 1)), ("any", (1, 5))],
        "prisms": [],
        "absorbers": [],
    },
    "absorb": {
        "towers": [("any", (3, 1))],
        "prisms": [],
        "absorbers": [(3, 6)],
    },
    "block": {
        "towers": [("any", (3, 1)), ("any", (3, 6))],
        "prisms": [],
        "absorbers": [],
    },
}

RECIPES = [
    ("split", "intersect"),
    ("nonsplit", "block"),
    ("chain", "absorb"),
    ("split", "nonsplit", "intersect"),
    ("chain", "block", "absorb"),
    ("split", "nonsplit", "absorb"),
]

RECIPE_REQUIREMENTS = {
    ("split", "intersect"): {"split_events": 1, "yellow_count": 1},
    ("nonsplit", "block"): {"prism_stops_no_split": 1, "tower_stops": 1},
    ("chain", "absorb"): {"max_depth": 2, "absorber_stops": 1},
    ("split", "nonsplit", "intersect"): {
        "split_events": 1,
        "prism_stops_no_split": 1,
        "yellow_count": 1,
    },
    ("chain", "block", "absorb"): {
        "max_depth": 2,
        "tower_stops": 1,
        "absorber_stops": 1,
    },
    ("split", "nonsplit", "absorb"): {
        "split_events": 1,
        "prism_stops_no_split": 1,
        "absorber_stops": 1,
    },
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _detect_towers(grid: np.ndarray) -> list[tuple[int, int, int]]:
    towers: list[tuple[int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color in (BLACK, GRAY, MAROON):
            continue
        for comp in _components(grid == color):
            coords = np.array(comp, dtype=int)
            assert coords.shape == (4, 2), "every tower must be a 2x2 monochrome component"
            r0, c0 = map(int, coords.min(axis=0))
            r1, c1 = map(int, coords.max(axis=0))
            assert r1 == r0 + 1 and c1 == c0 + 1, "tower bbox must be exactly 2x2"
            assert np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color), "tower must be solid"
            towers.append((int(color), r0, c0))
    assert towers, "expected at least one signal tower"
    towers.sort(key=lambda item: (item[1], item[2], item[0]))
    return towers


def _perpendicular_dirs(direction: int) -> tuple[int, int]:
    if direction in (UP, DOWN):
        return LEFT, RIGHT
    return UP, DOWN


def _simulate(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h <= 30 and w <= 30, "ARC grids must be at most 30x30"

    towers = _detect_towers(g)
    tower_cells = {
        (r + dr, c + dc)
        for _color, r, c in towers
        for dr, dc in TOWER_SHAPE
    }

    visit_count = np.zeros((h, w), dtype=int)
    visit_color = np.full((h, w), -1, dtype=int)

    split_events = 0
    prism_stops_no_split = 0
    absorber_stops = 0
    tower_stops = 0
    boundary_stops = 0
    max_depth = 0
    loop_breaks = 0

    queue: deque[tuple[int, int, int, int, int]] = deque()
    for color, top, left in towers:
        for direction in (UP, RIGHT, DOWN, LEFT):
            ar = top + LAUNCH_ANCHOR_OFFSETS[direction][0]
            ac = left + LAUNCH_ANCHOR_OFFSETS[direction][1]
            queue.append((ar, ac, direction, int(color), 0))

    seen_states: set[tuple[int, int, int, int]] = set()
    while queue:
        r, c, direction, color, depth = queue.popleft()
        state = (r, c, direction, color)
        if state in seen_states:
            loop_breaks += 1
            continue
        seen_states.add(state)
        max_depth = max(max_depth, int(depth))

        dr, dc = DIRS[direction]
        while True:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                boundary_stops += 1
                break
            if (nr, nc) in tower_cells:
                tower_stops += 1
                break

            cell = int(g[nr, nc])
            if cell == MAROON:
                absorber_stops += 1
                break
            if cell == GRAY:
                next_color = NEXT_COLOR.get(int(color))
                if next_color is None:
                    prism_stops_no_split += 1
                else:
                    split_events += 1
                    next_depth = depth + 1
                    max_depth = max(max_depth, next_depth)
                    for child_dir in _perpendicular_dirs(direction):
                        queue.append((nr, nc, child_dir, next_color, next_depth))
                break

            visit_count[nr, nc] += 1
            if visit_count[nr, nc] == 1:
                visit_color[nr, nc] = int(color)
            r, c = nr, nc

    out = g.copy()
    single = visit_count == 1
    multi = visit_count >= 2
    out[single] = visit_color[single]
    out[multi] = YELLOW

    if not return_meta:
        return out

    meta = {
        "visit_count": visit_count,
        "split_events": int(split_events),
        "prism_stops_no_split": int(prism_stops_no_split),
        "absorber_stops": int(absorber_stops),
        "tower_stops": int(tower_stops),
        "boundary_stops": int(boundary_stops),
        "max_depth": int(max_depth),
        "yellow_count": int(multi.sum()),
        "ray_cell_count": int((visit_count > 0).sum()),
        "tower_count": int(len(towers)),
        "prism_count": int((g == GRAY).sum()),
        "absorber_count": int((g == MAROON).sum()),
        "loop_breaks": int(loop_breaks),
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """Draw every orthogonal signal ray, including prism splits and yellow crossings.

    Input structure:
    - The background is black.
    - Signal towers are solid 2x2 monochrome blocks in non-black, non-grey, non-maroon colors.
    - Grey single pixels are prisms, and maroon single pixels are absorbers.

    Rule:
    - Each tower emits one thin ray upward, rightward, downward, and leftward from the four cyclic side-midpoint launch ports of its 2x2 block.
    - Rays advance through background cells only, stop before leaving the grid, before entering any tower, and before entering any maroon absorber.
    - When a ray would enter a grey prism, the incoming ray stops and the prism stays grey; if the ray color is red, green, or blue, the prism launches two perpendicular child rays colored by the fixed cycle red->green->blue->red, and those child rays obey the same rules recursively.
    - Any background cell visited by two or more ray paths becomes yellow; otherwise a visited cell keeps the single visiting ray color.

    Key insight:
    - The puzzle combines path tracing with a hidden color state machine: 2x2 towers define the four launch directions, prisms turn one incoming ray into two perpendicular rays, only the red/green/blue colors keep propagating through repeated splits, and all overlaps collapse to yellow.
    """
    return _simulate(grid)


def _rot90(cell: tuple[int, int], rotation: int, size: int = MODULE_SIZE) -> tuple[int, int]:
    r, c = cell
    for _ in range(rotation % 4):
        r, c = c, size - 1 - r
    return r, c


def _choose_color(role: str, available: list[int]) -> int | None:
    if role == "cycle":
        pool = [c for c in available if c in CYCLE_COLORS]
    elif role == "noncycle":
        pool = [c for c in available if c in NONCYCLE_COLORS]
    else:
        pool = list(available)
    if not pool:
        return None
    return int(random.choice(pool))


def _place_module(
    grid: np.ndarray,
    module_name: str,
    anchor: tuple[int, int],
    rotation: int,
    available_colors: list[int],
) -> list[int] | None:
    template = MODULE_TEMPLATES[module_name]
    used_colors: list[int] = []
    remaining = list(available_colors)
    paint_ops: list[tuple[int, int, int]] = []

    for role, tower_top_left in template["towers"]:
        color = _choose_color(role, remaining)
        if color is None:
            return None
        remaining.remove(color)
        used_colors.append(color)
        tr, tc = tower_top_left
        for dr, dc in TOWER_SHAPE:
            rr, cc = _rot90((tr + dr, tc + dc), rotation)
            paint_ops.append((anchor[0] + rr, anchor[1] + cc, int(color)))

    for cell in template["prisms"]:
        rr, cc = _rot90(cell, rotation)
        paint_ops.append((anchor[0] + rr, anchor[1] + cc, GRAY))
    for cell in template["absorbers"]:
        rr, cc = _rot90(cell, rotation)
        paint_ops.append((anchor[0] + rr, anchor[1] + cc, MAROON))

    h, w = grid.shape
    for r, c, _color in paint_ops:
        if not (0 <= r < h and 0 <= c < w):
            return None
        if int(grid[r, c]) != BLACK:
            return None

    for r, c, color in paint_ops:
        grid[r, c] = int(color)
    return used_colors


def _recipe_anchors(module_count: int) -> tuple[int, int, list[tuple[int, int]]]:
    if module_count == 2:
        h = random.randint(18, 24)
        w = random.randint(18, 24)
        rows = [1, h - MODULE_SIZE - 1]
        cols = [1, w - MODULE_SIZE - 1]
        if random.random() < 0.5:
            anchors = [(rows[0], cols[0]), (rows[1], cols[1])]
        else:
            anchors = [(rows[0], cols[1]), (rows[1], cols[0])]
        return h, w, anchors

    h = random.randint(26, 30)
    w = random.randint(26, 30)
    rows = [1, (h - MODULE_SIZE) // 2, h - MODULE_SIZE - 1]
    cols = [1, (w - MODULE_SIZE) // 2, w - MODULE_SIZE - 1]
    perm = random.sample(range(3), 3)
    anchors = [(rows[i], cols[perm[i]]) for i in range(3)]
    return h, w, anchors


def _quiet_empty_cells(grid: np.ndarray, forbidden_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BLACK or forbidden_mask[r, c]:
                continue
            r0 = max(0, r - 1)
            r1 = min(h, r + 2)
            c0 = max(0, c - 1)
            c1 = min(w, c + 2)
            if np.all(grid[r0:r1, c0:c1] == BLACK):
                cells.append((r, c))
    random.shuffle(cells)
    return cells


def _add_extras(grid: np.ndarray, visit_count: np.ndarray) -> None:
    forbidden = visit_count > 0
    quiet = _quiet_empty_cells(grid, forbidden)
    prism_budget = random.randint(0, EXTRA_OBJECTS_MAX)
    absorber_budget = random.randint(0, EXTRA_OBJECTS_MAX)

    idx = 0
    while prism_budget > 0 and idx < len(quiet):
        r, c = quiet[idx]
        idx += 1
        r0 = max(0, r - 1)
        r1 = min(grid.shape[0], r + 2)
        c0 = max(0, c - 1)
        c1 = min(grid.shape[1], c + 2)
        if int(grid[r, c]) == BLACK and np.all(grid[r0:r1, c0:c1] == BLACK):
            grid[r, c] = GRAY
            prism_budget -= 1

    while absorber_budget > 0 and idx < len(quiet):
        r, c = quiet[idx]
        idx += 1
        r0 = max(0, r - 1)
        r1 = min(grid.shape[0], r + 2)
        c0 = max(0, c - 1)
        c1 = min(grid.shape[1], c + 2)
        if int(grid[r, c]) == BLACK and np.all(grid[r0:r1, c0:c1] == BLACK):
            grid[r, c] = MAROON
            absorber_budget -= 1


def _meets_requirements(meta: dict[str, int], recipe: tuple[str, ...]) -> bool:
    if meta["loop_breaks"] != 0:
        return False
    if meta["ray_cell_count"] < 8:
        return False
    if meta["ray_cell_count"] > meta["visit_count"].size // 2:
        return False
    for key, value in RECIPE_REQUIREMENTS[recipe].items():
        if int(meta[key]) < int(value):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    recipe = tuple(random.choice(RECIPES))
    h, w, anchors = _recipe_anchors(len(recipe))
    grid = np.full((h, w), BLACK, dtype=int)

    available_colors = list(TOWER_COLOR_POOL)
    random.shuffle(available_colors)
    module_names = sorted(
        recipe,
        key=lambda name: sum(role != "any" for role, _ in MODULE_TEMPLATES[name]["towers"]),
        reverse=True,
    )
    random.shuffle(anchors)

    for module_name, anchor in zip(module_names, anchors):
        rotation = random.randrange(4)
        used = _place_module(grid, module_name, anchor, rotation, available_colors)
        if used is None:
            return None
        for color in used:
            available_colors.remove(color)

    _, first_meta = _simulate(grid, return_meta=True)
    _add_extras(grid, first_meta["visit_count"])
    _, meta = _simulate(grid, return_meta=True)

    if not _meets_requirements(meta, recipe):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-and-prism puzzle")
