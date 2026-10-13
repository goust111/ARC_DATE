from __future__ import annotations

import random

import numpy as np


BG = 0
OBSTACLE = 5
MARKER_TO_SIDE = {
    2: "up",
    3: "down",
    4: "left",
    7: "right",
}
SIDE_TO_MARKER = {side: color for color, side in MARKER_TO_SIDE.items()}
MARKER_COLORS = set(MARKER_TO_SIDE)
FLEX_COLORS = [1, 6, 8, 9]

GRID_HEIGHT_RANGE = (15, 30)
GRID_WIDTH_RANGE = (15, 30)
RECT_COUNT_OPTIONS = (2, 3)
RECT_OUTER_HEIGHT_RANGE = (5, 8)
RECT_OUTER_WIDTH_RANGE = (5, 8)
PLACEMENT_TRIES = 160
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(comp)
    return parts


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    cells = {(r0, c) for c in range(c0, c1 + 1)}
    cells.update((r1, c) for c in range(c0, c1 + 1))
    cells.update((r, c0) for r in range(r0, r1 + 1))
    cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _find_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    rects: list[dict[str, int]] = []
    for color in np.unique(grid):
        color = int(color)
        if color in MARKER_COLORS or color in (BG, OBSTACLE):
            continue
        for comp in _components(grid == color):
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            ring = _perimeter_cells(r0, c0, r1, c1)
            if set(comp) != ring:
                continue
            rects.append(
                {
                    "wall": color,
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                }
            )
    rects.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["r1"], rect["c1"]))
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open marker-specified walls of closed rectangles, fill their interiors, and spill the fill outward through each opening.

    Input structure:
    - The background is black.
    - Several single-pixel-thick closed rectangular walls of one wall color sit on the grid.
    - Inside each rectangle there is exactly one non-background fill-color pixel and zero, one, or two marker pixels.
    - Marker colors encode directions: red opens the top wall, green the bottom wall, yellow the left wall, and orange the right wall.
    - Gray single-pixel obstacles may appear outside the rectangles on the black background.

    Rule:
    - Detect every hollow rectangular wall component and read the marker colors inside it.
    - Flood-fill the full interior of the rectangle with its fill color, covering the marker pixels as well.
    - For each indicated side, remove that entire wall segment and replace it with the fill color to make a window.
    - Continue the fill straight outward from each window, lane by lane across the window width, stopping each lane at the first non-black pixel or the grid boundary.

    Key insight:
    - The marker colors are a direction code rather than paint colors: they tell you which wall segments become openings, and those openings alone determine the spill directions.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for rect in _find_rectangles(g):
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        inside_colors = {int(v) for v in np.unique(interior)}
        fill_candidates = [color for color in inside_colors if color not in MARKER_COLORS and color != BG]
        assert len(fill_candidates) == 1, "each rectangle must contain exactly one fill color"
        fill_color = int(fill_candidates[0])
        open_sides = {MARKER_TO_SIDE[int(color)] for color in inside_colors & MARKER_COLORS}

        out[r0 + 1 : r1, c0 + 1 : c1] = fill_color

        for side in open_sides:
            if side == "up":
                out[r0, c0 : c1 + 1] = fill_color
                for c in range(c0, c1 + 1):
                    rr = r0 - 1
                    while rr >= 0 and out[rr, c] == BG:
                        out[rr, c] = fill_color
                        rr -= 1
            elif side == "down":
                out[r1, c0 : c1 + 1] = fill_color
                for c in range(c0, c1 + 1):
                    rr = r1 + 1
                    while rr < out.shape[0] and out[rr, c] == BG:
                        out[rr, c] = fill_color
                        rr += 1
            elif side == "left":
                out[r0 : r1 + 1, c0] = fill_color
                for r in range(r0, r1 + 1):
                    cc = c0 - 1
                    while cc >= 0 and out[r, cc] == BG:
                        out[r, cc] = fill_color
                        cc -= 1
            else:
                out[r0 : r1 + 1, c1] = fill_color
                for r in range(r0, r1 + 1):
                    cc = c1 + 1
                    while cc < out.shape[1] and out[r, cc] == BG:
                        out[r, cc] = fill_color
                        cc += 1

    return out


def _side_count_plan(n_rects: int) -> list[int]:
    if n_rects == 3:
        counts = [0, 1, 2]
        random.shuffle(counts)
        return counts
    return random.sample([0, 1, 2], 2)


def _sample_sides(count: int) -> list[str]:
    all_sides = ["up", "down", "left", "right"]
    if count == 0:
        return []
    if count == 1:
        return [random.choice(all_sides)]
    return random.sample(all_sides, 2)


def _choose_obstacle_lanes(lanes: list[int], max_len: int, force_obstacles: bool) -> set[int]:
    if max_len < 1 or len(lanes) < 2:
        return set()
    if not force_obstacles and random.random() < 0.45:
        return set()
    max_blocked = max(1, min(len(lanes) - 1, len(lanes) // 3))
    blocked = random.randint(1, max_blocked)
    return set(random.sample(lanes, blocked))


def _project(side: str, r0: int, c0: int, r1: int, c1: int, lane: int, dist: int) -> tuple[int, int]:
    if side == "up":
        return r0 - dist, lane
    if side == "down":
        return r1 + dist, lane
    if side == "left":
        return lane, c0 - dist
    return lane, c1 + dist


def _build_spill(
    height: int,
    width: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    sides: list[str],
    force_obstacles: bool,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    spill_cells: set[tuple[int, int]] = set()
    obstacle_cells: set[tuple[int, int]] = set()
    forced_side = random.choice(sides) if force_obstacles and sides else None

    for side in sides:
        if side in ("up", "down"):
            lanes = list(range(c0, c1 + 1))
            max_len = r0 if side == "up" else height - 1 - r1
        else:
            lanes = list(range(r0, r1 + 1))
            max_len = c0 if side == "left" else width - 1 - c1

        blocked_lanes = _choose_obstacle_lanes(lanes, max_len, force_obstacles and side == forced_side)
        for lane in lanes:
            obstacle_dist = None
            if lane in blocked_lanes:
                obstacle_dist = random.randint(1, max_len)
                obstacle_cells.add(_project(side, r0, c0, r1, c1, lane, obstacle_dist))
            fill_len = max_len if obstacle_dist is None else obstacle_dist - 1
            for dist in range(1, fill_len + 1):
                spill_cells.add(_project(side, r0, c0, r1, c1, lane, dist))

    return spill_cells, obstacle_cells


def _corridor_clear(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int, side: str) -> bool:
    if side == "up":
        return not reserved[:r0, c0 : c1 + 1].any()
    if side == "down":
        return not reserved[r1 + 1 :, c0 : c1 + 1].any()
    if side == "left":
        return not reserved[r0 : r1 + 1, :c0].any()
    return not reserved[r0 : r1 + 1, c1 + 1 :].any()


def _place_rectangle(
    grid: np.ndarray,
    reserved: np.ndarray,
    wall_color: int,
    fill_color: int,
    sides: list[str],
    force_obstacles: bool,
) -> bool:
    height, width = grid.shape

    for _ in range(PLACEMENT_TRIES):
        outer_h = random.randint(*RECT_OUTER_HEIGHT_RANGE)
        outer_w = random.randint(*RECT_OUTER_WIDTH_RANGE)

        min_r0 = 1 if "up" in sides else 0
        max_r0 = height - outer_h - (1 if "down" in sides else 0)
        min_c0 = 1 if "left" in sides else 0
        max_c0 = width - outer_w - (1 if "right" in sides else 0)
        if max_r0 < min_r0 or max_c0 < min_c0:
            continue

        r0 = random.randint(min_r0, max_r0)
        c0 = random.randint(min_c0, max_c0)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1

        if reserved[r0 : r1 + 1, c0 : c1 + 1].any():
            continue
        if any(not _corridor_clear(reserved, r0, c0, r1, c1, side) for side in sides):
            continue

        spill_cells, obstacle_cells = _build_spill(height, width, r0, c0, r1, c1, sides, force_obstacles)

        grid[r0, c0 : c1 + 1] = wall_color
        grid[r1, c0 : c1 + 1] = wall_color
        grid[r0 : r1 + 1, c0] = wall_color
        grid[r0 : r1 + 1, c1] = wall_color

        interior_cells = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
        placed = random.sample(interior_cells, 1 + len(sides))
        fill_r, fill_c = placed[0]
        grid[fill_r, fill_c] = fill_color
        for (mr, mc), side in zip(placed[1:], sides, strict=False):
            grid[mr, mc] = SIDE_TO_MARKER[side]

        for rr, cc in obstacle_cells:
            grid[rr, cc] = OBSTACLE

        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
        for rr, cc in spill_cells | obstacle_cells:
            reserved[rr, cc] = True
        return True

    return False


def _add_spare_obstacles(grid: np.ndarray, reserved: np.ndarray) -> None:
    count = random.randint(0, 3)
    if count == 0:
        return
    empties = np.argwhere((grid == BG) & (~reserved))
    if len(empties) == 0:
        return
    random.shuffle(empties)
    for r, c in empties[:count]:
        grid[int(r), int(c)] = OBSTACLE
        reserved[int(r), int(c)] = True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    wall_color = random.choice(FLEX_COLORS)
    fill_palette = [color for color in FLEX_COLORS if color != wall_color]
    n_rects = random.choice(RECT_COUNT_OPTIONS)
    fill_colors = random.sample(fill_palette, n_rects)

    side_counts = _side_count_plan(n_rects)
    side_sets = [_sample_sides(count) for count in side_counts]
    open_rects = [idx for idx, sides in enumerate(side_sets) if sides]
    force_obstacles = bool(open_rects) and random.choice([False, True])
    forced_idx = random.choice(open_rects) if force_obstacles else None

    order = list(range(n_rects))
    random.shuffle(order)
    for idx in order:
        if not _place_rectangle(
            grid,
            reserved,
            wall_color,
            fill_colors[idx],
            side_sets[idx],
            idx == forced_idx,
        ):
            return None

    _add_spare_obstacles(grid, reserved)

    if force_obstacles and not np.any(grid == OBSTACLE):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
