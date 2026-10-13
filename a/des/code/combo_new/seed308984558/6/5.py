import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
MAROON = 9

REGION_COLORS = (BLUE, RED, GREEN, YELLOW, MAGENTA, ORANGE, CYAN)
SHADOW_DIRS = ("up", "down", "left", "right")
DIR_VECTORS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
LIGHT_SIDES = {
    "up": "bottom",
    "down": "top",
    "left": "right",
    "right": "left",
}
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN = 16
GRID_MAX = 24
OBJECT_COUNT_RANGE = (2, 4)
OBJECT_AREA_RANGE = (3, 8)
GEN_ATTEMPTS = 300
PLACEMENT_ATTEMPTS = 200
SHAPE_ATTEMPTS = 120
REGION_MARGIN = 4
MAX_SHAPE_SPAN = 4


def neighbors4(r: int, c: int):
    for dr, dc in NEIGHBORS4:
        yield r + dr, c + dc


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _transform_shape(cells: set[tuple[int, int]], rng: random.Random) -> set[tuple[int, int]]:
    out = set(cells)
    if rng.random() < 0.5:
        out = {(r, -c) for r, c in out}
    if rng.random() < 0.5:
        out = {(-r, c) for r, c in out}
    if rng.random() < 0.5:
        out = {(c, r) for r, c in out}
    return _normalize(out)


def _random_shape(rng: random.Random, shadow_dir: str) -> set[tuple[int, int]]:
    for _ in range(SHAPE_ATTEMPTS):
        target_area = rng.randint(*OBJECT_AREA_RANGE)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if not (-MAX_SHAPE_SPAN <= nr <= MAX_SHAPE_SPAN):
                        continue
                    if not (-MAX_SHAPE_SPAN <= nc <= MAX_SHAPE_SPAN):
                        continue
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(rng.choice(frontier))

        cells = _transform_shape(_normalize(cells), rng)
        height = 1 + max(r for r, _ in cells)
        width = 1 + max(c for _, c in cells)
        if height > MAX_SHAPE_SPAN or width > MAX_SHAPE_SPAN:
            continue
        if shadow_dir in ("up", "down") and width < 2:
            continue
        if shadow_dir in ("left", "right") and height < 2:
            continue
        return cells
    raise RuntimeError("failed to generate a valid shape")


def _color_components(grid: np.ndarray) -> list[tuple[int, set[tuple[int, int]]]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, set[tuple[int, int]]]] = []
    blocked = {BLACK, GREY, MAROON}

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in blocked or visited[r, c]:
                continue

            q = deque([(r, c)])
            visited[r, c] = True
            cells: set[tuple[int, int]] = set()

            while q:
                cr, cc = q.popleft()
                cells.add((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            components.append((color, cells))

    return components


def _shadow_direction(cells: set[tuple[int, int]], grey_cell: tuple[int, int]) -> tuple[int, int]:
    grey_r, grey_c = grey_cell
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]

    if grey_r < min(rows):
        return (1, 0)
    if grey_r > max(rows):
        return (-1, 0)
    if grey_c < min(cols):
        return (0, 1)
    if grey_c > max(cols):
        return (0, -1)

    for nr, nc in neighbors4(grey_r, grey_c):
        if (nr, nc) in cells:
            return nr - grey_r, nc - grey_c

    raise AssertionError("grey marker must touch its component")


def _project_component(
    grid: np.ndarray, cells: set[tuple[int, int]], shadow_step: tuple[int, int]
) -> set[tuple[int, int]]:
    h, w = grid.shape
    dr, dc = shadow_step
    hits: set[tuple[int, int]] = set()

    for r, c in cells:
        rr, cc = r + dr, c + dc
        last_black = None
        while 0 <= rr < h and 0 <= cc < w:
            if int(grid[rr, cc]) == MAROON:
                if last_black is not None:
                    hits.add(last_black)
                break
            if int(grid[rr, cc]) == BLACK:
                last_black = (rr, cc)
            rr += dr
            cc += dc

    return hits


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each colored region onto the opposite wall using its adjacent grey marker as the light-source cue.

    Input structure:
    - The background is black, and the whole grid edge is a continuous maroon wall.
    - Inside the arena are 2-4 small colored 4-connected regions with distinct colors.
    - Each colored region has exactly one grey pixel 4-adjacent to one side of the region, and that grey pixel is not shared by other regions.

    Rule:
    - For each colored region, find its unique adjacent grey pixel and treat the grey-side direction as the light-source direction.
    - Reverse that direction to get the shadow direction, then cast every pixel of the region straight toward the opposite wall.
    - Along each ray, ignore other colored pixels and markers, and color the last black cell before the maroon wall with the region's color.
    - Keep the original colored regions and grey markers unchanged; only add the wall shadows.

    Key insight:
    - The grey pixel is not part of the object to copy; it only encodes which wall receives the object's 1D silhouette projection.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color, cells in _color_components(g):
        grey_neighbors = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == GREY:
                    grey_neighbors.add((nr, nc))

        assert len(grey_neighbors) == 1, "each colored component must have exactly one adjacent grey marker"
        shadow_step = _shadow_direction(cells, next(iter(grey_neighbors)))
        for rr, cc in _project_component(g, cells, shadow_step):
            out[rr, cc] = color

    return out


def _wall_shadow_line(direction: str, h: int, w: int) -> tuple[str, int]:
    if direction == "up":
        return ("row", 1)
    if direction == "down":
        return ("row", h - 2)
    if direction == "left":
        return ("col", 1)
    return ("col", w - 2)


def _grey_candidates(shape_cells: set[tuple[int, int]], shadow_dir: str) -> list[tuple[int, int]]:
    light_side = LIGHT_SIDES[shadow_dir]
    if light_side == "top":
        edge = min(r for r, _ in shape_cells)
        return [(r - 1, c) for r, c in shape_cells if r == edge]
    if light_side == "bottom":
        edge = max(r for r, _ in shape_cells)
        return [(r + 1, c) for r, c in shape_cells if r == edge]
    if light_side == "left":
        edge = min(c for _, c in shape_cells)
        return [(r, c - 1) for r, c in shape_cells if c == edge]
    edge = max(c for _, c in shape_cells)
    return [(r, c + 1) for r, c in shape_cells if c == edge]


def _fits_direction_zone(
    top: int, left: int, height: int, width: int, h: int, w: int, shadow_dir: str
) -> bool:
    max_r = top + height - 1
    max_c = left + width - 1
    if shadow_dir == "up":
        return top >= h // 2
    if shadow_dir == "down":
        return max_r <= (h // 2)
    if shadow_dir == "left":
        return left >= w // 2
    return max_c <= (w // 2)


def _can_place(
    region_cells: set[tuple[int, int]],
    grey_cell: tuple[int, int],
    existing_special: set[tuple[int, int]],
    existing_regions: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    all_cells = set(region_cells)
    all_cells.add(grey_cell)

    for r, c in all_cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if (r, c) in existing_special:
            return False

    own_adj = sum((nr, nc) in region_cells for nr, nc in neighbors4(*grey_cell))
    if own_adj != 1:
        return False

    for r, c in region_cells:
        for nr, nc in neighbors4(r, c):
            if (nr, nc) in existing_special:
                return False

    for nr, nc in neighbors4(*grey_cell):
        if (nr, nc) in existing_regions:
            return False
        if (nr, nc) in existing_special:
            return False

    return True


def _validate_grid(grid: np.ndarray) -> bool:
    h, w = grid.shape
    if h > 30 or w > 30:
        return False
    if not np.all(grid[0, :] == MAROON):
        return False
    if not np.all(grid[-1, :] == MAROON):
        return False
    if not np.all(grid[:, 0] == MAROON):
        return False
    if not np.all(grid[:, -1] == MAROON):
        return False

    components = _color_components(grid)
    if not (2 <= len(components) <= 4):
        return False

    seen_colors = set()
    seen_greys = set()
    for color, cells in components:
        if color in seen_colors or color not in REGION_COLORS:
            return False
        seen_colors.add(color)
        if not (OBJECT_AREA_RANGE[0] <= len(cells) <= OBJECT_AREA_RANGE[1]):
            return False

        grey_neighbors = set()
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY:
                    grey_neighbors.add((nr, nc))
        if len(grey_neighbors) != 1:
            return False

        grey_cell = next(iter(grey_neighbors))
        if grey_cell in seen_greys:
            return False
        seen_greys.add(grey_cell)
        if sum((nr, nc) in cells for nr, nc in neighbors4(*grey_cell)) != 1:
            return False

    if int(np.sum(grid == GREY)) != len(components):
        return False

    out = main(grid)
    delta = np.argwhere(out != grid)
    if len(delta) < len(components):
        return False
    for r, c in delta:
        if int(grid[r, c]) != BLACK:
            return False
        if int(r) not in (1, h - 2) and int(c) not in (1, w - 2):
            return False

    for color, _cells in components:
        if not np.any((out == color) & (grid != color)):
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_objects = rng.randint(*OBJECT_COUNT_RANGE)
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    directions = rng.sample(SHADOW_DIRS, n_objects)
    colors = rng.sample(REGION_COLORS, n_objects)

    grid = np.full((h, w), BLACK, dtype=int)
    grid[0, :] = MAROON
    grid[-1, :] = MAROON
    grid[:, 0] = MAROON
    grid[:, -1] = MAROON

    occupied_regions: set[tuple[int, int]] = set()
    occupied_special: set[tuple[int, int]] = set()

    for color, shadow_dir in zip(colors, directions):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            shape = _random_shape(rng, shadow_dir)
            height = 1 + max(r for r, _ in shape)
            width = 1 + max(c for _, c in shape)

            top_lo = REGION_MARGIN
            left_lo = REGION_MARGIN
            top_hi = h - height - REGION_MARGIN
            left_hi = w - width - REGION_MARGIN
            if top_lo > top_hi or left_lo > left_hi:
                return None

            top = rng.randint(top_lo, top_hi)
            left = rng.randint(left_lo, left_hi)
            if not _fits_direction_zone(top, left, height, width, h, w, shadow_dir):
                continue

            region_cells = {(top + r, left + c) for r, c in shape}
            grey_options = _grey_candidates(region_cells, shadow_dir)
            rng.shuffle(grey_options)

            for grey_cell in grey_options:
                if not _can_place(region_cells, grey_cell, occupied_special, occupied_regions, h, w):
                    continue

                axis, wall_index = _wall_shadow_line(shadow_dir, h, w)
                if axis == "row":
                    shadow_coords = {(wall_index, c) for _, c in region_cells}
                else:
                    shadow_coords = {(r, wall_index) for r, _ in region_cells}
                if shadow_coords & occupied_special:
                    continue

                for r, c in region_cells:
                    grid[r, c] = color
                gr, gc = grey_cell
                grid[gr, gc] = GREY

                occupied_regions.update(region_cells)
                occupied_special.update(region_cells)
                occupied_special.add(grey_cell)
                placed = True
                break

            if placed:
                break

        if not placed:
            return None

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
