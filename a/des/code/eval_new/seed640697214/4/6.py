import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

TOP_WALL = RED
BOTTOM_WALL = BLUE
LEFT_WALL = GREEN
RIGHT_WALL = YELLOW

GRID_MIN = 11
GRID_MAX = 18
MAX_GENERATION_ATTEMPTS = 400


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(values, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: set[tuple[int, int]] = set()
            while q:
                cr, cc = q.popleft()
                comp.add((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _paint_walls(grid: np.ndarray) -> None:
    grid[0, :] = TOP_WALL
    grid[-1, :] = BOTTOM_WALL
    if grid.shape[0] > 2:
        grid[1:-1, 0] = LEFT_WALL
        grid[1:-1, -1] = RIGHT_WALL


def _wall_colors(grid: np.ndarray) -> dict[str, int]:
    h, w = grid.shape
    top = _mode(grid[0, 1:-1] if w > 2 else grid[0, :])
    bottom = _mode(grid[-1, 1:-1] if w > 2 else grid[-1, :])
    left = _mode(grid[1:-1, 0] if h > 2 else grid[:, 0])
    right = _mode(grid[1:-1, -1] if h > 2 else grid[:, -1])
    return {"up": int(top), "down": int(bottom), "left": int(left), "right": int(right)}


def _cargoes(grid: np.ndarray) -> list[tuple[int, int]]:
    cargos: list[tuple[int, int]] = []
    for comp in _components_of_color(grid, GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        if len(comp) != 4:
            continue
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 1 or c1 - c0 != 1:
            continue
        if comp == {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            cargos.append((int(r0), int(c0)))
    cargos.sort()
    return cargos


def _arrow_variants(r: int, c: int, direction: str) -> list[set[tuple[int, int]]]:
    if direction == "right":
        return [
            {(r, c - 1), (r + 1, c - 1), (r, c - 2)},
            {(r, c - 1), (r + 1, c - 1), (r + 1, c - 2)},
        ]
    if direction == "left":
        return [
            {(r, c + 2), (r + 1, c + 2), (r, c + 3)},
            {(r, c + 2), (r + 1, c + 2), (r + 1, c + 3)},
        ]
    if direction == "down":
        return [
            {(r - 1, c), (r - 1, c + 1), (r - 2, c)},
            {(r - 1, c), (r - 1, c + 1), (r - 2, c + 1)},
        ]
    if direction == "up":
        return [
            {(r + 2, c), (r + 2, c + 1), (r + 3, c)},
            {(r + 2, c), (r + 2, c + 1), (r + 3, c + 1)},
        ]
    raise ValueError(direction)


def _cargo_direction(
    cargo: tuple[int, int],
    orange_components: list[set[tuple[int, int]]],
) -> str:
    r, c = cargo
    matched: list[str] = []
    for direction in ("up", "down", "left", "right"):
        variants = _arrow_variants(r, c, direction)
        if any(variant in orange_components for variant in variants):
            matched.append(direction)
    assert len(matched) == 1, "each cargo must have exactly one adjacent arrow"
    return matched[0]


def _distance_to_wall(top: int, left: int, direction: str, h: int, w: int) -> int:
    if direction == "up":
        return int(top - 1)
    if direction == "down":
        return int((h - 3) - top)
    if direction == "left":
        return int(left - 1)
    if direction == "right":
        return int((w - 3) - left)
    raise ValueError(direction)


def _step(direction: str) -> tuple[int, int]:
    if direction == "up":
        return -1, 0
    if direction == "down":
        return 1, 0
    if direction == "left":
        return 0, -1
    if direction == "right":
        return 0, 1
    raise ValueError(direction)


def _can_place(top: int, left: int, occupied: np.ndarray, h: int, w: int) -> bool:
    if top < 1 or left < 1 or top + 1 > h - 2 or left + 1 > w - 2:
        return False
    return not bool(occupied[top : top + 2, left : left + 2].any())


def _front_colors(top: int, left: int, direction: str, parked: np.ndarray) -> list[int]:
    if direction == "up":
        coords = [(top - 1, left), (top - 1, left + 1)]
    elif direction == "down":
        coords = [(top + 2, left), (top + 2, left + 1)]
    elif direction == "left":
        coords = [(top, left - 1), (top + 1, left - 1)]
    else:
        coords = [(top, left + 2), (top + 1, left + 2)]
    colors = [int(parked[r, c]) for r, c in coords if int(parked[r, c]) >= 0]
    return colors


def _paint_block(grid: np.ndarray, top: int, left: int, color: int) -> None:
    grid[top : top + 2, left : left + 2] = int(color)


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _mode(g)
    wall = _wall_colors(g)
    cargos = _cargoes(g)
    orange_components = [comp for comp in _components_of_color(g, ORANGE) if len(comp) == 3]

    pieces = []
    for top, left in cargos:
        direction = _cargo_direction((top, left), orange_components)
        pieces.append(
            {
                "top": int(top),
                "left": int(left),
                "direction": direction,
                "dist": _distance_to_wall(int(top), int(left), direction, h, w),
            }
        )
    pieces.sort(key=lambda piece: (piece["dist"], piece["top"], piece["left"]))

    out = g.copy()
    out[g == GREY] = bg
    out[g == ORANGE] = bg

    occupied = np.zeros((h, w), dtype=bool)
    parked = np.full((h, w), -1, dtype=int)

    for piece in pieces:
        top = int(piece["top"])
        left = int(piece["left"])
        direction = piece["direction"]
        dr, dc = _step(direction)
        cur_r, cur_c = top, left
        while _can_place(cur_r + dr, cur_c + dc, occupied, h, w):
            cur_r += dr
            cur_c += dc

        if _distance_to_wall(cur_r, cur_c, direction, h, w) == 0:
            color = wall[direction]
        else:
            blockers = _front_colors(cur_r, cur_c, direction, parked)
            assert blockers, "a non-wall stop must touch a parked cargo"
            color = int(blockers[0])
            assert all(int(v) == int(color) for v in blockers), "front contact must be color-consistent"

        _paint_block(out, cur_r, cur_c, color)
        occupied[cur_r : cur_r + 2, cur_c : cur_c + 2] = True
        parked[cur_r : cur_r + 2, cur_c : cur_c + 2] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each grey 2x2 cargo toward the wall indicated by its adjacent L-shaped orange pusher.

    Input structure:
    - bg = most common color (black).
    - A 1-cell border wall frames the grid: top red, bottom blue, left green, right yellow.
    - Several isolated grey 2x2 cargo blocks lie in the interior.
    - Each cargo has one adjacent 3-cell orange L-shape: a 2-cell shaft flush with one cargo side plus a 1-cell tail extending away from the cargo.

    Rule:
    - Read the shaft side of each orange L-shape; the cargo moves from the arrow toward the opposite wall.
    - Sort cargos by their current distance to that target wall, breaking ties by smaller row then smaller column.
    - Move cargos in that order, sliding each until the next step would hit the border wall or an already parked cargo; erase all arrows.
    - Recolor a parked cargo to the wall color it touches, or if it stops against a parked cargo, inherit that cargo's current color.

    Key insight:
    - The orange mark is not a generic corner: its 2-cell side is pressed against the cargo, so the side it occupies encodes the push direction, and near-wall-first ordering determines the stacking and color-propagation chain.
    """
    return _solve(grid)


def _lane_tops(limit: int) -> list[int]:
    return [pos for pos in range(1, limit - 1, 3) if pos + 1 <= limit - 2]


def _right_pairs(width: int) -> list[tuple[int, int]]:
    pairs = []
    for back in range(2, width - 5):
        for front in range(back + 5, width - 3):
            pairs.append((int(front), int(back)))
    return pairs


def _left_pairs(width: int) -> list[tuple[int, int]]:
    pairs = []
    for front in range(2, width - 8):
        for back in range(front + 5, width - 3):
            pairs.append((int(front), int(back)))
    return pairs


def _down_pairs(height: int) -> list[tuple[int, int]]:
    pairs = []
    for back in range(2, height - 5):
        for front in range(back + 5, height - 3):
            pairs.append((int(front), int(back)))
    return pairs


def _up_pairs(height: int) -> list[tuple[int, int]]:
    pairs = []
    for front in range(2, height - 8):
        for back in range(front + 5, height - 3):
            pairs.append((int(front), int(back)))
    return pairs


def _paint_arrow(grid: np.ndarray, top: int, left: int, direction: str) -> None:
    arrow = random.choice(_arrow_variants(top, left, direction))
    for r, c in arrow:
        grid[r, c] = ORANGE


def _group_layouts(height: int, width: int, orientation: str):
    groups = []
    if orientation == "horizontal":
        lanes = _lane_tops(height)
        if len(lanes) < 2:
            return None
        group_count = random.randint(2, min(4, len(lanes)))
        dirs = ["left", "right"]
        while len(dirs) < group_count:
            dirs.append(random.choice(["left", "right"]))
        random.shuffle(dirs)
        rows = random.sample(lanes, group_count)
        row_counts = Counter(dirs)
        reuse_front = random.random() < 0.5
        reused_pairs: dict[str, tuple[int, int] | None] = {"left": None, "right": None}
        for row, direction in zip(rows, dirs):
            pool = _left_pairs(width) if direction == "left" else _right_pairs(width)
            if not pool:
                return None
            if row_counts[direction] > 1 and reuse_front and reused_pairs[direction] is not None and random.random() < 0.7:
                pair = reused_pairs[direction]
            else:
                pair = random.choice(pool)
                if reused_pairs[direction] is None:
                    reused_pairs[direction] = pair
            if direction == "left":
                inputs = [(row, pair[0]), (row, pair[1])]
                outputs = [(row, 1), (row, 3)]
            else:
                inputs = [(row, pair[0]), (row, pair[1])]
                outputs = [(row, width - 3), (row, width - 5)]
            groups.append({"direction": direction, "inputs": inputs, "outputs": outputs})
        return groups

    lanes = _lane_tops(width)
    if len(lanes) < 2:
        return None
    group_count = random.randint(2, min(4, len(lanes)))
    dirs = ["up", "down"]
    while len(dirs) < group_count:
        dirs.append(random.choice(["up", "down"]))
    random.shuffle(dirs)
    cols = random.sample(lanes, group_count)
    col_counts = Counter(dirs)
    reuse_front = random.random() < 0.5
    reused_pairs: dict[str, tuple[int, int] | None] = {"up": None, "down": None}
    for col, direction in zip(cols, dirs):
        pool = _up_pairs(height) if direction == "up" else _down_pairs(height)
        if not pool:
            return None
        if col_counts[direction] > 1 and reuse_front and reused_pairs[direction] is not None and random.random() < 0.7:
            pair = reused_pairs[direction]
        else:
            pair = random.choice(pool)
            if reused_pairs[direction] is None:
                reused_pairs[direction] = pair
        if direction == "up":
            inputs = [(pair[0], col), (pair[1], col)]
            outputs = [(1, col), (3, col)]
        else:
            inputs = [(pair[0], col), (pair[1], col)]
            outputs = [(height - 3, col), (height - 5, col)]
        groups.append({"direction": direction, "inputs": inputs, "outputs": outputs})
    return groups


def _render_input(height: int, width: int, groups) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    _paint_walls(grid)
    for group in groups:
        for top, left in group["inputs"]:
            assert np.all(grid[top : top + 2, left : left + 2] == BG)
            _paint_block(grid, top, left, GREY)
            _paint_arrow(grid, top, left, group["direction"])
    return grid


def _render_expected(height: int, width: int, groups) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    _paint_walls(grid)
    wall_color = {
        "up": TOP_WALL,
        "down": BOTTOM_WALL,
        "left": LEFT_WALL,
        "right": RIGHT_WALL,
    }
    for group in groups:
        color = wall_color[group["direction"]]
        for top, left in group["outputs"]:
            _paint_block(grid, top, left, color)
    return grid


def _background_is_dominant(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    return counts[BG] > max(count for color, count in counts.items() if color != BG)


def _attempt() -> np.ndarray | None:
    height = random.randint(GRID_MIN, GRID_MAX)
    width = random.randint(GRID_MIN, GRID_MAX)
    orientation = random.choice(["horizontal", "vertical"])
    groups = _group_layouts(height, width, orientation)
    if groups is None:
        return None

    inp = _render_input(height, width, groups)
    expected = _render_expected(height, width, groups)

    if not _background_is_dominant(inp):
        return None
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
