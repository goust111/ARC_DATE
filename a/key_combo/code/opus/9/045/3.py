from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
X_COLORS = (1, 2, 3, 4)
SPRITE_COLORS = (5, 6, 7, 9)

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)

COLOR_TO_DIR = {
    1: UP,
    2: RIGHT,
    3: DOWN,
    4: LEFT,
}
DIR_TO_COLOR = {value: key for key, value in COLOR_TO_DIR.items()}
SIDE_TO_DIR = {
    "top": UP,
    "right": RIGHT,
    "bottom": DOWN,
    "left": LEFT,
}
SIDE_ORDER = ("top", "right", "bottom", "left")
FORBIDDEN_CORNER_PLANS = (
    (("top", LEFT), ("left", UP)),
    (("top", RIGHT), ("right", UP)),
    (("bottom", LEFT), ("left", DOWN)),
    (("bottom", RIGHT), ("right", DOWN)),
)

X_OFFSETS = (
    (0, 0),
    (0, 4),
    (1, 1),
    (1, 3),
    (2, 2),
    (3, 1),
    (3, 3),
    (4, 0),
    (4, 4),
)

FOLD_MAP = {
    (0, 0): (0, 0),
    (1, 1): (0, 1),
    (0, 4): (0, 2),
    (1, 3): (1, 2),
    (2, 2): (1, 1),
    (3, 1): (1, 0),
    (4, 0): (2, 0),
    (3, 3): (2, 1),
    (4, 4): (2, 2),
}

POLYOMINO_ATTEMPTS = 80
PLACEMENT_ATTEMPTS = 160
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    return max(rows) + 1, max(cols) + 1


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    h, w = _shape_size(shape)
    return _normalize_shape({(c, h - 1 - r) for r, c in shape})


def _flip_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    _h, w = _shape_size(shape)
    return _normalize_shape({(r, w - 1 - c) for r, c in shape})


def _random_polyomino(rng: random.Random, size: int) -> tuple[tuple[int, int], ...]:
    for _ in range(POLYOMINO_ATTEMPTS):
        cells = {(0, 0)}
        while len(cells) < size:
            frontier = set()
            for r, c in cells:
                frontier.update(_neighbors4(r, c))
            frontier -= cells
            cells.add(rng.choice(tuple(frontier)))
        shape = _normalize_shape(cells)
        if rng.random() < 0.5:
            shape = _flip_shape(shape)
        for _ in range(rng.randrange(4)):
            shape = _rotate_shape(shape)
        h, w = _shape_size(shape)
        if h <= 4 and w <= 4:
            return shape
    raise RuntimeError("failed to sample a compact polyomino")


def _coords_from_shape(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> np.ndarray:
    return np.array([(r0 + r, c0 + c) for r, c in shape], dtype=int)


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros(mask.shape, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        cells = []
        while stack:
            cr, cc = stack.pop()
            cells.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc):
                if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(cells, dtype=int))
    return comps


def _find_x_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    x_mask = np.isin(grid, X_COLORS)
    coords = np.argwhere(x_mask)
    assert coords.shape[0] == 9, "expected exactly 9 X pixels"
    bbox = _bbox(coords)
    r0, c0, r1, c1 = bbox
    assert (r1 - r0 + 1, c1 - c0 + 1) == (5, 5), "expected a 5x5 X bbox"
    expected = {(r0 + dr, c0 + dc) for dr, dc in X_OFFSETS}
    actual = {tuple(map(int, coord)) for coord in coords}
    assert actual == expected, "expected foreground X pixels exactly on both diagonals"
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    off_diag = np.ones((5, 5), dtype=bool)
    for dr, dc in X_OFFSETS:
        off_diag[dr, dc] = False
    assert np.all(patch[off_diag] == BACKGROUND), "expected only background inside the X bbox off the diagonals"
    return bbox


def _folded_grid(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, _r1, _c1 = bbox
    folded = np.full((3, 3), -1, dtype=int)
    for (dr, dc), (fr, fc) in FOLD_MAP.items():
        folded[fr, fc] = int(grid[r0 + dr, c0 + dc])
    assert np.all(folded >= 0)
    return folded


def _instruction_colors(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, int]:
    folded = _folded_grid(grid, bbox)
    center = int(folded[1, 1])
    edge_pairs = {
        "top": (int(folded[0, 0]), int(folded[0, 1])),
        "right": (int(folded[0, 2]), int(folded[1, 2])),
        "left": (int(folded[1, 0]), int(folded[2, 0])),
        "bottom": (int(folded[2, 1]), int(folded[2, 2])),
    }
    result: dict[str, int] = {}
    for side, (a, b) in edge_pairs.items():
        result[side] = a if a == b else center
    return result


def _sprite_components(grid: np.ndarray, x_bbox: tuple[int, int, int, int]) -> dict[str, dict[str, np.ndarray | int]]:
    r0, c0, r1, c1 = x_bbox
    for dr, dc in X_OFFSETS:
        assert grid[r0 + dr, c0 + dc] in X_COLORS
    out: dict[str, dict[str, np.ndarray | int]] = {}
    for color in SPRITE_COLORS:
        coords = np.argwhere(grid == color)
        assert coords.shape[0] >= 1, "expected every sprite color exactly once"
        comps = _connected_components(grid == color)
        assert len(comps) == 1, "each sprite color must form one 4-connected component"
        comp = comps[0]
        assert np.all((comp[:, 0] < r0) | (comp[:, 0] > r1) | (comp[:, 1] < c0) | (comp[:, 1] > c1))
        touches = []
        if np.any(comp[:, 0] == 0):
            touches.append("top")
        if np.any(comp[:, 0] == grid.shape[0] - 1):
            touches.append("bottom")
        if np.any(comp[:, 1] == 0):
            touches.append("left")
        if np.any(comp[:, 1] == grid.shape[1] - 1):
            touches.append("right")
        assert len(touches) == 1, "each sprite must touch exactly one grid edge"
        side = touches[0]
        assert side not in out, "each edge must host exactly one sprite"
        out[side] = {"color": color, "coords": comp}
    assert set(out) == set(SIDE_ORDER), "missing edge sprite"
    return out


def _steps_to_boundary(coords: np.ndarray, direction: tuple[int, int], shape: tuple[int, int]) -> int:
    dr, dc = direction
    h, w = shape
    if dr == -1:
        return int(coords[:, 0].min())
    if dr == 1:
        return int(h - 1 - coords[:, 0].max())
    if dc == -1:
        return int(coords[:, 1].min())
    return int(w - 1 - coords[:, 1].max())


def _translate(coords: np.ndarray, direction: tuple[int, int], steps: int) -> np.ndarray:
    dr, dc = direction
    return coords + np.array([dr * steps, dc * steps], dtype=int)


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert np.any(g == BACKGROUND)
    x_bbox = _find_x_bbox(g)
    instructions = _instruction_colors(g, x_bbox)
    sprites = _sprite_components(g, x_bbox)

    out = np.full(g.shape, BACKGROUND, dtype=int)
    for side in SIDE_ORDER:
        sprite = sprites[side]
        coords = np.asarray(sprite["coords"], dtype=int)
        direction = COLOR_TO_DIR[instructions[side]]
        if direction == SIDE_TO_DIR[side]:
            continue
        shifted = _translate(coords, direction, _steps_to_boundary(coords, direction, g.shape))
        assert np.all(out[shifted[:, 0], shifted[:, 1]] == BACKGROUND), "moved sprites must not overlap"
        out[shifted[:, 0], shifted[:, 1]] = int(sprite["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the central X into edge votes and use them to push or erase the four edge sprites.

    Input structure:
    - The grid has black background, a central 5x5 X whose 9 diagonal pixels use only blue/red/green/yellow, and one monochrome 4-connected sprite on each of the four outer edges.
    - The X occupies exactly the two diagonals of its 5x5 box; the four edge sprites use distinct non-X colors and each touches exactly one grid boundary.

    Rule:
    - Fold the 5x5 X into a 3x3 outline: northwest arm -> top edge, northeast arm -> right edge, southwest arm -> left edge, southeast arm -> bottom edge, center stays centered.
    - For each folded side, read its two arm pixels: if they match, that color is the side's instruction; otherwise use the folded center color.
    - Decode blue/red/green/yellow as up/right/down/left and apply the direction to the sprite on the matching outer edge.
    - If a sprite is told to move further into the same edge it already touches, erase it; otherwise translate it straight in that direction until it reaches the grid boundary. Then remove the central X entirely.

    Key insight:
    - The X is a compressed instruction panel: its diagonal arms must first be folded onto the four sides, then each side's voted color becomes a direction for exactly one edge sprite.
    """
    return _transform(grid)


def _candidate_origins(
    side: str,
    shape: tuple[tuple[int, int], ...],
    grid_shape: tuple[int, int],
    x_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = _shape_size(shape)
    rows, cols = grid_shape
    r0, c0, r1, c1 = x_bbox
    origins: list[tuple[int, int]] = []
    if side == "top":
        left_end = c0 - w
        right_start = c1 + 1
        for c in range(1, left_end + 1):
            origins.append((0, c))
        for c in range(right_start, cols - w):
            origins.append((0, c))
    elif side == "bottom":
        left_end = c0 - w
        right_start = c1 + 1
        for c in range(1, left_end + 1):
            origins.append((rows - h, c))
        for c in range(right_start, cols - w):
            origins.append((rows - h, c))
    elif side == "left":
        top_end = r0 - h
        bottom_start = r1 + 1
        for r in range(1, top_end + 1):
            origins.append((r, 0))
        for r in range(bottom_start, rows - h):
            origins.append((r, 0))
    else:
        top_end = r0 - h
        bottom_start = r1 + 1
        for r in range(1, top_end + 1):
            origins.append((r, cols - w))
        for r in range(bottom_start, rows - h):
            origins.append((r, cols - w))
    return origins


def _all_non_background_except(grid: np.ndarray, own: set[tuple[int, int]]) -> set[tuple[int, int]]:
    blocked = set()
    for r, c in np.argwhere(grid != BACKGROUND):
        cell = (int(r), int(c))
        if cell not in own:
            blocked.add(cell)
    return blocked


def _swept_cells(coords: np.ndarray, direction: tuple[int, int], steps: int) -> set[tuple[int, int]]:
    own = {tuple(map(int, cell)) for cell in coords}
    swept = set()
    for step in range(1, steps + 1):
        shifted = _translate(coords, direction, step)
        for r, c in shifted:
            cell = (int(r), int(c))
            if cell not in own:
                swept.add(cell)
    return swept


def _plan_instructions(rng: random.Random) -> dict[str, int]:
    while True:
        erase_count = rng.choice((1, 2))
        erased_sides = set(rng.sample(SIDE_ORDER, erase_count))
        plan: dict[str, int] = {}
        for side in SIDE_ORDER:
            if side in erased_sides:
                plan[side] = DIR_TO_COLOR[SIDE_TO_DIR[side]]
            else:
                choices = [color for color, direction in COLOR_TO_DIR.items() if direction != SIDE_TO_DIR[side]]
                plan[side] = rng.choice(choices)
        if len(set(plan.values())) < 3:
            continue
        moved_dirs = [COLOR_TO_DIR[plan[side]] for side in SIDE_ORDER if side not in erased_sides]
        if not any(direction in (UP, DOWN) for direction in moved_dirs):
            continue
        if not any(direction in (LEFT, RIGHT) for direction in moved_dirs):
            continue
        if any(plan[a_side] == DIR_TO_COLOR[a_dir] and plan[b_side] == DIR_TO_COLOR[b_dir] for (a_side, a_dir), (b_side, b_dir) in FORBIDDEN_CORNER_PLANS):
            continue
        return plan


def _place_sprites(
    rng: random.Random,
    grid: np.ndarray,
    x_bbox: tuple[int, int, int, int],
    plan: dict[str, int],
) -> bool:
    rows, cols = grid.shape
    output_occupied: set[tuple[int, int]] = set()
    reserved_paths: set[tuple[int, int]] = set()
    side_colors = list(SPRITE_COLORS)
    rng.shuffle(side_colors)
    for side, color in zip(SIDE_ORDER, side_colors):
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            shape = _random_polyomino(rng, rng.randint(3, 5))
            origins = _candidate_origins(side, shape, (rows, cols), x_bbox)
            if not origins:
                continue
            rng.shuffle(origins)
            for origin in origins:
                coords = _coords_from_shape(shape, origin[0], origin[1])
                cells = {tuple(map(int, cell)) for cell in coords}
                if any(grid[r, c] != BACKGROUND for r, c in cells):
                    continue
                if cells & reserved_paths:
                    continue
                direction = COLOR_TO_DIR[plan[side]]
                if direction != SIDE_TO_DIR[side]:
                    steps = _steps_to_boundary(coords, direction, grid.shape)
                    if steps <= 0:
                        continue
                    blocked = _all_non_background_except(grid, cells)
                    swept = _swept_cells(coords, direction, steps)
                    if swept & blocked:
                        continue
                    shifted = _translate(coords, direction, steps)
                    shifted_cells = {tuple(map(int, cell)) for cell in shifted}
                    if shifted_cells & output_occupied:
                        continue
                else:
                    swept = set()
                    shifted_cells = set()
                for r, c in cells:
                    grid[r, c] = color
                output_occupied |= shifted_cells
                reserved_paths |= swept
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _paint_x(grid: np.ndarray, bbox: tuple[int, int, int, int], plan: dict[str, int], center_color: int) -> None:
    r0, c0, _r1, _c1 = bbox
    arm_colors = {
        (0, 0): plan["top"],
        (1, 1): plan["top"],
        (0, 4): plan["right"],
        (1, 3): plan["right"],
        (3, 1): plan["left"],
        (4, 0): plan["left"],
        (3, 3): plan["bottom"],
        (4, 4): plan["bottom"],
        (2, 2): center_color,
    }
    for (dr, dc), color in arm_colors.items():
        grid[r0 + dr, c0 + dc] = color


def _independent_semantic_checks(grid: np.ndarray, out: np.ndarray) -> None:
    x_bbox = _find_x_bbox(grid)
    instructions = _instruction_colors(grid, x_bbox)
    sprites = _sprite_components(grid, x_bbox)

    folded = _folded_grid(grid, x_bbox)
    assert folded[0, 0] == folded[0, 1]
    assert folded[0, 2] == folded[1, 2]
    assert folded[1, 0] == folded[2, 0]
    assert folded[2, 1] == folded[2, 2]

    erased = 0
    moved = 0
    for side in SIDE_ORDER:
        sprite = sprites[side]
        color = int(sprite["color"])
        coords = np.asarray(sprite["coords"], dtype=int)
        direction = COLOR_TO_DIR[instructions[side]]
        if direction == SIDE_TO_DIR[side]:
            erased += 1
            assert not np.any(out == color), "erased sprite must disappear"
            continue
        moved += 1
        steps = _steps_to_boundary(coords, direction, grid.shape)
        blocked = _all_non_background_except(grid, {tuple(map(int, cell)) for cell in coords})
        assert not (_swept_cells(coords, direction, steps) & blocked), "sprite path must be clear"
        shifted = _translate(coords, direction, steps)
        expected_cells = {tuple(map(int, cell)) for cell in shifted}
        actual_cells = {tuple(map(int, cell)) for cell in np.argwhere(out == color)}
        assert actual_cells == expected_cells, "sprite must move exactly to the boundary"
    assert erased >= 1 and moved >= 1, "generator must expose both erase and move branches"
    assert not np.any(np.isin(out, X_COLORS)), "output must not keep the X colors"
    assert np.count_nonzero(out) > 0, "output must retain at least one sprite"
    assert not np.array_equal(grid, out), "transform must change the grid"


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    rows = rng.choice((15, 17, 19))
    cols = rng.choice((15, 17, 19))
    grid = np.full((rows, cols), BACKGROUND, dtype=int)
    x_bbox = (rows // 2 - 2, cols // 2 - 2, rows // 2 + 2, cols // 2 + 2)
    plan = _plan_instructions(rng)
    center_color = rng.choice(X_COLORS)
    _paint_x(grid, x_bbox, plan, center_color)
    if not _place_sprites(rng, grid, x_bbox, plan):
        return None
    out = _transform(grid)
    _independent_semantic_checks(grid, out)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
