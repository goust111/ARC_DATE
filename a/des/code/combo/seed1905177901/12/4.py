import random

import numpy as np


BG_COLOR = 8
AXIS_COLOR = 9
OBJECT_COLORS = [0, 1, 2, 3, 4, 5, 6, 7]
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_SIDE_RANGE = (10, 18)
PAIR_COUNT_RANGE = (2, 5)
GENERATION_ATTEMPTS = 400
PAIR_PLACEMENT_ATTEMPTS = 160


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        if seen[rr, cc]:
            continue
        stack = [(rr, cc)]
        seen[rr, cc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            coords.append((cr, cc))
            for dr, dc in DIRS4:
                nr = cr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.array(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _reflect_point(r: int, c: int, axis_kind: str, axis_index: int) -> tuple[int, int]:
    if axis_kind == "H":
        return 2 * axis_index - r, c
    return r, 2 * axis_index - c


def _pair_matches_axis(
    pair: tuple[np.ndarray, np.ndarray],
    axis_kind: str,
    axis_index: int,
) -> bool:
    first, second = pair
    first_set = {tuple(map(int, rc)) for rc in first.tolist()}
    second_set = {tuple(map(int, rc)) for rc in second.tolist()}
    reflected = {
        _reflect_point(int(r), int(c), axis_kind, axis_index)
        for r, c in first_set
    }
    return reflected == second_set


def _collect_color_pairs(grid: np.ndarray) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != BG_COLOR]
    assert colors, "expected at least one foreground color"

    pairs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for color in colors:
        components = _component_coords(grid == color)
        assert len(components) == 2, "each foreground color must form exactly two objects"
        components.sort(key=lambda comp: tuple(_bbox(comp)))
        pairs[color] = (components[0], components[1])
    return pairs


def _infer_axis(grid: np.ndarray) -> tuple[str, int]:
    pairs = _collect_color_pairs(grid)

    row_mids4 = []
    col_mids4 = []
    for first, second in pairs.values():
        r0a, c0a, r1a, c1a = _bbox(first)
        r0b, c0b, r1b, c1b = _bbox(second)
        row_mids4.append(r0a + r1a + r0b + r1b)
        col_mids4.append(c0a + c1a + c0b + c1b)

    candidates: list[tuple[str, int]] = []
    if len(set(row_mids4)) == 1 and row_mids4[0] % 4 == 0:
        axis_row = row_mids4[0] // 4
        if all(_pair_matches_axis(pair, "H", axis_row) for pair in pairs.values()):
            candidates.append(("H", axis_row))
    if len(set(col_mids4)) == 1 and col_mids4[0] % 4 == 0:
        axis_col = col_mids4[0] // 4
        if all(_pair_matches_axis(pair, "V", axis_col) for pair in pairs.values()):
            candidates.append(("V", axis_col))

    assert len(candidates) == 1, "expected a unique global reflection axis"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color objects, infer their shared mirror axis, and draw that axis.

    The input uses azure background and contains exactly two connected components
    for each foreground color. Each color pair is placed as mirror images across
    one hidden horizontal or vertical axis shared by all pairs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    axis_kind, axis_index = _infer_axis(g)
    out = g.copy()

    if axis_kind == "H":
        out[axis_index, :] = AXIS_COLOR
    else:
        out[:, axis_index] = AXIS_COLOR

    return out


def _random_connected_mask(height: int, width: int) -> np.ndarray:
    area = height * width
    assert area >= 2
    min_cells = max(2, (area + 3) // 4)
    max_cells = max(min_cells, (3 * area + 2) // 4)
    target = random.randint(min_cells, max_cells)

    mask = np.zeros((height, width), dtype=bool)
    seed = (random.randrange(height), random.randrange(width))
    mask[seed] = True

    frontier = set()
    for dr, dc in DIRS4:
        nr = seed[0] + dr
        nc = seed[1] + dc
        if 0 <= nr < height and 0 <= nc < width:
            frontier.add((nr, nc))

    while int(mask.sum()) < target:
        nr, nc = random.choice(tuple(frontier))
        frontier.remove((nr, nc))
        mask[nr, nc] = True
        for dr, dc in DIRS4:
            rr = nr + dr
            cc = nc + dc
            if 0 <= rr < height and 0 <= cc < width and not mask[rr, cc]:
                frontier.add((rr, cc))

    return mask


def _coords_fit(occupied: np.ndarray, coords: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    unique = set(coords)
    if len(unique) != len(coords):
        return False

    for r, c in unique:
        if not (0 <= r < h and 0 <= c < w):
            return False
    for r, c in unique:
        r0 = max(0, r - 1)
        r1 = min(h, r + 2)
        c0 = max(0, c - 1)
        c1 = min(w, c + 2)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _place_pair(
    grid: np.ndarray,
    occupied: np.ndarray,
    axis_kind: str,
    axis_index: int,
    color: int,
) -> bool:
    h, w = grid.shape
    if axis_kind == "H":
        max_height = min(5, axis_index, h - axis_index - 1)
        if max_height < 1:
            return False
        for _ in range(PAIR_PLACEMENT_ATTEMPTS):
            height = random.randint(1, max_height)
            width = random.randint(2, min(6, w))
            if height * width < 2:
                continue
            max_gap = min(axis_index - height + 1, h - axis_index - height)
            if max_gap < 1:
                continue
            gap = random.randint(1, max_gap)
            top = axis_index - gap - height + 1
            left = random.randint(0, w - width)
            mask = _random_connected_mask(height, width)
            first = [(top + r, left + c) for r, c in np.argwhere(mask)]
            second = [_reflect_point(r, c, axis_kind, axis_index) for r, c in first]
            coords = first + second
            if not _coords_fit(occupied, coords):
                continue
            for r, c in coords:
                grid[r, c] = color
                occupied[r, c] = True
            return True
        return False

    max_width = min(5, axis_index, w - axis_index - 1)
    if max_width < 1:
        return False
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        height = random.randint(2, min(6, h))
        width = random.randint(1, max_width)
        if height * width < 2:
            continue
        max_gap = min(axis_index - width + 1, w - axis_index - width)
        if max_gap < 1:
            continue
        gap = random.randint(1, max_gap)
        top = random.randint(0, h - height)
        left = axis_index - gap - width + 1
        mask = _random_connected_mask(height, width)
        first = [(top + r, left + c) for r, c in np.argwhere(mask)]
        second = [_reflect_point(r, c, axis_kind, axis_index) for r, c in first]
        coords = first + second
        if not _coords_fit(occupied, coords):
            continue
        for r, c in coords:
            grid[r, c] = color
            occupied[r, c] = True
        return True
    return False


def _attempt_grid() -> np.ndarray | None:
    height = random.randint(*GRID_SIDE_RANGE)
    width = random.randint(*GRID_SIDE_RANGE)
    axis_kind = random.choice(("H", "V"))
    if axis_kind == "H":
        if height < 5:
            return None
        axis_index = random.randint(1, height - 2)
    else:
        if width < 5:
            return None
        axis_index = random.randint(1, width - 2)

    pair_cap = min(PAIR_COUNT_RANGE[1], len(OBJECT_COLORS), max(PAIR_COUNT_RANGE[0], (height * width) // 45))
    pair_count = random.randint(PAIR_COUNT_RANGE[0], pair_cap)
    colors = random.sample(OBJECT_COLORS, pair_count)

    grid = np.full((height, width), BG_COLOR, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for color in colors:
        if not _place_pair(grid, occupied, axis_kind, axis_index, color):
            return None

    try:
        inferred_kind, inferred_axis = _infer_axis(grid)
    except AssertionError:
        return None
    if inferred_kind != axis_kind or inferred_axis != axis_index:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-pair grid")
