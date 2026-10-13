import random
from collections import Counter

import numpy as np


BG = 0
RED = 2
YELLOW = 4
TEAL = 8
BLOCKER_COLORS = (RED, TEAL)
MAX_GENERATION_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        out.append(np.array(coords, dtype=int))
    return out


def crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    h, w = mask.shape
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = True
    return out


def find_axis(grid: np.ndarray, bg: int) -> tuple[str, int]:
    h, w = grid.shape
    candidates: list[tuple[str, int]] = []
    for r in range(h):
        if int(grid[r, 0]) != bg and int(grid[r, w - 1]) != bg and np.all(grid[r, 1 : w - 1] == bg):
            candidates.append(("h", r))
    for c in range(w):
        if int(grid[0, c]) != bg and int(grid[h - 1, c]) != bg and np.all(grid[1 : h - 1, c] == bg):
            candidates.append(("v", c))
    assert len(candidates) == 1, "expected exactly one clear boundary-to-boundary corridor"
    return candidates[0]


def mirror_coords(coords: np.ndarray, orient: str, axis: int) -> np.ndarray:
    mirrored = coords.copy()
    if orient == "h":
        mirrored[:, 0] = 2 * axis - mirrored[:, 0]
    else:
        mirrored[:, 1] = 2 * axis - mirrored[:, 1]
    return mirrored


def object_side(coords: np.ndarray, orient: str, axis: int) -> int:
    delta = coords[:, 0] - axis if orient == "h" else coords[:, 1] - axis
    assert np.all(delta != 0), "yellow objects cannot touch the corridor axis"
    if np.all(delta < 0):
        return -1
    if np.all(delta > 0):
        return 1
    raise AssertionError("yellow object cannot straddle the corridor axis")


def nearer_endpoint_color(grid: np.ndarray, coords: np.ndarray, orient: str, axis: int) -> int:
    h, w = grid.shape
    r0, c0, r1, c1 = bbox_coords(coords)
    if orient == "h":
        left_dist = c0
        right_dist = (w - 1) - c1
        assert left_dist != right_dist, "object must not be equally close to both corridor ends"
        return int(grid[axis, 0]) if left_dist < right_dist else int(grid[axis, w - 1])
    top_dist = r0
    bottom_dist = (h - 1) - r1
    assert top_dist != bottom_dist, "object must not be equally close to both corridor ends"
    return int(grid[0, axis]) if top_dist < bottom_dist else int(grid[h - 1, axis])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each yellow object across the unique clear boundary-to-boundary corridor and color the copy with the nearer endpoint color.

    Input structure
    - The background is black and the only source objects are yellow 4-connected sprites.
    - Red and teal pixels appear on the border and as internal blockers.
    - Exactly one full row or full column has non-background cells at both ends and only background cells inside it.
    - All yellow objects lie strictly on one side of that clear row or column.

    Rule
    - Treat the unique clear row or column as an implicit reflection axis.
    - Reflect every yellow object across that axis, preserving each object's shape and position relative to the axis.
    - Paint the reflected copy with the color of the corridor endpoint that is nearer along the corridor direction.
    - Keep every input pixel unchanged, including the empty corridor itself.

    Key insight
    - The important axis is not drawn as a colored separator; it is the only unobstructed background corridor whose colored endpoints also specify the reflected object's paint color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = get_bg(g)
    orient, axis = find_axis(g, bg)

    components = components_mask(g == YELLOW)
    assert components, "expected at least one yellow source object"
    components.sort(key=bbox_coords)

    sides = {object_side(coords, orient, axis) for coords in components}
    assert len(sides) == 1, "all yellow objects must lie on the same side of the corridor"

    out = g.copy()
    painted = np.zeros_like(g, dtype=bool)
    for coords in components:
        reflected = mirror_coords(coords, orient, axis)
        assert np.all(reflected >= 0), "reflected object must remain in bounds"
        assert np.all(reflected[:, 0] < g.shape[0]) and np.all(reflected[:, 1] < g.shape[1]), (
            "reflected object must remain in bounds"
        )
        assert np.all(g[reflected[:, 0], reflected[:, 1]] == bg), "mirror footprint must be background"
        assert not np.any(painted[reflected[:, 0], reflected[:, 1]]), "mirror footprints must be disjoint"
        color = nearer_endpoint_color(g, coords, orient, axis)
        out[reflected[:, 0], reflected[:, 1]] = color
        painted[reflected[:, 0], reflected[:, 1]] = True
    return out


def random_connected_shape() -> np.ndarray:
    box_h = random.randint(2, 5)
    box_w = random.randint(2, 6)
    min_cells = max(3, (box_h * box_w + 3) // 4)
    max_cells = max(min_cells, box_h * box_w - (1 if box_h * box_w > 4 else 0))
    target = random.randint(min_cells, max_cells)

    mask = np.zeros((box_h, box_w), dtype=bool)
    start = (random.randrange(box_h), random.randrange(box_w))
    mask[start] = True
    frontier = set()
    for cell in neighbors4(*start):
        if in_bounds(mask.shape, *cell):
            frontier.add(cell)

    while int(mask.sum()) < target:
        rr, cc = random.choice(tuple(frontier))
        frontier.remove((rr, cc))
        mask[rr, cc] = True
        for nr, nc in neighbors4(rr, cc):
            if in_bounds(mask.shape, nr, nc) and not mask[nr, nc]:
                frontier.add((nr, nc))

    return crop_mask(mask)


def endpoint_branch(c0: int, r0: int, shape: np.ndarray, orient: str, grid_shape: tuple[int, int]) -> int:
    h, w = grid_shape
    sh, sw = shape.shape
    if orient == "h":
        left_dist = c0
        right_dist = (w - 1) - (c0 + sw - 1)
        if left_dist == right_dist:
            return -1
        return 0 if left_dist < right_dist else 1
    top_dist = r0
    bottom_dist = (h - 1) - (r0 + sh - 1)
    if top_dist == bottom_dist:
        return -1
    return 0 if top_dist < bottom_dist else 1


def place_object(
    reserved_keepout: np.ndarray,
    reserved_bg: np.ndarray,
    orient: str,
    axis: int,
    source_side: int,
    branch: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    grid_shape = reserved_keepout.shape
    h, w = grid_shape
    shape = random_connected_shape()
    sh, sw = shape.shape
    gap = random.randint(1, 3)

    positions: list[tuple[int, int]] = []
    if orient == "h":
        row_values = range(1, axis - gap - sh + 1) if source_side < 0 else range(axis + gap + 1, h - sh)
        col_values = range(1, w - sw)
        for r0 in row_values:
            for c0 in col_values:
                if endpoint_branch(c0, r0, shape, orient, grid_shape) != branch:
                    continue
                positions.append((r0, c0))
    else:
        row_values = range(1, h - sh)
        col_values = range(1, axis - gap - sw + 1) if source_side < 0 else range(axis + gap + 1, w - sw)
        for r0 in row_values:
            for c0 in col_values:
                if endpoint_branch(c0, r0, shape, orient, grid_shape) != branch:
                    continue
                positions.append((r0, c0))

    random.shuffle(positions)
    shape_coords = np.argwhere(shape)
    for r0, c0 in positions:
        coords = shape_coords + np.array([r0, c0], dtype=int)
        reflected = mirror_coords(coords, orient, axis)
        if np.any(reflected < 0) or np.any(reflected[:, 0] >= h) or np.any(reflected[:, 1] >= w):
            continue

        obj_mask = np.zeros(grid_shape, dtype=bool)
        obj_mask[coords[:, 0], coords[:, 1]] = True
        ref_mask = np.zeros(grid_shape, dtype=bool)
        ref_mask[reflected[:, 0], reflected[:, 1]] = True

        if np.any(obj_mask & reserved_keepout) or np.any(ref_mask & reserved_keepout):
            continue
        if np.any(obj_mask & reserved_bg) or np.any(ref_mask & reserved_bg):
            continue

        keepout = dilate8(obj_mask | ref_mask)
        return obj_mask, ref_mask, keepout
    return None


def random_boundary_cell(edge: str, shape: tuple[int, int], forbidden: set[tuple[int, int]]) -> tuple[int, int] | None:
    h, w = shape
    if edge == "top":
        cells = [(0, c) for c in range(w)]
    elif edge == "bottom":
        cells = [(h - 1, c) for c in range(w)]
    elif edge == "left":
        cells = [(r, 0) for r in range(h)]
    else:
        cells = [(r, w - 1) for r in range(h)]
    cells = [cell for cell in cells if cell not in forbidden]
    return random.choice(cells) if cells else None


def corridor_candidates(grid: np.ndarray, bg: int) -> list[tuple[str, int]]:
    h, w = grid.shape
    out: list[tuple[str, int]] = []
    for r in range(h):
        if int(grid[r, 0]) != bg and int(grid[r, w - 1]) != bg and np.all(grid[r, 1 : w - 1] == bg):
            out.append(("h", r))
    for c in range(w):
        if int(grid[0, c]) != bg and int(grid[h - 1, c]) != bg and np.all(grid[1 : h - 1, c] == bg):
            out.append(("v", c))
    return out


def add_scatter(grid: np.ndarray, reserved_bg: np.ndarray, orient: str, axis: int) -> bool:
    h, w = grid.shape
    forbidden = set()
    if orient == "h":
        forbidden.update({(axis, 0), (axis, w - 1)})
    else:
        forbidden.update({(0, axis), (h - 1, axis)})

    min_extra = {
        "top": 1 if orient == "h" else 0,
        "bottom": 1 if orient == "h" else 0,
        "left": 1 if orient == "v" else 0,
        "right": 1 if orient == "v" else 0,
    }
    for edge in ("top", "bottom", "left", "right"):
        count = min_extra[edge] + random.randint(0, 2)
        for _ in range(count):
            cell = random_boundary_cell(edge, grid.shape, forbidden)
            if cell is None:
                return False
            forbidden.add(cell)
            grid[cell] = random.choice(BLOCKER_COLORS)

    blocker_forbidden = reserved_bg.copy()
    for r, c in np.argwhere(grid == YELLOW):
        blocker_forbidden[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True

    open_cells = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if int(grid[r, c]) == BG and not blocker_forbidden[r, c]
    ]
    random.shuffle(open_cells)
    blocker_count = random.randint(4, 9)
    if len(open_cells) < blocker_count:
        return False
    for r, c in open_cells[:blocker_count]:
        grid[r, c] = random.choice(BLOCKER_COLORS)
    return True


def invalidate_extra_corridors(grid: np.ndarray, reserved_bg: np.ndarray, target_axis: tuple[str, int]) -> bool:
    h, w = grid.shape
    for _ in range(20):
        changed = False
        for orient, axis in corridor_candidates(grid, BG):
            if (orient, axis) == target_axis:
                continue
            if orient == "h":
                choices = [(axis, c) for c in range(1, w - 1) if int(grid[axis, c]) == BG and not reserved_bg[axis, c]]
            else:
                choices = [(r, axis) for r in range(1, h - 1) if int(grid[r, axis]) == BG and not reserved_bg[r, axis]]
            if not choices:
                return False
            rr, cc = random.choice(choices)
            grid[rr, cc] = random.choice(BLOCKER_COLORS)
            changed = True
        if not changed:
            break
    return corridor_candidates(grid, BG) == [target_axis]


def gen_attempt() -> np.ndarray | None:
    h = random.randint(14, 22)
    w = random.randint(14, 22)
    orient = random.choice(("h", "v"))
    if orient == "h":
        axis = random.randint(5, h - 6)
    else:
        axis = random.randint(5, w - 6)

    source_side = random.choice((-1, 1))
    object_count = random.randint(2, 4)
    branches = [0, 1] + [random.randint(0, 1) for _ in range(object_count - 2)]
    random.shuffle(branches)

    keepout = np.zeros((h, w), dtype=bool)
    reserved_bg = np.zeros((h, w), dtype=bool)
    if orient == "h":
        reserved_bg[axis, 1 : w - 1] = True
    else:
        reserved_bg[1 : h - 1, axis] = True

    objects: list[np.ndarray] = []
    for branch in branches:
        placed = None
        for _ in range(80):
            placed = place_object(keepout, reserved_bg, orient, axis, source_side, branch)
            if placed is not None:
                break
        if placed is None:
            return None
        obj_mask, ref_mask, new_keepout = placed
        keepout |= new_keepout
        reserved_bg |= ref_mask
        objects.append(obj_mask)

    grid = np.zeros((h, w), dtype=int)
    for obj_mask in objects:
        grid[obj_mask] = YELLOW

    endpoint_colors = random.sample(list(BLOCKER_COLORS), 2)
    if orient == "h":
        grid[axis, 0] = endpoint_colors[0]
        grid[axis, w - 1] = endpoint_colors[1]
    else:
        grid[0, axis] = endpoint_colors[0]
        grid[h - 1, axis] = endpoint_colors[1]

    if not add_scatter(grid, reserved_bg, orient, axis):
        return None
    if not invalidate_extra_corridors(grid, reserved_bg, (orient, axis)):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
