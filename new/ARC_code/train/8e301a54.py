import numpy as np

from arc_utils import find_components, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every foreground object straight downward by a number of rows equal to its own pixel count.

    Input structure:
    - The grid has one dominant background color.
    - The foreground is made of separated 4-connected monochromatic objects.
    - Objects may move partly below the bottom edge of the grid.

    Rule:
    - Clear the canvas to the background.
    - For each foreground object, compute its area in pixels.
    - Redraw the object at the same columns and that many rows lower, clipping pixels below the grid.

    Key insight:
    - The downward displacement is the object's own area; color and shape only identify which pixels move together.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    assert objects, "expected at least one foreground object"

    out = np.full_like(g, bg)
    painted = np.zeros(g.shape, dtype=bool)
    H, W = g.shape

    for obj in objects:
        coords = np.asarray(obj["coords"], dtype=int)
        area = int(obj["area"])
        color = int(obj["color"])
        assert coords.ndim == 2 and coords.shape == (area, 2), "component coords must match area"
        assert np.all(g[coords[:, 0], coords[:, 1]] == color), "component must be monochromatic"

        target = coords.copy()
        target[:, 0] += area
        assert np.all((0 <= target[:, 1]) & (target[:, 1] < W)), "columns are preserved in bounds"

        keep = target[:, 0] < H
        target = target[keep]
        if target.size == 0:
            continue

        rr = target[:, 0]
        cc = target[:, 1]
        assert not painted[rr, cc].any(), "moved objects must not overlap"
        out[rr, cc] = color
        painted[rr, cc] = True

    return out

# --------------- generator (att2) ---------------

GRID_HEIGHT, GRID_WIDTH = 10, 10
BACKGROUND = 7
OBJECT_COLORS = [2, 5, 9]
OBJECT_COUNT = 3
AREA_MIN, AREA_MAX = 1, 5
ROTATION_COUNT = 4
MIRROR_PROB = 0.5
MAX_CLIPPED_OBJECTS = 2
NEIGHBORS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 300


def _gen_norm_coords(coords):
    arr = np.asarray(list(coords), dtype=int)
    arr[:, 0] -= arr[:, 0].min()
    arr[:, 1] -= arr[:, 1].min()
    return arr


def _gen_transform_coords(coords):
    arr = np.asarray(list(coords), dtype=int).copy()
    for _ in range(random.randrange(ROTATION_COUNT)):
        arr = np.column_stack((arr[:, 1], -arr[:, 0]))
    if random.random() < MIRROR_PROB:
        arr[:, 1] *= -1
    return _gen_norm_coords(map(tuple, arr))


def _gen_polyomino(area):
    cells = {(0, 0)}
    frontier = {(0, 1), (1, 0), (0, -1), (-1, 0)}
    while len(cells) < area:
        cell = random.choice(tuple(frontier))
        cells.add(cell)
        frontier.remove(cell)
        r, c = cell
        for dr, dc in NEIGHBORS_4:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in cells:
                frontier.add((nr, nc))
    return _gen_transform_coords(cells)


def _gen_output_bounds(coords, area, height):
    shape_h = int(coords[:, 0].max()) + 1
    shape_w = int(coords[:, 1].max()) + 1
    min_r = area + 1
    max_r = min(height - 1, height - shape_h + area)
    return min_r, max_r, shape_h, shape_w


def _gen_clip_indices(shapes, areas):
    clip_capable = []
    for i, (coords, area) in enumerate(zip(shapes, areas)):
        min_r, max_r, shape_h, _ = _gen_output_bounds(coords, area, GRID_HEIGHT)
        if max_r >= max(min_r, GRID_HEIGHT - shape_h + 1):
            clip_capable.append(i)
    clip_count = random.randint(0, min(MAX_CLIPPED_OBJECTS, len(clip_capable)))
    return set(random.sample(clip_capable, clip_count))


def _gen_candidate_positions(coords, area, out_used, in_used, want_clip):
    height, width = out_used.shape
    min_r, max_r, _, shape_w = _gen_output_bounds(coords, area, height)
    if min_r > max_r:
        return []

    candidates = []
    for out_r in range(min_r, max_r + 1):
        target = coords + np.array([out_r, 0])
        clipped = bool(np.any(target[:, 0] >= height))
        if clipped != want_clip:
            continue
        if not np.any(target[:, 0] < height):
            continue
        for out_c in range(0, width - shape_w + 1):
            target = coords + np.array([out_r, out_c])
            visible = target[target[:, 0] < height]
            inp = coords + np.array([out_r - area, out_c])
            if np.any(inp[:, 0] <= 0) or np.any(inp[:, 0] >= height):
                continue
            if out_used[visible[:, 0], visible[:, 1]].any():
                continue
            if in_used[inp[:, 0], inp[:, 1]].any():
                continue
            candidates.append((out_r, out_c, visible, inp))
    return candidates


def _gen_place_objects(colors, areas, shapes, clip_indices):
    expected = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND, dtype=int)
    inp = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND, dtype=int)
    out_used = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)
    in_used = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=bool)

    order = list(range(OBJECT_COUNT))
    random.shuffle(order)
    for i in order:
        coords, area, color = shapes[i], areas[i], colors[i]
        candidates = _gen_candidate_positions(
            coords, area, out_used, in_used, i in clip_indices
        )
        if not candidates:
            return None
        _, _, visible, input_coords = random.choice(candidates)
        expected[visible[:, 0], visible[:, 1]] = color
        out_used[visible[:, 0], visible[:, 1]] = True
        inp[input_coords[:, 0], input_coords[:, 1]] = color
        in_used[input_coords[:, 0], input_coords[:, 1]] = True

    return inp, expected


def _gen_attempt():
    colors = random.sample(OBJECT_COLORS, OBJECT_COUNT)
    areas = [random.randint(AREA_MIN, AREA_MAX) for _ in range(OBJECT_COUNT)]
    shapes = [_gen_polyomino(area) for area in areas]
    clip_indices = _gen_clip_indices(shapes, areas)
    placed = _gen_place_objects(colors, areas, shapes, clip_indices)
    if placed is None:
        return None

    inp, expected = placed
    if not np.all(inp[0] == BACKGROUND):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 8e301a54 input")
