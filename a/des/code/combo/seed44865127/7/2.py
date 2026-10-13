from __future__ import annotations

import random

import numpy as np


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "expected a unique dominant background color"
    return int(winners[0])


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    assert r0 < r1 and c0 < c1, "frame bbox must have positive height and width"

    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _unique_layer_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for lower, higher in edges:
        assert lower != higher, "layer constraint cannot be reflexive"
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap constraints must determine a unique stacking order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the bottom-to-top color order of overlapping rectangular frame layers.

    Each foreground color belongs to one 1-pixel-thick rectangular frame. Only the
    topmost frame color is visible at overlap cells. Reconstruct each frame from
    the bounding box of its visible pixels, infer pairwise "below -> above"
    relations from overlap cells, and return the unique bottom-to-top order as a
    single-column grid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _dominant_color(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert colors, "expected at least one foreground frame"

    masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == color
        box = _bbox(visible)
        assert box is not None, "every foreground color must be visible"
        mask = _frame_mask(g.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        masks[color] = mask

    edges: set[tuple[int, int]] = set()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            covering = [color for color in colors if masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "every overlap cell must show one of the covering frames"
            for lower in covering:
                if lower != top:
                    edges.add((lower, top))

    if len(colors) == 1:
        order = colors
    else:
        assert edges, "multiple frames need overlap constraints"
        order = _unique_layer_order(colors, edges)

    return np.array(order, dtype=int).reshape(-1, 1)


# --------------- generator ---------------

GRID_SIDE_RANGE = (14, 24)
FRAME_COUNT_RANGE = (2, 5)
FRAME_SIDE_MIN = 4
FRAME_OVERLAP_CHOICES = (2, 4)
EARLIER_OVERLAP_CHOICES = (0, 2, 4)
PLACEMENT_TRIES = 220
GENERATION_TRIES = 500


def _frame_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _random_frame_size(height: int, width: int) -> tuple[int, int]:
    max_h = min(height - 2, max(FRAME_SIDE_MIN, (3 * height) // 4))
    max_w = min(width - 2, max(FRAME_SIDE_MIN, (3 * width) // 4))
    h = random.randint(FRAME_SIDE_MIN, max_h)
    w = random.randint(FRAME_SIDE_MIN, max_w)
    return h, w


def _place_first_frame(height: int, width: int) -> tuple[int, int, int, int]:
    fh, fw = _random_frame_size(height, width)
    r0 = random.randint(1, height - fh - 1)
    c0 = random.randint(1, width - fw - 1)
    return r0, c0, r0 + fh - 1, c0 + fw - 1


def _place_next_frame(
    height: int,
    width: int,
    prev_box: tuple[int, int, int, int],
    earlier_sets: list[set[tuple[int, int]]],
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    pr0, pc0, pr1, pc1 = prev_box

    for _ in range(PLACEMENT_TRIES):
        fh, fw = _random_frame_size(height, width)
        r_lo = max(1, pr0 - fh + 1)
        r_hi = min(height - fh - 1, pr1)
        c_lo = max(1, pc0 - fw + 1)
        c_hi = min(width - fw - 1, pc1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        box = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        cells = _frame_cells(box)

        if len(cells & earlier_sets[-1]) not in FRAME_OVERLAP_CHOICES:
            continue
        if any(len(cells & other) not in EARLIER_OVERLAP_CHOICES for other in earlier_sets[:-1]):
            continue
        return box, cells

    return None


def _place_frame_stack(height: int, width: int, n_frames: int) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]]] | None:
    first = _place_first_frame(height, width)
    boxes = [first]
    cell_sets = [_frame_cells(first)]

    for _ in range(1, n_frames):
        placed = _place_next_frame(height, width, boxes[-1], cell_sets)
        if placed is None:
            return None
        box, cells = placed
        boxes.append(box)
        cell_sets.append(cells)

    return boxes, cell_sets


def _overlap_edges(order: list[int], cell_sets: list[set[tuple[int, int]]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if cell_sets[i] & cell_sets[j]:
                edges.add((order[i], order[j]))
    return edges


def _render_scene(height: int, width: int, bg: int, order: list[int], cell_sets: list[set[tuple[int, int]]]) -> np.ndarray:
    grid = np.full((height, width), bg, dtype=int)
    for color, cells in zip(order, cell_sets):
        rows, cols = zip(*cells)
        grid[rows, cols] = int(color)
    return grid


def _visible_boxes_match(grid: np.ndarray, order: list[int], boxes: list[tuple[int, int, int, int]]) -> bool:
    for color, box in zip(order, boxes):
        if _bbox(grid == int(color)) != box:
            return False
    return True


def _background_is_unique_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_count = int(counts[np.where(values == bg)[0][0]])
    return sum(int(count == bg_count) for count in counts) == 1 and all(bg_count > int(count) for value, count in zip(values, counts) if int(value) != bg)


def _solver_matches(grid: np.ndarray, order: list[int]) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    expected = np.array(order, dtype=int).reshape(-1, 1)
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    n_frames = random.randint(*FRAME_COUNT_RANGE)
    min_side = max(GRID_SIDE_RANGE[0], 9 + 2 * n_frames)
    height = random.randint(min_side, GRID_SIDE_RANGE[1])
    width = random.randint(min_side, GRID_SIDE_RANGE[1])

    bg = random.randint(0, 9)
    palette = [color for color in range(10) if color != bg]
    colors = random.sample(palette, n_frames)
    random.shuffle(colors)

    placed = _place_frame_stack(height, width, n_frames)
    if placed is None:
        return None
    boxes, cell_sets = placed

    edges = _overlap_edges(colors, cell_sets)
    try:
        topo = _unique_layer_order(colors, edges)
    except AssertionError:
        return None
    if topo != colors:
        return None

    grid = _render_scene(height, width, bg, colors, cell_sets)
    if not _visible_boxes_match(grid, colors, boxes):
        return None
    if not _background_is_unique_dominant(grid, bg):
        return None
    if not _solver_matches(grid, colors):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
