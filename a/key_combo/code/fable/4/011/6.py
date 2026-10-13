from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    assert r0 < r1 and c0 < c1, "each frame must have positive height and width"

    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _frame_cells(box: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _unique_layer_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}

    for lower, higher in edges:
        assert lower != higher, "layer constraint cannot point to itself"
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap constraints must determine a unique layer order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1

    return order


def _hidden_counts(order: list[int], frame_masks: dict[int, np.ndarray]) -> dict[int, int]:
    hidden: dict[int, int] = {}
    higher_union = np.zeros_like(next(iter(frame_masks.values())), dtype=bool)
    for color in reversed(order):
        mask = frame_masks[color]
        hidden[color] = int(np.count_nonzero(mask & higher_union))
        higher_union |= mask
    return hidden


def _build_readout(order: list[int], hidden: dict[int, int]) -> np.ndarray:
    values: list[int] = []
    for color in order:
        values.extend([color] * hidden[color])
    return np.array(values, dtype=int).reshape(-1, 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct stacked rectangular frames and output each color as many times as its hidden border pixels.

    Input structure:
    - The background contains several one-cell-thick monochrome rectangular frames.
    - Frames may overlap, and each overlap cell shows only the color of the higher frame.
    - Each frame keeps enough visible border pixels that its full bounding box is recoverable.

    Rule:
    - Reconstruct every frame as the border of the bounding box of its visible pixels.
    - Use overlap cells to recover the unique bottom-to-top layer order.
    - For each frame, count how many of its border cells are covered by any higher frame, then output a single column whose bottom-to-top color segments have exactly those lengths; frames with zero hidden cells contribute nothing.

    Key insight:
    - The answer counts pixels that are missing from the input, so the solver must first restore the complete frames and their occlusion order before any counting becomes possible.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) >= 2, "expected at least two overlapping frames"

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == color
        box = _bbox(visible)
        assert box is not None, "each frame color must remain visible"
        mask = _frame_mask(g.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        frame_masks[color] = mask

    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "each overlap cell must show one covering frame"
            for color in covering:
                if color != top:
                    edges.add((color, top))

    assert edges, "overlaps are required to determine the order"
    order = _unique_layer_order(colors, edges)
    hidden = _hidden_counts(order, frame_masks)
    return _build_readout(order, hidden)


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (16, 24)
GRID_WIDTH_RANGE = (16, 24)
FRAME_COUNT_CHOICES = (3, 4, 5)
FRAME_SIZE_MIN = 4
FRAME_SIZE_MAX = 11
GRID_BORDER = 1

REQUIRED_PREV_OVERLAPS = (2, 4)
ALLOWED_OLDER_OVERLAPS = (0, 2, 4)
MAX_OLDER_OVERLAP_FRAMES = 1
MAX_NEW_TOTAL_OVERLAPS = 8

PLACEMENT_TRIES = 600
GENERATION_TRIES = 400
COLOR_POOL = tuple(range(10))


def _random_frame_size(height: int, width: int) -> tuple[int, int]:
    max_h = min(FRAME_SIZE_MAX, height - 2 * GRID_BORDER)
    max_w = min(FRAME_SIZE_MAX, width - 2 * GRID_BORDER)
    return random.randint(FRAME_SIZE_MIN, max_h), random.randint(FRAME_SIZE_MIN, max_w)


def _sample_layer_order(bg: int) -> list[int]:
    count = random.choice(FRAME_COUNT_CHOICES)
    return random.sample([color for color in COLOR_POOL if color != bg], count)


def _place_first_frame(height: int, width: int) -> tuple[BBox, set[tuple[int, int]]]:
    frame_h, frame_w = _random_frame_size(height, width)
    r0 = random.randint(GRID_BORDER, height - GRID_BORDER - frame_h)
    c0 = random.randint(GRID_BORDER, width - GRID_BORDER - frame_w)
    box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
    return box, _frame_cells(box)


def _place_next_frame(
    height: int,
    width: int,
    prev_box: BBox,
    prior_cell_sets: list[set[tuple[int, int]]],
) -> tuple[BBox, set[tuple[int, int]]] | None:
    pr0, pc0, pr1, pc1 = prev_box
    for _ in range(PLACEMENT_TRIES):
        frame_h, frame_w = _random_frame_size(height, width)
        lo_r = max(GRID_BORDER, pr0 - frame_h + 2)
        hi_r = min(height - GRID_BORDER - frame_h, pr1 - 1)
        lo_c = max(GRID_BORDER, pc0 - frame_w + 2)
        hi_c = min(width - GRID_BORDER - frame_w, pc1 - 1)
        if lo_r > hi_r or lo_c > hi_c:
            continue

        r0 = random.randint(lo_r, hi_r)
        c0 = random.randint(lo_c, hi_c)
        box = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        cells = _frame_cells(box)

        prev_overlap = len(cells & prior_cell_sets[-1])
        older_overlaps = [len(cells & older) for older in prior_cell_sets[:-1]]
        if prev_overlap not in REQUIRED_PREV_OVERLAPS:
            continue
        if any(overlap not in ALLOWED_OLDER_OVERLAPS for overlap in older_overlaps):
            continue
        if sum(overlap > 0 for overlap in older_overlaps) > MAX_OLDER_OVERLAP_FRAMES:
            continue
        if prev_overlap + sum(older_overlaps) > MAX_NEW_TOTAL_OVERLAPS:
            continue
        return box, cells
    return None


def _place_frame_stack(height: int, width: int, count: int) -> tuple[list[BBox], list[set[tuple[int, int]]]] | None:
    first_box, first_cells = _place_first_frame(height, width)
    boxes = [first_box]
    cell_sets = [first_cells]

    for _ in range(1, count):
        placed = _place_next_frame(height, width, boxes[-1], cell_sets)
        if placed is None:
            return None
        box, cells = placed
        boxes.append(box)
        cell_sets.append(cells)

    return boxes, cell_sets


def _render_frames(
    shape: tuple[int, int],
    bg: int,
    layer_order: list[int],
    frame_cell_sets: list[set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for color, cells in zip(layer_order, frame_cell_sets):
        rows, cols = zip(*cells)
        grid[rows, cols] = color
    return grid


def _visible_bboxes_match(grid: np.ndarray, layer_order: list[int], boxes: list[BBox]) -> bool:
    for color, box in zip(layer_order, boxes):
        visible_box = _bbox(grid == color)
        if visible_box != box:
            return False
    return True


def _overlap_edges(layer_order: list[int], frame_cell_sets: list[set[tuple[int, int]]]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for lower_index, lower_color in enumerate(layer_order):
        for higher_index in range(lower_index + 1, len(layer_order)):
            if frame_cell_sets[lower_index] & frame_cell_sets[higher_index]:
                edges.add((lower_color, layer_order[higher_index]))
    return edges


def _expected_output(shape: tuple[int, int], layer_order: list[int], boxes: list[BBox]) -> np.ndarray:
    frame_masks = {color: _frame_mask(shape, box) for color, box in zip(layer_order, boxes)}
    hidden = _hidden_counts(layer_order, frame_masks)
    return _build_readout(layer_order, hidden)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg = random.choice(COLOR_POOL)
    layer_order = _sample_layer_order(bg)

    placed = _place_frame_stack(height, width, len(layer_order))
    if placed is None:
        return None
    boxes, frame_cell_sets = placed

    edges = _overlap_edges(layer_order, frame_cell_sets)
    if _unique_layer_order(layer_order, edges) != layer_order:
        return None

    grid = _render_frames((height, width), bg, layer_order, frame_cell_sets)
    if _get_bg(grid) != bg:
        return None
    if not _visible_bboxes_match(grid, layer_order, boxes):
        return None

    expected = _expected_output((height, width), layer_order, boxes)
    if expected.size == 0 or expected.shape[0] > 30:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
