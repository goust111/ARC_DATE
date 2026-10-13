from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG_COLOR = 0
OUTPUT_COLOR = 5
FRAME_COLORS = (1, 2, 3, 4, 7, 8)

GRID_HEIGHT_RANGE = (14, 18)
GRID_WIDTH_RANGE = (14, 18)
FRAME_COUNT_OPTIONS = (4, 6)
FRAME_HEIGHT_RANGE = (5, 10)
FRAME_WIDTH_RANGE = (5, 10)
REQUIRED_PREV_OVERLAPS = (2, 4)
ALLOWED_PRIOR_OVERLAPS = (0, 2, 4)
PLACEMENT_TRIES = 250
GENERATION_TRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    assert r0 < r1 and c0 < c1
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _frame_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(exclude)
    return sorted(colors)


def _unique_layer_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for lower, higher in edges:
        assert lower != higher
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap constraints must determine one layer order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return order


def _infer_frames_and_order(grid: np.ndarray) -> tuple[list[int], dict[int, tuple[int, int, int, int]], dict[int, np.ndarray]]:
    colors = _colors_present(grid, exclude={BG_COLOR})
    assert colors, "input must contain at least one frame"

    boxes: dict[int, tuple[int, int, int, int]] = {}
    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = grid == color
        box = _bbox(visible)
        assert box is not None, "each frame color must remain visible"
        mask = _frame_mask(grid.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame"
        boxes[color] = box
        frame_masks[color] = mask

    edges: set[tuple[int, int]] = set()
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(grid[r, c])
            assert top in covering, "an overlap cell must display one of the covering frames"
            for color in covering:
                if color != top:
                    edges.add((color, top))

    assert edges, "at least one overlap is required to infer the stack"
    order = _unique_layer_order(colors, edges)
    return order, boxes, frame_masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the frame stack, keep the even-depth layers, and redraw them as grey frames.

    Input structure:
    - A black background contains several distinct one-cell-thick hollow rectangles.
    - Each foreground color belongs to exactly one axis-aligned rectangular frame, but parts of its border may be hidden by higher layers.
    - At shared border cells, only the topmost frame color is visible.

    Rule:
    - For each foreground color, use the visible pixels' bounding box to reconstruct that frame's full rectangular border.
    - On every cell covered by multiple reconstructed frames, the visible color identifies the highest of those frames; these constraints determine the unique bottom-to-top layer order.
    - Erase the whole canvas, then redraw only layers 0, 2, 4, ... from that order as complete grey frames on black.

    Key insight:
    - The visible fragments encode both the hidden geometry of each frame and the global occlusion order, so the surviving even layers can be reconstructed after ordering the stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    order, _boxes, frame_masks = _infer_frames_and_order(g)
    out = np.full(g.shape, BG_COLOR, dtype=int)
    for depth, color in enumerate(order):
        if depth % 2 == 0:
            out[frame_masks[color]] = OUTPUT_COLOR
    return out


def _random_frame_size(limit: int, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    hi = min(hi, limit)
    assert lo <= hi
    return random.randint(lo, hi)


def _sample_layer_order(count: int) -> list[int]:
    order = random.sample(FRAME_COLORS, count)
    random.shuffle(order)
    return order


def _place_first_frame(rows: int, cols: int) -> tuple[int, int, int, int]:
    height = _random_frame_size(rows - 1, FRAME_HEIGHT_RANGE)
    width = _random_frame_size(cols - 1, FRAME_WIDTH_RANGE)
    r0 = random.randint(0, rows - height)
    c0 = random.randint(0, cols - width)
    return r0, c0, r0 + height - 1, c0 + width - 1


def _place_next_frame(
    rows: int,
    cols: int,
    prev_box: tuple[int, int, int, int],
    prior_cell_sets: list[set[tuple[int, int]]],
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    pr0, pc0, pr1, pc1 = prev_box
    for _ in range(PLACEMENT_TRIES):
        height = _random_frame_size(rows - 1, FRAME_HEIGHT_RANGE)
        width = _random_frame_size(cols - 1, FRAME_WIDTH_RANGE)

        r_lo = max(0, pr0 - height + 1)
        r_hi = min(rows - height, pr1)
        c_lo = max(0, pc0 - width + 1)
        c_hi = min(cols - width, pc1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        box = (r0, c0, r0 + height - 1, c0 + width - 1)
        cells = _frame_cells(box)

        if len(cells & prior_cell_sets[-1]) not in REQUIRED_PREV_OVERLAPS:
            continue
        if any(len(cells & older) not in ALLOWED_PRIOR_OVERLAPS for older in prior_cell_sets[:-1]):
            continue
        return box, cells
    return None


def _place_frame_stack(rows: int, cols: int, count: int) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]]] | None:
    first_box = _place_first_frame(rows, cols)
    boxes = [first_box]
    cell_sets = [_frame_cells(first_box)]
    for _ in range(1, count):
        placed = _place_next_frame(rows, cols, boxes[-1], cell_sets)
        if placed is None:
            return None
        box, cells = placed
        boxes.append(box)
        cell_sets.append(cells)
    return boxes, cell_sets


def _render_input(
    rows: int,
    cols: int,
    layer_order: list[int],
    cell_sets: list[set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((rows, cols), BG_COLOR, dtype=int)
    for color, cells in zip(layer_order, cell_sets):
        rs, cs = zip(*cells)
        grid[rs, cs] = color
    return grid


def _render_expected_output(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = np.full(shape, BG_COLOR, dtype=int)
    for depth, box in enumerate(boxes):
        if depth % 2 == 0:
            out[_frame_mask(shape, box)] = OUTPUT_COLOR
    return out


def _all_sides_visible(grid: np.ndarray, color: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    visible = grid == color
    return (
        bool(np.any(visible[r0, c0 : c1 + 1]))
        and bool(np.any(visible[r1, c0 : c1 + 1]))
        and bool(np.any(visible[r0 : r1 + 1, c0]))
        and bool(np.any(visible[r0 : r1 + 1, c1]))
    )


def _input_is_valid(grid: np.ndarray, layer_order: list[int], boxes: list[tuple[int, int, int, int]]) -> bool:
    for color, box in zip(layer_order, boxes):
        visible_box = _bbox(grid == color)
        if visible_box != box:
            return False
        if not _all_sides_visible(grid, color, box):
            return False

    try:
        inferred_order, _boxes, _masks = _infer_frames_and_order(grid)
    except Exception:
        return False
    return inferred_order == layer_order


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*GRID_HEIGHT_RANGE)
    cols = random.randint(*GRID_WIDTH_RANGE)
    count = random.choice(FRAME_COUNT_OPTIONS)
    placed = _place_frame_stack(rows, cols, count)
    if placed is None:
        return None

    boxes, cell_sets = placed
    layer_order = _sample_layer_order(count)
    inp = _render_input(rows, cols, layer_order, cell_sets)
    if not _input_is_valid(inp, layer_order, boxes):
        return None

    expected = _render_expected_output(inp.shape, boxes)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    color_counts = Counter(map(int, inp.flatten()))
    if color_counts[BG_COLOR] <= max(color_counts[color] for color in layer_order):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated retries")
