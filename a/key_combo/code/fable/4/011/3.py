from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _colors_present(grid: np.ndarray, exclude: set[int]) -> list[int]:
    return [int(color) for color in np.unique(grid) if int(color) not in exclude]


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    assert r0 < r1 and c0 < c1, "each frame must span at least two rows and two columns"

    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _frame_cells(bbox: BBox) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _unique_layer_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for lower, higher in edges:
        assert lower != higher, "a frame cannot be above itself"
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap constraints must determine a unique full layer order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden frame order, rotate frame colors upward by one layer, and redraw the stack.

    Input structure:
    - The background is a single color, and every other color belongs to one one-cell-thick rectangular frame.
    - Frames overlap, so cells covered by multiple reconstructed borders show only the color of the highest visible frame.
    - Each frame still has enough visible border pixels that the bounding box of its visible color recovers the full frame.

    Rule:
    - Reconstruct every frame as the hollow border of the bounding box of its visible pixels.
    - At each cell where multiple reconstructed frames pass, treat the visible color as the topmost covering frame and recover the unique bottom-to-top order.
    - Keep every frame in the same place and same order, but recolor each frame with the color of the frame directly above it; the top frame takes the bottom frame's color.

    Key insight:
    - The geometry stays fixed; only the colors move around the hidden depth cycle, so the main task is inferring layer order from occluded frame intersections.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = _colors_present(g, exclude={bg})
    assert len(colors) >= 2, "expected at least two foreground frames"

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = (g == color)
        bbox = _bbox(visible)
        assert bbox is not None, "every frame color must remain visible"
        mask = _frame_mask(g.shape, bbox)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed rectangular border"
        frame_masks[color] = mask

    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "an overlap cell must show one of the covering frames"
            for color in covering:
                if color != top:
                    edges.add((color, top))

    assert edges, "frames must overlap so their order can be recovered"
    layer_order = _unique_layer_order(colors, edges)

    out = np.full_like(g, bg)
    n = len(layer_order)
    for index, frame_color in enumerate(layer_order):
        rotated_color = int(layer_order[(index + 1) % n])
        out[frame_masks[frame_color]] = rotated_color
    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (14, 26)
GRID_WIDTH_RANGE = (14, 28)
FRAME_COUNT_CHOICES = (3, 4, 5)
FRAME_COUNT_WEIGHTS = (4, 4, 2)
OUTER_HEIGHT_RANGE = (4, 14)
OUTER_WIDTH_RANGE = (4, 16)
REQUIRED_PREV_OVERLAPS = {2, 4}
ALLOWED_OLDER_OVERLAPS = {0, 2, 4}
PLACEMENT_TRIES = 250
GENERATION_TRIES = 500


def _sample_frame_size(height: int, width: int) -> tuple[int, int] | None:
    max_h = min(OUTER_HEIGHT_RANGE[1], height - 1)
    max_w = min(OUTER_WIDTH_RANGE[1], width - 1)
    if max_h < OUTER_HEIGHT_RANGE[0] or max_w < OUTER_WIDTH_RANGE[0]:
        return None
    outer_h = random.randint(OUTER_HEIGHT_RANGE[0], max_h)
    outer_w = random.randint(OUTER_WIDTH_RANGE[0], max_w)
    return outer_h, outer_w


def _place_first_frame(height: int, width: int) -> tuple[BBox, set[tuple[int, int]]] | None:
    size = _sample_frame_size(height, width)
    if size is None:
        return None
    outer_h, outer_w = size
    r0 = random.randint(0, height - outer_h)
    c0 = random.randint(0, width - outer_w)
    bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
    return bbox, _frame_cells(bbox)


def _place_next_frame(
    height: int,
    width: int,
    previous_bbox: BBox,
    prior_cell_sets: list[set[tuple[int, int]]],
) -> tuple[BBox, set[tuple[int, int]]] | None:
    pr0, pc0, pr1, pc1 = previous_bbox

    for _ in range(PLACEMENT_TRIES):
        size = _sample_frame_size(height, width)
        if size is None:
            return None
        outer_h, outer_w = size

        row_lo = max(0, pr0 - outer_h + 2)
        row_hi = min(height - outer_h, pr1)
        col_lo = max(0, pc0 - outer_w + 2)
        col_hi = min(width - outer_w, pc1)
        if row_lo > row_hi or col_lo > col_hi:
            continue

        r0 = random.randint(row_lo, row_hi)
        c0 = random.randint(col_lo, col_hi)
        bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
        cells = _frame_cells(bbox)

        if len(cells & prior_cell_sets[-1]) not in REQUIRED_PREV_OVERLAPS:
            continue
        if any(len(cells & older) not in ALLOWED_OLDER_OVERLAPS for older in prior_cell_sets[:-1]):
            continue
        return bbox, cells

    return None


def _place_frame_stack(
    height: int,
    width: int,
    frame_count: int,
) -> tuple[list[BBox], list[set[tuple[int, int]]]] | None:
    first = _place_first_frame(height, width)
    if first is None:
        return None

    first_bbox, first_cells = first
    bboxes = [first_bbox]
    cell_sets = [first_cells]

    for _ in range(1, frame_count):
        placed = _place_next_frame(height, width, bboxes[-1], cell_sets)
        if placed is None:
            return None
        bbox, cells = placed
        bboxes.append(bbox)
        cell_sets.append(cells)

    return bboxes, cell_sets


def _render_stack(
    shape: tuple[int, int],
    bg: int,
    layer_order: list[int],
    frame_cell_sets: list[set[tuple[int, int]]],
    *,
    rotate_colors: bool,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    count = len(layer_order)
    for index, cells in enumerate(frame_cell_sets):
        color = layer_order[(index + 1) % count] if rotate_colors else layer_order[index]
        rows, cols = zip(*cells)
        out[rows, cols] = int(color)
    return out


def _visible_bboxes_match(grid: np.ndarray, layer_order: list[int], bboxes: list[BBox]) -> bool:
    for color, bbox in zip(layer_order, bboxes):
        if _bbox(grid == color) != bbox:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    bg = random.randint(0, 9)

    frame_count = random.choices(FRAME_COUNT_CHOICES, weights=FRAME_COUNT_WEIGHTS, k=1)[0]
    layer_order = random.sample([color for color in range(10) if color != bg], frame_count)

    placed = _place_frame_stack(height, width, frame_count)
    if placed is None:
        return None
    bboxes, cell_sets = placed

    inp = _render_stack((height, width), bg, layer_order, cell_sets, rotate_colors=False)
    if _get_bg(inp) != bg:
        return None
    if not _visible_bboxes_match(inp, layer_order, bboxes):
        return None

    expected = _render_stack((height, width), bg, layer_order, cell_sets, rotate_colors=True)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_TRIES} attempts")
