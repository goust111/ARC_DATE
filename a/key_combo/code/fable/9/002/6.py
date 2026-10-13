from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
COLOR_POOL = tuple(range(1, 10))

GRID_ROWS_RANGE = (18, 28)
GRID_COLS_RANGE = (18, 28)
FRAME_COUNT_RANGE = (4, 6)
FRAME_HEIGHT_RANGE = (4, 15)
FRAME_WIDTH_RANGE = (4, 15)
PLACEMENT_TRIES = 400
GENERATION_TRIES = 500
MAX_EDGE_COUNT_OFFSET = 2
FG_COVERAGE_RANGE = (0.10, 0.42)


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "each hollow rectangle needs a non-empty interior"

    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _frame_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _unique_bottom_to_top_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above: dict[int, set[int]] = {int(color): set() for color in colors}
    indegree: dict[int, int] = {int(color): 0 for color in colors}

    for lower, higher in edges:
        lo = int(lower)
        hi = int(higher)
        assert lo != hi, "a layer relation cannot point to itself"
        if hi not in above[lo]:
            above[lo].add(hi)
            indegree[hi] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap evidence must determine a unique total order"
        color = int(available[0])
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Recover the stacking order of occluding hollow rectangles and encode it as a vertical color strip.

    Input structure:
    - The grid has a dominant black background and several distinct non-black hollow rectangles.
    - Each foreground color belongs to exactly one monochrome axis-aligned rectangular frame.
    - When two frames cross, the visible cell shows the higher frame's color and locally breaks the lower frame's edge.

    Rule:
    - Reconstruct each frame as the border of the bounding box of its visible pixels.
    - At every cell covered by multiple reconstructed frames, treat the visible color as the upper frame and all other covering frames as lower.
    - Combine the direct overlap relations transitively into the unique bottom-to-top total order.
    - Output a single-column strip whose top cell is the topmost frame color and whose bottom cell is the bottommost frame color.

    Key insight:
    - Direct evidence is sparse and incomplete, but every visible break point gives a local above/below relation, and the full layer order follows by transitive closure along the connected intersection chain.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert colors, "expected at least one hollow rectangle"

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == int(color)
        box = _bbox(visible)
        assert box is not None, "every frame color must remain visible"
        mask = _frame_mask(g.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed rectangular border"
        frame_masks[int(color)] = mask

    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "an overlap cell must display one of the covering frame colors"
            for lower in covering:
                if lower != top:
                    edges.add((int(lower), top))

    assert edges, "expected at least one crossing relation"
    bottom_to_top = _unique_bottom_to_top_order(colors, edges)
    return np.array(bottom_to_top[::-1], dtype=int).reshape(-1, 1)


def _sample_first_box(height: int, width: int) -> tuple[int, int, int, int]:
    box_h = random.randint(FRAME_HEIGHT_RANGE[0], min(FRAME_HEIGHT_RANGE[1], height))
    box_w = random.randint(FRAME_WIDTH_RANGE[0], min(FRAME_WIDTH_RANGE[1], width))
    r0 = random.randint(0, height - box_h)
    c0 = random.randint(0, width - box_w)
    return r0, c0, r0 + box_h - 1, c0 + box_w - 1


def _sample_next_box(
    height: int,
    width: int,
    prev_box: tuple[int, int, int, int],
    prior_cells: list[set[tuple[int, int]]],
) -> tuple[int, int, int, int] | None:
    pr0, pc0, pr1, pc1 = prev_box
    for _ in range(PLACEMENT_TRIES):
        box_h = random.randint(FRAME_HEIGHT_RANGE[0], min(FRAME_HEIGHT_RANGE[1], height))
        box_w = random.randint(FRAME_WIDTH_RANGE[0], min(FRAME_WIDTH_RANGE[1], width))

        r_lo = max(0, pr0 - box_h + 1)
        r_hi = min(height - box_h, pr1)
        c_lo = max(0, pc0 - box_w + 1)
        c_hi = min(width - box_w, pc1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = random.randint(r_lo, r_hi)
        c0 = random.randint(c_lo, c_hi)
        box = (r0, c0, r0 + box_h - 1, c0 + box_w - 1)
        cells = _frame_cells(box)

        prev_overlap = len(cells & prior_cells[-1])
        if prev_overlap not in (2, 4):
            continue
        if any(len(cells & other) not in (0, 2, 4) for other in prior_cells[:-1]):
            continue
        return box
    return None


def _place_box_chain(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    first_box = _sample_first_box(height, width)
    boxes = [first_box]
    cell_sets = [_frame_cells(first_box)]

    for _ in range(1, count):
        box = _sample_next_box(height, width, boxes[-1], cell_sets)
        if box is None:
            return None
        boxes.append(box)
        cell_sets.append(_frame_cells(box))
    return boxes


def _adjacency_from_boxes(boxes: list[tuple[int, int, int, int]]) -> tuple[list[set[int]], int]:
    n = len(boxes)
    cell_sets = [_frame_cells(box) for box in boxes]
    neighbors = [set() for _ in range(n)]
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            overlap = len(cell_sets[i] & cell_sets[j])
            if overlap > 0:
                assert overlap in (2, 4), "generator only uses simple frame crossings"
                neighbors[i].add(j)
                neighbors[j].add(i)
                edge_count += 1
    return neighbors, edge_count


def _is_connected(neighbors: list[set[int]]) -> bool:
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for nxt in neighbors[node]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(neighbors)


def _render_input(
    shape: tuple[int, int],
    colors_bottom_to_top: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for color, box in zip(colors_bottom_to_top, boxes):
        grid[_frame_mask(shape, box)] = int(color)
    return grid


def _passes_layout_constraints(
    grid: np.ndarray,
    colors_bottom_to_top: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    fg_ratio = float(np.mean(grid != BACKGROUND))
    if not (FG_COVERAGE_RANGE[0] <= fg_ratio <= FG_COVERAGE_RANGE[1]):
        return False
    if _get_bg(grid) != BACKGROUND:
        return False

    neighbors, edge_count = _adjacency_from_boxes(boxes)
    if not _is_connected(neighbors):
        return False
    if edge_count < len(boxes) - 1 or edge_count > len(boxes) - 1 + MAX_EDGE_COUNT_OFFSET:
        return False
    if any((i + 1) not in neighbors[i] for i in range(len(boxes) - 1)):
        return False
    if all(j in neighbors[i] for i in range(len(boxes)) for j in range(i + 1, len(boxes))):
        return False
    if not any(j not in neighbors[i] for i in range(len(boxes)) for j in range(i + 2, len(boxes))):
        return False

    for color, box in zip(colors_bottom_to_top, boxes):
        visible_box = _bbox(grid == int(color))
        if visible_box != box:
            return False

    expected = np.array(colors_bottom_to_top[::-1], dtype=int).reshape(-1, 1)
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected)


def _generate_attempt() -> np.ndarray | None:
    count = random.randint(*FRAME_COUNT_RANGE)
    min_rows = min(GRID_ROWS_RANGE[1], GRID_ROWS_RANGE[0] + 2 * (count - FRAME_COUNT_RANGE[0]))
    min_cols = min(GRID_COLS_RANGE[1], GRID_COLS_RANGE[0] + 2 * (count - FRAME_COUNT_RANGE[0]))
    height = random.randint(min_rows, GRID_ROWS_RANGE[1])
    width = random.randint(min_cols, GRID_COLS_RANGE[1])
    colors_bottom_to_top = random.sample(COLOR_POOL, count)

    boxes = _place_box_chain(height, width, count)
    if boxes is None:
        return None

    grid = _render_input((height, width), colors_bottom_to_top, boxes)
    if not _passes_layout_constraints(grid, colors_bottom_to_top, boxes):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layering-archaeology input")
