from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


def _load_arc_utils():
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        candidate = parent / "new" / "solvers"
        if candidate.is_dir():
            if str(candidate) not in sys.path:
                sys.path.append(str(candidate))
            break
    from arc_utils import Color, bbox, colors_present, get_bg

    return Color, bbox, colors_present, get_bg


Color, bbox, colors_present, get_bg = _load_arc_utils()


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, box)
    assert r1 - r0 + 1 >= 3 and c1 - c0 + 1 >= 3, "frames must be at least 3x3"

    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _frame_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
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
        assert lower != higher, "layer constraints cannot self-loop"
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap constraints must determine a unique total order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1

    return order


def _render_output(order: list[int]) -> np.ndarray:
    n = len(order)
    doubled = np.resize(np.array(order, dtype=int), 2 * n)
    index = np.maximum.outer(np.arange(2 * n), np.arange(2 * n))
    return doubled[index]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the bottom-to-top frame order and project it into a doubled diagonal-index square.

    Input structure:
    - The background is black and the foreground consists of 3 to 5 distinct one-pixel-thick hollow rectangular frames.
    - Each foreground color appears on exactly one frame, and overlapping pixels show only the highest visible frame color.
    - The visible fragments of each color still determine that frame's full bounding box.

    Rule:
    - Reconstruct each frame as the border of the bounding box of its visible pixels.
    - Whenever several reconstructed frames cover the same cell, treat the visible color at that cell as lying above every other covering frame there.
    - Infer the unique bottom-to-top total order of frame colors.
    - Repeat that order cyclically to length 2n and fill a 2n by 2n square with color H[max(row, column)].

    Key insight:
    - The stacking order is not the final answer by itself; it becomes the one-dimensional header sequence that drives the diagonal-index projection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"
    colors = colors_present(g, exclude={bg})
    assert 3 <= len(colors) <= 5, "expected 3 to 5 frame colors"

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = (g == color)
        box = bbox(visible)
        assert box is not None, "each frame color must be visible"
        mask = _frame_mask(g.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        frame_masks[color] = mask

    edges: set[tuple[int, int]] = set()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "an overlap cell must show one of the covering frames"
            for color in covering:
                if color != top:
                    edges.add((color, top))

    assert edges, "frame overlaps must provide ordering evidence"
    order = _unique_layer_order(colors, edges)
    return _render_output(order)


GEN_BG = Color.BLACK
FRAME_PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK]
FRAME_COUNT_CHOICES = [3, 4, 5]
GRID_HEIGHT_RANGE = (12, 17)
GRID_WIDTH_RANGE = (12, 17)
FRAME_SPAN_RANGE = (3, 8)
REQUIRED_PREV_OVERLAPS = (2, 4)
ALLOWED_PRIOR_OVERLAPS = (0, 2, 4)
PLACEMENT_TRIES = 400
GENERATION_TRIES = 700
_FRAME_COUNT_BAG: list[int] = []


def _sample_frame_count(rng: random.Random) -> int:
    global _FRAME_COUNT_BAG
    if not _FRAME_COUNT_BAG:
        _FRAME_COUNT_BAG = FRAME_COUNT_CHOICES[:]
        rng.shuffle(_FRAME_COUNT_BAG)
    return _FRAME_COUNT_BAG.pop()


def _sample_layer_order(count: int, rng: random.Random) -> list[int]:
    colors = rng.sample(FRAME_PALETTE, count)
    rng.shuffle(colors)
    return colors


def _random_box(h: int, w: int, rng: random.Random) -> tuple[int, int, int, int]:
    bh = rng.randint(*FRAME_SPAN_RANGE)
    bw = rng.randint(*FRAME_SPAN_RANGE)
    r0 = rng.randint(0, h - bh)
    c0 = rng.randint(0, w - bw)
    return r0, c0, r0 + bh - 1, c0 + bw - 1


def _place_next_box(
    prev_box: tuple[int, int, int, int],
    prior_cell_sets: list[set[tuple[int, int]]],
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    pr0, pc0, pr1, pc1 = map(int, prev_box)
    for _ in range(PLACEMENT_TRIES):
        bh = rng.randint(*FRAME_SPAN_RANGE)
        bw = rng.randint(*FRAME_SPAN_RANGE)
        r_lo = max(0, pr0 - bh + 1)
        r_hi = min(h - bh, pr1)
        c_lo = max(0, pc0 - bw + 1)
        c_hi = min(w - bw, pc1)
        if r_lo > r_hi or c_lo > c_hi:
            continue

        r0 = rng.randint(r_lo, r_hi)
        c0 = rng.randint(c_lo, c_hi)
        box = (r0, c0, r0 + bh - 1, c0 + bw - 1)
        cells = _frame_cells(box)

        if len(cells & prior_cell_sets[-1]) not in REQUIRED_PREV_OVERLAPS:
            continue
        if any(len(cells & older) not in ALLOWED_PRIOR_OVERLAPS for older in prior_cell_sets[:-1]):
            continue
        return box, cells
    return None


def _place_frame_stack(
    count: int,
    h: int,
    w: int,
    rng: random.Random,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]]] | None:
    first_box = _random_box(h, w, rng)
    boxes = [first_box]
    cell_sets = [_frame_cells(first_box)]

    for _ in range(1, count):
        placed = _place_next_box(boxes[-1], cell_sets, h, w, rng)
        if placed is None:
            return None
        box, cells = placed
        boxes.append(box)
        cell_sets.append(cells)

    return boxes, cell_sets


def _render_frames(
    order: list[int],
    cell_sets: list[set[tuple[int, int]]],
    h: int,
    w: int,
) -> np.ndarray:
    grid = np.full((h, w), GEN_BG, dtype=int)
    for color, cells in zip(order, cell_sets):
        rows, cols = zip(*sorted(cells))
        grid[list(rows), list(cols)] = color
    return grid


def _visible_bboxes_match(
    grid: np.ndarray,
    order: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> bool:
    return all(bbox(grid == color) == box for color, box in zip(order, boxes))


def _actual_overlap_edges(
    order: list[int],
    cell_sets: list[set[tuple[int, int]]],
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if cell_sets[i] & cell_sets[j]:
                edges.add((order[i], order[j]))
    return edges


def _every_frame_overlaps(cell_sets: list[set[tuple[int, int]]]) -> bool:
    for i, cells in enumerate(cell_sets):
        if not any(cells & other for j, other in enumerate(cell_sets) if i != j):
            return False
    return True


def _matches_simple_sort(order: list[int], values: list[int]) -> bool:
    if len(set(values)) != len(values):
        return False
    ascending = [color for _, color in sorted(zip(values, order))]
    descending = [color for _, color in sorted(zip(values, order), reverse=True)]
    return order == ascending or order == descending


def _reject_shortcuts(order: list[int], boxes: list[tuple[int, int, int, int]]) -> bool:
    heights = [r1 - r0 + 1 for r0, _c0, r1, _c1 in boxes]
    widths = [c1 - c0 + 1 for _r0, c0, _r1, c1 in boxes]
    areas = [h * w for h, w in zip(heights, widths)]
    top_rows = [r0 for r0, _c0, _r1, _c1 in boxes]
    left_cols = [c0 for _r0, c0, _r1, _c1 in boxes]
    bottom_rows = [r1 for _r0, _c0, r1, _c1 in boxes]
    right_cols = [c1 for _r0, _c0, _r1, c1 in boxes]

    heuristics = [
        order,
        sorted(order),
        sorted(order, reverse=True),
    ]
    if order in heuristics[1:]:
        return True

    for values in [heights, widths, areas, top_rows, left_cols, bottom_rows, right_cols]:
        if _matches_simple_sort(order, values):
            return True
    return False


def _semantic_check(
    grid: np.ndarray,
    order: list[int],
    boxes: list[tuple[int, int, int, int]],
    cell_sets: list[set[tuple[int, int]]],
) -> bool:
    if not _visible_bboxes_match(grid, order, boxes):
        return False
    if not _every_frame_overlaps(cell_sets):
        return False

    actual_edges = _actual_overlap_edges(order, cell_sets)
    if len(actual_edges) < len(order) - 1:
        return False

    try:
        if _unique_layer_order(order, actual_edges) != order:
            return False
    except AssertionError:
        return False

    if _reject_shortcuts(order, boxes):
        return False

    try:
        predicted = main(grid)
    except Exception:
        return False
    return np.array_equal(predicted, _render_output(order))


def _gen_attempt(frame_count: int, rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_HEIGHT_RANGE)
    w = rng.randint(*GRID_WIDTH_RANGE)
    order = _sample_layer_order(frame_count, rng)
    placed = _place_frame_stack(len(order), h, w, rng)
    if placed is None:
        return None

    boxes, cell_sets = placed
    grid = _render_frames(order, cell_sets, h, w)
    if not _semantic_check(grid, order, boxes, cell_sets):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    frame_count = _sample_frame_count(rng)
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt(frame_count, rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_TRIES} attempts")
