from __future__ import annotations

import numpy as np

from arc_utils import bbox, bbox as _bbox, colors_present, get_bg
import random


def _frame_mask(shape: tuple[int, int], b: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = b
    assert r0 < r1 and c0 < c1, "each frame must have positive height and width"

    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the bottom-to-top stacking order of overlaid rectangular color frames.

    Input structure:
    - The background is one color, and every foreground color is a one-cell-thick rectangular frame.
    - Overlapping frame pixels show only the color of the frame that is highest at that location.

    Rule:
    - Reconstruct each frame as the border of the bounding box of its visible pixels.
    - At every cell covered by multiple reconstructed frames, the visible frame color is above the other frames covering that cell.
    - Return the unique bottom-to-top order of frame colors as a single-column grid.

    Key insight:
    The hidden parts of each frame do not need to be seen directly; the visible pixels reveal the full frame bbox, and overlap cells reveal the layer constraints.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert colors, "input must contain foreground frames"

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == color
        b = bbox(visible)
        assert b is not None, "each foreground color must be visible"
        mask = _frame_mask(g.shape, b)
        assert np.all(mask[visible]), "visible pixels must lie on their reconstructed frame border"
        frame_masks[color] = mask

    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "an overlap cell must show one of the frames covering it"
            for color in covering:
                if color != top:
                    edges.add((color, top))

    assert edges, "frames must overlap to determine their order"
    order = _unique_layer_order(colors, edges)
    return np.array(order, dtype=int).reshape(-1, 1)

# --------------- generator (att1) ---------------

# Grid and palette.
GRID_HEIGHT = 18
GRID_WIDTH = 19
BACKGROUND_COLOR = 8
FRAME_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]

# Frame geometry.
FRAME_COUNT_CHOICES = [4, 5]
FRAME_HEIGHT_RANGE = (2, 15)
FRAME_WIDTH_RANGE = (3, 16)
REQUIRED_PREV_OVERLAPS = (2, 4)
ALLOWED_PRIOR_OVERLAPS = (0, 2, 4)

# Retry limits.
PLACEMENT_TRIES = 200
GENERATION_TRIES = 300


def _frame_cells(r0, c0, r1, c1):
    """Return set of (r, c) for the 1-cell-thick border of a rectangle."""
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _sample_layer_order():
    n = random.choice(FRAME_COUNT_CHOICES)
    colors = random.sample(FRAME_COLORS, n)
    random.shuffle(colors)
    return colors


def _random_frame_size():
    fh = random.randint(*FRAME_HEIGHT_RANGE)
    fw = random.randint(*FRAME_WIDTH_RANGE)
    return fh, fw


def _place_first_frame():
    fh, fw = _random_frame_size()
    r0 = random.randint(0, GRID_HEIGHT - 1 - fh)
    c0 = random.randint(0, GRID_WIDTH - 1 - fw)
    return (r0, c0, r0 + fh, c0 + fw)


def _place_next_frame(prev, prior_cell_sets):
    """Sample one frame that crosses the previous frame and stays compatible with older frames."""
    pr0, pc0, pr1, pc1 = prev
    for _ in range(PLACEMENT_TRIES):
        fh, fw = _random_frame_size()
        r0 = random.randint(max(0, pr0 - fh + 1), min(GRID_HEIGHT - 1 - fh, pr1))
        c0 = random.randint(max(0, pc0 - fw + 1), min(GRID_WIDTH - 1 - fw, pc1))
        r1, c1 = r0 + fh, c0 + fw

        if r1 >= GRID_HEIGHT or c1 >= GRID_WIDTH:
            continue

        new_bb = (r0, c0, r1, c1)
        new_cells = _frame_cells(*new_bb)

        if len(new_cells & prior_cell_sets[-1]) not in REQUIRED_PREV_OVERLAPS:
            continue

        if any(len(new_cells & cells) not in ALLOWED_PRIOR_OVERLAPS for cells in prior_cell_sets[:-1]):
            continue

        return new_bb, new_cells
    return None


def _place_frame_stack(n):
    first = _place_first_frame()
    bboxes = [first]
    frame_cell_sets = [_frame_cells(*first)]

    for _ in range(1, n):
        placed = _place_next_frame(bboxes[-1], frame_cell_sets)
        if placed is None:
            return None
        bb, cells = placed
        bboxes.append(bb)
        frame_cell_sets.append(cells)

    return bboxes, frame_cell_sets


def _unique_topo_order(colors, edges):
    """Check if edges define a unique topological order, return it or None."""
    above = {c: set() for c in colors}
    indegree = {c: 0 for c in colors}
    for lo, hi in edges:
        if lo == hi:
            return None
        if hi not in above[lo]:
            above[lo].add(hi)
            indegree[hi] += 1
    order = []
    placed = set()
    while len(order) < len(colors):
        avail = [c for c in colors if c not in placed and indegree[c] == 0]
        if len(avail) != 1:
            return None
        c = avail[0]
        order.append(c)
        placed.add(c)
        for hi in above[c]:
            indegree[hi] -= 1
    return order


def _overlap_edges(layer_order, frame_cell_sets):
    edges = set()
    for i in range(len(layer_order)):
        for j in range(i + 1, len(layer_order)):
            ov = len(frame_cell_sets[i] & frame_cell_sets[j])
            if ov >= 2:
                edges.add((layer_order[i], layer_order[j]))
    return edges


def _render_frames(layer_order, frame_cell_sets):
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND_COLOR, dtype=int)
    for color, cells in zip(layer_order, frame_cell_sets):
        rows, cols = zip(*cells)
        grid[rows, cols] = color
    return grid


def _visible_bboxes_match(grid, layer_order, bboxes):
    for i in range(len(layer_order)):
        visible_bb = _bbox(grid == layer_order[i])
        if visible_bb != bboxes[i]:
            return False
    return True


def _solver_matches(grid, layer_order):
    try:
        out = main(grid)
        expected = np.array(layer_order, dtype=int).reshape(-1, 1)
        return np.array_equal(out, expected)
    except Exception:
        return False


def _gen_attempt():
    layer_order = _sample_layer_order()
    placed = _place_frame_stack(len(layer_order))
    if placed is None:
        return None

    bboxes, frame_cell_sets = placed
    edges = _overlap_edges(layer_order, frame_cell_sets)
    topo = _unique_topo_order(layer_order, edges)
    if topo is None or topo != layer_order:
        return None

    grid = _render_frames(layer_order, frame_cell_sets)
    if not _visible_bboxes_match(grid, layer_order, bboxes):
        return None
    if not _solver_matches(grid, layer_order):
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_TRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
