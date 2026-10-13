from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Canonical nested-output geometry: a 3x5 inner frame, then one background
# cell plus one border cell on each side for every additional layer.
INNER_HEIGHT = 3
INNER_WIDTH = 5

# Generator ranges.
FRAME_COUNT_RANGE = (3, 5)
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 28)
FRAME_HEIGHT_RANGE = (4, 12)
FRAME_WIDTH_RANGE = (4, 12)
ALLOWED_OVERLAPS = {0, 2, 4}
PLACEMENT_TRIES = 200
GENERATION_TRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    assert r0 < r1 and c0 < c1, "frame bbox must have positive height and width"

    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _frame_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for lower, higher in edges:
        assert lower != higher, "self-loop is not allowed"
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


def _render_nested_frames(order: list[int]) -> np.ndarray:
    n = len(order)
    height = INNER_HEIGHT + 4 * (n - 1)
    width = INNER_WIDTH + 4 * (n - 1)
    out = np.full((height, width), BACKGROUND, dtype=int)

    for i, color in enumerate(order):
        inset = 2 * i
        bbox = (inset, inset, height - 1 - inset, width - 1 - inset)
        out[_frame_mask(out.shape, bbox)] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the full bbox of each visible one-pixel rectangular frame, recover the
    unique bottom-to-top stacking order from overlap visibility, and rebuild the
    colors as centered concentric nested rectangular frames.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = [int(color) for color in np.unique(g) if int(color) != BACKGROUND]
    assert colors, "expected at least one visible frame color"

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == color
        bbox = _bbox(visible)
        assert bbox is not None, "every color must remain at least partially visible"
        mask = _frame_mask(g.shape, bbox)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        frame_masks[color] = mask

    if len(colors) == 1:
        return _render_nested_frames(colors)

    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "an overlap cell must show one of the covering frames"
            for lower in covering:
                if lower != top:
                    edges.add((lower, top))

    order = _unique_topological_order(colors, edges)
    return _render_nested_frames(order)


def _random_frame_size(height: int, width: int) -> tuple[int, int] | None:
    max_h = min(FRAME_HEIGHT_RANGE[1], height - 2)
    max_w = min(FRAME_WIDTH_RANGE[1], width - 2)
    if max_h < FRAME_HEIGHT_RANGE[0] or max_w < FRAME_WIDTH_RANGE[0]:
        return None
    return random.randint(FRAME_HEIGHT_RANGE[0], max_h), random.randint(FRAME_WIDTH_RANGE[0], max_w)


def _place_first_frame(height: int, width: int) -> tuple[int, int, int, int] | None:
    size = _random_frame_size(height, width)
    if size is None:
        return None
    fh, fw = size
    r0 = random.randint(1, height - fh - 1)
    c0 = random.randint(1, width - fw - 1)
    return (r0, c0, r0 + fh - 1, c0 + fw - 1)


def _place_next_frame(
    prev_bbox: tuple[int, int, int, int],
    prior_cell_sets: list[set[tuple[int, int]]],
    height: int,
    width: int,
    required_overlap: int,
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]]] | None:
    pr0, pc0, pr1, pc1 = prev_bbox

    for _ in range(PLACEMENT_TRIES):
        size = _random_frame_size(height, width)
        if size is None:
            return None
        fh, fw = size

        low_r = max(1, pr0 - fh + 2)
        high_r = min(height - fh - 1, pr1 - 1)
        low_c = max(1, pc0 - fw + 2)
        high_c = min(width - fw - 1, pc1 - 1)
        if low_r > high_r or low_c > high_c:
            continue

        r0 = random.randint(low_r, high_r)
        c0 = random.randint(low_c, high_c)
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        cells = _frame_cells(bbox)

        if len(cells & prior_cell_sets[-1]) != required_overlap:
            continue
        if any(len(cells & other) not in ALLOWED_OVERLAPS for other in prior_cell_sets[:-1]):
            continue

        return bbox, cells

    return None


def _render_stack(
    height: int,
    width: int,
    order: list[int],
    frame_cell_sets: list[set[tuple[int, int]]],
) -> np.ndarray:
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for color, cells in zip(order, frame_cell_sets):
        rows, cols = zip(*cells)
        grid[list(rows), list(cols)] = int(color)
    return grid


def _visible_bboxes_match(
    grid: np.ndarray,
    order: list[int],
    bboxes: list[tuple[int, int, int, int]],
) -> bool:
    for color, bbox in zip(order, bboxes):
        if _bbox(grid == color) != bbox:
            return False
    return True


def _solver_matches(grid: np.ndarray, order: list[int]) -> bool:
    try:
        expected = _render_nested_frames(order)
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _sample_layer_order() -> list[int]:
    n = random.randint(*FRAME_COUNT_RANGE)
    order = random.sample(PALETTE, n)
    random.shuffle(order)
    return order


def _place_frame_stack(
    height: int,
    width: int,
    count: int,
) -> tuple[list[tuple[int, int, int, int]], list[set[tuple[int, int]]]] | None:
    first = _place_first_frame(height, width)
    if first is None:
        return None

    bboxes = [first]
    frame_cell_sets = [_frame_cells(first)]
    for _ in range(1, count):
        required_overlap = random.choice((2, 4))
        placed = _place_next_frame(bboxes[-1], frame_cell_sets, height, width, required_overlap)
        if placed is None:
            return None
        bbox, cells = placed
        bboxes.append(bbox)
        frame_cell_sets.append(cells)
    return bboxes, frame_cell_sets


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    order = _sample_layer_order()

    placed = _place_frame_stack(height, width, len(order))
    if placed is None:
        return None
    bboxes, frame_cell_sets = placed

    grid = _render_stack(height, width, order, frame_cell_sets)
    if not _visible_bboxes_match(grid, order, bboxes):
        return None
    if not _solver_matches(grid, order):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_TRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("Failed to generate a valid input sample")
