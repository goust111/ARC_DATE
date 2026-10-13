from __future__ import annotations

import itertools
import random

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

PANEL_H_RANGE = (7, 10)
PANEL_W_RANGE = (7, 10)
LOCAL_OUTER_ATTEMPTS = 400
LOCAL_INNER_ATTEMPTS = 160
GLOBAL_ATTEMPTS = 500


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
    out = np.zeros(shape, dtype=bool)
    out[r0 : r1 + 1, c0] = True
    out[r0 : r1 + 1, c1] = True
    out[r0, c0 : c1 + 1] = True
    out[r1, c0 : c1 + 1] = True
    return out


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


def _unique_topological_order(
    colors: list[int], edges: set[tuple[int, int]]
) -> list[int] | None:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}
    for lower, higher in edges:
        if lower == higher:
            return None
        if higher not in above[lower]:
            above[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        if len(available) != 1:
            return None
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return order


def _find_separator_cross(grid: np.ndarray) -> tuple[int, int]:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == SEP)]
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == SEP)]
    assert len(rows) == 1 and len(cols) == 1, "expected exactly one separator row and column"
    cross_r = rows[0]
    cross_c = cols[0]
    assert cross_r > 0 and cross_r < grid.shape[0] - 1
    assert cross_c > 0 and cross_c < grid.shape[1] - 1
    assert cross_r == grid.shape[0] - cross_r - 1
    assert cross_c == grid.shape[1] - cross_c - 1
    return cross_r, cross_c


def _quadrant_slices(
    cross_r: int, cross_c: int, shape: tuple[int, int]
) -> list[tuple[slice, slice]]:
    height, width = shape
    return [
        (slice(0, cross_r), slice(0, cross_c)),
        (slice(0, cross_r), slice(cross_c + 1, width)),
        (slice(cross_r + 1, height), slice(0, cross_c)),
        (slice(cross_r + 1, height), slice(cross_c + 1, width)),
    ]


def _split_quadrants(grid: np.ndarray) -> list[np.ndarray]:
    cross_r, cross_c = _find_separator_cross(grid)
    panels = [grid[rs, cs].copy() for rs, cs in _quadrant_slices(cross_r, cross_c, grid.shape)]
    first_shape = panels[0].shape
    assert all(panel.shape == first_shape for panel in panels)
    return panels


def _infer_local_order(panel: np.ndarray) -> tuple[list[int], set[tuple[int, int]]]:
    colors = sorted(int(v) for v in np.unique(panel) if int(v) not in (BG, SEP))
    assert len(colors) == 3, "each quadrant must expose exactly three colors"

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = panel == color
        box = _bbox(visible)
        assert box is not None, "each visible color must have at least one pixel"
        mask = _frame_mask(panel.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed frame border"
        frame_masks[color] = mask

    edges: set[tuple[int, int]] = set()
    overlap_cells = 0
    for r in range(panel.shape[0]):
        for c in range(panel.shape[1]):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            overlap_cells += 1
            top = int(panel[r, c])
            assert top in covering, "an overlap cell must show the top visible frame"
            for lower in covering:
                if lower != top:
                    edges.add((lower, top))

    assert overlap_cells > 0, "every quadrant needs at least one overlap cell"
    order = _unique_topological_order(colors, edges)
    assert order is not None, "local overlap evidence must determine a unique three-color order"
    return order, edges


def _candidate_global_orders(
    all_colors: list[int], visible_sets: list[set[int]], local_orders: list[list[int]]
) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []
    for perm in itertools.permutations(all_colors):
        if all([color for color in perm if color in visible] == order for visible, order in zip(visible_sets, local_orders)):
            candidates.append(tuple(int(color) for color in perm))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the unique global bottom-to-top frame order from four quadrant-local overlap orders.

    Input structure:
    - A black background is split into four equal quadrants by one solid grey separator row and one solid grey separator column.
    - Across the whole puzzle there are exactly four foreground colors, each representing one hollow rectangular frame layer.
    - Inside each quadrant, exactly three of those colors are visible as partially overlapping one-cell-thick rectangular frames; the fourth color is missing from that quadrant.

    Rule:
    - In each quadrant, reconstruct every visible frame as the border of the bounding box of that color's visible pixels.
    - At every cell where reconstructed frames overlap, the shown color is above every other frame covering that cell, which yields the unique local order of the three visible colors.
    - Compare the four visible color sets to identify the missing color in each quadrant, then keep only the global four-color permutations whose quadrant subsequences match all four local orders.
    - Output the unique global order as a 4x1 column from bottom layer to top layer.

    Key insight:
    - No single quadrant places the absent color, so the answer comes from fusing four three-color subsequences of one hidden four-color total order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    panels = _split_quadrants(g)
    local_orders: list[list[int]] = []
    visible_sets: list[set[int]] = []
    all_color_set: set[int] = set()

    for panel in panels:
        order, _edges = _infer_local_order(panel)
        local_orders.append(order)
        visible = set(order)
        visible_sets.append(visible)
        all_color_set |= visible

    assert len(all_color_set) == 4, "the four quadrants together must expose exactly four colors"
    all_colors = sorted(all_color_set)

    missing_colors = []
    for visible in visible_sets:
        missing = list(all_color_set - visible)
        assert len(missing) == 1, "each quadrant must miss exactly one of the four colors"
        missing_colors.append(missing[0])
    assert len(set(missing_colors)) == 4, "the four quadrants must miss four different colors"

    candidates = _candidate_global_orders(all_colors, visible_sets, local_orders)
    assert len(candidates) == 1, "quadrant clues must determine one global order"
    return np.array(candidates[0], dtype=int).reshape(4, 1)


def _random_bbox(height: int, width: int) -> tuple[int, int, int, int]:
    side_h = random.randint(4, height)
    side_w = random.randint(4, width)
    r0 = random.randint(0, height - side_h)
    c0 = random.randint(0, width - side_w)
    return (r0, c0, r0 + side_h - 1, c0 + side_w - 1)


def _render_panel(
    order: list[int], bboxes: list[tuple[int, int, int, int]], shape: tuple[int, int]
) -> np.ndarray:
    panel = np.full(shape, BG, dtype=int)
    for color, box in zip(order, bboxes):
        for r, c in _frame_cells(box):
            panel[r, c] = int(color)
    return panel


def _generate_local_panel(
    order: list[int], height: int, width: int
) -> tuple[np.ndarray, dict[str, int]] | None:
    for _ in range(LOCAL_OUTER_ATTEMPTS):
        target_12 = random.choice([2, 4])
        target_23 = random.choice([2, 4])
        target_13 = random.choice([0, 2, 4])

        box1 = _random_bbox(height, width)
        cells1 = _frame_cells(box1)

        box2 = None
        cells2 = None
        for _ in range(LOCAL_INNER_ATTEMPTS):
            candidate = _random_bbox(height, width)
            if candidate == box1:
                continue
            candidate_cells = _frame_cells(candidate)
            if len(cells1 & candidate_cells) == target_12:
                box2 = candidate
                cells2 = candidate_cells
                break
        if box2 is None or cells2 is None:
            continue

        box3 = None
        for _ in range(LOCAL_INNER_ATTEMPTS):
            candidate = _random_bbox(height, width)
            if candidate in (box1, box2):
                continue
            candidate_cells = _frame_cells(candidate)
            if len(cells2 & candidate_cells) != target_23:
                continue
            if len(cells1 & candidate_cells) != target_13:
                continue
            box3 = candidate
            break
        if box3 is None:
            continue

        panel = _render_panel(order, [box1, box2, box3], (height, width))
        try:
            inferred_order, edges = _infer_local_order(panel)
        except AssertionError:
            continue
        if inferred_order != order:
            continue
        if len(edges) < 2:
            continue
        return panel, {"ov12": target_12, "ov23": target_23, "ov13": target_13}
    return None


def _assemble_grid(panels: list[np.ndarray]) -> np.ndarray:
    panel_h, panel_w = panels[0].shape
    grid = np.full((2 * panel_h + 1, 2 * panel_w + 1), BG, dtype=int)
    grid[panel_h, :] = SEP
    grid[:, panel_w] = SEP
    placements = [
        (0, 0),
        (0, panel_w + 1),
        (panel_h + 1, 0),
        (panel_h + 1, panel_w + 1),
    ]
    for panel, (r0, c0) in zip(panels, placements):
        grid[r0 : r0 + panel_h, c0 : c0 + panel_w] = panel
    return grid


def _validate_generated(grid: np.ndarray, expected_order: list[int]) -> None:
    expected = np.array(expected_order, dtype=int).reshape(4, 1)
    actual = main(grid)
    assert np.array_equal(actual, expected)

    panels = _split_quadrants(grid)
    local_orders: list[list[int]] = []
    visible_sets: list[set[int]] = []
    for panel in panels:
        order, edges = _infer_local_order(panel)
        local_orders.append(order)
        visible_sets.append(set(order))
        assert len(edges) >= 2

    all_colors = sorted(set().union(*visible_sets))
    assert set(all_colors) == set(expected_order)

    missing_colors = [tuple(sorted(set(all_colors) - visible)) for visible in visible_sets]
    assert all(len(missing) == 1 for missing in missing_colors)
    assert len({missing[0] for missing in missing_colors}) == 4

    candidates = _candidate_global_orders(all_colors, visible_sets, local_orders)
    assert candidates == [tuple(expected_order)]

    for visible, order in zip(visible_sets, local_orders):
        single_candidates = [
            perm
            for perm in itertools.permutations(all_colors)
            if [color for color in perm if color in visible] == order
        ]
        assert len(single_candidates) > 1, "a single quadrant must not determine the global order"

    assert len(set(tuple(order) for order in local_orders)) >= 2, "not all quadrants should show the same local order"
    assert any(len(edges) >= 3 for _, edges in [_infer_local_order(panel) for panel in panels])


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*PANEL_H_RANGE)
    panel_w = random.randint(*PANEL_W_RANGE)

    all_colors = random.sample(PALETTE, 4)
    global_order = random.sample(all_colors, 4)
    missing_by_quadrant = random.sample(all_colors, 4)

    panels: list[np.ndarray] = []
    for missing in missing_by_quadrant:
        local_order = [color for color in global_order if color != missing]
        built = _generate_local_panel(local_order, panel_h, panel_w)
        if built is None:
            return None
        panel, _meta = built
        panels.append(panel)

    grid = _assemble_grid(panels)
    try:
        _validate_generated(grid, global_order)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GLOBAL_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
