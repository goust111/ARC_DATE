from __future__ import annotations

import random

import numpy as np


BBox = tuple[int, int, int, int]


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(mask: np.ndarray) -> BBox | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _frame_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 1 and c1 - c0 >= 1, "frame bbox must have positive height and width"
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    higher_than: dict[int, set[int]] = {color: set() for color in colors}
    indegree: dict[int, int] = {color: 0 for color in colors}

    for lower, higher in edges:
        assert lower != higher, "self-overlap cannot define a layer relation"
        if higher not in higher_than[lower]:
            higher_than[lower].add(higher)
            indegree[higher] += 1

    order: list[int] = []
    placed: set[int] = set()
    while len(order) < len(colors):
        available = [color for color in colors if color not in placed and indegree[color] == 0]
        assert len(available) == 1, "overlap constraints must determine a unique layer order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in higher_than[color]:
            indegree[higher] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the bottom-to-top stacking order of overlapping rectangular frames.

    Input structure:
    - The background fills the grid.
    - Each foreground color is a one-cell-thick rectangular frame.
    - Overlaps show only the upper frame color.

    Rule:
    - Reconstruct each frame as the border of the bounding box of its visible pixels.
    - Every overlap cell says that the visible color is above every other frame covering that cell.
    - Deduce the unique bottom-to-top layer order and return it as an N x 1 column.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"

    bg = _mode_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    if not colors:
        return np.empty((0, 1), dtype=int)

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == color
        box = _bbox(visible)
        assert box is not None, "each foreground color must remain visible"
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
            for lower in covering:
                if lower != top:
                    edges.add((lower, top))

    if len(colors) == 1:
        order = colors
    else:
        assert edges, "multiple frames must create overlap constraints"
        order = _unique_topological_order(colors, edges)

    return np.array(order, dtype=int).reshape(-1, 1)


# --------------- generator ---------------

FRAME_COUNT_RANGE = (3, 6)
GRID_MIN_SIDE = 12
GRID_MAX_SIDE = 30
FRAME_SIDE_MIN = 4
FRAME_SIDE_MAX = 10
MARGIN_MIN = 1
GENERATION_ATTEMPTS = 400


def _choose_axis_params(count: int) -> tuple[int, int, int]:
    candidates: list[tuple[int, int, int]] = []
    for span in range(FRAME_SIDE_MIN, FRAME_SIDE_MAX + 1):
        min_step = (span + 1) // 2
        max_step = span - 2
        if min_step > max_step:
            continue
        max_fit_step = (GRID_MAX_SIDE - 2 * MARGIN_MIN - span) // (count - 1)
        max_step = min(max_step, max_fit_step)
        if min_step > max_step:
            continue
        for step in range(min_step, max_step + 1):
            structure = span + step * (count - 1)
            if structure + 2 * MARGIN_MIN <= GRID_MAX_SIDE:
                candidates.append((span, step, structure))
    assert candidates, "generator needs at least one feasible axis configuration"
    return random.choice(candidates)


def _symmetry_transforms() -> list:
    return [
        lambda x: x,
        np.flipud,
        np.fliplr,
        lambda x: np.flipud(np.fliplr(x)),
        np.transpose,
        lambda x: np.flipud(np.transpose(x)),
        lambda x: np.fliplr(np.transpose(x)),
        lambda x: np.flipud(np.fliplr(np.transpose(x))),
    ]


def _paint_frame(grid: np.ndarray, box: BBox, color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _generate_candidate() -> tuple[np.ndarray, list[int]] | None:
    count = random.randint(*FRAME_COUNT_RANGE)
    height, step_r, structure_h = _choose_axis_params(count)
    width, step_c, structure_w = _choose_axis_params(count)

    grid_h_min = max(GRID_MIN_SIDE, structure_h + 2 * MARGIN_MIN)
    grid_w_min = max(GRID_MIN_SIDE, structure_w + 2 * MARGIN_MIN)
    if grid_h_min > GRID_MAX_SIDE or grid_w_min > GRID_MAX_SIDE:
        return None

    grid_h = random.randint(grid_h_min, GRID_MAX_SIDE)
    grid_w = random.randint(grid_w_min, GRID_MAX_SIDE)
    top = random.randint(MARGIN_MIN, grid_h - structure_h - MARGIN_MIN)
    left = random.randint(MARGIN_MIN, grid_w - structure_w - MARGIN_MIN)

    palette = list(range(10))
    bg = random.choice(palette)
    palette.remove(bg)
    order = random.sample(palette, count)

    boxes = []
    for idx in range(count):
        r0 = top + idx * step_r
        c0 = left + idx * step_c
        boxes.append((r0, c0, r0 + height - 1, c0 + width - 1))

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    ascending = bool(random.getrandbits(1))
    geom_indices = list(range(count)) if ascending else list(range(count - 1, -1, -1))
    for color, geom_idx in zip(order, geom_indices, strict=False):
        _paint_frame(grid, boxes[geom_idx], color)

    transform = random.choice(_symmetry_transforms())
    grid = np.array(transform(grid), dtype=int, copy=True)

    if _mode_color(grid) != bg:
        return None
    if len(np.unique(grid)) != count + 1:
        return None

    expected = np.array(order, dtype=int).reshape(-1, 1)
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid, order


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _generate_candidate()
        if candidate is not None:
            return candidate[0]
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
