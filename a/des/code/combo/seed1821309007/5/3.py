from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
GRID_H_RANGE = (12, 23)
GRID_W_RANGE = (12, 23)
FRAME_COUNT_RANGE = (3, 6)
FRAME_H_RANGE = (3, 12)
FRAME_W_RANGE = (3, 12)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_TRIES = 300


def _colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    exclude = exclude or set()
    colors: list[int] = []
    seen: set[int] = set()
    for value in grid.ravel():
        color = int(value)
        if color in exclude or color in seen:
            continue
        seen.add(color)
        colors.append(color)
    return colors


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2
    mask = np.zeros(shape, dtype=bool)
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    return mask


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
        assert len(available) == 1, "overlap constraints must determine a unique order"
        color = available[0]
        order.append(color)
        placed.add(color)
        for higher in above[color]:
            indegree[higher] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    colors = _colors_present(g, exclude={BACKGROUND})
    assert colors, "expected at least one colored frame"

    frame_masks: dict[int, np.ndarray] = {}
    for color in colors:
        visible = g == color
        box = _bbox(visible)
        assert box is not None, "each color must remain visible"
        mask = _frame_mask(g.shape, box)
        assert np.all(mask[visible]), "visible pixels must lie on the reconstructed border"
        frame_masks[color] = mask

    edges: set[tuple[int, int]] = set()
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            covering = [color for color in colors if frame_masks[color][r, c]]
            if len(covering) < 2:
                continue
            top = int(g[r, c])
            assert top in covering, "visible overlap color must be one of the covering frames"
            for lower in covering:
                if lower != top:
                    edges.add((lower, top))

    assert edges, "expected at least one overlap"
    order = _unique_layer_order(colors, edges)
    return np.asarray(order, dtype=int).reshape(-1, 1)


def _sample_frame(shape: tuple[int, int], rng: random.Random) -> tuple[int, int, int, int]:
    h, w = shape
    fh = rng.randint(FRAME_H_RANGE[0], min(FRAME_H_RANGE[1], h - 1))
    fw = rng.randint(FRAME_W_RANGE[0], min(FRAME_W_RANGE[1], w - 1))
    r0 = rng.randint(0, h - fh)
    c0 = rng.randint(0, w - fw)
    return r0, c0, r0 + fh - 1, c0 + fw - 1


def _pair_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return len(_frame_cells(a) & _frame_cells(b))


def _place_frame_chain(shape: tuple[int, int], count: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = [_sample_frame(shape, rng)]
    for idx in range(1, count):
        prev = boxes[-1]
        for _ in range(PLACEMENT_TRIES):
            candidate = _sample_frame(shape, rng)
            overlap_prev = _pair_overlap(candidate, prev)
            if overlap_prev < 2:
                continue

            overlaps = [_pair_overlap(candidate, earlier) for earlier in boxes[:-1]]
            if idx == 1:
                allowed = True
            else:
                allowed = all(v in (0, 2, 4) for v in overlaps)
            if not allowed:
                continue

            boxes.append(candidate)
            break
        else:
            return None
    return boxes


def _render(boxes: list[tuple[int, int, int, int]], order: list[int], shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    for color, box in zip(order, boxes):
        mask = _frame_mask(shape, box)
        grid[mask] = color
    return grid


def _edges_from_boxes(boxes: list[tuple[int, int, int, int]], order: list[int]) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i in range(len(order)):
        for j in range(i + 1, len(order)):
            if _pair_overlap(boxes[i], boxes[j]) >= 2:
                edges.add((order[i], order[j]))
    return edges


def _visible_boxes_match(grid: np.ndarray, order: list[int], boxes: list[tuple[int, int, int, int]]) -> bool:
    for color, box in zip(order, boxes):
        visible_box = _bbox(grid == color)
        if visible_box != box:
            return False
    return True


def _solver_matches(grid: np.ndarray, order: list[int]) -> bool:
    try:
        expected = np.asarray(order, dtype=int).reshape(-1, 1)
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    shape = (
        rng.randint(*GRID_H_RANGE),
        rng.randint(*GRID_W_RANGE),
    )
    count = rng.randint(*FRAME_COUNT_RANGE)
    if min(shape) < 6 or count > len(PALETTE):
        return None

    order = rng.sample(PALETTE, count)
    boxes = _place_frame_chain(shape, count, rng)
    if boxes is None:
        return None

    edges = _edges_from_boxes(boxes, order)
    try:
        topo = _unique_layer_order(order, edges)
    except AssertionError:
        return None
    if topo != order:
        return None

    grid = _render(boxes, order, shape)
    if not _visible_boxes_match(grid, order, boxes):
        return None
    if not _solver_matches(grid, order):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
