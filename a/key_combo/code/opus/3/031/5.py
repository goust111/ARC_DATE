from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox, colors_present, get_bg, paint_coords


BBox = tuple[int, int, int, int]

_BG = 0
_MIN_SIDE = 4
_MAX_SIDE = 7
_MARGIN = 3
_GRID_H_RANGE = (22, 27)
_GRID_W_RANGE = (22, 27)
_LAYER_COUNT_CHOICES = [3, 3, 4]
_COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]
_MAX_GENERATION_ATTEMPTS = 400


def _expanded_box(box: BBox, delta: int) -> BBox:
    r0, c0, r1, c1 = box
    d = int(delta)
    return (r0 - d, c0 - d, r1 + d, c1 + d)


@lru_cache(maxsize=None)
def _ring_coords(box: BBox) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = map(int, box)
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    for c in range(c1, c0 - 1, -1):
        coords.append((r1, c))
    for r in range(r1 - 1, r0, -1):
        coords.append((r, c0))
    return tuple(coords)


@lru_cache(maxsize=None)
def _ring_set(box: BBox) -> frozenset[tuple[int, int]]:
    return frozenset(_ring_coords(box))


@lru_cache(maxsize=None)
def _all_boxes(shape: tuple[int, int]) -> tuple[BBox, ...]:
    h, w = shape
    boxes: list[BBox] = []
    for height in range(_MIN_SIDE, _MAX_SIDE + 1):
        for width in range(_MIN_SIDE, _MAX_SIDE + 1):
            for r0 in range(_MARGIN, h - _MARGIN - height + 1):
                r1 = r0 + height - 1
                for c0 in range(_MARGIN, w - _MARGIN - width + 1):
                    c1 = c0 + width - 1
                    boxes.append((r0, c0, r1, c1))
    return tuple(boxes)


def _candidate_boxes(grid: np.ndarray, bg: int, color: int) -> list[BBox]:
    visible_mask = grid == int(color)
    visible_bbox = bbox(visible_mask)
    assert visible_bbox is not None, f"missing visible pixels for color {color}"
    vr0, vc0, vr1, vc1 = visible_bbox
    visible = {tuple(map(int, cell)) for cell in np.argwhere(visible_mask)}
    h, w = grid.shape
    candidates: list[tuple[tuple[int, int, BBox], BBox]] = []

    for height in range(_MIN_SIDE, _MAX_SIDE + 1):
        r0_min = max(0, vr1 - height + 1)
        r0_max = min(vr0, h - height)
        if r0_min > r0_max:
            continue
        for width in range(_MIN_SIDE, _MAX_SIDE + 1):
            c0_min = max(0, vc1 - width + 1)
            c0_max = min(vc0, w - width)
            if c0_min > c0_max:
                continue
            for r0 in range(r0_min, r0_max + 1):
                r1 = r0 + height - 1
                for c0 in range(c0_min, c0_max + 1):
                    c1 = c0 + width - 1
                    box = (r0, c0, r1, c1)
                    if any(
                        not (r == r0 or r == r1 or c == c0 or c == c1)
                        for r, c in visible
                    ):
                        continue
                    ring = _ring_coords(box)
                    if any(int(grid[r, c]) == int(bg) for r, c in ring):
                        continue
                    if len(visible) * 5 < len(ring) * 3:
                        continue
                    score = (len(ring), height * width, box)
                    candidates.append((score, box))

    candidates.sort(key=lambda item: item[0])
    return [box for _score, box in candidates]


def _partial_consistent(
    grid: np.ndarray,
    assigned: dict[int, BBox],
    color: int,
    box: BBox,
) -> bool:
    ring = _ring_set(box)
    for r, c in ring:
        seen = int(grid[r, c])
        if seen == color:
            continue
        if seen in assigned and (r, c) not in _ring_set(assigned[seen]):
            return False

    for other_color, other_box in assigned.items():
        if other_color == color:
            continue
        for r, c in _ring_set(other_box):
            if int(grid[r, c]) == color and (r, c) not in ring:
                return False
    return True


def _top_order(grid: np.ndarray, bg: int, boxes: dict[int, BBox]) -> list[int]:
    above_to_below = {color: set() for color in boxes}
    indegree = {color: 0 for color in boxes}

    for color, box in boxes.items():
        for r, c in _ring_coords(box):
            seen = int(grid[r, c])
            if seen == int(bg):
                raise AssertionError("reconstructed border cannot contain background")
            if seen == color:
                continue
            if seen not in boxes:
                raise AssertionError("occluding color must also be a reconstructed rectangle")
            if (r, c) not in _ring_set(boxes[seen]):
                raise AssertionError("occluding color must occupy the shared border pixel")
            if color not in above_to_below[seen]:
                above_to_below[seen].add(color)
                indegree[color] += 1

    order: list[int] = []
    remaining = set(boxes)
    while remaining:
        frontier = sorted(color for color in remaining if indegree[color] == 0)
        if len(frontier) != 1:
            raise AssertionError("occlusion order must be uniquely determined")
        color = frontier[0]
        order.append(color)
        remaining.remove(color)
        for below in above_to_below[color]:
            indegree[below] -= 1
    return order


def _recover_boxes(grid: np.ndarray, bg: int) -> tuple[dict[int, BBox], list[int]]:
    colors = tuple(colors_present(grid, exclude={bg}))
    assert 3 <= len(colors) <= 4, "expected three or four rectangle colors"

    candidate_map = {color: _candidate_boxes(grid, bg, color) for color in colors}
    assert all(candidate_map[color] for color in colors), "each color needs a valid rectangle bbox"
    search_order = sorted(colors, key=lambda color: (len(candidate_map[color]), color))

    solutions: list[tuple[dict[int, BBox], list[int]]] = []

    def backtrack(index: int, assigned: dict[int, BBox]) -> None:
        if len(solutions) > 1:
            return
        if index == len(search_order):
            try:
                order = _top_order(grid, bg, assigned)
            except AssertionError:
                return
            solutions.append((assigned.copy(), order))
            return

        color = search_order[index]
        for box in candidate_map[color]:
            if not _partial_consistent(grid, assigned, color, box):
                continue
            assigned[color] = box
            backtrack(index + 1, assigned)
            assigned.pop(color)

    backtrack(0, {})
    if len(solutions) != 1:
        raise AssertionError("rectangle reconstruction must be unique")
    return solutions[0]


def _render_scene(shape: tuple[int, int], order_top_to_bottom: list[int], boxes: dict[int, BBox]) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    for color in reversed(order_top_to_bottom):
        paint_coords(grid, _ring_coords(boxes[color]), color)
    return grid


def _render_grown_scene(
    shape: tuple[int, int],
    order_top_to_bottom: list[int],
    boxes: dict[int, BBox],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    depth = {color: i for i, color in enumerate(order_top_to_bottom)}
    for color in reversed(order_top_to_bottom):
        for delta in range(depth[color] + 1):
            paint_coords(out, _ring_coords(_expanded_box(boxes[color], delta)), color)
    return out


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    boxes, order = _recover_boxes(g, bg)
    return _render_grown_scene(g.shape, order, boxes, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapping hollow rectangles, grow each by its occlusion depth, and redraw the layered scene.

    Input structure:
    - A black background contains three or four differently colored 1-pixel hollow rectangles.
    - The rectangles partially overlap, so lower borders may appear only as visible fragments while upper borders remain visible.
    - Each rectangle fits inside a 4x4 to 7x7 bbox, and every hidden border pixel is replaced by another rectangle color.

    Rule:
    - For each color, recover the unique full rectangle whose border contains all visible pixels of that color and whose missing border pixels are explained by other visible colors.
    - A border pixel currently showing color C' on color C's recovered rectangle means C' is above C; topologically sort these relations from top to bottom.
    - Let the top rectangle have depth 0, the next depth 1, and so on; rectangle depth D keeps its original border and adds D outer concentric rectangle borders in the same color.
    - Render the grown rectangles on a black canvas from bottom layer to top layer, clipping any growth that leaves the grid.

    Key insight:
    - Occlusion is used twice: first to reconstruct each hidden hollow rectangle, then to convert layer depth into the number of extra concentric borders that must be grown.
    """
    return _solve(grid)


def _build_layer_boxes(shape: tuple[int, int], layer_count: int) -> list[BBox] | None:
    all_boxes = list(_all_boxes(shape))
    random.shuffle(all_boxes)
    top_box = random.choice(all_boxes)
    boxes = [top_box]
    higher_rings = [set(_ring_set(top_box))]

    for _ in range(1, layer_count):
        prev_ring = higher_rings[-1]
        higher_before_prev = set().union(*higher_rings[:-1]) if len(higher_rings) > 1 else set()
        covered_by_higher = set().union(*higher_rings)
        candidates: list[BBox] = []

        for box in all_boxes:
            if box in boxes:
                continue
            ring = set(_ring_set(box))
            perimeter = len(ring)
            hidden = len(ring & covered_by_higher)
            if hidden == 0 or hidden * 5 > perimeter * 2:
                continue
            if not ((ring & prev_ring) - higher_before_prev):
                continue
            if len(ring & prev_ring) > max(6, perimeter // 3):
                continue
            candidates.append(box)

        if not candidates:
            return None

        box = random.choice(candidates)
        boxes.append(box)
        higher_rings.append(set(_ring_set(box)))

    return boxes


def _expected_output(shape: tuple[int, int], order_top_to_bottom: list[int], boxes: dict[int, BBox]) -> np.ndarray:
    return _render_grown_scene(shape, order_top_to_bottom, boxes, _BG)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*_GRID_H_RANGE)
    w = random.randint(*_GRID_W_RANGE)
    layer_count = random.choice(_LAYER_COUNT_CHOICES)
    boxes_list = _build_layer_boxes((h, w), layer_count)
    if boxes_list is None:
        return None

    colors = random.sample(_COLOR_POOL, layer_count)
    order_top_to_bottom = colors[:]
    boxes = {color: box for color, box in zip(order_top_to_bottom, boxes_list)}
    expected = _expected_output((h, w), order_top_to_bottom, boxes)
    grid = _render_scene((h, w), order_top_to_bottom, boxes)

    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
