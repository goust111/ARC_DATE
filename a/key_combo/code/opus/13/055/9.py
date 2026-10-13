from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


SIDES = ("top", "bottom", "left", "right")
COLOR_POOL = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MIN_RECTS = 3
MAX_RECTS = 5
MIN_SIZE = 3
MAX_SIZE = 8
GEN_ATTEMPTS = 600


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _border_cells(box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _side_cells(box: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    if side == "top":
        return [(r0, c) for c in range(c0, c1 + 1)]
    if side == "bottom":
        return [(r1, c) for c in range(c0, c1 + 1)]
    if side == "left":
        return [(r, c0) for r in range(r0, r1 + 1)]
    if side == "right":
        return [(r, c1) for r in range(r0, r1 + 1)]
    raise ValueError(f"unknown side: {side}")


def _paint_border(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    for r, c in _border_cells(box):
        grid[r, c] = int(color)


def _box_dims(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = map(int, box)
    return r1 - r0 + 1, c1 - c0 + 1


def _box_perimeter_len(box: tuple[int, int, int, int]) -> int:
    h, w = _box_dims(box)
    return 2 * (h + w) - 4


def _reconstruct_boxes(
    grid: np.ndarray, colors: list[int]
) -> dict[int, tuple[int, int, int, int]]:
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        coords = np.argwhere(grid == int(color))
        assert coords.size > 0, "every rectangle color must remain at least partly visible"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        box = (int(r0), int(c0), int(r1), int(c1))
        h, w = _box_dims(box)
        assert MIN_SIZE <= h <= MAX_SIZE and MIN_SIZE <= w <= MAX_SIZE

        visible = {(int(r), int(c)) for r, c in coords}
        border = _border_cells(box)
        assert visible.issubset(border), "visible pixels must lie on the inferred hollow border"
        for side in SIDES:
            assert any(cell in visible for cell in _side_cells(box, side)), (
                "each side must have some visible evidence so the bbox is recoverable"
            )
        assert any(
            all(int(grid[r, c]) == int(color) for r, c in _side_cells(box, side))
            for side in SIDES
        ), "each rectangle needs one fully visible edge"
        boxes[int(color)] = box
    return boxes


def _pairwise_front(
    grid: np.ndarray,
    color_a: int,
    box_a: tuple[int, int, int, int],
    color_b: int,
    box_b: tuple[int, int, int, int],
) -> tuple[int, int]:
    overlap = _border_cells(box_a) & _border_cells(box_b)
    assert overlap, "every pair of rectangles must theoretically overlap on the border"
    winners = {int(grid[r, c]) for r, c in overlap if int(grid[r, c]) in (int(color_a), int(color_b))}
    assert winners, "at least one pairwise overlap point must remain visible for direct comparison"
    assert len(winners) == 1, "a pair of rectangles cannot contradict itself about layer order"
    winner = winners.pop()
    if winner == int(color_a):
        return int(color_a), int(color_b)
    return int(color_b), int(color_a)


def _topological_order(colors: list[int], comparisons: list[tuple[int, int]]) -> list[int]:
    edges: dict[int, set[int]] = defaultdict(set)
    indeg = {int(color): 0 for color in colors}
    for front, back in comparisons:
        if int(back) not in edges[int(front)]:
            edges[int(front)].add(int(back))
            indeg[int(back)] += 1

    order: list[int] = []
    remaining = set(map(int, colors))
    while remaining:
        layer = sorted(color for color in remaining if indeg[color] == 0)
        assert len(layer) == 1, "pairwise comparisons must determine a unique total order"
        color = layer[0]
        order.append(int(color))
        remaining.remove(color)
        for nxt in edges[color]:
            indeg[nxt] -= 1
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapped hollow rectangles and output their front-to-back color order.

    Input structure:
    - A black-like background contains 3 to 5 axis-aligned hollow rectangles with
      unique colors and 1-pixel-thick borders.
    - The rectangles have already been overlaid into one final image, so lower
      borders may be partially hidden wherever higher borders pass through them.
    - Every rectangle still has enough visible border fragments to reconstruct its
      full bounding box, and every pair of rectangles has at least one border-overlap
      location that reveals which one is in front.

    Rule:
    - For each visible color, reconstruct the full hollow rectangle by taking the
      bounding box of its surviving border pixels and treating that box as the
      rectangle outline.
    - For every pair of reconstructed rectangles, inspect their theoretical border
      overlap cells; whichever color is visible there must be the front rectangle.
    - Combine all pairwise front/back relations into a unique total order.
    - Output a 1-row grid whose cells list the rectangle colors from frontmost to backmost.

    Key insight:
    - The visible image is not the answer itself: you must first rebuild each hidden
      hollow rectangle, then compare colors only at the reconstructed overlap points
      to decode the layer stack.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != int(bg))
    assert MIN_RECTS <= len(colors) <= MAX_RECTS

    boxes = _reconstruct_boxes(g, colors)
    comparisons: list[tuple[int, int]] = []
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            comparisons.append(_pairwise_front(g, color_a, boxes[color_a], color_b, boxes[color_b]))
    order = _topological_order(colors, comparisons)
    return np.array([order], dtype=int)


def _enumerate_boxes(
    height: int,
    width: int,
    hub: tuple[int, int],
    used_rows: set[int],
    used_cols: set[int],
    higher_boxes: list[tuple[int, int, int, int]],
    exposed_side: str | None,
) -> list[tuple[int, int, int, int]]:
    hub_r, hub_c = map(int, hub)
    top_limit = min((box[0] for box in higher_boxes), default=hub_r)
    bottom_limit = max((box[2] for box in higher_boxes), default=hub_r)
    left_limit = min((box[1] for box in higher_boxes), default=hub_c)
    right_limit = max((box[3] for box in higher_boxes), default=hub_c)

    boxes = []
    for r0 in range(1, hub_r):
        if r0 in used_rows:
            continue
        for r1 in range(hub_r + 1, height - 1):
            if r1 in used_rows:
                continue
            if not (MIN_SIZE <= r1 - r0 + 1 <= MAX_SIZE):
                continue
            if exposed_side == "top" and higher_boxes and not (r0 < top_limit):
                continue
            if exposed_side == "bottom" and higher_boxes and not (r1 > bottom_limit):
                continue
            for c0 in range(1, hub_c):
                if c0 in used_cols:
                    continue
                for c1 in range(hub_c + 1, width - 1):
                    if c1 in used_cols:
                        continue
                    if not (MIN_SIZE <= c1 - c0 + 1 <= MAX_SIZE):
                        continue
                    if exposed_side == "left" and higher_boxes and not (c0 < left_limit):
                        continue
                    if exposed_side == "right" and higher_boxes and not (c1 > right_limit):
                        continue
                    boxes.append((int(r0), int(c0), int(r1), int(c1)))
    random.shuffle(boxes)
    return boxes


def _render_scene(
    shape: tuple[int, int],
    order_colors: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for color, box in zip(reversed(order_colors), reversed(boxes)):
        _paint_border(grid, box, color)
    return grid


def _boxes_overlap_on_border(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> bool:
    return bool(_border_cells(box_a) & _border_cells(box_b))


def _feature_shortcut(order_colors: list[int], boxes: dict[int, tuple[int, int, int, int]], visible_counts: dict[int, int]) -> bool:
    def area(color: int) -> int:
        h, w = _box_dims(boxes[color])
        return h * w

    def height(color: int) -> int:
        return _box_dims(boxes[color])[0]

    def width(color: int) -> int:
        return _box_dims(boxes[color])[1]

    features = [
        lambda c: boxes[c][0],
        lambda c: boxes[c][2],
        lambda c: boxes[c][1],
        lambda c: boxes[c][3],
        height,
        width,
        area,
        lambda c: _box_perimeter_len(boxes[c]),
        lambda c: visible_counts[c],
        lambda c: c,
    ]
    target = list(map(int, order_colors))
    for key_fn in features:
        if sorted(target, key=key_fn) == target:
            return True
        if sorted(target, key=key_fn, reverse=True) == target:
            return True
    return False


def _semantic_checks(
    grid: np.ndarray,
    order_colors: list[int],
    boxes: list[tuple[int, int, int, int]],
    exposed_sides: list[str],
) -> bool:
    reconstructed = _reconstruct_boxes(grid, order_colors)
    for color, box in zip(order_colors, boxes):
        if reconstructed[int(color)] != tuple(map(int, box)):
            return False

    visible_counts = {int(color): int(np.count_nonzero(grid == int(color))) for color in order_colors}
    for idx, (color, box, side) in enumerate(zip(order_colors, boxes, exposed_sides)):
        if idx > 0 and all(int(grid[r, c]) == int(color) for r, c in _side_cells(box, side)):
            pass
        elif idx > 0:
            return False
        if idx > 0 and visible_counts[int(color)] >= _box_perimeter_len(box):
            return False

    pair_comparisons = []
    for i, color_a in enumerate(order_colors):
        for j in range(i + 1, len(order_colors)):
            color_b = order_colors[j]
            front, back = _pairwise_front(grid, color_a, reconstructed[color_a], color_b, reconstructed[color_b])
            if front != int(color_a) or back != int(color_b):
                return False
            pair_comparisons.append((front, back))
    if _topological_order(order_colors, pair_comparisons) != list(map(int, order_colors)):
        return False

    if len(set(exposed_sides[1:])) < 2:
        return False
    if _feature_shortcut(order_colors, reconstructed, visible_counts):
        return False
    return bool(np.array_equal(main(grid), np.array([order_colors], dtype=int)))


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(MIN_RECTS, MAX_RECTS)
    min_span = max(13, 2 * n_rects + 3)
    if min_span > 16:
        return None

    height = random.randint(min_span, 16)
    width = random.randint(min_span, 16)
    hub_r = random.randint(n_rects + 1, height - n_rects - 2)
    hub_c = random.randint(n_rects + 1, width - n_rects - 2)

    order_colors = random.sample(COLOR_POOL, n_rects)
    exposed_sides: list[str] = []
    boxes: list[tuple[int, int, int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()

    side_plan = [None]
    side_choices = list(SIDES)
    random.shuffle(side_choices)
    side_plan.extend(side_choices[: min(len(side_choices), n_rects - 1)])
    while len(side_plan) < n_rects:
        side_plan.append(random.choice(SIDES))

    for idx in range(n_rects):
        exposed_side = side_plan[idx]
        if idx > 0:
            random.shuffle(side_choices)
            if len(set(side_plan[1:])) < 2:
                need = [side for side in SIDES if side not in set(side_plan[1:idx])]
                if need:
                    exposed_side = random.choice(need)
        candidates = _enumerate_boxes(
            height,
            width,
            (hub_r, hub_c),
            used_rows,
            used_cols,
            boxes,
            exposed_side,
        )
        if not candidates:
            return None
        chosen = None
        for box in candidates:
            if box in boxes:
                continue
            if any(not _boxes_overlap_on_border(box, higher_box) for higher_box in boxes):
                continue
            chosen = box
            break
        if chosen is None:
            return None
        boxes.append(chosen)
        exposed_sides.append("top" if exposed_side is None else exposed_side)
        used_rows.update((chosen[0], chosen[2]))
        used_cols.update((chosen[1], chosen[3]))

    grid = _render_scene((height, width), order_colors, boxes)
    if not _semantic_checks(grid, order_colors, boxes, exposed_sides):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid rectangle-layer input")
