from __future__ import annotations

import random
import sys

import numpy as np

try:
    from arc_utils import bbox_coords, colors_present, get_bg
except ImportError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_coords, colors_present, get_bg


_GRID_LIMIT = 30
_GEN_ATTEMPTS = 500


def _box_shape(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    return r1 - r0 + 1, c1 - c0 + 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _visible_bboxes(g: np.ndarray, bg: int) -> dict[int, tuple[int, int, int, int]]:
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        assert coords.size, "each foreground color must be visible"
        boxes[int(color)] = bbox_coords(coords)
    assert boxes, "expected at least two colored rectangle fragments"
    return boxes


def _candidate_boxes_for_color(
    g: np.ndarray,
    color: int,
    bg: int,
    rect_h: int,
    rect_w: int,
) -> list[tuple[int, int, int, int]]:
    coords = np.argwhere(g == int(color))
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h0 = int(r1 - r0 + 1)
    w0 = int(c1 - c0 + 1)
    if rect_h < h0 or rect_w < w0:
        return []

    H, W = g.shape
    row_lo = max(0, int(r1) - rect_h + 1)
    row_hi = min(int(r0), H - rect_h)
    col_lo = max(0, int(c1) - rect_w + 1)
    col_hi = min(int(c0), W - rect_w)
    if row_lo > row_hi or col_lo > col_hi:
        return []

    out: list[tuple[int, int, int, int]] = []
    for rr in range(row_lo, row_hi + 1):
        for cc in range(col_lo, col_hi + 1):
            region = g[rr : rr + rect_h, cc : cc + rect_w]
            if np.any(region == int(bg)):
                continue
            out.append((rr, cc, rr + rect_h - 1, cc + rect_w - 1))
    return out


def _infer_order(
    g: np.ndarray,
    bg: int,
    boxes: dict[int, tuple[int, int, int, int]],
) -> list[int] | None:
    colors = sorted(boxes)
    above_to_below = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    edges: set[tuple[int, int]] = set()

    shape_set = {_box_shape(box) for box in boxes.values()}
    if len(shape_set) != 1:
        return None

    for color, (r0, c0, r1, c1) in boxes.items():
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(region == int(bg)):
            return None
        coverers = set(map(int, np.unique(region))) - {int(bg), int(color)}
        for upper in coverers:
            if (upper, color) in edges:
                continue
            edges.add((upper, color))
            above_to_below[upper].add(color)
            indegree[color] += 1

    if len(colors) > 1 and len(edges) < len(colors) - 1:
        return None

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        tops = sorted(color for color in remaining if indegree[color] == 0)
        if len(tops) != 1:
            return None
        top = tops[0]
        order.append(top)
        remaining.remove(top)
        for below in above_to_below[top]:
            indegree[below] -= 1
    return order


def _search_boxes(
    g: np.ndarray,
    bg: int,
    candidate_map: dict[int, list[tuple[int, int, int, int]]],
) -> tuple[dict[int, tuple[int, int, int, int]], list[int]] | None:
    colors = sorted(candidate_map, key=lambda color: (len(candidate_map[color]), color))
    chosen: dict[int, tuple[int, int, int, int]] = {}

    def backtrack(idx: int):
        if idx == len(colors):
            order = _infer_order(g, bg, chosen)
            if order is None:
                return None
            return chosen.copy(), order

        color = colors[idx]
        for box in candidate_map[color]:
            chosen[color] = box
            solved = backtrack(idx + 1)
            if solved is not None:
                return solved
        chosen.pop(color, None)
        return None

    return backtrack(0)


def _solve_scene(g: np.ndarray) -> tuple[int, dict[int, tuple[int, int, int, int]], list[int]]:
    bg = get_bg(g)
    visible_boxes = _visible_bboxes(g, bg)

    direct_order = _infer_order(g, bg, visible_boxes)
    if direct_order is not None:
        return int(bg), visible_boxes, direct_order

    H, W = g.shape
    box_shapes = [_box_shape(box) for box in visible_boxes.values()]
    min_h = max(h for h, _ in box_shapes)
    min_w = max(w for _, w in box_shapes)
    sizes = sorted((h, w) for h in range(min_h, H + 1) for w in range(min_w, W + 1))

    for rect_h, rect_w in sizes:
        candidate_map: dict[int, list[tuple[int, int, int, int]]] = {}
        failed = False
        for color in visible_boxes:
            candidates = _candidate_boxes_for_color(g, color, bg, rect_h, rect_w)
            if not candidates:
                failed = True
                break
            candidate_map[color] = candidates
        if failed:
            continue

        solved = _search_boxes(g, bg, candidate_map)
        if solved is not None:
            boxes, order = solved
            return int(bg), boxes, order

    raise AssertionError("failed to recover a unique equal-size rectangle layer chain")


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recover the top-to-bottom rectangle order, then cycle colors one layer downward.

    Input structure:
    - The dominant color is the background.
    - Every other color is the visible fragment of one filled axis-aligned rectangle.
    - All rectangles have the same size, and visible foreign pixels inside one
      rectangle's footprint come from rectangles above it in a single layer chain.

    Rule:
    - Reconstruct the common rectangle footprints and infer the unique
      top-to-bottom order from which colors visibly occupy whose footprint.
    - Keep every visible foreground pixel exactly where it is.
    - Recolor each rectangle's visible pixels with the color of the rectangle
      immediately below it; the bottom rectangle takes the top rectangle's color.
    """
    g = np.asarray(input_grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    bg, _boxes, order = _solve_scene(g)

    color_map = {
        order[idx]: order[(idx + 1) % len(order)]
        for idx in range(len(order))
    }
    out = g.copy()
    for src, dst in color_map.items():
        out[g == int(src)] = int(dst)
    out[g == int(bg)] = int(bg)
    return out


def _render_scene(
    shape: tuple[int, int],
    bg: int,
    order: list[int],
    boxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    g = np.full(shape, int(bg), dtype=int)
    for color, (r0, c0, r1, c1) in reversed(list(zip(order, boxes))):
        g[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return g


def _sample_chain_boxes(
    rect_count: int,
    rect_h: int,
    rect_w: int,
) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    for _ in range(200):
        positions = [(0, 0)]
        ok = True
        for _layer in range(1, rect_count):
            placed = False
            for _inner in range(200):
                dr = random.choice([d for d in range(-(rect_h - 1), rect_h) if d != 0])
                dc = random.choice([d for d in range(-(rect_w - 1), rect_w) if d != 0])
                nr = positions[-1][0] + dr
                nc = positions[-1][1] + dc
                candidate = (nr, nc, nr + rect_h - 1, nc + rect_w - 1)
                earlier_boxes = [
                    (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
                    for r0, c0 in positions[:-1]
                ]
                if any(_boxes_overlap(candidate, box) for box in earlier_boxes):
                    continue
                positions.append((nr, nc))
                placed = True
                break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        row_lo = min(r for r, _ in positions)
        col_lo = min(c for _, c in positions)
        shifted = [(r - row_lo, c - col_lo) for r, c in positions]
        max_row = max(r for r, _ in shifted) + rect_h
        max_col = max(c for _, c in shifted) + rect_w
        pad_top = random.randint(0, max(0, _GRID_LIMIT - max_row))
        pad_left = random.randint(0, max(0, _GRID_LIMIT - max_col))
        shifted = [(r + pad_top, c + pad_left) for r, c in shifted]
        max_row = max(r for r, _ in shifted) + rect_h
        max_col = max(c for _, c in shifted) + rect_w
        if max_row > _GRID_LIMIT or max_col > _GRID_LIMIT:
            continue

        boxes = [(r, c, r + rect_h - 1, c + rect_w - 1) for r, c in shifted]
        good = True
        for idx in range(rect_count - 1):
            if not _boxes_overlap(boxes[idx], boxes[idx + 1]):
                good = False
                break
        for i in range(rect_count):
            for j in range(i + 2, rect_count):
                if _boxes_overlap(boxes[i], boxes[j]):
                    good = False
                    break
            if not good:
                break
        if not good:
            continue

        return (max_row, max_col), boxes
    return None


def _generate_one() -> tuple[np.ndarray, list[int]]:
    bg = random.randrange(10)
    rect_count = random.randint(2, 5)
    palette = [color for color in range(10) if color != bg]
    order = random.sample(palette, rect_count)

    for _ in range(_GEN_ATTEMPTS):
        rect_h = random.randint(4, 8)
        rect_w = random.randint(4, 8)
        sampled = _sample_chain_boxes(rect_count, rect_h, rect_w)
        if sampled is None:
            continue
        shape, boxes = sampled
        grid = _render_scene(shape, bg, order, boxes)

        try:
            solved_bg, _solved_boxes, solved_order = _solve_scene(grid)
        except AssertionError:
            continue
        if solved_bg != bg or solved_order != order:
            continue
        return grid, order

    raise AssertionError("generator failed to sample a solver-consistent rectangle chain")


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid, _order = _generate_one()
        if grid.ndim != 2:
            continue
        if grid.shape[0] > _GRID_LIMIT or grid.shape[1] > _GRID_LIMIT:
            continue
        return grid
    raise AssertionError("failed to generate a valid ARC input grid")
