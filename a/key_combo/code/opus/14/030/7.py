from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
RECT_COUNT_CHOICES = (4, 5)
SIZE_RANGE = range(3, 7)
PAD_VALUES = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 400
CORNER_NAMES = ("TL", "TR", "BL", "BR")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _intersect(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _kahn_unique_order(nodes: list[int], edges: set[tuple[int, int]]) -> list[int]:
    succ = {node: set() for node in nodes}
    indeg = {node: 0 for node in nodes}
    for low, high in edges:
        if high not in succ[low]:
            succ[low].add(high)
            indeg[high] += 1

    ready = sorted(node for node in nodes if indeg[node] == 0)
    order: list[int] = []
    while ready:
        if len(ready) != 1:
            raise AssertionError("stacking order is not uniquely determined")
        node = ready.pop(0)
        order.append(node)
        for nxt in sorted(succ[node]):
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
        ready.sort()

    if len(order) != len(nodes):
        raise AssertionError("stacking constraints contain a cycle")
    return order


def _solve_order(grid: np.ndarray) -> list[int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert 4 <= len(colors) <= 5, "expected 4 or 5 visible rectangle colors"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.size > 0, "each rectangle color must be visible"
        bboxes[color] = _bbox_from_coords(coords)

    edges: set[tuple[int, int]] = set()
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            inter = _intersect(bboxes[color_a], bboxes[color_b])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            window = g[r0 : r1 + 1, c0 : c1 + 1]
            sees_a = bool(np.any(window == color_a))
            sees_b = bool(np.any(window == color_b))
            if sees_a and sees_b:
                raise AssertionError("overlap region cannot show both rectangle colors")
            if sees_a:
                edges.add((color_b, color_a))
            elif sees_b:
                edges.add((color_a, color_b))

    assert len(edges) >= 3, "expected at least three direct overlap comparisons"
    return _kahn_unique_order(colors, edges)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the bottom-to-top stacking order of overlapping solid rectangles and output it as a compressed color row.

    Input structure:
    - The background is black.
    - There are 4 or 5 distinct non-black colors, each belonging to one solid axis-aligned rectangle.
    - Rectangles partially overlap, so only the topmost color is visible in each covered cell, but every rectangle still keeps some visible area.

    Rule:
    - For each visible color, infer its full rectangle from the bounding box of all pixels of that color.
    - For every pair of inferred rectangles whose bounding boxes overlap, inspect the intersection region: if exactly one of the two colors is visible there, that rectangle is above the other.
    - Combine all such pairwise comparisons into the unique total order and output a 1xN grid listing the rectangle colors from bottom layer to top layer.

    Key insight:
    - The final picture is the result of overwriting solid rectangles, so each bbox intersection encodes a local priority comparison, and those local comparisons can be merged into the full stacking sequence.
    """
    order = _solve_order(np.asarray(grid, dtype=int))
    return np.asarray([order], dtype=int)


def _corner_position(
    prev: tuple[int, int, int, int],
    cur_h: int,
    cur_w: int,
    corner: str,
    overlap_h: int,
    overlap_w: int,
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = prev
    if corner == "TL":
        return pr0 + overlap_h - cur_h, pc0 + overlap_w - cur_w
    if corner == "TR":
        return pr0 + overlap_h - cur_h, pc1 - overlap_w + 1
    if corner == "BL":
        return pr1 - overlap_h + 1, pc0 + overlap_w - cur_w
    if corner == "BR":
        return pr1 - overlap_h + 1, pc1 - overlap_w + 1
    raise ValueError(f"unknown corner {corner}")


def _render(
    rects: list[dict[str, int]], shape: tuple[int, int], bg: int = BACKGROUND
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for rect in rects:
        r0 = int(rect["r0"])
        c0 = int(rect["c0"])
        h = int(rect["h"])
        w = int(rect["w"])
        color = int(rect["color"])
        grid[r0 : r0 + h, c0 : c0 + w] = color
    return grid


def _visible_bbox_matches(grid: np.ndarray, rect: dict[str, int]) -> bool:
    color = int(rect["color"])
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return False
    visible_bbox = _bbox_from_coords(coords)
    full_bbox = (
        int(rect["r0"]),
        int(rect["c0"]),
        int(rect["r0"] + rect["h"] - 1),
        int(rect["c0"] + rect["w"] - 1),
    )
    return visible_bbox == full_bbox


def _actual_edges(rects: list[dict[str, int]], grid: np.ndarray) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for i, low in enumerate(rects):
        low_bbox = (
            int(low["r0"]),
            int(low["c0"]),
            int(low["r0"] + low["h"] - 1),
            int(low["c0"] + low["w"] - 1),
        )
        for high in rects[i + 1 :]:
            high_bbox = (
                int(high["r0"]),
                int(high["c0"]),
                int(high["r0"] + high["h"] - 1),
                int(high["c0"] + high["w"] - 1),
            )
            inter = _intersect(low_bbox, high_bbox)
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            pair_colors = {int(low["color"]), int(high["color"])}
            visible_pair = {int(v) for v in np.unique(window) if int(v) in pair_colors}
            if visible_pair != {int(high["color"])}:
                return set()
            edges.add((int(low["color"]), int(high["color"])))
    return edges


def _nontrivial_visibility(rects: list[dict[str, int]], grid: np.ndarray) -> bool:
    for rect in rects:
        color = int(rect["color"])
        visible = int(np.count_nonzero(grid == color))
        if visible < 3:
            return False
        if not _visible_bbox_matches(grid, rect):
            return False
    return True


def _sequence_is_monotone(values: list[int]) -> bool:
    return values == sorted(values) or values == sorted(values, reverse=True)


def _rect_union_area(rects: list[dict[str, int]]) -> int:
    min_r = min(int(rect["r0"]) for rect in rects)
    min_c = min(int(rect["c0"]) for rect in rects)
    max_r = max(int(rect["r0"] + rect["h"] - 1) for rect in rects)
    max_c = max(int(rect["c0"] + rect["w"] - 1) for rect in rects)
    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for rect in rects:
        r0 = int(rect["r0"]) - min_r
        c0 = int(rect["c0"]) - min_c
        mask[r0 : r0 + int(rect["h"]), c0 : c0 + int(rect["w"])] = True
    return int(mask.sum())


def _normalize_and_embed(rects: list[dict[str, int]]) -> tuple[list[dict[str, int]], tuple[int, int]] | None:
    min_r = min(int(rect["r0"]) for rect in rects)
    min_c = min(int(rect["c0"]) for rect in rects)
    max_r = max(int(rect["r0"] + rect["h"] - 1) for rect in rects)
    max_c = max(int(rect["c0"] + rect["w"] - 1) for rect in rects)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1
    if span_h > 16 or span_w > 16:
        return None

    pad_top = random.choice(PAD_VALUES)
    pad_left = random.choice(PAD_VALUES)
    pad_bottom = random.choice(PAD_VALUES)
    pad_right = random.choice(PAD_VALUES)
    out_h = span_h + pad_top + pad_bottom
    out_w = span_w + pad_left + pad_right
    if out_h > 18 or out_w > 18:
        return None

    shifted: list[dict[str, int]] = []
    for rect in rects:
        shifted.append(
            {
                **rect,
                "r0": int(rect["r0"]) - min_r + pad_top,
                "c0": int(rect["c0"]) - min_c + pad_left,
            }
        )
    return shifted, (out_h, out_w)


def _generate_chain(n_rects: int) -> tuple[list[dict[str, int]], tuple[int, int]] | None:
    colors = random.sample(PALETTE, n_rects)
    if _sequence_is_monotone(colors):
        return None

    sizes = [(random.choice(SIZE_RANGE), random.choice(SIZE_RANGE)) for _ in range(n_rects)]
    areas = [h * w for h, w in sizes]
    if _sequence_is_monotone(areas):
        return None

    rects: list[dict[str, int]] = [
        {"r0": 0, "c0": 0, "h": sizes[0][0], "w": sizes[0][1], "color": colors[0]}
    ]
    corners = [random.choice(CORNER_NAMES) for _ in range(n_rects - 1)]
    if len(set(corners)) == 1:
        return None

    for idx in range(1, n_rects):
        prev = rects[-1]
        cur_h, cur_w = sizes[idx]
        max_overlap_h = min(int(prev["h"]) - 1, cur_h - 1, 3)
        max_overlap_w = min(int(prev["w"]) - 1, cur_w - 1, 3)
        if max_overlap_h < 1 or max_overlap_w < 1:
            return None
        overlap_h = random.randint(1, max_overlap_h)
        overlap_w = random.randint(1, max_overlap_w)
        prev_bbox = (
            int(prev["r0"]),
            int(prev["c0"]),
            int(prev["r0"] + prev["h"] - 1),
            int(prev["c0"] + prev["w"] - 1),
        )
        r0, c0 = _corner_position(prev_bbox, cur_h, cur_w, corners[idx - 1], overlap_h, overlap_w)
        rects.append({"r0": r0, "c0": c0, "h": cur_h, "w": cur_w, "color": colors[idx]})

    embedded = _normalize_and_embed(rects)
    if embedded is None:
        return None
    return embedded


def _semantic_checks(rects: list[dict[str, int]], shape: tuple[int, int]) -> np.ndarray | None:
    grid = _render(rects, shape)
    if _get_bg(grid) != BACKGROUND:
        return None
    if not _nontrivial_visibility(rects, grid):
        return None

    actual_edges = _actual_edges(rects, grid)
    if not actual_edges:
        return None

    order_colors = [int(rect["color"]) for rect in rects]
    expected_edges = {(order_colors[i], order_colors[i + 1]) for i in range(len(order_colors) - 1)}
    if actual_edges != expected_edges:
        return None
    if len(actual_edges) < 3:
        return None

    union_area = _rect_union_area(rects)
    total_area = sum(int(rect["h"]) * int(rect["w"]) for rect in rects)
    overlap_area = total_area - union_area
    if overlap_area < len(rects) - 1 or overlap_area > total_area // 2:
        return None

    try:
        solved_order = _solve_order(grid)
    except Exception:
        return None
    if solved_order != order_colors:
        return None

    out = main(grid)
    if out.shape != (1, len(rects)) or list(map(int, out[0])) != order_colors:
        return None
    return grid


def _gen_attempt(n_rects: int) -> np.ndarray | None:
    generated = _generate_chain(n_rects)
    if generated is None:
        return None
    rects, shape = generated
    return _semantic_checks(rects, shape)


def generate_input() -> np.ndarray:
    target_n = random.choice(RECT_COUNT_CHOICES)
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(target_n)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid overlapping-rectangle decoding input")
