import random

import numpy as np


AZURE = 8
ARC_MAX_SIDE = 30
MIN_RECT_COUNT = 3
MAX_RECT_COUNT = 5
MIN_RECT_HEIGHT = 3
MAX_RECT_HEIGHT = 8
MIN_RECT_WIDTH = 3
MAX_RECT_WIDTH = 7
MIN_PADDING = 1
MAX_PADDING = 4
MAX_OVERLAP = 3
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_TRIES_PER_RECT = 80
DIRECTIONS = ("NW", "NE", "SW", "SE")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == AZURE):
        return AZURE
    return _most_common_color(grid)


def _bbox_of_color(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size, f"missing color {color}"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_intersection(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(box_a[0], box_b[0])
    c0 = max(box_a[1], box_b[1])
    r1 = min(box_a[2], box_b[2])
    c1 = min(box_a[3], box_b[3])
    if r0 > r1 or c0 > c1:
        return None
    return r0, c0, r1, c1


def _recover_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    bg = _background_color(grid)
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert MIN_RECT_COUNT <= len(colors) <= MAX_RECT_COUNT

    rectangles: list[dict[str, object]] = []
    for color in colors:
        bbox = _bbox_of_color(grid, color)
        rectangles.append({"color": color, "bbox": bbox})
    return rectangles


def _layer_edges(
    grid: np.ndarray, rectangles: list[dict[str, object]]
) -> list[set[int]]:
    edges = [set() for _ in rectangles]
    for lower_candidate in range(len(rectangles)):
        color_a = int(rectangles[lower_candidate]["color"])
        box_a = rectangles[lower_candidate]["bbox"]
        for upper_candidate in range(lower_candidate + 1, len(rectangles)):
            color_b = int(rectangles[upper_candidate]["color"])
            box_b = rectangles[upper_candidate]["bbox"]
            overlap = _bbox_intersection(box_a, box_b)
            if overlap is None:
                continue

            r0, c0, r1, c1 = overlap
            window = grid[r0 : r1 + 1, c0 : c1 + 1]
            a_visible = bool(np.any(window == color_a))
            b_visible = bool(np.any(window == color_b))

            if a_visible and b_visible:
                raise AssertionError("overlap order is inconsistent")
            if a_visible:
                edges[upper_candidate].add(lower_candidate)
            elif b_visible:
                edges[lower_candidate].add(upper_candidate)
    return edges


def _topological_order(edges: list[set[int]]) -> list[int]:
    indegree = [0] * len(edges)
    for src, targets in enumerate(edges):
        for dst in targets:
            indegree[dst] += 1

    remaining = set(range(len(edges)))
    order: list[int] = []
    while remaining:
        ready = [node for node in remaining if indegree[node] == 0]
        assert len(ready) == 1, "layer order must be uniquely determined"
        node = ready[0]
        order.append(node)
        remaining.remove(node)
        for dst in edges[node]:
            indegree[dst] -= 1
    return order


def _pack_rectangles(
    rectangles: list[dict[str, object]], order: list[int], bg: int
) -> np.ndarray:
    dims = []
    for idx in order:
        r0, c0, r1, c1 = rectangles[idx]["bbox"]
        dims.append((r1 - r0 + 1, c1 - c0 + 1, int(rectangles[idx]["color"])))

    out_h = max(height for height, _width, _color in dims)
    out_w = sum(width for _height, width, _color in dims) + len(dims) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    col = 0
    for height, width, color in dims:
        out[out_h - height :, col : col + width] = color
        col += width + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each layered solid rectangle, infer the drawing order from occlusion,
    then pack the full rectangles from bottom to top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    rectangles = _recover_rectangles(g)
    edges = _layer_edges(g, rectangles)
    order = _topological_order(edges)
    return _pack_rectangles(rectangles, order, bg)


def _rect_bounds(rect: dict[str, int]) -> tuple[int, int, int, int]:
    return rect["r0"], rect["c0"], rect["r0"] + rect["h"], rect["c0"] + rect["w"]


def _bbox_from_rect(rect: dict[str, int]) -> tuple[int, int, int, int]:
    return rect["r0"], rect["c0"], rect["r0"] + rect["h"] - 1, rect["c0"] + rect["w"] - 1


def _intersection_area(rect_a: dict[str, int], rect_b: dict[str, int]) -> int:
    ar0, ac0, ar1, ac1 = _rect_bounds(rect_a)
    br0, bc0, br1, bc1 = _rect_bounds(rect_b)
    h = max(0, min(ar1, br1) - max(ar0, br0))
    w = max(0, min(ac1, bc1) - max(ac0, bc0))
    return h * w


def _place_overlapping_corner(
    prev: dict[str, int], height: int, width: int, ov_h: int, ov_w: int, direction: str
) -> dict[str, int]:
    if direction == "SE":
        r0 = prev["r0"] + prev["h"] - ov_h
        c0 = prev["c0"] + prev["w"] - ov_w
    elif direction == "NE":
        r0 = prev["r0"] - (height - ov_h)
        c0 = prev["c0"] + prev["w"] - ov_w
    elif direction == "SW":
        r0 = prev["r0"] + prev["h"] - ov_h
        c0 = prev["c0"] - (width - ov_w)
    elif direction == "NW":
        r0 = prev["r0"] - (height - ov_h)
        c0 = prev["c0"] - (width - ov_w)
    else:
        raise ValueError(direction)
    return {"r0": r0, "c0": c0, "h": height, "w": width}


def _sample_dimensions(n_rects: int) -> list[tuple[int, int]] | None:
    width_budget = ARC_MAX_SIDE - (n_rects - 1)
    for _ in range(120):
        widths = [random.randint(MIN_RECT_WIDTH, MAX_RECT_WIDTH) for _ in range(n_rects)]
        if sum(widths) > width_budget:
            continue
        heights = [random.randint(MIN_RECT_HEIGHT, MAX_RECT_HEIGHT) for _ in range(n_rects)]
        return list(zip(heights, widths))
    return None


def _translate_rectangles(
    rectangles: list[dict[str, int]], dr: int, dc: int
) -> list[dict[str, int]]:
    shifted = []
    for rect in rectangles:
        shifted.append(
            {
                "r0": rect["r0"] + dr,
                "c0": rect["c0"] + dc,
                "h": rect["h"],
                "w": rect["w"],
                "color": rect["color"],
            }
        )
    return shifted


def _paint_layers(
    rectangles: list[dict[str, int]], shape: tuple[int, int], bg: int
) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for rect in rectangles:
        r0, c0 = rect["r0"], rect["c0"]
        grid[r0 : r0 + rect["h"], c0 : c0 + rect["w"]] = rect["color"]
    return grid


def _expected_output(rectangles: list[dict[str, int]], bg: int) -> np.ndarray:
    packed = []
    for rect in rectangles:
        packed.append({"color": rect["color"], "bbox": _bbox_from_rect(rect)})
    order = list(range(len(packed)))
    return _pack_rectangles(packed, order, bg)


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(MIN_RECT_COUNT, MAX_RECT_COUNT)
    dims = _sample_dimensions(n_rects)
    if dims is None:
        return None

    colors = random.sample([0, 1, 2, 3, 4, 5, 6, 7, 9], n_rects)
    rectangles: list[dict[str, int]] = [
        {"r0": 0, "c0": 0, "h": dims[0][0], "w": dims[0][1], "color": colors[0]}
    ]

    for idx in range(1, n_rects):
        height, width = dims[idx]
        prev = rectangles[-1]
        max_ov_h = min(prev["h"], height) - 1
        max_ov_w = min(prev["w"], width) - 1
        if max_ov_h < 1 or max_ov_w < 1:
            return None

        placed = None
        for _ in range(PLACEMENT_TRIES_PER_RECT):
            direction = random.choice(DIRECTIONS)
            ov_h = random.randint(1, min(max_ov_h, MAX_OVERLAP))
            ov_w = random.randint(1, min(max_ov_w, MAX_OVERLAP))
            candidate = _place_overlapping_corner(prev, height, width, ov_h, ov_w, direction)
            if any(_intersection_area(candidate, older) > 0 for older in rectangles[:-1]):
                continue
            candidate["color"] = colors[idx]
            placed = candidate
            break
        if placed is None:
            return None
        rectangles.append(placed)

    min_r = min(rect["r0"] for rect in rectangles)
    min_c = min(rect["c0"] for rect in rectangles)
    max_r = max(rect["r0"] + rect["h"] for rect in rectangles)
    max_c = max(rect["c0"] + rect["w"] for rect in rectangles)
    span_h = max_r - min_r
    span_w = max_c - min_c

    max_top = min(MAX_PADDING, ARC_MAX_SIDE - span_h - MIN_PADDING)
    max_bottom = min(MAX_PADDING, ARC_MAX_SIDE - span_h - MIN_PADDING)
    max_left = min(MAX_PADDING, ARC_MAX_SIDE - span_w - MIN_PADDING)
    max_right = min(MAX_PADDING, ARC_MAX_SIDE - span_w - MIN_PADDING)
    if min(max_top, max_bottom, max_left, max_right) < MIN_PADDING:
        return None

    pad_top = random.randint(MIN_PADDING, max_top)
    pad_bottom = random.randint(MIN_PADDING, min(max_bottom, ARC_MAX_SIDE - span_h - pad_top))
    pad_left = random.randint(MIN_PADDING, max_left)
    pad_right = random.randint(MIN_PADDING, min(max_right, ARC_MAX_SIDE - span_w - pad_left))

    shifted = _translate_rectangles(rectangles, pad_top - min_r, pad_left - min_c)
    shape = (span_h + pad_top + pad_bottom, span_w + pad_left + pad_right)
    if shape[0] > ARC_MAX_SIDE or shape[1] > ARC_MAX_SIDE:
        return None

    grid = _paint_layers(shifted, shape, AZURE)
    expected = _expected_output(shifted, AZURE)
    if expected.shape[0] > ARC_MAX_SIDE or expected.shape[1] > ARC_MAX_SIDE:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
