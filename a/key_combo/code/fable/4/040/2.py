from __future__ import annotations

import itertools

import numpy as np


BBox = tuple[int, int, int, int]


def _bbox(mask: np.ndarray) -> BBox:
    """Return the inclusive (row, column) bbox of a nonempty mask."""
    coords = np.argwhere(mask)
    assert coords.size, "expected a nonempty color mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangles(g: np.ndarray) -> dict[int, BBox]:
    colors = [int(color) for color in np.unique(g) if int(color) != 0]
    assert len(colors) >= 2, "expected at least two colored rectangles"
    return {color: _bbox(g == color) for color in colors}


def _occlusion_edges(g: np.ndarray, boxes: dict[int, BBox]) -> set[tuple[int, int]]:
    """Infer A -> B whenever a visible A pixel lies inside B's rectangle."""
    edges: set[tuple[int, int]] = set()
    for upper in boxes:
        upper_cells = np.argwhere(g == upper)
        for lower, (r0, c0, r1, c1) in boxes.items():
            if upper == lower:
                continue
            inside = (
                (upper_cells[:, 0] >= r0)
                & (upper_cells[:, 0] <= r1)
                & (upper_cells[:, 1] >= c0)
                & (upper_cells[:, 1] <= c1)
            )
            if bool(np.any(inside)):
                edges.add((upper, lower))
    return edges


def _unique_chain(colors: set[int], edges: set[tuple[int, int]]) -> list[int]:
    """Return the unique top-to-bottom topological order."""
    remaining = set(colors)
    indegree = {color: 0 for color in colors}
    followers = {color: set() for color in colors}
    for upper, lower in edges:
        assert upper in colors and lower in colors and upper != lower
        if lower not in followers[upper]:
            followers[upper].add(lower)
            indegree[lower] += 1

    order: list[int] = []
    while remaining:
        choices = [color for color in remaining if indegree[color] == 0]
        assert len(choices) == 1, "occlusion evidence must determine one total chain"
        color = choices[0]
        order.append(color)
        remaining.remove(color)
        for lower in followers[color]:
            indegree[lower] -= 1

    assert len(order) == len(colors), "occlusion relation must be acyclic"
    return order


def _render_stack(
    boxes: dict[int, BBox], order: list[int], align_top: bool, align_left: bool
) -> np.ndarray:
    heights = {color: box[2] - box[0] + 1 for color, box in boxes.items()}
    widths = {color: box[3] - box[1] + 1 for color, box in boxes.items()}
    out_h = max(heights.values())
    out_w = max(widths.values())
    out = np.zeros((out_h, out_w), dtype=int)
    for color in order:
        height, width = heights[color], widths[color]
        r0 = 0 if align_top else out_h - height
        c0 = 0 if align_left else out_w - width
        out[r0 : r0 + height, c0 : c0 + width] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Restack the rectangles in reverse occlusion order at the corner selected by the chain endpoints.

    Input structure:
    - A black background contains differently colored filled rectangles whose overlaps show a unique top-to-bottom occlusion chain.
    - Each color's visible pixels retain the full rectangle bbox, and the top endpoint is diagonally offset from the bottom endpoint.

    Rule:
    - Recover every rectangle from its visible-color bbox and infer A above B when a visible A pixel lies inside B's bbox.
    - Make a max-height by max-width black canvas, align all rectangles to the corner indicated by the top rectangle relative to the bottom one, and paint top first through bottom last.

    Key insight:
    - Layer order comes from pairwise occlusion evidence rather than area; only after finding the chain endpoints can their relative row and column positions select the alignment corner.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"
    assert np.any(g == 0), "expected black background"

    boxes = _rectangles(g)
    edges = _occlusion_edges(g, boxes)
    order = _unique_chain(set(boxes), edges)

    top_box = boxes[order[0]]
    bottom_box = boxes[order[-1]]
    top_center_r2 = top_box[0] + top_box[2]
    top_center_c2 = top_box[1] + top_box[3]
    bottom_center_r2 = bottom_box[0] + bottom_box[2]
    bottom_center_c2 = bottom_box[1] + bottom_box[3]
    assert top_center_r2 != bottom_center_r2 and top_center_c2 != bottom_center_c2

    return _render_stack(
        boxes,
        order,
        align_top=top_center_r2 < bottom_center_r2,
        align_left=top_center_c2 < bottom_center_c2,
    )


# Generator geometry.  Every pair is Pareto-incomparable, so no rectangle can
# erase another completely after corner alignment.  Areas are all distinct.
_DIMENSION_POOL = ((6, 11), (7, 10), (8, 9), (9, 7), (10, 6))
_STEP = 6
_MAX_ATTEMPTS = 600
_PLACEMENT_ATTEMPTS = 300


def _boxes_overlap(a: BBox, b: BBox) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _is_nonmonotone(values: list[int]) -> bool:
    deltas = [b - a for a, b in zip(values, values[1:])]
    return any(delta > 0 for delta in deltas) and any(delta < 0 for delta in deltas)


def _choose_dimensions(rng: np.random.Generator, count: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        chosen_indices = rng.choice(len(_DIMENSION_POOL), size=count, replace=False)
        dims = [_DIMENSION_POOL[int(index)] for index in chosen_indices]
        rng.shuffle(dims)
        areas = [height * width for height, width in dims]
        if len(set(areas)) != count:
            continue
        if areas == sorted(areas) or areas == sorted(areas, reverse=True):
            continue
        return dims
    return None


def _signed_steps(rng: np.random.Generator, count: int, endpoint_sign: int) -> list[int]:
    # One contrary step makes the coordinate sequence visibly nonmonotone while
    # a strict majority fixes the endpoint side without using color IDs.
    steps = [endpoint_sign] * (count - 2) + [-endpoint_sign]
    rng.shuffle(steps)
    return steps


def _place_chain(
    rng: np.random.Generator, dims: list[tuple[int, int]]
) -> tuple[list[BBox], bool, bool] | None:
    count = len(dims)
    for _ in range(_PLACEMENT_ATTEMPTS):
        endpoint_row_sign = int(rng.choice((-1, 1)))
        endpoint_col_sign = int(rng.choice((-1, 1)))
        row_steps = _signed_steps(rng, count, endpoint_row_sign)
        col_steps = _signed_steps(rng, count, endpoint_col_sign)

        positions = [(0, 0)]
        for dr, dc in zip(row_steps, col_steps):
            r, c = positions[-1]
            positions.append((r + dr * _STEP, c + dc * _STEP))

        raw_boxes = [
            (r, c, r + height - 1, c + width - 1)
            for (r, c), (height, width) in zip(positions, dims)
        ]
        if not all(_boxes_overlap(raw_boxes[i], raw_boxes[i + 1]) for i in range(count - 1)):
            continue
        if any(
            _boxes_overlap(raw_boxes[i], raw_boxes[j])
            for i in range(count)
            for j in range(i + 2, count)
        ):
            continue

        center_rows = [box[0] + box[2] for box in raw_boxes]
        center_cols = [box[1] + box[3] for box in raw_boxes]
        if not _is_nonmonotone(center_rows) or not _is_nonmonotone(center_cols):
            continue
        if (center_rows[-1] - center_rows[0]) * endpoint_row_sign <= 0:
            continue
        if (center_cols[-1] - center_cols[0]) * endpoint_col_sign <= 0:
            continue

        min_r = min(box[0] for box in raw_boxes)
        min_c = min(box[1] for box in raw_boxes)
        max_r = max(box[2] for box in raw_boxes)
        max_c = max(box[3] for box in raw_boxes)
        span_h, span_w = max_r - min_r + 1, max_c - min_c + 1
        if span_h > 28 or span_w > 28:
            continue

        grid_h = int(rng.integers(max(18, span_h + 2), 31))
        grid_w = int(rng.integers(max(18, span_w + 2), 31))
        top_margin = int(rng.integers(1, grid_h - span_h))
        left_margin = int(rng.integers(1, grid_w - span_w))
        translated = [
            (
                box[0] - min_r + top_margin,
                box[1] - min_c + left_margin,
                box[2] - min_r + top_margin,
                box[3] - min_c + left_margin,
            )
            for box in raw_boxes
        ]
        # A positive top-to-bottom displacement means the chain top is above
        # (or left of) its bottom, hence alignment to the corresponding top/left.
        return translated, endpoint_row_sign > 0, endpoint_col_sign > 0
    return None


def _paint_input(shape: tuple[int, int], colors: list[int], boxes: list[BBox]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    # Bottom is painted first, so the earlier chain member is visible in each
    # adjacent overlap and supplies the directed occlusion clue.
    for color, (r0, c0, r1, c1) in reversed(list(zip(colors, boxes))):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid


def _expected_stack(
    dims_by_color: dict[int, tuple[int, int]],
    order: list[int],
    align_top: bool,
    align_left: bool,
) -> np.ndarray:
    out_h = max(height for height, _width in dims_by_color.values())
    out_w = max(width for _height, width in dims_by_color.values())
    out = np.zeros((out_h, out_w), dtype=int)
    for color in order:
        height, width = dims_by_color[color]
        r0 = 0 if align_top else out_h - height
        c0 = 0 if align_left else out_w - width
        out[r0 : r0 + height, c0 : c0 + width] = color
    return out


def _semantic_check(
    grid: np.ndarray,
    colors: list[int],
    boxes: list[BBox],
    dims: list[tuple[int, int]],
    align_top: bool,
    align_left: bool,
) -> bool:
    recovered = {color: _bbox(grid == color) for color in colors}
    if any(recovered[color] != box for color, box in zip(colors, boxes)):
        return False

    # With no non-neighbor intersections, every one of these direct clues is an
    # indispensable link in the only possible top-to-bottom chain.
    edges = _occlusion_edges(grid, recovered)
    required_edges = {(colors[i], colors[i + 1]) for i in range(len(colors) - 1)}
    if edges != required_edges:
        return False
    try:
        if _unique_chain(set(colors), edges) != colors:
            return False
    except AssertionError:
        return False
    for edge in required_edges:
        try:
            _unique_chain(set(colors), edges - {edge})
            return False
        except AssertionError:
            pass

    dims_by_color = dict(zip(colors, dims))
    expected = _expected_stack(dims_by_color, colors, align_top, align_left)
    if set(map(int, np.unique(expected))) != {0, *colors}:
        return False

    # Reject examples for which endpoint direction, reverse drawing, or the
    # tempting area sort happens to be observationally irrelevant.
    for other_top, other_left in itertools.product((False, True), repeat=2):
        if (other_top, other_left) != (align_top, align_left):
            alternative = _expected_stack(dims_by_color, colors, other_top, other_left)
            if np.array_equal(alternative, expected):
                return False
    if np.array_equal(
        _expected_stack(dims_by_color, list(reversed(colors)), align_top, align_left), expected
    ):
        return False
    area_order = sorted(colors, key=lambda color: dims_by_color[color][0] * dims_by_color[color][1], reverse=True)
    if area_order == colors or np.array_equal(
        _expected_stack(dims_by_color, area_order, align_top, align_left), expected
    ):
        return False

    if np.count_nonzero(grid == 0) < grid.size * 0.35:
        return False
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return False
    return np.array_equal(actual, expected)


def generate_input() -> np.ndarray:
    """Generate a varied, nondegenerate occlusion-chain instance."""
    rng = np.random.default_rng()
    for _ in range(_MAX_ATTEMPTS):
        count = int(rng.integers(4, 6))
        dims = _choose_dimensions(rng, count)
        if dims is None:
            continue
        placement = _place_chain(rng, dims)
        if placement is None:
            continue
        boxes, align_top, align_left = placement

        colors = [int(color) for color in rng.choice(np.arange(1, 10), size=count, replace=False)]
        grid_h = max(box[2] for box in boxes) + 1
        grid_w = max(box[3] for box in boxes) + 1
        # Preserve the sampled lower/right slack as well as the translated bboxes.
        grid_h = int(rng.integers(grid_h, 31)) if grid_h < 30 else grid_h
        grid_w = int(rng.integers(grid_w, 31)) if grid_w < 30 else grid_w
        grid = _paint_input((grid_h, grid_w), colors, boxes)

        if _semantic_check(grid, colors, boxes, dims, align_top, align_left):
            return grid
    raise RuntimeError("failed to generate a valid occlusion-chain restacking input")
