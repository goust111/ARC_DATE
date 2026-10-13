import numpy as np


BACKGROUND = 0
AVAILABLE_COLORS = list(range(1, 10))
GEN_ATTEMPTS = 600
SIZE_ATTEMPTS = 80
HEURISTIC_FEATURES = (
    "left",
    "right",
    "top",
    "bottom",
    "center_x",
    "center_y",
    "area",
    "height",
    "width",
    "visible_area",
)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected non-empty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_h(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[2] - bbox[0] + 1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _intersect_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    r0 = max(int(a[0]), int(b[0]))
    c0 = max(int(a[1]), int(b[1]))
    r1 = min(int(a[2]), int(b[2]))
    c1 = min(int(a[3]), int(b[3]))
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _rectangles_from_visible(
    grid: np.ndarray,
) -> tuple[int, dict[int, tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _background_color(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert colors, "expected at least one foreground rectangle"

    rects: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        rects[color] = _bbox_from_mask(g == color)
    return bg, rects


def _occlusion_edges(
    grid: np.ndarray, rects: dict[int, tuple[int, int, int, int]]
) -> set[tuple[int, int]]:
    g = np.asarray(grid, dtype=int)
    colors = list(rects)
    edges: set[tuple[int, int]] = set()
    for i, color_a in enumerate(colors):
        for color_b in colors[i + 1 :]:
            inter = _intersect_bbox(rects[color_a], rects[color_b])
            if inter is None:
                continue
            r0, c0, r1, c1 = inter
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            sees_a = bool(np.any(patch == color_a))
            sees_b = bool(np.any(patch == color_b))
            assert not (sees_a and sees_b), "pairwise overlap cannot show both orders"
            if sees_a:
                edges.add((int(color_b), int(color_a)))
            elif sees_b:
                edges.add((int(color_a), int(color_b)))
    return edges


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    indegree = {int(color): 0 for color in colors}
    outgoing = {int(color): set() for color in colors}
    for lower, upper in edges:
        if upper not in outgoing[lower]:
            outgoing[lower].add(upper)
            indegree[upper] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        zeros = sorted(color for color in remaining if indegree[color] == 0)
        assert len(zeros) == 1, "expected a unique occlusion order"
        current = zeros[0]
        order.append(int(current))
        remaining.remove(current)
        for nxt in outgoing[current]:
            indegree[nxt] -= 1
    return order


def _drop_rectangles(
    shape: tuple[int, int],
    rects: dict[int, tuple[int, int, int, int]],
    order: list[int],
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    H, W = map(int, shape)
    top_occupied = np.full(W, H, dtype=int)
    out = np.full((H, W), BACKGROUND, dtype=int)
    landed: dict[int, tuple[int, int, int, int]] = {}

    for color in order:
        r0, c0, r1, c1 = rects[color]
        h = _bbox_h((r0, c0, r1, c1))
        landing_top = int(np.min(top_occupied[c0 : c1 + 1]) - h)
        assert landing_top >= 0, "stack exceeds the canvas height"
        assert np.all(out[landing_top : landing_top + h, c0 : c1 + 1] == BACKGROUND)
        out[landing_top : landing_top + h, c0 : c1 + 1] = int(color)
        top_occupied[c0 : c1 + 1] = np.minimum(top_occupied[c0 : c1 + 1], landing_top)
        landed[color] = (landing_top, c0, landing_top + h - 1, c1)

    return out, landed


def _render_occluded_input(
    shape: tuple[int, int],
    rects: dict[int, tuple[int, int, int, int]],
    order: list[int],
) -> np.ndarray:
    H, W = map(int, shape)
    grid = np.full((H, W), BACKGROUND, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rects[color]
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return grid


def _visible_edge_flags(
    grid: np.ndarray, color: int, bbox: tuple[int, int, int, int]
) -> tuple[bool, bool, bool, bool]:
    r0, c0, r1, c1 = bbox
    mask = grid[r0 : r1 + 1, c0 : c1 + 1] == int(color)
    return (
        bool(mask[0, :].any()),
        bool(mask[-1, :].any()),
        bool(mask[:, 0].any()),
        bool(mask[:, -1].any()),
    )


def _visible_area(grid: np.ndarray, color: int) -> int:
    return int(np.count_nonzero(grid == int(color)))


def _feature_values(
    rects: dict[int, tuple[int, int, int, int]], grid: np.ndarray
) -> dict[str, dict[int, int]]:
    features: dict[str, dict[int, int]] = {name: {} for name in HEURISTIC_FEATURES}
    for color, bbox in rects.items():
        r0, c0, r1, c1 = bbox
        features["left"][color] = int(c0)
        features["right"][color] = int(c1)
        features["top"][color] = int(r0)
        features["bottom"][color] = int(r1)
        features["center_x"][color] = int(c0 + c1)
        features["center_y"][color] = int(r0 + r1)
        features["height"][color] = int(_bbox_h(bbox))
        features["width"][color] = int(_bbox_w(bbox))
        features["area"][color] = int(_bbox_h(bbox) * _bbox_w(bbox))
        features["visible_area"][color] = int(_visible_area(grid, color))
    return features


def _is_simple_sort(order: list[int], rects: dict[int, tuple[int, int, int, int]], grid: np.ndarray) -> bool:
    features = _feature_values(rects, grid)
    for name in HEURISTIC_FEATURES:
        values = features[name]
        if len(set(values.values())) != len(values):
            continue
        ascending = [color for color, _value in sorted(values.items(), key=lambda item: item[1])]
        descending = list(reversed(ascending))
        if order == ascending or order == descending:
            return True
    return False


def _swap_changes_everything(
    shape: tuple[int, int],
    rects: dict[int, tuple[int, int, int, int]],
    order: list[int],
    baseline: np.ndarray,
) -> bool:
    for i in range(len(order) - 1):
        alt = list(order)
        alt[i], alt[i + 1] = alt[i + 1], alt[i]
        try:
            alt_out, _ = _drop_rectangles(shape, rects, alt)
        except AssertionError:
            return True
        if np.array_equal(alt_out, baseline):
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the hidden rectangle order from occlusion and let the rectangles fall in that order onto an empty canvas.

    Input structure:
    - A black background contains several solid monochrome axis-aligned rectangles.
    - Rectangles partially overlap, and each overlap shows only the color of the rectangle that sits above the others there.
    - Every rectangle still has visible pixels on all four sides, so the full rectangle can be recovered from the bounding box of its visible color.

    Rule:
    - For each non-background color, reconstruct its full solid rectangle from the bounding box of its visible pixels.
    - For each pair of reconstructed rectangles, inspect their overlap area: if color B is visible inside the pairwise overlap, then B is above A.
    - Use the uniquely determined bottom-to-top occlusion order as the drop order, and let each rectangle fall straight down on an empty canvas of the same size until it hits the floor or a previously dropped rectangle.

    Key insight:
    - The visible occlusion is the timestamp: the hidden rectangle is earlier and therefore drops first, so the correct output requires recovering order before applying gravity.
    """
    g = np.asarray(grid, dtype=int)
    bg, rects = _rectangles_from_visible(g)
    assert bg == BACKGROUND, "expected black background"

    colors = sorted(rects)
    edges = _occlusion_edges(g, rects)
    order = _unique_topological_order(colors, edges)
    out, _landed = _drop_rectangles(g.shape, rects, order)
    return out


def _random_sizes(rng: np.random.Generator, n: int) -> tuple[list[int], list[int]]:
    if n == 3:
        heights = [int(rng.integers(3, 8)) for _ in range(n)]
        widths = [int(rng.integers(5, 10)) for _ in range(n)]
    elif n == 4:
        heights = [int(rng.integers(3, 7)) for _ in range(n)]
        widths = [int(rng.integers(5, 9)) for _ in range(n)]
    else:
        heights = [int(rng.integers(3, 6)) for _ in range(n)]
        widths = [int(rng.integers(4, 8)) for _ in range(n)]
    return heights, widths


def _random_walk_layout(
    rng: np.random.Generator, heights: list[int], widths: list[int]
) -> list[tuple[int, int, int, int]]:
    n = len(heights)
    r0s = [0]
    c0s = [0]
    prev_dx = None
    prev_dy = None
    for i in range(1, n):
        max_ox = min(3, widths[i - 1] - 1, widths[i] - 1)
        max_oy = min(3, heights[i - 1] - 1, heights[i] - 1)
        ox = int(rng.integers(1, max_ox + 1))
        oy = int(rng.integers(1, max_oy + 1))

        sx = 1 if rng.random() < 0.5 else -1
        sy = 1 if rng.random() < 0.5 else -1
        if prev_dx is not None and prev_dy is not None and rng.random() < 0.65:
            if rng.random() < 0.5:
                sx = -prev_dx
            else:
                sy = -prev_dy

        dx = widths[i - 1] - ox if sx > 0 else -(widths[i] - ox)
        dy = heights[i - 1] - oy if sy > 0 else -(heights[i] - oy)
        c0s.append(c0s[-1] + int(dx))
        r0s.append(r0s[-1] + int(dy))
        prev_dx, prev_dy = sx, sy

    min_r = min(r0s)
    min_c = min(c0s)
    max_r = max(r0 + h for r0, h in zip(r0s, heights))
    max_c = max(c0 + w for c0, w in zip(c0s, widths))
    span_h = int(max_r - min_r)
    span_w = int(max_c - min_c)

    sum_heights = int(sum(heights))
    if span_h > 30 or span_w > 30 or sum_heights > 30:
        raise AssertionError("layout too large")

    H_min = max(span_h, sum_heights)
    W_min = span_w
    H = int(rng.integers(H_min, min(30, H_min + 5) + 1))
    W = int(rng.integers(W_min, min(30, W_min + 5) + 1))
    row_shift = int(rng.integers(0, H - span_h + 1)) - min_r
    col_shift = int(rng.integers(0, W - span_w + 1)) - min_c

    rects = []
    for r0, c0, h, w in zip(r0s, c0s, heights, widths):
        rr0 = int(r0 + row_shift)
        cc0 = int(c0 + col_shift)
        rects.append((rr0, cc0, rr0 + h - 1, cc0 + w - 1))
    return rects


def _has_unique_bbox_recovery(
    grid: np.ndarray, rects: dict[int, tuple[int, int, int, int]]
) -> bool:
    for color, bbox in rects.items():
        mask = grid == int(color)
        if not mask.any():
            return False
        if _bbox_from_mask(mask) != bbox:
            return False
        if not all(_visible_edge_flags(grid, color, bbox)):
            return False
    return True


def _movement_is_nontrivial(
    landed: dict[int, tuple[int, int, int, int]], rects: dict[int, tuple[int, int, int, int]]
) -> bool:
    return any(landed[color][0] != rects[color][0] for color in rects)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(3, 6))
    colors = list(AVAILABLE_COLORS)
    rng.shuffle(colors)
    order = [int(color) for color in colors[:n]]

    for _ in range(SIZE_ATTEMPTS):
        heights, widths = _random_sizes(rng, n)
        try:
            bboxes = _random_walk_layout(rng, heights, widths)
        except AssertionError:
            continue

        rects = {color: bbox for color, bbox in zip(order, bboxes)}
        H = max(bbox[2] for bbox in bboxes) + 1
        W = max(bbox[3] for bbox in bboxes) + 1
        shape = (int(H), int(W))

        input_grid = _render_occluded_input(shape, rects, order)
        if not _has_unique_bbox_recovery(input_grid, rects):
            continue

        edges = _occlusion_edges(input_grid, rects)
        if any((order[i], order[i + 1]) not in edges for i in range(n - 1)):
            continue

        try:
            recovered = _unique_topological_order(sorted(rects), edges)
        except AssertionError:
            continue
        if recovered != order:
            continue

        if _is_simple_sort(order, rects, input_grid):
            continue

        try:
            expected, landed = _drop_rectangles(shape, rects, order)
        except AssertionError:
            continue

        if not _movement_is_nontrivial(landed, rects):
            continue
        if not _swap_changes_everything(shape, rects, order, expected):
            continue
        if np.array_equal(input_grid, expected):
            continue

        try:
            actual = main(input_grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue

        return input_grid

    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
