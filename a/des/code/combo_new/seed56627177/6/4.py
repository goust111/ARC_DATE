import random

import numpy as np


COLOR_VALUES = tuple(range(10))
GEN_RETRIES = 400
RECT_COUNT_RANGE = (2, 4)
RECT_SIZE_RANGE = (3, 7)
PADDING_BUDGET_RANGE = (2, 10)
DIRECTIONS = (
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 0:
        return []
    assert total >= parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _infer_one_rectangle(
    grid: np.ndarray, color: int, rect_h: int, rect_w: int, coords: np.ndarray
) -> tuple[int, int, int, int]:
    h, w = grid.shape
    vr0, vc0, vr1, vc1 = _bbox(coords)
    mask = grid == color
    candidates: list[tuple[int, int, int, int]] = []

    corner_specs = [
        ("tl", vr0, vc0),
        ("tr", vr0, vc1 - rect_w + 1),
        ("bl", vr1 - rect_h + 1, vc0),
        ("br", vr1 - rect_h + 1, vc1 - rect_w + 1),
    ]

    for corner, r0, c0 in corner_specs:
        r1 = r0 + rect_h - 1
        c1 = c0 + rect_w - 1
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        inside = (
            (coords[:, 0] >= r0)
            & (coords[:, 0] <= r1)
            & (coords[:, 1] >= c0)
            & (coords[:, 1] <= c1)
        )
        if not bool(np.all(inside)):
            continue

        if "t" in corner and not bool(np.all(mask[r0, c0 : c1 + 1])):
            continue
        if "b" in corner and not bool(np.all(mask[r1, c0 : c1 + 1])):
            continue
        if "l" in corner and not bool(np.all(mask[r0 : r1 + 1, c0])):
            continue
        if "r" in corner and not bool(np.all(mask[r0 : r1 + 1, c1])):
            continue
        candidates.append((r0, c0, r1, c1))

    unique = sorted(set(candidates))
    assert len(unique) == 1, f"failed to infer rectangle for color {color}: {unique}"
    return unique[0]


def _infer_rectangles(
    grid: np.ndarray, bg: int
) -> tuple[int, int, dict[int, tuple[int, int, int, int]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert colors, "expected at least one foreground rectangle"

    rect_h = 0
    rect_w = 0
    coords_by_color: dict[int, np.ndarray] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        coords_by_color[color] = coords
        r0, c0, r1, c1 = _bbox(coords)
        rect_h = max(rect_h, r1 - r0 + 1)
        rect_w = max(rect_w, c1 - c0 + 1)

    rectangles = {
        color: _infer_one_rectangle(grid, color, rect_h, rect_w, coords)
        for color, coords in coords_by_color.items()
    }
    return rect_h, rect_w, rectangles


def _add_edge(
    edges: dict[int, set[int]], indegree: dict[int, int], below: int, above: int
) -> None:
    if above not in edges[below]:
        edges[below].add(above)
        indegree[above] += 1


def _layer_order(
    grid: np.ndarray, rectangles: dict[int, tuple[int, int, int, int]]
) -> list[int]:
    colors = list(rectangles)
    edges = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for i, color_a in enumerate(colors):
        ar0, ac0, ar1, ac1 = rectangles[color_a]
        for color_b in colors[i + 1 :]:
            br0, bc0, br1, bc1 = rectangles[color_b]
            ir0 = max(ar0, br0)
            ic0 = max(ac0, bc0)
            ir1 = min(ar1, br1)
            ic1 = min(ac1, bc1)
            if ir0 > ir1 or ic0 > ic1:
                continue

            seen_a = False
            seen_b = False
            window = grid[ir0 : ir1 + 1, ic0 : ic1 + 1]
            if np.any(window == color_a):
                seen_a = True
            if np.any(window == color_b):
                seen_b = True

            assert not (seen_a and seen_b), "pairwise overlap shows conflicting layer evidence"
            if seen_a:
                _add_edge(edges, indegree, color_b, color_a)
            elif seen_b:
                _add_edge(edges, indegree, color_a, color_b)

    order: list[int] = []
    available = sorted(color for color in colors if indegree[color] == 0)
    while available:
        assert len(available) == 1, "layer order is not uniquely determined"
        color = available.pop()
        order.append(color)
        for nxt in sorted(edges[color]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                available.append(nxt)
                available.sort()

    assert len(order) == len(colors), "layer graph must be acyclic"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Recover overlapped same-size rectangles and lay them out from bottom layer to top layer.

    Input structure:
    - The background is the dominant color.
    - Each non-background color is the visible part of one solid axis-aligned rectangle.
    - All rectangles have the same height and width, but lower rectangles may be partly hidden by higher ones.
    - For every rectangle, at least one corner keeps two adjacent full edges visible, so its original bbox is recoverable.

    Rule:
    - Infer the common rectangle size from the largest visible height and width.
    - For each color, test the four corner-anchored bbox candidates and keep the unique one whose corresponding two adjacent edges are fully visible.
    - Whenever two recovered rectangles overlap, the visible color inside that overlap is above the other one; collect these above/below relations and derive the unique bottom-to-top order.
    - Return a grid of rectangle height by (rectangle count × rectangle width), filled with solid color blocks arranged left to right from bottom layer to top layer.

    Key insight:
    - The visible L-shaped corner of each cropped region reveals its full rectangle, and overlap visibility turns the 2D composite back into a 1D layer ordering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    rect_h, rect_w, rectangles = _infer_rectangles(g, bg)
    order = _layer_order(g, rectangles)

    out = np.empty((rect_h, rect_w * len(order)), dtype=int)
    for idx, color in enumerate(order):
        out[:, idx * rect_w : (idx + 1) * rect_w] = color
    return out


def _expected_output(colors: list[int], rect_h: int, rect_w: int) -> np.ndarray:
    out = np.empty((rect_h, rect_w * len(colors)), dtype=int)
    for idx, color in enumerate(colors):
        out[:, idx * rect_w : (idx + 1) * rect_w] = color
    return out


def _render_mask(positions: list[tuple[int, int]], rect_h: int, rect_w: int) -> np.ndarray:
    max_r = max(r for r, _ in positions) + rect_h
    max_c = max(c for _, c in positions) + rect_w
    mask = np.zeros((max_r, max_c), dtype=bool)
    for r, c in positions:
        mask[r : r + rect_h, c : c + rect_w] = True
    return mask


def _sample_scene() -> tuple[np.ndarray, np.ndarray] | None:
    n_rects = random.randint(*RECT_COUNT_RANGE)
    min_size = max(RECT_SIZE_RANGE[0], n_rects)
    max_size = RECT_SIZE_RANGE[1]
    if min_size > max_size:
        return None

    rect_h = random.randint(min_size, max_size)
    rect_w = random.randint(min_size, max_size)
    if n_rects * rect_w > 30:
        return None

    bg = random.choice(COLOR_VALUES)
    colors = random.sample([c for c in COLOR_VALUES if c != bg], n_rects)
    sign_r, sign_c = random.choice(DIRECTIONS)

    step_count = n_rects - 1
    total_dr = random.randint(step_count, rect_h - 1)
    total_dc = random.randint(step_count, rect_w - 1)
    drs = _random_composition(total_dr, step_count)
    dcs = _random_composition(total_dc, step_count)

    positions = [(0, 0)]
    for dr, dc in zip(drs, dcs, strict=False):
        prev_r, prev_c = positions[-1]
        positions.append((prev_r + sign_r * dr, prev_c + sign_c * dc))

    min_r = min(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    positions = [(r - min_r, c - min_c) for r, c in positions]

    bbox_h = max(r for r, _ in positions) + rect_h
    bbox_w = max(c for _, c in positions) + rect_w
    if bbox_h > 30 or bbox_w > 30:
        return None

    union_mask = _render_mask(positions, rect_h, rect_w)
    union_area = int(np.sum(union_mask))

    pad_h_max = min(PADDING_BUDGET_RANGE[1], 30 - bbox_h)
    pad_w_max = min(PADDING_BUDGET_RANGE[1], 30 - bbox_w)
    if pad_h_max < PADDING_BUDGET_RANGE[0] or pad_w_max < PADDING_BUDGET_RANGE[0]:
        return None

    total_pad_h = random.randint(PADDING_BUDGET_RANGE[0], pad_h_max)
    total_pad_w = random.randint(PADDING_BUDGET_RANGE[0], pad_w_max)
    grid_h = bbox_h + total_pad_h
    grid_w = bbox_w + total_pad_w
    bg_count = grid_h * grid_w - union_area
    if bg_count <= rect_h * rect_w:
        return None

    top_pad = random.randint(0, total_pad_h)
    left_pad = random.randint(0, total_pad_w)
    positions = [(r + top_pad, c + left_pad) for r, c in positions]

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for color, (r, c) in zip(colors, positions, strict=False):
        grid[r : r + rect_h, c : c + rect_w] = color

    expected = _expected_output(colors, rect_h, rect_w)
    if not np.array_equal(main(grid), expected):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        scene = _sample_scene()
        if scene is not None:
            grid, _expected = scene
            return grid
    raise RuntimeError("failed to generate a valid layered-rectangle scene")
