import numpy as np


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == 0):
        return 0
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _frame_points(bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    points: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        points.append((r0, c))
        if r1 != r0:
            points.append((r1, c))
    for r in range(r0 + 1, r1):
        points.append((r, c0))
        if c1 != c0:
            points.append((r, c1))
    return points


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    for r, c in _frame_points(bbox):
        grid[r, c] = color


def _visible_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == color)
    assert coords.size > 0, "each frame color must remain visible"
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    r0, c0 = map(int, mins)
    r1, c1 = map(int, maxs)
    border_mask = (
        (coords[:, 0] == r0)
        | (coords[:, 0] == r1)
        | (coords[:, 1] == c0)
        | (coords[:, 1] == c1)
    )
    assert np.all(border_mask), "visible pixels must lie on a single rectangle frame"
    return r0, c0, r1, c1


def _topological_order(
    bboxes: dict[int, tuple[int, int, int, int]],
    edges: dict[int, set[int]],
) -> list[int]:
    indegree = {color: 0 for color in bboxes}
    adjacency = {color: set() for color in bboxes}
    for lower, uppers in edges.items():
        for upper in uppers:
            if upper not in bboxes or upper == lower or upper in adjacency[lower]:
                continue
            adjacency[lower].add(upper)
            indegree[upper] += 1

    order: list[int] = []
    ready = sorted((color for color, deg in indegree.items() if deg == 0), key=lambda c: bboxes[c])
    while ready:
        color = ready.pop(0)
        order.append(color)
        for upper in sorted(adjacency[color], key=lambda c: bboxes[c]):
            indegree[upper] -= 1
            if indegree[upper] == 0:
                ready.append(upper)
                ready.sort(key=lambda c: bboxes[c])

    assert len(order) == len(bboxes), "occlusion relations must form an acyclic order"
    return order


def _render_nested(order: list[int], bg: int) -> np.ndarray:
    size = 2 * len(order) - 1
    out = np.full((size, size), bg, dtype=int)
    for depth, color in enumerate(order):
        _draw_frame(out, (depth, depth, size - 1 - depth, size - 1 - depth), int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder overlapping rectangle frames into a concentric nesting by occlusion depth.

    Input structure:
    - The background is black.
    - Each non-background color belongs to a distinct one-cell-thick axis-aligned rectangular frame.
    - Frames may overlap, and every overlap cell shows only the color of the upper frame.

    Rule:
    - Recover each full frame by taking the bounding box of that color's visible pixels.
    - On a recovered frame border, any visible different color must come from a frame above it; use these relations to build a bottom-to-top layering order.
    - Redraw the colors as concentric one-cell-thick rectangular frames on a `(2K-1) x (2K-1)` canvas, from the lowest layer as the outer border to the highest layer as the inner border.

    Key insight:
    - The input geometry only serves to reveal which frame covers which at shared border cells; the output is a canonical rerendering of that recovered occlusion order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)

    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one frame color"

    bboxes = {color: _visible_bbox(g, color) for color in colors}
    edges = {color: set() for color in colors}
    for color, bbox in bboxes.items():
        for r, c in _frame_points(bbox):
            seen = int(g[r, c])
            if seen != bg and seen != color:
                edges[color].add(seen)

    order = _topological_order(bboxes, edges)
    return _render_nested(order, bg)


_BG = 0
_PALETTE = tuple(range(1, 10))
_FRAME_COUNT_RANGE = (2, 7)
_GRID_MAX = 30
_MIN_INNER_SPAN = 5
_MAX_GENERATION_ATTEMPTS = 300


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _sample_nested_edges(
    rng: np.random.Generator,
    length: int,
    count: int,
) -> tuple[list[int], list[int]]:
    center = _randint(rng, count + 2, length - count - 3)
    upper_pool = np.arange(1, center - 1)
    lower_pool = np.arange(center + 2, length - 1)
    tops = np.sort(rng.choice(upper_pool, size=count, replace=False)).astype(int).tolist()
    bottoms = np.sort(rng.choice(lower_pool, size=count, replace=False))[::-1].astype(int).tolist()
    assert bottoms[-1] - tops[-1] + 1 >= _MIN_INNER_SPAN
    return tops, bottoms


def _symmetry(grid: np.ndarray, code: int) -> np.ndarray:
    out = grid
    if code & 1:
        out = np.flipud(out)
    if code & 2:
        out = np.fliplr(out)
    if code & 4:
        out = out.T
    return np.array(out, dtype=int, copy=True)


def _cross_count(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    return len(set(_frame_points(a)) & set(_frame_points(b)))


def _valid_overlap_pattern(bboxes: list[tuple[int, int, int, int]]) -> bool:
    point_hits: dict[tuple[int, int], int] = {}
    for bbox in bboxes:
        for point in _frame_points(bbox):
            point_hits[point] = point_hits.get(point, 0) + 1
    if max(point_hits.values(), default=0) > 2:
        return False
    for i in range(len(bboxes)):
        for j in range(i + 1, len(bboxes)):
            if _cross_count(bboxes[i], bboxes[j]) != 4:
                return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    max_frames = min(_FRAME_COUNT_RANGE[1], len(_PALETTE))
    count = _randint(rng, _FRAME_COUNT_RANGE[0], max_frames)
    min_side = max(2 * count + _MIN_INNER_SPAN, 11)
    height = _randint(rng, min_side, _GRID_MAX)
    width = _randint(rng, min_side, _GRID_MAX)

    tops, bottoms = _sample_nested_edges(rng, height, count)
    lefts, rights = _sample_nested_edges(rng, width, count)

    bboxes = [
        (tops[i], lefts[count - 1 - i], bottoms[i], rights[count - 1 - i])
        for i in range(count)
    ]
    if not _valid_overlap_pattern(bboxes):
        return None

    colors = [int(c) for c in rng.choice(_PALETTE, size=count, replace=False)]
    paint_order = [int(i) for i in rng.permutation(count)]
    expected_order = [colors[i] for i in paint_order]

    grid = np.full((height, width), _BG, dtype=int)
    for idx in paint_order:
        _draw_frame(grid, bboxes[idx], colors[idx])

    grid = _symmetry(grid, _randint(rng, 0, 7))
    expected = _render_nested(expected_order, _BG)
    actual = main(grid)
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    if len(np.unique(grid)) != count + 1:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
