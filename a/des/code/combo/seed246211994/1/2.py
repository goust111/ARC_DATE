from __future__ import annotations

import numpy as np


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _contains(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _infer_rectangles(
    grid: np.ndarray, bg: int
) -> tuple[list[int], dict[int, tuple[int, int, int, int]]]:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert colors, "expected at least one foreground rectangle"

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color in colors:
        coords = np.argwhere(grid == color)
        assert coords.size > 0, "every foreground color must be visible"
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert not np.any(sub == bg), "an inferred solid rectangle cannot contain exposed background"
        bboxes[color] = bbox
    return colors, bboxes


def _occlusion_edges(
    grid: np.ndarray,
    colors: list[int],
    bboxes: dict[int, tuple[int, int, int, int]],
    bg: int,
) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for r_raw, c_raw in np.argwhere(grid != bg):
        r = int(r_raw)
        c = int(c_raw)
        visible = int(grid[r, c])
        covering = [color for color in colors if _contains(bboxes[color], r, c)]
        assert visible in covering, "visible foreground must belong to its inferred rectangle"
        for hidden in covering:
            if hidden != visible:
                edges.add((hidden, visible))
    return edges


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    outgoing = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for lower, upper in edges:
        if upper not in outgoing[lower]:
            outgoing[lower].add(upper)
            indegree[upper] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        roots = [color for color in colors if color in remaining and indegree[color] == 0]
        assert len(roots) == 1, "occlusion evidence must determine a unique total order"
        color = roots[0]
        remaining.remove(color)
        order.append(color)
        for upper in outgoing[color]:
            indegree[upper] -= 1
    return order


def _render_nested(order: list[int]) -> np.ndarray:
    size = 2 * len(order) - 1
    out = np.empty((size, size), dtype=int)
    for layer, color in enumerate(order):
        end = size - layer - 1
        out[layer : end + 1, layer] = color
        out[layer : end + 1, end] = color
        out[layer, layer : end + 1] = color
        out[end, layer : end + 1] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered solid rectangles by occlusion order, then redraw the order as nested square rings.

    Input structure
    - The background is the dominant color.
    - Each foreground color belongs to one solid axis-aligned rectangle.
    - Rectangles may overlap; only the topmost color remains visible at each covered cell.

    Rule
    - Infer each rectangle from the bounding box of its visible pixels.
    - At every foreground cell, every other rectangle covering that cell lies below the visible color.
    - Recover the unique bottom-to-top order from those directed occlusion constraints.
    - Output a square of side 2*n-1 whose nested rings encode that order from outside to inside.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    colors, bboxes = _infer_rectangles(g, bg)
    edges = _occlusion_edges(g, colors, bboxes, bg)
    if len(colors) > 1:
        assert edges, "multiple rectangles must produce occlusion evidence"
    order = _unique_topological_order(colors, edges)
    return _render_nested(order)


# --------------- generator ---------------

PALETTE = tuple(range(10))
MIN_RECT_COUNT = 3
MAX_RECT_COUNT = 5
MIN_SIDE = 5
MAX_SIDE = 10
MAX_SHIFT = 3
GRID_MIN = 14
GRID_MAX = 32
MARGIN_MIN = 1
MARGIN_MAX = 6
GENERATION_ATTEMPTS = 400


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _sample_colors(rng: np.random.Generator) -> tuple[int, list[int]]:
    bg = _randint(rng, 0, 9)
    palette = np.array([color for color in PALETTE if color != bg], dtype=int)
    rng.shuffle(palette)
    count = _randint(rng, MIN_RECT_COUNT, MAX_RECT_COUNT)
    return bg, [int(color) for color in palette[:count]]


def _next_side(rng: np.random.Generator, prev: int) -> int:
    if rng.random() < 0.7:
        lo = max(MIN_SIDE, prev - 2)
        hi = min(MAX_SIDE, prev + 2)
    else:
        lo = MIN_SIDE
        hi = MAX_SIDE
    return _randint(rng, lo, hi)


def _build_local_chain(
    rng: np.random.Generator, count: int
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    r = 0
    c = 0
    h = _randint(rng, MIN_SIDE, MAX_SIDE)
    w = _randint(rng, MIN_SIDE, MAX_SIDE)
    rects.append((r, c, h, w))

    for _ in range(1, count):
        prev_r, prev_c, prev_h, prev_w = rects[-1]
        max_dr = min(MAX_SHIFT, prev_h - 3)
        max_dc = min(MAX_SHIFT, prev_w - 3)
        if max_dr < 1 or max_dc < 1:
            return None
        dr = _randint(rng, 1, max_dr)
        dc = _randint(rng, 1, max_dc)
        r = prev_r + dr
        c = prev_c + dc
        h = _next_side(rng, prev_h)
        w = _next_side(rng, prev_w)
        rects.append((r, c, h, w))
    return rects


def _transform_bbox(
    bbox: tuple[int, int, int, int],
    height: int,
    width: int,
    flip_v: bool,
    flip_h: bool,
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    if flip_v:
        r0, r1 = height - 1 - r1, height - 1 - r0
    if flip_h:
        c0, c1 = width - 1 - c1, width - 1 - c0
    return int(r0), int(c0), int(r1), int(c1)


def _embed_bboxes(
    rng: np.random.Generator, rects: list[tuple[int, int, int, int]]
) -> tuple[int, int, list[tuple[int, int, int, int]]] | None:
    local_h = max(r + h for r, _, h, _ in rects)
    local_w = max(c + w for _, c, _, w in rects)

    top = _randint(rng, MARGIN_MIN, MARGIN_MAX)
    left = _randint(rng, MARGIN_MIN, MARGIN_MAX)
    bottom = _randint(rng, MARGIN_MIN, MARGIN_MAX)
    right = _randint(rng, MARGIN_MIN, MARGIN_MAX)
    height = top + local_h + bottom
    width = left + local_w + right
    if not (GRID_MIN <= height <= GRID_MAX and GRID_MIN <= width <= GRID_MAX):
        return None

    flip_v = bool(_randint(rng, 0, 1))
    flip_h = bool(_randint(rng, 0, 1))

    bboxes = []
    for r, c, h, w in rects:
        bbox = (top + r, left + c, top + r + h - 1, left + c + w - 1)
        bboxes.append(_transform_bbox(bbox, height, width, flip_v, flip_h))
    return height, width, bboxes


def _paint_scene(
    bg: int,
    ordering: list[int],
    height: int,
    width: int,
    bboxes: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, dict[int, tuple[int, int, int, int]]]:
    grid = np.full((height, width), bg, dtype=int)
    true_bboxes = {int(color): bbox for color, bbox in zip(ordering, bboxes)}
    for color in ordering:
        r0, c0, r1, c1 = true_bboxes[color]
        grid[r0 : r1 + 1, c0 : c1 + 1] = color
    return grid, true_bboxes


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    count_map = {int(value): int(count) for value, count in zip(values, counts)}
    bg_count = count_map.get(int(bg), 0)
    return all(bg_count > count_map[color] for color in count_map if color != int(bg))


def _visible_bboxes_match(
    grid: np.ndarray, true_bboxes: dict[int, tuple[int, int, int, int]]
) -> bool:
    for color, bbox in true_bboxes.items():
        coords = np.argwhere(grid == int(color))
        if coords.size == 0:
            return False
        if _bbox_from_coords(coords) != bbox:
            return False
    return True


def _passes_oracle(grid: np.ndarray, ordering: list[int]) -> bool:
    expected = _render_nested(ordering)
    try:
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    bg, ordering = _sample_colors(rng)

    # Consecutive rectangles drift diagonally so each new layer leaves a
    # visible witness cell inside the previous layer while keeping all bbox
    # extremes visible.
    rects = _build_local_chain(rng, len(ordering))
    if rects is None:
        return None

    embedded = _embed_bboxes(rng, rects)
    if embedded is None:
        return None
    height, width, bboxes = embedded

    grid, true_bboxes = _paint_scene(bg, ordering, height, width, bboxes)
    if not _background_is_dominant(grid, bg):
        return None
    if not _visible_bboxes_match(grid, true_bboxes):
        return None
    if not _passes_oracle(grid, ordering):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid occluded-rectangle scene")
