from __future__ import annotations

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> BBox:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _perimeter_cells(bbox: BBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
    for c in range(c0, c1 + 1):
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
    for r in range(r0 + 1, r1):
        cells.append((r, c1))
    return cells


def _draw_perimeter(grid: np.ndarray, bbox: BBox, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _infer_rectangles(grid: np.ndarray, bg: int) -> list[dict]:
    items: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue

        coords = np.argwhere(grid == color)
        assert coords.size, "each non-background color must remain visible"
        bbox = _bbox(coords)
        r0, c0, r1, c1 = bbox

        on_border = (
            (coords[:, 0] == r0)
            | (coords[:, 0] == r1)
            | (coords[:, 1] == c0)
            | (coords[:, 1] == c1)
        )
        assert bool(np.all(on_border)), "visible pixels must belong to a rectangular border"

        items.append(
            {
                "color": color,
                "bbox": bbox,
                "visible_set": {(int(r), int(c)) for r, c in coords},
                "perimeter_set": set(_perimeter_cells(bbox)),
            }
        )

    assert 3 <= len(items) <= 5, "expected 3-5 rectangular frames"
    return items


def _layer_order_bottom_to_top(items: list[dict]) -> list[int]:
    n = len(items)
    lower_to_higher = [set() for _ in range(n)]

    for hi, item_hi in enumerate(items):
        for cell in item_hi["visible_set"]:
            for lo, item_lo in enumerate(items):
                if hi == lo:
                    continue
                if cell in item_lo["perimeter_set"]:
                    lower_to_higher[lo].add(hi)

    indegree = [0] * n
    for lo in range(n):
        for hi in lower_to_higher[lo]:
            indegree[hi] += 1

    key = lambda idx: tuple(int(v) for v in items[idx]["bbox"])
    ready = sorted([idx for idx in range(n) if indegree[idx] == 0], key=key)
    order: list[int] = []

    while ready:
        idx = ready.pop(0)
        order.append(idx)
        for nxt in sorted(lower_to_higher[idx], key=key):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort(key=key)

    if len(order) < n:
        remaining = sorted([idx for idx in range(n) if idx not in order], key=key)
        order.extend(remaining)

    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each partially hidden rectangular frame and redraw the complete stack in layer order.

    Input structure:
    - The background is black.
    - Each foreground color forms the still-visible fragments of one axis-aligned rectangular border that is one cell thick.
    - Several such frames overlap, so higher frames overwrite lower frames on shared border cells and can hide long segments.

    Rule:
    - For each visible color, infer its full rectangle from the extreme rows and columns of its surviving border fragments.
    - Build the occlusion order: if color A is visible on a cell that lies on color B's inferred border, then B is below A.
    - Clear the grid, then redraw every full rectangular border from the bottom layer to the top layer so higher borders overwrite lower ones on overlaps.

    Key insight:
    - Missing border segments are not damage; they are occlusion evidence. The visible fragments reveal each rectangle's full bbox, and visible overwrite cells reveal the stacking order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    items = _infer_rectangles(g, bg)
    order = _layer_order_bottom_to_top(items)

    out = np.full_like(g, bg)
    for idx in order:
        item = items[idx]
        _draw_perimeter(out, item["bbox"], int(item["color"]))
    return out


# --------------- generator ---------------

_BG = 0
_COLORS = tuple(range(1, 10))

_GRID_H_RANGE = (18, 29)
_GRID_W_RANGE = (18, 29)
_RECT_COUNT_RANGE = (3, 6)
_MIN_SIDE = 5
_MAX_SIDE = 16
_POOL_SIZE_RANGE = (5, 8)
_PAIR_PICK_RETRIES = 500
_POOL_RETRIES = 60
_GEN_RETRIES = 600


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _perimeters_intersect(a: BBox, b: BBox) -> bool:
    return bool(set(_perimeter_cells(a)) & set(_perimeter_cells(b)))


def _has_long_shared_segment(a: BBox, b: BBox) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    for ra in (ar0, ar1):
        for rb in (br0, br1):
            if ra == rb and _overlap_len(ac0, ac1, bc0, bc1) >= 3:
                return True
    for ca in (ac0, ac1):
        for cb in (bc0, bc1):
            if ca == cb and _overlap_len(ar0, ar1, br0, br1) >= 3:
                return True
    return False


def _sample_axis_pairs(
    rng: np.random.Generator, limit: int
) -> list[tuple[int, int]] | None:
    inner = np.arange(1, limit - 1)
    if inner.size < 6:
        return None

    for _ in range(_POOL_RETRIES):
        pool_size = min(int(rng.integers(*_POOL_SIZE_RANGE)), inner.size)
        values = sorted(int(v) for v in rng.choice(inner, size=pool_size, replace=False))
        pairs = [
            (values[i], values[j])
            for i in range(len(values))
            for j in range(i + 1, len(values))
            if _MIN_SIDE <= values[j] - values[i] + 1 <= _MAX_SIDE
        ]
        if pairs:
            return pairs
    return None


def _sample_rectangles(
    rng: np.random.Generator, h: int, w: int, n_rects: int
) -> list[BBox] | None:
    row_pairs = _sample_axis_pairs(rng, h)
    col_pairs = _sample_axis_pairs(rng, w)
    if row_pairs is None or col_pairs is None:
        return None

    rectangles: list[BBox] = []
    used: set[BBox] = set()
    for idx in range(n_rects):
        picked: BBox | None = None
        for _ in range(_PAIR_PICK_RETRIES):
            r0, r1 = row_pairs[int(rng.integers(len(row_pairs)))]
            c0, c1 = col_pairs[int(rng.integers(len(col_pairs)))]
            bbox = (int(r0), int(c0), int(r1), int(c1))
            if bbox in used:
                continue
            if idx > 0 and not any(_perimeters_intersect(bbox, other) for other in rectangles):
                continue
            picked = bbox
            break
        if picked is None:
            return None
        used.add(picked)
        rectangles.append(picked)
    return rectangles


def _render_scene(shape: tuple[int, int], rectangles: list[BBox], colors: list[int], order: list[int]) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    for idx in order:
        _draw_perimeter(grid, rectangles[idx], colors[idx])
    return grid


def _side_visibility(rect: BBox, visible: set[tuple[int, int]]) -> tuple[bool, bool, bool, bool]:
    r0, c0, r1, c1 = rect
    return (
        any(r == r0 for r, _c in visible),
        any(r == r1 for r, _c in visible),
        any(c == c0 for _r, c in visible),
        any(c == c1 for _r, c in visible),
    )


def _full_perimeter_size(rect: BBox) -> int:
    r0, c0, r1, c1 = rect
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    return 2 * (h + w) - 4


def _generated_scene_is_valid(
    scene: np.ndarray,
    rectangles: list[BBox],
    colors: list[int],
    force_segment_overlap: bool,
) -> bool:
    hidden_count = 0
    for rect, color in zip(rectangles, colors):
        coords = np.argwhere(scene == color)
        if coords.size == 0:
            return False
        if _bbox(coords) != rect:
            return False

        visible = {(int(r), int(c)) for r, c in coords}
        if not all(_side_visibility(rect, visible)):
            return False
        if len(visible) < _full_perimeter_size(rect):
            hidden_count += 1

    if hidden_count == 0:
        return False

    if force_segment_overlap:
        if not any(
            _has_long_shared_segment(rectangles[i], rectangles[j])
            for i in range(len(rectangles))
            for j in range(i + 1, len(rectangles))
        ):
            return False

    return np.array_equal(main(scene), scene)


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*_GRID_H_RANGE))
    w = int(rng.integers(*_GRID_W_RANGE))
    n_rects = int(rng.integers(*_RECT_COUNT_RANGE))
    force_segment_overlap = bool(rng.integers(2))

    rectangles = _sample_rectangles(rng, h, w, n_rects)
    if rectangles is None:
        return None

    colors = [int(c) for c in rng.choice(np.array(_COLORS), size=n_rects, replace=False)]
    order = [int(i) for i in rng.permutation(n_rects)]
    scene = _render_scene((h, w), rectangles, colors, order)

    if _generated_scene_is_valid(scene, rectangles, colors, force_segment_overlap):
        return scene
    return None


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        scene = _generate_attempt(rng)
        if scene is not None:
            return scene
    raise RuntimeError("failed to generate a valid occluded-rectangle scene")
