from __future__ import annotations

from collections import Counter
import random

import numpy as np


Rect = tuple[int, int, int, int]


COLOR_POOL = tuple(range(10))
GEN_RETRIES = 400

BODY_HEIGHT_RANGE = (14, 20)
BODY_WIDTH_RANGE = (16, 24)
RECT_COUNT_RANGE = (3, 5)
MAX_BODY_FILL_FRACTION = 0.5
LEGEND_RETRIES = 120
ROOT_RETRIES = 160
LEAF_MIN_SIDE = 2
ROOT_GAP = 1


def _bg_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> Rect:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _intersection(a: Rect, b: Rect) -> Rect | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 > r1 or c0 > c1:
        return None
    return int(r0), int(c0), int(r1), int(c1)


def _strictly_inside(inner: Rect, outer: Rect) -> bool:
    return bool(
        inner[0] > outer[0]
        and inner[1] > outer[1]
        and inner[2] < outer[2]
        and inner[3] < outer[3]
    )


def _render_body(
    height: int,
    width: int,
    bg: int,
    rectangles: dict[int, Rect],
    order: list[int],
) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rectangles[int(color)]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _legend_order(row: np.ndarray, bg: int) -> list[int]:
    return [int(v) for v in row if int(v) != bg]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover each partially occluded solid rectangle from its visible bbox, then redraw
    all rectangles using the new bottom legend order from bottom to top.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2

    bg = _bg_color(g)
    body = g[:-1, :]
    legend = g[-1, :]
    order = _legend_order(legend, bg)
    assert order and len(order) == len(set(order))

    body_colors = {int(v) for v in np.unique(body) if int(v) != bg}
    assert body_colors == set(order)

    rectangles: dict[int, Rect] = {}
    for color in order:
        coords = np.argwhere(body == int(color))
        assert coords.size > 0
        rectangles[int(color)] = _bbox(coords)

    return _render_body(body.shape[0], body.shape[1], bg, rectangles, order)


def _visible_bbox_matches(body: np.ndarray, color: int, rect: Rect) -> bool:
    coords = np.argwhere(body == int(color))
    return bool(coords.size > 0 and _bbox(coords) == rect)


def _union_area(height: int, width: int, rectangles: dict[int, Rect]) -> int:
    mask = np.zeros((height, width), dtype=bool)
    for r0, c0, r1, c1 in rectangles.values():
        mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return int(mask.sum())


def _sample_spaced_positions(length: int, n: int) -> list[int] | None:
    if length < 2 * n - 1:
        return None
    candidates = list(range(length))
    for _ in range(LEGEND_RETRIES):
        picked = sorted(random.sample(candidates, n))
        if all(b - a >= 2 for a, b in zip(picked, picked[1:])):
            return picked
    return None


def _separated(a: Rect, b: Rect, gap: int = ROOT_GAP) -> bool:
    return bool(
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _chain_partitions(n: int) -> list[tuple[int, ...]]:
    if n == 3:
        return [(3,), (2, 1)]
    if n == 4:
        return [(4,), (3, 1), (2, 2)]
    return [(5,), (4, 1), (3, 2)]


def _sample_root_rect(height: int, width: int, depth: int, existing: list[Rect]) -> Rect | None:
    min_side = LEAF_MIN_SIDE + 2 * (depth - 1)
    max_h = min(height, min_side + 5)
    max_w = min(width, min_side + 5)
    if min_side > max_h or min_side > max_w:
        return None

    for _ in range(ROOT_RETRIES):
        rect_h = random.randint(min_side, max_h)
        rect_w = random.randint(min_side, max_w)
        r0 = random.randint(0, height - rect_h)
        c0 = random.randint(0, width - rect_w)
        rect = (int(r0), int(c0), int(r0 + rect_h - 1), int(c0 + rect_w - 1))
        if all(_separated(rect, other) for other in existing):
            return rect
    return None


def _sample_child_rect(parent: Rect, remaining_descendants: int) -> Rect | None:
    min_side = LEAF_MIN_SIDE + 2 * remaining_descendants
    max_h = parent[2] - parent[0] - 1
    max_w = parent[3] - parent[1] - 1
    if min_side > max_h or min_side > max_w:
        return None

    rect_h = random.randint(min_side, max_h)
    rect_w = random.randint(min_side, max_w)
    r0 = random.randint(parent[0] + 1, parent[2] - rect_h)
    c0 = random.randint(parent[1] + 1, parent[3] - rect_w)
    return int(r0), int(c0), int(r0 + rect_h - 1), int(c0 + rect_w - 1)


def _interleave_chains(chains: list[list[int]]) -> list[int]:
    queues = [chain[:] for chain in chains]
    out: list[int] = []
    while any(queues):
        active = [i for i, chain in enumerate(queues) if chain]
        idx = random.choice(active)
        out.append(int(queues[idx].pop(0)))
    return out


def _sample_scene() -> np.ndarray | None:
    body_h = random.randint(*BODY_HEIGHT_RANGE)
    body_w = random.randint(*BODY_WIDTH_RANGE)
    rect_count = random.randint(*RECT_COUNT_RANGE)

    bg = random.choice(COLOR_POOL)
    colors = random.sample([c for c in COLOR_POOL if c != bg], rect_count)
    chain_lengths = random.choice(_chain_partitions(rect_count))

    rectangles: dict[int, Rect] = {}
    color_chains: list[list[int]] = []
    roots: list[Rect] = []
    color_idx = 0

    for chain_len in chain_lengths:
        chain_colors = colors[color_idx : color_idx + chain_len]
        color_idx += chain_len

        root = _sample_root_rect(body_h, body_w, chain_len, roots)
        if root is None:
            return None
        roots.append(root)
        rectangles[int(chain_colors[0])] = root
        color_chains.append([int(chain_colors[0])])

        parent = root
        for depth_index, color in enumerate(chain_colors[1:], start=1):
            remaining_descendants = chain_len - depth_index - 1
            child = _sample_child_rect(parent, remaining_descendants)
            if child is None:
                return None
            rectangles[int(color)] = child
            color_chains[-1].append(int(color))
            parent = child

    order = _interleave_chains(color_chains)

    if _union_area(body_h, body_w, rectangles) > MAX_BODY_FILL_FRACTION * body_h * body_w:
        return None

    body = _render_body(body_h, body_w, bg, rectangles, order)
    if not all(_visible_bbox_matches(body, color, rectangles[color]) for color in order):
        return None

    for _ in range(60):
        new_order = random.sample(order, len(order))
        if new_order == order:
            continue
        expected = _render_body(body_h, body_w, bg, rectangles, new_order)
        if not np.array_equal(expected, body):
            break
    else:
        return None

    legend_cols = _sample_spaced_positions(body_w, len(new_order))
    if legend_cols is None:
        return None
    legend = np.full(body_w, bg, dtype=int)
    for col, color in zip(legend_cols, new_order):
        legend[int(col)] = int(color)

    grid = np.vstack([body, legend[np.newaxis, :]])
    if _bg_color(grid) != bg:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _sample_scene()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
