from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4, 6)
SIZE_CHOICES = tuple((h, w) for h in range(2, 6) for w in range(2, 6))
GEN_ATTEMPTS = 1200
FIVE_LAYOUT = (
    (6, 15, 3, 2),
    (18, 10, 4, 4),
    (1, 13, 4, 5),
    (14, 4, 5, 4),
    (10, 9, 5, 5),
)


def _find_rectangles(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros_like(g, dtype=bool)
    rects: list[dict[str, int | tuple[int, int, int, int]]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BLACK or seen[r, c]:
                continue

            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            rows = [p[0] for p in cells]
            cols = [p[1] for p in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            assert np.all(patch == color), "each foreground object must be a solid rectangle"
            rects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "h": r1 - r0 + 1,
                    "w": c1 - c0 + 1,
                    "area": (r1 - r0 + 1) * (c1 - c0 + 1),
                }
            )

    assert rects, "expected at least one rectangle"
    return rects


def _ray_hits(
    grid: np.ndarray,
    start: tuple[int, int],
    step: tuple[int, int],
    self_color: int,
) -> int:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r, c = start
    dr, dc = step
    hits: set[int] = set()

    while 0 <= r < h and 0 <= c < w:
        color = int(g[r, c])
        if color != BLACK and color != self_color:
            hits.add(color)
        r += dr
        c += dc
    return len(hits)


def _dominance(grid: np.ndarray, rect: dict[str, int | tuple[int, int, int, int]]) -> int:
    r0, c0, r1, c1 = rect["bbox"]  # type: ignore[assignment]
    color = int(rect["color"])
    return sum(
        [
            _ray_hits(grid, (r0 - 1, c0 - 1), (-1, -1), color),
            _ray_hits(grid, (r0 - 1, c1 + 1), (-1, 1), color),
            _ray_hits(grid, (r1 + 1, c0 - 1), (1, -1), color),
            _ray_hits(grid, (r1 + 1, c1 + 1), (1, 1), color),
        ]
    )


def _rank_rectangles(grid: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    rects = _find_rectangles(grid)
    scored = []
    for rect in rects:
        item = dict(rect)
        item["dominance"] = _dominance(grid, rect)
        scored.append(item)

    scored.sort(
        key=lambda rect: (
            -int(rect["dominance"]),
            -int(rect["area"]),
            int(rect["bbox"][0]),  # type: ignore[index]
            int(rect["bbox"][1]),  # type: ignore[index]
        )
    )
    return scored


def _build_output(ordered_rects: list[dict[str, int | tuple[int, int, int, int]]]) -> np.ndarray:
    max_side = max(max(int(rect["h"]), int(rect["w"])) for rect in ordered_rects)
    size = max_side + 2 * (len(ordered_rects) - 1)
    out = np.zeros((size, size), dtype=int)
    out[:, :] = int(ordered_rects[0]["color"])

    for depth, rect in enumerate(ordered_rects[1:], start=1):
        rh = int(rect["h"])
        rw = int(rect["w"])
        left = depth
        bottom = size - 1 - depth
        top = bottom - rh + 1
        right = left + rw - 1
        out[top : bottom + 1, left : right + 1] = int(rect["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort solid rectangles by diagonal-ray penetration count, then nest them into the output from the lower-left.

    Input structure:
    - The grid has black background and 4-5 disconnected solid color rectangles.
    - Each rectangle is monochromatic, axis-aligned, and separated from the others by at least one background cell.
    - Rectangle colors are distinct controls; their sizes and positions do not directly encode the nesting order.

    Rule:
    - Detect each solid rectangle and, from its four corners, cast one outward 45-degree ray in the matching diagonal direction until the grid edge.
    - For each ray, count how many other rectangles it intersects, counting each crossed rectangle at most once per ray.
    - Sum the four ray counts to get that rectangle's dominance score, then sort rectangles from highest score to lowest.
    - Create a square output whose side is the largest rectangle side plus twice the number of inner nesting levels.
    - Fill the whole output with the highest-score rectangle color, then place each remaining rectangle at the lower-left corner of the next inner nesting level, preserving its original size and letting later placements overwrite earlier ones.

    Key insight:
    - The nesting order is not determined by area: it is encoded by how many other rectangles each rectangle's outward diagonal rays can penetrate.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    ordered = _rank_rectangles(g)
    return _build_output(ordered)


def _boxes_clear(box: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = box
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + 1 < pr0 or pr1 + 1 < r0 or c1 + 1 < pc0 or pc1 + 1 < c0):
            return False
    return True


def _paint_rectangles(
    shape: tuple[int, int],
    rect_specs: list[tuple[int, int, int, int, int]],
) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for color, r0, c0, rh, rw in rect_specs:
        grid[r0 : r0 + rh, c0 : c0 + rw] = color
    return grid


def _area_order(rects: list[dict[str, int | tuple[int, int, int, int]]]) -> list[int]:
    return [
        int(rect["color"])
        for rect in sorted(
            rects,
            key=lambda rect: (-int(rect["area"]), -max(int(rect["h"]), int(rect["w"]))),
        )
    ]


def _choose_sizes(count: int) -> list[tuple[int, int]]:
    options = list(SIZE_CHOICES)
    random.shuffle(options)
    return options[:count]


def _random_positions(
    height: int,
    width: int,
    sizes: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    order = list(range(len(sizes)))
    random.shuffle(order)
    placed_boxes: list[tuple[int, int, int, int]] = [(-2, -2, -2, -2)]
    positions: list[tuple[int, int] | None] = [None] * len(sizes)

    center_r = (height - 1) / 2.0
    center_c = (width - 1) / 2.0

    for idx in order:
        rh, rw = sizes[idx]
        candidates = []
        for r0 in range(1, height - rh):
            for c0 in range(1, width - rw):
                box = (r0, c0, r0 + rh - 1, c0 + rw - 1)
                if not _boxes_clear(box, placed_boxes[1:]):
                    continue
                cr = r0 + (rh - 1) / 2.0
                cc = c0 + (rw - 1) / 2.0
                centrality = abs(cr - center_r) + abs(cc - center_c)
                candidates.append((centrality, random.random(), r0, c0, box))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[0], item[1]))
        top = candidates[: max(12, len(candidates) // 5)]
        _, _, r0, c0, box = random.choice(top)
        positions[idx] = (r0, c0)
        placed_boxes.append(box)

    return [pos for pos in positions if pos is not None]


def _is_interesting(grid: np.ndarray) -> bool:
    rects = _find_rectangles(grid)
    ranked = _rank_rectangles(grid)
    dominance_values = [int(rect["dominance"]) for rect in ranked]
    if len(set(dominance_values)) != len(dominance_values):
        return False
    if max(dominance_values) < 2:
        return False
    if sum(dominance_values) < len(dominance_values) + 2:
        return False
    if [int(rect["color"]) for rect in ranked] == _area_order(rects):
        return False
    out = _build_output(ranked)
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False
    if len(np.unique(out[out != BLACK])) < 4:
        return False
    return True


def _generate_four_attempt() -> np.ndarray | None:
    height = random.randint(16, 20)
    width = random.randint(16, 20)
    sizes = _choose_sizes(4)
    positions = _random_positions(height, width, sizes)
    if positions is None:
        return None

    colors = random.sample(PALETTE, 4)
    specs: list[tuple[int, int, int, int, int]] = []
    for color, (rh, rw), (r0, c0) in zip(colors, sizes, positions):
        specs.append((color, r0, c0, rh, rw))

    grid = _paint_rectangles((height, width), specs)
    try:
        if _is_interesting(grid):
            return grid
    except AssertionError:
        return None
    return None


def _transform_layout(
    rects: list[tuple[int, int, int, int]],
    canvas_size: int,
    *,
    transpose: bool,
    flip_rows: bool,
    flip_cols: bool,
) -> list[tuple[int, int, int, int]]:
    out = rects[:]
    if transpose:
        out = [(c0, r0, rw, rh) for r0, c0, rh, rw in out]
    if flip_rows:
        out = [(canvas_size - (r0 + rh), c0, rh, rw) for r0, c0, rh, rw in out]
    if flip_cols:
        out = [(r0, canvas_size - (c0 + rw), rh, rw) for r0, c0, rh, rw in out]
    return out


def _generate_five_attempt() -> np.ndarray | None:
    base_size = 22
    target_size = random.randint(22, 24)
    layout = _transform_layout(
        list(FIVE_LAYOUT),
        base_size,
        transpose=bool(random.getrandbits(1)),
        flip_rows=bool(random.getrandbits(1)),
        flip_cols=bool(random.getrandbits(1)),
    )

    min_r = min(r0 for r0, _c0, _rh, _rw in layout)
    min_c = min(c0 for _r0, c0, _rh, _rw in layout)
    max_r = max(r0 + rh - 1 for r0, _c0, rh, _rw in layout)
    max_c = max(c0 + rw - 1 for _r0, c0, _rh, rw in layout)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1
    if span_h > target_size or span_w > target_size:
        return None

    shift_r = random.randint(0, target_size - span_h)
    shift_c = random.randint(0, target_size - span_w)
    colors = random.sample(PALETTE, 5)
    specs: list[tuple[int, int, int, int, int]] = []
    for color, (r0, c0, rh, rw) in zip(colors, layout):
        specs.append((color, r0 - min_r + shift_r, c0 - min_c + shift_c, rh, rw))

    grid = _paint_rectangles((target_size, target_size), specs)
    try:
        if _is_interesting(grid):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    branches = [_generate_four_attempt, _generate_five_attempt]
    if random.random() < 0.5:
        branches.reverse()
    for branch in branches:
        for _ in range(GEN_ATTEMPTS):
            grid = branch()
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid input")
