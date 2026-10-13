import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
RECT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 500


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1


def _foreground_colors(g: np.ndarray) -> list[int]:
    colors: list[int] = []
    seen = {BLACK}
    for value in g.reshape(-1):
        color = int(value)
        if color not in seen:
            seen.add(color)
            colors.append(color)
    return colors


def _visible_bbox(g: np.ndarray, color: int) -> tuple[int, int, int, int]:
    rr, cc = np.where(g == color)
    assert len(rr) > 0, "each rectangle color must have visible pixels"
    return int(rr.min()), int(cc.min()), int(rr.max()), int(cc.max())


def _intersection(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> tuple[int, int, int, int] | None:
    r0 = max(a[0], b[0])
    c0 = max(a[1], b[1])
    r1 = min(a[2], b[2])
    c1 = min(a[3], b[3])
    if r0 <= r1 and c0 <= c1:
        return r0, c0, r1, c1
    return None


def _layer_order(g: np.ndarray, colors: list[int], bboxes: dict[int, tuple[int, int, int, int]]) -> list[int]:
    above: dict[int, set[int]] = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}

    for i, lower_candidate in enumerate(colors):
        for upper_candidate in colors[i + 1:]:
            inter = _intersection(bboxes[lower_candidate], bboxes[upper_candidate])
            if inter is None:
                continue

            r0, c0, r1, c1 = inter
            patch = g[r0:r1 + 1, c0:c1 + 1]
            lower_seen = np.any(patch == lower_candidate)
            upper_seen = np.any(patch == upper_candidate)

            if lower_seen:
                assert upper_candidate not in above[lower_candidate], "conflicting overlap order"
                if lower_candidate not in above[upper_candidate]:
                    above[upper_candidate].add(lower_candidate)
                    indegree[lower_candidate] += 1

            if upper_seen:
                assert lower_candidate not in above[upper_candidate], "conflicting overlap order"
                if upper_candidate not in above[lower_candidate]:
                    above[lower_candidate].add(upper_candidate)
                    indegree[upper_candidate] += 1

    order: list[int] = []
    remaining = set(colors)
    while remaining:
        choices = [color for color in colors if color in remaining and indegree[color] == 0]
        assert len(choices) == 1, "layering order must be uniquely determined"
        color = choices[0]
        order.append(color)
        remaining.remove(color)
        for upper in above[color]:
            indegree[upper] -= 1

    return order


def _render_with_grey_frames(order: list[int], bboxes: dict[int, tuple[int, int, int, int]]) -> np.ndarray:
    min_r = min(bboxes[color][0] for color in order) - 1
    min_c = min(bboxes[color][1] for color in order) - 1
    max_r = max(bboxes[color][2] for color in order) + 1
    max_c = max(bboxes[color][3] for color in order) + 1
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), BLACK, dtype=int)

    for color in order:
        r0, c0, r1, c1 = bboxes[color]
        br0, bc0 = r0 - 1 - min_r, c0 - 1 - min_c
        br1, bc1 = r1 + 1 - min_r, c1 + 1 - min_c
        rr0, cc0 = r0 - min_r, c0 - min_c
        rr1, cc1 = r1 - min_r, c1 - min_c
        out[br0:br1 + 1, bc0:bc1 + 1] = GREY
        out[rr0:rr1 + 1, cc0:cc1 + 1] = int(color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore layered overlapping colored rectangles and redraw them with grey outer frames.

    Input structure:
    - The grid background is Black.
    - Each non-Black foreground color is the visible part of one solid rectangle layer.
    - Higher rectangles hide lower rectangles in overlap regions, while each color's visible pixels still span that rectangle's full bounding box.

    Rule:
    - Infer one complete rectangle bounding box from the visible pixels of each foreground color.
    - For every pair of inferred rectangles, read their overlap: a color visible there must be above the other rectangle.
    - Use the unique bottom-to-top order implied by those overlap constraints.
    - On a new cropped Black canvas, redraw the complete rectangles from bottom to top, giving each rectangle a one-pixel Grey outer border before filling its original colored interior.

    Key insight: The visible overlap colors encode the layer order; the output makes that hidden order explicit by redrawing every full rectangle with a Grey frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = _foreground_colors(g)
    assert colors, "expected at least one foreground rectangle"
    assert GREY not in colors, "Grey is reserved for output borders"

    bboxes = {color: _visible_bbox(g, color) for color in colors}
    order = _layer_order(g, colors, bboxes)
    return _render_with_grey_frames(order, bboxes)


def _draw_rect(grid: np.ndarray, rect: Rect) -> None:
    grid[rect.r0:rect.r1 + 1, rect.c0:rect.c1 + 1] = rect.color


def _render_input(shape: tuple[int, int], rects: list[Rect]) -> np.ndarray:
    grid = np.full(shape, BLACK, dtype=int)
    for rect in rects:
        _draw_rect(grid, rect)
    return grid


def _expected_output(rects: list[Rect]) -> np.ndarray:
    bboxes = {rect.color: (rect.r0, rect.c0, rect.r1, rect.c1) for rect in rects}
    return _render_with_grey_frames([rect.color for rect in rects], bboxes)


def _choose_inner_offset(outer_len: int, inner_len: int, mode: str) -> int:
    slack = outer_len - inner_len
    assert slack >= 2
    if mode == "near_start":
        return 1
    if mode == "near_end":
        return slack - 1
    if mode == "center":
        return slack // 2
    return random.randint(1, slack - 1)


def _nested_rectangles() -> tuple[tuple[int, int], list[Rect]] | None:
    count = random.randint(3, 6)
    min_outer = 3 + 2 * (count - 1)
    outer_h = random.randint(min_outer, 28)
    outer_w = random.randint(min_outer, 28)

    sizes = [(outer_h, outer_w)]
    for remaining in range(count - 1, 0, -1):
        prev_h, prev_w = sizes[-1]
        min_next = 3 + 2 * (remaining - 1)
        max_h = prev_h - 2
        max_w = prev_w - 2
        if max_h < min_next or max_w < min_next:
            return None
        next_h = random.randint(min_next, max_h)
        next_w = random.randint(min_next, max_w)
        sizes.append((next_h, next_w))

    grid_h = random.randint(outer_h, 30)
    grid_w = random.randint(outer_w, 30)
    base_r = random.randint(0, grid_h - outer_h)
    base_c = random.randint(0, grid_w - outer_w)

    colors = random.sample(RECT_COLORS, count)
    rects = [Rect(colors[0], base_r, base_c, base_r + outer_h - 1, base_c + outer_w - 1)]
    row_modes = ["near_start", "near_end", "center", "random"]
    col_modes = ["near_start", "near_end", "center", "random"]

    for idx, (height, width) in enumerate(sizes[1:], start=1):
        parent = rects[-1]
        row_mode = random.choice(row_modes)
        col_mode = random.choice(col_modes)
        dr = _choose_inner_offset(parent.height, height, row_mode)
        dc = _choose_inner_offset(parent.width, width, col_mode)
        r0 = parent.r0 + dr
        c0 = parent.c0 + dc
        rects.append(Rect(colors[idx], r0, c0, r0 + height - 1, c0 + width - 1))

    return (grid_h, grid_w), rects


def _gen_attempt() -> np.ndarray | None:
    result = _nested_rectangles()
    if result is None:
        return None
    shape, rects = result
    grid = _render_input(shape, rects)
    expected = _expected_output(rects)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if any(np.count_nonzero(grid == rect.color) == 0 for rect in rects):
        return None

    try:
        got = main(grid)
    except Exception:
        return None
    if not np.array_equal(got, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-rectangle input")
