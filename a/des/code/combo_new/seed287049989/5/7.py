from __future__ import annotations

import random

import numpy as np


Rect = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> Rect:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _inside(box: Rect, row: int, col: int) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= row <= r1 and c0 <= col <= c1


def _detect_palette(grid: np.ndarray, bg: int) -> tuple[str, list[int]]:
    h, w = grid.shape
    assert w >= 3, "expected room for a palette column and object area"

    candidates: list[tuple[int, str, tuple[int, ...]]] = []
    for side, col, inward in (("left", 0, 1), ("right", w - 1, w - 2)):
        r = 0
        while r < h:
            if int(grid[r, col]) == bg:
                r += 1
                continue
            start = r
            colors: list[int] = []
            while r < h and int(grid[r, col]) != bg:
                colors.append(int(grid[r, col]))
                r += 1
            end = r - 1
            if len(colors) < 2 or len(colors) != len(set(colors)):
                continue
            if np.all(grid[start : end + 1, inward] == bg):
                candidates.append((len(colors), side, tuple(colors)))

    assert candidates, "expected a unique edge palette strip"
    best_len = max(item[0] for item in candidates)
    best = [item for item in candidates if item[0] == best_len]
    assert len(best) == 1, "palette strip must be unique"
    _length, side, colors = best[0]
    return side, list(colors)


def _strip_palette(grid: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return np.array(grid[:, 1:], dtype=int, copy=True)
    assert side == "right"
    return np.array(grid[:, :-1], dtype=int, copy=True)


def _infer_rect_size(core: np.ndarray, bg: int, colors: list[int]) -> tuple[int, int]:
    candidates: list[tuple[int, int, int]] = []
    for color in colors:
        coords = np.argwhere(core == color)
        assert coords.size > 0
        r0, c0, r1, c1 = _bbox(coords)
        patch = core[r0 : r1 + 1, c0 : c1 + 1]
        if np.all(patch != bg):
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            candidates.append((rect_h * rect_w, rect_h, rect_w))

    if not candidates:
        for color in colors:
            coords = np.argwhere(core == color)
            r0, c0, r1, c1 = _bbox(coords)
            rect_h = r1 - r0 + 1
            rect_w = c1 - c0 + 1
            if int(coords.shape[0]) == rect_h * rect_w:
                candidates.append((rect_h * rect_w, rect_h, rect_w))

    assert candidates, "expected at least one recoverable rectangle size"
    max_area = max(area for area, _h, _w in candidates)
    dims = {(h, w) for area, h, w in candidates if area == max_area}
    assert len(dims) == 1, "rectangle size must be unique"
    return next(iter(dims))


def _candidate_boxes(core: np.ndarray, bg: int, color: int, rect_h: int, rect_w: int) -> list[Rect]:
    coords = np.argwhere(core == color)
    assert coords.size > 0
    rmin, cmin, rmax, cmax = _bbox(coords)
    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1
    occupied = core != bg

    if bbox_h == rect_h and bbox_w == rect_w and np.all(occupied[rmin : rmax + 1, cmin : cmax + 1]):
        return [(rmin, cmin, rmax, cmax)]

    h, w = core.shape
    candidates: set[Rect] = set()
    row_lo = max(0, rmax - rect_h + 1)
    row_hi = min(rmin, h - rect_h)
    col_lo = max(0, cmax - rect_w + 1)
    col_hi = min(cmin, w - rect_w)
    for r0 in range(row_lo, row_hi + 1):
        for c0 in range(col_lo, col_hi + 1):
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            if np.all(occupied[r0 : r1 + 1, c0 : c1 + 1]):
                candidates.add((r0, c0, r1, c1))

    result = sorted(candidates)
    assert result, f"no candidate rectangle for color {color}"
    return result


def _unique_topological_order(colors: list[int], edges: dict[int, set[int]]) -> list[int] | None:
    indegree = {color: 0 for color in colors}
    for src in colors:
        for dst in edges[src]:
            indegree[dst] += 1

    remaining = set(colors)
    order: list[int] = []
    while remaining:
        frontier = [color for color in remaining if indegree[color] == 0]
        if len(frontier) != 1:
            return None
        node = frontier[0]
        remaining.remove(node)
        order.append(node)
        for dst in edges[node]:
            indegree[dst] -= 1
    return order


def _assignment_order(core: np.ndarray, bg: int, boxes: dict[int, Rect], colors: list[int]) -> list[int] | None:
    edges = {color: set() for color in colors}
    for row, col in np.argwhere(core != bg):
        r = int(row)
        c = int(col)
        observed = int(core[r, c])
        carriers = [color for color in colors if _inside(boxes[color], r, c)]
        if observed not in carriers:
            return None
        for color in carriers:
            if color != observed:
                edges[color].add(observed)
    return _unique_topological_order(colors, edges)


def _render(
    shape: tuple[int, int],
    bg: int,
    boxes: dict[int, Rect],
    bottom_to_top: list[int],
    color_map: dict[int, int] | None = None,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in bottom_to_top:
        paint = color if color_map is None else int(color_map[color])
        r0, c0, r1, c1 = boxes[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = paint
    return out


def _solve_core(core: np.ndarray, palette: list[int], bg: int) -> np.ndarray:
    colors = [int(color) for color in np.unique(core) if int(color) != bg]
    assert colors, "expected at least one rectangle color"
    assert len(colors) == len(palette), "palette depth count must match rectangle count"

    rect_h, rect_w = _infer_rect_size(core, bg, colors)
    candidates = {color: _candidate_boxes(core, bg, color, rect_h, rect_w) for color in colors}
    search_order = sorted(colors, key=lambda color: (len(candidates[color]), -int(np.count_nonzero(core == color))))

    chosen: dict[int, Rect] = {}
    outputs: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    def backtrack(index: int) -> None:
        if index == len(search_order):
            order = _assignment_order(core, bg, chosen, colors)
            if order is None:
                return
            top_to_bottom = list(reversed(order))
            recolor = {color: palette[i] for i, color in enumerate(top_to_bottom)}
            out = _render(core.shape, bg, chosen, order, recolor)
            outputs[(out.shape, out.tobytes())] = out.copy()
            return

        color = search_order[index]
        for box in candidates[color]:
            chosen[color] = box
            backtrack(index + 1)
            del chosen[color]

    backtrack(0)
    assert outputs, "failed to reconstruct rectangle stack"
    assert len(outputs) == 1, "reconstruction must have a unique output"
    return next(iter(outputs.values()))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapping equal-size rectangles, reorder them by occlusion depth, and repaint them from an edge palette.

    Input structure:
    - The most common color is the background.
    - Several solid rectangles of identical size but different colors overlap one another, so lower rectangles may be only partially visible.
    - A one-column vertical palette strip sits on the left or right edge; its colors run from top to bottom and encode topmost to bottommost depth.

    Rule:
    - Locate and remove the edge palette strip from the working canvas.
    - Infer the common rectangle size, then recover one full rectangle box for each visible rectangle color by finding the only same-size placement that contains that color's visible cells and never covers background.
    - Use occlusion to recover layer order: whenever a recovered rectangle's area contains visible pixels of another rectangle color, that other rectangle must lie above it.
    - Repaint the recovered rectangles with the palette colors by depth, using the first palette color for the top layer and the last for the bottom layer, then redraw the rectangles in the original stacking order.

    Key insight:
    - The puzzle combines three linked inferences: the edge strip is a depth legend, identical rectangle size lets you complete partially hidden boxes, and visible overlap colors reveal the unique front-to-back layering.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    side, palette = _detect_palette(g, bg)
    core = _strip_palette(g, side)
    return _solve_core(core, palette, bg)


MAX_GENERATION_ATTEMPTS = 300
RECT_COUNT_RANGE = (3, 4)
RECT_SIDE_MIN = 4
RECT_SIDE_MAX = 7
STEP_MAG_MAX = 3
MARGIN_MIN = 1
MARGIN_MAX = 3
SIDE_OPTIONS = ("left", "right")
DIAGONAL_SIGNS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _gen_attempt() -> np.ndarray | None:
    count = random.randint(*RECT_COUNT_RANGE)
    rect_h = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)
    rect_w = random.randint(RECT_SIDE_MIN, RECT_SIDE_MAX)

    sign_r, sign_c = random.choice(DIAGONAL_SIGNS)
    step_r = sign_r * random.randint(1, min(STEP_MAG_MAX, rect_h - 1))
    step_c = sign_c * random.randint(1, min(STEP_MAG_MAX, rect_w - 1))

    row_offsets = [depth * step_r for depth in range(count)]
    col_offsets = [depth * step_c for depth in range(count)]
    min_row_off, max_row_off = min(row_offsets), max(row_offsets)
    min_col_off, max_col_off = min(col_offsets), max(col_offsets)

    top_margin = random.randint(MARGIN_MIN, MARGIN_MAX)
    bottom_margin = random.randint(MARGIN_MIN, MARGIN_MAX)
    left_margin = random.randint(MARGIN_MIN, MARGIN_MAX)
    right_margin = random.randint(MARGIN_MIN, MARGIN_MAX)

    work_h = top_margin + (max_row_off - min_row_off) + rect_h + bottom_margin
    work_w = left_margin + (max_col_off - min_col_off) + rect_w + right_margin
    if max(work_h, work_w + 1) > 30:
        return None

    bg = random.randint(0, 9)
    palette_pool = [color for color in range(10) if color != bg]
    if len(palette_pool) < 2 * count:
        return None
    sampled = random.sample(palette_pool, 2 * count)
    object_colors = sampled[:count]
    palette_colors = sampled[count:]

    base_r = top_margin - min_row_off
    base_c = left_margin - min_col_off
    boxes: dict[int, Rect] = {}
    bottom_to_top: list[int] = []
    for depth, color in enumerate(object_colors):
        r0 = base_r + row_offsets[depth]
        c0 = base_c + col_offsets[depth]
        boxes[color] = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
        bottom_to_top.append(color)

    core = _render((work_h, work_w), bg, boxes, bottom_to_top)
    recolor = {color: palette_colors[count - 1 - depth] for depth, color in enumerate(bottom_to_top)}
    expected = _render((work_h, work_w), bg, boxes, bottom_to_top, recolor)

    side = random.choice(SIDE_OPTIONS)
    palette_start = random.randint(0, work_h - count)
    inp = np.full((work_h, work_w + 1), bg, dtype=int)
    if side == "left":
        inp[:, 1:] = core
        palette_col = 0
    else:
        inp[:, :work_w] = core
        palette_col = work_w
    inp[palette_start : palette_start + count, palette_col] = np.asarray(palette_colors, dtype=int)

    try:
        solved = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
