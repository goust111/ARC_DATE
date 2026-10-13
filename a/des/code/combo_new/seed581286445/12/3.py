from __future__ import annotations

import random

import numpy as np


def _border_background(g: np.ndarray) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    values, counts = np.unique(border, return_counts=True)
    return int(values[counts.argmax()])


def _separator_row(g: np.ndarray, bg: int) -> int:
    candidates: list[int] = []
    for r in range(1, g.shape[0] - 1):
        row = g[r]
        if int(row[0]) == bg:
            continue
        if np.all(row == row[0]) and np.any(g[:r] != bg) and np.any(g[r + 1 :] != bg):
            candidates.append(r)
    assert len(candidates) == 1, "expected one solid separator row"
    return int(candidates[0])


def _legend_order(top: np.ndarray, bg: int) -> list[int]:
    colors = [int(color) for color in np.unique(top) if int(color) != bg]
    blocks: list[tuple[int, int]] = []
    for color in colors:
        coords = np.argwhere(top == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        assert int(coords.shape[0]) == 9, "legend block must be 3x3"
        assert int(r1 - r0) == 2 and int(c1 - c0) == 2, "legend block must be 3x3"
        assert np.all(top[r0 : r1 + 1, c0 : c1 + 1] == color)
        blocks.append((int(c0), color))
    blocks.sort()
    order = [color for _c0, color in blocks]
    assert order and len(order) == len(set(order)), "legend colors must be unique"
    return order


def _prefix_sum(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(int), ((1, 0), (1, 0)))
    return padded.cumsum(axis=0).cumsum(axis=1)


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _reconstruct_rectangle(visible: np.ndarray, covered: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(visible)
    assert coords.size > 0, "every legend color must stay visible on the canvas"

    rmin, cmin = coords.min(axis=0)
    rmax, cmax = coords.max(axis=0)
    invalid = ~(visible | covered)
    prefix = _prefix_sum(invalid)

    best_key: tuple[int, int, int, int, int] | None = None
    best_rect: tuple[int, int, int, int] | None = None

    for r0 in range(int(rmin) + 1):
        for r1 in range(int(rmax), visible.shape[0]):
            for c0 in range(int(cmin) + 1):
                for c1 in range(int(cmax), visible.shape[1]):
                    if _rect_sum(prefix, r0, c0, r1, c1) != 0:
                        continue
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    key = (area, -r0, r1, -c0, c1)
                    if best_key is None or key > best_key:
                        best_key = key
                        best_rect = (r0, c0, r1, c1)

    assert best_rect is not None, "could not infer a full rectangle for one color"
    return best_rect


def _render_stack(
    shape: tuple[int, int],
    order: list[int],
    rectangles: dict[int, tuple[int, int, int, int]],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in order:
        r0, c0, r1, c1 = rectangles[color]
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def _packed_output(order: list[int], rectangles: dict[int, tuple[int, int, int, int]], bg: int) -> np.ndarray:
    dims = [(r1 - r0 + 1, c1 - c0 + 1) for r0, c0, r1, c1 in (rectangles[color] for color in order)]
    out_h = max(h for h, _w in dims)
    out_w = sum(w for _h, w in dims) + len(dims) - 1
    out = np.full((out_h, out_w), bg, dtype=int)

    c = 0
    for color, (h, w) in zip(order, dims, strict=False):
        out[:h, c : c + w] = int(color)
        c += w + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the legend-defined overprint order, recover the full hidden rectangles, and lay them out side by side.

    Input structure:
    - The background is black, and one solid grey row separates an upper legend area from a lower canvas area.
    - The legend area contains 3x3 monochrome color blocks arranged left to right with one background column between blocks.
    - The canvas area contains solid monochrome rectangles whose colors match the legend blocks, and higher-priority colors visibly cover lower-priority rectangles where they overlap.

    Rule:
    - Read the legend blocks from left to right to obtain the rectangle order from bottom layer to top layer.
    - Reconstruct each full axis-aligned rectangle on the canvas by processing colors from top layer down: a color's true rectangle is the largest rectangle containing its visible cells whose hidden cells are fully explained by already reconstructed higher layers.
    - Remove the legend and separator, then place the recovered rectangles left to right in legend order with one background column between them and their top edges aligned.

    Key insight:
    - The legend does not label shapes directly; it encodes the stacking order needed to explain which missing cells are hidden by higher rectangles rather than absent from the object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _border_background(g)
    sep = _separator_row(g, bg)
    top = g[:sep]
    canvas = g[sep + 1 :]
    assert top.size > 0 and canvas.size > 0

    order = _legend_order(top, bg)
    rectangles: dict[int, tuple[int, int, int, int]] = {}
    covered = np.zeros_like(canvas, dtype=bool)

    for color in reversed(order):
        visible = canvas == int(color)
        rect = _reconstruct_rectangle(visible, covered)
        rectangles[color] = rect
        r0, c0, r1, c1 = rect
        covered[r0 : r1 + 1, c0 : c1 + 1] = True

    assert np.array_equal(_render_stack(canvas.shape, order, rectangles, bg), canvas)
    out = _packed_output(order, rectangles, bg)
    assert max(out.shape) <= 30
    return out


BG = 0
SEP = 5
LEGEND_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
LEGEND_BLOCK = 3
MAX_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 120


def _sample_dimensions(n_rects: int) -> list[tuple[int, int]] | None:
    dims: list[tuple[int, int]] = []
    remaining_width = 30 - (n_rects - 1)
    for i in range(n_rects):
        min_rest = 2 * (n_rects - i - 1)
        max_w = min(8, remaining_width - min_rest)
        if max_w < 2:
            return None
        w = random.randint(2, max_w)
        h = random.randint(2, 8)
        dims.append((h, w))
        remaining_width -= w
    return dims


def _rect_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> int:
    ar, ac, ah, aw = a
    br, bc, bh, bw = b
    dh = min(ar + ah, br + bh) - max(ar, br)
    dw = min(ac + aw, bc + bw) - max(ac, bc)
    if dh <= 0 or dw <= 0:
        return 0
    return int(dh * dw)


def _free_position(h: int, w: int, canvas_h: int, canvas_w: int) -> tuple[int, int] | None:
    if canvas_h - h - 1 < 1 or canvas_w - w - 1 < 1:
        return None
    return (
        random.randint(1, canvas_h - h - 1),
        random.randint(1, canvas_w - w - 1),
    )


def _overlap_position(
    h: int,
    w: int,
    canvas_h: int,
    canvas_w: int,
    target: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    tr, tc, th, tw = target
    r_lo = max(1, tr - h + 1)
    r_hi = min(canvas_h - h - 1, tr + th - 1)
    c_lo = max(1, tc - w + 1)
    c_hi = min(canvas_w - w - 1, tc + tw - 1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    for _ in range(20):
        r = random.randint(r_lo, r_hi)
        c = random.randint(c_lo, c_hi)
        if _rect_overlap((r, c, h, w), target) > 0:
            return r, c
    return None


def _aligned_inside_position(
    h: int,
    w: int,
    canvas_h: int,
    canvas_w: int,
    target: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    tr, tc, th, tw = target
    if h > th or w > tw:
        return None
    options: list[tuple[int, int]] = []
    for r in (tr, tr + th - h):
        for c in range(tc, tc + tw - w + 1):
            options.append((r, c))
    for c in (tc, tc + tw - w):
        for r in range(tr, tr + th - h + 1):
            options.append((r, c))
    options = [
        (r, c)
        for r, c in options
        if 1 <= r <= canvas_h - h - 1 and 1 <= c <= canvas_w - w - 1
    ]
    if not options:
        return None
    return random.choice(options)


def _place_rectangles(dims: list[tuple[int, int]], canvas_h: int, canvas_w: int) -> list[tuple[int, int, int, int]] | None:
    rectangles: list[tuple[int, int, int, int]] = []
    overlap_count = 0

    for index, (h, w) in enumerate(dims):
        placed = None
        for _ in range(PLACEMENT_ATTEMPTS):
            if index == 0:
                pos = _free_position(h, w, canvas_h, canvas_w)
            else:
                need_overlap = overlap_count == 0 or random.random() < 0.7
                target = random.choice(rectangles)
                if need_overlap:
                    mode = random.choice(["aligned", "overlap"])
                else:
                    mode = random.choice(["free", "overlap", "aligned"])
                if mode == "free":
                    pos = _free_position(h, w, canvas_h, canvas_w)
                elif mode == "aligned":
                    pos = _aligned_inside_position(h, w, canvas_h, canvas_w, target)
                    if pos is None:
                        pos = _overlap_position(h, w, canvas_h, canvas_w, target)
                else:
                    pos = _overlap_position(h, w, canvas_h, canvas_w, target)
            if pos is None:
                continue
            rect = (pos[0], pos[1], h, w)
            if index > 0 and all(_rect_overlap(rect, prev) == 0 for prev in rectangles) and need_overlap:
                continue
            placed = rect
            break
        if placed is None:
            return None
        overlap_count += sum(_rect_overlap(placed, prev) > 0 for prev in rectangles)
        rectangles.append(placed)

    if overlap_count == 0:
        return None
    return rectangles


def _render_canvas(
    order: list[int],
    rectangles: list[tuple[int, int, int, int]],
    canvas_h: int,
    canvas_w: int,
) -> np.ndarray:
    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    for color, (r, c, h, w) in zip(order, rectangles, strict=False):
        canvas[r : r + h, c : c + w] = int(color)
    return canvas


def _render_input(
    order: list[int],
    rectangles: list[tuple[int, int, int, int]],
    canvas_h: int,
    canvas_w: int,
) -> np.ndarray:
    legend_w = len(order) * LEGEND_BLOCK + (len(order) - 1)
    top_margin = random.randint(1, 2)
    bottom_margin = random.randint(0, 1)
    top_h = top_margin + LEGEND_BLOCK + bottom_margin
    assert legend_w + 2 <= canvas_w

    grid = np.full((top_h + 1 + canvas_h, canvas_w), BG, dtype=int)
    legend_left = random.randint(1, canvas_w - legend_w - 1)
    c = legend_left
    for color in order:
        grid[top_margin : top_margin + LEGEND_BLOCK, c : c + LEGEND_BLOCK] = int(color)
        c += LEGEND_BLOCK + 1
    grid[top_h, :] = SEP
    grid[top_h + 1 :] = _render_canvas(order, rectangles, canvas_h, canvas_w)
    return grid


def _expected_output(order: list[int], dims: list[tuple[int, int]]) -> np.ndarray:
    out_h = max(h for h, _w in dims)
    out_w = sum(w for _h, w in dims) + len(dims) - 1
    out = np.full((out_h, out_w), BG, dtype=int)
    c = 0
    for color, (h, w) in zip(order, dims, strict=False):
        out[:h, c : c + w] = int(color)
        c += w + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    n_rects = random.randint(3, 5)
    order = random.sample(LEGEND_COLORS, n_rects)
    dims = _sample_dimensions(n_rects)
    if dims is None:
        return None

    legend_w = n_rects * LEGEND_BLOCK + (n_rects - 1)
    max_h = max(h for h, _w in dims)
    max_w = max(w for _h, w in dims)

    canvas_h_lo = max(max_h + 3, 10)
    canvas_h_hi = min(22, max_h + 10)
    canvas_w_lo = max(max_w + 3, legend_w + 2, 10)
    canvas_w_hi = min(28, max(canvas_w_lo, max_w + 10, legend_w + 8))
    if canvas_h_lo > canvas_h_hi or canvas_w_lo > canvas_w_hi:
        return None

    canvas_h = random.randint(canvas_h_lo, canvas_h_hi)
    canvas_w = random.randint(canvas_w_lo, canvas_w_hi)
    rectangles = _place_rectangles(dims, canvas_h, canvas_w)
    if rectangles is None:
        return None

    grid = _render_input(order, rectangles, canvas_h, canvas_w)
    expected = _expected_output(order, dims)

    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return None
    if np.count_nonzero(grid == BG) <= grid.size // 2:
        return None
    if any(not np.any(grid == color) for color in order):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_ATTEMPTS} attempts")
