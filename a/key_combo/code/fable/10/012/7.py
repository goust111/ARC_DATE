import itertools
import random

import numpy as np


GEN_RETRIES = 400
BG = 0
PALETTE = tuple(range(1, 10))


def _is_contiguous(indices: np.ndarray) -> bool:
    if indices.size == 0:
        return False
    return bool(np.all(np.diff(indices) == 1))


def _projection_strips(canvas: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = canvas.shape
    top = np.full(w, bg, dtype=int)
    left = np.full(h, bg, dtype=int)

    for c in range(w):
        rows = np.flatnonzero(canvas[:, c] != bg)
        if rows.size:
            top[c] = int(canvas[int(rows[0]), c])

    for r in range(h):
        cols = np.flatnonzero(canvas[r, :] != bg)
        if cols.size:
            left[r] = int(canvas[r, int(cols[0])])

    return top, left


def _render_canvas(shape: tuple[int, int], rects: list[dict], order: tuple[int, ...], bg: int) -> np.ndarray:
    canvas = np.full(shape, bg, dtype=int)
    for idx in order:
        rect = rects[idx]
        canvas[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return canvas


def _parse_rectangles(grid: np.ndarray) -> tuple[int, np.ndarray, np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 2, "expected a 2D grid with projection strips"

    bg = int(g[0, 0])
    top = g[0, 1:]
    left = g[1:, 0]
    hints = g[1:, 1:]
    colors = sorted(set(map(int, top)) | set(map(int, left)) | set(map(int, hints.ravel())))
    colors = [color for color in colors if color != bg]
    assert colors, "expected at least one rectangle color"

    rects: list[dict] = []
    for color in colors:
        hint_cells = np.argwhere(hints == color)
        row_vis = np.flatnonzero(left == color)
        col_vis = np.flatnonzero(top == color)
        assert hint_cells.shape[0] == 1, "each color must have exactly one interior hint"
        assert row_vis.size > 0 and col_vis.size > 0, "each color must appear in both projection strips"
        assert _is_contiguous(row_vis), "left-strip projection for each color must be one contiguous segment"
        assert _is_contiguous(col_vis), "top-strip projection for each color must be one contiguous segment"

        hr, hc = map(int, hint_cells[0])
        r0 = min(hr, int(row_vis[0]))
        r1 = max(hr, int(row_vis[-1]))
        c0 = min(hc, int(col_vis[0]))
        c1 = max(hc, int(col_vis[-1]))
        rects.append({"color": int(color), "r0": r0, "c0": c0, "r1": r1, "c1": c1, "hint": (hr, hc)})

    return bg, top, left, rects


def _matching_canvases(shape: tuple[int, int], rects: list[dict], top: np.ndarray, left: np.ndarray, bg: int) -> list[np.ndarray]:
    matches: dict[bytes, np.ndarray] = {}
    for order in itertools.permutations(range(len(rects))):
        canvas = _render_canvas(shape, rects, order, bg)
        top_now, left_now = _projection_strips(canvas, bg)
        if np.array_equal(top_now, top) and np.array_equal(left_now, left):
            matches[canvas.tobytes()] = canvas
    return list(matches.values())


def main(grid: np.ndarray) -> np.ndarray:
    """Rebuild overlapping solid rectangles from top and left projection strips plus one hint pixel per color.

    Input structure:
    - The top row and left column are projection strips over an otherwise empty interior canvas.
    - In the top strip, each column shows the first rectangle color seen from above in that interior column.
    - In the left strip, each row shows the first rectangle color seen from the left in that interior row.
    - Inside the canvas, each rectangle contributes exactly one same-color hint pixel somewhere in its true area.

    Rule:
    - For each color, take its contiguous top-strip segment, contiguous left-strip segment, and single interior hint.
    - The rectangle's full row span is the hull of its left-strip rows and hint row; its full column span is the hull
      of its top-strip columns and hint column.
    - Reconstruct all solid rectangles and choose the unique back-to-front painting order whose rendered top and left
      projections exactly match the two strips.
    - Clear the two projection strips back to background and keep only the rebuilt rectangles in the interior canvas.

    Key insight:
    - Each one-dimensional view may miss one side of a rectangle because a nearer rectangle hides that part, so the
      hint pixel reveals the hidden extremity while the two strips together determine the only consistent overlap order.
    """
    g = np.asarray(grid, dtype=int)
    bg, top, left, rects = _parse_rectangles(g)
    matches = _matching_canvases(g[1:, 1:].shape, rects, top, left, bg)
    assert matches, "no rectangle layering matches the two projection strips"
    assert len(matches) == 1, "expected a unique reconstructed canvas"

    out = np.full_like(g, bg)
    out[1:, 1:] = matches[0]
    return out


def _apply_symmetry(arr: np.ndarray, flip_ud: bool, flip_lr: bool, transpose: bool) -> np.ndarray:
    out = arr.copy()
    if flip_ud:
        out = np.flipud(out).copy()
    if flip_lr:
        out = np.fliplr(out).copy()
    if transpose:
        out = out.T.copy()
    return out


def _hint_axis_candidates(lo: int, hi: int, vis: np.ndarray) -> list[int]:
    if int(vis[0]) > lo:
        return [int(lo)]
    if int(vis[-1]) < hi:
        return [int(hi)]
    if lo == hi:
        return [int(lo)]
    return [int(lo), int(hi)]


def _build_base_pair() -> list[tuple[int, int, int, int]]:
    front_r1 = random.randint(2, 5)
    front_c1 = random.randint(3, 6)
    front = (0, 0, front_r1, front_c1)

    back_r0 = 0
    back_c0 = random.randint(1, front_c1)
    back_r1 = front_r1 + random.randint(2, 4)
    back_c1 = front_c1 + random.randint(2, 4)
    back = (back_r0, back_c0, back_r1, back_c1)
    return [front, back]


def _shift_rects(rects: list[tuple[int, int, int, int]], dr: int, dc: int) -> list[tuple[int, int, int, int]]:
    return [(r0 + dr, c0 + dc, r1 + dr, c1 + dc) for r0, c0, r1, c1 in rects]


def _paint_rectangles(shape: tuple[int, int], rects: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    canvas = np.full(shape, BG, dtype=int)
    for (r0, c0, r1, c1), color in reversed(list(zip(rects, colors))):
        canvas[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return canvas


def _make_attempt() -> tuple[np.ndarray, np.ndarray]:
    colors = random.sample(PALETTE, 2)
    rects = _build_base_pair()

    last_r1 = rects[-1][2]
    last_c1 = rects[-1][3]
    top_margin = random.randint(0, 4)
    left_margin = random.randint(0, 4)
    max_bottom = 29 - (last_r1 + 1 + top_margin)
    max_right = 29 - (last_c1 + 1 + left_margin)
    assert max_bottom >= 0 and max_right >= 0
    bottom_margin = random.randint(0, min(5, max_bottom))
    right_margin = random.randint(0, min(5, max_right))

    rects = _shift_rects(rects, top_margin, left_margin)
    canvas_h = rects[-1][2] + 1 + bottom_margin
    canvas_w = rects[-1][3] + 1 + right_margin
    canvas = _paint_rectangles((canvas_h, canvas_w), rects, colors)
    layers = []
    for (r0, c0, r1, c1), color in zip(rects, colors):
        layer = np.full((canvas_h, canvas_w), BG, dtype=int)
        layer[r0 : r1 + 1, c0 : c1 + 1] = int(color)
        layers.append(layer)

    orient = random.choice(("base", "transpose"))
    flip_ud = False
    flip_lr = False
    transpose = orient == "transpose"
    canvas = _apply_symmetry(canvas, flip_ud, flip_lr, transpose)
    layers = [_apply_symmetry(layer, flip_ud, flip_lr, transpose) for layer in layers]
    top, left = _projection_strips(canvas, BG)

    hints = np.full_like(canvas, BG)
    used_hint_cells: set[tuple[int, int]] = set()
    for layer, color in zip(layers, colors):
        coords = np.argwhere(layer == color)
        r0 = int(coords[:, 0].min())
        r1 = int(coords[:, 0].max())
        c0 = int(coords[:, 1].min())
        c1 = int(coords[:, 1].max())
        row_vis = np.flatnonzero(left == color)
        col_vis = np.flatnonzero(top == color)
        row_options = _hint_axis_candidates(r0, r1, row_vis)
        col_options = _hint_axis_candidates(c0, c1, col_vis)
        candidates = [(rr, cc) for rr in row_options for cc in col_options]
        random.shuffle(candidates)
        hint_cell = next((cell for cell in candidates if cell not in used_hint_cells), None)
        assert hint_cell is not None, "hint pixels must be unique"
        used_hint_cells.add(hint_cell)
        hints[hint_cell] = int(color)

    grid = np.full((canvas.shape[0] + 1, canvas.shape[1] + 1), BG, dtype=int)
    grid[0, 1:] = top
    grid[1:, 0] = left
    grid[1:, 1:] = hints
    expected = np.full_like(grid, BG)
    expected[1:, 1:] = canvas
    return grid, expected


def _semantic_checks(inp: np.ndarray, out: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if inp.shape != out.shape:
        return False
    if not np.array_equal(out, expected):
        return False

    bg, top, left, rects = _parse_rectangles(inp)
    if len(_matching_canvases(inp[1:, 1:].shape, rects, top, left, bg)) != 1:
        return False

    hidden_both = 0
    overlaps = 0
    for i, rect in enumerate(rects):
        hr, hc = rect["hint"]
        row_vis = np.flatnonzero(left == rect["color"])
        col_vis = np.flatnonzero(top == rect["color"])
        row_hidden = hr < int(row_vis[0]) or hr > int(row_vis[-1])
        col_hidden = hc < int(col_vis[0]) or hc > int(col_vis[-1])
        if row_hidden and col_hidden:
            hidden_both += 1

        for other in rects[i + 1 :]:
            row_overlap = not (rect["r1"] < other["r0"] or other["r1"] < rect["r0"])
            col_overlap = not (rect["c1"] < other["c0"] or other["c1"] < rect["c0"])
            if row_overlap and col_overlap:
                overlaps += 1

    if hidden_both < 1:
        return False
    if overlaps < 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        try:
            inp, expected = _make_attempt()
            out = main(inp)
            if _semantic_checks(inp, out, expected):
                return inp
        except AssertionError:
            continue
    raise RuntimeError("failed to generate a valid dual-view rectangle reconstruction input")
