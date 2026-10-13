from __future__ import annotations

import random

import numpy as np


Rect = tuple[int, int, int, int]
PlacedRect = tuple[int, int, Rect, int]


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    winners = values[counts == counts.max()]
    assert winners.size == 1, "background color must be uniquely dominant"
    return int(winners[0])


def _bbox(mask: np.ndarray) -> Rect | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rect_area(rect: Rect) -> int:
    r0, c0, r1, c1 = rect
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    assert w % 2 == 1, "equal panels with a one-cell separator require odd width"
    sep = w // 2
    sep_vals = np.unique(grid[:, sep])
    assert sep_vals.size == 1, "separator must be a uniform full-height column"
    sep_color = int(sep_vals[0])
    assert sep_color != bg, "separator color must differ from background"
    return sep, sep_color


def _extract_left_rectangles(left: np.ndarray, bg: int) -> list[PlacedRect]:
    rectangles: list[PlacedRect] = []
    for color0 in np.unique(left):
        color = int(color0)
        if color == bg:
            continue
        mask = left == color
        rect = _bbox(mask)
        assert rect is not None
        area = _rect_area(rect)
        assert int(mask.sum()) == area, "every foreground color must form one solid rectangle"
        r0, c0, r1, c1 = rect
        assert np.all(left[r0 : r1 + 1, c0 : c1 + 1] == color), "rectangle interior must be solid"
        rectangles.append((color, area, rect, area))
    assert rectangles, "left panel must contain at least one rectangle"
    areas = [area for _color, area, _rect, _area_copy in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must be unique for overlap tie-breaks"
    return rectangles


def _paint_right_panel(panel_shape: tuple[int, int], rectangles: list[PlacedRect], bg: int) -> np.ndarray:
    h, w = panel_shape
    best_area = np.full((h, w), np.iinfo(np.int32).max, dtype=int)
    out = np.full((h, w), bg, dtype=int)

    for color, area, rect, _area_copy in rectangles:
        r0, c0, r1, c1 = rect
        mc0 = w - 1 - c1
        mc1 = w - 1 - c0
        current = np.zeros((h, w), dtype=bool)
        current[r0 : r1 + 1, mc0 : mc1 + 1] = True

        ambiguous = current & (best_area == area) & (out != bg) & (out != color)
        assert not bool(ambiguous.any()), "equal-area overlaps are ambiguous"

        update = current & (area < best_area)
        out[update] = color
        best_area[update] = area

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the left-panel rectangles into the right panel by matching rows and reversed columns.

    Input structure:
    - The background is the uniquely dominant color.
    - A single full-height non-background column splits the grid into two equal-width panels.
    - The left panel contains several non-overlapping solid rectangles in distinct colors, while the right panel is entirely background.

    Rule:
    - Keep the left panel and separator unchanged.
    - For each left-panel rectangle, copy its occupied row interval directly to the right panel.
    - Map the rectangle's column interval to mirrored panel columns, so the leftmost column of the left panel corresponds to the rightmost column of the right panel.
    - Fill the right-panel intersection of that row interval and mirrored column interval with the rectangle's color; if projected regions overlap, the smaller-area rectangle wins.

    Key insight:
    - Each rectangle encodes two independent selectors: its rows transfer directly, but its columns reappear only after a left-right mirror inside the opposite panel.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    sep, sep_color = _find_separator(g, bg)
    left = g[:, :sep]
    right = g[:, sep + 1 :]
    assert left.shape == right.shape, "the separator must split two equal panels"
    assert not bool(np.any(left == sep_color)), "separator color may only appear on the separator column"
    assert bool(np.all(right == bg)), "right panel must start as pure background"

    rectangles = _extract_left_rectangles(left, bg)

    out = g.copy()
    out[:, sep + 1 :] = _paint_right_panel(right.shape, rectangles, bg)
    return out


# --------------- generator (custom) ---------------

GEN_PANEL_HEIGHT_RANGE = (7, 16)
GEN_PANEL_WIDTH_RANGE = (6, 14)
GEN_RECT_COUNT_RANGE = (2, 4)
GEN_MAX_RECT_SIDE = 5
GEN_MIN_RECT_AREA = 2
GEN_PLACEMENT_ATTEMPTS = 120
GEN_RETRIES = 400
GEN_MODES = ("rows", "cols", "scatter")
GEN_PALETTE = tuple(range(10))


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return not (a1 < b0 or b1 < a0)


def _pairwise_row_overlap(rects: list[Rect]) -> bool:
    for i, (r0, _c0, r1, _c1) in enumerate(rects):
        for rr0, _cc0, rr1, _cc1 in rects[i + 1 :]:
            if _intervals_overlap(r0, r1, rr0, rr1):
                return True
    return False


def _pairwise_col_overlap(rects: list[Rect]) -> bool:
    for i, (_r0, c0, _r1, c1) in enumerate(rects):
        for _rr0, cc0, _rr1, cc1 in rects[i + 1 :]:
            if _intervals_overlap(c0, c1, cc0, cc1):
                return True
    return False


def _valid_layout_mode(rects: list[Rect], mode: str) -> bool:
    if mode == "rows":
        return _pairwise_col_overlap(rects)
    if mode == "cols":
        return _pairwise_row_overlap(rects)
    return _pairwise_row_overlap(rects) and _pairwise_col_overlap(rects)


def _size_options(panel_h: int, panel_w: int, mode: str, count: int) -> list[tuple[int, int]]:
    max_h = min(GEN_MAX_RECT_SIDE, panel_h - 1)
    max_w = min(GEN_MAX_RECT_SIDE, panel_w - 1)
    options: list[tuple[int, int]] = []
    for h in range(1, max_h + 1):
        for w in range(1, max_w + 1):
            if h * w < GEN_MIN_RECT_AREA:
                continue
            if mode == "rows" and h > max(1, panel_h // count):
                continue
            if mode == "cols" and w > max(1, panel_w // count):
                continue
            options.append((h, w))
    random.shuffle(options)
    return options


def _rect_conflicts(rect: Rect, existing: list[Rect], mode: str) -> bool:
    r0, c0, r1, c1 = rect
    for er0, ec0, er1, ec1 in existing:
        if not (r1 < er0 - 1 or er1 < r0 - 1 or c1 < ec0 - 1 or ec1 < c0 - 1):
            return True
        if mode == "rows" and _intervals_overlap(r0, r1, er0, er1):
            return True
        if mode == "cols" and _intervals_overlap(c0, c1, ec0, ec1):
            return True
    return False


def _sample_rectangles(panel_h: int, panel_w: int, count: int, mode: str) -> list[Rect] | None:
    options = _size_options(panel_h, panel_w, mode, count)
    if not options:
        return None

    rects: list[Rect] = []
    used_areas: set[int] = set()

    for _idx in range(count):
        placed = False
        for _attempt in range(GEN_PLACEMENT_ATTEMPTS):
            h, w = random.choice(options)
            area = h * w
            if area in used_areas:
                continue
            r0 = random.randint(0, panel_h - h)
            c0 = random.randint(0, panel_w - w)
            rect = (r0, c0, r0 + h - 1, c0 + w - 1)
            if _rect_conflicts(rect, rects, mode):
                continue
            rects.append(rect)
            used_areas.add(area)
            placed = True
            break
        if not placed:
            return None

    if not _valid_layout_mode(rects, mode):
        return None
    return rects


def _render_input(panel_h: int, panel_w: int, rectangles: list[Rect]) -> np.ndarray:
    colors = random.sample(list(GEN_PALETTE), len(rectangles) + 2)
    bg = colors[0]
    sep_color = colors[1]
    rect_colors = colors[2:]

    grid = np.full((panel_h, 2 * panel_w + 1), bg, dtype=int)
    grid[:, panel_w] = sep_color

    for color, rect in zip(rect_colors, rectangles):
        r0, c0, r1, c1 = rect
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    return grid


def _gen_attempt() -> np.ndarray | None:
    panel_h = random.randint(*GEN_PANEL_HEIGHT_RANGE)
    panel_w = random.randint(*GEN_PANEL_WIDTH_RANGE)
    count_hi = min(GEN_RECT_COUNT_RANGE[1], max(2, min(panel_h, panel_w) // 2))
    count = random.randint(GEN_RECT_COUNT_RANGE[0], count_hi)
    mode = random.choice(GEN_MODES)

    rectangles = _sample_rectangles(panel_h, panel_w, count, mode)
    if rectangles is None:
        return None

    grid = _render_input(panel_h, panel_w, rectangles)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
