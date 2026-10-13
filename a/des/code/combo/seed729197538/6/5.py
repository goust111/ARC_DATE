import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Rect:
    top: int
    left: int
    bottom: int
    right: int
    border: int
    fill: int


BACKGROUND = 0
NON_BG_COLORS = tuple(range(1, 10))
GENERATION_RETRIES = 500
GEOMETRY_RETRIES = 120


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _edge_run(grid: np.ndarray, bg: int, start: tuple[int, int], step: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    r, c = start
    dr, dc = step
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) != bg:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _line_is_isolated(grid: np.ndarray, bg: int, cells: list[tuple[int, int]]) -> bool:
    if not cells:
        return False
    h, w = grid.shape
    if len(cells) == 1:
        return False
    rows = {r for r, _ in cells}
    cols = {c for _, c in cells}
    if len(rows) == 1:
        row = next(iter(rows))
        inward = 1 if row == 0 else -1
        for r, c in cells:
            rr = r + inward
            if 0 <= rr < h and int(grid[rr, c]) != bg:
                return False
    elif len(cols) == 1:
        col = next(iter(cols))
        inward = 1 if col == 0 else -1
        for r, c in cells:
            cc = c + inward
            if 0 <= cc < w and int(grid[r, cc]) != bg:
                return False
    else:
        return False
    return True


def _detect_legend(grid: np.ndarray, bg: int) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    specs = [
        ((0, 0), (0, 1), (1, 0)),
        ((0, w - 1), (0, -1), (1, 0)),
        ((h - 1, 0), (0, 1), (-1, 0)),
        ((h - 1, w - 1), (0, -1), (-1, 0)),
    ]
    for corner, horizontal_step, vertical_step in specs:
        if int(grid[corner]) == bg:
            continue
        horiz = _edge_run(grid, bg, corner, horizontal_step)
        vert = _edge_run(grid, bg, corner, vertical_step)
        candidates: list[list[tuple[int, int]]] = []
        if _line_is_isolated(grid, bg, horiz):
            candidates.append(horiz)
        if _line_is_isolated(grid, bg, vert):
            candidates.append(vert)
        if not candidates:
            continue
        line = max(candidates, key=len)
        colors = [int(grid[r, c]) for r, c in line]
        if len(set(colors)) != len(colors):
            continue
        mask = np.zeros_like(grid, dtype=bool)
        for r, c in line:
            mask[r, c] = True
        return colors, mask
    raise AssertionError("could not locate corner legend")


def _inside_color_votes(
    grid: np.ndarray,
    rect: Rect,
    legend_mask: np.ndarray,
    bg: int,
) -> Counter:
    votes: Counter = Counter()
    for r in range(rect.top, rect.bottom + 1):
        for c in range(rect.left, rect.right + 1):
            if legend_mask[r, c] or int(grid[r, c]) != rect.border:
                continue
            if r == rect.top and rect.top + 1 <= rect.bottom - 1:
                val = int(grid[r + 1, c])
                if val != bg and val != rect.border and not legend_mask[r + 1, c]:
                    votes[val] += 1
            if r == rect.bottom and rect.top + 1 <= rect.bottom - 1:
                val = int(grid[r - 1, c])
                if val != bg and val != rect.border and not legend_mask[r - 1, c]:
                    votes[val] += 1
            if c == rect.left and rect.left + 1 <= rect.right - 1:
                val = int(grid[r, c + 1])
                if val != bg and val != rect.border and not legend_mask[r, c + 1]:
                    votes[val] += 1
            if c == rect.right and rect.left + 1 <= rect.right - 1:
                val = int(grid[r, c - 1])
                if val != bg and val != rect.border and not legend_mask[r, c - 1]:
                    votes[val] += 1
    return votes


def _infer_fill_color(grid: np.ndarray, rect: Rect, legend_mask: np.ndarray, bg: int) -> int:
    votes = _inside_color_votes(grid, rect, legend_mask, bg)
    if votes:
        return int(votes.most_common(1)[0][0])

    interior = grid[rect.top + 1 : rect.bottom, rect.left + 1 : rect.right]
    interior_mask = legend_mask[rect.top + 1 : rect.bottom, rect.left + 1 : rect.right]
    fallback: Counter = Counter()
    for val, keep in zip(interior.ravel(), (~interior_mask).ravel()):
        if not keep:
            continue
        color = int(val)
        if color != bg and color != rect.border:
            fallback[color] += 1
    if fallback:
        return int(fallback.most_common(1)[0][0])
    raise AssertionError("could not infer rectangle fill color")


def _extract_rectangles(grid: np.ndarray) -> tuple[list[Rect], list[int], np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    legend_colors, legend_mask = _detect_legend(g, bg)

    rects: list[Rect] = []
    for border in legend_colors:
        coords = np.argwhere((g == border) & ~legend_mask)
        if coords.size == 0:
            raise AssertionError("every legend color must appear in a visible rectangle border")
        top = int(coords[:, 0].min())
        left = int(coords[:, 1].min())
        bottom = int(coords[:, 0].max())
        right = int(coords[:, 1].max())
        if bottom - top < 2 or right - left < 2:
            raise AssertionError("reconstructed rectangles must have an interior")
        rect = Rect(top=top, left=left, bottom=bottom, right=right, border=int(border), fill=-1)
        fill = _infer_fill_color(g, rect, legend_mask, bg)
        rects.append(Rect(top=top, left=left, bottom=bottom, right=right, border=int(border), fill=fill))
    return rects, legend_colors, legend_mask, bg


def _draw_rect(grid: np.ndarray, rect: Rect) -> None:
    grid[rect.top : rect.bottom + 1, rect.left : rect.right + 1] = rect.fill
    grid[rect.top, rect.left : rect.right + 1] = rect.border
    grid[rect.bottom, rect.left : rect.right + 1] = rect.border
    grid[rect.top : rect.bottom + 1, rect.left] = rect.border
    grid[rect.top : rect.bottom + 1, rect.right] = rect.border


def _legend_positions(h: int, w: int, n: int, corner: str, horizontal: bool) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, i) for i in range(n)] if horizontal else [(i, 0) for i in range(n)]
    if corner == "tr":
        return [(0, w - 1 - i) for i in range(n)] if horizontal else [(i, w - 1) for i in range(n)]
    if corner == "bl":
        return [(h - 1, i) for i in range(n)] if horizontal else [(h - 1 - i, 0) for i in range(n)]
    if corner == "br":
        return [(h - 1, w - 1 - i) for i in range(n)] if horizontal else [(h - 1 - i, w - 1) for i in range(n)]
    raise ValueError(f"unknown corner: {corner}")


def _render_scene(
    h: int,
    w: int,
    rects: list[Rect],
    draw_order: list[int],
    legend_colors: list[int],
    corner: str,
    horizontal: bool,
) -> np.ndarray:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    for border in draw_order:
        rect = next(rect for rect in rects if rect.border == border)
        _draw_rect(grid, rect)
    for (r, c), color in zip(_legend_positions(h, w, len(legend_colors), corner, horizontal), legend_colors):
        grid[r, c] = color
    return grid


def _overlap_area(a: Rect, b: Rect) -> int:
    top = max(a.top, b.top)
    left = max(a.left, b.left)
    bottom = min(a.bottom, b.bottom)
    right = min(a.right, b.right)
    if top > bottom or left > right:
        return 0
    return (bottom - top + 1) * (right - left + 1)


def _visible_counts(grid: np.ndarray, rect: Rect) -> dict[str, int]:
    counts = {
        "top": int(np.sum(grid[rect.top, rect.left : rect.right + 1] == rect.border)),
        "bottom": int(np.sum(grid[rect.bottom, rect.left : rect.right + 1] == rect.border)),
        "left": int(np.sum(grid[rect.top : rect.bottom + 1, rect.left] == rect.border)),
        "right": int(np.sum(grid[rect.top : rect.bottom + 1, rect.right] == rect.border)),
        "fill": int(np.sum(grid[rect.top + 1 : rect.bottom, rect.left + 1 : rect.right] == rect.fill)),
    }
    return counts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the corner legend as border-color priority and redraw the rectangles in that order.
    """
    rects, legend_colors, legend_mask, bg = _extract_rectangles(np.asarray(grid, dtype=int))
    out = np.full_like(grid, bg)
    out[legend_mask] = np.asarray(grid, dtype=int)[legend_mask]
    for rect in rects:
        _draw_rect(out, rect)
    return out


def _sample_rectangles(rng: random.Random, h: int, w: int, n: int) -> list[Rect] | None:
    usable_top = 2
    usable_left = 2
    usable_bottom = h - 3
    usable_right = w - 3
    if usable_bottom - usable_top + 1 < 8 or usable_right - usable_left + 1 < 8:
        return None

    max_core_h = min(4, usable_bottom - usable_top - 1)
    max_core_w = min(4, usable_right - usable_left - 1)
    if max_core_h < 2 or max_core_w < 2:
        return None

    core_h = rng.randint(2, max_core_h)
    core_w = rng.randint(2, max_core_w)
    top_lo = usable_top + 1
    top_hi = usable_bottom - core_h - 1
    left_lo = usable_left + 1
    left_hi = usable_right - core_w - 1
    if top_lo > top_hi or left_lo > left_hi:
        return None
    core_top = rng.randint(top_lo, top_hi)
    core_left = rng.randint(left_lo, left_hi)
    core_bottom = core_top + core_h - 1
    core_right = core_left + core_w - 1

    rects: list[tuple[int, int, int, int]] = []
    for _ in range(n):
        for _attempt in range(30):
            top = core_top - rng.randint(1, min(5, core_top - usable_top))
            left = core_left - rng.randint(1, min(5, core_left - usable_left))
            bottom = core_bottom + rng.randint(1, min(5, usable_bottom - core_bottom))
            right = core_right + rng.randint(1, min(5, usable_right - core_right))
            if bottom - top + 1 < 5 or right - left + 1 < 5:
                continue
            bbox = (top, left, bottom, right)
            if bbox in rects:
                continue
            rects.append(bbox)
            break
        else:
            return None

    rect_objs: list[Rect] = []
    border_colors = rng.sample(list(NON_BG_COLORS), n)
    fill_pool = [c for c in NON_BG_COLORS if c not in border_colors]
    fills = [rng.choice(fill_pool) for _ in range(n)]
    for bbox, border, fill in zip(rects, border_colors, fills):
        rect_objs.append(Rect(*bbox, border=border, fill=fill))
    return rect_objs


def _rects_match_by_border(expected: list[Rect], inferred: list[Rect]) -> bool:
    exp_by_border = {rect.border: rect for rect in expected}
    inf_by_border = {rect.border: rect for rect in inferred}
    if exp_by_border.keys() != inf_by_border.keys():
        return False
    for border, exp in exp_by_border.items():
        if inf_by_border[border] != exp:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.choice([2, 2, 3, 3, 4])
    h = rng.randint(max(12, n + 8), 22)
    w = rng.randint(max(12, n + 8), 22)
    corner = rng.choice(["tl", "tr", "bl", "br"])
    horizontal = bool(rng.randint(0, 1))

    for _ in range(GEOMETRY_RETRIES):
        rects = _sample_rectangles(rng, h, w, n)
        if rects is None:
            continue
        if max(_overlap_area(a, b) for i, a in enumerate(rects) for b in rects[i + 1 :]) <= 0:
            continue

        legend_colors = [rect.border for rect in rects]
        rng.shuffle(legend_colors)
        draw_order = legend_colors[:]
        while draw_order == legend_colors:
            rng.shuffle(draw_order)

        inp = _render_scene(h, w, rects, draw_order, legend_colors, corner, horizontal)
        expected = _render_scene(h, w, rects, legend_colors, legend_colors, corner, horizontal)
        if np.array_equal(inp, expected):
            continue
        if int(np.count_nonzero(inp != expected)) < 6:
            continue

        if any(min(_visible_counts(inp, rect).values()) < 1 for rect in rects):
            continue
        if any(_visible_counts(inp, rect)["fill"] < 2 for rect in rects):
            continue

        try:
            inferred, inferred_legend, _legend_mask, _bg = _extract_rectangles(inp)
        except AssertionError:
            continue
        if inferred_legend != legend_colors:
            continue
        if not _rects_match_by_border(rects, inferred):
            continue
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
