from __future__ import annotations

from dataclasses import dataclass
import random

import numpy as np


BG_COLOR = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

GRID_MIN = 18
GRID_MAX = 30
RECT_COUNT_RANGE = (3, 5)
RECT_SIDE_MIN = 5
RECT_SIDE_MAX = 14

ROOT_MARGIN = 2
OUTER_MARGIN = 1
GEN_ATTEMPTS = 400
STEP_ATTEMPTS = 120

ORIENTATIONS = ("SE", "SW", "NE", "NW")


@dataclass(frozen=True)
class Rect:
    color: int
    r0: int
    c0: int
    r1: int
    c1: int


def _bg_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _on_border(rect: Rect, r: int, c: int) -> bool:
    return rect.r0 <= r <= rect.r1 and rect.c0 <= c <= rect.c1 and (
        r == rect.r0 or r == rect.r1 or c == rect.c0 or c == rect.c1
    )


def _paint_border(grid: np.ndarray, rect: Rect) -> None:
    grid[rect.r0, rect.c0 : rect.c1 + 1] = rect.color
    grid[rect.r1, rect.c0 : rect.c1 + 1] = rect.color
    grid[rect.r0 : rect.r1 + 1, rect.c0] = rect.color
    grid[rect.r0 : rect.r1 + 1, rect.c1] = rect.color


def _render_input(shape: tuple[int, int], rects_front_to_back: list[Rect]) -> np.ndarray:
    grid = np.full(shape, BG_COLOR, dtype=int)
    for rect in reversed(rects_front_to_back):
        _paint_border(grid, rect)
    return grid


def _extract_rectangles(grid: np.ndarray, bg: int) -> list[Rect]:
    rects: list[Rect] = []
    for color in sorted(set(map(int, grid.flat)) - {int(bg)}):
        coords = np.argwhere(grid == color)
        assert coords.size > 0
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        rect = Rect(int(color), int(r0), int(c0), int(r1), int(c1))
        assert rect.r1 - rect.r0 >= 2 and rect.c1 - rect.c0 >= 2

        seen_top = seen_bottom = seen_left = seen_right = False
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            assert _on_border(rect, rr, cc), "visible pixels of a color must lie on one rectangle border"
            seen_top |= rr == rect.r0
            seen_bottom |= rr == rect.r1
            seen_left |= cc == rect.c0
            seen_right |= cc == rect.c1

        assert seen_top and seen_bottom and seen_left and seen_right, (
            "each rectangle needs at least one visible pixel on every side"
        )
        rects.append(rect)
    return rects


def _shared_border_cells(a: Rect, b: Rect) -> list[tuple[int, int]]:
    r0 = max(a.r0, b.r0)
    r1 = min(a.r1, b.r1)
    c0 = max(a.c0, b.c0)
    c1 = min(a.c1, b.c1)
    if r0 > r1 or c0 > c1:
        return []

    shared: list[tuple[int, int]] = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _on_border(a, r, c) and _on_border(b, r, c):
                shared.append((r, c))
    return shared


def _infer_order(grid: np.ndarray, rects: list[Rect]) -> list[Rect]:
    by_color = {rect.color: rect for rect in rects}
    covers: dict[int, set[int]] = {rect.color: set() for rect in rects}
    indegree: dict[int, int] = {rect.color: 0 for rect in rects}

    for i, a in enumerate(rects):
        for b in rects[i + 1 :]:
            shared = _shared_border_cells(a, b)
            if not shared:
                continue

            visible = {int(grid[r, c]) for r, c in shared}
            a_visible = a.color in visible
            b_visible = b.color in visible
            assert not (a_visible and b_visible), "shared border cells must respect one global layer order"

            if a_visible == b_visible:
                continue

            top, bottom = (a, b) if a_visible else (b, a)
            if bottom.color not in covers[top.color]:
                covers[top.color].add(bottom.color)
                indegree[bottom.color] += 1

    remaining = set(by_color)
    order: list[Rect] = []
    while remaining:
        ready = sorted(color for color in remaining if indegree[color] == 0)
        assert len(ready) == 1, "occlusion graph must determine a unique front-to-back order"
        color = ready[0]
        remaining.remove(color)
        order.append(by_color[color])
        for nxt in covers[color]:
            indegree[nxt] -= 1
    return order


def _paint_layer(out: np.ndarray, rect: Rect, depth: int, bg: int) -> None:
    if depth == 1:
        return

    for r in range(rect.r0 + 1, rect.r1):
        rr = r - (rect.r0 + 1)
        for c in range(rect.c0 + 1, rect.c1):
            if out[r, c] != bg:
                continue
            cc = c - (rect.c0 + 1)

            if depth == 2:
                fill = rr % 2 == 0
            elif depth == 3:
                fill = True
            else:
                fill = (rr + cc) % 2 == 0

            if fill:
                out[r, c] = rect.color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover overlapping hollow rectangles, order them by border occlusion, and encode depth as interior fill patterns.

    Input structure:
    - The background is black, and every non-background color belongs to one axis-aligned rectangular frame.
    - Each rectangle has a one-pixel border and a background interior; rectangles may overlap, so higher borders hide lower borders on shared border cells.
    - At least three differently colored rectangles appear, and their visible border overlaps determine a unique front-to-back order.

    Rule:
    - Reconstruct each rectangle from the extrema of its visible same-color border fragments.
    - Compare colors on cells where reconstructed borders coincide to infer which rectangle lies above which, then topologically sort the rectangles from front to back.
    - Keep the original visible borders unchanged, and decorate only background interior cells in that order: layer 1 stays empty, layer 2 paints horizontal stripes on every other interior row starting from the first, layer 3 paints solid, and layer 4 or deeper paints a checkerboard.

    Key insight:
    - The task combines two stages: first rebuild partially hidden frames and their occlusion order from border evidence, then map the recovered depth rank to a fill pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _bg_color(g)
    rects = _extract_rectangles(g, bg)
    order = _infer_order(g, rects)

    out = g.copy()
    for depth, rect in enumerate(order, start=1):
        _paint_layer(out, rect, depth, bg)
    return out


def _rand_interval(lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return random.randint(lo, hi)


def _sample_root(height: int, width: int, color: int) -> Rect | None:
    max_h = min(RECT_SIDE_MAX, height - 2 * ROOT_MARGIN)
    max_w = min(RECT_SIDE_MAX, width - 2 * ROOT_MARGIN)
    if max_h < RECT_SIDE_MIN or max_w < RECT_SIDE_MIN:
        return None

    rect_h = random.randint(RECT_SIDE_MIN, max_h)
    rect_w = random.randint(RECT_SIDE_MIN, max_w)
    r0 = random.randint(ROOT_MARGIN, height - rect_h - ROOT_MARGIN)
    c0 = random.randint(ROOT_MARGIN, width - rect_w - ROOT_MARGIN)
    return Rect(color, r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)


def _sample_next(prev: Rect, height: int, width: int, color: int) -> Rect | None:
    for orientation in random.sample(ORIENTATIONS, len(ORIENTATIONS)):
        for _ in range(STEP_ATTEMPTS):
            if orientation == "SE":
                nr0 = _rand_interval(prev.r0 + 1, prev.r1 - 2)
                nc0 = _rand_interval(prev.c0 + 1, prev.c1 - 2)
                if nr0 is None or nc0 is None:
                    continue
                nr1 = _rand_interval(max(prev.r1 + 1, nr0 + RECT_SIDE_MIN - 1), height - 1 - OUTER_MARGIN)
                nc1 = _rand_interval(max(prev.c1 + 1, nc0 + RECT_SIDE_MIN - 1), width - 1 - OUTER_MARGIN)
                if nr1 is None or nc1 is None:
                    continue
            elif orientation == "SW":
                nr0 = _rand_interval(prev.r0 + 1, prev.r1 - 2)
                nc1 = _rand_interval(prev.c0 + 2, prev.c1 - 1)
                if nr0 is None or nc1 is None:
                    continue
                nr1 = _rand_interval(max(prev.r1 + 1, nr0 + RECT_SIDE_MIN - 1), height - 1 - OUTER_MARGIN)
                nc0 = _rand_interval(OUTER_MARGIN, min(prev.c0 - 1, nc1 - RECT_SIDE_MIN + 1))
                if nr1 is None or nc0 is None:
                    continue
            elif orientation == "NE":
                nr1 = _rand_interval(prev.r0 + 2, prev.r1 - 1)
                nc0 = _rand_interval(prev.c0 + 1, prev.c1 - 2)
                if nr1 is None or nc0 is None:
                    continue
                nr0 = _rand_interval(OUTER_MARGIN, min(prev.r0 - 1, nr1 - RECT_SIDE_MIN + 1))
                nc1 = _rand_interval(max(prev.c1 + 1, nc0 + RECT_SIDE_MIN - 1), width - 1 - OUTER_MARGIN)
                if nr0 is None or nc1 is None:
                    continue
            else:  # NW
                nr1 = _rand_interval(prev.r0 + 2, prev.r1 - 1)
                nc1 = _rand_interval(prev.c0 + 2, prev.c1 - 1)
                if nr1 is None or nc1 is None:
                    continue
                nr0 = _rand_interval(OUTER_MARGIN, min(prev.r0 - 1, nr1 - RECT_SIDE_MIN + 1))
                nc0 = _rand_interval(OUTER_MARGIN, min(prev.c0 - 1, nc1 - RECT_SIDE_MIN + 1))
                if nr0 is None or nc0 is None:
                    continue

            rect = Rect(color, int(nr0), int(nc0), int(nr1), int(nc1))
            if rect.r1 - rect.r0 + 1 < RECT_SIDE_MIN or rect.c1 - rect.c0 + 1 < RECT_SIDE_MIN:
                continue
            if rect == prev:
                continue
            return rect
    return None


def _all_sides_visible(grid: np.ndarray, rect: Rect) -> bool:
    top = np.any(grid[rect.r0, rect.c0 : rect.c1 + 1] == rect.color)
    bottom = np.any(grid[rect.r1, rect.c0 : rect.c1 + 1] == rect.color)
    left = np.any(grid[rect.r0 : rect.r1 + 1, rect.c0] == rect.color)
    right = np.any(grid[rect.r0 : rect.r1 + 1, rect.c1] == rect.color)
    return bool(top and bottom and left and right)


def _visible_fill_exists(out: np.ndarray, rect: Rect) -> bool:
    if rect.r1 - rect.r0 <= 1 or rect.c1 - rect.c0 <= 1:
        return False
    interior = out[rect.r0 + 1 : rect.r1, rect.c0 + 1 : rect.c1]
    return bool(np.any(interior == rect.color))


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        height = random.randint(GRID_MIN, GRID_MAX)
        width = random.randint(GRID_MIN, GRID_MAX)
        count = random.randint(*RECT_COUNT_RANGE)
        colors = random.sample(COLOR_POOL, count)

        rects: list[Rect] = []
        root = _sample_root(height, width, colors[0])
        if root is None:
            continue
        rects.append(root)

        ok = True
        for color in colors[1:]:
            nxt = _sample_next(rects[-1], height, width, color)
            if nxt is None:
                ok = False
                break
            rects.append(nxt)
        if not ok:
            continue

        inp = _render_input((height, width), rects)

        if not all(_all_sides_visible(inp, rect) for rect in rects):
            continue

        try:
            recovered = _extract_rectangles(inp, BG_COLOR)
            order = _infer_order(inp, recovered)
            out = main(inp)
        except AssertionError:
            continue

        if [rect.color for rect in order] != [rect.color for rect in rects]:
            continue

        visible_depths = 0
        for depth, rect in enumerate(rects, start=1):
            if depth >= 2 and _visible_fill_exists(out, rect):
                visible_depths += 1
        if visible_depths < min(len(rects) - 1, 2):
            continue

        return inp

    raise RuntimeError("failed to generate a valid overlapping-rectangles puzzle")
