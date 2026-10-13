from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
RULER_COLORS = (2, 1, 3, 4, 7, 6)  # red, blue, green, yellow, orange, magenta

# --------------- solver ---------------


def _grey_rectangles(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    rects: list[tuple[int, int, int, int]] = []

    for r in range(1, h):
        for c in range(1, w):
            if seen[r, c] or int(grid[r, c]) != GREY:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 1 <= nr < h and 1 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == GREY:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            assert len(coords) == area, "grey object must be a solid rectangle"
            assert bool(np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == GREY))
            rects.append((r0, c0, r1, c1))

    rects.sort()
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the top-row and left-column rulers to recolor each grey rectangle with stripes.

    - Read each rectangle's row and column span.
    - If width >= height, repaint by columns using the top ruler colors over that span.
    - Otherwise repaint by rows using the left ruler colors over that span.
    - Keep the rulers unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3
    assert int(g[0, 0]) == BLACK

    out = g.copy()
    top_ruler = g[0]
    left_ruler = g[:, 0]

    for r0, c0, r1, c1 in _grey_rectangles(g):
        rect_h = r1 - r0 + 1
        rect_w = c1 - c0 + 1
        if rect_w >= rect_h:
            colors = top_ruler[c0 : c1 + 1]
            for j, color in enumerate(colors):
                out[r0 : r1 + 1, c0 + j] = int(color)
        else:
            colors = left_ruler[r0 : r1 + 1]
            for i, color in enumerate(colors):
                out[r0 + i, c0 : c1 + 1] = int(color)

    return out


# --------------- generator ---------------

GRID_HEIGHT_RANGE = (9, 17)
GRID_WIDTH_RANGE = (9, 17)
OBJECT_COUNT_RANGE = (2, 5)
GENERATION_RETRIES = 400
PLACEMENT_RETRIES = 120
RULER_RETRIES = 80


def _sample_ruler(length: int) -> np.ndarray:
    for _ in range(RULER_RETRIES):
        arr = np.array([random.choice(RULER_COLORS) for _ in range(length)], dtype=int)
        if len(set(map(int, arr))) >= 3:
            return arr
    raise RuntimeError("failed to sample a diverse ruler")


def _sample_orientations(count: int) -> list[str]:
    wide_count = count // 2
    tall_count = count - wide_count
    if count % 2 == 1 and random.random() < 0.5:
        wide_count, tall_count = tall_count, wide_count
    kinds = ["wide"] * wide_count + ["tall"] * tall_count
    random.shuffle(kinds)
    return kinds


def _sample_size(kind: str, inner_h: int, inner_w: int) -> tuple[int, int] | None:
    if kind == "wide":
        max_h = min(4, inner_h)
        if max_h < 1:
            return None
        rect_h = random.randint(1, max_h)
        min_w = rect_h + 1
        max_w = min(7, inner_w)
        if min_w > max_w:
            return None
        rect_w = random.randint(min_w, max_w)
        return rect_h, rect_w

    max_w = min(4, inner_w)
    if max_w < 1:
        return None
    rect_w = random.randint(1, max_w)
    min_h = rect_w + 1
    max_h = min(7, inner_h)
    if min_h > max_h:
        return None
    rect_h = random.randint(min_h, max_h)
    return rect_h, rect_w


def _place_rectangles(
    inner_h: int,
    inner_w: int,
    kinds: list[str],
) -> list[tuple[int, int, int, int, str]] | None:
    blocked = np.zeros((inner_h, inner_w), dtype=bool)
    rects: list[tuple[int, int, int, int, str]] = []

    for kind in kinds:
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            size = _sample_size(kind, inner_h, inner_w)
            if size is None:
                return None
            rect_h, rect_w = size

            max_r = inner_h - rect_h
            max_c = inner_w - rect_w
            if max_r < 0 or max_c < 0:
                continue

            top = random.randint(0, max_r)
            left = random.randint(0, max_c)
            r0 = max(0, top - 1)
            c0 = max(0, left - 1)
            r1 = min(inner_h, top + rect_h + 1)
            c1 = min(inner_w, left + rect_w + 1)
            if bool(blocked[r0:r1, c0:c1].any()):
                continue

            blocked[r0:r1, c0:c1] = True
            rects.append((top + 1, left + 1, top + rect_h, left + rect_w, kind))
            placed = True
            break
        if not placed:
            return None

    rects.sort()
    return rects


def _segments_are_visible(
    top_ruler: np.ndarray,
    left_ruler: np.ndarray,
    rects: list[tuple[int, int, int, int, str]],
) -> bool:
    for r0, c0, r1, c1, kind in rects:
        horiz = top_ruler[c0 - 1 : c1]
        vert = left_ruler[r0 - 1 : r1]
        if kind == "wide":
            if len(set(map(int, horiz))) < 2:
                return False
        else:
            if len(set(map(int, vert))) < 2:
                return False

    return True


def _expected_output(
    grid: np.ndarray,
    rects: list[tuple[int, int, int, int, str]],
) -> np.ndarray:
    out = grid.copy()
    top_ruler = grid[0]
    left_ruler = grid[:, 0]

    for r0, c0, r1, c1, kind in rects:
        if kind == "wide":
            colors = top_ruler[c0 : c1 + 1]
            for j, color in enumerate(colors):
                out[r0 : r1 + 1, c0 + j] = int(color)
        else:
            colors = left_ruler[r0 : r1 + 1]
            for i, color in enumerate(colors):
                out[r0 + i, c0 : c1 + 1] = int(color)

    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    inner_h = height - 1
    inner_w = width - 1

    object_count = random.randint(*OBJECT_COUNT_RANGE)
    kinds = _sample_orientations(object_count)
    rects = _place_rectangles(inner_h, inner_w, kinds)
    if rects is None:
        return None

    top_ruler = _sample_ruler(inner_w)
    left_ruler = _sample_ruler(inner_h)
    if not _segments_are_visible(top_ruler, left_ruler, rects):
        return None

    grid = np.full((height, width), BLACK, dtype=int)
    grid[0, 1:] = top_ruler
    grid[1:, 0] = left_ruler
    for r0, c0, r1, c1, _kind in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY

    expected = _expected_output(grid, rects)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
