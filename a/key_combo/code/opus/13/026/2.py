from __future__ import annotations

import random

import numpy as np


BG = 0
SEP = 7
RECT_COLORS = (1, 2, 3)
GEN_ATTEMPTS = 400


def _find_separator_row(grid: np.ndarray) -> tuple[int, int]:
    rows: list[tuple[int, int]] = []
    for r in range(grid.shape[0]):
        vals = np.unique(grid[r])
        if vals.size == 1 and int(vals[0]) != BG:
            rows.append((r, int(vals[0])))
    assert len(rows) == 1, "expected exactly one full-width non-background separator row"
    return rows[0]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _checkerboard_carve(
    out: np.ndarray,
    rect: tuple[int, int, int, int],
    color: int,
    parity_to_black: int,
) -> None:
    r0, c0, r1, c1 = rect
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return
    interior = out[r0 + 1 : r1, c0 + 1 : c1]
    rows, cols = np.indices(interior.shape)
    interior[(rows + cols) % 2 == parity_to_black] = BG
    interior[(rows + cols) % 2 != parity_to_black] = color


def _find_source_rectangles(grid: np.ndarray, sep_row: int, sep_color: int) -> list[tuple[int, tuple[int, int, int, int]]]:
    mask = (grid != BG) & (grid != sep_color)
    comps = _components(mask)
    assert comps, "expected at least one source rectangle"

    rects: list[tuple[int, tuple[int, int, int, int]]] = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 < sep_row, "all source rectangles must lie above the separator"
        color = int(grid[r0, c0])
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == color), "each source object must be a solid monochrome rectangle"
        rects.append((color, (r0, c0, r1, c1)))

    rects.sort(key=lambda item: (item[1][1], item[1][0]))
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the upper solid rectangles across the horizontal separator, fill each gap with the separator color, and hollow originals and mirrors with opposite checkerboard phases.

    Input structure:
    - A black background contains exactly one full-width horizontal separator row in a non-black color.
    - Two or more solid monochrome rectangles sit strictly above that separator, with at least one blank row between each rectangle and the separator.
    - The area below the separator is otherwise background in the intended task instances.

    Rule:
    - Reflect every upper rectangle vertically across the separator row, keeping its color and column span.
    - Fill the entire vertical strip between each original rectangle and its reflection, over the same columns, with the separator color.
    - Preserve each rectangle's 1-cell border, then carve its interior into a checkerboard: originals turn even-parity interior cells black, while mirrored copies turn odd-parity interior cells black.
    - Leave all other background and untouched separator cells unchanged.

    Key insight:
    - The lower rectangles are ordinary reflections, but their interior checkerboard phase is the opposite of the corresponding upper originals, so object identity matters in addition to geometry.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_row, sep_color = _find_separator_row(g)
    rects = _find_source_rectangles(g, sep_row, sep_color)

    out = g.copy()
    for color, (r0, c0, r1, c1) in rects:
        mr0 = 2 * sep_row - r1
        mr1 = 2 * sep_row - r0
        assert sep_row < mr0 <= mr1 < g.shape[0], "mirror rectangle must fit below the separator"

        out[r1 + 1 : mr0, c0 : c1 + 1] = sep_color
        out[mr0 : mr1 + 1, c0 : c1 + 1] = color

        _checkerboard_carve(out, (r0, c0, r1, c1), color, parity_to_black=0)
        _checkerboard_carve(out, (mr0, c0, mr1, c1), color, parity_to_black=1)

    return out


def _rand_composition(total: int, parts: int) -> list[int]:
    if total <= 0:
        return [0] * parts
    return list(np.random.multinomial(int(total), [1.0 / parts] * parts))


def _pack_intervals(total_width: int, widths: list[int], min_gap: int) -> list[tuple[int, int]] | None:
    extra = total_width - sum(widths) - min_gap * (len(widths) - 1)
    if extra < 0:
        return None
    gaps = _rand_composition(extra, len(widths) + 1)
    intervals: list[tuple[int, int]] = []
    c = gaps[0]
    for i, width in enumerate(widths):
        intervals.append((c, c + width - 1))
        c += width
        if i < len(widths) - 1:
            c += min_gap + gaps[i + 1]
    return intervals


def _sample_rectangles(height: int, width: int, sep_row: int, count: int) -> list[tuple[int, int, int, int]] | None:
    lower_bound_top = max(0, 2 * sep_row - (height - 1))
    rect_heights: list[int] = []
    rect_tops: list[int] = []
    rect_widths: list[int] = []

    for _ in range(count):
        candidates: list[tuple[int, int]] = []
        for rect_h in range(4, min(7, sep_row) + 1):
            max_top = sep_row - rect_h - 1
            if lower_bound_top <= max_top:
                for top in range(lower_bound_top, max_top + 1):
                    candidates.append((rect_h, top))
        if not candidates:
            return None
        rect_h, top = random.choice(candidates)
        rect_heights.append(rect_h)
        rect_tops.append(top)
        rect_widths.append(random.randint(4, 8))

    if len(set(zip(rect_heights, rect_widths))) == 1:
        idx = random.randrange(count)
        rect_widths[idx] = min(8, rect_widths[idx] + 1) if rect_widths[idx] < 8 else max(4, rect_widths[idx] - 1)

    intervals = _pack_intervals(width, rect_widths, min_gap=2)
    if intervals is None:
        return None

    rects = []
    for top, rect_h, (c0, c1) in zip(rect_tops, rect_heights, intervals):
        rects.append((top, c0, top + rect_h - 1, c1))
    return rects


def _valid_input(grid: np.ndarray) -> bool:
    try:
        sep_row, sep_color = _find_separator_row(grid)
        rects = _find_source_rectangles(grid, sep_row, sep_color)
        if len(rects) not in {2, 3}:
            return False
        if any((r1 - r0 + 1) < 4 or (c1 - c0 + 1) < 4 for _, (r0, c0, r1, c1) in rects):
            return False
        out = main(grid)
    except AssertionError:
        return False

    for color, (r0, c0, r1, c1) in rects:
        mr0 = 2 * sep_row - r1
        mr1 = 2 * sep_row - r0
        mirror = out[mr0 : mr1 + 1, c0 : c1 + 1]
        if np.count_nonzero(mirror == color) == mirror.size:
            return False
        corridor = out[r1 + 1 : mr0, c0 : c1 + 1]
        if corridor.size == 0 or not np.all(corridor == sep_color):
            return False
        if np.any(grid[mr0 : mr1 + 1, c0 : c1 + 1] != BG):
            return False
    return True


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(15, 24)
    width = random.randint(18, 27)

    sep_candidates = [
        r
        for r in range(5, height - 6)
        if 0.35 <= r / float(height - 1) <= 0.5 and 2 * r >= height - 1
    ]
    if not sep_candidates:
        return None
    sep_row = random.choice(sep_candidates)

    count = random.choice((2, 3))
    rects = _sample_rectangles(height, width, sep_row, count)
    if rects is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[sep_row, :] = SEP

    colors = random.sample(RECT_COLORS, count)
    for color, (r0, c0, r1, c1) in zip(colors, rects):
        grid[r0 : r1 + 1, c0 : c1 + 1] = color

    if not _valid_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
