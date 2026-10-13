from __future__ import annotations

import random

import numpy as np


GREEN = 3
BLACK = 0
MARKER_COLORS = (1, 2, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _completed_outline(grid: np.ndarray) -> tuple[np.ndarray, int]:
    outline = np.zeros_like(grid, dtype=bool)
    green_coords = np.argwhere(grid == GREEN)
    assert green_coords.size > 0, "expected a green half-outline"

    c_max = int(green_coords[:, 1].max())
    h, w = grid.shape
    for r, c in green_coords:
        rr = int(r)
        cc = int(c)
        mc = 2 * c_max + 1 - cc
        assert 0 <= mc < w, "mirrored outline would leave the grid"
        outline[rr, cc] = True
        outline[rr, mc] = True
    return outline, c_max


def _interior_mask(outline: np.ndarray) -> np.ndarray:
    h, w = outline.shape
    interior = np.zeros((h, w), dtype=bool)
    for r in range(h):
        cols = np.flatnonzero(outline[r])
        if cols.size >= 2:
            left = int(cols[0])
            right = int(cols[-1])
            if right - left >= 2:
                interior[r, left + 1 : right] = True
    interior &= ~outline
    return interior


def _majority_internal_marker(grid: np.ndarray, interior: np.ndarray) -> int:
    marker_mask = (grid != BLACK) & (grid != GREEN) & interior
    colors, counts = np.unique(grid[marker_mask], return_counts=True)
    assert colors.size > 0, "expected at least one internal marker"
    best = int(np.argmax(counts))
    assert list(counts).count(int(counts[best])) == 1, "expected a unique majority marker color"
    return int(colors[best])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-complete the green half-outline, then fill the enclosed interior with the internal majority marker color.

    Input structure:
    - A black background contains a green one-pixel-thick half-outline whose rightmost
      green column touches the vertical reflection seam.
    - Several single-pixel markers of 2-3 other colors are scattered across the grid.
    - Some markers fall inside the region enclosed after mirror completion, and others stay outside.

    Rule:
    - Reflect every green pixel across the vertical axis immediately to the right of the
      half-outline's rightmost green column, producing a full symmetric outline.
    - For each row, treat the cells strictly between the leftmost and rightmost green
      outline pixels as interior cells.
    - Count the colors of marker pixels that lie in that interior, fill the whole interior
      with the unique majority marker color, keep the interior markers on top, erase all
      exterior markers, and preserve the green outline.

    Key insight:
    - The fill color is determined only after the geometry is completed: mirroring the
      half-outline reveals which scattered markers count as interior votes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30, "expected a 2D ARC-sized grid"

    outline, _ = _completed_outline(g)
    interior = _interior_mask(outline)
    fill_color = _majority_internal_marker(g, interior)

    out = np.zeros_like(g)
    out[interior] = fill_color
    inside_markers = (g != BLACK) & (g != GREEN) & interior
    out[inside_markers] = g[inside_markers]
    out[outline] = GREEN
    return out


def _sample_spans(shape_h: int, max_span: int) -> list[int]:
    spans = [random.randint(1, max_span)]
    for _ in range(1, shape_h):
        prev = spans[-1]
        lo = max(1, prev - 1)
        hi = min(max_span, prev + 1)
        spans.append(random.randint(lo, hi))
    if max(spans) < max_span:
        idx = random.randint(1, shape_h - 2)
        spans[idx] = max_span
    return spans


def _build_half_outline(grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid_shape
    shape_h = random.randint(8, min(12, h - 2))
    max_span = random.randint(2, 4)
    top = random.randint(1, h - shape_h - 1)
    c_mid = random.randint(max_span + 1, w - max_span - 2)

    spans = _sample_spans(shape_h, max_span)
    rows = [top + i for i in range(shape_h)]
    left_cols = [c_mid - span for span in spans]

    inp = np.zeros((h, w), dtype=int)
    for r, c in zip(rows, left_cols):
        inp[r, c] = GREEN

    top_row = rows[0]
    bot_row = rows[-1]
    inp[top_row, left_cols[0] : c_mid + 1] = GREEN
    inp[bot_row, left_cols[-1] : c_mid + 1] = GREEN

    outline, _ = _completed_outline(inp)
    interior = _interior_mask(outline)
    return inp, outline, interior


def _distribute_positive(total: int, parts: int, cap: int) -> list[int] | None:
    base = [1] * parts
    extra = total - parts
    if extra < 0:
        return None
    for _ in range(80):
        counts = base.copy()
        ok = True
        for i in range(parts):
            if i == parts - 1:
                add = extra
            else:
                add = random.randint(0, extra)
            counts[i] += add
            extra -= add
            if counts[i] > cap:
                ok = False
                break
        if ok and extra == 0:
            random.shuffle(counts)
            return counts
        extra = total - parts
    return None


def _sample_internal_counts(color_count: int, max_internal: int) -> list[int] | None:
    min_internal = max(2, color_count + (1 if color_count > 1 else 0))
    if max_internal < min_internal:
        return None
    for _ in range(120):
        internal_total = random.randint(min_internal, max_internal)
        majority = random.randint(2, internal_total - (color_count - 1))
        rest_total = internal_total - majority
        if color_count == 1:
            return [majority]
        if not (color_count - 1 <= rest_total <= (color_count - 1) * (majority - 1)):
            continue
        rest = _distribute_positive(rest_total, color_count - 1, majority - 1)
        if rest is None:
            continue
        counts = rest + [majority]
        random.shuffle(counts)
        if counts.count(max(counts)) == 1:
            return counts
    return None


def _place_markers(
    grid: np.ndarray,
    interior: np.ndarray,
    outline: np.ndarray,
) -> np.ndarray | None:
    inside_cells = [tuple(map(int, rc)) for rc in np.argwhere(interior)]
    outside_mask = ~(interior | outline)
    outside_cells = [tuple(map(int, rc)) for rc in np.argwhere(outside_mask)]
    if len(inside_cells) < 4 or len(outside_cells) < 4:
        return None

    total_markers = random.randint(6, 10)
    color_count = random.randint(2, 3)
    colors = random.sample(MARKER_COLORS, color_count)
    internal_counts = _sample_internal_counts(color_count, min(6, total_markers - 1, len(inside_cells)))
    if internal_counts is None:
        return None

    n_inside = sum(internal_counts)
    n_outside = total_markers - n_inside
    if n_outside < 1 or n_outside > len(outside_cells):
        return None

    random.shuffle(inside_cells)
    random.shuffle(outside_cells)
    for color, count in zip(colors, internal_counts):
        for _ in range(count):
            r, c = inside_cells.pop()
            grid[r, c] = color

    for _ in range(n_outside):
        r, c = outside_cells.pop()
        grid[r, c] = random.choice(colors)
    return grid


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.randint(13, 18)
    inp, outline, interior = _build_half_outline((h, w))
    if np.count_nonzero(interior) < 6:
        return None

    candidate = _place_markers(inp.copy(), interior, outline)
    if candidate is None:
        return None

    try:
        out = main(candidate)
    except AssertionError:
        return None

    if out.shape != candidate.shape or max(out.shape) > 30:
        return None
    if np.count_nonzero((candidate != BLACK) & (candidate != GREEN) & interior) < 2:
        return None
    return candidate


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
