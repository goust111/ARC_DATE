from __future__ import annotations

from collections import Counter

import numpy as np


BACKGROUND = 0
OUTLINE_COLOR = 1
SEPARATOR_COLOR = 3
MARKER_COLORS = (2, 4, 6, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _infer_puzzle_colors(grid: np.ndarray, bg: int) -> tuple[int, int]:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert len(colors) == 2, "expected one outline color and one marker color"
    counts = {color: int(np.count_nonzero(grid == color)) for color in colors}
    marker_candidates = [color for color, count in counts.items() if count == 1]
    assert len(marker_candidates) == 1, "expected exactly one singleton marker color"
    marker_color = marker_candidates[0]
    outline_color = next(color for color in colors if color != marker_color)
    return outline_color, marker_color


def _mirror_outline(mask: np.ndarray) -> tuple[np.ndarray, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "outline must be non-empty"
    c_max = int(coords[:, 1].max())
    full_mask = mask.copy()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        mc = 2 * c_max - cc + 1
        assert 0 <= mc < mask.shape[1], "mirrored outline exceeds the canvas"
        full_mask[rr, mc] = True
    return full_mask, c_max


def _find_separator_row(full_mask: np.ndarray) -> tuple[tuple[int, int, int, int], int]:
    shape_box = _bbox(full_mask)
    assert shape_box is not None, "mirrored outline must have a bbox"
    r0, c0, r1, c1 = shape_box
    separators = [
        r
        for r in range(r0 + 1, r1)
        if not np.any(full_mask[r, c0 : c1 + 1])
    ]
    assert len(separators) == 1, "expected exactly one separator row"
    return shape_box, int(separators[0])


def main(grid: np.ndarray) -> np.ndarray:
    """Mirror the half-outline, erase it, and fill the two regions split by the unique blank row.

    Input structure:
    - A black background contains a left-half outline of a vertically symmetric shape.
    - The outline color appears many times, while a second non-background color appears once as an isolated marker outside the mirrored shape's bbox.
    - The mirrored full shape has exactly one completely empty interior row inside its bbox.

    Rule:
    - Infer the outline color and the singleton marker color.
    - Mirror the half-outline across the vertical axis immediately to the right of its global rightmost column.
    - Erase the outline, find the unique empty row inside the mirrored bbox, and treat it as a separator.
    - For rows strictly between the bbox top/bottom and the separator, fill the span between that row's leftmost and rightmost mirrored outline pixels.
    - Use the outline color above the separator, the marker color below it, and paint the separator row green across the whole canvas.

    Key insight:
    - Three clues interact: the half-outline defines the hidden symmetric container, the unique blank row splits it into two zones, and the isolated marker supplies the lower zone's fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    outline_color, marker_color = _infer_puzzle_colors(g, bg)
    outline_mask = g == outline_color

    full_mask, _c_max = _mirror_outline(outline_mask)
    (r0, c0, r1, c1), separator_row = _find_separator_row(full_mask)

    out = np.full_like(g, bg)
    for r in range(r0 + 1, r1):
        if r == separator_row:
            continue
        row_cols = np.flatnonzero(full_mask[r])
        assert row_cols.size >= 2, "filled rows need mirrored boundaries"
        left = int(row_cols.min())
        right = int(row_cols.max())
        fill_color = outline_color if r < separator_row else marker_color
        if left + 1 < right:
            out[r, left + 1 : right] = fill_color

    out[separator_row, :] = SEPARATOR_COLOR
    return out


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _build_section_spans(count: int, half_width: int) -> list[int]:
    assert count >= 2
    spans = [_randint(1, min(2, half_width))]
    while len(spans) < count:
        prev = spans[-1]
        low = prev
        high = min(half_width, prev + 1)
        if len(spans) == count - 1:
            low = max(low, 2)
        spans.append(_randint(low, high))
    if spans[-1] == spans[0] and spans[-1] < half_width and np.random.rand() < 0.8:
        spans[-1] += 1
    return spans


def _choose_insets(top_spans: list[int], bottom_spans: list[int]) -> tuple[list[int], list[int]]:
    top_insets: list[int] = []
    bottom_insets: list[int] = []

    for idx, span in enumerate(top_spans):
        if idx == 0 or span <= 1:
            top_insets.append(0)
        else:
            top_insets.append(_randint(0, min(2, span - 1)))
    for idx, span in enumerate(bottom_spans):
        if idx == len(bottom_spans) - 1 or span <= 1:
            bottom_insets.append(0)
        else:
            bottom_insets.append(_randint(0, min(2, span - 1)))

    interior_slots = []
    for idx, span in enumerate(top_spans):
        if idx > 0 and span > 1:
            interior_slots.append(("top", idx))
    for idx, span in enumerate(bottom_spans):
        if idx < len(bottom_spans) - 1 and span > 1:
            interior_slots.append(("bottom", idx))

    if interior_slots and all(v == 0 for v in top_insets[1:] + bottom_insets[:-1]):
        side, idx = interior_slots[_randint(0, len(interior_slots) - 1)]
        if side == "top":
            top_insets[idx] = 1
        else:
            bottom_insets[idx] = 1

    return top_insets, bottom_insets


def _marker_candidates(grid: np.ndarray, full_box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    h, w = grid.shape
    r0, c0, r1, c1 = full_box
    candidates: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if grid[r, c] != BACKGROUND:
                continue
            isolated = True
            for rr in range(max(0, r - 1), min(h, r + 2)):
                for cc in range(max(0, c - 1), min(w, c + 2)):
                    if (rr, cc) != (r, c) and grid[rr, cc] != BACKGROUND:
                        isolated = False
                        break
                if not isolated:
                    break
            if isolated:
                candidates.append((r, c))
    return candidates


def _generate_attempt() -> np.ndarray | None:
    top_count = _randint(2, 4)
    bottom_count = _randint(2, 4)
    shape_h = top_count + 1 + bottom_count
    half_width = _randint(3, 5)
    full_width = 2 * half_width

    top_margin = _randint(2, 4)
    bottom_margin = _randint(2, 4)
    left_margin = _randint(2, 4)
    right_margin = _randint(2, 4)
    h = top_margin + shape_h + bottom_margin
    w = left_margin + full_width + right_margin

    if max(h, w) > 30:
        return None

    grid = np.full((h, w), BACKGROUND, dtype=int)
    marker_color = int(np.random.choice(MARKER_COLORS))

    r0 = top_margin
    c0 = left_margin
    c_max = c0 + half_width - 1
    separator_local = top_count
    separator_row = r0 + separator_local
    full_box = (r0, c0, r0 + shape_h - 1, c0 + full_width - 1)

    top_spans = _build_section_spans(top_count, half_width)
    bottom_spans = list(reversed(_build_section_spans(bottom_count, half_width)))
    top_insets, bottom_insets = _choose_insets(top_spans, bottom_spans)

    for idx, span in enumerate(top_spans):
        outer_left = c_max - span + 1
        inner_left = c_max - top_insets[idx]
        grid[r0 + idx, outer_left : inner_left + 1] = OUTLINE_COLOR

    for idx, span in enumerate(bottom_spans):
        outer_left = c_max - span + 1
        inner_left = c_max - bottom_insets[idx]
        grid[separator_row + 1 + idx, outer_left : inner_left + 1] = OUTLINE_COLOR

    candidates = _marker_candidates(grid, full_box)
    if not candidates:
        return None

    marker_row, marker_col = candidates[_randint(0, len(candidates) - 1)]
    grid[marker_row, marker_col] = marker_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.count_nonzero(grid == marker_color) != 1:
        return None
    if np.any(grid[full_box[0] : full_box[2] + 1, full_box[1] : full_box[3] + 1] == marker_color):
        return None
    if not np.all(out[separator_row, :] == SEPARATOR_COLOR):
        return None
    if np.count_nonzero(out == OUTLINE_COLOR) == 0:
        return None
    if np.count_nonzero(out == marker_color) == 0:
        return None
    if not np.array_equal(out[separator_row, :], np.full(w, SEPARATOR_COLOR, dtype=int)):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
