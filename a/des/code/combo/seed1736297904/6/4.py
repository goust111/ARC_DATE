from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GRAY = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
GRID_H_RANGE = (18, 31)
GRID_W_RANGE = (18, 31)
RECT_COUNT_CHOICES = (2, 3, 4)
RECT_H_RANGE = (6, 14)
RECT_W_RANGE = (6, 14)
RECT_MARGIN = 1
RECT_PLACE_TRIES = 200
MAX_GENERATE_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            part: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                part.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(part)
    return parts


def _touches_boundary(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _corners(rect: tuple[int, int, int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0, r1, c1 = rect
    return ((r0, c0), (r0, c1), (r1, c0), (r1, c1))


def _paint_border(mask: np.ndarray, rect: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = rect
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _border_mask(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for rect in rectangles:
        _paint_border(mask, rect)
    return mask


def _point_on_border(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = point
    r0, c0, r1, c1 = rect
    return ((r == r0 or r == r1) and c0 <= c <= c1) or ((c == c0 or c == c1) and r0 <= r <= r1)


def _interiors_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return max(a[0], b[0]) + 1 <= min(a[2], b[2]) - 1 and max(a[1], b[1]) + 1 <= min(a[3], b[3]) - 1


def _container_signature(point: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> tuple[int, ...]:
    r, c = point
    return tuple(
        idx
        for idx, (r0, c0, r1, c1) in enumerate(rectangles)
        if r0 < r < r1 and c0 < c < c1
    )


def _region_signature(cells: list[tuple[int, int]], rectangles: list[tuple[int, int, int, int]]) -> tuple[int, ...]:
    sig = _container_signature(cells[0], rectangles)
    for cell in cells[1:]:
        assert _container_signature(cell, rectangles) == sig, "region signature must be uniform within a component"
    return sig


def _rectangles_for_color(grid: np.ndarray, color: int) -> list[tuple[int, int, int, int]]:
    mask = grid == int(color)
    coords = np.argwhere(mask)
    if len(coords) < 12:
        return []

    h, w = mask.shape
    top_lefts: list[tuple[int, int]] = []
    top_rights: list[tuple[int, int]] = []
    bottom_lefts: list[tuple[int, int]] = []
    bottom_rights: list[tuple[int, int]] = []

    for r, c in coords:
        up = r > 0 and bool(mask[r - 1, c])
        down = r + 1 < h and bool(mask[r + 1, c])
        left = c > 0 and bool(mask[r, c - 1])
        right = c + 1 < w and bool(mask[r, c + 1])

        if right and down and not up and not left:
            top_lefts.append((int(r), int(c)))
        elif left and down and not up and not right:
            top_rights.append((int(r), int(c)))
        elif right and up and not down and not left:
            bottom_lefts.append((int(r), int(c)))
        elif left and up and not down and not right:
            bottom_rights.append((int(r), int(c)))

    n = len(top_lefts)
    if n == 0 or n != len(top_rights) or n != len(bottom_lefts) or n != len(bottom_rights):
        return []

    top_rows = {r for r, _ in top_lefts}
    left_cols = {c for _, c in top_lefts}
    if len(top_rows) != n or len(left_cols) != n:
        return []
    if top_rows != {r for r, _ in top_rights}:
        return []
    if left_cols != {c for _, c in bottom_lefts}:
        return []

    top_right_by_row = {r: c for r, c in top_rights}
    bottom_left_by_col = {c: r for r, c in bottom_lefts}
    bottom_right_set = set(bottom_rights)

    rectangles: list[tuple[int, int, int, int]] = []
    for r0, c0 in sorted(top_lefts):
        c1 = top_right_by_row.get(r0)
        r1 = bottom_left_by_col.get(c0)
        if c1 is None or r1 is None or c1 <= c0 + 2 or r1 <= r0 + 2:
            return []
        if (r1, c1) not in bottom_right_set:
            return []
        if not np.all(mask[r0, c0 : c1 + 1]):
            return []
        if not np.all(mask[r1, c0 : c1 + 1]):
            return []
        if not np.all(mask[r0 : r1 + 1, c0]):
            return []
        if not np.all(mask[r0 : r1 + 1, c1]):
            return []
        rectangles.append((r0, c0, r1, c1))

    if len(set(rectangles)) != n:
        return []

    coverage = _border_mask(grid.shape, rectangles)
    if not np.array_equal(coverage, mask):
        return []

    return rectangles


def _detect_gray_and_rectangles(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    candidates: list[tuple[tuple[int, int], int, list[tuple[int, int, int, int]]]] = []
    for color in map(int, np.unique(grid)):
        if color == BLACK:
            continue
        rectangles = _rectangles_for_color(grid, color)
        if rectangles:
            score = (len(rectangles), int(np.count_nonzero(grid == color)))
            candidates.append((score, color, rectangles))

    assert candidates, "expected at least one recoverable frame color"
    best_score = max(score for score, _, _ in candidates)
    best = [(color, rects) for score, color, rects in candidates if score == best_score]
    assert len(best) == 1, "expected a unique frame color"
    gray_color, rectangles = best[0]
    return int(gray_color), sorted(rectangles)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill closed regions by matching each region's containing-rectangle set to the one encoded by a marker point.

    Input structure:
    - Black background.
    - One-pixel-thick grey rectangular frames, all the same color, possibly overlapping.
    - Colored singleton markers placed in different enclosed regions.

    Rule:
    - Recover every rectangle.
    - For each marker, determine the set of rectangles whose interiors strictly contain it.
    - For each enclosed non-grey region separated by the frame lines, compute the same container set.
    - Fill the whole region with the marker color when the sets match; otherwise leave it black.
    - Remove the markers and keep the grey frame lines unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    gray_color, rectangles = _detect_gray_and_rectangles(g)

    region_to_color: dict[tuple[int, ...], int] = {}
    for color in map(int, np.unique(g)):
        if color in (BLACK, gray_color):
            continue
        coords = np.argwhere(g == color)
        assert len(coords) == 1, "each marker color must appear exactly once"
        r, c = map(int, coords[0])
        sig = _container_signature((r, c), rectangles)
        assert sig not in region_to_color, "marker signatures must be unique"
        region_to_color[sig] = color

    out = np.full_like(g, BLACK)
    out[g == gray_color] = gray_color

    passable = g != gray_color
    for cells in _components(passable):
        if _touches_boundary(cells, g.shape):
            continue
        sig = _region_signature(cells, rectangles)
        fill_color = region_to_color.get(sig, BLACK)
        for r, c in cells:
            out[r, c] = fill_color

    return out


def _choose_rect_size(limit: int, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return random.randint(lo, min(hi, limit))


def _sample_rectangles(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    rectangles: list[tuple[int, int, int, int]] = []
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    corner_points: set[tuple[int, int]] = set()

    max_rect_h = min(RECT_H_RANGE[1], height - 2 * RECT_MARGIN)
    max_rect_w = min(RECT_W_RANGE[1], width - 2 * RECT_MARGIN)
    if max_rect_h < RECT_H_RANGE[0] or max_rect_w < RECT_W_RANGE[0]:
        return None

    for idx in range(count):
        placed = False
        for _ in range(RECT_PLACE_TRIES):
            rect_h = _choose_rect_size(max_rect_h, RECT_H_RANGE)
            rect_w = _choose_rect_size(max_rect_w, RECT_W_RANGE)
            r0 = random.randint(RECT_MARGIN, height - rect_h - RECT_MARGIN)
            c0 = random.randint(RECT_MARGIN, width - rect_w - RECT_MARGIN)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1
            rect = (r0, c0, r1, c1)

            if r0 in used_rows or r1 in used_rows or c0 in used_cols or c1 in used_cols:
                continue
            if idx > 0 and not any(_interiors_overlap(rect, prev) for prev in rectangles):
                continue
            if any(_point_on_border(point, rect) for point in corner_points):
                continue

            tentative = sorted(rectangles + [rect])
            rendered = _render_frames((height, width), tentative)
            if _rectangles_for_color(rendered, GRAY) != tentative:
                continue

            rectangles = tentative
            used_rows.update((r0, r1))
            used_cols.update((c0, c1))
            corner_points.update(_corners(rect))
            placed = True
            break
        if not placed:
            return None

    return sorted(rectangles)


def _render_frames(shape: tuple[int, int], rectangles: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for r0, c0, r1, c1 in rectangles:
        grid[r0, c0 : c1 + 1] = GRAY
        grid[r1, c0 : c1 + 1] = GRAY
        grid[r0 : r1 + 1, c0] = GRAY
        grid[r0 : r1 + 1, c1] = GRAY
    return grid


def _enclosed_regions(
    grid: np.ndarray, rectangles: list[tuple[int, int, int, int]]
) -> list[tuple[list[tuple[int, int]], tuple[int, ...]]]:
    regions: list[tuple[list[tuple[int, int]], tuple[int, ...]]] = []
    for cells in _components(grid != GRAY):
        if _touches_boundary(cells, grid.shape):
            continue
        regions.append((cells, _region_signature(cells, rectangles)))
    return regions


def _choose_marker_signatures(signatures: list[tuple[int, ...]]) -> list[tuple[int, ...]]:
    max_markers = min(len(signatures), len(MARKER_COLORS))
    assert max_markers >= 2

    if len(signatures) == 2:
        mark_count = 2
    else:
        upper = max_markers if random.random() < 0.5 else max(2, max_markers - 1)
        mark_count = random.randint(2, upper)

    chosen: list[tuple[int, ...]] = []
    singletons = [sig for sig in signatures if len(sig) == 1]
    overlaps = [sig for sig in signatures if len(sig) >= 2]
    if singletons and overlaps and mark_count >= 2:
        chosen.append(random.choice(singletons))
        chosen.append(random.choice(overlaps))

    remaining = [sig for sig in signatures if sig not in chosen]
    random.shuffle(remaining)
    while len(chosen) < mark_count:
        chosen.append(remaining.pop())
    return chosen


def _expected_output(
    shape: tuple[int, int],
    rectangles: list[tuple[int, int, int, int]],
    marker_mapping: dict[tuple[int, ...], int],
) -> np.ndarray:
    base = _render_frames(shape, rectangles)
    out = base.copy()
    for cells, sig in _enclosed_regions(base, rectangles):
        fill_color = marker_mapping.get(sig, BLACK)
        for r, c in cells:
            out[r, c] = fill_color
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        height = random.randint(*GRID_H_RANGE)
        width = random.randint(*GRID_W_RANGE)
        rect_count = random.choice(RECT_COUNT_CHOICES)
        rectangles = _sample_rectangles(height, width, rect_count)
        if rectangles is None:
            continue

        base = _render_frames((height, width), rectangles)
        try:
            detected_gray, detected_rectangles = _detect_gray_and_rectangles(base)
        except AssertionError:
            continue
        if detected_gray != GRAY or detected_rectangles != rectangles:
            continue

        regions = _enclosed_regions(base, rectangles)
        if not regions:
            continue

        by_signature: dict[tuple[int, ...], list[list[tuple[int, int]]]] = {}
        for cells, sig in regions:
            by_signature.setdefault(sig, []).append(cells)

        signatures = sorted(by_signature)
        if len(signatures) < 2:
            continue
        if max(len(sig) for sig in signatures) < 2:
            continue

        chosen_signatures = _choose_marker_signatures(signatures)
        marker_colors = random.sample(MARKER_COLORS, len(chosen_signatures))
        mapping = dict(zip(chosen_signatures, marker_colors))

        inp = base.copy()
        for sig, color in mapping.items():
            candidates = [cells for cells in by_signature[sig] if len(cells) >= 2]
            if not candidates:
                candidates = by_signature[sig]
            cells = random.choice(candidates)
            r, c = random.choice(cells)
            inp[r, c] = color

        expected = _expected_output(inp.shape, rectangles, mapping)
        actual = main(inp)
        if not np.array_equal(actual, expected):
            continue
        if np.array_equal(inp, actual):
            continue
        if len({tuple(sig) for sig in chosen_signatures}) != len(chosen_signatures):
            continue
        return inp

    raise RuntimeError("failed to generate a valid ARC input")
