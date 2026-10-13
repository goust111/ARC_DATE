from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
RED = 2
BODY_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)

VERTICAL_HINT = 0
HORIZONTAL_HINT = 1

GEN_ATTEMPTS = 300
OBJECT_COUNT_RANGE = (2, 6)  # upper bound exclusive
SIZE_RANGE = (3, 8)  # upper bound exclusive
MASK_ATTEMPTS = 80
SHAPE_ATTEMPTS = 60


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    seen = np.zeros(mask.shape, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    height, width = map(int, mask.shape)
    for r in range(height):
        for c in range(width):
            if seen[r, c] or not mask[r, c]:
                continue
            seen[r, c] = True
            queue: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(comp)
    return parts


def _marker_hint(marker: tuple[int, int], shape: tuple[int, int]) -> int:
    mr, mc = marker
    center_r = (shape[0] - 1) / 2.0
    center_c = (shape[1] - 1) / 2.0
    dr = abs(mr - center_r)
    dc = abs(mc - center_c)
    if dc <= 1.0 and dr > dc:
        return VERTICAL_HINT
    if dr <= 1.0 and dc > dr:
        return HORIZONTAL_HINT
    raise AssertionError("marker direction is ambiguous")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Collapse each marked shape into a one-cell-wide projection bar chosen by the marker's direction from the bbox center.

    Input structure:
    - The background is black.
    - Several disconnected 4-connected shapes appear in distinct non-red colors.
    - Each shape contains exactly one red marker pixel that replaces one shape cell, and shape bounding boxes do not overlap.

    Rule:
    - For each non-black connected component, treat every non-black cell, including the red marker, as part of the shape.
    - Compute the geometric center of the component's bounding box and inspect the marker's offset from that center.
    - If the marker lies mainly above or below the center (within one column of it), compress the shape vertically: keep only a horizontal bar on the bbox middle row, coloring the columns that contain any shape cell.
    - If the marker lies mainly left or right of the center (within one row of it), compress the shape horizontally: keep only a vertical bar on the bbox middle column, coloring the rows that contain any shape cell.
    - Erase the original shape and the red marker; preserve only the projected bar in the shape's base color.

    Key insight:
    - The red pixel is not part of the final picture; its position relative to the shape's bbox center selects which projection axis to use before the shape is flattened into a central bar.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = np.full_like(g, BG)
    for comp in _components(g != BG):
        coords = np.array(comp, dtype=int)
        values = g[coords[:, 0], coords[:, 1]]
        colors = set(map(int, values))
        assert RED in colors, "each shape must contain one red marker"
        body_colors = colors - {RED}
        assert len(body_colors) == 1, "each shape must have exactly one base color"
        body_color = int(next(iter(body_colors)))
        assert int(np.sum(values == RED)) == 1, "each shape must contain exactly one red marker"

        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True

        marker_local = tuple(map(int, (coords[values == RED][0] - np.array([r0, c0]))))
        hint = _marker_hint(marker_local, local.shape)

        if hint == VERTICAL_HINT:
            mid_row = r0 + (local.shape[0] - 1) // 2
            present = np.any(local, axis=0)
            for j, filled in enumerate(present):
                if filled:
                    out[mid_row, c0 + j] = body_color
        else:
            mid_col = c0 + (local.shape[1] - 1) // 2
            present = np.any(local, axis=1)
            for i, filled in enumerate(present):
                if filled:
                    out[r0 + i, mid_col] = body_color

    return out


def _build_interval_mask(height: int, width: int, rng: np.random.Generator) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        span = int(rng.integers(max(1, width // 2), width + 1))
        left = int(rng.integers(0, width - span + 1))
        intervals = [(left, left + span - 1)]

        for _ in range(1, height):
            prev_left, prev_right = intervals[-1]
            prev_span = prev_right - prev_left + 1
            span_lo = max(1, prev_span - 2)
            span_hi = min(width, prev_span + 2)
            span = int(rng.integers(span_lo, span_hi + 1))
            left_lo = max(0, prev_left - span + 1)
            left_hi = min(width - span, prev_right)
            if left_lo > left_hi:
                break
            left = int(rng.integers(left_lo, left_hi + 1))
            intervals.append((left, left + span - 1))

        if len(intervals) != height:
            continue

        mask = np.zeros((height, width), dtype=bool)
        for r, (c0, c1) in enumerate(intervals):
            mask[r, c0 : c1 + 1] = True

        area = int(mask.sum())
        if area == height * width:
            continue
        if area < max(height + width - 2, (height * width + 2) // 3):
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        return mask

    return None


def _vertical_marker_candidates(height: int, width: int, row: int) -> list[int]:
    center_r = (height - 1) / 2.0
    center_c = (width - 1) / 2.0
    row_offset = abs(row - center_r)
    return [c for c in range(width) if abs(c - center_c) <= 1.0 and abs(c - center_c) < row_offset]


def _force_marker_row(mask: np.ndarray, row: int, target_col: int) -> None:
    cols = np.flatnonzero(mask[row])
    assert cols.size > 0, "every row must already contain the shape"
    left = min(int(cols[0]), int(target_col))
    right = max(int(cols[-1]), int(target_col))
    mask[row, :] = False
    mask[row, left : right + 1] = True


def _apply_random_flips(
    mask: np.ndarray, marker: tuple[int, int], rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = map(int, mask.shape)
    mr, mc = marker
    if bool(rng.integers(0, 2)):
        mask = np.flipud(mask).copy()
        mr = h - 1 - mr
    if bool(rng.integers(0, 2)):
        mask = np.fliplr(mask).copy()
        mc = w - 1 - mc
    return mask, (mr, mc)


def _make_vertical_shape(
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int]] | None:
    height = int(rng.integers(*SIZE_RANGE))
    width = int(rng.integers(*SIZE_RANGE))
    mask = _build_interval_mask(height, width, rng)
    if mask is None:
        return None

    marker_row = 0 if bool(rng.integers(0, 2)) else height - 1
    candidates = _vertical_marker_candidates(height, width, marker_row)
    if not candidates:
        return None
    target_col = int(rng.choice(candidates))
    _force_marker_row(mask, marker_row, target_col)
    marker = (marker_row, target_col)
    if _marker_hint(marker, mask.shape) != VERTICAL_HINT:
        return None
    return mask, marker


def _make_shape(
    hint: int, rng: np.random.Generator
) -> tuple[np.ndarray, tuple[int, int]] | None:
    for _ in range(SHAPE_ATTEMPTS):
        built = _make_vertical_shape(rng)
        if built is None:
            continue
        mask, marker = built
        if hint == HORIZONTAL_HINT:
            mask = mask.T.copy()
            marker = (marker[1], marker[0])
        mask, marker = _apply_random_flips(mask, marker, rng)
        if _marker_hint(marker, mask.shape) == hint:
            return mask, marker
    return None


def _paint_patch(mask: np.ndarray, marker: tuple[int, int], color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    patch[marker] = RED
    return patch


def _find_positions(blocked: np.ndarray, patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    height, width = map(int, blocked.shape)
    ph, pw = map(int, patch_shape)
    positions: list[tuple[int, int]] = []
    for r0 in range(height - ph + 1):
        for c0 in range(width - pw + 1):
            rr0 = max(0, r0 - 1)
            rr1 = min(height, r0 + ph + 1)
            cc0 = max(0, c0 - 1)
            cc1 = min(width, c0 + pw + 1)
            if not blocked[rr0:rr1, cc0:cc1].any():
                positions.append((r0, c0))
    return positions


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    ph, pw = map(int, patch.shape)
    region = grid[top : top + ph, left : left + pw]
    region[patch != BG] = patch[patch != BG]


def _draw_expected(
    expected: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    hint: int,
    color: int,
) -> None:
    height, width = map(int, mask.shape)
    if hint == VERTICAL_HINT:
        row = top + (height - 1) // 2
        for col, filled in enumerate(np.any(mask, axis=0)):
            if filled:
                expected[row, left + col] = int(color)
    else:
        col = left + (width - 1) // 2
        for row, filled in enumerate(np.any(mask, axis=1)):
            if filled:
                expected[top + row, col] = int(color)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_objects = int(rng.integers(*OBJECT_COUNT_RANGE))

    hints = [VERTICAL_HINT, HORIZONTAL_HINT]
    while len(hints) < n_objects:
        hints.append(int(rng.integers(0, 2)))
    rng.shuffle(hints)

    colors = [int(c) for c in rng.choice(np.array(BODY_COLORS), size=n_objects, replace=False)]
    shapes: list[tuple[int, int, np.ndarray, tuple[int, int]]] = []
    for hint, color in zip(hints, colors):
        built = _make_shape(hint, rng)
        if built is None:
            return None
        mask, marker = built
        shapes.append((hint, color, mask, marker))

    shapes.sort(key=lambda item: (-int(item[2].shape[0] * item[2].shape[1]), rng.random()))
    total_bbox_area = sum(int(mask.shape[0] * mask.shape[1]) for _, _, mask, _ in shapes)
    max_h = max(int(mask.shape[0]) for _, _, mask, _ in shapes)
    max_w = max(int(mask.shape[1]) for _, _, mask, _ in shapes)

    min_height = max(max_h + 2, int(np.ceil(np.sqrt(total_bbox_area * 2.3))))
    min_width = max(max_w + 2, int(np.ceil(np.sqrt(total_bbox_area * 2.3))))
    min_height = min(min_height, 30)
    min_width = min(min_width, 30)
    if min_height > 30 or min_width > 30:
        return None

    height = int(rng.integers(min_height, 31))
    width = int(rng.integers(min_width, 31))
    grid = np.full((height, width), BG, dtype=int)
    expected = np.full_like(grid, BG)
    blocked = np.zeros((height, width), dtype=bool)

    for hint, color, mask, marker in shapes:
        positions = _find_positions(blocked, mask.shape)
        if not positions:
            return None
        top, left = positions[int(rng.integers(0, len(positions)))]

        patch = _paint_patch(mask, marker, color)
        _place_patch(grid, patch, top, left)
        _draw_expected(expected, top, left, mask, hint, color)

        ph, pw = map(int, patch.shape)
        rr0 = max(0, top - 1)
        rr1 = min(height, top + ph + 1)
        cc0 = max(0, left - 1)
        cc1 = min(width, left + pw + 1)
        blocked[rr0:rr1, cc0:cc1] = True

    try:
        if len(_components(grid != BG)) != n_objects:
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
