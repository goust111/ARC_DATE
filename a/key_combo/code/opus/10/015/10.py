from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
OUTER = 2
INNER = 3
ASYMMETRIC = 4


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    if coords.size == 0:
        raise ValueError("expected a non-empty mask")
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _rectangle_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _bbox(grid == OUTER)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "outer rectangle must enclose an inner area"
    assert (c1 - c0 + 1) % 2 == 1, "rectangle width must be odd"
    assert int(grid[r0, c0]) == OUTER
    assert int(grid[r0, c1]) == OUTER
    assert int(grid[r1, c0]) == OUTER
    assert int(grid[r1, c1]) == OUTER
    assert np.all(grid[r0, c0 : c1 + 1] == OUTER)
    assert np.all(grid[r1, c0 : c1 + 1] == OUTER)
    assert np.all(grid[r0 : r1 + 1, c0] == OUTER)
    assert np.all(grid[r0 : r1 + 1, c1] == OUTER)
    assert np.any(grid[r0 + 1 : r1, c0 + 1 : c1] == INNER), "rectangle must contain an inner fill"
    return r0, c0, r1, c1


def _protrusion_rows(grid: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[set[int], set[int]]:
    r0, c0, r1, c1 = rect
    _, width = grid.shape
    left_rows: set[int] = set()
    right_rows: set[int] = set()

    for r in range(r0 + 1, r1):
        if c0 - 1 >= 0 and int(grid[r, c0 - 1]) == INNER and int(grid[r, c0]) == OUTER:
            left_rows.add(r)
        if c1 + 1 < width and int(grid[r, c1 + 1]) == INNER and int(grid[r, c1]) == OUTER:
            right_rows.add(r)

    assert left_rows or right_rows, "expected at least one side protrusion"

    allowed_outside = {(r, c0 - 1) for r in left_rows if c0 - 1 >= 0}
    allowed_outside |= {(r, c1 + 1) for r in right_rows if c1 + 1 < width}

    for rr, cc in np.argwhere(grid == INNER):
        r = int(rr)
        c = int(cc)
        inside_rect = r0 <= r <= r1 and c0 <= c <= c1
        if not inside_rect:
            assert (r, c) in allowed_outside, "unexpected inner-color cell outside the rectangle"

    return left_rows, right_rows


def _extension_masks(
    shape: tuple[int, int],
    rect: tuple[int, int, int, int],
    left_rows: set[int],
    right_rows: set[int],
) -> tuple[np.ndarray, np.ndarray]:
    _, c0, _, c1 = rect
    extension_mask = np.zeros(shape, dtype=bool)
    marker_mask = np.zeros(shape, dtype=bool)

    for r in left_rows:
        extension_mask[r, :c0] = True
        marker_mask[r, c0 - 1] = True
    for r in right_rows:
        extension_mask[r, c1 + 1 :] = True
        marker_mask[r, c1 + 1] = True

    return extension_mask, marker_mask


def _transform_from_parts(
    grid: np.ndarray,
    rect: tuple[int, int, int, int],
    left_rows: set[int],
    right_rows: set[int],
) -> np.ndarray:
    _, c0, _, c1 = rect
    axis_col = (c0 + c1) // 2
    out = grid.copy()

    for r in sorted(left_rows | right_rows):
        out[r, c0 : c1 + 1] = OUTER

    for r in left_rows:
        out[r, :c0] = INNER
    for r in right_rows:
        out[r, c1 + 1 :] = INNER

    extension_mask, marker_mask = _extension_masks(grid.shape, rect, left_rows, right_rows)
    coords = np.argwhere(extension_mask)
    width = grid.shape[1]

    for rr, cc in coords:
        r = int(rr)
        c = int(cc)
        mirror_c = 2 * axis_col - c
        has_partner = 0 <= mirror_c < width and bool(extension_mask[r, mirror_c])
        if not has_partner and not marker_mask[r, c]:
            out[r, c] = ASYMMETRIC

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extend side leaf rows outward from a two-color rectangle and mark one-sided rays yellow.

    Input structure:
    - A black background contains one red rectangle with a one-cell-thick border and a green interior.
    - Green single-cell protrusions sit just outside the rectangle on its left and/or right side edges.
    - The rectangle width is odd, so its vertical middle column is a unique reflection axis.

    Rule:
    - Each protrusion selects its row.
    - For every selected row, recolor the full rectangle row to red.
    - A left-side protrusion extends a green ray from the rectangle's left edge to the canvas edge on that row; a right-side protrusion does the same on the right.
    - Compare every ray pixel against its mirror across the rectangle's vertical middle column: mirrored ray pixels stay green, while non-marker ray pixels without a mirrored partner become yellow.
    - Keep the original protrusion cells, the rest of the rectangle, and the black background unchanged.

    Key insight:
    - The protrusions do not merely name rows to cut; they also determine which side(s) of each row emit rays, so symmetry must be checked after the side-specific extensions are built.
    """
    g = np.asarray(grid, dtype=int)
    rect = _rectangle_bbox(g)
    left_rows, right_rows = _protrusion_rows(g, rect)
    return _transform_from_parts(g, rect, left_rows, right_rows)


def _sample_geometry() -> tuple[int, int, int, int, int, int]:
    while True:
        rect_h = random.randint(7, 11)
        rect_w = random.choice([7, 9, 11, 13])
        side_margin = random.randint(3, min(8, (30 - rect_w) // 2))
        top_margin = random.randint(2, 7)
        bottom_margin = random.randint(2, 7)
        height = top_margin + rect_h + bottom_margin
        width = side_margin + rect_w + side_margin
        if height <= 30 and width <= 30:
            return height, width, top_margin, side_margin, rect_h, rect_w


def _sample_row_roles(interior_rows: list[int]) -> tuple[set[int], set[int]]:
    while True:
        pair_count = random.randint(1, min(2, len(interior_rows) // 2))
        total_protrusions = random.randint(4, 6)
        unpaired_total = total_protrusions - 2 * pair_count
        unique_rows = pair_count + unpaired_total
        if unpaired_total < 1 or unique_rows > len(interior_rows):
            continue

        chosen = random.sample(interior_rows, unique_rows)
        pair_rows = set(chosen[:pair_count])
        extras = chosen[pair_count:]
        random.shuffle(extras)

        if unpaired_total == 1:
            left_only_count = 1 if random.random() < 0.5 else 0
        else:
            if random.random() < 0.5:
                left_only_count = random.randint(1, unpaired_total - 1)
            else:
                left_only_count = unpaired_total if random.random() < 0.5 else 0

        left_rows = pair_rows | set(extras[:left_only_count])
        right_rows = pair_rows | set(extras[left_only_count:])

        if left_rows != right_rows:
            return left_rows, right_rows


def generate_input() -> np.ndarray:
    height, width, r0, c0, rect_h, rect_w = _sample_geometry()
    r1 = r0 + rect_h - 1
    c1 = c0 + rect_w - 1

    grid = np.full((height, width), BACKGROUND, dtype=int)
    grid[r0 : r1 + 1, c0 : c1 + 1] = OUTER
    grid[r0 + 1 : r1, c0 + 1 : c1] = INNER

    interior_rows = list(range(r0 + 1, r1))
    left_rows, right_rows = _sample_row_roles(interior_rows)

    for r in left_rows:
        grid[r, c0 - 1] = INNER
    for r in right_rows:
        grid[r, c1 + 1] = INNER

    total_protrusions = len(left_rows) + len(right_rows)
    assert 4 <= total_protrusions <= 6
    assert left_rows & right_rows, "need at least one mirrored protrusion pair"
    assert left_rows != right_rows, "need at least one one-sided protrusion row"
    assert max(grid.shape) <= 30

    return grid
