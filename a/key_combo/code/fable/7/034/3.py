"""ARC task 3: stamp successively rotated copies of a framed template."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
FRAME = 5
MARKER = 8
TEMPLATE_COLORS = (1, 2, 3, 4, 6, 7, 9)
MAX_ATTEMPTS = 500


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    """Return the inclusive (row, column) bbox of a nonempty coordinate array."""
    assert coords.ndim == 2 and coords.shape[0] > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _stamp_clipped(dst: np.ndarray, patch: np.ndarray, center: tuple[int, int]) -> None:
    """Opaque-paste a 3x3 patch around center, clipping at canvas edges."""
    h, w = dst.shape
    cr, cc = center
    for pr in range(3):
        for pc in range(3):
            r, c = cr + pr - 1, cc + pc - 1
            if 0 <= r < h and 0 <= c < w:
                dst[r, c] = patch[pr, pc]


def _scan_order(markers: np.ndarray) -> list[tuple[int, int]]:
    """Use the marker bbox's stated reference frame to choose the scan axis."""
    r0, c0, r1, c1 = _bbox(markers)
    height, width = r1 - r0 + 1, c1 - c0 + 1
    coords = [tuple(map(int, p)) for p in markers]
    if height >= width:
        return sorted(coords, key=lambda p: (p[0], p[1]))
    return sorted(coords, key=lambda p: (p[1], p[0]))


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp successively counterclockwise-rotated copies of a framed 3x3 template.

    Input structure:
    - A black grid contains one complete grey 5x5 rectangular frame whose
      interior is a fully colored 3x3 template, plus isolated teal marker pixels
      outside the frame.
    - The template has four visibly distinct quarter-turn orientations.
    Rule:
    - Take the inclusive bounding box of all teal markers.  If its height is at
      least its width, order markers row-first; otherwise order them column-first.
    - At marker k (starting at zero), opaquely stamp the template rotated k
      quarter-turns counterclockwise, centered on the marker; clip at grid edges.
      Preserve the grey frame and its original template.
    Key insight:
    - The marker bbox's longer axis chooses the scan order, and each marker's
      ordinal in that order—not its location or color—chooses its rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    grey = np.argwhere(g == FRAME)
    assert len(grey) == 16, "expected one complete grey 5x5 frame"
    r0, c0, r1, c1 = _bbox(grey)
    assert (r1 - r0 + 1, c1 - c0 + 1) == (5, 5)
    border = np.zeros(g.shape, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True
    assert np.array_equal(g == FRAME, border), "grey must be exactly the frame"

    template = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    assert template.shape == (3, 3)
    assert np.all((template != BACKGROUND) & (template != FRAME) & (template != MARKER))
    assert all(not np.array_equal(template, np.rot90(template, k)) for k in (1, 2, 3))

    markers = np.argwhere(g == MARKER)
    assert len(markers) >= 2, "expected multiple teal markers"
    order = _scan_order(markers)

    out = g.copy()
    for k, center in enumerate(order):
        _stamp_clipped(out, np.rot90(template, k % 4), center)
    return out


def _make_template() -> np.ndarray:
    """Make a multicolor, rotation-asymmetric, fully occupied template."""
    while True:
        colors = random.sample(TEMPLATE_COLORS, random.randint(3, 5))
        template = np.array(
            [[random.choice(colors) for _ in range(3)] for _ in range(3)], dtype=int
        )
        # Every selected color is visible, and no quarter-turn can be a shortcut.
        for i, color in enumerate(colors):
            template.ravel()[i] = color
        if all(not np.array_equal(template, np.rot90(template, k)) for k in (1, 2, 3)):
            return template


def _nonoverlapping(points: list[tuple[int, int]], point: tuple[int, int]) -> bool:
    """True when centered 3x3 stamp boxes do not overlap."""
    r, c = point
    return all(abs(r - rr) >= 3 or abs(c - cc) >= 3 for rr, cc in points)


def _sample_markers(
    region: tuple[int, int, int, int], count: int
) -> list[tuple[int, int]] | None:
    """Fill all four sides of a prescribed marker bbox, then add interior points."""
    r0, c0, r1, c1 = region
    sides = [
        [(r0, c) for c in range(c0, c1 + 1)],
        [(r1, c) for c in range(c0, c1 + 1)],
        [(r, c0) for r in range(r0, r1 + 1)],
        [(r, c1) for r in range(r0, r1 + 1)],
    ]
    all_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]

    for _ in range(150):
        points: list[tuple[int, int]] = []
        for choices in sides:
            viable = [p for p in choices if p not in points and _nonoverlapping(points, p)]
            if not viable:
                break
            points.append(random.choice(viable))
        if len(points) != 4:
            continue
        shuffled = all_cells.copy()
        random.shuffle(shuffled)
        for point in shuffled:
            if len(points) == count:
                break
            if point not in points and _nonoverlapping(points, point):
                points.append(point)
        if len(points) != count:
            continue

        arr = np.array(points, dtype=int)
        if _bbox(arr) != region:
            continue
        row_order = sorted(points, key=lambda p: (p[0], p[1]))
        col_order = sorted(points, key=lambda p: (p[1], p[0]))
        row_rank = {p: i % 4 for i, p in enumerate(row_order)}
        col_rank = {p: i % 4 for i, p in enumerate(col_order)}
        # This is a direct counterexample to always using the opposite scan axis.
        if sum(row_rank[p] != col_rank[p] for p in points) < 4:
            continue
        return points
    return None


def _layout(shape: tuple[int, int], mode: str) -> tuple[tuple[int, int, int, int], tuple[int, int]]:
    """Choose a marker bbox and a well-separated frame origin."""
    h, w = shape
    if mode == "tall":
        mh = random.randint(14, min(18, h - 2))
        mw = random.randint(9, min(12, mh - 2, w - 10))
        left = random.choice((True, False))
        c0 = 0 if left else w - mw
        r0 = random.randint(1, h - mh - 1)
        frame = (random.randint(1, h - 6), w - 6 if left else 1)
    elif mode == "wide":
        mw = random.randint(14, min(18, w - 2))
        mh = random.randint(9, min(12, mw - 2, h - 10))
        top = random.choice((True, False))
        r0 = 0 if top else h - mh
        c0 = random.randint(1, w - mw - 1)
        frame = (h - 6 if top else 1, random.randint(1, w - 6))
    else:
        side = random.randint(12, min(16, h - 2, w - 10))
        mh = mw = side
        left = random.choice((True, False))
        c0 = 0 if left else w - mw
        r0 = random.randint(1, h - mh - 1)
        frame = (random.randint(1, h - 6), w - 6 if left else 1)
    return (r0, c0, r0 + mh - 1, c0 + mw - 1), frame


def _boxes_disjoint(marker: tuple[int, int], frame_origin: tuple[int, int]) -> bool:
    """Check that a marker's clipped 3x3 stamp cannot touch the 5x5 frame."""
    r, c = marker
    fr, fc = frame_origin
    return r + 1 < fr or r - 1 > fr + 4 or c + 1 < fc or c - 1 > fc + 4


def _render_expected(
    inp: np.ndarray, template: np.ndarray, markers: list[tuple[int, int]]
) -> np.ndarray:
    """Independent direct rendering used as the generator's semantic oracle."""
    arr = np.array(markers, dtype=int)
    r0, c0, r1, c1 = _bbox(arr)
    if r1 - r0 + 1 >= c1 - c0 + 1:
        ordered = sorted(markers, key=lambda p: (p[0], p[1]))
    else:
        ordered = sorted(markers, key=lambda p: (p[1], p[0]))
    expected = inp.copy()
    h, w = inp.shape
    for k, (cr, cc) in enumerate(ordered):
        rotated = np.rot90(template, k % 4)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, col = cr + dr, cc + dc
                if 0 <= rr < h and 0 <= col < w:
                    expected[rr, col] = rotated[dr + 1, dc + 1]
    return expected


def _generate_attempt() -> np.ndarray | None:
    h, w = random.randint(23, 30), random.randint(23, 30)
    mode = random.choice(("tall", "wide", "tie"))
    region, frame_origin = _layout((h, w), mode)
    count = random.randint(8, 10)
    markers = _sample_markers(region, count)
    if markers is None or not all(_boxes_disjoint(p, frame_origin) for p in markers):
        return None

    # The marker region touches one canvas edge, so clipping is genuinely tested.
    if not any(r in (0, h - 1) or c in (0, w - 1) for r, c in markers):
        return None

    template = _make_template()
    grid = np.zeros((h, w), dtype=int)
    fr, fc = frame_origin
    grid[fr, fc : fc + 5] = FRAME
    grid[fr + 4, fc : fc + 5] = FRAME
    grid[fr : fr + 5, fc] = FRAME
    grid[fr : fr + 5, fc + 4] = FRAME
    grid[fr + 1 : fr + 4, fc + 1 : fc + 4] = template
    for r, c in markers:
        grid[r, c] = MARKER

    expected = _render_expected(grid, template, markers)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if not np.array_equal(solved[fr : fr + 5, fc : fc + 5], grid[fr : fr + 5, fc : fc + 5]):
        return None

    # Validate that both composition steps matter, not merely that code closes.
    unrotated = grid.copy()
    for point in _scan_order(np.array(markers, dtype=int)):
        _stamp_clipped(unrotated, template, point)
    if np.array_equal(unrotated, expected):
        return None

    marker_array = np.array(markers, dtype=int)
    br0, bc0, br1, bc1 = _bbox(marker_array)
    intended_is_row_first = br1 - br0 + 1 >= bc1 - bc0 + 1
    opposite_order = sorted(
        markers,
        key=(lambda p: (p[1], p[0])) if intended_is_row_first else (lambda p: (p[0], p[1])),
    )
    wrong_axis = grid.copy()
    for k, point in enumerate(opposite_order):
        _stamp_clipped(wrong_axis, np.rot90(template, k % 4), point)
    if np.array_equal(wrong_axis, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate diverse, nondegenerate examples satisfying all solver assumptions."""
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ordinal-stamping input")
