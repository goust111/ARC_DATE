"""ARC task: reconstruct an erased rectangle from four occluded edge projections."""

from __future__ import annotations

import random

import numpy as np


Rect = tuple[int, int, int, int]


def _shadow_color(grid: np.ndarray) -> int:
    """Return the sole non-black color occurring on the outer boundary."""
    border = np.concatenate((grid[0], grid[-1], grid[1:-1, 0], grid[1:-1, -1]))
    colors = set(map(int, np.unique(border))) - {0}
    assert len(colors) == 1, "the boundary must identify exactly one shadow color"
    return colors.pop()


def _trace_masks(blockers: np.ndarray, rect: Rect) -> tuple[np.ndarray, ...]:
    """Render the four edge projections for an erased inclusive rectangle."""
    h, w = blockers.shape
    r0, c0, r1, c1 = rect
    left = np.zeros(h, dtype=bool)
    right = np.zeros(h, dtype=bool)
    top = np.zeros(w, dtype=bool)
    bottom = np.zeros(w, dtype=bool)

    for r in range(r0, r1 + 1):
        left[r] = not bool(blockers[r, 1:c0].any())
        right[r] = not bool(blockers[r, c1 + 1 : w - 1].any())
    for c in range(c0, c1 + 1):
        top[c] = not bool(blockers[1:r0, c].any())
        bottom[c] = not bool(blockers[r1 + 1 : h - 1, c].any())
    return left, right, top, bottom


def _observed_masks(grid: np.ndarray, shadow: int) -> tuple[np.ndarray, ...]:
    return (
        grid[:, 0] == shadow,
        grid[:, -1] == shadow,
        grid[0, :] == shadow,
        grid[-1, :] == shadow,
    )


def _candidate_rectangles(grid: np.ndarray, shadow: int) -> list[Rect]:
    """Find every erased rectangle exactly explaining all four testimonies."""
    h, w = grid.shape
    observed = _observed_masks(grid, shadow)
    side_rows = np.flatnonzero(observed[0] | observed[1])
    side_cols = np.flatnonzero(observed[2] | observed[3])
    if side_rows.size == 0 or side_cols.size == 0:
        return []

    blockers = (grid != 0) & (grid != shadow)
    empty = grid == 0
    occupied_prefix = np.pad((~empty).astype(int), ((1, 0), (1, 0))).cumsum(0).cumsum(1)

    def rectangle_is_empty(rect: Rect) -> bool:
        r0, c0, r1, c1 = rect
        count = (
            occupied_prefix[r1 + 1, c1 + 1]
            - occupied_prefix[r0, c1 + 1]
            - occupied_prefix[r1 + 1, c0]
            + occupied_prefix[r0, c0]
        )
        return int(count) == 0

    min_r, max_r = int(side_rows.min()), int(side_rows.max())
    min_c, max_c = int(side_cols.min()), int(side_cols.max())
    answers: list[Rect] = []

    # A candidate must contain every coordinate that cast a visible projection.
    for r0 in range(1, min_r + 1):
        for r1 in range(max_r, h - 1):
            for c0 in range(1, min_c + 1):
                for c1 in range(max_c, w - 1):
                    rect = (r0, c0, r1, c1)
                    if not rectangle_is_empty(rect):
                        continue
                    predicted = _trace_masks(blockers, rect)
                    if all(np.array_equal(a, b) for a, b in zip(predicted, observed)):
                        answers.append(rect)
    return answers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct an erased solid rectangle from its four blocker-interrupted edge projections.

    Input structure:
    - The background is black; one shadow color occurs only in line segments on the four outer edges.
    - Several internally placed, non-shadow-colored solid blockers interrupt horizontal or vertical sight lines.
    - The shadow-colored source rectangle itself has been erased, so its intended cells are black.

    Rule:
    - Test interior rectangles against all four edge traces: every source row projects left and right, and every
      source column projects up and down, exactly when no blocker lies between that rectangle and the relevant edge.
    - The unique rectangle whose four predicted traces equal the visible traces is filled with the shadow color.
    - Preserve every edge trace, blocker, and all other input cells.

    Key insight:
    - A run endpoint beside a gap can be caused by an occluding blocker, so only the joint, blocker-aware testimony
      of opposite and orthogonal edges determines the never-drawn rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) >= 3
    shadow = _shadow_color(g)
    assert not bool((g[1:-1, 1:-1] == shadow).any()), "shadow color must be absent internally"
    assert bool(np.all(g[0, [0, -1]] == 0)) and bool(np.all(g[-1, [0, -1]] == 0))

    candidates = _candidate_rectangles(g, shadow)
    assert len(candidates) == 1, f"expected one reconstructable rectangle, found {len(candidates)}"
    r0, c0, r1, c1 = candidates[0]
    out = g.copy()
    out[r0 : r1 + 1, c0 : c1 + 1] = shadow
    return out


def _paint_blocker(
    grid: np.ndarray,
    rects: list[Rect],
    rect: Rect,
    color: int,
) -> bool:
    """Place a solid rectangle if it is separated from all earlier blockers."""
    r0, c0, r1, c1 = rect
    h, w = grid.shape
    if not (1 <= r0 <= r1 < h - 1 and 1 <= c0 <= c1 < w - 1):
        return False
    for ar0, ac0, ar1, ac1 in rects:
        if not (r1 + 1 < ar0 or ar1 + 1 < r0 or c1 + 1 < ac0 or ac1 + 1 < c0):
            return False
    if bool(np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != 0)):
        return False
    grid[r0 : r1 + 1, c0 : c1 + 1] = color
    rects.append(rect)
    return True


def _random_horizontal_blocker(row: int, lo: int, hi: int) -> Rect | None:
    """A 1-by-(1 or 2) blocker inside an inclusive column corridor."""
    if lo > hi:
        return None
    width = random.randint(1, min(2, hi - lo + 1))
    c0 = random.randint(lo, hi - width + 1)
    return row, c0, row, c0 + width - 1


def _random_vertical_blocker(col: int, lo: int, hi: int) -> Rect | None:
    """A (1 or 2)-by-1 blocker inside an inclusive row corridor."""
    if lo > hi:
        return None
    height = random.randint(1, min(2, hi - lo + 1))
    r0 = random.randint(lo, hi - height + 1)
    return r0, col, r0 + height - 1, col


def _render_input(base: np.ndarray, rect: Rect, shadow: int) -> np.ndarray:
    blockers = base != 0
    left, right, top, bottom = _trace_masks(blockers, rect)
    grid = base.copy()
    grid[left, 0] = shadow
    grid[right, -1] = shadow
    grid[0, top] = shadow
    grid[-1, bottom] = shadow
    return grid


def _has_internal_gap(mask: np.ndarray, start: int, end: int) -> bool:
    values = mask[start : end + 1]
    return bool(values.any() and (~values).any() and (~values[1:-1]).any())


def _validate_generated(
    grid: np.ndarray,
    rect: Rect,
    shadow: int,
    blocker_rects: list[Rect],
) -> bool:
    """Check the stated mechanism, uniqueness, blocker witnesses, and anti-shortcuts."""
    h, w = grid.shape
    r0, c0, r1, c1 = rect
    if max(grid.shape) > 30 or min(grid.shape) < 3:
        return False
    if _shadow_color(grid) != shadow or bool((grid[1:-1, 1:-1] == shadow).any()):
        return False
    if len(blocker_rects) < 6 or len(blocker_rects) > 8:
        return False

    blockers = (grid != 0) & (grid != shadow)
    predicted = _trace_masks(blockers, rect)
    observed = _observed_masks(grid, shadow)
    if not all(np.array_equal(a, b) for a, b in zip(predicted, observed)):
        return False
    if not all(bool(side.any()) for side in observed):
        return False

    # Opposite sides contribute different true ends; no single side gives the interval.
    left, right, top, bottom = observed
    if not (left[r0] and not right[r0] and not left[r1] and right[r1]):
        return False
    if not (top[c0] and not bottom[c0] and not top[c1] and bottom[c1]):
        return False
    if not (_has_internal_gap(left | right, r0, r1) or _has_internal_gap(left, r0, r1) or _has_internal_gap(right, r0, r1)):
        return False
    if not (_has_internal_gap(top | bottom, c0, c1) or _has_internal_gap(top, c0, c1) or _has_internal_gap(bottom, c0, c1)):
        return False

    # Every generated blocker must be visible evidence: removing it changes a trace.
    for br0, bc0, br1, bc1 in blocker_rects:
        reduced = blockers.copy()
        reduced[br0 : br1 + 1, bc0 : bc1 + 1] = False
        changed = _trace_masks(reduced, rect)
        if all(np.array_equal(a, b) for a, b in zip(changed, predicted)):
            return False

    candidates = _candidate_rectangles(grid, shadow)
    if candidates != [rect]:
        return False
    expected = grid.copy()
    expected[r0 : r1 + 1, c0 : c1 + 1] = shadow
    try:
        solved = main(grid)
    except AssertionError:
        return False
    return np.array_equal(solved, expected) and not np.array_equal(solved, grid)


def _generation_attempt() -> np.ndarray | None:
    h = random.randint(18, 28)
    w = random.randint(18, 28)
    rh = random.randint(5, min(8, h - 10))
    rw = random.randint(5, min(8, w - 10))
    r0 = random.randint(5, h - rh - 5)
    c0 = random.randint(5, w - rw - 5)
    r1, c1 = r0 + rh - 1, c0 + rw - 1
    target = (r0, c0, r1, c1)

    shadow = random.randint(1, 9)
    blocker_color_count = random.choice((2, 3))
    blocker_colors = random.sample([c for c in range(1, 10) if c != shadow], blocker_color_count)
    base = np.zeros((h, w), dtype=int)
    blocker_rects: list[Rect] = []

    # Four mandatory occluders create complementary false ends on opposite sides.
    specifications: list[tuple[str, int]] = [
        ("left", r1),
        ("right", r0),
        ("top", c1),
        ("bottom", c0),
    ]

    # At least one internal horizontal and vertical gap; optional second gaps vary count.
    horizontal_sides = random.sample(["left", "right"], random.randint(1, 2))
    vertical_sides = random.sample(["top", "bottom"], random.randint(1, 2))
    for side in horizontal_sides:
        lo = r0 + 1 if side == "left" else r0 + 2
        hi = r1 - 2 if side == "left" else r1 - 1
        if lo > hi:
            return None
        specifications.append((side, random.randint(lo, hi)))
    for side in vertical_sides:
        lo = c0 + 1 if side == "top" else c0 + 2
        hi = c1 - 2 if side == "top" else c1 - 1
        if lo > hi:
            return None
        specifications.append((side, random.randint(lo, hi)))

    random.shuffle(specifications)
    assigned_colors = (blocker_colors * 4)[: len(specifications)]
    random.shuffle(assigned_colors)
    for (side, axis), color in zip(specifications, assigned_colors):
        if side == "left":
            blocker = _random_horizontal_blocker(axis, 2, c0 - 2)
        elif side == "right":
            blocker = _random_horizontal_blocker(axis, c1 + 2, w - 3)
        elif side == "top":
            blocker = _random_vertical_blocker(axis, 2, r0 - 2)
        else:
            blocker = _random_vertical_blocker(axis, r1 + 2, h - 3)
        if blocker is None or not _paint_blocker(base, blocker_rects, blocker, color):
            return None

    # Each selected color must be structurally instantiated, without color-id semantics.
    present = set(map(int, np.unique(base))) - {0}
    if present != set(blocker_colors):
        return None

    grid = _render_input(base, target, shadow)
    return grid if _validate_generated(grid, target, shadow, blocker_rects) else None


def generate_input() -> np.ndarray:
    """Generate a diverse, blocker-witnessed instance with a unique reconstruction."""
    for _ in range(500):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a unique shadow-testimony instance")
