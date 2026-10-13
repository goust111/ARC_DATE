from __future__ import annotations

from collections import deque
import random

import numpy as np


# ARC palette constants used by the puzzle statement.
BLACK = 0
GRAY = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

Coord = tuple[int, int]
Rect = tuple[int, int, int, int]  # inclusive (r0, c0, r1, c1)


def _neighbors4(r: int, c: int):
    """Yield neighbors in the shared ARC convention: coordinates are (row, col)."""
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, col)."""
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    result: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        coords: list[Coord] = []
        while queue:
            rr, cc = queue.popleft()
            coords.append((rr, cc))
            for nr, nc in _neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        result.append(np.asarray(coords, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> Rect:
    """Inclusive bounding box, matching arc_utils.bbox_coords."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _enclosed_black_components(grid: np.ndarray) -> list[np.ndarray]:
    """Find black 4-components that cannot be reached from the canvas boundary."""
    h, w = grid.shape
    black = grid == BLACK
    outside = np.zeros(grid.shape, dtype=bool)
    queue: deque[Coord] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if black[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and black[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))

    return _components(black & ~outside)


def _extract_stamp(grid: np.ndarray) -> np.ndarray:
    """Extract the unique enclosed black component in its own square bbox."""
    holes = _enclosed_black_components(grid)
    assert len(holes) == 1, "expected exactly one boundary-inaccessible black hole"
    coords = holes[0]
    r0, c0, r1, c1 = _bbox(coords)
    side_h, side_w = r1 - r0 + 1, c1 - c0 + 1
    assert side_h == side_w and side_h % 2 == 1, "hole bbox must be an odd square"

    stamp = np.zeros((side_h, side_w), dtype=bool)
    stamp[coords[:, 0] - r0, coords[:, 1] - c0] = True

    # Every exposed edge of the negative-space component belongs to the gray wall.
    hole_cells = set(map(tuple, coords.tolist()))
    h, w = grid.shape
    for r, c in hole_cells:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in hole_cells:
                assert 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GRAY
    return stamp


def _find_bars(grid: np.ndarray) -> list[tuple[int, int, int, str]]:
    """Return (center_r, center_c, color, orientation) for all three-cell bars."""
    colored = (grid != BLACK) & (grid != GRAY)
    bars: list[tuple[int, int, int, str]] = []
    for coords in _components(colored):
        assert len(coords) == 3, "every colored object must be a three-cell bar"
        colors = {int(grid[r, c]) for r, c in coords}
        assert len(colors) == 1, "a bar must have one visible color"
        color = colors.pop()
        rows = sorted(map(int, coords[:, 0]))
        cols = sorted(map(int, coords[:, 1]))
        if len(set(rows)) == 1 and cols[1] == cols[0] + 1 and cols[2] == cols[1] + 1:
            bars.append((rows[0], cols[1], color, "H"))
        elif len(set(cols)) == 1 and rows[1] == rows[0] + 1 and rows[2] == rows[1] + 1:
            bars.append((rows[1], cols[0], color, "V"))
        else:
            raise AssertionError("colored component is not a straight consecutive three-cell bar")
    assert bars, "expected at least one orientation bar"
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the unique enclosed negative-space shape at each bar, rotated by the bar direction and recolored by its color.

    Input structure:
    - A black canvas contains one connected gray wall object enclosing exactly one black hole.
    - The hole is connected, asymmetric, and has an odd square bounding box.
    - Separate colored objects are straight three-cell horizontal or vertical bars away from the wall.

    Rule:
    - Find the black component unreachable from the canvas boundary and use its bbox-relative mask as the stamp.
    - Center the unrotated stamp on every horizontal bar; rotate it 90 degrees clockwise for every vertical bar.
    - Paint each stamp with its bar's color on a fresh black canvas, erasing the gray wall, hole, and uncovered bar cells.

    Key insight:
    - The source motif appears only as negative space; bar geometry, rather than color, selects its rotation before recoloring.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    stamp = _extract_stamp(g)
    bars = _find_bars(g)
    assert {orientation for _, _, _, orientation in bars} == {"H", "V"}, (
        "expected both horizontal and vertical orientation bars"
    )
    side = stamp.shape[0]
    radius = side // 2
    h, w = g.shape

    out = np.zeros_like(g)
    occupied = np.zeros(g.shape, dtype=bool)
    for center_r, center_c, color, orientation in bars:
        oriented = stamp if orientation == "H" else np.rot90(stamp, -1)
        r0, c0 = center_r - radius, center_c - radius
        assert 0 <= r0 and r0 + side <= h and 0 <= c0 and c0 + side <= w, "stamp exceeds canvas"
        target = occupied[r0 : r0 + side, c0 : c0 + side]
        assert not np.any(target & oriented), "stamp marks must not overlap"
        patch = out[r0 : r0 + side, c0 : c0 + side]
        patch[oriented] = int(color)
        target[oriented] = True
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _mask_has_no_symmetry(mask: np.ndarray) -> bool:
    variants = [np.rot90(mask, k) for k in (1, 2, 3)]
    variants.extend(np.fliplr(np.rot90(mask, k)) for k in range(4))
    return all(not np.array_equal(mask, other) for other in variants)


def _random_path(mask: np.ndarray, start: Coord, goal: Coord) -> None:
    """Add a randomized shortest orthogonal path, preserving connectivity."""
    r, c = start
    gr, gc = goal
    mask[r, c] = True
    while (r, c) != (gr, gc):
        steps = []
        if r < gr:
            steps.append((r + 1, c))
        elif r > gr:
            steps.append((r - 1, c))
        if c < gc:
            steps.append((r, c + 1))
        elif c > gc:
            steps.append((r, c - 1))
        r, c = random.choice(steps)
        mask[r, c] = True


def _make_stamp(side: int) -> np.ndarray | None:
    """Create a connected, full-bbox, asymmetric negative-space mask."""
    center = side // 2
    for _ in range(250):
        mask = np.zeros((side, side), dtype=bool)
        mask[center, center] = True
        goals = (
            (0, random.randint(1, side - 2)),
            (side - 1, random.randint(1, side - 2)),
            (random.randint(1, side - 2), 0),
            (random.randint(1, side - 2), side - 1),
        )
        shuffled = list(goals)
        random.shuffle(shuffled)
        for goal in shuffled:
            starts = list(map(tuple, np.argwhere(mask)))
            _random_path(mask, random.choice(starts), goal)

        target_area = random.randint(int(0.38 * side * side), int(0.58 * side * side))
        while int(mask.sum()) < target_area:
            frontier: set[Coord] = set()
            for r, c in map(tuple, np.argwhere(mask)):
                for nr, nc in _neighbors4(int(r), int(c)):
                    if 0 <= nr < side and 0 <= nc < side and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            mask[random.choice(tuple(frontier))] = True

        # The hole and the surrounding gray material must each be one component.
        wall_patch = np.ones((side + 2, side + 2), dtype=bool)
        wall_patch[1:-1, 1:-1][mask] = False
        if len(_components(mask)) != 1 or len(_components(wall_patch)) != 1:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        if not mask[center, center] or not _mask_has_no_symmetry(mask):
            continue
        if int(np.count_nonzero(mask != np.rot90(mask, -1))) < 6:
            continue
        # Every bar retains its center under the stamp but also has an uncovered cell erased.
        if np.all(mask[center, center - 1 : center + 2]):
            continue
        if np.all(mask[center - 1 : center + 2, center]):
            continue
        return mask
    return None


def _rect_options(shape: tuple[int, int], side: int) -> list[Rect]:
    h, w = shape
    return [
        (r, c, r + side - 1, c + side - 1)
        for r in range(1, h - side)
        for c in range(1, w - side)
    ]


def _rects_separated(a: Rect, b: Rect, gap: int = 1) -> bool:
    """True when rectangles have at least `gap` wholly blank rows or columns."""
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_rectangles(shape: tuple[int, int], stamp_side: int, count: int) -> tuple[Rect, list[Rect]] | None:
    """Place one wall bbox and disjoint stamp bboxes using randomized backtracking."""
    wall_side = stamp_side + 2
    wall_options = _rect_options(shape, wall_side)
    random.shuffle(wall_options)
    stamp_options = _rect_options(shape, stamp_side)

    for wall in wall_options[: min(80, len(wall_options))]:
        candidates = [rect for rect in stamp_options if _rects_separated(rect, wall)]
        random.shuffle(candidates)

        def choose(start: int, selected: list[Rect]) -> list[Rect] | None:
            if len(selected) == count:
                return selected
            # Random candidates already provide diversity; this cap keeps retries quick.
            examined = 0
            for index in range(start, len(candidates)):
                rect = candidates[index]
                if all(_rects_separated(rect, old) for old in selected):
                    result = choose(index + 1, selected + [rect])
                    if result is not None:
                        return result
                    examined += 1
                    if examined >= 120:
                        break
            return None

        stamps = choose(0, [])
        if stamps is not None:
            random.shuffle(stamps)
            return wall, stamps
    return None


def _orientation_and_colors(count: int) -> list[tuple[str, int]]:
    """Cover both branches and reuse one visible color across the two branches."""
    orientations = ["H", "H", "V", "V"]
    orientations.extend(random.choice(("H", "V")) for _ in range(count - 4))
    random.shuffle(orientations)

    shared = random.choice(BAR_COLORS)
    specs: list[tuple[str, int] | None] = [None] * count
    h_index = random.choice([i for i, ori in enumerate(orientations) if ori == "H"])
    v_index = random.choice([i for i, ori in enumerate(orientations) if ori == "V"])
    specs[h_index] = ("H", shared)
    specs[v_index] = ("V", shared)

    remaining_colors = list(set(BAR_COLORS) - {shared})
    random.shuffle(remaining_colors)
    for i, ori in enumerate(orientations):
        if specs[i] is None:
            specs[i] = (ori, remaining_colors.pop())
    return [spec for spec in specs if spec is not None]


def _render_expected(
    shape: tuple[int, int], stamp: np.ndarray, bars: list[tuple[int, int, str, int]]
) -> np.ndarray:
    out = np.zeros(shape, dtype=int)
    radius = stamp.shape[0] // 2
    for r, c, orientation, color in bars:
        oriented = stamp if orientation == "H" else np.rot90(stamp, -1)
        patch = out[r - radius : r + radius + 1, c - radius : c + radius + 1]
        assert not np.any((patch != BLACK) & oriented)
        patch[oriented] = color
    return out


def _validate_generated(
    grid: np.ndarray,
    stamp: np.ndarray,
    wall_rect: Rect,
    bars: list[tuple[int, int, str, int]],
) -> bool:
    """Check puzzle semantics, branch necessity, uniqueness, and non-degeneracy."""
    h, w = grid.shape
    if grid.ndim != 2 or max(h, w) > 30 or min(h, w) < 1:
        return False

    holes = _enclosed_black_components(grid)
    if len(holes) != 1:
        return False
    hr0, hc0, hr1, hc1 = _bbox(holes[0])
    wr0, wc0, wr1, wc1 = wall_rect
    if (hr0, hc0, hr1, hc1) != (wr0 + 1, wc0 + 1, wr1 - 1, wc1 - 1):
        return False
    observed = np.zeros_like(stamp)
    observed[holes[0][:, 0] - hr0, holes[0][:, 1] - hc0] = True
    if not np.array_equal(observed, stamp):
        return False
    if len(_components(grid == GRAY)) != 1:
        return False

    parsed = {(r, c, ori, color) for r, c, color, ori in _find_bars(grid)}
    specified = set(bars)
    if parsed != specified or len(parsed) != len(bars):
        return False
    if {ori for _, _, ori, _ in bars} != {"H", "V"}:
        return False
    # A repeated color across directions is a visible counterexample to color-based orientation.
    if not any(
        any(ori == "H" and color == candidate for _, _, ori, color in bars)
        and any(ori == "V" and color == candidate for _, _, ori, color in bars)
        for candidate in BAR_COLORS
    ):
        return False

    expected = _render_expected(grid.shape, stamp, bars)
    if int(np.count_nonzero(expected)) != len(bars) * int(stamp.sum()):
        return False
    try:
        actual = main(grid)
    except (AssertionError, ValueError, IndexError):
        return False
    if not np.array_equal(actual, expected):
        return False
    if np.any(actual == GRAY) or np.array_equal(actual, grid):
        return False

    # Both orientation branches must affect the answer; neither fixed rotation can substitute.
    all_horizontal = [(r, c, "H", color) for r, c, _, color in bars]
    all_vertical = [(r, c, "V", color) for r, c, _, color in bars]
    if np.array_equal(expected, _render_expected(grid.shape, stamp, all_horizontal)):
        return False
    if np.array_equal(expected, _render_expected(grid.shape, stamp, all_vertical)):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    stamp_side = random.choice((5, 7))
    if stamp_side == 5:
        shape = (random.randint(24, 30), random.randint(24, 30))
        bar_count = random.choice((4, 5))
    else:
        shape = (random.randint(27, 30), random.randint(27, 30))
        bar_count = 4

    stamp = _make_stamp(stamp_side)
    if stamp is None:
        return None
    placement = _place_rectangles(shape, stamp_side, bar_count)
    if placement is None:
        return None
    wall_rect, stamp_rects = placement

    grid = np.zeros(shape, dtype=int)
    wr0, wc0, wr1, wc1 = wall_rect
    grid[wr0 : wr1 + 1, wc0 : wc1 + 1] = GRAY
    interior = grid[wr0 + 1 : wr1, wc0 + 1 : wc1]
    interior[stamp] = BLACK

    bars: list[tuple[int, int, str, int]] = []
    for rect, (orientation, color) in zip(stamp_rects, _orientation_and_colors(bar_count)):
        r0, c0, r1, c1 = rect
        center_r, center_c = (r0 + r1) // 2, (c0 + c1) // 2
        if orientation == "H":
            grid[center_r, center_c - 1 : center_c + 2] = color
        else:
            grid[center_r - 1 : center_r + 2, center_c] = color
        bars.append((center_r, center_c, orientation, color))

    if not _validate_generated(grid, stamp, wall_rect, bars):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied valid instance with both bar orientations and semantic counterexamples."""
    for _ in range(400):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid negative-space stamp instance")
