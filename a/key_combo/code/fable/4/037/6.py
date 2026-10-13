from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_BG = 0
_COLORS = tuple(range(1, 10))
_SIDES = ("top", "bottom", "left", "right")
_DIRECTIONS = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
_direction_bag: list[str] = []


def _bbox(cells: set[Coord]) -> BBox:
    assert cells, "expected nonempty coordinates"
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter(box: BBox) -> set[Coord]:
    r0, c0, r1, c1 = box
    assert r0 < r1 and c0 < c1, "a frame needs positive height and width"
    return (
        {(r0, c) for c in range(c0, c1 + 1)}
        | {(r1, c) for c in range(c0, c1 + 1)}
        | {(r, c0) for r in range(r0, r1 + 1)}
        | {(r, c1) for r in range(r0, r1 + 1)}
    )


def _tab_side(tab: Coord, box: BBox) -> str | None:
    """Return the outward side of a non-corner tab, or None."""
    tr, tc = tab
    r0, c0, r1, c1 = box
    if tr == r0 - 1 and c0 < tc < c1:
        return "top"
    if tr == r1 + 1 and c0 < tc < c1:
        return "bottom"
    if tc == c0 - 1 and r0 < tr < r1:
        return "left"
    if tc == c1 + 1 and r0 < tr < r1:
        return "right"
    return None


def _find_tabbed_frame(grid: np.ndarray) -> tuple[int, set[Coord], BBox, Coord, str]:
    """Identify the unique color that is exactly a hollow rectangle plus one tab."""
    candidates = []
    for color in sorted(set(map(int, grid.flat)) - {_BG}):
        cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
        for tab in cells:
            ring = cells - {tab}
            if len(ring) < 8:
                continue
            box = _bbox(ring)
            r0, c0, r1, c1 = box
            if r1 - r0 < 2 or c1 - c0 < 2:
                continue
            if ring != _perimeter(box):
                continue
            side = _tab_side(tab, box)
            if side is not None:
                candidates.append((color, cells, box, tab, side))

    assert len(candidates) == 1, "expected one uniquely identifiable tabbed hollow frame"
    return candidates[0]


def _shift_cells(cells: set[Coord], dr: int, dc: int, distance: int) -> set[Coord]:
    return {(r + dr * distance, c + dc * distance) for r, c in cells}


def _shift_box(box: BBox, dr: int, dc: int, distance: int) -> BBox:
    r0, c0, r1, c1 = box
    return (
        r0 + dr * distance,
        c0 + dc * distance,
        r1 + dr * distance,
        c1 + dc * distance,
    )


def _strictly_inside(cells: set[Coord], box: BBox) -> bool:
    r0, c0, r1, c1 = box
    return all(r0 < r < r1 and c0 < c < c1 for r, c in cells)


def _all_in_bounds(cells: set[Coord], shape: tuple[int, int]) -> bool:
    h, w = shape
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def main(grid: np.ndarray) -> np.ndarray:
    """Slide a tabbed hollow frame in the tab direction until its interior first strictly contains the whole pixel cluster.

    Input structure:
    - A black-background grid contains one one-cell-thick hollow rectangular frame
      with one same-colored pixel tab immediately outside a non-corner side cell.
    - Every pixel of the other foreground color belongs to one sparse cluster ahead
      of the tab, and the cluster bbox is smaller than the frame interior.

    Rule:
    - Move the complete frame, including its tab, one cell at a time in the outward
      tab direction. Ignore intersections with cluster pixels during the motion.
    - Stop at the first translation where every cluster pixel is strictly inside the
      hollow interior and none lies on the ring; erase the old frame, draw the moved
      frame and tab, and leave every cluster pixel fixed.

    Key insight:
    - The tab determines only the direction. The stopping distance comes from full
      containment of the entire cluster, so contact with its first pixel is too early.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids may not exceed 30 in either dimension"
    assert int(np.count_nonzero(g == _BG)) > g.size // 2, "black must be the background"

    foreground = sorted(set(map(int, g.flat)) - {_BG})
    assert len(foreground) == 2, "expected exactly two foreground colors"

    frame_color, frame_cells, frame_box, _tab, side = _find_tabbed_frame(g)
    cluster_color = next(color for color in foreground if color != frame_color)
    cluster_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == cluster_color)}
    assert len(cluster_cells) >= 2, "expected a multi-pixel cluster"

    cr0, cc0, cr1, cc1 = _bbox(cluster_cells)
    r0, c0, r1, c1 = frame_box
    assert cr1 - cr0 + 1 < r1 - r0, "cluster bbox must be smaller than interior height"
    assert cc1 - cc0 + 1 < c1 - c0, "cluster bbox must be smaller than interior width"

    if side == "top":
        assert cr1 < r0 and c0 < cc0 <= cc1 < c1, "cluster must lie ahead in the frame's column lane"
    elif side == "bottom":
        assert r1 < cr0 and c0 < cc0 <= cc1 < c1, "cluster must lie ahead in the frame's column lane"
    elif side == "left":
        assert cc1 < c0 and r0 < cr0 <= cr1 < r1, "cluster must lie ahead in the frame's row lane"
    else:
        assert c1 < cc0 and r0 < cr0 <= cr1 < r1, "cluster must lie ahead in the frame's row lane"

    dr, dc = _DIRECTIONS[side]
    stop = None
    for distance in range(1, max(g.shape) + 1):
        moved_cells = _shift_cells(frame_cells, dr, dc, distance)
        assert _all_in_bounds(moved_cells, g.shape), "frame left the grid before containing the cluster"
        moved_box = _shift_box(frame_box, dr, dc, distance)
        if _strictly_inside(cluster_cells, moved_box):
            stop = distance
            break
    assert stop is not None, "the frame never fully contains the cluster"

    destination = _shift_cells(frame_cells, dr, dc, stop)
    assert destination.isdisjoint(cluster_cells), "strict containment must keep cluster pixels off the frame"

    out = g.copy()
    for r, c in frame_cells:
        out[r, c] = _BG
    assert all(int(out[r, c]) == _BG for r, c in destination), "moved frame must land only on background"
    for r, c in destination:
        out[r, c] = frame_color
    assert all(int(out[r, c]) == cluster_color for r, c in cluster_cells), "cluster pixels must not move"
    return out


# ------------------------------ generator ------------------------------


def _next_side() -> str:
    """Balance the four visible direction branches across consecutive examples."""
    if not _direction_bag:
        _direction_bag.extend(_SIDES)
        random.shuffle(_direction_bag)
    return _direction_bag.pop()


def _component_count(cells: set[Coord]) -> int:
    remaining = set(cells)
    count = 0
    while remaining:
        count += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for nxt in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if nxt in remaining:
                    remaining.remove(nxt)
                    stack.append(nxt)
    return count


def _scatter_in_bbox(height: int, width: int) -> set[Coord] | None:
    """Make a sparse, not-necessarily-connected cluster whose bbox uses both axes."""
    cells = [(r, c) for r in range(height) for c in range(width)]
    max_count = min(8, len(cells) - 1, max(3, (len(cells) + 1) // 2))
    if max_count < 3:
        return None
    for _ in range(100):
        count = random.randint(3, max_count)
        chosen = set(random.sample(cells, count))
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if min(rows) != 0 or max(rows) != height - 1:
            continue
        if min(cols) != 0 or max(cols) != width - 1:
            continue
        if _component_count(chosen) < 2:
            continue
        return chosen
    return None


def _render_expected(
    inp: np.ndarray,
    frame_cells: set[Coord],
    frame_color: int,
    direction: tuple[int, int],
    distance: int,
) -> np.ndarray:
    out = inp.copy()
    for r, c in frame_cells:
        out[r, c] = _BG
    dr, dc = direction
    for r, c in _shift_cells(frame_cells, dr, dc, distance):
        out[r, c] = frame_color
    return out


def _semantic_check(
    inp: np.ndarray,
    expected: np.ndarray,
    frame_cells: set[Coord],
    frame_box: BBox,
    cluster_cells: set[Coord],
    side: str,
    distance: int,
    frame_color: int,
    cluster_color: int,
) -> bool:
    """Check intended semantics independently of the solver's output equality."""
    dr, dc = _DIRECTIONS[side]
    tab_candidates = frame_cells - _perimeter(frame_box)
    if len(tab_candidates) != 1 or _tab_side(next(iter(tab_candidates)), frame_box) != side:
        return False

    if _strictly_inside(cluster_cells, frame_box):
        return False
    for step in range(1, distance):
        if _strictly_inside(cluster_cells, _shift_box(frame_box, dr, dc, step)):
            return False
    final_box = _shift_box(frame_box, dr, dc, distance)
    if not _strictly_inside(cluster_cells, final_box):
        return False

    # The ring must actually cross cluster pixels before the containment stop.
    ring = _perimeter(frame_box)
    intersections = [
        step
        for step in range(1, distance)
        if not _shift_cells(ring, dr, dc, step).isdisjoint(cluster_cells)
    ]
    if not intersections or min(intersections) >= distance:
        return False

    cr0, cc0, cr1, cc1 = _bbox(cluster_cells)
    r0, c0, r1, c1 = frame_box
    if not (cr1 - cr0 + 1 < r1 - r0 and cc1 - cc0 + 1 < c1 - c0):
        return False
    if _component_count(cluster_cells) < 2:
        return False
    if not frame_cells.isdisjoint(_shift_cells(frame_cells, dr, dc, distance)):
        return False
    if any(int(expected[r, c]) != cluster_color for r, c in cluster_cells):
        return False
    if int(np.sum(expected == frame_color)) != len(frame_cells):
        return False
    return not np.array_equal(inp, expected)


def _gen_attempt(side: str) -> np.ndarray | None:
    height = random.randint(16, 30)
    width = random.randint(16, 30)
    frame_h = random.randint(6, min(10, height - 4))
    frame_w = random.randint(6, min(10, width - 4))

    cluster_h = random.randint(2, frame_h - 3)
    cluster_w = random.randint(2, frame_w - 3)
    if cluster_h == cluster_w == 2:
        if frame_h >= 6 and random.choice((True, False)):
            cluster_h = 3
        else:
            cluster_w = 3

    vertical = side in {"top", "bottom"}
    axis_limit = height if vertical else width
    cross_limit = width if vertical else height
    frame_axis = frame_h if vertical else frame_w
    frame_cross = frame_w if vertical else frame_h
    cluster_axis = cluster_h if vertical else cluster_w
    cluster_cross = cluster_w if vertical else cluster_h

    # A large axial span plus this gap makes old and new frames visibly disjoint.
    cluster_axis = random.randint(max(2, frame_axis - 4), frame_axis - 3)
    if vertical:
        cluster_h = cluster_axis
    else:
        cluster_w = cluster_axis
    gap_low = max(1, frame_axis - cluster_axis - 1)
    gap = random.randint(gap_low, gap_low + 2)
    distance = gap + cluster_axis + 2

    cross_start_low = 1
    cross_start_high = cross_limit - frame_cross - 1
    if cross_start_low > cross_start_high:
        return None
    cross_start = random.randint(cross_start_low, cross_start_high)
    cluster_cross_offset = random.randint(1, frame_cross - cluster_cross - 1)
    cluster_cross_start = cross_start + cluster_cross_offset

    positive = side in {"bottom", "right"}
    if positive:
        axis_low = 1
        axis_high = axis_limit - 2 - distance - frame_axis
    else:
        axis_low = distance + 2
        axis_high = axis_limit - frame_axis - 1
    if axis_low > axis_high:
        return None
    axis_start = random.randint(axis_low, axis_high)

    if positive:
        cluster_axis_start = axis_start + frame_axis + gap + 1
    else:
        cluster_axis_end = axis_start - gap - 2
        cluster_axis_start = cluster_axis_end - cluster_axis + 1

    if vertical:
        r0, c0 = axis_start, cross_start
        cr0, cc0 = cluster_axis_start, cluster_cross_start
    else:
        r0, c0 = cross_start, axis_start
        cr0, cc0 = cluster_cross_start, cluster_axis_start
    r1, c1 = r0 + frame_h - 1, c0 + frame_w - 1
    frame_box = (r0, c0, r1, c1)

    ring = _perimeter(frame_box)
    if side == "top":
        tab = (r0 - 1, random.randint(c0 + 1, c1 - 1))
    elif side == "bottom":
        tab = (r1 + 1, random.randint(c0 + 1, c1 - 1))
    elif side == "left":
        tab = (random.randint(r0 + 1, r1 - 1), c0 - 1)
    else:
        tab = (random.randint(r0 + 1, r1 - 1), c1 + 1)
    frame_cells = ring | {tab}

    local_cluster = _scatter_in_bbox(cluster_h, cluster_w)
    if local_cluster is None:
        return None
    cluster_cells = {(cr0 + r, cc0 + c) for r, c in local_cluster}

    frame_color, cluster_color = random.sample(_COLORS, 2)
    inp = np.full((height, width), _BG, dtype=int)
    for r, c in frame_cells:
        inp[r, c] = frame_color
    for r, c in cluster_cells:
        if int(inp[r, c]) != _BG:
            return None
        inp[r, c] = cluster_color

    expected = _render_expected(
        inp,
        frame_cells,
        frame_color,
        _DIRECTIONS[side],
        distance,
    )
    if not _semantic_check(
        inp,
        expected,
        frame_cells,
        frame_box,
        cluster_cells,
        side,
        distance,
        frame_color,
        cluster_color,
    ):
        return None
    try:
        solved = main(inp)
    except (AssertionError, StopIteration):
        return None
    return inp if np.array_equal(solved, expected) else None


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked instance of the puzzle."""
    side = _next_side()
    for _ in range(400):
        grid = _gen_attempt(side)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid tabbed-frame input for side {side}")
