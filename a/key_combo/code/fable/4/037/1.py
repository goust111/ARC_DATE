from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    """Return the most frequent color without assigning meaning to its id."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """Return 4-connected components as (N, 2) arrays of (row, column)."""
    h, w = mask.shape
    unseen = mask.astype(bool).copy()
    result: list[np.ndarray] = []
    for sr, sc in np.argwhere(unseen):
        sr, sc = int(sr), int(sc)
        if not unseen[sr, sc]:
            continue
        unseen[sr, sc] = False
        queue = deque([(sr, sc)])
        cells: list[Coord] = []
        while queue:
            r, c = queue.popleft()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < w and unseen[rr, cc]:
                    unseen[rr, cc] = False
                    queue.append((rr, cc))
        result.append(np.asarray(cells, dtype=int))
    return result


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_solid_rectangle(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    return len(coords) == (r1 - r0 + 1) * (c1 - c0 + 1)


def _motion(coords: np.ndarray) -> Coord:
    """Use the mover's unique long axis and point it toward r=c."""
    r0, c0, r1, c1 = _bbox(coords)
    height, width = r1 - r0 + 1, c1 - c0 + 1
    assert height != width, "moving objects need a unique long axis"

    offsets = coords[:, 1] - coords[:, 0]
    assert np.all(offsets > 0) or np.all(offsets < 0), "a mover must lie on one side of the diagonal"
    above = bool(np.all(offsets > 0))
    if width > height:
        return (0, -1) if above else (0, 1)
    return (1, 0) if above else (-1, 0)


def _slide(
    coords: np.ndarray,
    direction: Coord,
    forbidden: set[Coord],
    shape: tuple[int, int],
) -> tuple[np.ndarray, int, set[Coord]]:
    """Slide until the next unit step intersects a forbidden cell or the edge."""
    h, w = shape
    dr, dc = direction
    shift = 0
    hit: set[Coord] = set()
    while True:
        nxt = coords + np.array(((shift + 1) * dr, (shift + 1) * dc), dtype=int)
        cells = {(int(r), int(c)) for r, c in nxt}
        outside = {(r, c) for r, c in cells if not (0 <= r < h and 0 <= c < w)}
        hit = outside | (cells & forbidden)
        if hit:
            break
        shift += 1
        assert shift <= max(h, w), "motion failed to encounter its diagonal target"
    final = coords + np.array((shift * dr, shift * dc), dtype=int)
    return final, shift, hit


def _parse(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] == g.shape[1], "expected a square 2D grid"
    n = g.shape[0]
    bg = _background(g)
    diag = {(i, i) for i in range(n)}

    foreground = [int(v) for v in np.unique(g) if int(v) != bg]
    line_colors = [v for v in foreground if {(int(r), int(c)) for r, c in np.argwhere(g == v)} == diag]
    assert len(line_colors) == 1, "expected one complete main-diagonal marker line"
    line_color = line_colors[0]

    object_colors = [v for v in foreground if v != line_color]
    assert len(object_colors) == 2, "expected one anchor color and one moving color"
    by_color = {v: _components(g == v) for v in object_colors}
    anchor_candidates = [v for v in object_colors if by_color[v] and all(_is_solid_rectangle(x) for x in by_color[v])]
    mover_candidates = [v for v in object_colors if by_color[v] and all(not _is_solid_rectangle(x) for x in by_color[v])]
    assert len(anchor_candidates) == len(mover_candidates) == 1, "rectangle structure must distinguish anchors from movers"
    anchor_color, mover_color = anchor_candidates[0], mover_candidates[0]
    anchors, movers = by_color[anchor_color], by_color[mover_color]
    assert 1 <= len(anchors) == len(movers) <= 2, "expected matching counts of one or two anchors and movers"

    anchor_signs = {int(np.sign(int(c) - int(r))) for obj in anchors for r, c in obj}
    mover_signs = {int(np.sign(int(c) - int(r))) for obj in movers for r, c in obj}
    assert len(anchor_signs) == len(mover_signs) == 1
    assert 0 not in anchor_signs and anchor_signs == {-next(iter(mover_signs))}, "roles must occupy opposite sides"
    return g, bg, line_color, anchor_color, mover_color, anchors, movers, diag


def main(grid: np.ndarray) -> np.ndarray:
    """Slide oriented objects toward the main diagonal, stopping at invisible mirrored anchors.

    Input structure:
    - A square grid has a single-color marker on every cell of the main diagonal r=c.
    - One side contains one or two separate solid rectangular anchors; the other side
      contains the same number of monochrome non-rectangular movers of another color.
    - Each mover has a unique horizontal or vertical long axis, visibly specifying its
      slide axis, and its path contains no drawn obstacle.

    Rule:
    - Reflect every anchor across r=c by swapping each (row, column) coordinate.
    - Move each object along its long axis in the sign that approaches r=c, preserving
      its shape, until its next unit step would overlap the diagonal or any reflected
      anchor cell. Erase its old position and never draw the reflected obstacles.
    - Keep the diagonal and all real anchors unchanged.

    Key insight:
    - The apparent empty space contains collision walls obtained only by transposing
      the opposite-side anchors; some movers hit these virtual walls while others reach
      the visible diagonal.
    """
    g, bg, _line_color, _anchor_color, mover_color, anchors, movers, diag = _parse(grid)
    virtual = {(int(c), int(r)) for obj in anchors for r, c in obj}
    forbidden = diag | virtual

    out = g.copy()
    out[out == mover_color] = bg
    occupied: set[Coord] = set()
    for mover in movers:
        final, shift, _hit = _slide(mover, _motion(mover), forbidden, g.shape)
        assert shift > 0, "each mover must advance"
        cells = {(int(r), int(c)) for r, c in final}
        assert not cells & occupied, "final moving objects must not overlap"
        assert all(int(out[r, c]) == bg for r, c in cells), "a mover must land only on visible background"
        out[final[:, 0], final[:, 1]] = mover_color
        occupied |= cells
    return out


# ------------------------------ generator ------------------------------


def _random_long_axis_shape(axis: str) -> np.ndarray:
    """Create a connected, visibly irregular object with one unique long axis."""
    short = random.choice((2, 3))
    long = random.randint(short + 2, min(6, short + 4))
    horizontal = np.zeros((short, long), dtype=bool)
    spine = random.randrange(short)
    horizontal[spine, :] = True
    connector = random.randrange(long)
    for r in range(short):
        if r == spine:
            continue
        run = random.randint(1, max(1, long - 2))
        start_min = max(0, connector - run + 1)
        start_max = min(connector, long - run)
        start = random.randint(start_min, start_max)
        horizontal[r, start : start + run] = True
    assert horizontal.any() and not horizontal.all()
    return horizontal if axis == "horizontal" else horizontal.T


def _rect_cells(top: int, left: int, height: int = 2, width: int = 2) -> set[Coord]:
    return {(r, c) for r in range(top, top + height) for c in range(left, left + width)}


def _virtual_hit_layout(n: int) -> tuple[set[Coord], np.ndarray, np.ndarray, str]:
    """Construct a horizontal mover whose next step meets a 2x2 virtual wall."""
    shape = _random_long_axis_shape("horizontal")
    local = np.argwhere(shape)
    height, width = shape.shape

    top = random.randint(2, 4)
    left = top + random.randint(7, 8)
    max_shift = n - left - width - 1
    shift = random.randint(3, min(6, max_shift))

    final = local + np.array((top, left), dtype=int)
    start = final + np.array((0, shift), dtype=int)

    # Every generated shape has a full-width spine.  Put the virtual rectangle
    # immediately beyond that spine's left frontier: final cells remain clear,
    # while the next one-cell move is guaranteed to collide.
    spine_rows = np.flatnonzero(np.all(shape, axis=1))
    wall_top = top + int(random.choice(spine_rows.tolist()))
    virtual = _rect_cells(wall_top, left - 2)
    return virtual, start, final, "virtual"


def _line_hit_layout(n: int) -> tuple[set[Coord], np.ndarray, np.ndarray, str]:
    """Construct a vertical mover one clear step away from the diagonal."""
    shape = _random_long_axis_shape("vertical")
    local = np.argwhere(shape)
    _height, width = shape.shape

    left = random.randint(n - width - 3, n - width - 2)
    min_local_offset = int(np.min(local[:, 1] - local[:, 0]))
    top = left + min_local_offset - 1
    shift = random.randint(3, min(6, top - 2))

    final = local + np.array((top, left), dtype=int)
    start = final - np.array((shift, 0), dtype=int)

    # This anchor supplies the required matching real/virtual object count but
    # lies in a separate lane, so this mover reaches the visible diagonal.
    wall_top = random.randint(8, 10)
    virtual = _rect_cells(wall_top, wall_top + 4)
    return virtual, start, final, "line"


def _validate_layout(
    inp: np.ndarray,
    expected: np.ndarray,
    virtual: set[Coord],
    specs: list[tuple[np.ndarray, np.ndarray, str]],
) -> None:
    """Fail fast if a direct construction ever violates its geometric contract."""
    diag = {(i, i) for i in range(inp.shape[0])}
    for start, final, wanted in specs:
        got, shift, hit = _slide(start, _motion(start), diag | virtual, inp.shape)
        reason = "virtual" if hit & virtual else "line" if hit & diag else "edge"
        assert reason == wanted and shift >= 3 and np.array_equal(got, final)
        if wanted == "virtual":
            no_wall, no_wall_shift, _ = _slide(start, _motion(start), diag, inp.shape)
            assert no_wall_shift > shift and not np.array_equal(no_wall, final)
    assert not np.array_equal(inp, expected)
    assert np.array_equal(main(inp), expected)


def generate_input() -> np.ndarray:
    """Generate varied one/two-mover scenes with explicit semantic counterexamples."""
    count = random.choice((1, 2))
    n = random.randint(26 if count == 2 else 22, 30)

    layouts = []
    if count == 2:
        layouts = [_virtual_hit_layout(n), _line_hit_layout(n)]
    elif random.getrandbits(1):
        layouts = [_virtual_hit_layout(n)]
    else:
        layouts = [_line_hit_layout(n)]

    virtual: set[Coord] = set()
    specs: list[tuple[np.ndarray, np.ndarray, str]] = []
    for wall, start, final, kind in layouts:
        assert not virtual & wall
        virtual |= wall
        specs.append((start, final, kind))

    # Transposition swaps the occupied side and the horizontal/vertical roles
    # without changing reflection across the main diagonal.
    if random.getrandbits(1):
        virtual = {(c, r) for r, c in virtual}
        specs = [(start[:, ::-1].copy(), final[:, ::-1].copy(), kind) for start, final, kind in specs]

    bg, line_color, anchor_color, mover_color = random.sample(range(10), 4)
    inp = np.full((n, n), bg, dtype=int)
    np.fill_diagonal(inp, line_color)
    for r, c in virtual:
        inp[c, r] = anchor_color
    for start, _final, _kind in specs:
        inp[start[:, 0], start[:, 1]] = mover_color

    expected = inp.copy()
    expected[expected == mover_color] = bg
    for _start, final, _kind in specs:
        expected[final[:, 0], final[:, 1]] = mover_color

    _validate_layout(inp, expected, virtual, specs)
    return inp
