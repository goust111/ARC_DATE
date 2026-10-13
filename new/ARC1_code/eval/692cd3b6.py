from collections import Counter, deque

import numpy as np

from arc_utils import Color, get_bg, in_bounds


def _sign(n: int) -> int:
    if n > 0:
        return 1
    if n < 0:
        return -1
    return 0


def _reachable(
    g: np.ndarray,
    bg: int,
    seed: tuple[int, int],
    moves: list[tuple[int, int]],
    bounds: tuple[int, int, int, int],
) -> np.ndarray:
    r0, c0, r1, c1 = bounds

    def valid(r: int, c: int) -> bool:
        return r0 <= r <= r1 and c0 <= c <= c1 and g[r, c] == bg

    assert valid(*seed)
    seen = np.zeros(g.shape, dtype=bool)
    seen[seed] = True
    q = deque([seed])
    while q:
        r, c = q.popleft()
        for dr, dc in moves:
            rr, cc = r + dr, c + dc
            if valid(rr, cc) and not seen[rr, cc]:
                seen[rr, cc] = True
                q.append((rr, cc))
    return seen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint yellow the background cells on the unobstructed shortest monotone corridor between two doorway frames.

    Input structure
    - The grid has a background, two small frame objects in one foreground color, and one center marker in each frame.
    - Each marker has exactly one background 4-neighbor, the doorway gap; the next cell outward is the exterior doorway cell.

    Rule
    - Treat all existing foreground cells as blockers.
    - Between the two exterior doorway cells, allow only row/column steps that move toward the other exterior cell.
    - Paint every background cell that lies on some such path from one exterior cell to the other, plus the two doorway gaps, yellow.

    Key insight
    - The output is not a single chosen route: it is the union of all shortest monotone paths that can pass between the two frames.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    counts = Counter(map(int, g.ravel()))
    fg_colors = [color for color in counts if color != bg]
    assert len(fg_colors) == 2

    marker_colors = [color for color in fg_colors if counts[color] == 2]
    assert len(marker_colors) == 1
    marker_color = marker_colors[0]
    frame_colors = [color for color in fg_colors if color != marker_color]
    assert len(frame_colors) == 1
    frame_color = frame_colors[0]

    doors: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r0, c0 in zip(*np.where(g == marker_color)):
        r, c = int(r0), int(c0)
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = r + dr, c + dc
            assert in_bounds(g.shape, rr, cc)
            neighbors.append((int(g[rr, cc]), dr, dc, rr, cc))

        gaps = [item for item in neighbors if item[0] == bg]
        frame_neighbors = [item for item in neighbors if item[0] == frame_color]
        assert len(gaps) == 1
        assert len(frame_neighbors) == 3

        _, dr, dc, gap_r, gap_c = gaps[0]
        ext_r, ext_c = r + 2 * dr, c + 2 * dc
        assert in_bounds(g.shape, ext_r, ext_c)
        assert g[ext_r, ext_c] == bg
        doors.append(((gap_r, gap_c), (ext_r, ext_c)))

    assert len(doors) == 2
    start = doors[0][1]
    end = doors[1][1]
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    assert row_step != 0 or col_step != 0

    moves = []
    if row_step:
        moves.append((row_step, 0))
    if col_step:
        moves.append((0, col_step))

    r0, r1 = sorted([start[0], end[0]])
    c0, c1 = sorted([start[1], end[1]])
    bounds = (r0, c0, r1, c1)
    forward = _reachable(g, bg, start, moves, bounds)
    backward = _reachable(g, bg, end, [(-dr, -dc) for dr, dc in moves], bounds)
    path = forward & backward
    assert path[start]
    assert path[end]

    out = g.copy()
    out[path] = Color.YELLOW
    for gap, _ in doors:
        assert g[gap] == bg
        out[gap] = Color.YELLOW
    return out

# --------------- generator (o5x) ---------------

# Colors used by the fixed 692cd3b6 object/corridor vocabulary.
GEN_BG, GEN_FRAME, GEN_MARKER, GEN_FILL = Color.BLACK, Color.RED, Color.GREY, Color.YELLOW

# Canvas and frame geometry.
GEN_GRID_SHAPE = (15, 15)
GEN_FRAME_RADIUS, GEN_EXTERIOR_STEP = 1, 2
GEN_DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Marker center sampling bounds and required diagonal separation.
GEN_TOP_ROWS, GEN_BOT_ROWS = (2, 5), (9, 12)
GEN_LEFT_COLS, GEN_RIGHT_COLS = (2, 5), (9, 12)
GEN_MIN_SEP = 5

# Sampling controls and retry budget.
GEN_TL_BR_ORIENTATION, GEN_ORIENTATION_RANGE = 0, (0, 1)
GEN_MAX_ATTEMPTS = 300


def _gen_randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _gen_choose_centers() -> list[tuple[int, int]]:
    top_r = _gen_randint(*GEN_TOP_ROWS)
    bot_r = _gen_randint(max(GEN_BOT_ROWS[0], top_r + GEN_MIN_SEP), GEN_BOT_ROWS[1])

    if _gen_randint(*GEN_ORIENTATION_RANGE) == GEN_TL_BR_ORIENTATION:
        left_c = _gen_randint(*GEN_LEFT_COLS)
        right_c = _gen_randint(max(GEN_RIGHT_COLS[0], left_c + GEN_MIN_SEP), GEN_RIGHT_COLS[1])
        return [(top_r, left_c), (bot_r, right_c)]

    right_c = _gen_randint(*GEN_RIGHT_COLS)
    left_c = _gen_randint(GEN_LEFT_COLS[0], min(GEN_LEFT_COLS[1], right_c - GEN_MIN_SEP))
    return [(top_r, right_c), (bot_r, left_c)]


def _gen_place_frame(
    grid: np.ndarray, r: int, c: int, direction: tuple[int, int]
) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    gap = (r + dr, c + dc)
    exterior = (r + GEN_EXTERIOR_STEP * dr, c + GEN_EXTERIOR_STEP * dc)

    r0, r1 = r - GEN_FRAME_RADIUS, r + GEN_FRAME_RADIUS + 1
    c0, c1 = c - GEN_FRAME_RADIUS, c + GEN_FRAME_RADIUS + 1
    grid[r0:r1, c0:c1] = GEN_FRAME
    grid[r, c] = GEN_MARKER
    grid[gap] = GEN_BG
    return gap, exterior


def _gen_expected_output(
    base: np.ndarray, doors: list[tuple[tuple[int, int], tuple[int, int]]]
) -> np.ndarray:
    start = doors[0][1]
    end = doors[1][1]
    row_step = _sign(end[0] - start[0])
    col_step = _sign(end[1] - start[1])
    assert row_step != 0 and col_step != 0

    moves = [(row_step, 0), (0, col_step)]
    r0, r1 = sorted([start[0], end[0]])
    c0, c1 = sorted([start[1], end[1]])
    bounds = (r0, c0, r1, c1)
    forward = _reachable(base, GEN_BG, start, moves, bounds)
    backward = _reachable(base, GEN_BG, end, [(-dr, -dc) for dr, dc in moves], bounds)
    path = forward & backward
    assert path[start] and path[end]

    out = base.copy()
    out[path] = GEN_FILL
    for gap, _ in doors:
        out[gap] = GEN_FILL
    return out


def _gen_attempt() -> np.ndarray | None:
    base = np.full(GEN_GRID_SHAPE, GEN_BG, dtype=int)
    doors = []

    for r, c in _gen_choose_centers():
        direction = GEN_DIRECTIONS[_gen_randint(0, len(GEN_DIRECTIONS) - 1)]
        doors.append(_gen_place_frame(base, r, c, direction))

    try:
        expected = _gen_expected_output(base, doors)
        inp = expected.copy()
        inp[inp == GEN_FILL] = GEN_BG
        if not np.array_equal(main(inp), expected):
            return None
        return inp
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 692cd3b6 input")
