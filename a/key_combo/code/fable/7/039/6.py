from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]

_DIAGONALS: tuple[Direction, ...] = ((1, 1), (1, -1))
_CARDINALS: tuple[Direction, ...] = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _in_bounds(shape: tuple[int, int], cell: Coord) -> bool:
    r, c = cell
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _background(grid: np.ndarray) -> int:
    counts = Counter(map(int, grid.ravel()))
    largest = max(counts.values())
    choices = [color for color, count in counts.items() if count == largest]
    assert len(choices) == 1, "background must be uniquely most frequent"
    return choices[0]


def _source(grid: np.ndarray, bg: int) -> tuple[Coord, int, Direction]:
    h, w = grid.shape
    border: list[Coord] = []
    for c in range(w):
        border.extend(((0, c), (h - 1, c)))
    for r in range(1, h - 1):
        border.extend(((r, 0), (r, w - 1)))
    marked = [(r, c) for r, c in border if int(grid[r, c]) != bg]
    assert len(marked) == 1, "expected one non-background edge source"
    r, c = marked[0]
    if r == 0:
        direction = (1, 0)
    elif r == h - 1:
        direction = (-1, 0)
    elif c == 0:
        direction = (0, 1)
    else:
        direction = (0, -1)
    return (r, c), int(grid[r, c]), direction


def _mirror_lines(points: set[Coord]) -> list[tuple[Direction, tuple[Coord, ...]]]:
    """Return maximal exact two-step diagonal runs containing at least three dots."""
    lines: list[tuple[Direction, tuple[Coord, ...]]] = []
    for dr, dc in _DIAGONALS:
        for start in sorted(points):
            if (start[0] - 2 * dr, start[1] - 2 * dc) in points:
                continue
            run: list[Coord] = []
            cell = start
            while cell in points:
                run.append(cell)
                cell = (cell[0] + 2 * dr, cell[1] + 2 * dc)
            if len(run) >= 3:
                lines.append(((dr, dc), tuple(run)))
    return lines


def _dense_mirrors(points: set[Coord]) -> tuple[dict[Coord, Direction], list[tuple[Direction, tuple[Coord, ...]]]]:
    lines = _mirror_lines(points)
    slopes: dict[Coord, Direction] = {}
    for (dr, dc), run in lines:
        r, c = run[0]
        end = run[-1]
        while True:
            old = slopes.get((r, c))
            assert old is None or old == (dr, dc), "crossing mirror segments are ambiguous"
            slopes[(r, c)] = (dr, dc)
            if (r, c) == end:
                break
            r += dr
            c += dc
    return slopes, lines


def _reflect(direction: Direction, slope: Direction) -> Direction:
    dr, dc = direction
    if slope == (1, 1):       # backslash mirror
        return dc, dr
    assert slope == (1, -1)  # slash mirror
    return -dc, -dr


def _trace(
    shape: tuple[int, int],
    source: Coord,
    initial: Direction,
    mirrors: dict[Coord, Direction],
) -> tuple[list[Coord], list[Coord]]:
    """Trace cardinal beam cells and collision cells, rejecting optical loops."""
    r, c = source
    dr, dc = initial
    cells = [source]
    collisions: list[Coord] = []
    states: set[tuple[int, int, int, int]] = set()
    while True:
        r += dr
        c += dc
        if not _in_bounds(shape, (r, c)):
            return cells, collisions
        state = (r, c, dr, dc)
        assert state not in states, "beam entered a reflection loop"
        states.add(state)
        cells.append((r, c))
        if (r, c) in mirrors:
            collisions.append((r, c))
            dr, dc = _reflect((dr, dc), mirrors[(r, c)])


def main(grid: np.ndarray) -> np.ndarray:
    """Densify sparse diagonal mirrors, then draw the edge-source beam reflected between them.

    Input structure:
    - A uniquely most-common background surrounds one single-pixel source on an edge.
    - All other foreground pixels share one mirror color: separated two-step 45-degree
      dot runs contain at least three dots, while a few isolated dots are distractors.

    Rule:
    - Fill every maximal two-step diagonal run of at least three collinear dots into
      a continuous 45-degree mirror. Send the source inward perpendicular to its edge;
      at each dense mirror cell turn it 90 degrees according to that mirror's slope.
      Paint only traversed background cells in the source color, preserving mirrors
      and isolated distractor dots, and stop when the ray leaves the grid.

    Key insight:
    - Reflection uses the completed diagonal segments, not merely the visible dots;
      an isolated same-color dot remains unchanged and has no optical effect.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= max(g.shape) <= 30, "expected an ARC-sized 2D grid"
    bg = _background(g)
    source, source_color, initial = _source(g, bg)
    foreground = set(map(int, g.ravel())) - {bg, source_color}
    assert len(foreground) == 1, "expected one shared mirror/distractor color"
    mirror_color = foreground.pop()

    visible_points = {tuple(map(int, rc)) for rc in np.argwhere(g == mirror_color)}
    mirrors, lines = _dense_mirrors(visible_points)
    assert lines, "expected at least one qualifying sparse mirror segment"

    out = g.copy()
    for r, c in mirrors:
        out[r, c] = mirror_color

    path, _ = _trace(g.shape, source, initial, mirrors)
    for r, c in path:
        if int(g[r, c]) == bg and (r, c) not in mirrors:
            out[r, c] = source_color
    return out


# ------------------------------ generator ------------------------------

_PALETTE = tuple(range(10))
_MAX_ATTEMPTS = 600


def _perpendicular(direction: Direction) -> tuple[Direction, Direction]:
    dr, dc = direction
    return ((dc, -dr), (-dc, dr))


def _slope_for(incoming: Direction, outgoing: Direction) -> Direction:
    for slope in _DIAGONALS:
        if _reflect(incoming, slope) == outgoing:
            return slope
    raise AssertionError("turn is not a 45-degree reflection")


def _continuous_segment(turn: Coord, slope: Direction, hidden_hit: bool) -> tuple[set[Coord], set[Coord]]:
    dr, dc = slope
    if hidden_hit:
        offsets = (-3, -1, 1, 3)
    else:
        half = random.choice((2, 4))
        offsets = tuple(range(-half, half + 1, 2))
    visible = {(turn[0] + k * dr, turn[1] + k * dc) for k in offsets}
    lo, hi = min(offsets), max(offsets)
    dense = {(turn[0] + k * dr, turn[1] + k * dc) for k in range(lo, hi + 1)}
    return visible, dense


def _segments_separated(segments: list[set[Coord]]) -> bool:
    for i, first in enumerate(segments):
        for second in segments[i + 1 :]:
            if min(max(abs(a[0] - b[0]), abs(a[1] - b[1])) for a in first for b in second) < 3:
                return False
    return True


def _make_turns(shape: tuple[int, int], count: int) -> tuple[Coord, Direction, list[Coord], list[Direction]] | None:
    h, w = shape
    margin = 5
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        source = (0, random.randint(margin, w - margin - 1))
        initial = (1, 0)
        first = (random.randint(margin, h - margin - 1), source[1])
    elif side == "bottom":
        source = (h - 1, random.randint(margin, w - margin - 1))
        initial = (-1, 0)
        first = (random.randint(margin, h - margin - 1), source[1])
    elif side == "left":
        source = (random.randint(margin, h - margin - 1), 0)
        initial = (0, 1)
        first = (source[0], random.randint(margin, w - margin - 1))
    else:
        source = (random.randint(margin, h - margin - 1), w - 1)
        initial = (0, -1)
        first = (source[0], random.randint(margin, w - margin - 1))

    turns = [first]
    slopes: list[Direction] = []
    incoming = initial
    current = first
    for index in range(count):
        choices: list[tuple[Direction, Coord | None]] = []
        for outgoing in _perpendicular(incoming):
            if index == count - 1:
                choices.append((outgoing, None))
                continue
            for distance in range(5, 10):
                nxt = (current[0] + distance * outgoing[0], current[1] + distance * outgoing[1])
                if margin <= nxt[0] < h - margin and margin <= nxt[1] < w - margin:
                    choices.append((outgoing, nxt))
        if not choices:
            return None
        outgoing, nxt = random.choice(choices)
        slopes.append(_slope_for(incoming, outgoing))
        if nxt is not None:
            turns.append(nxt)
            current = nxt
            incoming = outgoing
    return source, initial, turns, slopes


def _add_distractors(
    grid: np.ndarray,
    mirror_color: int,
    visible: set[Coord],
    intended_dense: set[Coord],
    path: list[Coord],
) -> set[Coord] | None:
    h, w = grid.shape
    distractors: set[Coord] = set()

    def isolated(cell: Coord) -> bool:
        occupied = visible | distractors
        return all(max(abs(cell[0] - r), abs(cell[1] - c)) >= 2 for r, c in occupied)

    on_path = [
        cell for cell in path[2:-2]
        if cell not in intended_dense and 1 <= cell[0] < h - 1 and 1 <= cell[1] < w - 1 and isolated(cell)
    ]
    random.shuffle(on_path)
    for cell in on_path:
        trial = visible | {cell}
        try:
            dense, _ = _dense_mirrors(trial)
        except AssertionError:
            continue
        if set(dense) == intended_dense:
            distractors.add(cell)
            break
    if not distractors:
        return None

    target = random.randint(1, 3)
    candidates = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(candidates)
    for cell in candidates:
        if len(distractors) >= target:
            break
        if cell in intended_dense or cell in path or not isolated(cell):
            continue
        trial = visible | distractors | {cell}
        try:
            dense, _ = _dense_mirrors(trial)
        except AssertionError:
            continue
        if set(dense) == intended_dense:
            distractors.add(cell)
    for r, c in distractors:
        grid[r, c] = mirror_color
    return distractors


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(19, 27)
    w = random.randint(19, 27)
    shape = (h, w)
    turn_count = random.randint(2, 4)
    plan = _make_turns(shape, turn_count)
    if plan is None:
        return None
    source, initial, turns, slopes = plan

    hidden = [random.random() < 0.6 for _ in turns]
    hidden[random.randrange(len(hidden))] = True
    visible_segments: list[set[Coord]] = []
    dense_segments: list[set[Coord]] = []
    for turn, slope, hide in zip(turns, slopes, hidden):
        sparse, dense = _continuous_segment(turn, slope, hide)
        visible_segments.append(sparse)
        dense_segments.append(dense)
    if not _segments_separated(dense_segments):
        return None

    visible = set().union(*visible_segments)
    intended_dense = set().union(*dense_segments)
    try:
        found, lines = _dense_mirrors(visible)
    except AssertionError:
        return None
    if set(found) != intended_dense or len(lines) != turn_count:
        return None

    path, collisions = _trace(shape, source, initial, found)
    if collisions != turns or len(set(path)) != len(path):
        return None

    sparse_slopes: dict[Coord, Direction] = {}
    for segment, slope in zip(visible_segments, slopes):
        for cell in segment:
            sparse_slopes[cell] = slope
    shortcut_path, shortcut_collisions = _trace(shape, source, initial, sparse_slopes)
    if shortcut_path == path or shortcut_collisions == collisions:
        return None

    bg, mirror_color, source_color = random.sample(_PALETTE, 3)
    grid = np.full(shape, bg, dtype=int)
    for r, c in visible:
        grid[r, c] = mirror_color
    grid[source] = source_color
    distractors = _add_distractors(grid, mirror_color, visible, intended_dense, path)
    if distractors is None:
        return None

    # Semantic checks: exact recovery, every mirror matters, hidden hit changes the ray,
    # and an isolated dot lying on the ray is preserved rather than treated as a mirror.
    all_points = {tuple(map(int, rc)) for rc in np.argwhere(grid == mirror_color)}
    try:
        recovered, recovered_lines = _dense_mirrors(all_points)
    except AssertionError:
        return None
    if set(recovered) != intended_dense or len(recovered_lines) != turn_count:
        return None
    if not all(turn in intended_dense for turn in turns):
        return None
    if not any(turn not in visible for turn in turns):
        return None
    if not (distractors & set(path)):
        return None

    expected = grid.copy()
    for r, c in intended_dense:
        expected[r, c] = mirror_color
    for r, c in path:
        if int(grid[r, c]) == bg and (r, c) not in intended_dense:
            expected[r, c] = source_color
    try:
        actual = main(grid)
    except (AssertionError, ValueError):
        return None
    if not np.array_equal(actual, expected) or np.array_equal(actual, grid):
        return None
    if any(int(actual[cell]) != mirror_color for cell in distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, uniquely traceable sparse-mirror instance."""
    for _ in range(_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sparse-mirror input")
