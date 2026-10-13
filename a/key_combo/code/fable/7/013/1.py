from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
Direction = tuple[int, int]

DIRS: tuple[Direction, ...] = ((-1, 0), (0, 1), (1, 0), (0, -1))
STAMP_ORTHO: tuple[Direction, ...] = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
STAMP_DIAG: tuple[Direction, ...] = ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))


def _get_bg(grid: np.ndarray) -> int:
    """Return the most frequent color, following arc_utils.get_bg."""
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    """Use the canonical arc_utils (row, column) coordinate convention."""
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_border(cell: Coord, shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or r == h - 1 or c == 0 or c == w - 1


def _roles(grid: np.ndarray) -> tuple[int, int, int, Coord, Coord, Direction]:
    """Identify colors and orient the two-cell start bar from edge to interior."""
    bg = _get_bg(grid)
    foreground = [int(v) for v in np.unique(grid) if int(v) != bg]
    assert len(foreground) == 2, "expected start and blocker colors"

    counts = {color: int(np.count_nonzero(grid == color)) for color in foreground}
    start_candidates = [color for color in foreground if counts[color] == 2]
    assert len(start_candidates) == 1, "exactly one foreground color must form a two-pixel bar"
    start_color = start_candidates[0]
    blocker_color = next(color for color in foreground if color != start_color)
    assert counts[blocker_color] > 2, "blockers must be distinguishable from the start bar"

    cells = [tuple(map(int, rc)) for rc in np.argwhere(grid == start_color)]
    assert abs(cells[0][0] - cells[1][0]) + abs(cells[0][1] - cells[1][1]) == 1, (
        "start pixels must be orthogonally adjacent"
    )
    edge_cells = [cell for cell in cells if _is_border(cell, grid.shape)]
    assert len(edge_cells) == 1, "exactly one endpoint of the start bar must touch the edge"
    edge = edge_cells[0]
    inner = cells[1] if cells[0] == edge else cells[0]
    direction = (inner[0] - edge[0], inner[1] - edge[1])
    return bg, start_color, blocker_color, edge, inner, direction


def _trace_turns(grid: np.ndarray, blocker_color: int, start: Coord, direction: Direction) -> list[tuple[Coord, str]]:
    """Trace the invisible route and record each collision cell and turn hand."""
    r, c = start
    dr, dc = direction
    turns: list[tuple[Coord, str]] = []
    seen: set[tuple[int, int, int, int]] = set()

    while True:
        state = (r, c, dr, dc)
        assert state not in seen, "route must terminate rather than cycle"
        seen.add(state)

        nr, nc = r + dr, c + dc
        if not _in_bounds(grid.shape, nr, nc):
            return turns
        if int(grid[nr, nc]) != blocker_color:
            r, c = nr, nc
            continue

        # In row/column coordinates these are respectively a visual left and right turn.
        candidates: list[tuple[str, Direction]] = []
        for hand, (tdr, tdc) in (("left", (-dc, dr)), ("right", (dc, -dr))):
            tr, tc = r + tdr, c + tdc
            if _in_bounds(grid.shape, tr, tc) and int(grid[tr, tc]) != blocker_color:
                candidates.append((hand, (tdr, tdc)))
        assert len(candidates) == 1, "every collision must expose one perpendicular continuation"
        hand, (dr, dc) = candidates[0]
        turns.append(((r, c), hand))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each invisible corridor turn with a handedness-dependent cross without drawing the route.

    Input structure:
    - The most common color is background; the two foreground roles are a two-pixel
      start bar and a larger blocker corridor.
    - Exactly one endpoint of the start bar touches the grid edge, fixing the inward
      initial direction, and every blocker collision has one open perpendicular side.

    Rule:
    - Walk from the edge endpoint through the inner start pixel, continuing straight
      through non-blockers; at a blocker, take the unique open perpendicular direction,
      and stop when the next straight step leaves the grid.
    - Do not draw the route. At each collision cell, overlay an orthogonal five-cell
      cross for a left turn or a diagonal five-cell cross for a right turn, using the
      start color and changing only cells that were background in the input.

    Key insight:
    - The blocker corridor defines an invisible path, while only its turning events and
      their left/right handedness survive as orthogonal or diagonal stamp shapes.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg, start_color, blocker_color, _edge, inner, direction = _roles(g)
    turns = _trace_turns(g, blocker_color, inner, direction)

    out = g.copy()
    for (r, c), hand in turns:
        offsets = STAMP_ORTHO if hand == "left" else STAMP_DIAG
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == bg:
                out[rr, cc] = start_color
    return out


# ------------------------------ generator ------------------------------

MIN_SIZE = 13
MAX_SIZE = 23
MIN_TURNS = 3
MAX_TURNS = 6
MIN_SEGMENT = 2
MAX_SEGMENT = 6
GENERATION_ATTEMPTS = 1200


def _entry_bar(h: int, w: int) -> tuple[Coord, Coord, Direction]:
    side = random.choice(("top", "right", "bottom", "left"))
    if side == "top":
        c = random.randint(2, w - 3)
        return (0, c), (1, c), (1, 0)
    if side == "right":
        r = random.randint(2, h - 3)
        return (r, w - 1), (r, w - 2), (0, -1)
    if side == "bottom":
        c = random.randint(2, w - 3)
        return (h - 1, c), (h - 2, c), (-1, 0)
    r = random.randint(2, h - 3)
    return (r, 0), (r, 1), (0, 1)


def _steps_to_edge(cell: Coord, direction: Direction, shape: tuple[int, int]) -> int:
    r, c = cell
    dr, dc = direction
    steps = 0
    while _in_bounds(shape, r + dr, c + dc):
        r, c = r + dr, c + dc
        steps += 1
    return steps


def _sample_hands(n_turns: int) -> list[str]:
    # Both branches occur in every generated example; neither can be inferred as a rare exception.
    while True:
        hands = [random.choice(("left", "right")) for _ in range(n_turns)]
        if len(set(hands)) == 2:
            return hands


def _build_geometry(h: int, w: int, n_turns: int):
    edge, inner, direction = _entry_bar(h, w)
    hands = _sample_hands(n_turns)
    path: list[Coord] = [edge, inner]
    turns: list[tuple[Coord, str, Direction, Direction]] = []
    r, c = inner
    dr, dc = direction

    for hand in hands:
        room = _steps_to_edge((r, c), (dr, dc), (h, w))
        max_len = min(MAX_SEGMENT, room - 1)
        if max_len < MIN_SEGMENT:
            return None
        length = random.randint(MIN_SEGMENT, max_len)
        for _ in range(length):
            r, c = r + dr, c + dc
            path.append((r, c))

        old_direction = (dr, dc)
        new_direction = (-dc, dr) if hand == "left" else (dc, -dr)
        turns.append(((r, c), hand, old_direction, new_direction))
        dr, dc = new_direction

    final_room = _steps_to_edge((r, c), (dr, dc), (h, w))
    if final_room < 2:
        return None
    for _ in range(final_room):
        r, c = r + dr, c + dc
        path.append((r, c))

    if len(path) != len(set(path)):
        return None
    centers = [turn[0] for turn in turns]
    if any(max(abs(r1 - r2), abs(c1 - c2)) <= 2
           for i, (r1, c1) in enumerate(centers) for r2, c2 in centers[i + 1:]):
        return None
    if any(not (1 <= r < h - 1 and 1 <= c < w - 1) for r, c in centers):
        return None
    return edge, inner, direction, path, turns


def _expected_output(inp: np.ndarray, bg: int, start_color: int,
                     turns: list[tuple[Coord, str, Direction, Direction]]) -> np.ndarray:
    """Independent declarative stamp oracle used by the generator."""
    expected = inp.copy()
    for (r, c), hand, _old, _new in turns:
        offsets = STAMP_ORTHO if hand == "left" else STAMP_DIAG
        for dr, dc in offsets:
            rr, cc = r + dr, c + dc
            if _in_bounds(inp.shape, rr, cc) and int(inp[rr, cc]) == bg:
                expected[rr, cc] = start_color
    return expected


def _semantic_checks(inp: np.ndarray, expected: np.ndarray, bg: int, start_color: int,
                     blocker_color: int, edge: Coord, inner: Coord, direction: Direction,
                     path: list[Coord], turns: list[tuple[Coord, str, Direction, Direction]]) -> bool:
    """Check uniqueness, handedness, preservation, and anti-shortcut properties."""
    try:
        found_bg, found_start, found_blocker, found_edge, found_inner, found_dir = _roles(inp)
        traced = _trace_turns(inp, blocker_color, inner, direction)
        actual = main(inp)
    except (AssertionError, IndexError, ValueError):
        return False

    known = [(cell, hand) for cell, hand, _old, _new in turns]
    if (found_bg, found_start, found_blocker, found_edge, found_inner, found_dir) != (
        bg, start_color, blocker_color, edge, inner, direction
    ):
        return False
    if traced != known or not np.array_equal(actual, expected):
        return False
    if set(hand for _cell, hand in traced) != {"left", "right"}:
        return False

    # Only original background may change; both semantic objects are immutable.
    changed = actual != inp
    if not np.any(changed) or np.any(changed & (inp != bg)):
        return False
    if not np.array_equal(actual[inp == blocker_color], inp[inp == blocker_color]):
        return False
    if not np.array_equal(actual[inp == start_color], inp[inp == start_color]):
        return False

    # A genuine unpainted straight-path witness rules out ordinary path drawing.
    turn_neighborhood = {
        (r + dr, c + dc)
        for (r, c), hand in traced
        for dr, dc in (STAMP_ORTHO if hand == "left" else STAMP_DIAG)
    }
    witnesses = [cell for cell in path[2:] if cell not in turn_neighborhood and inp[cell] == bg]
    if not witnesses or any(actual[cell] != bg for cell in witnesses):
        return False

    # Each right turn exposes all four diagonals; each left turn exposes center,
    # incoming arm, and outgoing arm while the two collision-side arms stay blocked.
    for (r, c), hand, old, new in turns:
        if hand == "right":
            if any(inp[r + dr, c + dc] != bg for dr, dc in STAMP_DIAG):
                return False
        else:
            visible = ((0, 0), (-old[0], -old[1]), new)
            if any(inp[r + dr, c + dc] != bg for dr, dc in visible):
                return False
    return True


def _generation_attempt() -> np.ndarray | None:
    h = random.randint(MIN_SIZE, MAX_SIZE)
    w = random.randint(MIN_SIZE, MAX_SIZE)
    n_turns = random.randint(MIN_TURNS, MAX_TURNS)
    geometry = _build_geometry(h, w, n_turns)
    if geometry is None:
        return None
    edge, inner, direction, path, turns = geometry

    bg, start_color, blocker_color = random.sample(range(10), 3)
    grid = np.full((h, w), bg, dtype=int)
    grid[edge] = start_color
    grid[inner] = start_color
    path_set = set(path)

    blockers: set[Coord] = set()
    for (r, c), _hand, (odr, odc), (ndr, ndc) in turns:
        ahead = (r + odr, c + odc)
        wrong_side = (r - ndr, c - ndc)
        if not _in_bounds(grid.shape, *ahead) or not _in_bounds(grid.shape, *wrong_side):
            return None
        if ahead in path_set or wrong_side in path_set:
            return None
        blockers.update((ahead, wrong_side))
    if len(blockers) != 2 * n_turns:
        return None

    # Do not let a blocker for one collision clip another turn's local stamp.
    for br, bc in blockers:
        if any(max(abs(br - tr), abs(bc - tc)) <= 1 and (br, bc) not in {
            (tr + old[0], tc + old[1]), (tr - new[0], tc - new[1])
        } for (tr, tc), _hand, old, new in turns):
            return None
    for cell in blockers:
        grid[cell] = blocker_color

    # Sparse, remote distractors vary obstacle count without changing the corridor trace.
    remote = [
        (r, c) for r in range(h) for c in range(w)
        if grid[r, c] == bg
        and (r, c) not in path_set
        and all(abs(r - pr) + abs(c - pc) >= 2 for pr, pc in path_set)
        and all(max(abs(r - tr), abs(c - tc)) >= 2 for (tr, tc), *_ in turns)
    ]
    random.shuffle(remote)
    for cell in remote[:random.randint(2, min(10, len(remote)))]:
        grid[cell] = blocker_color

    values, counts = np.unique(grid, return_counts=True)
    if int(values[int(np.argmax(counts))]) != bg or int(np.count_nonzero(grid == blocker_color)) <= 2:
        return None

    expected = _expected_output(grid, bg, start_color, turns)
    if not _semantic_checks(
        grid, expected, bg, start_color, blocker_color, edge, inner, direction, path, turns
    ):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, oracle-checked corridor with both turn-handedness branches."""
    for _ in range(GENERATION_ATTEMPTS):
        result = _generation_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid turning-stamp puzzle")
