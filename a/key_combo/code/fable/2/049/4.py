from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, get_bg, in_bounds

Coord = tuple[int, int]
Direction = tuple[int, int]
Segment = tuple[Coord, Direction, int]

BG = Color.BLACK
HINGE = Color.GREY
START_COLORS = [color for color in Color.NOT_BLACK if color != HINGE]

MIN_SEG_LEN = 2
MAX_ATTEMPTS = 500
MAX_DIM = 22

_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("top", "L"),
    ("bottom", "R"),
    ("left", "LR"),
    ("right", "RL"),
    ("top", "LRL"),
    ("bottom", "RLR"),
    ("left", "LRRL"),
    ("right", "RLL"),
)
_GEN_COUNTER = 0


def _add(p: Coord, d: Direction, steps: int = 1) -> Coord:
    return int(p[0] + d[0] * steps), int(p[1] + d[1] * steps)


def _turn_left(d: Direction) -> Direction:
    return int(-d[1]), int(d[0])


def _turn_right(d: Direction) -> Direction:
    return int(d[1]), int(-d[0])


def _edge_distance(shape: tuple[int, int], p: Coord, d: Direction) -> int:
    h, w = map(int, shape)
    r, c = map(int, p)
    dr, dc = map(int, d)
    if (dr, dc) == (-1, 0):
        return int(r)
    if (dr, dc) == (1, 0):
        return int(h - 1 - r)
    if (dr, dc) == (0, -1):
        return int(c)
    if (dr, dc) == (0, 1):
        return int(w - 1 - c)
    raise ValueError("direction must be axis-aligned")


def _ray_cells(origin: Coord, d: Direction, length: int) -> list[Coord]:
    return [_add(origin, d, step) for step in range(1, int(length) + 1)]


def _turn_direction(shape: tuple[int, int], hinge: Coord, incoming: Direction) -> Direction | None:
    left = _turn_left(incoming)
    right = _turn_right(incoming)
    left_space = _edge_distance(shape, hinge, left)
    right_space = _edge_distance(shape, hinge, right)
    if left_space == right_space:
        return None
    return left if left_space > right_space else right


def _turn_side(shape: tuple[int, int], hinge: Coord, incoming: Direction) -> str | None:
    chosen = _turn_direction(shape, hinge, incoming)
    if chosen is None:
        return None
    return "L" if chosen == _turn_left(incoming) else "R"


def _inward_direction(shape: tuple[int, int], start: Coord) -> Direction:
    h, w = map(int, shape)
    r, c = map(int, start)
    choices: list[Direction] = []
    if r == 0:
        choices.append((1, 0))
    if r == h - 1:
        choices.append((-1, 0))
    if c == 0:
        choices.append((0, 1))
    if c == w - 1:
        choices.append((0, -1))
    assert len(choices) == 1, "start marker must lie on exactly one edge"
    return choices[0]


def _find_start(grid: np.ndarray, bg: int) -> tuple[Coord, int]:
    start_mask = (grid != int(bg)) & (grid != HINGE)
    coords = np.argwhere(start_mask)
    assert len(coords) == 1, "expected one colored edge start marker"
    r, c = map(int, coords[0])
    return (r, c), int(grid[r, c])


def _trace_records(
    grid: np.ndarray,
) -> tuple[np.ndarray, tuple[Coord, ...], tuple[Coord, ...], tuple[tuple[Coord, Direction], ...]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    assert bg == BG, "background must be black"

    start, start_color = _find_start(g, bg)
    assert int(g[start]) != HINGE
    other_colors = set(map(int, np.unique(g))) - {int(bg), int(HINGE), int(start_color)}
    assert not other_colors, "input should contain only background, one start color, and gray hinges"
    hinge_count = int((g == HINGE).sum())
    assert hinge_count >= 2, "expected used hinges plus distractor hinges"

    out = g.copy()
    direction = _inward_direction(g.shape, start)
    pos = start
    seen_states: set[tuple[Coord, Direction]] = set()
    used_hinges: list[Coord] = []
    painted: list[Coord] = []
    hinge_records: list[tuple[Coord, Direction]] = []

    while in_bounds(g.shape, *pos):
        state = (pos, direction)
        assert state not in seen_states, "ray must leave the grid instead of looping"
        seen_states.add(state)

        if int(g[pos]) == HINGE:
            hinge_records.append((pos, direction))
            new_direction = _turn_direction(g.shape, pos, direction)
            assert new_direction is not None, "hit hinges must have a unique larger side"
            direction = new_direction
            used_hinges.append(pos)
        else:
            out[pos] = start_color
            painted.append(pos)

        pos = _add(pos, direction)

    return out, tuple(used_hinges), tuple(painted), tuple(hinge_records)


def _trace_path(grid: np.ndarray) -> tuple[np.ndarray, tuple[Coord, ...], tuple[Coord, ...]]:
    out, used_hinges, painted, _hinge_records = _trace_records(grid)
    return out, used_hinges, painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace one edge-entering light ray that turns at gray hinges toward the side with more remaining space.

    Input structure:
    - The background is black.
    - A single non-gray start pixel sits on exactly one grid edge; its inward normal gives the initial ray direction.
    - Several gray hinge pixels are scattered in the grid, but only some lie on the ray's eventual route.

    Rule:
    - Move the ray cell by cell from the start marker into the grid.
    - Whenever the ray enters a gray hinge, compare the left-turn and right-turn distances from that hinge to the grid edge and rotate 90 degrees toward the larger one.
    - Continue until the ray steps out of bounds, recoloring every traversed non-gray cell to the start color while keeping all gray hinges unchanged.

    Key insight:
    - Hinges do not encode a fixed mirror orientation; the turn is decided locally from that hinge's current left/right open space, and untouched gray dots are pure distractors.
    """
    return _trace_path(np.asarray(grid, dtype=int))[0]


def _edge_positions(shape: tuple[int, int], edge: str) -> list[Coord]:
    h, w = map(int, shape)
    if edge == "top":
        return [(0, c) for c in range(1, w - 1)]
    if edge == "bottom":
        return [(h - 1, c) for c in range(1, w - 1)]
    if edge == "left":
        return [(r, 0) for r in range(1, h - 1)]
    if edge == "right":
        return [(r, w - 1) for r in range(1, h - 1)]
    raise ValueError(f"unknown edge {edge}")


def _sample_start(shape: tuple[int, int], edge: str, first_turn: str) -> tuple[Coord, Direction] | None:
    candidates: list[tuple[Coord, Direction]] = []
    for pos in _edge_positions(shape, edge):
        direction = _inward_direction(shape, pos)
        if _turn_side(shape, pos, direction) != first_turn:
            continue
        if _edge_distance(shape, pos, direction) <= MIN_SEG_LEN:
            continue
        candidates.append((pos, direction))
    if not candidates:
        return None
    return random.choice(candidates)


def _build_segments(
    shape: tuple[int, int],
    pos: Coord,
    direction: Direction,
    pattern: str,
    idx: int,
    occupied: set[Coord],
) -> list[Segment] | None:
    space = _edge_distance(shape, pos, direction)

    if idx == len(pattern):
        if space < MIN_SEG_LEN:
            return None
        exit_cells = _ray_cells(pos, direction, space)
        if any(cell in occupied for cell in exit_cells):
            return None
        return [(pos, direction, space)]

    chosen_side = _turn_side(shape, pos, direction)
    if chosen_side != pattern[idx]:
        return None

    next_direction = _turn_direction(shape, pos, direction)
    assert next_direction is not None
    next_space = _edge_distance(shape, pos, next_direction)
    needed_after = MIN_SEG_LEN if idx + 1 == len(pattern) else MIN_SEG_LEN + 1
    if next_space < needed_after:
        return None

    if space <= MIN_SEG_LEN:
        return None

    candidate_lengths = list(range(MIN_SEG_LEN, space))
    random.shuffle(candidate_lengths)
    for length in candidate_lengths:
        cells = _ray_cells(pos, direction, length)
        if any(cell in occupied for cell in cells):
            continue
        hinge = cells[-1]
        rest = _build_segments(
            shape,
            hinge,
            next_direction,
            pattern,
            idx + 1,
            occupied | set(cells),
        )
        if rest is not None:
            return [(pos, direction, length)] + rest
    return None


def _path_details(start: Coord, segments: list[Segment]) -> tuple[set[Coord], set[Coord], list[Coord], list[tuple[Coord, Direction, Direction]]]:
    full_path: set[Coord] = {start}
    painted: set[Coord] = {start}
    used_hinges: list[Coord] = []
    hinge_infos: list[tuple[Coord, Direction, Direction]] = []

    for index, (origin, direction, length) in enumerate(segments):
        cells = _ray_cells(origin, direction, length)
        full_path.update(cells)
        if index + 1 == len(segments):
            painted.update(cells)
            continue
        hinge = cells[-1]
        full_path.add(hinge)
        painted.update(cells[:-1])
        used_hinges.append(hinge)
        hinge_infos.append((hinge, direction, segments[index + 1][1]))

    return full_path, painted, used_hinges, hinge_infos


def _manhattan(a: Coord, b: Coord) -> int:
    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


def _can_place_gray(cell: Coord, blocked: set[Coord], gray_cells: set[Coord]) -> bool:
    if cell in blocked or cell in gray_cells:
        return False
    return all(_manhattan(cell, other) > 1 for other in gray_cells)


def _branch_distractor_candidates(
    shape: tuple[int, int],
    hinge_infos: list[tuple[Coord, Direction, Direction]],
    blocked: set[Coord],
) -> list[Coord]:
    candidates: list[Coord] = []
    for hinge, incoming, chosen in hinge_infos:
        rejected = _turn_right(incoming) if chosen == _turn_left(incoming) else _turn_left(incoming)
        for cell in _ray_cells(hinge, rejected, _edge_distance(shape, hinge, rejected)):
            if cell not in blocked:
                candidates.append(cell)
    random.shuffle(candidates)
    return candidates


def _general_distractor_candidates(shape: tuple[int, int], blocked: set[Coord]) -> list[Coord]:
    h, w = map(int, shape)
    cells = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(1, w - 1)
        if (r, c) not in blocked
    ]
    random.shuffle(cells)
    return cells


def _place_distractors(
    shape: tuple[int, int],
    full_path: set[Coord],
    used_hinges: list[Coord],
    hinge_infos: list[tuple[Coord, Direction, Direction]],
) -> list[Coord] | None:
    min_distractors = 2 if len(used_hinges) == 1 else 1
    target = random.randint(min_distractors, min(4, min_distractors + 2))

    gray_cells = set(used_hinges)
    distractors: list[Coord] = []

    for cell in _branch_distractor_candidates(shape, hinge_infos, full_path):
        if _can_place_gray(cell, full_path, gray_cells):
            distractors.append(cell)
            gray_cells.add(cell)
            if len(distractors) >= target:
                return distractors

    for cell in _general_distractor_candidates(shape, full_path):
        if _can_place_gray(cell, full_path, gray_cells):
            distractors.append(cell)
            gray_cells.add(cell)
            if len(distractors) >= target:
                return distractors

    return distractors if len(distractors) >= min_distractors else None


def _render_expected(grid: np.ndarray, start_color: int, painted: set[Coord]) -> np.ndarray:
    out = np.array(grid, copy=True)
    for r, c in painted:
        if int(out[r, c]) != HINGE:
            out[r, c] = int(start_color)
    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    used_hinges: list[Coord],
    distractors: list[Coord],
) -> bool:
    try:
        actual, traced_hinges, painted, hinge_records = _trace_records(grid)
    except AssertionError:
        return False

    if not np.array_equal(actual, expected):
        return False
    if list(traced_hinges) != used_hinges:
        return False
    if not traced_hinges:
        return False
    if set(traced_hinges) & set(distractors):
        return False
    if len(painted) <= 3 or np.array_equal(grid, expected):
        return False

    for hinge, incoming in hinge_records:
        left = _turn_left(incoming)
        right = _turn_right(incoming)
        if _edge_distance(grid.shape, hinge, left) == _edge_distance(grid.shape, hinge, right):
            return False

    for hinge in used_hinges:
        alt = np.array(grid, copy=True)
        alt[hinge] = BG
        try:
            changed = main(alt)
        except AssertionError:
            return False
        if np.array_equal(changed, expected):
            return False

    for hinge in distractors:
        alt = np.array(grid, copy=True)
        alt[hinge] = BG
        try:
            alt_out, alt_hinges, alt_painted, _alt_records = _trace_records(alt)
        except AssertionError:
            return False
        if list(alt_hinges) != used_hinges:
            return False
        if set(alt_painted) != set(painted):
            return False

    return True


def _scenario() -> tuple[str, str]:
    global _GEN_COUNTER
    edge, pattern = _SCENARIOS[_GEN_COUNTER % len(_SCENARIOS)]
    _GEN_COUNTER += 1
    return edge, pattern


def _sample_shape(pattern: str) -> tuple[int, int]:
    min_dim = max(10, 8 + 2 * len(pattern))
    h = random.randint(min_dim, MAX_DIM)
    w = random.randint(min_dim, MAX_DIM)
    return h, w


def _build_candidate(edge: str, pattern: str) -> np.ndarray | None:
    shape = _sample_shape(pattern)
    sampled = _sample_start(shape, edge, pattern[0])
    if sampled is None:
        return None
    start, direction = sampled

    segments = _build_segments(shape, start, direction, pattern, 0, {start})
    if segments is None:
        return None

    full_path, painted, used_hinges, hinge_infos = _path_details(start, segments)
    distractors = _place_distractors(shape, full_path, used_hinges, hinge_infos)
    if distractors is None:
        return None

    start_color = random.choice(START_COLORS)
    grid = np.full(shape, BG, dtype=int)
    grid[start] = int(start_color)
    for hinge in used_hinges:
        grid[hinge] = HINGE
    for hinge in distractors:
        grid[hinge] = HINGE

    expected = _render_expected(grid, start_color, painted)
    if not _semantic_checks(grid, expected, used_hinges, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    edge, pattern = _scenario()
    for _ in range(MAX_ATTEMPTS):
        grid = _build_candidate(edge, pattern)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-turn input")
