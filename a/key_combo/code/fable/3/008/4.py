from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
COLORS = tuple(range(1, 10))

UP = (-1, 0)
RIGHT = (0, 1)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT_TURN = {
    UP: RIGHT,
    RIGHT: DOWN,
    DOWN: LEFT,
    LEFT: UP,
}
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

GEN_RETRY_LIMIT = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return int(r) in (0, h - 1) or int(c) in (0, w - 1)


def _is_internal(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 < int(r) < h - 1 and 0 < int(c) < w - 1


def _edge_direction(shape: tuple[int, int], r: int, c: int) -> tuple[int, int]:
    h, w = map(int, shape)
    directions: list[tuple[int, int]] = []
    if int(r) == 0:
        directions.append(DOWN)
    if int(r) == h - 1:
        directions.append(UP)
    if int(c) == 0:
        directions.append(RIGHT)
    if int(c) == w - 1:
        directions.append(LEFT)
    assert len(directions) == 1, "the start pixel must be on exactly one border edge, not a corner"
    return directions[0]


def _colored_cells(g: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(g != BACKGROUND)]


def _find_start(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = g.shape
    border: set[tuple[int, int]] = set()
    for c in range(w):
        border.add((0, int(c)))
        border.add((h - 1, int(c)))
    for r in range(h):
        border.add((int(r), 0))
        border.add((int(r), w - 1))

    starts = [(r, c) for r, c in border if int(g[r, c]) != BACKGROUND]
    assert len(starts) == 1, "expected exactly one non-black boundary start pixel"
    start = starts[0]
    return start, _edge_direction(g.shape, *start)


def _assert_isolated_controls(g: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    controls = [(r, c) for r, c in _colored_cells(g) if (r, c) != start]
    assert controls, "expected internal isolated control pixels"

    for r, c in controls:
        assert _is_internal(g.shape, r, c), "all non-start colored pixels must be internal controls"

    for i, (r0, c0) in enumerate(controls):
        for r1, c1 in controls[i + 1 :]:
            assert max(abs(int(r0) - int(r1)), abs(int(c0) - int(c1))) > 1, (
                "control pixels must be mutually 8-isolated"
            )
    return controls


def _trace(grid: np.ndarray, *, details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    assert h >= 5 and w >= 5, "grid must be large enough for an internal turning ray"
    assert h <= 30 and w <= 30, "ARC grids must not exceed 30 by 30"
    assert np.all((0 <= g) & (g <= 9)), "ARC colors must be in 0..9"

    start, direction = _find_start(g)
    _assert_isolated_controls(g, start)

    out = g.copy()
    pos = start
    color = int(g[start])
    hits: list[tuple[int, int]] = []
    hit_seen: set[tuple[int, int]] = set()
    drawn: list[tuple[int, int, int]] = []
    state_seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for _step in range(h * w * 10 + 50):
        state = (pos, direction, int(color))
        assert state not in state_seen, "the chained ray must eventually leave the grid"
        state_seen.add(state)

        nr = int(pos[0] + direction[0])
        nc = int(pos[1] + direction[1])
        if not _in_bounds(g.shape, nr, nc):
            break

        next_color = int(g[nr, nc])
        if next_color != BACKGROUND:
            assert (nr, nc) != start, "the ray may not run back into the boundary start"
            assert not _is_border(g.shape, nr, nc), "turning markers must be internal pixels"
            assert (nr, nc) not in hit_seen, "a control pixel may be hit only once"
            hits.append((nr, nc))
            hit_seen.add((nr, nc))
            direction = RIGHT_TURN[direction]
            color = next_color
            continue

        pos = (nr, nc)
        out[nr, nc] = int(color)
        drawn.append((nr, nc, int(color)))
    else:
        raise AssertionError("ray tracing exceeded the finite-grid step bound")

    assert hits, "expected the ray to hit at least one turning control"
    assert drawn, "expected the ray to paint at least one background cell"

    if details:
        return out, hits, drawn
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the chained right-turn ray, recoloring it at each marker it hits.

    Input structure:
    - The background is black.
    - Exactly one non-black pixel lies on a non-corner boundary cell; this is the ray start.
    - The other non-black pixels are mutually 8-isolated internal control pixels, with some placed off the walked path as distractors.

    Rule:
    - Send the ray inward from the start, perpendicular to the border edge containing the start pixel.
    - Paint every input-background cell the ray enters with the current line color, initially the start color.
    - When the next cell in the current direction is an internal control pixel, stop before that pixel, turn right, and change the line color to the control pixel's color.
    - Repeat until the ray steps out of bounds; keep all control pixels and the start pixel unchanged.

    Key insight:
    - The isolated pixels are not independent line sources: only pixels encountered as the next cell of the same chained walk turn and recolor the ray, while unhit pixels are distractors.
    """
    return _trace(grid)


def _segment_cells(
    start: tuple[int, int],
    end: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    sr, sc = map(int, start)
    er, ec = map(int, end)
    dr, dc = map(int, direction)
    if dr == 0:
        assert sr == er
        steps = (ec - sc) * dc
    else:
        assert sc == ec
        steps = (er - sr) * dr
    assert steps >= 1
    return [(sr + i * dr, sc + i * dc) for i in range(1, int(steps) + 1)]


def _can_add_control(
    grid: np.ndarray,
    cell: tuple[int, int],
    forbidden: set[tuple[int, int]],
) -> bool:
    r, c = map(int, cell)
    if not _is_internal(grid.shape, r, c):
        return False
    if (r, c) in forbidden or int(grid[r, c]) != BACKGROUND:
        return False

    for rr, cc in _colored_cells(grid):
        if max(abs(int(r) - int(rr)), abs(int(c) - int(cc))) <= 1:
            return False
    return True


def _paint_top_plan(
    h: int,
    w: int,
    start: tuple[int, int],
    turns: list[tuple[int, int]],
    colors: list[int],
) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int], int]]]:
    grid = np.full((h, w), BACKGROUND, dtype=int)
    grid[start] = int(colors[0])

    pos = start
    direction = DOWN
    marker_infos: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for idx, turn in enumerate(turns):
        for r, c in _segment_cells(pos, turn, direction):
            assert _in_bounds(grid.shape, r, c)
            assert int(grid[r, c]) == BACKGROUND

        mr = int(turn[0] + direction[0])
        mc = int(turn[1] + direction[1])
        assert _is_internal(grid.shape, mr, mc)
        assert int(grid[mr, mc]) == BACKGROUND
        grid[mr, mc] = int(colors[idx + 1])
        marker_infos.append(((mr, mc), direction, int(colors[idx + 1])))

        pos = turn
        direction = RIGHT_TURN[direction]

    return grid, marker_infos


def _sample_top_plan(rng: random.Random) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int], int]], int] | None:
    h = rng.randint(12, 30)
    w = rng.randint(12, 30)
    turn_count = rng.choices([2, 3, 4], weights=[3, 3, 4])[0]

    s_min = 5
    s_max = w - 4
    r1_min = 5
    r1_max = h - 4
    if s_min > s_max or r1_min > r1_max:
        return None

    start = (0, rng.randint(s_min, s_max))
    r1 = rng.randint(r1_min, r1_max)
    c2_max = int(start[1]) - 3
    if c2_max < 2:
        return None
    c2 = rng.randint(2, c2_max)

    turns = [(r1, int(start[1])), (r1, c2)]

    if turn_count >= 3:
        if r1 - 3 < 2:
            return None
        r3 = rng.randint(2, r1 - 3)
        turns.append((r3, c2))

    if turn_count >= 4:
        c4_options = [c for c in range(c2 + 3, w - 2) if c not in (int(start[1]) - 1, int(start[1]))]
        if not c4_options:
            return None
        c4 = rng.choice(c4_options)
        turns.append((turns[2][0], c4))

    colors = rng.sample(COLORS, turn_count + 1)
    try:
        grid, marker_infos = _paint_top_plan(h, w, start, turns, colors)
        _out, hits, _drawn = _trace(grid, details=True)
    except AssertionError:
        return None

    intended_hits = [marker for marker, _direction, _color in marker_infos]
    if hits != intended_hits:
        return None
    return grid, marker_infos, turn_count


def _place_distractors(
    rng: random.Random,
    grid: np.ndarray,
    marker_infos: list[tuple[tuple[int, int], tuple[int, int], int]],
    path_cells: set[tuple[int, int]],
) -> bool:
    placed = 0
    aligned = False
    target = rng.randint(2, 5)

    shuffled_infos = marker_infos[:]
    rng.shuffle(shuffled_infos)
    for (mr, mc), direction, _color in shuffled_infos:
        distances = list(range(2, 7))
        rng.shuffle(distances)
        for dist in distances:
            cell = (int(mr + direction[0] * dist), int(mc + direction[1] * dist))
            if not _can_add_control(grid, cell, path_cells):
                continue
            grid[cell] = rng.choice(COLORS)
            placed += 1
            aligned = True
            break
        if aligned:
            break

    if not aligned:
        return False

    for _try in range(600):
        if placed >= target:
            break
        cell = (rng.randint(1, grid.shape[0] - 2), rng.randint(1, grid.shape[1] - 2))
        if not _can_add_control(grid, cell, path_cells):
            continue
        grid[cell] = rng.choice(COLORS)
        placed += 1

    return placed >= 2


def _semantic_checks(grid: np.ndarray, expected_hit_count: int) -> bool:
    try:
        out, hits, drawn = _trace(grid, details=True)
    except AssertionError:
        return False

    start, _direction = _find_start(grid)
    controls = [(r, c) for r, c in _colored_cells(grid) if (r, c) != start]
    hit_set = set(hits)
    distractors = [cell for cell in controls if cell not in hit_set]
    if len(hits) != int(expected_hit_count) or not distractors:
        return False
    if np.array_equal(out, grid):
        return False

    drawn_colors = {int(color) for _r, _c, color in drawn}
    if int(grid[start]) not in drawn_colors:
        return False
    for marker in hits:
        if int(grid[marker]) not in drawn_colors:
            return False

    if not any(_is_border(grid.shape, r, c) for r, c, _color in drawn):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sampled = _sample_top_plan(rng)
    if sampled is None:
        return None
    grid, marker_infos, turn_count = sampled

    try:
        _out, hits, drawn = _trace(grid, details=True)
    except AssertionError:
        return None
    if hits != [marker for marker, _direction, _color in marker_infos]:
        return None

    path_cells = {(int(r), int(c)) for r, c, _color in drawn}
    if not _place_distractors(rng, grid, marker_infos, path_cells):
        return None
    if not _semantic_checks(grid, turn_count):
        return None

    rotations = rng.randrange(4)
    result = np.rot90(grid, rotations).copy()
    if not _semantic_checks(result, turn_count):
        return None

    out, hits_after, drawn_after = _trace(result, details=True)
    if len(hits_after) != turn_count:
        return None
    start_after, _direction_after = _find_start(result)
    segment_source_cells = [start_after] + hits_after
    if len({int(result[cell]) for cell in segment_source_cells}) < turn_count + 1:
        return None
    if not drawn_after or np.array_equal(out, result):
        return None
    return result


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chained turning ray input")
