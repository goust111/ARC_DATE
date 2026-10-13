from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

Coord = tuple[int, int]
Velocity = tuple[int, int]


def _single_coord(grid: np.ndarray, color: int) -> Coord:
    coords = np.argwhere(grid == color)
    assert coords.shape == (1, 2), f"color {color} must mark exactly one pixel"
    return int(coords[0, 0]), int(coords[0, 1])


def _reflected_step(position: Coord, velocity: Velocity, shape: tuple[int, int]):
    """Return (next position, post-bounce velocity, reflected-axis flags)."""
    p = [int(position[0]), int(position[1])]
    v = [int(velocity[0]), int(velocity[1])]
    reflected = [False, False]
    for axis, limit in enumerate(shape):
        if not 0 <= p[axis] + v[axis] < int(limit):
            v[axis] = -v[axis]
            reflected[axis] = True
        assert 0 <= p[axis] + v[axis] < int(limit), "step is too large for one-bounce motion"
    nxt = (p[0] + v[0], p[1] + v[1])
    return nxt, (v[0], v[1]), (reflected[0], reflected[1])


def _history_candidates(grid: np.ndarray):
    """Find the unique chronological ordering of the two blue pixels."""
    start = _single_coord(grid, GREEN)
    blues = [tuple(map(int, rc)) for rc in np.argwhere(grid == BLUE)]
    assert len(blues) == 2, "expected exactly two blue landing markers"

    candidates = []
    for first, second in ((blues[0], blues[1]), (blues[1], blues[0])):
        initial = (first[0] - start[0], first[1] - start[1])
        if initial == (0, 0):
            continue
        expected_second, current, _ = _reflected_step(first, initial, grid.shape)
        if expected_second == second:
            candidates.append((start, first, second, current))
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """Continue the marked discrete billiard trajectory until it lands exactly on a yellow target.

    Input structure:
    - A black background contains one green start marker, two blue consecutive
      landing markers, and several single-pixel yellow candidate targets.
    - The displacement from green to the chronologically first blue marker is
      the fixed step; the second blue marker either continues it or displays a
      row/column reflection at the boundary.

    Rule:
    - Infer the unique order of the blue markers and the current signed step.
      Continue from the second blue marker in discrete jumps. Before each jump,
      negate every component whose proposed coordinate would leave the grid.
    - Recolor each new landing red. Stop only when a landing is exactly yellow;
      keep that target and all other yellow pixels unchanged.

    Key insight:
    - The bend in the three supplied landings teaches mirror reflection, while
      yellow pixels between landings are decoys: only discrete landing positions
      can stop the trajectory.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
    assert set(map(int, np.unique(g))) == {BACKGROUND, BLUE, GREEN, YELLOW}
    assert np.count_nonzero(g == YELLOW) >= 2, "expected a target and distractors"

    histories = _history_candidates(g)
    assert len(histories) == 1, "the visible markers must determine one trajectory"
    _start, _first, position, velocity = histories[0]

    out = g.copy()
    seen = {(position, velocity)}
    while True:
        position, velocity, _ = _reflected_step(position, velocity, g.shape)
        if int(g[position]) == YELLOW:
            return out
        out[position] = RED
        state = (position, velocity)
        assert state not in seen, "no yellow candidate is reachable from the marked trajectory"
        seen.add(state)


def _primitive_direction() -> Velocity:
    choices = [
        (r, c)
        for r in (-2, -1, 1, 2)
        for c in (-2, -1, 1, 2)
        if abs(r) == 1 or abs(c) == 1
    ]
    return random.choice(choices)


def _trajectory(position: Coord, velocity: Velocity, shape: tuple[int, int]):
    """List one full state cycle after the supplied second blue marker."""
    entries = []
    seen = {(position, velocity)}
    while True:
        position, velocity, bounced = _reflected_step(position, velocity, shape)
        state = (position, velocity)
        if state in seen:
            break
        entries.append((position, velocity, bounced))
        seen.add(state)
    return entries


def _between(a: Coord, b: Coord, divisor: int) -> list[Coord]:
    return [
        (a[0] + (b[0] - a[0]) * j // divisor, a[1] + (b[1] - a[1]) * j // divisor)
        for j in range(1, divisor)
    ]


def _render_candidate(desired_history: str) -> np.ndarray | None:
    height = random.randint(14, 25)
    width = random.randint(14, 25)
    divisor = random.choice((2, 3))
    ur, uc = _primitive_direction()
    velocity = (divisor * ur, divisor * uc)

    valid_starts = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if 0 <= r + velocity[0] < height and 0 <= c + velocity[1] < width
    ]
    random.shuffle(valid_starts)
    if not valid_starts:
        return None
    start = valid_starts[0]
    first = (start[0] + velocity[0], start[1] + velocity[1])
    second, current, history_bounce = _reflected_step(first, velocity, (height, width))
    bounce_count = int(history_bounce[0]) + int(history_bounce[1])
    if desired_history == "straight" and bounce_count != 0:
        return None
    if desired_history == "bend" and bounce_count != 1:
        return None
    if len({start, first, second}) != 3:
        return None

    entries = _trajectory(second, current, (height, width))
    if len(entries) < 7:
        return None
    if bounce_count == 1:
        # The observed bend must materially determine the continuation.  In a
        # very narrow phase, keeping the old velocity can trigger an immediate
        # second reflection and accidentally produce the same next landing.
        wrong_next, _wrong_velocity, _wrong_bounce = _reflected_step(
            second, velocity, (height, width)
        )
        if wrong_next == entries[0][0]:
            return None

    supplied = {start, first, second}
    target_indices = []
    positions: list[Coord] = []
    first_future_bounce = None
    prefix_valid = True
    for i, (pos, _vel, bounced) in enumerate(entries, start=1):
        if pos in supplied or pos in positions:
            prefix_valid = False
        positions.append(pos)
        if any(bounced) and first_future_bounce is None:
            first_future_bounce = i
        if (
            prefix_valid
            and 5 <= i <= 12
            and first_future_bounce is not None
            and i >= first_future_bounce + 1
        ):
            target_indices.append(i)
    if not target_indices:
        return None
    target_index = random.choice(target_indices)
    target = positions[target_index - 1]

    # Put yellow decoys strictly inside several jump segments. Because every
    # velocity component is divisible by `divisor`, these are visible integer
    # pixels on the continuous line but are never discrete landings.
    landing_prefix = [second] + positions[:target_index]
    segment_decoys = []
    segment_ids = list(range(target_index))
    random.shuffle(segment_ids)
    for i in segment_ids:
        for point in _between(landing_prefix[i], landing_prefix[i + 1], divisor):
            if point not in supplied and point != target and point not in segment_decoys:
                segment_decoys.append(point)
                break
        if len(segment_decoys) >= random.choice((2, 3)):
            break
    if len(segment_decoys) < 2:
        return None

    all_orbit_landings = {pos for pos, _vel, _bounce in entries} | supplied
    forbidden = all_orbit_landings | set(segment_decoys)
    free = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if (r, c) not in forbidden
    ]
    if len(free) < 3:
        return None
    off_path = random.sample(free, random.randint(1, 3))

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for point in segment_decoys + off_path + [target]:
        grid[point] = YELLOW
    grid[start] = GREEN
    grid[first] = BLUE
    grid[second] = BLUE

    if len(_history_candidates(grid)) != 1:
        return None

    # Independent construction-time oracle: every pre-target landing must be
    # new, visible red; the exact target must remain yellow.
    expected = grid.copy()
    for point in positions[: target_index - 1]:
        if int(expected[point]) != BACKGROUND:
            return None
        expected[point] = RED
    if int(expected[target]) != YELLOW:
        return None
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.count_nonzero(actual == RED) < 4:
        return None
    return grid


_GEN_CALL_COUNT = 0


def generate_input() -> np.ndarray:
    """Generate varied, non-degenerate examples with balanced history branches."""
    global _GEN_CALL_COUNT
    desired_history = "straight" if _GEN_CALL_COUNT % 2 == 0 else "bend"
    _GEN_CALL_COUNT += 1
    for _ in range(2000):
        grid = _render_candidate(desired_history)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid {desired_history} billiard input")
