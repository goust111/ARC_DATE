from collections import Counter, defaultdict

import numpy as np
import random


def _side_from_top_repetition(g: np.ndarray) -> str | None:
    h, w = g.shape
    assert h >= 2 and w >= 2, "input must be large enough for a 2x2 crop"

    columns_by_color: dict[int, list[int]] = defaultdict(list)
    for c, color in enumerate(map(int, g[0])):
        columns_by_color[color].append(c)

    repeated_groups = [cols for cols in columns_by_color.values() if len(cols) > 1]
    if not repeated_groups:
        return None

    candidates: list[tuple[int, str]] = []
    for cols in repeated_groups:
        for c in cols:
            left_dist = c
            right_dist = w - 1 - c
            if left_dist < right_dist:
                candidates.append((left_dist, "left"))
            elif right_dist < left_dist:
                candidates.append((right_dist, "right"))

    assert candidates, "expected a repeated top-row color occurrence nearest to one side"
    nearest = min(dist for dist, _side in candidates)
    nearest_sides = [side for dist, side in candidates if dist == nearest]
    assert len(nearest_sides) == 1, "expected a unique nearest repeated top-row occurrence"
    return nearest_sides[0]


def _side_from_rarer_top_corner(g: np.ndarray) -> str:
    counts = Counter(map(int, g.ravel()))
    left_count = counts[int(g[0, 0])]
    right_count = counts[int(g[0, -1])]
    assert left_count != right_count, "expected one top corner color to be rarer"
    return "left" if left_count < right_count else "right"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the top-left or top-right 2x2 corner crop selected by the top edge clue.

    Input structure
    - The input is a square color grid whose answer is one of the two 2x2 crops
      touching the top edge and a grid corner.
    - A repeated color on the top row, when present, acts as the primary side cue.
    - Otherwise, the two top corner colors have different whole-grid frequencies.

    Rule
    - If a top-row color repeats, find the repeated top-row occurrence nearest to
      a left or right edge and take the 2x2 top-corner crop on that side.
    - If the top row has no repeated color, compare the whole-grid frequencies of
      the two top-corner colors and take the side whose corner color is rarer.

    Key insight
    - The task is not asking for an arbitrary 2x2 search: the decisive information
      is on the top edge, and it selects which top corner crop to keep.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "input must be a 2D grid"
    h, w = g.shape
    assert h == w and h >= 2, "input must be a square grid with a 2x2 top crop"

    side = _side_from_top_repetition(g)
    if side is None:
        side = _side_from_rarer_top_corner(g)

    c0 = 0 if side == "left" else w - 2
    return g[:2, c0 : c0 + 2].copy()

# --------------- generator (att2) ---------------

# Retry limits.
_GEN_MAX_ATTEMPTS = 300
# Canvas and palette bounds.
_GEN_MIN_SIZE = 3
_GEN_MAX_SIZE = 8
_GEN_COLORS = tuple(range(10))
_GEN_MIN_DISTINCT_COLORS = 6
_GEN_MAX_DISTINCT_COLORS = 10
# Cue choices.
_GEN_SIDES = ("left", "right")
_GEN_MECHANISMS = ("repetition", "rarer_corner")


def _gen_repetition_pairs(n: int, side: str) -> list[tuple[int, int]]:
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            candidates = []
            for c in (a, b):
                left_dist = c
                right_dist = n - 1 - c
                if left_dist < right_dist:
                    candidates.append((left_dist, "left"))
                elif right_dist < left_dist:
                    candidates.append((right_dist, "right"))
            if not candidates or not any(0 < c < n - 1 for c in (a, b)):
                continue
            nearest = min(dist for dist, _ in candidates)
            nearest_sides = [s for dist, s in candidates if dist == nearest]
            if len(nearest_sides) == 1 and nearest_sides[0] == side:
                pairs.append((a, b))
    return pairs


def _gen_body_positions(n: int) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(1, n) for c in range(n)]
    random.shuffle(positions)
    return positions


def _gen_distinct_color_target(present_count: int, open_count: int, reserve: int = 0) -> int:
    return random.randint(
        max(_GEN_MIN_DISTINCT_COLORS, present_count),
        min(_GEN_MAX_DISTINCT_COLORS, present_count + open_count - reserve),
    )


def _gen_new_colors(present: set[int], count: int) -> list[int]:
    return random.sample([c for c in _GEN_COLORS if c not in present], count)


def _gen_empty_grid(top: list[int]) -> np.ndarray:
    n = len(top)
    grid = np.empty((n, n), dtype=int)
    grid[0] = np.array(top, dtype=int)
    return grid


def _gen_free_body_values(present: set[int], position_count: int) -> list[int]:
    target_total = _gen_distinct_color_target(len(present), position_count)
    forced = _gen_new_colors(present, target_total - len(present))
    allowed = list(present) + forced
    values = forced + [
        random.choice(allowed) for _ in range(position_count - len(forced))
    ]
    random.shuffle(values)
    return values


def _gen_fill_free_body(grid: np.ndarray) -> None:
    positions = _gen_body_positions(grid.shape[0])
    present = set(map(int, grid[0]))
    values = _gen_free_body_values(present, len(positions))
    for (r, c), color in zip(positions, values):
        grid[r, c] = color


def _gen_repetition_grid(n: int, side: str) -> np.ndarray | None:
    pairs = _gen_repetition_pairs(n, side)
    if not pairs:
        return None

    a, b = random.choice(pairs)
    colors = random.sample(_GEN_COLORS, n - 1)
    repeated = colors[0]
    others = colors[1:]

    top = [None] * n
    top[a] = repeated
    top[b] = repeated
    for c in range(n):
        if top[c] is None:
            top[c] = others.pop()

    grid = _gen_empty_grid(top)
    _gen_fill_free_body(grid)
    return grid


def _gen_rarer_corner_adds(remaining: int) -> tuple[int, int]:
    count_pairs = [
        (target_add, other_add)
        for target_add in range(remaining + 1)
        for other_add in range(target_add + 1, remaining + 1)
        if target_add + other_add <= remaining
    ]
    return random.choice(count_pairs)


def _gen_rarer_corner_values(present, position_count, target_color, other_color):
    target_total = _gen_distinct_color_target(len(present), position_count, reserve=1)
    forced_new = _gen_new_colors(present, target_total - len(present))
    remaining = position_count - len(forced_new)
    target_add, other_add = _gen_rarer_corner_adds(remaining)

    allowed = list(present) + forced_new
    neutral_colors = [c for c in allowed if c not in (target_color, other_color)]
    values = (
        forced_new
        + [target_color] * target_add
        + [other_color] * other_add
        + [
            random.choice(neutral_colors)
            for _ in range(remaining - target_add - other_add)
        ]
    )
    random.shuffle(values)
    return values


def _gen_rarer_corner_grid(n: int, side: str) -> np.ndarray:
    top = random.sample(_GEN_COLORS, n)
    grid = _gen_empty_grid(top)

    target_color = top[0] if side == "left" else top[-1]
    other_color = top[-1] if side == "left" else top[0]
    positions = _gen_body_positions(n)
    values = _gen_rarer_corner_values(set(top), len(positions), target_color, other_color)
    for (r, c), color in zip(positions, values):
        grid[r, c] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(_GEN_MIN_SIZE, _GEN_MAX_SIZE)
    side = random.choice(_GEN_SIDES)
    mechanism = random.choice(_GEN_MECHANISMS)

    if mechanism == "repetition":
        grid = _gen_repetition_grid(n, side)
        if grid is None:
            return None
    else:
        grid = _gen_rarer_corner_grid(n, side)

    expected = grid[:2, :2] if side == "left" else grid[:2, -2:]
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {_GEN_MAX_ATTEMPTS} attempts")
