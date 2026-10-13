from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


GRAY = 5
LEFT = (-1,)
RIGHT = (1,)
BOTH = (-1, 1)


def _dominant_and_rarest(grid: np.ndarray) -> tuple[int, int]:
    """Return the unique dominant background and unique rarest other color."""
    counts = Counter(map(int, grid.flat))
    largest = max(counts.values())
    backgrounds = [color for color, count in counts.items() if count == largest]
    assert len(backgrounds) == 1, "background color must be uniquely dominant"
    bg = int(backgrounds[0])

    non_bg = {color: count for color, count in counts.items() if color != bg}
    assert non_bg, "expected non-background objects"
    smallest = min(non_bg.values())
    rarest = [color for color, count in non_bg.items() if count == smallest]
    assert len(rarest) == 1, "water color must be uniquely rarest"
    return bg, int(rarest[0])


def _decode_legends(
    grid: np.ndarray, bg: int, water: int
) -> dict[int, tuple[int, ...]]:
    """Decode every horizontal grey/shelf/grey-or-bg 1x3 legend."""
    h, w = grid.shape
    legends: dict[int, tuple[int, ...]] = {}
    occurrences: Counter[int] = Counter()

    for r in range(h):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            if center in {bg, water, GRAY}:
                continue
            left = int(grid[r, c - 1])
            right = int(grid[r, c + 1])
            if left not in {bg, GRAY} or right not in {bg, GRAY}:
                continue
            if left != GRAY and right != GRAY:
                continue

            directions = tuple(
                direction
                for direction, wing in ((-1, left), (1, right))
                if wing == GRAY
            )
            occurrences[center] += 1
            legends[center] = directions

    assert legends, "expected at least one 1x3 corner legend"
    assert all(count == 1 for count in occurrences.values()), (
        "each shelf color must have exactly one legend"
    )
    return legends


def _find_shelves(
    grid: np.ndarray, mapping: dict[int, tuple[int, ...]]
) -> list[tuple[int, int, int, int]]:
    """Find maximal horizontal shelf runs, excluding one-cell legend centers."""
    h, w = grid.shape
    shelves: list[tuple[int, int, int, int]] = []
    shelf_colors = set(mapping)
    for r in range(h):
        c = 0
        while c < w:
            color = int(grid[r, c])
            if color not in shelf_colors:
                c += 1
                continue
            a = c
            while c + 1 < w and int(grid[r, c + 1]) == color:
                c += 1
            b = c
            if b - a + 1 >= 2:
                shelves.append((r, a, b, color))
            c += 1
    assert shelves, "expected horizontal colored shelves"
    assert set(color for _r, _a, _b, color in shelves) == shelf_colors, (
        "every legend color must occur on a shelf"
    )
    return shelves


def _shelves_by_row(
    shelves: list[tuple[int, int, int, int]]
) -> dict[int, list[tuple[int, int, int]]]:
    by_row: dict[int, list[tuple[int, int, int]]] = {}
    for r, a, b, color in shelves:
        by_row.setdefault(r, []).append((a, b, color))
    for row in by_row.values():
        row.sort()
    return by_row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route falling water at each colored shelf using the matching corner legend.

    Input structure:
    - The unique dominant color is the background, and the unique rarest
      non-background color marks one or more water sources on the top row.
    - Every shelf is a monochromatic horizontal run.  In a corner legend area,
      each shelf color appears as the center of a horizontal 1x3 entry whose
      grey left and/or right wings encode the permitted overflow sides.

    Rule:
    - Water falls vertically through background cells.  On reaching a shelf,
      it consults that shelf color's legend, paints only from the impact column
      toward each permitted end along the row above, continues one cell beyond
      that end, and resumes falling there.  A two-wing legend splits the stream.
      Shelves, sources, and all legend cells remain unchanged.

    Key insight:
    - Geometry locates a shelf, but its visible color-to-legend match determines
      which branch or branches continue, so the complete cascade requires both
      decoding the corner key and recursively simulating gravity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    bg, water = _dominant_and_rarest(g)
    assert water != GRAY, "grey is reserved for legend indicators"

    mapping = _decode_legends(g, bg, water)
    shelves = _find_shelves(g, mapping)
    by_row = _shelves_by_row(shelves)

    sources = [(0, int(c)) for c in np.flatnonzero(g[0] == water)]
    assert sources, "at least one water source must lie on the top row"
    assert int(np.count_nonzero(g == water)) == len(sources), (
        "input water pixels must all be top-row sources"
    )

    out = g.copy()
    queue = deque(sources)
    started: set[tuple[int, int]] = set()

    while queue:
        r, c = queue.popleft()
        if (r, c) in started:
            continue
        started.add((r, c))

        rr = int(r)
        if int(g[rr, c]) in {bg, water}:
            out[rr, c] = water

        while rr + 1 < h:
            nr = rr + 1
            hit: tuple[int, int, int] | None = None
            for a, b, color in by_row.get(nr, []):
                if a <= c <= b:
                    hit = (a, b, color)
                    break

            if hit is not None:
                a, b, color = hit
                for direction in mapping[color]:
                    overflow = a - 1 if direction < 0 else b + 1
                    assert 0 <= overflow < w, "an allowed shelf end must have overflow space"
                    lo, hi = sorted((c, overflow))
                    for cc in range(lo, hi + 1):
                        assert int(g[rr, cc]) in {bg, water}, (
                            "the row above a shelf must be clear to its allowed end"
                        )
                        out[rr, cc] = water
                    queue.append((rr, overflow))
                break

            below = int(g[nr, c])
            if below not in {bg, water}:
                break
            rr = nr
            out[rr, c] = water

    return out


# ------------------------------ generator ------------------------------


def _paint_shelf(
    grid: np.ndarray,
    shelves: list[tuple[int, int, int, int]],
    bg: int,
    r: int,
    a: int,
    b: int,
    color: int,
) -> None:
    assert 0 < a <= b < grid.shape[1] - 1
    assert np.all(grid[r, a : b + 1] == bg), "shelves must not overlap"
    grid[r, a : b + 1] = color
    shelves.append((r, a, b, color))


def _oracle_from_spec(
    grid: np.ndarray,
    bg: int,
    water: int,
    sources: list[int],
    shelves: list[tuple[int, int, int, int]],
    mapping: dict[int, tuple[int, ...]],
) -> tuple[np.ndarray, list[tuple[int, int, int, int, int, int, int]]]:
    """Independent generator oracle using the construction metadata directly."""
    h, w = grid.shape
    by_row = _shelves_by_row(shelves)
    out = grid.copy()
    queue = deque((0, c, 0) for c in sources)
    seen: set[tuple[int, int]] = set()
    events: list[tuple[int, int, int, int, int, int, int]] = []

    while queue:
        r, c, depth = queue.popleft()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        rr = r
        if int(grid[rr, c]) in {bg, water}:
            out[rr, c] = water

        while rr + 1 < h:
            shelf = next(
                (
                    (a, b, color)
                    for a, b, color in by_row.get(rr + 1, [])
                    if a <= c <= b
                ),
                None,
            )
            if shelf is not None:
                a, b, color = shelf
                events.append((rr + 1, a, b, color, c, depth, len(mapping[color])))
                for direction in mapping[color]:
                    overflow = a - 1 if direction < 0 else b + 1
                    if not 0 <= overflow < w:
                        return out, []
                    lo, hi = sorted((c, overflow))
                    if any(int(grid[rr, cc]) not in {bg, water} for cc in range(lo, hi + 1)):
                        return out, []
                    out[rr, lo : hi + 1] = water
                    queue.append((rr, overflow, depth + 1))
                break

            if int(grid[rr + 1, c]) not in {bg, water}:
                break
            rr += 1
            out[rr, c] = water

    return out, events


def _legend_values(bg: int, color: int, directions: tuple[int, ...]) -> list[int]:
    return [GRAY if -1 in directions else bg, color, GRAY if 1 in directions else bg]


def _intervals_separate(
    r: int,
    a: int,
    b: int,
    shelves: list[tuple[int, int, int, int]],
) -> bool:
    return all(sr != r or b + 1 < sa or sb + 1 < a for sr, sa, sb, _ in shelves)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(22, 30)
    w = random.randint(24, 30)

    palette = [color for color in range(10) if color != GRAY]
    bg, water, left_color, right_color, both_color = random.sample(palette, 5)
    mapping = {
        left_color: LEFT,
        right_color: RIGHT,
        both_color: BOTH,
    }

    grid = np.full((h, w), bg, dtype=int)
    corner = random.choice(("left", "right"))
    legend_c0 = 0 if corner == "left" else w - 3
    legend_rows = [0, 2, 4]
    legend_colors = [left_color, right_color, both_color]
    random.shuffle(legend_colors)
    legend_positions: dict[int, tuple[int, int]] = {}
    for r, color in zip(legend_rows, legend_colors):
        grid[r, legend_c0 : legend_c0 + 3] = _legend_values(bg, color, mapping[color])
        legend_positions[color] = (r, legend_c0 + 1)

    shelves: list[tuple[int, int, int, int]] = []
    root_c = w // 2 + random.randint(-1, 1)
    root_r = random.randint(7, 9)
    root_a = root_c - random.randint(2, 3)
    root_b = root_c + random.randint(2, 3)
    _paint_shelf(grid, shelves, bg, root_r, root_a, root_b, both_color)

    left_in = root_a - 1
    right_in = root_b + 1
    left_r = root_r + random.randint(4, 5)
    right_r = root_r + random.randint(4, 5)

    left_long = random.randint(3, 4)
    left_short = random.randint(0, 1)
    left_a = left_in - left_long
    left_b = left_in + left_short
    right_long = random.randint(3, 4)
    right_short = random.randint(0, 1)
    right_a = right_in - right_short
    right_b = right_in + right_long
    if left_a <= 1 or right_b >= w - 2:
        return None
    _paint_shelf(grid, shelves, bg, left_r, left_a, left_b, left_color)
    _paint_shelf(grid, shelves, bg, right_r, right_a, right_b, right_color)

    left_out = left_a - 1
    right_out = right_b + 1
    target = random.choice((left_out, right_out))
    grand_r = max(left_r, right_r) + random.randint(4, 5)
    if grand_r >= h - 2:
        return None
    grand_left = random.randint(2, 3)
    grand_right = random.randint(2, 3)
    grand_a = target - grand_left
    grand_b = target + grand_right
    if grand_a <= 0 or grand_b >= w - 1:
        return None
    if not _intervals_separate(grand_r, grand_a, grand_b, shelves):
        return None
    _paint_shelf(grid, shelves, bg, grand_r, grand_a, grand_b, both_color)

    sources = [root_c]
    grid[0, root_c] = water

    base_expected, base_events = _oracle_from_spec(
        grid, bg, water, sources, shelves, mapping
    )
    if not base_events:
        return None

    # About half the examples receive a second, independent top source.  It is
    # chosen only from a column untouched by the cascade and by every shelf.
    if random.random() < 0.55:
        candidates = []
        for c in range(1, w - 1):
            if legend_c0 <= c < legend_c0 + 3:
                continue
            if int(grid[0, c]) != bg:
                continue
            if any(a <= c <= b for _r, a, b, _color in shelves):
                continue
            if np.any(base_expected[:, c] == water):
                continue
            candidates.append(c)
        if candidates:
            second = random.choice(candidates)
            sources.append(second)
            grid[0, second] = water

    expected, events = _oracle_from_spec(grid, bg, water, sources, shelves, mapping)
    if not events:
        return None

    # Add zero to two genuinely inactive shelves.  Each candidate is accepted
    # only when the independent oracle proves it changes neither water nor hits.
    distractor_goal = random.randint(0, 2)
    for _ in range(distractor_goal):
        for _try in range(50):
            r = random.randint(7, h - 2)
            length = random.randint(3, 6)
            a = random.randint(1, w - length - 1)
            b = a + length - 1
            color = random.choice([left_color, right_color, both_color])
            if not _intervals_separate(r, a, b, shelves):
                continue
            if np.any(grid[r, a : b + 1] != bg):
                continue
            trial_grid = grid.copy()
            trial_grid[r, a : b + 1] = color
            trial_shelves = shelves + [(r, a, b, color)]
            trial_expected, trial_events = _oracle_from_spec(
                trial_grid, bg, water, sources, trial_shelves, mapping
            )
            if not np.array_equal(trial_expected == water, expected == water):
                continue
            if trial_events != events:
                continue
            grid = trial_grid
            shelves = trial_shelves
            expected = trial_expected
            break

    # Semantic checks: all three rules are exercised, the route cascades at
    # least two decisions deep, and nearest-edge routing is contradicted.
    hit_colors = {event[3] for event in events}
    if hit_colors != set(mapping):
        return None
    if max(event[5] for event in events) < 2:
        return None
    if sum(event[3] == both_color for event in events) < 2:
        return None

    single_hits = [event for event in events if event[3] in {left_color, right_color}]
    if len(single_hits) < 2:
        return None
    for _r, a, b, color, impact, _depth, _branches in single_hits:
        allowed = impact - (a - 1) if color == left_color else (b + 1) - impact
        forbidden = (b + 1) - impact if color == left_color else impact - (a - 1)
        if allowed <= forbidden:
            return None

    new_water = int(np.count_nonzero((expected == water) & (grid == bg)))
    if new_water < 18 or new_water > int(0.38 * grid.size):
        return None

    # Every color-to-direction association is necessary: changing just one
    # encoded behavior must change the independently simulated result.
    alternatives = {left_color: RIGHT, right_color: LEFT, both_color: LEFT}
    for color, alternative in alternatives.items():
        changed = dict(mapping)
        changed[color] = alternative
        changed_expected, changed_events = _oracle_from_spec(
            grid, bg, water, sources, shelves, changed
        )
        if not changed_events or np.array_equal(changed_expected, expected):
            return None

    # Decode and solve the actual rendered input, then test each rendered
    # legend mutation against the corresponding independent specification.
    try:
        solved = main(grid)
    except (AssertionError, IndexError, KeyError):
        return None
    if not np.array_equal(solved, expected):
        return None

    for color, alternative in alternatives.items():
        mutated = grid.copy()
        r, c = legend_positions[color]
        mutated[r, c - 1 : c + 2] = _legend_values(bg, color, alternative)
        changed = dict(mapping)
        changed[color] = alternative
        oracle, _ = _oracle_from_spec(mutated, bg, water, sources, shelves, changed)
        try:
            got = main(mutated)
        except (AssertionError, IndexError, KeyError):
            return None
        if np.array_equal(got, solved) or not np.array_equal(got, oracle):
            return None

    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, semantically checked colored-shelf waterfall input."""
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            assert grid.ndim == 2 and max(grid.shape) <= 30
            return grid
    raise RuntimeError("failed to generate a valid colored-shelf waterfall")
