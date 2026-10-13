import random
from collections import Counter

import numpy as np


BLACK = 0
RED = 2
MARKER_POOL = (1, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 800


def _quadrant_of(r: int, c: int, h: int, w: int) -> tuple[int, int]:
    return (0 if r < h // 2 else 1, 0 if c < w // 2 else 1)


def _quadrant_bounds(h: int, w: int, quad: tuple[int, int]) -> tuple[int, int, int, int]:
    qr, qc = quad
    r0 = 0 if qr == 0 else h // 2
    r1 = h // 2 - 1 if qr == 0 else h - 1
    c0 = 0 if qc == 0 else w // 2
    c1 = w // 2 - 1 if qc == 0 else w - 1
    return r0, c0, r1, c1


def _boundary_depth(r: int, c: int, bounds: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bounds
    return min(r - r0, r1 - r, c - c0, c1 - c)


def _all_cells(bounds: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bounds
    return [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]


def _ring_cells(bounds: tuple[int, int, int, int], depth: int) -> list[tuple[int, int]]:
    return [(r, c) for r, c in _all_cells(bounds) if _boundary_depth(r, c, bounds) == depth]


def _max_depth(bounds: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bounds
    return (min(r1 - r0 + 1, c1 - c0 + 1) - 1) // 2


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    reds = np.argwhere(g == RED)
    if len(reds) != 1:
        raise ValueError("expected exactly one red control pixel")

    rr, cc = map(int, reds[0])
    quad = _quadrant_of(rr, cc, h, w)
    bounds = _quadrant_bounds(h, w, quad)
    counts = Counter(map(int, g.flat))

    r0, c0, r1, c1 = bounds
    active = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            color = int(g[r, c])
            if color in (BLACK, RED):
                continue
            if counts[color] == 1:
                active.append((_boundary_depth(r, c, bounds), r, c, color))

    active.sort(key=lambda item: (item[0], item[1], item[2]))
    palette = [BLACK] + [color for _d, _r, _c, color in active]

    out = np.zeros_like(g)
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            depth = _boundary_depth(r, c, bounds)
            out[r, c] = palette[depth % len(palette)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the control pixel's quadrant with concentric bands whose palette comes from that quadrant's unique marker pixels.

    Input structure:
    - An even-sized black grid contains one red control pixel, 2-3 single-cell marker pixels of globally unique colors, and several repeated-color distractor pixels.
    - The red control pixel lies in one of the four equal quadrants of the grid.
    - At least two unique marker pixels lie in the same quadrant as the red control pixel, while distractors may appear both inside and outside that quadrant.

    Rule:
    - Split the grid into four equal quadrants and keep only the quadrant containing the red control pixel.
    - Treat every non-black, non-red color that appears more than once in the whole grid as a distractor and erase it.
    - Among the remaining pixels in the chosen quadrant, use the unique-color marker pixels to build a palette: black first, then marker colors sorted by increasing distance to the chosen quadrant's nearest edge, breaking ties by row then column.
    - Repaint the entire chosen quadrant so each cell takes palette[distance_to_quadrant_edge mod palette_length], and leave the other three quadrants black.

    Key insight: The same quadrant split does two jobs at once: the red pixel chooses which quadrant survives, and only the unique markers inside that quadrant contribute their edge-depth order to the cyclic band palette.
    """
    return _solve(np.asarray(grid, dtype=int))


def _expected_from_metadata(
    shape: tuple[int, int], bounds: tuple[int, int, int, int], active: list[tuple[int, int, int, int]]
) -> np.ndarray:
    palette = [BLACK] + [color for _d, _r, _c, color in sorted(active, key=lambda item: (item[0], item[1], item[2]))]
    out = np.zeros(shape, dtype=int)
    r0, c0, r1, c1 = bounds
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            out[r, c] = palette[_boundary_depth(r, c, bounds) % len(palette)]
    return out


def _sample_even(rng: random.Random, low: int, high: int) -> int:
    return rng.choice([n for n in range(low, high + 1) if n % 2 == 0])


def _sample_distractor_group_sizes(rng: random.Random, total: int) -> list[int]:
    options = {3: [[3]], 4: [[4], [2, 2]], 5: [[5], [2, 3]]}
    return list(rng.choice(options[total]))


def _pick_from(cells: list[tuple[int, int]], used: set[tuple[int, int]], rng: random.Random) -> tuple[int, int] | None:
    free = [cell for cell in cells if cell not in used]
    if not free:
        return None
    return rng.choice(free)


def _sample_active_records(
    bounds: tuple[int, int, int, int], colors: list[int], rng: random.Random
) -> list[tuple[int, int, int, int]] | None:
    max_depth = _max_depth(bounds)
    if max_depth + 1 < len(colors):
        return None

    for _ in range(200):
        depths = sorted(rng.sample(range(max_depth + 1), len(colors)))
        used: set[tuple[int, int]] = set()
        records = []
        for depth, color in zip(depths, colors):
            pos = _pick_from(_ring_cells(bounds, depth), used, rng)
            if pos is None:
                break
            used.add(pos)
            r, c = pos
            records.append((depth, r, c, color))
        if len(records) != len(colors):
            continue
        reading_depths = [item[0] for item in sorted(records, key=lambda item: (item[1], item[2]))]
        if reading_depths != sorted(reading_depths):
            return records
    return None


def _validate_attempt(
    grid: np.ndarray,
    quad: tuple[int, int],
    bounds: tuple[int, int, int, int],
    control: tuple[int, int],
    active: list[tuple[int, int, int, int]],
    outside_markers: list[tuple[int, int, int]],
    distractors: list[tuple[int, int, int]],
) -> bool:
    h, w = grid.shape
    counts = Counter(map(int, grid.flat))

    if h % 2 or w % 2 or max(h, w) > 30:
        return False
    if len(np.argwhere(grid == RED)) != 1 or int(grid[control]) != RED:
        return False
    if len(active) < 2:
        return False
    if len({depth for depth, _r, _c, _color in active}) != len(active):
        return False

    active_colors = {color for _d, _r, _c, color in active}
    outside_colors = {color for _r, _c, color in outside_markers}
    distractor_colors = {color for _r, _c, color in distractors}
    if active_colors & outside_colors or active_colors & distractor_colors or outside_colors & distractor_colors:
        return False

    for color in active_colors | outside_colors:
        if counts[color] != 1:
            return False
    for color in distractor_colors:
        if counts[color] < 2:
            return False

    if not any(_quadrant_of(r, c, h, w) == quad for r, c, _color in distractors):
        return False
    if not any(_quadrant_of(r, c, h, w) != quad for r, c, _color in outside_markers + distractors):
        return False

    reading_depths = [item[0] for item in sorted(active, key=lambda item: (item[1], item[2]))]
    if reading_depths == sorted(reading_depths):
        return False

    expected = _expected_from_metadata(grid.shape, bounds, active)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False
    if not np.all(actual[: bounds[0], :] == 0):
        return False
    if not np.all(actual[bounds[2] + 1 :, :] == 0):
        return False
    if not np.all(actual[bounds[0] : bounds[2] + 1, : bounds[1]] == 0):
        return False
    if not np.all(actual[bounds[0] : bounds[2] + 1, bounds[3] + 1 :] == 0):
        return False
    if len(set(map(int, actual[bounds[0] : bounds[2] + 1, bounds[1] : bounds[3] + 1].flat))) < 3:
        return False
    return True


def _generate_once(rng: random.Random) -> np.ndarray | None:
    h = _sample_even(rng, 12, 20)
    w = _sample_even(rng, 12, 20)
    quad = (rng.randrange(2), rng.randrange(2))
    bounds = _quadrant_bounds(h, w, quad)

    active_count = 3 if rng.random() < 0.45 else 2
    total_markers = active_count + (1 if active_count == 2 and rng.random() < 0.75 else 0)
    distractor_total = rng.randint(3, 5)
    distractor_sizes = _sample_distractor_group_sizes(rng, distractor_total)

    distinct_needed = total_markers + len(distractor_sizes)
    if distinct_needed > len(MARKER_POOL):
        return None
    color_pool = rng.sample(list(MARKER_POOL), distinct_needed)
    active_colors = color_pool[:active_count]
    extra_marker_colors = color_pool[active_count:total_markers]
    distractor_colors = color_pool[total_markers:]

    active = _sample_active_records(bounds, active_colors, rng)
    if active is None:
        return None

    used = {(r, c) for _d, r, c, _color in active}
    target_cells = _all_cells(bounds)
    control = _pick_from(target_cells, used, rng)
    if control is None:
        return None
    used.add(control)

    outside_cells = [(r, c) for r in range(h) for c in range(w) if _quadrant_of(r, c, h, w) != quad]
    outside_markers = []
    for color in extra_marker_colors:
        pos = _pick_from(outside_cells, used, rng)
        if pos is None:
            return None
        used.add(pos)
        outside_markers.append((pos[0], pos[1], color))

    distractors = []
    first_count = distractor_sizes[0]
    inside_first = rng.randint(1, first_count - 1)
    outside_first = first_count - inside_first

    for _ in range(inside_first):
        pos = _pick_from(target_cells, used, rng)
        if pos is None:
            return None
        used.add(pos)
        distractors.append((pos[0], pos[1], distractor_colors[0]))
    for _ in range(outside_first):
        pos = _pick_from(outside_cells, used, rng)
        if pos is None:
            return None
        used.add(pos)
        distractors.append((pos[0], pos[1], distractor_colors[0]))

    for color, count in zip(distractor_colors[1:], distractor_sizes[1:]):
        inside_count = rng.randint(0, count)
        outside_count = count - inside_count
        for _ in range(inside_count):
            pos = _pick_from(target_cells, used, rng)
            if pos is None:
                return None
            used.add(pos)
            distractors.append((pos[0], pos[1], color))
        for _ in range(outside_count):
            pos = _pick_from(outside_cells, used, rng)
            if pos is None:
                return None
            used.add(pos)
            distractors.append((pos[0], pos[1], color))

    grid = np.zeros((h, w), dtype=int)
    grid[control] = RED
    for _d, r, c, color in active:
        grid[r, c] = color
    for r, c, color in outside_markers:
        grid[r, c] = color
    for r, c, color in distractors:
        grid[r, c] = color

    if not _validate_attempt(grid, quad, bounds, control, active, outside_markers, distractors):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_once(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
