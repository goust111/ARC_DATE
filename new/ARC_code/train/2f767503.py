from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg
import itertools
import math
import random


def _find_marker_bar(g: np.ndarray, bg: int) -> tuple[int, str, tuple[int, int]]:
    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    candidates: list[tuple[int, str, tuple[int, int]]] = []

    for color in colors:
        coords = np.argwhere(g == int(color))
        if coords.shape[0] != 3:
            continue

        rows = sorted(set(map(int, coords[:, 0])))
        cols = sorted(set(map(int, coords[:, 1])))
        if len(rows) == 1 and len(cols) == 3 and cols == list(range(cols[0], cols[0] + 3)):
            candidates.append((int(color), "horizontal", (rows[0], cols[1])))
        elif len(cols) == 1 and len(rows) == 3 and rows == list(range(rows[0], rows[0] + 3)):
            candidates.append((int(color), "vertical", (rows[1], cols[0])))

    assert len(candidates) == 1, "expected a unique three-pixel marker bar"
    return candidates[0]


def _roles(g: np.ndarray) -> tuple[int, int, tuple[int, int], tuple[int, int]]:
    bg = get_bg(g)
    colors = [int(color) for color in np.unique(g) if int(color) != int(bg)]
    assert len(colors) == 3, "expected target color, marker bar color, and control-pixel color"

    bar_color, axis, center = _find_marker_bar(g, bg)
    control_candidates: list[tuple[int, tuple[int, int]]] = []
    for color in colors:
        if int(color) == int(bar_color):
            continue
        coords = np.argwhere(g == int(color))
        if coords.shape[0] != 1:
            continue
        r, c = map(int, coords[0])
        if abs(r - center[0]) + abs(c - center[1]) == 1:
            control_candidates.append((int(color), (r, c)))

    assert len(control_candidates) == 1, "expected one singleton control pixel adjacent to the bar center"
    control_color, control = control_candidates[0]

    target_colors = [color for color in colors if color not in (bar_color, control_color)]
    assert len(target_colors) == 1, "expected one remaining target color"

    dr = center[0] - control[0]
    dc = center[1] - control[1]
    assert abs(dr) + abs(dc) == 1, "control pixel must touch the marker bar center"
    if axis == "horizontal":
        assert dc == 0, "horizontal marker bars emit vertically"
    else:
        assert dr == 0, "vertical marker bars emit horizontally"

    return int(bg), int(target_colors[0]), center, (dr, dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase each target-color object crossed by the ray pointed by the bar-and-control marker.

    Input structure:
    - The background is the dominant color.
    - There is one three-pixel straight marker bar and one singleton control pixel adjacent to the bar center.
    - The remaining non-background color forms multiple 4-connected target objects.

    Rule:
    - Cast a straight ray from the marker bar center in the direction away from the adjacent control pixel.
    - Find every 4-connected target-color object that intersects that ray.
    - Recolor all pixels of those hit target objects to the background; all other pixels stay unchanged.

    Key insight:
    - The whole object is erased when the center ray touches any of its pixels; the full marker bar width is not swept.
    """
    g = np.asarray(grid, dtype=int)
    bg, target_color, center, step = _roles(g)
    height, width = g.shape

    ray: set[tuple[int, int]] = set()
    r = center[0] + step[0]
    c = center[1] + step[1]
    while 0 <= r < height and 0 <= c < width:
        ray.add((int(r), int(c)))
        r += step[0]
        c += step[1]

    target_objects = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=[target_color])
    hit_objects = [
        obj
        for obj in target_objects
        if any(tuple(map(int, coord)) in ray for coord in obj["coords"])
    ]
    assert hit_objects, "expected at least one target object intersecting the marker ray"

    out = g.copy()
    for obj in hit_objects:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = int(bg)
    return out

# --------------- generator (o5x) ---------------

# Palette.
BG = 7
TARGET = 4
BAR = 5
CONTROL = 9

# Canvas size bounds.
HEIGHT_MIN, HEIGHT_MAX = 9, 14
WIDTH_MIN, WIDTH_MAX = 10, 13

# Marker and ray geometry.
BAR_CENTER_MARGIN = 2
RAY_LEN_MIN, RAY_LEN_MAX = 6, 8
NEIGHBOR_STEPS_4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
AXIS_HORIZONTAL, AXIS_VERTICAL = "horizontal", "vertical"

# Target component counts, density, and size caps.
DENSITY_MIN, DENSITY_MAX = 0.17, 0.21
OBJECT_COUNT_MIN, OBJECT_COUNT_MAX = 10, 18
EXPECTED_HIT_OBJECTS, CONNECTIVITY = 2, 4
HIT_EXTRA_CAP, NON_HIT_EXTRA_CAP = 6, 4

# Retry budget for rejection sampling.
GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in NEIGHBOR_STEPS_4:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _ray_cells(center: tuple[int, int], step: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = center[0] + step[0], center[1] + step[1]
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += step[0]
        c += step[1]
    return cells


def _marker_candidates(
    h: int,
    w: int,
) -> list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]] = []

    for length in range(RAY_LEN_MIN, RAY_LEN_MAX + 1):
        for c in range(BAR_CENTER_MARGIN, w - BAR_CENTER_MARGIN):
            if 1 <= length <= h - 2:
                candidates.append((AXIS_HORIZONTAL, (length, c), (-1, 0), (length + 1, c)))
            r = h - 1 - length
            if 1 <= r <= h - 2:
                candidates.append((AXIS_HORIZONTAL, (r, c), (1, 0), (r - 1, c)))

        for r in range(BAR_CENTER_MARGIN, h - BAR_CENTER_MARGIN):
            if 1 <= length <= w - 2:
                candidates.append((AXIS_VERTICAL, (r, length), (0, -1), (r, length + 1)))
            c = w - 1 - length
            if 1 <= c <= w - 2:
                candidates.append((AXIS_VERTICAL, (r, c), (0, 1), (r, c - 1)))

    return candidates


def _bar_cells(axis: str, center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    if axis == AXIS_HORIZONTAL:
        return [(r, c - 1), (r, c), (r, c + 1)]
    return [(r - 1, c), (r, c), (r + 1, c)]


def _place_marker(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], set[tuple[int, int]], list[tuple[int, int]]] | None:
    """Draw one marker and return center, ray step, occupied marker cells, and ray cells."""
    h, w = g.shape
    candidates = _marker_candidates(h, w)
    if not candidates:
        return None

    axis, center, step, control = random.choice(candidates)
    marker: set[tuple[int, int]] = {control}
    bar_cells = _bar_cells(axis, center)
    marker.update(bar_cells)

    for rr, cc in bar_cells:
        g[rr, cc] = BAR
    g[control] = CONTROL
    return center, step, marker, _ray_cells(center, step, h, w)


def _feasible_size_choices(area: int) -> list[tuple[int, int, int]]:
    lo = math.ceil(area * DENSITY_MIN)
    hi = math.floor(area * DENSITY_MAX)
    choices: list[tuple[int, int, int]] = []
    for total_pixels in range(lo, hi + 1):
        for count in range(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX + 1):
            if count > total_pixels:
                continue
            extras = total_pixels - count
            caps = [HIT_EXTRA_CAP] * EXPECTED_HIT_OBJECTS + [NON_HIT_EXTRA_CAP] * (count - EXPECTED_HIT_OBJECTS)
            max_clustered = (count - 1) // EXPECTED_HIT_OBJECTS
            for clustered in range(max_clustered + 1):
                if clustered > extras:
                    continue
                if extras <= sum(sorted(caps, reverse=True)[:clustered]):
                    choices.append((total_pixels, count, clustered))
    return choices


def _choose_sizes(area: int) -> tuple[list[int], list[int]] | None:
    choices = _feasible_size_choices(area)
    if not choices:
        return None

    total_pixels, count, clustered = random.choice(choices)
    extras = total_pixels - count
    caps = [HIT_EXTRA_CAP] * EXPECTED_HIT_OBJECTS + [NON_HIT_EXTRA_CAP] * (count - EXPECTED_HIT_OBJECTS)

    if clustered == 0:
        if extras != 0:
            return None
        return [1] * EXPECTED_HIT_OBJECTS, [1] * (count - EXPECTED_HIT_OBJECTS)

    combos = [
        combo
        for combo in itertools.combinations(range(count), clustered)
        if extras <= sum(caps[i] for i in combo)
    ]
    if not combos:
        return None

    chosen = random.choice(combos)
    sizes = [1] * count
    for idx in chosen:
        sizes[idx] += 1
    remaining = extras - clustered

    while remaining > 0:
        available = [idx for idx in chosen if sizes[idx] < caps[idx] + 1]
        if not available:
            return None
        idx = random.choice(available)
        sizes[idx] += 1
        remaining -= 1

    return sizes[:EXPECTED_HIT_OBJECTS], sizes[EXPECTED_HIT_OBJECTS:]


def _touches_existing_target(
    r: int,
    c: int,
    existing: np.ndarray,
    current: set[tuple[int, int]],
) -> bool:
    h, w = existing.shape
    for rr, cc in _neighbors4(r, c, h, w):
        if existing[rr, cc] and (rr, cc) not in current:
            return True
    return False


def _place_object(
    g: np.ndarray,
    size: int,
    forbidden: set[tuple[int, int]],
    seed_pool: list[tuple[int, int]] | None = None,
) -> set[tuple[int, int]] | None:
    """Grow one 4-connected target object without 4-touching prior target objects."""
    h, w = g.shape
    existing = g == TARGET

    if seed_pool is None:
        seed_pool = [(r, c) for r in range(h) for c in range(w)]

    seeds = [
        (r, c)
        for r, c in seed_pool
        if (r, c) not in forbidden
        and g[r, c] == BG
        and not _touches_existing_target(r, c, existing, set())
    ]
    if not seeds:
        return None

    current = {random.choice(seeds)}
    while len(current) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in current:
            frontier.update(_neighbors4(r, c, h, w))

        options = [
            (r, c)
            for r, c in frontier
            if (r, c) not in current
            and (r, c) not in forbidden
            and g[r, c] == BG
            and not _touches_existing_target(r, c, existing, current)
        ]
        if not options:
            return None
        current.add(random.choice(options))

    for r, c in current:
        g[r, c] = TARGET
    return current


def _hit_count(g: np.ndarray, ray: set[tuple[int, int]]) -> int:
    objects = find_components(g, bg=BG, connectivity=CONNECTIVITY, monochromatic=True, colors=[TARGET])
    return sum(any((int(r), int(c)) in ray for r, c in obj["coords"]) for obj in objects)


def _place_objects(
    g: np.ndarray,
    sizes: list[int],
    forbidden: set[tuple[int, int]],
    seed_pool: list[tuple[int, int]] | None = None,
) -> bool:
    for size in sorted(sizes, reverse=True):
        if _place_object(g, size, forbidden, seed_pool) is None:
            return False
    return True


def _valid_generated_pair(inp: np.ndarray, out: np.ndarray, ray: set[tuple[int, int]]) -> bool:
    objects = find_components(inp, bg=BG, connectivity=CONNECTIVITY, monochromatic=True, colors=[TARGET])
    density = float(np.count_nonzero(inp == TARGET)) / inp.size
    if len(objects) < OBJECT_COUNT_MIN or len(objects) > OBJECT_COUNT_MAX:
        return False
    if not (DENSITY_MIN <= density <= DENSITY_MAX):
        return False
    if _hit_count(inp, ray) != EXPECTED_HIT_OBJECTS:
        return False

    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, out)


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(HEIGHT_MIN, HEIGHT_MAX)
    w = random.randint(WIDTH_MIN, WIDTH_MAX)
    out = np.full((h, w), BG, dtype=int)

    marker = _place_marker(out)
    if marker is None:
        return None
    _, _, marker_cells, ray_cells = marker
    ray = set(ray_cells)

    sizes = _choose_sizes(out.size)
    if sizes is None:
        return None
    hit_sizes, non_hit_sizes = sizes

    blocked_for_output = marker_cells | ray
    if not _place_objects(out, non_hit_sizes, blocked_for_output):
        return None

    inp = out.copy()
    if not _place_objects(inp, hit_sizes, marker_cells, ray_cells):
        return None

    if not _valid_generated_pair(inp, out, ray):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 2f767503 input")
