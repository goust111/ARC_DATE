import random

import numpy as np


BACKGROUND = 0
MAX_GENERATION_ATTEMPTS = 300


def _horizontal_objects(grid: np.ndarray):
    """Return monochrome horizontal walls and isolated foreground pixels."""
    walls = []
    drops = []
    for r in range(grid.shape[0]):
        cols = np.flatnonzero(grid[r] != BACKGROUND)
        if cols.size == 0:
            continue
        cuts = np.flatnonzero(np.diff(cols) > 1) + 1
        for run in np.split(cols, cuts):
            c0, c1 = int(run[0]), int(run[-1])
            colors = np.unique(grid[r, c0 : c1 + 1])
            assert colors.size == 1, "each horizontal object must be monochrome"
            color = int(colors[0])
            if len(run) == 1:
                drops.append((r, c0, color))
            else:
                walls.append((r, c0, c1, color))
    return walls, drops


def main(grid: np.ndarray) -> np.ndarray:
    """Drop each top pixel through recoloring walls until its current color matches a wall.

    Input structure:
    - A black-background grid has isolated colored pixels in its upper part, with
      at most one pixel in each column.
    - Below them are non-touching monochrome horizontal wall segments at distinct
      heights and over different column intervals.

    Rule:
    - For each pixel, inspect only the walls crossed in its own column, from top
      to bottom.  A differently colored wall recolors the pixel and lets it pass;
      a wall matching its current color stops it one cell above that wall.
    - If no wall stops it, place it on the bottom row in its final color.  Erase
      every original pixel and preserve every wall.

    Key insight:
    - Matching uses the falling pixel's repeatedly updated current color, so an
      earlier wall can make the next same-colored wall become a blocker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0 and max(g.shape) <= 30, "expected an ARC-sized grid"
    assert np.all((0 <= g) & (g <= 9)), "expected ARC palette colors"

    walls, drops = _horizontal_objects(g)
    assert walls, "expected at least one horizontal wall"
    assert drops, "expected at least one isolated falling pixel"
    assert len({r for r, _c0, _c1, _color in walls}) == len(walls), (
        "walls must occupy distinct rows"
    )
    assert len({c for _r, c, _color in drops}) == len(drops), (
        "each falling column may contain at most one pixel"
    )

    first_wall_row = min(r for r, _c0, _c1, _color in walls)
    assert all(r < first_wall_row for r, _c, _color in drops), (
        "all falling pixels must start above all walls"
    )

    out = g.copy()
    for r, c, _color in drops:
        out[r, c] = BACKGROUND

    h, _w = g.shape
    occupied_destinations = set()
    for _r, c, initial_color in drops:
        current_color = initial_color
        destination = (h - 1, c)
        crossed = sorted(
            (wall for wall in walls if wall[1] <= c <= wall[2]),
            key=lambda wall: wall[0],
        )
        for wall_r, _c0, _c1, wall_color in crossed:
            if wall_color == current_color:
                destination = (wall_r - 1, c)
                break
            current_color = wall_color

        assert destination not in occupied_destinations
        assert int(out[destination]) == BACKGROUND, (
            "a falling pixel's destination must be background"
        )
        out[destination] = current_color
        occupied_destinations.add(destination)

    return out


def _segment_containing(zone, target, used):
    lo, hi = zone
    candidates = [
        (c0, c1)
        for c0 in range(lo, target + 1)
        for c1 in range(target, hi + 1)
        if c1 - c0 + 1 >= 3 and (c0, c1) not in used
    ]
    if not candidates:
        return None
    segment = random.choice(candidates)
    used.add(segment)
    return segment


def _sample_separated_rows(count, h):
    candidates = list(range(6, h - 2))
    for _ in range(100):
        rows = sorted(random.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(rows, rows[1:])):
            return rows
    return None


def _explicit_oracle(grid, walls, drops):
    """Simulate from the generator's object list, independently of parsing."""
    out = grid.copy()
    for drop in drops:
        out[drop[0], drop[1]] = BACKGROUND
    for _r, c, initial in drops:
        color = initial
        dest_r = grid.shape[0] - 1
        for wr, c0, c1, wall_color in sorted(walls):
            if not c0 <= c <= c1:
                continue
            if color == wall_color:
                dest_r = wr - 1
                break
            color = wall_color
        out[dest_r, c] = color
    return out


def _encounters(walls, drop):
    _r, c, _color = drop
    return sorted((wall for wall in walls if wall[1] <= c <= wall[2]))


def _validate_generated(grid, walls, drops, roles, expected):
    """Check structure, semantic branches, and anti-shortcut dependencies."""
    h, w = grid.shape
    if not (h <= 30 and w <= 30 and h >= 1 and w >= 1):
        return False

    try:
        parsed_walls, parsed_drops = _horizontal_objects(grid)
    except AssertionError:
        return False
    if sorted(parsed_walls) != sorted(walls) or sorted(parsed_drops) != sorted(drops):
        return False
    if len({r for r, _c0, _c1, _color in walls}) != len(walls):
        return False
    if any(b[0] - a[0] < 2 for a, b in zip(sorted(walls), sorted(walls)[1:])):
        return False
    if len({c for _r, c, _color in drops}) != len(drops):
        return False

    # Every falling pixel is genuinely isolated in its input position.
    for r, c, _color in drops:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr, cc] != BACKGROUND:
                    return False

    chain = roles["chain"]
    chain_hits = _encounters(walls, chain)
    if not (
        len(chain_hits) == 2
        and chain[2] != chain_hits[0][3]
        and chain_hits[0][3] == chain_hits[1][3]
    ):
        return False
    chain_dest = (chain_hits[1][0] - 1, chain[1])
    if int(expected[chain_dest]) != chain_hits[0][3]:
        return False

    # A solver that compares every wall only with the initial color misses this stop.
    initial_only_stop = next(
        (wall[0] - 1 for wall in chain_hits if wall[3] == chain[2]), h - 1
    )
    if initial_only_stop == chain_dest[0]:
        return False

    immediate = roles["immediate"]
    immediate_hits = _encounters(walls, immediate)
    if not (
        len(immediate_hits) == 1
        and immediate[2] == immediate_hits[0][3]
        and int(expected[immediate_hits[0][0] - 1, immediate[1]]) == immediate[2]
    ):
        return False

    dyed_bottom = roles["dyed_bottom"]
    dye_hits = _encounters(walls, dyed_bottom)
    dye_colors = [wall[3] for wall in dye_hits]
    if not (
        dye_hits
        and dyed_bottom[2] != dye_colors[0]
        and all(a != b for a, b in zip(dye_colors, dye_colors[1:]))
        and int(expected[h - 1, dyed_bottom[1]]) == dye_colors[-1]
        and dye_colors[-1] != dyed_bottom[2]
    ):
        return False

    free = roles["free"]
    if _encounters(walls, free) or int(expected[h - 1, free[1]]) != free[2]:
        return False

    # Walls are unchanged, all sources are erased, and the solver agrees with
    # the generator's explicit semantic simulation.
    for wr, c0, c1, color in walls:
        if not np.all(expected[wr, c0 : c1 + 1] == color):
            return False
    if any(expected[r, c] != BACKGROUND for r, c, _color in drops):
        return False
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return np.array_equal(actual, expected) and not np.array_equal(grid, expected)


def _generate_attempt():
    h = random.randint(20, 28)
    w = random.randint(22, 30)
    bottom_wall_count = random.randint(1, 3)
    wall_count = 3 + bottom_wall_count

    widths = [random.randint(4, 6) for _ in range(3)]
    block_width = sum(widths) + 4  # two two-column gaps
    if block_width > w - 2:
        return None
    start = random.randint(1, w - block_width - 1)
    physical_zones = []
    cursor = start
    for width in widths:
        physical_zones.append((cursor, cursor + width - 1))
        cursor += width + 2

    labels = ["chain", "immediate", "dyed_bottom"]
    random.shuffle(labels)
    zones = dict(zip(labels, physical_zones))
    targets = {
        label: random.randint(zone[0] + 1, zone[1] - 1)
        for label, zone in zones.items()
    }

    gaps = []
    ordered_zones = sorted(physical_zones)
    for left, right in zip(ordered_zones, ordered_zones[1:]):
        gaps.extend(range(left[1] + 1, right[0]))
    free_col = random.choice(gaps)

    rows = _sample_separated_rows(wall_count, h)
    if rows is None:
        return None
    random.shuffle(rows)
    chain_rows = sorted(rows[:2])
    immediate_row = rows[2]
    bottom_rows = sorted(rows[3:])

    palette = random.sample(range(1, 10), 9)
    chain_wall_color = palette[0]
    chain_initial_color = palette[1]
    immediate_color = palette[2]
    bottom_colors = palette[3 : 3 + bottom_wall_count]
    bottom_initial_color = palette[6]
    free_color = palette[7]

    walls = []
    used = set()
    for row in chain_rows:
        segment = _segment_containing(zones["chain"], targets["chain"], used)
        if segment is None:
            return None
        walls.append((row, segment[0], segment[1], chain_wall_color))

    segment = _segment_containing(zones["immediate"], targets["immediate"], used)
    if segment is None:
        return None
    walls.append((immediate_row, segment[0], segment[1], immediate_color))

    for row, color in zip(bottom_rows, bottom_colors):
        segment = _segment_containing(zones["dyed_bottom"], targets["dyed_bottom"], used)
        if segment is None:
            return None
        walls.append((row, segment[0], segment[1], color))

    fixed = [
        (targets["chain"], chain_initial_color, "chain"),
        (targets["immediate"], immediate_color, "immediate"),
        (targets["dyed_bottom"], bottom_initial_color, "dyed_bottom"),
        (free_col, free_color, "free"),
    ]
    random.shuffle(fixed)
    drops = []
    roles = {}
    for index, (col, color, role) in enumerate(fixed):
        row = (index + random.randint(0, 3)) % 4
        drop = (row, col, color)
        drops.append(drop)
        roles[role] = drop

    # Vary the number of falling objects while retaining all four semantic cases.
    optional_count = random.randint(0, 3)
    candidates = [
        c
        for c in range(1, w - 1)
        if all(abs(c - existing[1]) >= 2 for existing in drops)
    ]
    random.shuffle(candidates)
    selected_optional = []
    for col in candidates:
        if len(selected_optional) >= optional_count:
            break
        if any(abs(col - chosen) < 2 for chosen in selected_optional):
            continue
        selected_optional.append(col)
    for col in selected_optional:
        row = random.randint(0, 3)
        color = random.randint(1, 9)
        drops.append((row, col, color))

    grid = np.zeros((h, w), dtype=int)
    for wr, c0, c1, color in walls:
        grid[wr, c0 : c1 + 1] = color
    for r, c, color in drops:
        if grid[r, c] != BACKGROUND:
            return None
        grid[r, c] = color

    expected = _explicit_oracle(grid, walls, drops)
    if not _validate_generated(grid, walls, drops, roles, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    """Generate a varied, non-degenerate instance covering all rule branches."""
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 'pass-through dyes' input")
