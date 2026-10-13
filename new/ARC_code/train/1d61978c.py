import numpy as np

from arc_utils import Color, get_bg
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor the two diagonal foreground-bar orientations, assigning teal to the larger orientation and red to the smaller one.

    Input structure
    - The grid has one dominant background color and one foreground color.
    - Foreground pixels form one-cell-thick diagonal bars.
    - Every foreground pixel belongs to exactly one diagonal orientation.

    Rule
    - Keep the background unchanged.
    - Classify each foreground pixel by whether it has a diagonal neighbor on the NW/SE axis or the NE/SW axis.
    - Recolor the orientation with more foreground pixels teal and the other orientation red.

    Key insight
    - The output colors depend on the relative population of the two diagonal orientations, not on a fixed slash direction or position.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(fg_colors) == 1

    fg = g != bg
    out = g.copy()
    height, width = g.shape
    groups = {"nw_se": [], "ne_sw": []}

    for r0, c0 in np.argwhere(fg):
        r, c = int(r0), int(c0)
        has_nw_se = (
            (r > 0 and c > 0 and fg[r - 1, c - 1])
            or (r + 1 < height and c + 1 < width and fg[r + 1, c + 1])
        )
        has_ne_sw = (
            (r > 0 and c + 1 < width and fg[r - 1, c + 1])
            or (r + 1 < height and c > 0 and fg[r + 1, c - 1])
        )
        assert has_nw_se ^ has_ne_sw
        groups["nw_se" if has_nw_se else "ne_sw"].append((r, c))

    nw_se_count = len(groups["nw_se"])
    ne_sw_count = len(groups["ne_sw"])
    assert nw_se_count > 0 and ne_sw_count > 0
    assert nw_se_count != ne_sw_count

    larger = "nw_se" if nw_se_count > ne_sw_count else "ne_sw"
    smaller = "ne_sw" if larger == "nw_se" else "nw_se"

    for r, c in groups[larger]:
        out[r, c] = Color.TEAL
    for r, c in groups[smaller]:
        out[r, c] = Color.RED

    return out

# --------------- generator (o5x) ---------------

# Grid and palette.
GRID_HEIGHT = 16
GRID_WIDTH = 16
BACKGROUND_COLOR, FOREGROUND_COLOR = Color.ORANGE, Color.GRAY
LARGER_COLOR, SMALLER_COLOR = Color.TEAL, Color.RED

# Bar orientation labels and diagonal column deltas.
NW_SE = "nw_se"
NE_SW = "ne_sw"
ORIENTATIONS = (NW_SE, NE_SW)
ORIENTATION_DC = {NW_SE: 1, NE_SW: -1}

# Bar count, length, and foreground density bounds.
TOTAL_BARS_MIN, TOTAL_BARS_MAX = 2, 6
BAR_LENGTH_MIN, BAR_LENGTH_MAX = 2, 10
FG_PIXELS_MIN, FG_PIXELS_MAX = 5, 29

# Placement exclusion and retry controls.
BLOCK_RADIUS = 1
BLOCK_DELTAS = tuple(
    (dr, dc)
    for dr in range(-BLOCK_RADIUS, BLOCK_RADIUS + 1)
    for dc in range(-BLOCK_RADIUS, BLOCK_RADIUS + 1)
)
GENERATION_ATTEMPTS = 300


def _coords_for_bar(orientation, length, r0, c0):
    dc = ORIENTATION_DC[orientation]
    return tuple((r0 + i, c0 + dc * i) for i in range(length))


def _make_bar_catalog():
    catalog = {}
    for orientation in ORIENTATIONS:
        for length in range(BAR_LENGTH_MIN, BAR_LENGTH_MAX + 1):
            bars = []
            for r in range(GRID_HEIGHT - length + 1):
                if orientation == NW_SE:
                    cols = range(GRID_WIDTH - length + 1)
                else:
                    cols = range(length - 1, GRID_WIDTH)
                for c in cols:
                    bars.append(_coords_for_bar(orientation, length, r, c))
            catalog[(orientation, length)] = tuple(bars)
    return catalog


_BAR_CATALOG = _make_bar_catalog()


def _blocked_after(coords):
    blocked = set()
    for r, c in coords:
        for dr, dc in BLOCK_DELTAS:
            rr, cc = r + dr, c + dc
            if 0 <= rr < GRID_HEIGHT and 0 <= cc < GRID_WIDTH:
                blocked.add((rr, cc))
    return blocked


def _adjust_random_length(lengths, predicate, delta, required=True):
    choices = [i for i, length in enumerate(lengths) if predicate(length)]
    if not choices and not required:
        return False
    lengths[random.choice(choices)] += delta
    return True


def _random_lengths(n):
    lengths = [random.randint(BAR_LENGTH_MIN, BAR_LENGTH_MAX) for _ in range(n)]
    while sum(lengths) > FG_PIXELS_MAX:
        if not _adjust_random_length(lengths, lambda length: length > BAR_LENGTH_MIN, -1, required=False):
            break
    if sum(lengths) < FG_PIXELS_MIN:
        _adjust_random_length(lengths, lambda length: length < BAR_LENGTH_MAX, 1)
    random.shuffle(lengths)
    return lengths


def _break_equal_orientation_lengths(nw_lengths, ne_lengths, total_pixels):
    groups = [nw_lengths, ne_lengths]
    group = random.choice(groups)
    if total_pixels < FG_PIXELS_MAX:
        if not _adjust_random_length(group, lambda length: length < BAR_LENGTH_MAX, 1, required=False):
            other = ne_lengths if group is nw_lengths else nw_lengths
            _adjust_random_length(other, lambda length: length > BAR_LENGTH_MIN, -1)
    else:
        if not _adjust_random_length(group, lambda length: length > BAR_LENGTH_MIN, -1, required=False):
            other = ne_lengths if group is nw_lengths else nw_lengths
            _adjust_random_length(other, lambda length: length < BAR_LENGTH_MAX, 1)


def _make_bar_specs():
    total_count = random.randint(TOTAL_BARS_MIN, TOTAL_BARS_MAX)
    nw_count = random.randint(1, total_count - 1)
    ne_count = total_count - nw_count

    lengths = _random_lengths(total_count)
    nw_lengths = lengths[:nw_count]
    ne_lengths = lengths[nw_count:]

    nw_sum = sum(nw_lengths)
    ne_sum = sum(ne_lengths)
    if nw_sum == ne_sum:
        _break_equal_orientation_lengths(nw_lengths, ne_lengths, sum(lengths))

    specs = [(NW_SE, length) for length in nw_lengths]
    specs.extend((NE_SW, length) for length in ne_lengths)
    specs.sort(key=lambda item: (-item[1], random.random()))
    return specs


def _place_bars(specs):
    placed = []
    blocked = set()
    for orientation, length in specs:
        candidates = [
            coords
            for coords in _BAR_CATALOG[(orientation, length)]
            if all(coord not in blocked for coord in coords)
        ]
        if not candidates:
            return None
        coords = random.choice(candidates)
        placed.append((orientation, coords))
        blocked.update(_blocked_after(coords))
    return placed


def _paint_coords(grid, coords, color):
    rows, cols = zip(*coords)
    grid[rows, cols] = color


def _render_input_grid(placed):
    grid = np.full((GRID_HEIGHT, GRID_WIDTH), BACKGROUND_COLOR, dtype=int)
    for _, coords in placed:
        _paint_coords(grid, coords, FOREGROUND_COLOR)
    return grid


def _expected_output(grid, placed):
    counts = {orientation: 0 for orientation in ORIENTATIONS}
    for orientation, coords in placed:
        counts[orientation] += len(coords)
    if counts[NW_SE] == counts[NE_SW]:
        return None

    larger = NW_SE if counts[NW_SE] > counts[NE_SW] else NE_SW
    out = grid.copy()
    for orientation, coords in placed:
        color = LARGER_COLOR if orientation == larger else SMALLER_COLOR
        _paint_coords(out, coords, color)
    return out


def _gen_attempt():
    specs = _make_bar_specs()
    placed = _place_bars(specs)
    if placed is None:
        return None

    grid = _render_input_grid(placed)
    expected = _expected_output(grid, placed)
    if expected is None:
        return None
    try:
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input")
