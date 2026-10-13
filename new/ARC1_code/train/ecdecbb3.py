import numpy as np

from arc_utils import colors_present, get_bg
import itertools
import random


def _complete_lines(g: np.ndarray, color: int) -> tuple[list[int], list[int]]:
    H, W = g.shape
    rows = [r for r in range(H) if np.all(g[r, :] == color)]
    cols = [c for c in range(W) if np.all(g[:, c] == color)]
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each control pixel perpendicular to the full bars, marking every bar hit with a 3x3 contact marker.

    Input structure
    - The background is the dominant color.
    - Exactly one foreground color forms one or more complete parallel rows or complete parallel columns.
    - The other foreground color appears as isolated control pixels off those bars.

    Rule
    - For each control pixel, cast rays in both directions perpendicular to the bars.
    - Stop each ray at the first bar cell it reaches and color the control-to-hit segment with the control color.
    - After all rays are drawn, stamp a 3x3 block of the bar color centered on each hit, restoring the center to the control color.
    - Directions with no bar before the grid edge emit nothing.

    Key insight
    - The only complete rows or columns identify the bars and their orientation; the contact markers are layered after the rays.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    bg = get_bg(g)
    colors = colors_present(g, exclude={bg})

    bar_candidates: list[tuple[int, list[int], list[int]]] = []
    for color in colors:
        rows, cols = _complete_lines(g, color)
        if rows or cols:
            bar_candidates.append((color, rows, cols))

    assert len(bar_candidates) == 1
    bar_color, bar_rows, bar_cols = bar_candidates[0]
    assert bool(bar_rows) ^ bool(bar_cols)

    control_colors = [color for color in colors if color != bar_color]
    assert len(control_colors) == 1
    control_color = control_colors[0]
    controls = np.argwhere(g == control_color)
    assert len(controls) > 0

    out = g.copy()
    hits: set[tuple[int, int]] = set()

    if bar_rows:
        for r0, c0 in controls:
            r = int(r0)
            c = int(c0)
            for dr in (-1, 1):
                rr = r + dr
                while 0 <= rr < H:
                    if g[rr, c] == bar_color:
                        lo, hi = sorted((r, rr))
                        out[lo : hi + 1, c] = control_color
                        hits.add((rr, c))
                        break
                    rr += dr
    else:
        for r0, c0 in controls:
            r = int(r0)
            c = int(c0)
            for dc in (-1, 1):
                cc = c + dc
                while 0 <= cc < W:
                    if g[r, cc] == bar_color:
                        lo, hi = sorted((c, cc))
                        out[r, lo : hi + 1] = control_color
                        hits.add((r, cc))
                        break
                    cc += dc

    assert hits
    for r, c in hits:
        assert 0 < r < H - 1
        assert 0 < c < W - 1
        out[r - 1 : r + 2, c - 1 : c + 2] = bar_color
        out[r, c] = control_color

    return out

# --------------- generator (o5x) ---------------
# Canvas and palette.
GRID_SIZE_RANGE = (12, 19)
BACKGROUND_COLOR = 0
BAR_COLOR = 8
CTRL_COLOR = 2
# Bar and control placement.
BAR_COUNT_RANGE = (1, 2)
BAR_EDGE_MARGIN = 3
BAR_MIN_GAP = 5
CTRL_EDGE_MARGIN = 1
CTRL_COLUMN_MIN_GAP = 4
SINGLE_BAR_CONTROL_COUNT_RANGE = (2, 4)
TWO_BAR_CONTROL_COUNT_RANGE = (1, 4)
# Ray/hit constraints.
RAY_DISTANCE_RANGE = (3, 12)
HIT_COUNT_RANGE = (2, 5)
# Generation control.
ORIENTATION_CHOICES = (False, True)
MAX_GENERATION_ATTEMPTS = 300


def _gen_distance_buckets(height, rows_at_distance):
    return {
        distance: [r for r in rows_at_distance(distance) if 0 <= r < height]
        for distance in range(RAY_DISTANCE_RANGE[0], RAY_DISTANCE_RANGE[1] + 1)
    }


def _gen_spaced_choice(values, count, min_gap):
    combos = [
        combo
        for combo in itertools.combinations(values, count)
        if all(b - a >= min_gap for a, b in zip(combo, combo[1:]))
    ]
    if not combos:
        return None
    return list(random.choice(combos))


def _gen_choice_by_distance(rows_by_distance):
    distances = [d for d, rows in rows_by_distance.items() if rows]
    if not distances:
        return None
    return random.choice(rows_by_distance[random.choice(distances)])


def _gen_single_bar_controls(height, bar_row):
    rows_by_distance = _gen_distance_buckets(
        height, lambda distance: (bar_row - distance, bar_row + distance)
    )
    if not any(rows_by_distance.values()):
        return None
    n_ctrl = random.randint(*SINGLE_BAR_CONTROL_COUNT_RANGE)
    return [_gen_choice_by_distance(rows_by_distance) for _ in range(n_ctrl)]


def _gen_two_bar_controls(height, bar_rows):
    top, bottom = sorted(bar_rows)
    between_rows = list(range(top + BAR_EDGE_MARGIN, bottom - BAR_EDGE_MARGIN + 1))
    outside_by_distance = _gen_distance_buckets(
        height, lambda distance: (top - distance, bottom + distance)
    )

    feasible = []
    for n_ctrl in range(TWO_BAR_CONTROL_COUNT_RANGE[0], TWO_BAR_CONTROL_COUNT_RANGE[1] + 1):
        for n_between in range(n_ctrl + 1):
            n_outside = n_ctrl - n_between
            if n_between and not between_rows:
                continue
            if n_outside and not any(outside_by_distance.values()):
                continue
            n_hits = n_outside + 2 * n_between
            if HIT_COUNT_RANGE[0] <= n_hits <= HIT_COUNT_RANGE[1]:
                feasible.append((n_ctrl, n_between, n_outside))
    if not feasible:
        return None

    _, n_between, n_outside = random.choice(feasible)
    control_rows = [random.choice(between_rows) for _ in range(n_between)]
    control_rows += [_gen_choice_by_distance(outside_by_distance) for _ in range(n_outside)]
    random.shuffle(control_rows)
    return control_rows


def _gen_bar_rows(height):
    bar_candidates = list(range(BAR_EDGE_MARGIN, height - BAR_EDGE_MARGIN))
    pairs = [
        pair
        for pair in itertools.combinations(bar_candidates, 2)
        if pair[1] - pair[0] >= BAR_MIN_GAP
    ]

    n_bars = random.randint(*BAR_COUNT_RANGE)
    if n_bars == 1:
        return [random.choice(bar_candidates)]
    return list(random.choice(pairs))


def _draw_rays(output, bar_rows, controls):
    hits = set()
    if len(bar_rows) == 1:
        bar_row = bar_rows[0]
        for r, c in controls:
            lo, hi = sorted((r, bar_row))
            output[lo : hi + 1, c] = CTRL_COLOR
            hits.add((bar_row, c))
        return hits

    top, bottom = sorted(bar_rows)
    for r, c in controls:
        if top < r < bottom:
            output[top : bottom + 1, c] = CTRL_COLOR
            hits.add((top, c))
            hits.add((bottom, c))
        elif r < top:
            output[r : top + 1, c] = CTRL_COLOR
            hits.add((top, c))
        else:
            output[bottom : r + 1, c] = CTRL_COLOR
            hits.add((bottom, c))
    return hits


def _render_horizontal_scene(height, width, bar_rows, controls):
    output = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    output[bar_rows, :] = BAR_COLOR
    for r, c in _draw_rays(output, bar_rows, controls):
        output[r - 1 : r + 2, c - 1 : c + 2] = BAR_COLOR
        output[r, c] = CTRL_COLOR

    inp = np.full_like(output, BACKGROUND_COLOR)
    inp[bar_rows, :] = BAR_COLOR
    rows, cols = zip(*controls)
    inp[rows, cols] = CTRL_COLOR
    return inp, output


def _gen_horizontal_scene(height, width):
    bar_rows = _gen_bar_rows(height)
    if len(bar_rows) == 1:
        control_rows = _gen_single_bar_controls(height, bar_rows[0])
    else:
        control_rows = _gen_two_bar_controls(height, bar_rows)
    if control_rows is None:
        return None

    columns = _gen_spaced_choice(
        list(range(CTRL_EDGE_MARGIN, width - CTRL_EDGE_MARGIN)),
        len(control_rows),
        CTRL_COLUMN_MIN_GAP,
    )
    if columns is None:
        return None

    controls = list(zip(control_rows, columns))
    return _render_horizontal_scene(height, width, bar_rows, controls)


def _gen_attempt():
    height = random.randint(*GRID_SIZE_RANGE)
    width = random.randint(*GRID_SIZE_RANGE)
    vertical = random.choice(ORIENTATION_CHOICES)

    if vertical:
        made = _gen_horizontal_scene(width, height)
        if made is None:
            return None
        inp_t, output_t = made
        inp = inp_t.T.copy()
        output = output_t.T.copy()
    else:
        made = _gen_horizontal_scene(height, width)
        if made is None:
            return None
        inp, output = made

    if np.array_equal(main(inp), output):
        return inp
    return None


def generate_input():
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid ecdecbb3 input")
