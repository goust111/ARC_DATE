import numpy as np

from arc_utils import Color, bbox_h, bbox_w, find_components, get_bg, in_bounds
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Move each maroon marker toward the blue rectangle and recolor the rectangle red.

    Input structure
    A background canvas contains one solid blue rectangular object and several isolated
    maroon marker pixels. Each marker is horizontally or vertically aligned with the
    rectangle and lies outside it.

    Rule
    Start with a background canvas, repaint the blue rectangle red, and move each
    maroon marker straight toward the rectangle along its row or column. Exactly one
    marker is strictly closer to the rectangle than all others and enters the
    rectangle boundary; every other marker lands in the adjacent exterior cell just
    outside its aligned side.

    Key insight
    The single entering marker is the unique global nearest marker to the rectangle,
    not the nearest marker on each of the four sides.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    components = find_components(g, bg=bg, connectivity=4, monochromatic=True)
    rectangles = []
    markers = []
    for comp in components:
        h = bbox_h(comp["bbox"])
        w = bbox_w(comp["bbox"])
        if comp["area"] == h * w and comp["area"] > 1:
            rectangles.append(comp)
        elif comp["area"] == 1:
            markers.append(comp)

    assert len(rectangles) == 1
    assert markers
    rect = rectangles[0]
    rect_color = int(rect["color"])
    marker_colors = {int(comp["color"]) for comp in markers}
    assert marker_colors == {Color.MAROON}
    marker_color = Color.MAROON
    assert rect_color == Color.BLUE
    assert set(map(int, g.flat)) == {bg, rect_color, marker_color}

    r0, c0, r1, c1 = map(int, rect["bbox"])
    h = bbox_h(rect["bbox"])
    w = bbox_w(rect["bbox"])

    out = np.full_like(g, bg)
    out[r0 : r1 + 1, c0 : c1 + 1] = Color.RED

    destinations = []
    marker_entries: list[tuple[int, tuple[int, int], bool]] = []
    for comp in markers:
        r, c = map(int, comp["coords"][0])
        candidates = []
        if c0 <= c <= c1 and r < r0:
            d = r0 - r
            enters = d < h
            candidates.append((r0 if enters else r0 - 1, c))
            marker_entries.append((d, (r, c), enters))
        if c0 <= c <= c1 and r > r1:
            d = r - r1
            enters = d < h
            candidates.append((r1 if enters else r1 + 1, c))
            marker_entries.append((d, (r, c), enters))
        if r0 <= r <= r1 and c < c0:
            d = c0 - c
            enters = d < w
            candidates.append((r, c0 if enters else c0 - 1))
            marker_entries.append((d, (r, c), enters))
        if r0 <= r <= r1 and c > c1:
            d = c - c1
            enters = d < w
            candidates.append((r, c1 if enters else c1 + 1))
            marker_entries.append((d, (r, c), enters))

        assert len(candidates) == 1
        dr, dc = candidates[0]
        assert in_bounds(g.shape, dr, dc)
        assert (dr, dc) not in destinations
        destinations.append((dr, dc))

    min_dist = min(dist for dist, _pos, _enters in marker_entries)
    nearest = [pos for dist, pos, _enters in marker_entries if dist == min_dist]
    entering = [pos for _dist, pos, enters in marker_entries if enters]
    assert len(nearest) == 1, "the nearest marker to the rectangle must be unique"
    assert len(entering) == 1, "exactly one marker may enter the rectangle"
    assert entering[0] == nearest[0], "the entering marker must be the global nearest"

    for r, c in destinations:
        out[r, c] = marker_color

    return out

# --------------- generator (att2) ---------------

GRID_SIZE_MIN, GRID_SIZE_MAX = 8, 16
RECT_SIDE_MIN, RECT_SIDE_MAX = 3, 5
GRID_TO_RECT_RATIO_MIN, GRID_TO_RECT_RATIO_MAX = 2, 4
MARKER_COUNT_MAX = 7
SIDE_COUNT_MIN, SIDE_COUNT_MAX = 2, 4
MIN_POSITION_SIDES_WITH_MARGIN, MIN_POSITION_MARGIN = 2, 2
CLOSE_DIST_MIN, DIST_MAX = 2, 7
KIND_CLOSE, KIND_FAR = "close", "far"
BG_COLOR, RECT_INPUT_COLOR, RECT_OUTPUT_COLOR, MARKER_COLOR = 7, 1, 2, 9
GEN_RETRIES = 300
SIDE_TOP, SIDE_BOTTOM, SIDE_LEFT, SIDE_RIGHT = "top", "bottom", "left", "right"
HORIZONTAL_SIDES = (SIDE_TOP, SIDE_BOTTOM)

def _gen_d6e50e54_margins(n, r0, c0, side):
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    return {
        SIDE_TOP: r0,
        SIDE_BOTTOM: n - r1 - 1,
        SIDE_LEFT: c0,
        SIDE_RIGHT: n - c1 - 1,
    }

def _gen_d6e50e54_count_allocations(total, caps):
    if not caps:
        return [()] if total == 0 else []

    out = []
    rest_min = len(caps) - 1
    rest_max = sum(caps[1:])
    for first in range(1, caps[0] + 1):
        remaining = total - first
        if rest_min <= remaining <= rest_max:
            for tail in _gen_d6e50e54_count_allocations(remaining, caps[1:]):
                out.append((first,) + tail)
    return out

def _gen_d6e50e54_gap_lane_sets(values, count):
    out = []

    def rec(start, chosen):
        if len(chosen) == count:
            out.append(tuple(chosen))
            return
        for i in range(start, len(values)):
            if chosen and values[i] == chosen[-1] + 1:
                continue
            rec(i + 1, chosen + [values[i]])

    rec(0, [])
    return out

def _gen_d6e50e54_lanes(values, count, single_distance):
    if single_distance:
        choices = _gen_d6e50e54_gap_lane_sets(list(values), count)
        if not choices:
            return None
        return list(random.choice(choices))
    return random.sample(list(values), count)

def _gen_d6e50e54_valid_options(margin, side_len):
    options = []
    close_max = min(side_len - 1, margin, DIST_MAX)
    for dist in range(CLOSE_DIST_MIN, close_max + 1):
        options.append((KIND_CLOSE, dist))
    if margin >= side_len:
        for dist in range(side_len, min(margin, DIST_MAX) + 1):
            options.append((KIND_FAR, dist))
    return options


def _gen_d6e50e54_side_cap(margin, side_len):
    if len({dist for _kind, dist in _gen_d6e50e54_valid_options(margin, side_len)}) <= 1:
        return (side_len + 1) // 2
    return side_len


def _gen_d6e50e54_coords(side_name, lane, kind, dist, r0, c0, r1, c1):
    if side_name == SIDE_TOP:
        dest = (r0 if kind == KIND_CLOSE else r0 - 1, lane)
        src = (r0 - dist, lane)
    elif side_name == SIDE_BOTTOM:
        dest = (r1 if kind == KIND_CLOSE else r1 + 1, lane)
        src = (r1 + dist, lane)
    elif side_name == SIDE_LEFT:
        dest = (lane, c0 if kind == KIND_CLOSE else c0 - 1)
        src = (lane, c0 - dist)
    else:
        dest = (lane, c1 if kind == KIND_CLOSE else c1 + 1)
        src = (lane, c1 + dist)
    return dest, src


def _gen_d6e50e54_assign(slots, margins, side_len, r0, c0, r1, c1):
    random.shuffle(slots)

    def rec(i, close_index, used_dest, used_src, assigned):
        if i == len(slots):
            return assigned

        side_name, lane = slots[i]
        margin = margins[side_name]
        required_kind = KIND_CLOSE if i == close_index else KIND_FAR
        options = [
            option
            for option in _gen_d6e50e54_valid_options(margin, side_len)
            if option[0] == required_kind
        ]
        random.shuffle(options)

        for kind, dist in options:
            dest, src = _gen_d6e50e54_coords(
                side_name, lane, kind, dist, r0, c0, r1, c1
            )
            if dest in used_dest or src in used_src:
                continue
            if any(abs(src[0] - old[0]) + abs(src[1] - old[1]) == 1 for old in used_src):
                continue
            result = rec(
                i + 1,
                close_index,
                used_dest | {dest},
                used_src | {src},
                assigned + [(dest, src)],
            )
            if result is not None:
                return result
        return None

    close_indices = list(range(len(slots)))
    random.shuffle(close_indices)
    for close_index in close_indices:
        result = rec(0, close_index, set(), set(), [])
        if result is not None:
            return result
    return None


def _gen_d6e50e54_choose_rect():
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    side_choices = [
        s
        for s in range(RECT_SIDE_MIN, RECT_SIDE_MAX + 1)
        if GRID_TO_RECT_RATIO_MIN <= n / s <= GRID_TO_RECT_RATIO_MAX
    ]
    side_len = random.choice(side_choices)
    positions = []
    for r0 in range(n - side_len + 1):
        for c0 in range(n - side_len + 1):
            margins = _gen_d6e50e54_margins(n, r0, c0, side_len)
            enough_sides = (
                sum(m >= MIN_POSITION_MARGIN for m in margins.values())
                >= MIN_POSITION_SIDES_WITH_MARGIN
            )
            if enough_sides and max(margins.values()) >= side_len:
                positions.append((r0, c0, margins))
    if not positions:
        return None
    r0, c0, margins = random.choice(positions)
    r1 = r0 + side_len - 1
    c1 = c0 + side_len - 1
    return n, side_len, r0, c0, r1, c1, margins


def _gen_d6e50e54_choose_side_counts(margins, side_len):
    available = [name for name, margin in margins.items() if margin >= side_len]
    far_available = [name for name in available if margins[name] >= side_len]
    if len(available) < SIDE_COUNT_MIN or not far_available:
        return None

    side_count = random.randint(SIDE_COUNT_MIN, min(SIDE_COUNT_MAX, len(available)))
    selected = [random.choice(far_available)]
    selected += random.sample(
        [name for name in available if name != selected[0]], side_count - 1
    )
    random.shuffle(selected)
    caps = [_gen_d6e50e54_side_cap(margins[name], side_len) for name in selected]
    total = random.randint(len(selected), min(MARKER_COUNT_MAX, sum(caps)))
    allocations = _gen_d6e50e54_count_allocations(total, caps)
    if not allocations:
        return None
    return selected, random.choice(allocations)


def _gen_d6e50e54_make_slots(selected, counts, margins, side_len, r0, c0, r1, c1):
    slots = []
    for side_name, count in zip(selected, counts):
        lane_values = range(c0, c1 + 1) if side_name in HORIZONTAL_SIDES else range(r0, r1 + 1)
        single_distance = _gen_d6e50e54_side_cap(margins[side_name], side_len) < side_len
        lanes = _gen_d6e50e54_lanes(lane_values, count, single_distance)
        if lanes is None:
            return None
        slots.extend((side_name, lane) for lane in lanes)
    return slots


def _gen_d6e50e54_make_grids(n, r0, c0, r1, c1, assigned):
    inp = np.full((n, n), BG_COLOR, dtype=int)
    inp[r0 : r1 + 1, c0 : c1 + 1] = RECT_INPUT_COLOR
    for _dest, src in assigned:
        inp[src] = MARKER_COLOR
    out = np.full((n, n), BG_COLOR, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = RECT_OUTPUT_COLOR
    for dest, _src in assigned:
        out[dest] = MARKER_COLOR
    return inp, out


def _gen_attempt():
    rect = _gen_d6e50e54_choose_rect()
    if rect is None:
        return None
    n, side_len, r0, c0, r1, c1, margins = rect

    side_counts = _gen_d6e50e54_choose_side_counts(margins, side_len)
    if side_counts is None:
        return None
    selected, counts = side_counts
    slots = _gen_d6e50e54_make_slots(
        selected, counts, margins, side_len, r0, c0, r1, c1
    )
    if slots is None:
        return None
    assigned = _gen_d6e50e54_assign(slots, margins, side_len, r0, c0, r1, c1)
    if assigned is None:
        return None

    inp, out = _gen_d6e50e54_make_grids(n, r0, c0, r1, c1, assigned)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input():
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate d6e50e54 input")
