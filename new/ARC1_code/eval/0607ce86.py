from collections import Counter

import numpy as np

from arc_utils import get_bg


def _dense_runs(counts: np.ndarray) -> list[tuple[int, int]]:
    values = sorted(set(map(int, counts)))
    assert len(values) >= 2

    gaps = [(b - a, b) for a, b in zip(values, values[1:])]
    max_gap = max(gap for gap, _ in gaps)
    thresholds = [upper for gap, upper in gaps if gap == max_gap]
    assert len(thresholds) == 1

    high = np.asarray(counts) >= thresholds[0]
    runs = []
    i = 0
    while i < len(high):
        if not high[i]:
            i += 1
            continue
        j = i
        while j < len(high) and high[j]:
            j += 1
        runs.append((i, j))
        i = j

    assert len(runs) >= 2
    assert len({b - a for a, b in runs}) == 1
    assert len({runs[i + 1][0] - runs[i][1] for i in range(len(runs) - 1)}) == 1
    return runs


def _mode_color(values: list[int]) -> int:
    counts = Counter(values)
    max_count = max(counts.values())
    modes = [color for color, count in counts.items() if count == max_count]
    assert len(modes) == 1
    return modes[0]


def main(grid: np.ndarray) -> np.ndarray:
    """Clean a noisy regular array by reconstructing its repeated rectangular sprite and erasing everything outside the slots.

    Input structure
    - The background is the most common color.
    - Non-background cells form a regular grid of solid rectangular sprite slots, plus sparse stray foreground noise.
    - Every slot contains a corrupted copy of the same sprite, preserving non-background occupancy.

    Rule
    - Find the dense row bands and dense column bands of the slot array.
    - For each relative position inside the sprite rectangle, take the unique majority color across all slots.
    - Fill every slot with that recovered sprite and set all other cells to background.

    Key insight
    - The intended sprite support is the Cartesian product of dense row bands and dense column bands; noise is too sparse to enter the high-density bands consistently.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    fg = g != bg

    row_runs = _dense_runs(fg.sum(axis=1))
    col_runs = _dense_runs(fg.sum(axis=0))
    slot_h = row_runs[0][1] - row_runs[0][0]
    slot_w = col_runs[0][1] - col_runs[0][0]
    assert slot_h > 1 and slot_w > 1
    assert len(row_runs) * len(col_runs) >= 4

    blocks = []
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            block = g[r0:r1, c0:c1]
            assert block.shape == (slot_h, slot_w)
            assert np.all(block != bg)
            blocks.append(block)

    sprite = np.empty((slot_h, slot_w), dtype=int)
    for r in range(slot_h):
        for c in range(slot_w):
            sprite[r, c] = _mode_color([int(block[r, c]) for block in blocks])

    out = np.full_like(g, bg)
    for r0, r1 in row_runs:
        for c0, c1 in col_runs:
            out[r0:r1, c0:c1] = sprite
    return out

# --------------- generator (o5x) ---------------

# Canvas and slot layout.
GRID_HEIGHT_MIN, GRID_HEIGHT_MAX = 21, 24
GRID_WIDTH = 22
SLOT_ROWS = 3
SLOT_COL_OPTIONS = (2, 3)
SLOT_SIZE_RANGE = range(4, 7)
ROW_GAP_OPTIONS, COL_GAP_OPTIONS = (1, 2), (1, 2)
TOP_MARGIN = 1
LEFT_MARGIN_OPTIONS = (1, 2)
BOTTOM_MARGIN_RANGE = range(3, 6)
RIGHT_MARGIN_RANGE = range(4, 7)

# Sprite construction.
SPRITE_COLOR_COUNT_RANGE = (3, 5)
FOREGROUND_COLORS = np.arange(1, 10)
REGION_SPLIT_MIN_SIZE = 2
REGION_SPLIT_MIN_FRACTION, REGION_SPLIT_MAX_FRACTION = 0.35, 0.65
SPRITE_EDGE_JITTER_PROB = 0.55
SPRITE_EDGE_JITTER_STEPS_MIN, SPRITE_EDGE_JITTER_MIN_STOP = 1, 2
SPRITE_EDGE_JITTER_AREA_DIVISOR = 5
NEIGHBOR_DELTAS_4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Slot corruption.
SLOT_CORRUPT_RATE_RANGE = (0.07, 0.09)
CORRUPT_CAP_MIN, CORRUPT_CAP_MARGIN = 1, 1

# Outside-slot noise.
OUTSIDE_NOISE_COUNT_RANGE = (26, 33)
NOISE_LINE_CAP = 3
NOISE_SPLIT_DENOMINATOR, NOISE_SPLIT_MAX_NUMERATOR = 3, 2
ROW_AXIS, COL_AXIS = 0, 1

# Generation retry budget.
GEN_RETRY_LIMIT = 300


def _gen_layouts():
    layouts = []
    for slot_cols in SLOT_COL_OPTIONS:
        for slot_h in SLOT_SIZE_RANGE:
            for slot_w in SLOT_SIZE_RANGE:
                for row_gap in ROW_GAP_OPTIONS:
                    for col_gap in COL_GAP_OPTIONS:
                        for left_margin in LEFT_MARGIN_OPTIONS:
                            for bottom_margin in BOTTOM_MARGIN_RANGE:
                                for right_margin in RIGHT_MARGIN_RANGE:
                                    height = TOP_MARGIN + SLOT_ROWS * slot_h + (SLOT_ROWS - 1) * row_gap + bottom_margin
                                    width = left_margin + slot_cols * slot_w + (slot_cols - 1) * col_gap + right_margin
                                    if GRID_HEIGHT_MIN <= height <= GRID_HEIGHT_MAX and width == GRID_WIDTH:
                                        layouts.append((height, width, slot_cols, slot_h, slot_w, row_gap, col_gap, left_margin))
    return layouts


LAYOUTS = _gen_layouts()


def _gen_choose_layout(rng):
    layout = LAYOUTS[int(rng.integers(len(LAYOUTS)))]
    height, width, slot_cols, slot_h, slot_w, row_gap, col_gap, left_margin = layout
    row_starts = [TOP_MARGIN + i * (slot_h + row_gap) for i in range(SLOT_ROWS)]
    col_starts = [left_margin + i * (slot_w + col_gap) for i in range(slot_cols)]
    return height, width, slot_h, slot_w, row_starts, col_starts


def _gen_cut_span(length, rng):
    lo = max(1, int(np.floor(length * REGION_SPLIT_MIN_FRACTION)))
    hi = min(length - 1, int(np.ceil(length * REGION_SPLIT_MAX_FRACTION)))
    return int(rng.integers(lo, hi + 1))


def _gen_split_region(region, rng):
    r0, r1, c0, c1 = region
    h, w = r1 - r0, c1 - c0
    choices = []
    if h >= REGION_SPLIT_MIN_SIZE:
        choices.append("h")
    if w >= REGION_SPLIT_MIN_SIZE:
        choices.append("v")
    if not choices:
        return None
    orient = choices[int(rng.integers(len(choices)))]
    if orient == "h":
        cut = _gen_cut_span(h, rng)
        return [(r0, r0 + cut, c0, c1), (r0 + cut, r1, c0, c1)]

    cut = _gen_cut_span(w, rng)
    return [(r0, r1, c0, c0 + cut), (r0, r1, c0 + cut, c1)]


def _gen_partition_regions(slot_h, slot_w, n_regions, rng):
    """Split the full sprite box into exactly n_regions non-empty rectangles."""
    regions = [(0, slot_h, 0, slot_w)]
    while len(regions) < n_regions:
        splittable = [
            i
            for i, (r0, r1, c0, c1) in enumerate(regions)
            if r1 - r0 >= REGION_SPLIT_MIN_SIZE or c1 - c0 >= REGION_SPLIT_MIN_SIZE
        ]
        if not splittable:
            return None
        areas = np.array(
            [(regions[i][1] - regions[i][0]) * (regions[i][3] - regions[i][2]) for i in splittable],
            dtype=float,
        )
        idx = splittable[int(rng.choice(len(splittable), p=areas / areas.sum()))]
        region = regions.pop(idx)
        pieces = _gen_split_region(region, rng)
        if pieces is None:
            return None
        regions.extend(pieces)
    return regions


def _gen_jitter_sprite_edges(sprite, colors, rng):
    """Move a few boundary cells without removing any sprite color."""
    if rng.random() >= SPRITE_EDGE_JITTER_PROB:
        return

    slot_h, slot_w = sprite.shape
    counts = {color: int(np.sum(sprite == color)) for color in colors}
    stop = max(SPRITE_EDGE_JITTER_MIN_STOP, slot_h * slot_w // SPRITE_EDGE_JITTER_AREA_DIVISOR)
    for _ in range(int(rng.integers(SPRITE_EDGE_JITTER_STEPS_MIN, stop))):
        r = int(rng.integers(slot_h))
        c = int(rng.integers(slot_w))
        old = int(sprite[r, c])
        nbrs = []
        for dr, dc in NEIGHBOR_DELTAS_4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < slot_h and 0 <= cc < slot_w and sprite[rr, cc] != old:
                nbrs.append(int(sprite[rr, cc]))
        if nbrs and counts[old] > 1:
            new = nbrs[int(rng.integers(len(nbrs)))]
            sprite[r, c] = new
            counts[old] -= 1
            counts[new] += 1


def _gen_build_sprite(slot_h, slot_w, rng):
    n_colors = int(rng.integers(*SPRITE_COLOR_COUNT_RANGE))
    colors = rng.choice(FOREGROUND_COLORS, size=n_colors, replace=False).astype(int).tolist()
    regions = _gen_partition_regions(slot_h, slot_w, n_colors, rng)
    if regions is None:
        return None, None

    rng.shuffle(colors)
    sprite = np.zeros((slot_h, slot_w), dtype=int)
    for color, (r0, r1, c0, c1) in zip(colors, regions):
        sprite[r0:r1, c0:c1] = color
    _gen_jitter_sprite_edges(sprite, colors, rng)
    if len(set(map(int, sprite.ravel()))) != n_colors or np.any(sprite == 0):
        return None, None
    return sprite, colors


def _gen_axis_coord(major, minor, major_axis):
    if major_axis == ROW_AXIS:
        return major, minor
    return minor, major


def _gen_balanced_cross_noise(grid, count, major_indices, minor_indices, colors, rng, major_axis):
    """Place noise while balancing the major axis and capping each minor line."""
    major_extras = {major: 0 for major in major_indices}
    minor_counts = {minor: 0 for minor in minor_indices}
    used = set()

    for _ in range(count):
        possible_major = []
        for major in major_indices:
            if any(
                minor_counts[minor] < NOISE_LINE_CAP
                and _gen_axis_coord(major, minor, major_axis) not in used
                for minor in minor_indices
            ):
                possible_major.append(major)
        if not possible_major:
            return False

        min_extra = min(major_extras[major] for major in possible_major)
        major_choices = [major for major in possible_major if major_extras[major] == min_extra]
        major = major_choices[int(rng.integers(len(major_choices)))]
        possible_minor = [
            minor
            for minor in minor_indices
            if minor_counts[minor] < NOISE_LINE_CAP
            and _gen_axis_coord(major, minor, major_axis) not in used
        ]
        min_minor_count = min(minor_counts[minor] for minor in possible_minor)
        minor_choices = [
            minor for minor in possible_minor if minor_counts[minor] == min_minor_count
        ]
        minor = minor_choices[int(rng.integers(len(minor_choices)))]

        r, c = _gen_axis_coord(major, minor, major_axis)
        grid[r, c] = colors[int(rng.integers(len(colors)))]
        major_extras[major] += 1
        minor_counts[minor] += 1
        used.add((r, c))
    return True


def _gen_scatter_noise(grid, row_starts, col_starts, slot_h, slot_w, colors, rng):
    slot_row_mask = np.zeros(grid.shape[0], dtype=bool)
    slot_col_mask = np.zeros(grid.shape[1], dtype=bool)
    for r0 in row_starts:
        slot_row_mask[r0 : r0 + slot_h] = True
    for c0 in col_starts:
        slot_col_mask[c0 : c0 + slot_w] = True

    noise_count = int(rng.integers(*OUTSIDE_NOISE_COUNT_RANGE))
    cap_a = int(np.sum(~slot_row_mask)) * NOISE_LINE_CAP
    cap_b = int(np.sum(~slot_col_mask)) * NOISE_LINE_CAP
    lo = max(0, noise_count - cap_b, noise_count // NOISE_SPLIT_DENOMINATOR)
    hi = min(noise_count, cap_a, (NOISE_SPLIT_MAX_NUMERATOR * noise_count) // NOISE_SPLIT_DENOMINATOR)
    if lo > hi:
        lo = max(0, noise_count - cap_b)
        hi = min(noise_count, cap_a)
    if lo > hi:
        return False
    count_a = int(rng.integers(lo, hi + 1))
    count_b = noise_count - count_a
    low_rows = np.flatnonzero(~slot_row_mask).astype(int).tolist()
    high_cols = np.flatnonzero(slot_col_mask).astype(int).tolist()
    high_rows = np.flatnonzero(slot_row_mask).astype(int).tolist()
    low_cols = np.flatnonzero(~slot_col_mask).astype(int).tolist()

    return _gen_balanced_cross_noise(grid, count_a, high_cols, low_rows, colors, rng, COL_AXIS) and _gen_balanced_cross_noise(
        grid, count_b, high_rows, low_cols, colors, rng, ROW_AXIS
    )


def _gen_corrupt_slots(grid, sprite, row_starts, col_starts, colors, rng):
    slot_h, slot_w = sprite.shape
    n_slots = len(row_starts) * len(col_starts)
    total = n_slots * slot_h * slot_w
    lo = int(np.ceil(total * SLOT_CORRUPT_RATE_RANGE[0]))
    hi = int(np.floor(total * SLOT_CORRUPT_RATE_RANGE[1]))
    if lo > hi:
        hi = lo
    target = int(rng.integers(lo, hi + 1))
    per_position_cap = max(CORRUPT_CAP_MIN, n_slots // 2 - CORRUPT_CAP_MARGIN)
    slot_coords = [(r0, c0) for r0 in row_starts for c0 in col_starts]
    candidates = [
        (r, c, slot_i)
        for r in range(slot_h)
        for c in range(slot_w)
        for slot_i in range(n_slots)
    ]
    counts = np.zeros((slot_h, slot_w), dtype=int)
    made = 0
    for idx in rng.permutation(len(candidates)):
        r, c, slot_i = candidates[int(idx)]
        if counts[r, c] >= per_position_cap:
            continue
        r0, c0 = slot_coords[slot_i]
        old = int(sprite[r, c])
        choices = [color for color in colors if color != old]
        grid[r0 + r, c0 + c] = choices[int(rng.integers(len(choices)))]
        counts[r, c] += 1
        made += 1
        if made == target:
            return True
    return False


def _gen_render_output(shape, sprite, row_starts, col_starts):
    out = np.zeros(shape, dtype=int)
    slot_h, slot_w = sprite.shape
    for r0 in row_starts:
        for c0 in col_starts:
            out[r0 : r0 + slot_h, c0 : c0 + slot_w] = sprite
    return out


def _gen_attempt():
    rng = np.random.default_rng()
    height, width, slot_h, slot_w, row_starts, col_starts = _gen_choose_layout(rng)

    sprite, colors = _gen_build_sprite(slot_h, slot_w, rng)
    if sprite is None:
        return None

    out = _gen_render_output((height, width), sprite, row_starts, col_starts)
    inp = out.copy()
    if not _gen_corrupt_slots(inp, sprite, row_starts, col_starts, colors, rng):
        return None
    if not _gen_scatter_noise(inp, row_starts, col_starts, slot_h, slot_w, colors, rng):
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input():
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 0607ce86 input")
