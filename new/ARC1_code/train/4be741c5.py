import numpy as np
import functools
import random


def _compressed(values: list[int]) -> list[int]:
    out: list[int] = []
    for value in values:
        if not out or out[-1] != value:
            out.append(value)
    return out


def _axis_runs(g: np.ndarray, axis: int) -> tuple[bool, list[int]]:
    colors = [int(color) for color in np.unique(g)]
    color_means: dict[int, float] = {}
    for color in colors:
        coords = np.argwhere(g == color)[:, axis]
        assert coords.size > 0, "each listed color must occur"
        color_means[color] = float(coords.mean())

    line_majorities: list[int] = []
    for index in range(g.shape[axis]):
        line = g[index, :] if axis == 0 else g[:, index]
        values, counts = np.unique(line, return_counts=True)
        max_count = int(counts.max())
        winners = [int(value) for value, count in zip(values, counts) if int(count) == max_count]
        if len(winners) == 1:
            line_majorities.append(winners[0])
            continue

        distances = [(abs(color_means[color] - index), color) for color in winners]
        min_distance = min(distance for distance, _ in distances)
        closest = [color for distance, color in distances if distance == min_distance]
        assert len(closest) == 1, "tied line must be resolved by band position"
        line_majorities.append(closest[0])

    runs = _compressed(line_majorities)
    valid = len(runs) == len(colors) and set(runs) == set(colors)
    return valid, runs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Summarize the jagged color bands as a one-pixel sequence in their spatial order.

    Input structure:
    - The grid is fully covered by a small set of colors with no separate background.
    - Along exactly one axis, line-majority colors form consecutive broad bands.
    - The compressed band sequence contains every input color exactly once.

    Rule:
    - Find the unique band axis by compressing row-majorities and column-majorities.
    - Keep one pixel for each band color, ordered top-to-bottom for row bands or
      left-to-right for column bands.
    - Return a column vector for horizontal bands and a row vector for vertical bands.

    Key insight:
    - The irregular boundaries are noise; the line-majority run sequence reveals the
      coarse color order that the output preserves.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.size > 0, "expected a nonempty grid"
    assert np.unique(g).size >= 2, "expected at least two color bands"

    candidates: list[tuple[int, list[int]]] = []
    for axis in (0, 1):
        valid, runs = _axis_runs(g, axis)
        if valid:
            candidates.append((axis, runs))
    assert len(candidates) == 1, "expected exactly one axis with one run per color"

    axis, runs = candidates[0]
    out = np.array(runs, dtype=int)
    if axis == 0:
        return out.reshape((-1, 1))
    return out.reshape((1, -1))

# --------------- generator (o5x) ---------------

# Color and orientation sampling.
COLOR_COUNT_CHOICES = [3, 4]
COLOR_POOL = range(1, 10)
AXIS_COUNT, HORIZONTAL_AXIS = 2, 0

# Canvas dimensions and band thicknesses.
BAND_DIM_RANGE = (9, 16)
HORIZONTAL_CROSS_DIM_RANGE = (7, 16)
VERTICAL_CROSS_DIM_RANGE = (9, 16)
THICKNESS_RANGE = (2, 7)

# Boundary walk shape controls.
WALK_START_POS, WALK_START_LAST = 1, 0
WALK_STEP_MAX = 1
WALK_SIGN_LIMIT_DIVISOR = 2
MAX_BOUNDARY_AMP, BASE_AMP_CAP = 2, 1
DELTA_AMP_PROBABILITY = 0.75
DELTA_SCORE_CROSS_DIVISOR = 3

# Intrusion targets and retry limit.
INTRUSION_PCT_RANGE, DELTA_BASE_INTRUSION_PCT_RANGE = (9, 13), (8, 11)
INTRUSION_FRACTION_RANGE = (0.09, 0.13)
GENERATION_ATTEMPTS = 300


def _gen_can_partition(total, parts):
    return THICKNESS_RANGE[0] * parts <= total <= THICKNESS_RANGE[1] * parts


@functools.lru_cache(maxsize=None)
def _gen_partition_count(total, parts):
    if parts == 0:
        return int(total == 0)
    count = 0
    for value in range(THICKNESS_RANGE[0], THICKNESS_RANGE[1] + 1):
        remaining = total - value
        if _gen_can_partition(remaining, parts - 1):
            count += _gen_partition_count(remaining, parts - 1)
    return count


def _gen_weighted_choice(items, weights):
    total = sum(weights)
    pick = random.randrange(total)
    running = 0
    for item, weight in zip(items, weights):
        running += weight
        if pick < running:
            return item
    return items[-1]


def _gen_partition(total, parts):
    out = []
    remaining = total
    for parts_left in range(parts, 0, -1):
        choices = []
        weights = []
        for value in range(THICKNESS_RANGE[0], THICKNESS_RANGE[1] + 1):
            rest = remaining - value
            if _gen_can_partition(rest, parts_left - 1):
                weight = _gen_partition_count(rest, parts_left - 1)
                if weight:
                    choices.append(value)
                    weights.append(weight)
        value = _gen_weighted_choice(choices, weights)
        out.append(value)
        remaining -= value
    return out


@functools.lru_cache(maxsize=None)
def _gen_walk_count(width, amp, pos, last, score_left, pos_count, neg_count):
    if score_left < 0:
        return 0
    limit = (width - 1) // WALK_SIGN_LIMIT_DIVISOR
    if pos_count > limit or neg_count > limit:
        return 0
    if pos == width:
        return int(score_left == 0)

    total = 0
    lo = max(-amp, last - WALK_STEP_MAX)
    hi = min(amp, last + WALK_STEP_MAX)
    for nxt in range(lo, hi + 1):
        total += _gen_walk_count(
            width,
            amp,
            pos + 1,
            nxt,
            score_left - abs(nxt),
            pos_count + int(nxt > 0),
            neg_count + int(nxt < 0),
        )
    return total


def _gen_walk(width, target_score, amp):
    if _gen_walk_count(width, amp, WALK_START_POS, WALK_START_LAST, target_score, 0, 0) == 0:
        return None

    offsets = [WALK_START_LAST]
    pos = WALK_START_POS
    last = WALK_START_LAST
    score_left = target_score
    pos_count = 0
    neg_count = 0
    while pos < width:
        choices = []
        weights = []
        lo = max(-amp, last - WALK_STEP_MAX)
        hi = min(amp, last + WALK_STEP_MAX)
        for nxt in range(lo, hi + 1):
            weight = _gen_walk_count(
                width,
                amp,
                pos + 1,
                nxt,
                score_left - abs(nxt),
                pos_count + int(nxt > 0),
                neg_count + int(nxt < 0),
            )
            if weight:
                choices.append(nxt)
                weights.append(weight)
        nxt = _gen_weighted_choice(choices, weights)
        offsets.append(nxt)
        pos += 1
        last = nxt
        score_left -= abs(nxt)
        pos_count += int(nxt > 0)
        neg_count += int(nxt < 0)
    return np.array(offsets, dtype=int)


def _gen_walk_score(band_dim, cross_dim, n_boundaries, amp, pct_range):
    cells = band_dim * cross_dim
    lo_pct, hi_pct = pct_range
    lo = (lo_pct * cells + 100 * n_boundaries - 1) // (100 * n_boundaries)
    hi = (hi_pct * cells) // (100 * n_boundaries)
    scores = [
        score
        for score in range(lo, hi + 1)
        if _gen_walk_count(cross_dim, amp, WALK_START_POS, WALK_START_LAST, score, 0, 0)
    ]
    if not scores:
        return None
    return random.choice(scores)


def _gen_horizontal_grid(colors, thicknesses, cross_dim, offsets):
    band_dim = sum(thicknesses)
    offsets = np.asarray(offsets, dtype=int)
    if offsets.ndim == 1:
        offsets = np.tile(offsets[None, :], (len(thicknesses) - 1, 1))
    boundaries = np.cumsum(thicknesses[:-1])[:, None] + offsets
    rows = np.arange(band_dim)[:, None, None]
    band_indices = np.count_nonzero(rows >= boundaries.T[None, :, :], axis=2)
    return np.asarray(colors, dtype=int)[band_indices]


def _gen_intrusion_fraction(grid, axis):
    total = 0
    n_lines = grid.shape[axis]
    for index in range(n_lines):
        line = grid[index, :] if axis == 0 else grid[:, index]
        _, counts = np.unique(line, return_counts=True)
        total += line.size - int(counts.max())
    return total / grid.size


def _gen_delta_amp_limit(thicknesses, amps, boundary_index, base_amp):
    """Return the largest extra offset amplitude that keeps adjacent bands nonempty."""
    n_boundaries = len(thicknesses) - 1
    if boundary_index == 0:
        left_limit = thicknesses[0] - 1 - base_amp
    else:
        left_limit = thicknesses[boundary_index] - 1 - amps[boundary_index - 1]
    if boundary_index == n_boundaries - 1:
        right_limit = thicknesses[-1] - 1 - base_amp
    else:
        right_limit = thicknesses[boundary_index + 1] - 1 - amps[boundary_index + 1]
    return min(MAX_BOUNDARY_AMP - base_amp, left_limit, right_limit)


def _gen_delta_amps(thicknesses, base_amp):
    n_boundaries = len(thicknesses) - 1
    amps = [0] * n_boundaries
    for j in random.sample(range(n_boundaries), n_boundaries):
        max_amp = _gen_delta_amp_limit(thicknesses, amps, j, base_amp)
        if max_amp > 0 and random.random() < DELTA_AMP_PROBABILITY:
            amps[j] = random.randint(1, max_amp)
    return amps


def _gen_boundary_offsets(thicknesses, cross_dim):
    band_dim = sum(thicknesses)
    n_boundaries = len(thicknesses) - 1
    base_amp = min(BASE_AMP_CAP, thicknesses[0] - 1, thicknesses[-1] - 1)
    delta_amps = _gen_delta_amps(thicknesses, base_amp)

    if any(delta_amps):
        pct_range = DELTA_BASE_INTRUSION_PCT_RANGE
    else:
        pct_range = INTRUSION_PCT_RANGE
    base_score = _gen_walk_score(band_dim, cross_dim, n_boundaries, base_amp, pct_range)
    if base_score is None:
        return None
    base = _gen_walk(cross_dim, base_score, base_amp)
    if base is None:
        return None

    offsets = []
    for amp in delta_amps:
        if amp == 0:
            delta = np.zeros(cross_dim, dtype=int)
        else:
            max_score = min(max(1, cross_dim // DELTA_SCORE_CROSS_DIVISOR), cross_dim - 1)
            delta_score = random.randint(1, max_score)
            delta = _gen_walk(cross_dim, delta_score, amp)
            if delta is None:
                return None
        offsets.append(base + delta)
    return np.array(offsets, dtype=int)


def _gen_sample_layout():
    """Sample colors, orientation, dimensions, and band thicknesses in generator order."""
    n_colors = random.choice(COLOR_COUNT_CHOICES)
    colors = random.sample(COLOR_POOL, n_colors)
    axis = random.randrange(AXIS_COUNT)

    band_dim = random.randint(*BAND_DIM_RANGE)
    cross_range = HORIZONTAL_CROSS_DIM_RANGE if axis == HORIZONTAL_AXIS else VERTICAL_CROSS_DIM_RANGE
    cross_dim = random.randint(*cross_range)
    thicknesses = _gen_partition(band_dim, n_colors)
    return colors, axis, thicknesses, cross_dim


def _gen_oriented_candidate(colors, axis, thicknesses, cross_dim, offsets):
    """Render canonical horizontal bands, then orient grid and expected output."""
    canonical = _gen_horizontal_grid(colors, thicknesses, cross_dim, offsets)
    if axis == HORIZONTAL_AXIS:
        return canonical, np.array(colors, dtype=int).reshape((-1, 1))
    return canonical.T, np.array(colors, dtype=int).reshape((1, -1))


def _gen_matches_task(grid, axis, expected):
    """Check intrusion bounds and solver agreement for a generated candidate."""
    intrusion = _gen_intrusion_fraction(grid, axis)
    if not INTRUSION_FRACTION_RANGE[0] <= intrusion <= INTRUSION_FRACTION_RANGE[1]:
        return False

    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt():
    colors, axis, thicknesses, cross_dim = _gen_sample_layout()

    offsets = _gen_boundary_offsets(thicknesses, cross_dim)
    if offsets is None:
        return None

    grid, expected = _gen_oriented_candidate(colors, axis, thicknesses, cross_dim, offsets)
    if not _gen_matches_task(grid, axis, expected):
        return None
    return grid


def generate_input():
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 4be741c5 input")
