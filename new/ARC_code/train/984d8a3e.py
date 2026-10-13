import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shift each row's dominant middle-color run left until its right edge aligns with the leftmost original right edge, moving displaced left-side pixels after the run.

    Input structure:
    - The grid has three colors.
    - The most common color forms one contiguous horizontal run in every row.
    - Pixels before those runs share one side color, and pixels after those runs share the other side color.

    Rule:
    - Find the minimum right edge column among the middle-color runs.
    - In each row, keep the run width and the trailing right-side color count.
    - Place the middle-color run so its right edge is at that minimum column, unless the run is too wide, in which case start it at column 0.
    - Fill any remaining non-trailing cells with the left-side color.

    Key insight:
    The varying row prefixes are movable padding; the stable feature is the leftmost right boundary of the middle-color shape.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    colors, counts = np.unique(g, return_counts=True)
    assert colors.size == 3, "grid must contain exactly three colors"
    count_order = np.argsort(-counts)
    assert counts[count_order[0]] > counts[count_order[1]], "middle color must be uniquely most common"
    middle_color = int(colors[count_order[0]])

    spans: list[tuple[int, int]] = []
    left_colors: set[int] = set()
    right_colors: set[int] = set()
    for r in range(h):
        row = g[r]
        middle_cols = np.flatnonzero(row == middle_color)
        assert middle_cols.size > 0, "each row must contain the middle-color run"
        start = int(middle_cols[0])
        end = int(middle_cols[-1])
        assert middle_cols.size == end - start + 1, "middle-color pixels must form one run per row"

        spans.append((start, end))
        left_colors.update(map(int, row[:start]))
        right_colors.update(map(int, row[end + 1 :]))

    assert len(left_colors) == 1, "all left-side pixels must share one color"
    assert len(right_colors) == 1, "all right-side pixels must share one color"
    left_color = next(iter(left_colors))
    right_color = next(iter(right_colors))
    assert len({left_color, middle_color, right_color}) == 3, "the three color roles must be distinct"
    assert set(map(int, colors)) == {left_color, middle_color, right_color}

    anchor_end = min(end for _, end in spans)
    max_left_prefix = max(start for start, _ in spans)
    assert 0 <= anchor_end - max_left_prefix <= 2, "left-side coverage must stop at or just before the middle-run boundary"

    out = np.empty_like(g)
    for r, (start, end) in enumerate(spans):
        run_width = end - start + 1
        right_count = w - end - 1
        new_start = max(0, anchor_end - run_width + 1)
        assert 0 <= new_start <= start, "aligned run must fit within the row's left padding"

        out[r, :new_start] = left_color
        out[r, new_start : new_start + run_width] = middle_color
        out[r, new_start + run_width : w - right_count] = left_color
        out[r, w - right_count :] = right_color

    assert out.shape == g.shape
    return out

# --------------- generator (att2) ---------------

_GEN_RNG = np.random.default_rng()

GRID_MIN_SIZE = 10
GRID_MAX_SIZE_EXCLUSIVE = 18
COLOR_COUNT = 3
MIN_RUN_WIDTH = 3
MAX_RUN_WIDTH_FRAC = 0.70
MIN_MIDDLE_FRAC = 0.40
MAX_MIDDLE_FRAC = 0.60
MIN_DISTINCT_WIDTHS = 3
ANCHOR_LOW_FRAC = 0.30
ANCHOR_HIGH_FRAC = 0.60
ANCHOR_RIGHT_PADDING = 9
MAX_LEFT_PREFIX = 6
MAX_RIGHT_SUFFIX = 8
MAX_PREFIX_GAP = 2
CLAMP_RIGHT_PADDING = 2
GEN_ATTEMPTS = 300


def _gen_adjust_values(vals, mins, caps, target, rng, locked=None):
    vals = np.asarray(vals, dtype=int).copy()
    mins = np.asarray(mins, dtype=int)
    caps = np.asarray(caps, dtype=int)
    if locked is None:
        locked = np.zeros(vals.size, dtype=bool)
    else:
        locked = np.asarray(locked, dtype=bool)

    guard = int(vals.size * (int(caps.max(initial=0)) + int(vals.max(initial=0)) + 5))
    for _ in range(max(guard, 1)):
        total = int(vals.sum())
        if total == target:
            return vals
        if total < target:
            choices = np.flatnonzero((vals < caps) & ~locked)
            if choices.size == 0:
                return None
            vals[int(rng.choice(choices))] += 1
        else:
            choices = np.flatnonzero((vals > mins) & ~locked)
            if choices.size == 0:
                return None
            vals[int(rng.choice(choices))] -= 1
    return None


def _gen_adjust_widths(widths, min_total, max_total, min_w, max_w, anchor, rng):
    widths = np.asarray(widths, dtype=int).copy()
    guard = int(widths.size * (max_w - min_w + 3))
    for _ in range(max(guard, 1)):
        total = int(widths.sum())
        if min_total <= total <= max_total:
            return widths

        has_anchor = widths <= anchor
        has_clamp = widths > anchor + 1
        if total < min_total:
            choices = []
            for i, w in enumerate(widths):
                if w >= max_w:
                    continue
                if has_anchor.sum() == 1 and has_anchor[i] and w + 1 > anchor:
                    continue
                choices.append(i)
            if not choices:
                return None
            widths[int(rng.choice(choices))] += 1
        else:
            choices = []
            for i, w in enumerate(widths):
                if w <= min_w:
                    continue
                if has_clamp.sum() == 1 and has_clamp[i] and w - 1 <= anchor + 1:
                    continue
                choices.append(i)
            if not choices:
                return None
            widths[int(rng.choice(choices))] -= 1
    return None


def _gen_make_rights_varied(rights, mins, caps, locked, rng):
    rights = np.asarray(rights, dtype=int).copy()
    if np.unique(rights).size >= 2:
        return rights

    incs = np.flatnonzero((rights < caps) & ~locked)
    decs = np.flatnonzero((rights > mins) & ~locked)
    pairs = [(int(i), int(j)) for i in incs for j in decs if int(i) != int(j)]
    if not pairs:
        return rights
    i, j = pairs[int(rng.integers(0, len(pairs)))]
    rights[i] += 1
    rights[j] -= 1
    return rights


def _gen_choose_widths(n, total_cells, max_w, anchor, rng):
    effective_min_w = max(MIN_RUN_WIDTH, anchor - 5)
    widths = rng.integers(effective_min_w, max_w + 1, size=n)
    seed_rows = rng.choice(np.arange(n), 3, replace=False)
    widths[int(seed_rows[0])] = int(rng.integers(effective_min_w, anchor + 1))
    widths[int(seed_rows[1])] = int(rng.integers(anchor + 2, max_w + 1))
    seeded = {int(widths[seed_rows[0]]), int(widths[seed_rows[1]])}
    other_widths = [w for w in range(effective_min_w, max_w + 1) if w not in seeded]
    widths[int(seed_rows[2])] = int(rng.choice(other_widths))

    min_middle = int(np.ceil(MIN_MIDDLE_FRAC * total_cells))
    max_middle = int(np.floor(MAX_MIDDLE_FRAC * total_cells))
    widths = _gen_adjust_widths(widths, min_middle, max_middle, effective_min_w, max_w, anchor, rng)
    if widths is None:
        return None
    if (
        np.unique(widths).size < MIN_DISTINCT_WIDTHS
        or not np.any(widths <= anchor)
        or not np.any(widths > anchor + 1)
    ):
        return None
    return widths


def _gen_choose_rights(n, total_cells, widths, anchor, rng):
    out_starts = np.maximum(0, anchor - widths + 1)
    out_ends = out_starts + widths - 1
    caps = np.minimum(MAX_RIGHT_SUFFIX, n - out_ends - 1)
    caps = np.minimum(caps, n - anchor - 1)
    caps = np.where(widths > anchor, np.minimum(caps, n - anchor - CLAMP_RIGHT_PADDING), caps)
    mins = np.maximum(0, n - widths - MAX_LEFT_PREFIX)
    mins = np.maximum(mins, n - widths - anchor)
    if np.any(caps < 0) or np.any(mins > caps):
        return None

    anchor_candidates = np.flatnonzero(widths <= anchor)
    anchor_idx = int(rng.choice(anchor_candidates))
    anchor_right = n - anchor - 1
    if not (mins[anchor_idx] <= anchor_right <= caps[anchor_idx]):
        return None

    rights = rng.integers(mins, caps + 1)
    rights[anchor_idx] = anchor_right
    locked = np.zeros(n, dtype=bool)
    locked[anchor_idx] = True

    prefix_floor = max(0, anchor - MAX_PREFIX_GAP)
    prefix_options = []
    for i, m in enumerate(widths):
        if i == anchor_idx:
            left = n - int(m) - anchor_right
            if prefix_floor <= left <= anchor:
                prefix_options.append((i, anchor_right, anchor_right))
            continue

        low_right = max(int(mins[i]), n - int(m) - anchor)
        high_right = min(int(caps[i]), n - int(m) - prefix_floor)
        if low_right <= high_right:
            prefix_options.append((i, low_right, high_right))
    if not prefix_options:
        return None

    prefix_idx, low_right, high_right = prefix_options[int(rng.integers(0, len(prefix_options)))]
    if prefix_idx != anchor_idx:
        rights[prefix_idx] = int(rng.integers(low_right, high_right + 1))
        locked[prefix_idx] = True

    middle_total = int(widths.sum())
    locked_right_total = int(rights[locked].sum())
    min_right_total = max(int(locked_right_total + mins[~locked].sum()), total_cells - 2 * middle_total + 1)
    max_right_total = min(middle_total - 1, total_cells - middle_total - 1, int(locked_right_total + caps[~locked].sum()))
    if min_right_total > max_right_total:
        return None

    current_right_total = int(rights.sum())
    if min_right_total <= current_right_total <= max_right_total:
        target_right_total = current_right_total
    else:
        target_right_total = int(rng.integers(min_right_total, max_right_total + 1))

    rights = _gen_adjust_values(rights, mins, caps, target_right_total, rng, locked=locked)
    if rights is None:
        return None
    rights = _gen_make_rights_varied(rights, mins, caps, locked, rng)

    if rights[anchor_idx] != anchor_right or np.any(rights > caps) or np.any(rights < mins):
        return None
    if int(rights.max()) != anchor_right:
        return None
    lefts = n - widths - rights
    if not (0 <= anchor - int(lefts.max()) <= MAX_PREFIX_GAP):
        return None

    left_total = total_cells - middle_total - int(rights.sum())
    if not (middle_total > int(rights.sum()) and middle_total > left_total and left_total > 0):
        return None
    return rights, out_starts


def _gen_render_rows(n, widths, rights, out_starts, colors):
    left_color, middle_color, right_color = colors
    inp = np.empty((n, n), dtype=int)
    expected = np.empty((n, n), dtype=int)
    for r, (m, right, out_start) in enumerate(zip(widths, rights, out_starts)):
        input_start = n - int(m) - int(right)
        if input_start < 0:
            return None
        inp[r, :input_start] = left_color
        inp[r, input_start : input_start + int(m)] = middle_color
        inp[r, input_start + int(m) :] = right_color

        expected[r, : int(out_start)] = left_color
        expected[r, int(out_start) : int(out_start) + int(m)] = middle_color
        expected[r, int(out_start) + int(m) : n - int(right)] = left_color
        expected[r, n - int(right) :] = right_color
    return inp, expected


def _gen_attempt():
    rng = _GEN_RNG
    n = int(rng.integers(GRID_MIN_SIZE, GRID_MAX_SIZE_EXCLUSIVE))
    total_cells = n * n
    max_w = int(np.floor(MAX_RUN_WIDTH_FRAC * n))

    anchor_low = max(int(np.floor(ANCHOR_LOW_FRAC * n)), n - ANCHOR_RIGHT_PADDING)
    anchor_high = min(int(np.floor(ANCHOR_HIGH_FRAC * n)), max_w - MAX_PREFIX_GAP)
    if anchor_low > anchor_high:
        return None
    anchor = int(rng.integers(anchor_low, anchor_high + 1))

    colors = tuple(map(int, rng.choice(np.arange(10), COLOR_COUNT, replace=False)))

    widths = _gen_choose_widths(n, total_cells, max_w, anchor, rng)
    if widths is None:
        return None

    right_result = _gen_choose_rights(n, total_cells, widths, anchor, rng)
    if right_result is None:
        return None
    rights, out_starts = right_result

    if (rendered := _gen_render_rows(n, widths, rights, out_starts, colors)) is None:
        return None
    inp, expected = rendered

    if len(np.unique(inp)) != COLOR_COUNT:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input():
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
