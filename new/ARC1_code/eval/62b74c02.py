import numpy as np

from arc_utils import get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each row by stretching its left closed color prefix and repeating the prefix interior at the right edge.

    Input structure
    Each row starts with a contiguous non-background prefix, followed only by background cells.
    The first and last cells of that prefix have the same boundary color.

    Rule
    For each row, keep the prefix in place, fill the middle background gap with the prefix boundary color, and copy the prefix without its first cell into the last cells of the row.

    Key insight
    The left prefix is a closed horizontal motif: its boundary color forms the long span, while the interior symbols reappear at the right end to close the motif again.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    out = g.copy()

    for r in range(h):
        row = g[r]
        non_bg_cols = np.flatnonzero(row != bg)
        assert non_bg_cols.size >= 2
        assert np.array_equal(non_bg_cols, np.arange(non_bg_cols[-1] + 1))

        prefix_len = int(non_bg_cols[-1] + 1)
        prefix = row[:prefix_len]
        boundary_color = int(prefix[0])
        assert boundary_color == int(prefix[-1])
        assert w >= 2 * prefix_len - 1

        suffix_len = prefix_len - 1
        out[r, prefix_len : w - suffix_len] = boundary_color
        out[r, w - suffix_len :] = prefix[1:]

    return out

# --------------- generator (o5x) ---------------
# Geometry, palette, and retry controls; rng.integers upper bounds are exclusive.
_GEN_HEIGHT_RANGE = (3, 5)
_GEN_PREFIX_LEN_RANGE = (3, 6)
_GEN_GAP_RANGE = (5, 8)
_GEN_WIDTH_RANGE = (10, 14)
_GEN_BG = 0
_GEN_COLORS = np.array([1, 2, 3, 8], dtype=int)
_GEN_COLOR_COUNT_RANGE = (2, 5)
_GEN_MIN_DISTINCT_ROWS = 2
_GEN_RETRY_LIMIT = 300
_GEN_PREFIX_GAP_PAIRS = tuple(
    (p, gap) for p in range(*_GEN_PREFIX_LEN_RANGE) for gap in range(*_GEN_GAP_RANGE)
    if _GEN_WIDTH_RANGE[0] <= 2 * p + gap - 1 <= _GEN_WIDTH_RANGE[1]
)


def _gen_prefix(boundary, interiors):
    return tuple([int(boundary), *[int(x) for x in interiors], int(boundary)])


def _gen_canvas_shape(rng):
    h = int(rng.integers(*_GEN_HEIGHT_RANGE))
    prefix_len, gap = _GEN_PREFIX_GAP_PAIRS[int(rng.integers(len(_GEN_PREFIX_GAP_PAIRS)))]
    return h, prefix_len, 2 * prefix_len + gap - 1


def _gen_interior_choices(colors, boundary):
    return [int(c) for c in colors if int(c) != int(boundary)]


def _gen_random_prefix(rng, palette, prefix_len):
    boundary = int(rng.choice(palette))
    interior_choices = _gen_interior_choices(palette, boundary)
    interiors = rng.choice(interior_choices, size=prefix_len - 2, replace=True)
    return _gen_prefix(boundary, interiors)


def _gen_prefix_with_anchor(rng, colors, prefix_len, boundary, anchor):
    interiors = [anchor]
    if prefix_len > 3:
        interior_choices = _gen_interior_choices(colors, boundary)
        interiors.extend(rng.choice(interior_choices, size=prefix_len - 3, replace=True))
    return _gen_prefix(boundary, interiors)


def _gen_cover_rows(rng, palette, prefix_len):
    rows = []
    colors = [int(c) for c in palette]
    for i in range(0, len(colors), 2):
        boundary = colors[i]
        partner = colors[(i + 1) % len(colors)]
        rows.append(_gen_prefix_with_anchor(rng, colors, prefix_len, boundary, partner))

    if len(set(rows)) < _GEN_MIN_DISTINCT_ROWS:
        boundary = colors[1]
        rows.append(_gen_prefix_with_anchor(rng, colors, prefix_len, boundary, colors[0]))

    return rows


def _gen_rows(rng, palette, prefix_len, h):
    prefixes = _gen_cover_rows(rng, palette, prefix_len)
    while len(prefixes) < h:
        prefixes.append(_gen_random_prefix(rng, palette, prefix_len))
    rng.shuffle(prefixes)
    return prefixes[:h]


def _gen_expected_output(inp, prefix_len):
    _, w = inp.shape
    out = inp.copy()
    suffix_len = prefix_len - 1
    out[:, prefix_len : w - suffix_len] = inp[:, :1]
    out[:, w - suffix_len :] = inp[:, 1:prefix_len]
    return out


def _gen_passes_oracle(inp, prefix_len):
    expected = _gen_expected_output(inp, prefix_len)
    try:
        actual = main(inp)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt():
    rng = np.random.default_rng()
    h, prefix_len, w = _gen_canvas_shape(rng)
    color_count = int(rng.integers(*_GEN_COLOR_COUNT_RANGE))
    palette = rng.choice(_GEN_COLORS, size=color_count, replace=False)
    prefixes = _gen_rows(rng, palette, prefix_len, h)
    inp = np.full((h, w), _GEN_BG, dtype=int)
    inp[:, :prefix_len] = np.asarray(prefixes, dtype=int)
    return inp if _gen_passes_oracle(inp, prefix_len) else None


def generate_input():
    for _ in range(_GEN_RETRY_LIMIT):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
