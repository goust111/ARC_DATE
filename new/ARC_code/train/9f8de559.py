import numpy as np

from arc_utils import Color, get_bg


DIRS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def main(grid: np.ndarray) -> np.ndarray:
    """Erase the first border pixel hit by extending the red marker away from its pink trail.

    Input structure
    - The dominant background surrounds a unique red marker and a four-pixel
      pink trail.
    - The pink pixels form one straight 8-connected ray starting next to the
      red marker.
    - In the opposite direction from that trail, the red marker faces a
      non-background border or obstacle pixel beyond background cells.

    Rule
    - Keep the whole grid fixed except for the first non-background pixel on
      the ray from the red marker away from the pink trail.
    - Recolor that one hit pixel to the background color.

    Key insight
    - The pink trail marks where the red marker came from, so the edit is found
      by ray-casting from the red marker in the opposite direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape
    bg = get_bg(g)

    red_coords = np.argwhere(g == Color.RED)
    assert len(red_coords) == 1
    red_r, red_c = map(int, red_coords[0])

    pink_coords = [tuple(map(int, p)) for p in np.argwhere(g == Color.PINK)]
    assert len(pink_coords) == 4
    pink_set = set(pink_coords)

    trail_dirs: list[tuple[int, int]] = []
    for dr, dc in DIRS8:
        expected = {
            (red_r + step * dr, red_c + step * dc)
            for step in range(1, len(pink_coords) + 1)
        }
        if expected == pink_set:
            trail_dirs.append((dr, dc))
    assert len(trail_dirs) == 1

    trail_dr, trail_dc = trail_dirs[0]
    ray_dr, ray_dc = -trail_dr, -trail_dc

    hits: list[tuple[int, int]] = []
    r, c = red_r + ray_dr, red_c + ray_dc
    while 0 <= r < H and 0 <= c < W:
        if g[r, c] != bg:
            assert g[r, c] not in (Color.RED, Color.PINK)
            hits.append((r, c))
            break
        r += ray_dr
        c += ray_dc
    assert len(hits) == 1

    out = g.copy()
    hit_r, hit_c = hits[0]
    out[hit_r, hit_c] = bg
    return out

# --------------- generator (att2) ---------------

_GEN_ORANGE, _GEN_RED, _GEN_PINK = 7, 2, 6
_GEN_FRAME_COLORS = (5, 8, 9)

# Canvas, border, and inner rectangle bounds.
_GEN_GRID_SIZE = 14
_GEN_BORDER_RANGE = (1, 3)
_GEN_INNER_HEIGHT_RANGE = (9, 12)
_GEN_INNER_WIDTH_RANGE = (9, 11)

# Trail and ray constraints.
_GEN_TRAIL_LEN = 4
_GEN_TRAIL_CLEARANCE_RANGE = (1, 2)
_GEN_RAY_ORANGE_RANGE = (1, 6)

# Frame segmentation and dominance constraints.
_GEN_BAND_COUNT = 4
_GEN_SEGMENTS_PER_BAND_RANGE = (1, 4)
_GEN_TOTAL_SEGMENTS_RANGE = (9, 13)
_GEN_DOMINANT_LEN_RANGE = (8, 11)

# Generation retry limit.
_GEN_MAX_ATTEMPTS = 300

_GEN_DIRECTIONS = DIRS8


def _gen_choice(seq):
    return seq[int(np.random.randint(len(seq)))]


def _gen_split_axis(inner_min, inner_max):
    inner = int(np.random.randint(inner_min, inner_max + 1))
    border_min, border_max = _GEN_BORDER_RANGE
    pairs = []
    for before in range(border_min, border_max + 1):
        after = _GEN_GRID_SIZE - inner - before
        if border_min <= after <= border_max:
            pairs.append((before, after))
    before, after = _gen_choice(pairs)
    return before, inner, after


def _gen_positive_parts(total, count):
    if count == 0:
        return []
    if count == 1:
        return [total]
    cuts = np.random.choice(np.arange(1, total), count - 1, replace=False)
    cuts = [0] + sorted(int(c) for c in cuts) + [total]
    return [cuts[i + 1] - cuts[i] for i in range(count)]


def _gen_segment_counts(duplicate_band_indices):
    total_min, total_max = _GEN_TOTAL_SEGMENTS_RANGE
    segment_min, segment_max = _GEN_SEGMENTS_PER_BAND_RANGE
    total = int(np.random.randint(total_min, total_max + 1))
    choices = []
    span = segment_max - segment_min + 1
    for offsets in np.ndindex((span,) * _GEN_BAND_COUNT):
        counts = tuple(offset + segment_min for offset in offsets)
        if sum(counts) != total:
            continue
        if any(counts[i] == segment_min for i in duplicate_band_indices):
            continue
        choices.append(counts)
    return _gen_choice(choices)


def _gen_band_pattern(dominant_color, segments, duplicate_color):
    segment_min, _ = _GEN_SEGMENTS_PER_BAND_RANGE
    dominant_min, dominant_max = _GEN_DOMINANT_LEN_RANGE
    if segments == segment_min:
        return np.full(_GEN_GRID_SIZE, dominant_color, dtype=int)

    if dominant_color == duplicate_color:
        dominant_len = dominant_min
    else:
        max_len = min(dominant_max, _GEN_GRID_SIZE - (segments - 1))
        dominant_len = int(np.random.randint(dominant_min, max_len + 1))

    dominant_index = int(np.random.randint(segments))
    remaining_lengths = _gen_positive_parts(_GEN_GRID_SIZE - dominant_len, segments - 1)
    np.random.shuffle(remaining_lengths)

    lengths = []
    for idx in range(segments):
        if idx == dominant_index:
            lengths.append(dominant_len)
        else:
            lengths.append(remaining_lengths.pop())

    other_colors = [color for color in _GEN_FRAME_COLORS if color != dominant_color]
    runs = []
    prev = None
    for idx in range(segments):
        if idx == dominant_index:
            color = dominant_color
        else:
            choices = [color for color in other_colors if color != prev]
            if dominant_color != duplicate_color and len(choices) > 1 and duplicate_color in choices:
                choices = [color for color in choices if color != duplicate_color]
            color = _gen_choice(choices)
        runs.extend([color] * lengths[idx])
        prev = color
    return np.array(runs, dtype=int)


def _gen_fill_horizontal(grid, rows, cols, pattern):
    for c in cols:
        grid[rows, c] = pattern[c]


def _gen_fill_vertical(grid, rows, cols, pattern):
    for r in rows:
        grid[r, cols] = pattern[r]


def _gen_place_corner(grid, row_slice, col_slice, horizontal_pattern, vertical_pattern):
    if np.random.randint(2) == 0:
        cols = range(col_slice.start, col_slice.stop)
        _gen_fill_horizontal(grid, row_slice, cols, horizontal_pattern)
    else:
        rows = range(row_slice.start, row_slice.stop)
        _gen_fill_vertical(grid, rows, col_slice, vertical_pattern)


def _gen_frame(grid, top, bottom, left, right):
    band_names = ("top", "bottom", "left", "right")
    colors = list(_GEN_FRAME_COLORS)
    np.random.shuffle(colors)
    duplicate_color = _gen_choice(colors)
    dominants = colors + [duplicate_color]
    np.random.shuffle(dominants)
    duplicate_band_indices = [idx for idx, color in enumerate(dominants) if color == duplicate_color]
    segment_counts = _gen_segment_counts(duplicate_band_indices)
    patterns = {
        band_names[idx]: _gen_band_pattern(dominants[idx], segment_counts[idx], duplicate_color)
        for idx in range(_GEN_BAND_COUNT)
    }

    inner_rows = range(top, _GEN_GRID_SIZE - bottom)
    inner_cols = range(left, _GEN_GRID_SIZE - right)
    top_rows = slice(0, top)
    bottom_rows = slice(_GEN_GRID_SIZE - bottom, _GEN_GRID_SIZE)
    left_cols = slice(0, left)
    right_cols = slice(_GEN_GRID_SIZE - right, _GEN_GRID_SIZE)

    _gen_fill_horizontal(grid, top_rows, inner_cols, patterns["top"])
    _gen_fill_horizontal(grid, bottom_rows, inner_cols, patterns["bottom"])
    _gen_fill_vertical(grid, inner_rows, left_cols, patterns["left"])
    _gen_fill_vertical(grid, inner_rows, right_cols, patterns["right"])

    _gen_place_corner(grid, top_rows, left_cols, patterns["top"], patterns["left"])
    _gen_place_corner(grid, top_rows, right_cols, patterns["top"], patterns["right"])
    _gen_place_corner(grid, bottom_rows, left_cols, patterns["bottom"], patterns["left"])
    _gen_place_corner(grid, bottom_rows, right_cols, patterns["bottom"], patterns["right"])


def _gen_red_location(r0, c0, r1, c1, trail_dr, trail_dc):
    clearance_min, clearance_max = _GEN_TRAIL_CLEARANCE_RANGE
    clearance = int(np.random.randint(clearance_min, clearance_max + 1))
    trail_distance = _GEN_TRAIL_LEN + clearance

    if trail_dr < 0:
        red_r = r0 + trail_distance
    elif trail_dr > 0:
        red_r = r1 - trail_distance
    else:
        red_r = int(np.random.randint(r0, r1 + 1))

    if trail_dc < 0:
        red_c = c0 + trail_distance
    elif trail_dc > 0:
        red_c = c1 - trail_distance
    else:
        red_c = int(np.random.randint(c0, c1 + 1))

    return red_r, red_c


def _gen_ray_hit(out, r0, c0, r1, c1, red, ray_dir):
    red_r, red_c = red
    ray_dr, ray_dc = ray_dir

    ray_cells = []
    r, c = red_r + ray_dr, red_c + ray_dc
    while r0 <= r <= r1 and c0 <= c <= c1:
        ray_cells.append((r, c))
        r += ray_dr
        c += ray_dc
    ray_min, ray_max = _GEN_RAY_ORANGE_RANGE
    if not (
        ray_min <= len(ray_cells) <= ray_max
        and 0 <= r < _GEN_GRID_SIZE
        and 0 <= c < _GEN_GRID_SIZE
    ):
        return None

    hit_r, hit_c = r, c
    hit_color = int(out[hit_r, hit_c])
    if hit_color not in _GEN_FRAME_COLORS:
        return None
    return hit_r, hit_c, hit_color


def _gen_oracle_accepts(inp, out):
    orange_count = np.count_nonzero(inp == _GEN_ORANGE)
    max_frame_count = max(np.count_nonzero(inp == color) for color in _GEN_FRAME_COLORS)
    if orange_count <= max_frame_count:
        return False
    try:
        predicted = main(inp)
    except Exception:
        return False
    return np.array_equal(predicted, out)


def _gen_attempt():
    h_min, h_max = _GEN_INNER_HEIGHT_RANGE
    w_min, w_max = _GEN_INNER_WIDTH_RANGE
    top, inner_h, bottom = _gen_split_axis(h_min, h_max)
    left, inner_w, right = _gen_split_axis(w_min, w_max)
    r0, c0 = top, left
    r1, c1 = top + inner_h - 1, left + inner_w - 1

    trail_dr, trail_dc = _gen_choice(_GEN_DIRECTIONS)
    red_r, red_c = _gen_red_location(r0, c0, r1, c1, trail_dr, trail_dc)
    ray_dir = (-trail_dr, -trail_dc)

    out = np.zeros((_GEN_GRID_SIZE, _GEN_GRID_SIZE), dtype=int)
    _gen_frame(out, top, bottom, left, right)
    out[r0 : r1 + 1, c0 : c1 + 1] = _GEN_ORANGE
    out[red_r, red_c] = _GEN_RED
    for step in range(1, _GEN_TRAIL_LEN + 1):
        out[red_r + step * trail_dr, red_c + step * trail_dc] = _GEN_PINK

    hit = _gen_ray_hit(out, r0, c0, r1, c1, (red_r, red_c), ray_dir)
    if hit is None:
        return None
    hit_r, hit_c, hit_color = hit
    out[hit_r, hit_c] = _GEN_ORANGE
    inp = out.copy()
    inp[hit_r, hit_c] = hit_color

    if _gen_oracle_accepts(inp, out):
        return inp
    return None


def generate_input():
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 9f8de559 input")
