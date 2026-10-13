import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat each row by the number of distinct non-black colors it contains."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] > 0 and g.shape[1] > 0, "grid must be non-empty"

    repeats = []
    for row in g:
        colors = np.unique(row[row != 0])
        assert len(colors) > 0, "each row must contain at least one foreground color"
        repeats.append(int(len(colors)))

    return np.repeat(g, repeats, axis=0)


# --------------- generator ---------------

_RNG = np.random.default_rng()

COLOR_POOL = tuple(range(1, 10))
HEIGHT_RANGE = (4, 8)
WIDTH_RANGE = (8, 16)
MAX_ROW_COLORS = 5
MAX_EXTRA_SEGMENTS = 2
MAX_GENERATION_ATTEMPTS = 300


def _sample_parts(total: int, parts: int) -> np.ndarray:
    """Split total into positive integer parts."""
    assert 1 <= parts <= total
    if parts == 1:
        return np.array([total], dtype=int)

    cuts = np.sort(_RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    return np.diff(np.concatenate(([0], cuts, [total]))).astype(int)


def _segment_count(row: np.ndarray) -> int:
    return 1 + int(np.count_nonzero(row[1:] != row[:-1]))


def _sample_segment_sequence(colors: list[int], segment_count: int) -> list[int]:
    """Build a no-touching segment color sequence that uses every chosen color."""
    assert len(colors) <= segment_count
    if len(colors) == 1:
        assert segment_count == 1
        return [int(colors[0])]

    sequence = _RNG.permutation(np.array(colors, dtype=int)).tolist()
    while len(sequence) < segment_count:
        candidates = [color for color in colors if color != sequence[-1]]
        sequence.append(int(_RNG.choice(candidates)))

    if _RNG.random() < 0.5:
        sequence.reverse()
    return [int(color) for color in sequence]


def _build_row(width: int, distinct_colors: int, force_repeated_color: bool, palette: np.ndarray) -> np.ndarray | None:
    if distinct_colors == 1:
        min_segments = 1
        max_segments = 1
    else:
        min_segments = distinct_colors
        if force_repeated_color:
            min_segments += 1
        max_segments = min(width, distinct_colors + MAX_EXTRA_SEGMENTS)

    if min_segments > max_segments:
        return None

    segment_count = int(_RNG.integers(min_segments, max_segments + 1))
    colors = _RNG.choice(palette, size=distinct_colors, replace=False).astype(int).tolist()
    sequence = _sample_segment_sequence(colors, segment_count)
    lengths = _sample_parts(width, segment_count)

    row = np.empty(width, dtype=int)
    c = 0
    for color, seg_len in zip(sequence, lengths):
        row[c : c + seg_len] = int(color)
        c += int(seg_len)

    assert c == width
    assert int(len(np.unique(row))) == distinct_colors
    assert _segment_count(row) == segment_count
    if force_repeated_color:
        assert segment_count > distinct_colors
    return row


def _target_color_counts(height: int, max_colors: int) -> list[int]:
    counts = [1, max_colors]
    if height >= 3 and max_colors >= 3:
        counts.append(int(_RNG.integers(2, max_colors)))

    while len(counts) < height:
        if _RNG.random() < 0.5:
            counts.append(int(_RNG.choice(counts)))
        else:
            counts.append(int(_RNG.integers(1, max_colors + 1)))

    _RNG.shuffle(counts)
    return counts[:height]


def _is_informative(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or not np.all(grid != 0):
        return False

    color_counts = [int(len(np.unique(row))) for row in grid]
    if len(set(color_counts)) < 2:
        return False
    if not any(count == 1 for count in color_counts):
        return False
    if not any(count >= 3 for count in color_counts):
        return False
    if not any(_segment_count(row) > int(len(np.unique(row))) for row in grid):
        return False

    expected = np.repeat(grid, color_counts, axis=0)
    try:
        actual = main(grid)
    except Exception:
        return False
    return np.array_equal(actual, expected)


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(HEIGHT_RANGE[0], HEIGHT_RANGE[1] + 1))
    width = int(_RNG.integers(WIDTH_RANGE[0], WIDTH_RANGE[1] + 1))
    max_colors = min(MAX_ROW_COLORS, width, len(COLOR_POOL))

    color_counts = _target_color_counts(height, max_colors)
    palette_size = int(_RNG.integers(max(max(color_counts), 4), len(COLOR_POOL) + 1))
    palette = _RNG.choice(np.array(COLOR_POOL, dtype=int), size=palette_size, replace=False)

    repeated_candidates = [i for i, count in enumerate(color_counts) if count >= 2]
    if not repeated_candidates:
        return None
    forced_repeat_index = int(_RNG.choice(repeated_candidates))

    rows = []
    for i, distinct_colors in enumerate(color_counts):
        force_repeated_color = i == forced_repeat_index
        if distinct_colors >= 2 and i != forced_repeat_index and _RNG.random() < 0.35 and distinct_colors < width:
            force_repeated_color = True

        row = _build_row(width, distinct_colors, force_repeated_color, palette)
        if row is None:
            return None
        rows.append(row)

    grid = np.stack(rows, axis=0)
    if _is_informative(grid):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
