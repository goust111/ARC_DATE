from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
TEAL = 8


def _marked_rows(grid: np.ndarray) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for r in range(grid.shape[0]):
        blue_cols = np.flatnonzero(grid[r] == BLUE)
        teal_cols = np.flatnonzero(grid[r] == TEAL)
        if len(blue_cols) == 0 and len(teal_cols) == 0:
            continue
        assert len(blue_cols) == 1 and len(teal_cols) == 1, "each marked row must have one blue and one teal marker"
        left = int(blue_cols[0])
        right = int(teal_cols[0])
        assert left < right, "blue marker must be left of teal marker"
        rows.append((r, left, right))
    assert rows, "expected at least one marked row"
    width = rows[0][2] - rows[0][1] - 1
    assert width > 0 and width % 2 == 1, "segment width must be positive and odd"
    assert all(right - left - 1 == width for _, left, right in rows), "all extracted segments must share one width"
    return rows


def _repair_segment(segment: np.ndarray) -> tuple[np.ndarray, int]:
    seg = np.asarray(segment, dtype=int)
    assert seg.ndim == 1 and seg.size % 2 == 1, "expected one odd-width segment"

    out = seg.copy()
    width = int(seg.size)
    center = width // 2
    repaired = 0

    for c in range(width):
        if int(seg[c]) != GREEN:
            continue
        assert c != center, "green placeholder cannot lie on the symmetry axis"
        mirror = width - 1 - c
        fill = int(seg[mirror])
        assert fill in (BLACK, RED), "green pixels must mirror a visible black/red cell"
        out[c] = fill
        repaired += 1

    return out, repaired


def main(grid: np.ndarray) -> np.ndarray:
    """Extract bracketed segments, repair them by horizontal symmetry, then stack them by repair count.

    Input structure:
    - The grid is a black/red field with some marked rows containing exactly one blue pixel on the left
      and one teal pixel on the right.
    - The cells strictly between each blue/teal pair form a same-width odd-length segment.
    - Inside each segment, some cells are green placeholders while the remaining visible cells form a
      left-right symmetric red/black pattern.

    Rule:
    - Find every marked row and extract the cells strictly between the blue and teal markers.
    - For each segment, use the segment's own vertical middle column as the reflection axis and replace
      every green placeholder with the color at its mirrored position.
    - Count how many cells were repaired in each segment, sort segments by that count ascending, and stack
      the repaired segments from top to bottom.

    Key insight:
    - The control pair defines both the cropped segment and the local symmetry domain, and the number of
      repaired cells is the hidden key that determines the final row order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    repaired_rows: list[tuple[int, int, np.ndarray]] = []
    width: int | None = None

    for row_index, left, right in _marked_rows(g):
        segment = g[row_index, left + 1 : right]
        assert not np.isin(segment, [BLUE, TEAL]).any(), "markers only delimit segments"
        repaired, count = _repair_segment(segment)
        if width is None:
            width = int(repaired.size)
        assert int(repaired.size) == width
        repaired_rows.append((count, row_index, repaired))

    repaired_rows.sort(key=lambda item: (item[0], item[1]))
    return np.vstack([row for _, _, row in repaired_rows])


# --------------- generator (gpt-5.4) ---------------

GEN_ATTEMPTS = 400
HEIGHT_RANGE = (9, 20)
WIDTH_RANGE = (15, 29)
SEGMENT_WIDTH_CHOICES = (9, 11, 13)
MARKER_COUNT_CHOICES = (2, 3, 4)
MARKER_COUNT_PROBS = (0.15, 0.30, 0.55)
NON_MARKER_RED_FRAC = (0.30, 0.65)
MARKER_ROW_RED_FRAC = (0.25, 0.55)
MARKER_GAP_RANGE = (1, 3)


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(lo, hi + 1))


def _sample_count(rng: np.random.Generator) -> int:
    return int(rng.choice(np.array(MARKER_COUNT_CHOICES), p=np.array(MARKER_COUNT_PROBS)))


def _random_red_black_row(rng: np.random.Generator, width: int, frac_range: tuple[float, float]) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    lo = max(0, min(width, int(np.ceil(frac_range[0] * width))))
    hi = max(lo, min(width, int(np.floor(frac_range[1] * width))))
    red_count = _randint(rng, lo, hi)
    if red_count:
        cols = rng.choice(width, size=red_count, replace=False)
        row[np.asarray(cols, dtype=int)] = RED
    return row


def _symmetric_segment(rng: np.random.Generator, width: int) -> np.ndarray:
    half = width // 2
    left = rng.choice(np.array([BLACK, RED], dtype=int), size=half, replace=True).astype(int)
    if np.all(left == left[0]):
        flip_at = _randint(rng, 0, half - 1)
        left[flip_at] = RED if int(left[flip_at]) == BLACK else BLACK
    center = int(rng.choice(np.array([BLACK, RED], dtype=int)))
    return np.concatenate([left, np.array([center], dtype=int), left[::-1]])


def _damage_segment(rng: np.random.Generator, segment: np.ndarray, green_count: int) -> np.ndarray:
    width = int(segment.size)
    half = width // 2
    assert 1 <= green_count <= half
    damaged = segment.copy()
    pair_indices = rng.choice(half, size=green_count, replace=False)
    for pair_idx in np.asarray(pair_indices, dtype=int):
        if bool(rng.integers(0, 2)):
            damaged[pair_idx] = GREEN
        else:
            damaged[width - 1 - pair_idx] = GREEN
    return damaged


def _marker_rows(rng: np.random.Generator, height: int, count: int) -> list[int]:
    if count == 1:
        return [_randint(rng, 0, height - 1)]

    min_span = MARKER_GAP_RANGE[0] * (count - 1)
    max_span = min(MARKER_GAP_RANGE[1] * (count - 1), height - 1)
    if min_span > max_span:
        raise ValueError("height too small for spaced marker rows")

    span = _randint(rng, min_span, max_span)
    start = _randint(rng, 0, height - 1 - span)
    remaining = span
    gaps: list[int] = []
    for i in range(count - 1):
        slots = count - i - 2
        lo = max(MARKER_GAP_RANGE[0], remaining - MARKER_GAP_RANGE[1] * slots)
        hi = min(MARKER_GAP_RANGE[1], remaining - MARKER_GAP_RANGE[0] * slots)
        gap = _randint(rng, lo, hi)
        gaps.append(gap)
        remaining -= gap

    rows = [start]
    for gap in gaps:
        rows.append(rows[-1] + gap)
    return rows


def _count_assignment(rng: np.random.Generator, count: int) -> list[int]:
    counts = list(rng.choice(np.array([1, 2, 3, 4], dtype=int), size=count, replace=False).astype(int))
    if count > 1:
        for _ in range(20):
            rng.shuffle(counts)
            if counts != sorted(counts):
                break
        else:
            counts = counts[::-1]
    return counts


def _attempt_generate(rng: np.random.Generator) -> np.ndarray | None:
    marker_count = _sample_count(rng)
    seg_width = int(rng.choice(np.array(SEGMENT_WIDTH_CHOICES)))
    height = _randint(rng, max(HEIGHT_RANGE[0], marker_count + 2), HEIGHT_RANGE[1])
    grid_width = _randint(rng, max(WIDTH_RANGE[0], seg_width + 4), WIDTH_RANGE[1])

    max_left = grid_width - seg_width - 2
    if max_left < 0:
        return None

    grid = np.vstack([
        _random_red_black_row(rng, grid_width, NON_MARKER_RED_FRAC)
        for _ in range(height)
    ])

    rows = _marker_rows(rng, height, marker_count)
    counts = _count_assignment(rng, marker_count)

    left_choices = list(range(0, max_left + 1))
    if not left_choices:
        return None
    left_cols = [int(rng.choice(left_choices)) for _ in range(marker_count)]
    if marker_count > 1 and len(set(left_cols)) == 1 and len(left_choices) > 1:
        idx = _randint(rng, 0, marker_count - 1)
        alternatives = [c for c in left_choices if c != left_cols[idx]]
        left_cols[idx] = int(rng.choice(np.array(alternatives)))

    expected_rows: list[tuple[int, np.ndarray]] = []
    for r, left, green_count in zip(rows, left_cols, counts):
        right = left + seg_width + 1
        row = _random_red_black_row(rng, grid_width, MARKER_ROW_RED_FRAC)
        clean_segment = _symmetric_segment(rng, seg_width)
        damaged_segment = _damage_segment(rng, clean_segment, green_count)

        row[left] = BLUE
        row[left + 1 : right] = damaged_segment
        row[right] = TEAL
        grid[r] = row
        expected_rows.append((green_count, clean_segment))

    expected = np.vstack([segment for _, segment in sorted(expected_rows, key=lambda item: item[0])])

    try:
        actual = main(grid)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None
    if len(_marked_rows(grid)) != marker_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
