from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import Color
import random


def _border_background(g: np.ndarray, placeholder: int) -> int:
    border = np.concatenate([g[0, :], g[-1, :], g[:, 0], g[:, -1]])
    counts = Counter(int(v) for v in border if int(v) != placeholder).most_common()
    assert counts, "expected at least one non-placeholder border pixel"
    if len(counts) > 1:
        assert counts[0][1] > counts[1][1], "border background color must be unique"
    return int(counts[0][0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every grey placeholder pixel by the control color at the left edge of its row.

    Input structure
    - The grid has a background color, grey placeholder pixels, and colored row control pixels.
    - Each row containing grey pixels has exactly one non-background, non-grey control pixel in column 0.

    Rule
    - Preserve the background and all non-grey control pixels.
    - For each row, replace every grey pixel by the color of that row's column-0 control pixel.

    Key insight
    - The connected grey shapes are not filled object by object; each row is independently keyed by its left-edge marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.size

    placeholder = int(Color.GREY)
    assert np.any(g == placeholder), "expected grey placeholder pixels"
    bg = _border_background(g, placeholder)
    assert bg != placeholder

    out = g.copy()
    col_indices = np.arange(g.shape[1])
    for r in range(g.shape[0]):
        row = g[r]
        placeholder_mask = row == placeholder
        marked_cols = np.flatnonzero((row != bg) & (row != placeholder))

        if np.any(placeholder_mask):
            control = int(row[0])
            assert control not in (bg, placeholder)
            assert marked_cols.size == 1 and int(marked_cols[0]) == 0
            out[r, placeholder_mask] = control
        else:
            assert np.all((row == bg) | ((col_indices == 0) & (row != placeholder)))

    return out

# --------------- generator (o5x) ---------------

# Grid and colors.
GRID_SIZE, BACKGROUND_COLOR, PLACEHOLDER_COLOR, CONTROL_COL = 12, 0, 5, 0
CONTROL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# Retry and target area bounds.
GEN_RETRIES = 300
MIN_TARGET_AREA, MAX_TARGET_AREA = 36, 79
LARGE_TARGET_AREA, HIGH_AREA_MIN_WIDTH_TARGET = 62, 50

# Layout probabilities.
FULL_ACTIVE_START, TOP_BLANK_ACTIVE_START = 0, 1
TOP_ROW_ACTIVE_PROB, SPINE_PATTERN_PROB = 0.35, 0.55
LOW_AREA_MIN_COL_ONE_PROB, LARGE_AREA_MIN_COL_ONE_PROB = 0.15, 0.50

# Component and rectangle geometry.
MIN_COMPONENTS, MAX_COMPONENTS = 2, 3
MIN_COL_ONE, MIN_COL_DEFAULT, COMPONENT_GAP = 1, 2, 1
MIN_RECT_WIDTH, MAX_RECT_HEIGHT = 2, 4
SHORT_COMPONENT_MIN_HEIGHT, SHORT_COMPONENT_MAX_HEIGHT = 2, 4
LARGE_MIN_WIDTH = 4

# Column band widths.
TWO_COMPONENT_WIDTH_MIN, TWO_COMPONENT_WIDTH_MAX = 2, 5
TWO_COMPONENT_SPLIT_MIN_WIDTH, TWO_COMPONENT_MIN_SUM = 3, 7
THREE_COMPONENT_WIDTH_MIN, THREE_COMPONENT_WIDTH_MAX = 2, 3
THREE_COMPONENT_SPINE_MIN_SUM, THREE_COMPONENT_SPLIT_MIN_SUM = 7, 8
LARGE_WIDTH_MIN, LARGE_WIDTH_MAX = 4, 5

# Row-color runs.
MIN_CONTROL_COLORS, MAX_CONTROL_COLORS = 2, 3
MIN_COLOR_ROW_COUNT, MAX_COLOR_ROW_COUNT, MAX_COLOR_RUN = 3, 7, 4

# Split layout lengths and growth search.
SPLIT_TWO_MIN_LENGTH, SPLIT_TWO_MAX_LENGTH = 4, 8
SPLIT_THREE_FULL_ROWS = [4, 4, 4]
SPLIT_THREE_TOP_BLANK_ROWS = [3, 4, 4]
SPLIT_THREE_EXTRA_ATTACH_MIN, SPLIT_THREE_EXTRA_ATTACH_MAX = 1, 2
GROWTH_LARGE_AREA_MARGIN = 2

PATTERN_SPINE, PATTERN_SPLIT = "spine", "split"


def _gen_column_bands(widths: list[int], min_col: int) -> list[tuple[int, int]]:
    slack = (GRID_SIZE - min_col) - (sum(widths) + COMPONENT_GAP * (len(widths) - 1))
    spaces = [0] * (len(widths) + 1)
    for _ in range(slack):
        spaces[random.randrange(len(spaces))] += 1

    bands = []
    c = min_col + spaces[0]
    for i, w in enumerate(widths):
        bands.append((c, c + w - 1))
        c += w
        if i + 1 < len(widths):
            c += COMPONENT_GAP + spaces[i + 1]
    return bands


def _gen_widths(n_components: int, min_col: int, pattern: str) -> list[int]:
    usable = GRID_SIZE - min_col
    if n_components == 2:
        candidates = [
            (a, b)
            for a in range(TWO_COMPONENT_WIDTH_MIN, TWO_COMPONENT_WIDTH_MAX + 1)
            for b in range(TWO_COMPONENT_WIDTH_MIN, TWO_COMPONENT_WIDTH_MAX + 1)
            if a + b + COMPONENT_GAP <= usable
        ]
        if pattern == PATTERN_SPLIT:
            candidates = [
                p
                for p in candidates
                if min(p) >= TWO_COMPONENT_SPLIT_MIN_WIDTH
                and sum(p) >= TWO_COMPONENT_MIN_SUM
            ]
        else:
            candidates = [p for p in candidates if max(p) >= TWO_COMPONENT_SPLIT_MIN_WIDTH]
    else:
        candidates = [
            (a, b, c)
            for a in range(THREE_COMPONENT_WIDTH_MIN, THREE_COMPONENT_WIDTH_MAX + 1)
            for b in range(THREE_COMPONENT_WIDTH_MIN, THREE_COMPONENT_WIDTH_MAX + 1)
            for c in range(THREE_COMPONENT_WIDTH_MIN, THREE_COMPONENT_WIDTH_MAX + 1)
            if a + b + c + COMPONENT_GAP * 2 <= usable
        ]
        minimum_sum = (
            THREE_COMPONENT_SPLIT_MIN_SUM
            if pattern == PATTERN_SPLIT
            else THREE_COMPONENT_SPINE_MIN_SUM
        )
        candidates = [p for p in candidates if sum(p) >= minimum_sum]

    widths = list(random.choice(candidates))
    random.shuffle(widths)
    return widths


def _gen_height_chunks(height: int) -> list[int]:
    if height <= MAX_RECT_HEIGHT:
        return [height]
    if height <= MAX_RECT_HEIGHT * 2:
        lo = max(1, height - MAX_RECT_HEIGHT)
        hi = min(MAX_RECT_HEIGHT, height - 1)
        first = random.randint(lo, hi)
        return [first, height - first]
    chunks = [MAX_RECT_HEIGHT, MAX_RECT_HEIGHT, height - MAX_RECT_HEIGHT * 2]
    random.shuffle(chunks)
    return chunks


def _gen_subrange(
    band: tuple[int, int], anchor: int | None = None, min_width: int = MIN_RECT_WIDTH
) -> tuple[int, int]:
    c0, c1 = band
    width = c1 - c0 + 1
    if width <= MIN_RECT_WIDTH:
        return c0, c1
    if anchor is None:
        anchor = random.randint(c0, c1)
    w = random.randint(min(width, max(MIN_RECT_WIDTH, min_width)), width)
    lo = max(c0, anchor - w + 1)
    hi = min(anchor, c1 - w + 1)
    left = random.randint(lo, hi)
    return left, left + w - 1


def _gen_paint_vertical_component(
    mask: np.ndarray,
    row0: int,
    row1: int,
    band: tuple[int, int],
    min_width: int = MIN_RECT_WIDTH,
) -> None:
    c0, c1 = band
    width = c1 - c0 + 1
    anchor = random.randint(c0 + 1, c1 - 1) if width > 2 else c0
    r = row0
    for h in _gen_height_chunks(row1 - row0 + 1):
        a, b = _gen_subrange(band, anchor, min_width)
        mask[r : r + h, a : b + 1] = True
        r += h


def _gen_attach_rect(
    mask: np.ndarray,
    base: tuple[int, int],
    band: tuple[int, int],
    active_start: int,
    active_end: int,
    min_width: int = MIN_RECT_WIDTH,
) -> tuple[int, int, int, int] | None:
    row0, row1 = base
    choices = []
    if row0 > active_start:
        choices.append("up")
    if row1 < active_end:
        choices.append("down")
    if not choices:
        return None

    direction = random.choice(choices)
    if direction == "up":
        max_h = min(MAX_RECT_HEIGHT, row0 - active_start)
        min_h = SHORT_COMPONENT_MIN_HEIGHT if max_h >= SHORT_COMPONENT_MIN_HEIGHT else 1
        h = random.randint(min_h, max_h)
        r0, r1 = row0 - h, row0 - 1
    else:
        max_h = min(MAX_RECT_HEIGHT, active_end - row1)
        min_h = SHORT_COMPONENT_MIN_HEIGHT if max_h >= SHORT_COMPONENT_MIN_HEIGHT else 1
        h = random.randint(min_h, max_h)
        r0, r1 = row1 + 1, row1 + h

    c0, c1 = _gen_subrange(band, min_width=min_width)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return r0, c0, r1, c1


def _gen_color_counts(n_colors: int, n_rows: int) -> list[int]:
    if n_colors == MIN_CONTROL_COLORS:
        candidates = [
            (a, n_rows - a)
            for a in range(MIN_COLOR_ROW_COUNT, MAX_COLOR_ROW_COUNT + 1)
            if MIN_COLOR_ROW_COUNT <= n_rows - a <= MAX_COLOR_ROW_COUNT
        ]
    else:
        candidates = [
            (a, b, n_rows - a - b)
            for a in range(MIN_COLOR_ROW_COUNT, MAX_COLOR_ROW_COUNT + 1)
            for b in range(MIN_COLOR_ROW_COUNT, MAX_COLOR_ROW_COUNT + 1)
            if MIN_COLOR_ROW_COUNT <= n_rows - a - b <= MAX_COLOR_ROW_COUNT
        ]
    return list(random.choice(candidates))


def _gen_count_chunks(count: int) -> list[int]:
    if count <= MAX_COLOR_RUN:
        return [count]
    first = random.randint(max(1, count - MAX_COLOR_RUN), min(MAX_COLOR_RUN, count - 1))
    chunks = [first, count - first]
    random.shuffle(chunks)
    return chunks


def _gen_row_colors(n_rows: int) -> list[int] | None:
    n_colors = random.randint(MIN_CONTROL_COLORS, MAX_CONTROL_COLORS)
    colors = random.sample(CONTROL_COLORS, n_colors)
    chunks_by_color = {
        color: _gen_count_chunks(count)
        for color, count in zip(colors, _gen_color_counts(n_colors, n_rows))
    }

    seq = []
    prev = None
    while any(chunks_by_color.values()):
        choices = [c for c, chunks in chunks_by_color.items() if chunks and c != prev]
        if not choices:
            return None
        max_chunks = max(len(chunks_by_color[c]) for c in choices)
        best = [c for c in choices if len(chunks_by_color[c]) == max_chunks]
        max_remaining = max(sum(chunks_by_color[c]) for c in best)
        best = [c for c in best if sum(chunks_by_color[c]) == max_remaining]
        color = random.choice(best)
        seq.extend([color] * chunks_by_color[color].pop(0))
        prev = color
    return seq if len(seq) == n_rows else None


def _gen_spine_mask(
    mask: np.ndarray,
    bands: list[tuple[int, int]],
    active_start: int,
    min_width: int,
) -> None:
    active_end = GRID_SIZE - 1
    spine_idx = max(range(len(bands)), key=lambda i: bands[i][1] - bands[i][0])
    _gen_paint_vertical_component(
        mask, active_start, active_end, bands[spine_idx], min_width
    )

    for i, band in enumerate(bands):
        if i == spine_idx:
            continue
        h = random.randint(SHORT_COMPONENT_MIN_HEIGHT, SHORT_COMPONENT_MAX_HEIGHT)
        r0 = random.randint(active_start, active_end - h + 1)
        c0, c1 = band
        mask[r0 : r0 + h, c0 : c1 + 1] = True

def _gen_split_mask(
    mask: np.ndarray,
    bands: list[tuple[int, int]],
    active_start: int,
    min_width: int,
) -> None:
    active_end = GRID_SIZE - 1
    n_components = len(bands)
    n_rows = active_end - active_start + 1
    if n_components == 2:
        length_options = [
            (a, n_rows - a)
            for a in range(
                SPLIT_TWO_MIN_LENGTH,
                min(SPLIT_TWO_MAX_LENGTH, n_rows - (SPLIT_TWO_MIN_LENGTH - 1)) + 1,
            )
            if SPLIT_TWO_MIN_LENGTH <= n_rows - a <= SPLIT_TWO_MAX_LENGTH
        ]
        lengths = list(random.choice(length_options))
    else:
        lengths = (
            SPLIT_THREE_FULL_ROWS.copy()
            if n_rows == GRID_SIZE
            else SPLIT_THREE_TOP_BLANK_ROWS.copy()
        )
        random.shuffle(lengths)

    intervals = []
    r = active_start
    for length in lengths:
        intervals.append((r, r + length - 1))
        r += length

    for interval, band in zip(intervals, bands):
        _gen_paint_vertical_component(mask, interval[0], interval[1], band, min_width)

    if n_components == 3:
        n_extra = random.randint(
            SPLIT_THREE_EXTRA_ATTACH_MIN, SPLIT_THREE_EXTRA_ATTACH_MAX
        )
        for idx in random.sample(range(n_components), n_extra):
            _gen_attach_rect(
                mask, intervals[idx], bands[idx], active_start, active_end, min_width
            )


def _gen_large_widths(min_col: int, n_rows: int) -> list[int]:
    usable = GRID_SIZE - min_col
    candidates = []
    for a in range(LARGE_WIDTH_MIN, LARGE_WIDTH_MAX + 1):
        for b in range(LARGE_WIDTH_MIN, LARGE_WIDTH_MAX + 1):
            if a + b + COMPONENT_GAP > usable:
                continue
            max_w, min_w = max(a, b), min(a, b)
            if n_rows * max_w + MAX_RECT_HEIGHT * min_w <= MAX_TARGET_AREA:
                candidates.append((a, b))
    widths = list(random.choice(candidates))
    random.shuffle(widths)
    return widths


def _gen_rect_touches(mask: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    return (
        (r0 > 0 and np.any(mask[r0 - 1, c0 : c1 + 1]))
        or (r1 + 1 < mask.shape[0] and np.any(mask[r1 + 1, c0 : c1 + 1]))
        or (c0 > 0 and np.any(mask[r0 : r1 + 1, c0 - 1]))
        or (c1 + 1 < mask.shape[1] and np.any(mask[r0 : r1 + 1, c1 + 1]))
    )

def _gen_grow_to_area(
    mask: np.ndarray,
    bands: list[tuple[int, int]],
    active_start: int,
    target_area: int,
) -> bool:
    while int(mask.sum()) < target_area:
        remaining = target_area - int(mask.sum())
        candidates = []
        for c0, c1 in bands:
            band_width = c1 - c0 + 1
            for h in range(1, min(MAX_RECT_HEIGHT, remaining) + 1):
                max_w = min(band_width, remaining // h)
                min_w = 1 if remaining == 1 else MIN_RECT_WIDTH
                for w in range(min_w, max_w + 1):
                    for r in range(active_start, GRID_SIZE + 1 - h):
                        for c in range(c0, c1 - w + 2):
                            if np.any(mask[r : r + h, c : c + w]):
                                continue
                            if _gen_rect_touches(mask, r, c, h, w):
                                candidates.append((h * w, r, c, h, w))
        if not candidates:
            return False
        candidates.sort(reverse=True)
        max_area = candidates[0][0]
        large = [cand for cand in candidates if cand[0] >= max(1, max_area - GROWTH_LARGE_AREA_MARGIN)]
        _, r, c, h, w = random.choice(large)
        mask[r : r + h, c : c + w] = True
    return True


def _gen_layout(active_start: int, target_area: int) -> tuple[str, list[tuple[int, int]], int]:
    if target_area >= LARGE_TARGET_AREA:
        pattern = PATTERN_SPINE
        use_col_one = active_start == TOP_BLANK_ACTIVE_START and random.random() < LARGE_AREA_MIN_COL_ONE_PROB
        min_col = MIN_COL_ONE if use_col_one else MIN_COL_DEFAULT
        widths = _gen_large_widths(min_col, GRID_SIZE - active_start)
        min_width = LARGE_MIN_WIDTH
    else:
        n_components = random.randint(MIN_COMPONENTS, MAX_COMPONENTS)
        pattern = PATTERN_SPINE if random.random() < SPINE_PATTERN_PROB else PATTERN_SPLIT
        min_col = MIN_COL_ONE if random.random() < LOW_AREA_MIN_COL_ONE_PROB else MIN_COL_DEFAULT
        widths = _gen_widths(n_components, min_col, pattern)
        min_width = TWO_COMPONENT_SPLIT_MIN_WIDTH if target_area >= HIGH_AREA_MIN_WIDTH_TARGET else MIN_RECT_WIDTH

    return pattern, _gen_column_bands(widths, min_col), min_width

def _mask_matches_constraints(
    mask: np.ndarray,
    target_area: int,
    active_rows: list[int],
    active_start: int,
) -> bool:
    area = int(mask.sum())
    if area != target_area or not (MIN_TARGET_AREA <= area <= MAX_TARGET_AREA):
        return False
    if np.any(mask[:, CONTROL_COL]):
        return False
    if not all(np.any(mask[r]) for r in active_rows):
        return False
    if active_start == TOP_BLANK_ACTIVE_START and np.any(mask[0]):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    active_start = FULL_ACTIVE_START if random.random() < TOP_ROW_ACTIVE_PROB else TOP_BLANK_ACTIVE_START
    active_rows = list(range(active_start, GRID_SIZE))
    target_area = random.randint(MIN_TARGET_AREA, MAX_TARGET_AREA)

    pattern, bands, min_width = _gen_layout(active_start, target_area)
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    if pattern == PATTERN_SPINE:
        _gen_spine_mask(mask, bands, active_start, min_width)
    else:
        _gen_split_mask(mask, bands, active_start, min_width)

    if int(mask.sum()) > target_area:
        return None
    if not _gen_grow_to_area(mask, bands, active_start, target_area):
        return None
    if not _mask_matches_constraints(mask, target_area, active_rows, active_start):
        return None

    row_colors = _gen_row_colors(len(active_rows))
    if row_colors is None:
        return None

    out = np.full((GRID_SIZE, GRID_SIZE), BACKGROUND_COLOR, dtype=int)
    for r, color in zip(active_rows, row_colors):
        out[r, CONTROL_COL] = color
        out[r, mask[r]] = color
    inp = out.copy()
    inp[mask] = PLACEHOLDER_COLOR

    try:
        actual = main(inp)
    except Exception:
        return None
    return inp if np.array_equal(actual, out) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid c9f8e694 input")
