from __future__ import annotations

import random
from collections import Counter

import numpy as np


PALETTE = tuple(range(10))

# Generator configuration.
KEY_COUNT_RANGE = (2, 4)
INNER_WIDTH_RANGE = (5, 10)
EXTRA_ROWS_RANGE = (2, 5)
SEGMENT_COUNT_RANGE = (2, 4)
EXTRA_FILLER_COLORS_RANGE = (1, 3)
MAX_ATTEMPTS = 400


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a control column, use the top-row index colors to locate template rows,
    and stack the referenced template rows in control order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 2 and w >= 2, "grid must contain control/index margins"

    bg = get_bg(g)

    color_to_col: dict[int, int] = {}
    for c in range(1, w):
        color = int(g[0, c])
        if color == bg:
            continue
        assert color not in color_to_col, "index-row colors must be unique"
        color_to_col[color] = c
    assert color_to_col, "expected at least one indexed color in the top row"

    candidate_rows = [r for r in range(1, h) if np.any(g[r, 1:] != bg)]
    assert candidate_rows, "expected at least one template row"

    template_by_color: dict[int, np.ndarray] = {}
    for color, col in color_to_col.items():
        matches = [r for r in candidate_rows if int(g[r, col]) == color]

        if len(matches) != 1:
            labeled_matches = [r for r in matches if int(g[r, 1]) == color]
            if len(labeled_matches) == 1:
                matches = labeled_matches
            else:
                fallback = [r for r in candidate_rows if int(g[r, 1]) == color]
                assert len(fallback) == 1, "color-to-row mapping must be unambiguous"
                matches = fallback

        template_by_color[color] = g[matches[0], 1:].copy()

    control_sequence = [int(g[r, 0]) for r in range(1, h) if int(g[r, 0]) != bg]
    assert control_sequence, "control column must contain at least one non-background color"
    assert all(color in template_by_color for color in control_sequence), "unknown control color"

    return np.stack([template_by_color[color] for color in control_sequence], axis=0)


def _pick_distinct_colors(rng: random.Random, bg: int, count: int) -> list[int]:
    return rng.sample([c for c in PALETTE if c != bg], count)


def _choice_excluding(rng: random.Random, colors: list[int], banned: set[int]) -> int:
    options = [c for c in colors if c not in banned]
    if not options:
        options = list(colors)
    return int(rng.choice(options))


def _make_segment_row(
    rng: random.Random,
    inner_w: int,
    key_color: int,
    key_col: int,
    color_to_col: dict[int, int],
    palette: list[int],
) -> np.ndarray:
    segment_count = rng.randint(SEGMENT_COUNT_RANGE[0], min(SEGMENT_COUNT_RANGE[1], inner_w))
    cuts = sorted(rng.sample(range(1, inner_w), segment_count - 1))
    edges = [0, *cuts, inner_w]

    row = np.empty(inner_w, dtype=int)
    prev_color = None
    for start, end in zip(edges, edges[1:]):
        allowed = [c for c in palette if c != prev_color]
        seg_color = int(rng.choice(allowed))
        row[start:end] = seg_color
        prev_color = seg_color

    row[0] = key_color
    row[key_col - 1] = key_color

    # Keep the column-based lookup unique for every indexed color.
    for other_color, other_col in color_to_col.items():
        idx = other_col - 1
        if other_color == key_color:
            row[idx] = key_color
            continue
        if int(row[idx]) == other_color:
            row[idx] = _choice_excluding(rng, palette, {other_color})

    # Add a small accent or two so rows are visually distinct.
    free_positions = [i for i in range(1, inner_w) if i != key_col - 1]
    rng.shuffle(free_positions)
    accent_count = min(len(free_positions), rng.randint(0, 2))
    for pos in free_positions[:accent_count]:
        banned = {int(row[pos])}
        if pos + 1 < inner_w:
            banned.add(int(row[pos + 1]))
        if pos - 1 >= 0:
            banned.add(int(row[pos - 1]))
        row[pos] = _choice_excluding(rng, palette, banned)

    if len(set(map(int, row.tolist()))) == 1:
        pos = 1 if inner_w > 2 and key_col - 1 != 1 else inner_w - 1
        row[pos] = _choice_excluding(rng, palette, {key_color})

    for other_color, other_col in color_to_col.items():
        idx = other_col - 1
        if other_color == key_color:
            row[idx] = key_color
        elif int(row[idx]) == other_color:
            row[idx] = _choice_excluding(rng, palette, {other_color})

    assert len(set(map(int, row.tolist()))) >= 2, "template rows must be multi-color"
    return row


def _make_control_sequence(
    rng: random.Random,
    key_colors: list[int],
    length: int,
    repeat_mode: bool,
) -> list[int]:
    if repeat_mode:
        while True:
            seq = [int(rng.choice(key_colors)) for _ in range(length)]
            if len(set(seq)) < len(seq) and len(set(seq)) >= 2:
                return seq

    seq = rng.sample(key_colors, length)
    rng.shuffle(seq)
    return [int(c) for c in seq]


def _control_rows(
    rng: random.Random,
    inner_h: int,
    template_rows: list[int],
    control_len: int,
) -> list[int]:
    all_rows = list(range(1, inner_h + 1))
    overlap_mode = rng.random() < 0.5

    if overlap_mode:
        max_overlap = min(len(template_rows), control_len)
        overlap_count = rng.randint(1, max_overlap)
        overlap_rows = rng.sample(template_rows, overlap_count)
        remaining_pool = [r for r in all_rows if r not in overlap_rows]
        if control_len - overlap_count > len(remaining_pool):
            chosen = overlap_rows + remaining_pool
        else:
            chosen = overlap_rows + rng.sample(remaining_pool, control_len - overlap_count)
        return sorted(chosen)

    non_template_rows = [r for r in all_rows if r not in template_rows]
    if len(non_template_rows) >= control_len:
        return sorted(rng.sample(non_template_rows, control_len))
    return sorted(rng.sample(all_rows, control_len))


def _bg_is_unique_mode(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.ravel()))
    bg_count = counts[int(bg)]
    return all(color == bg or bg_count > count for color, count in counts.items())


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = int(rng.choice(PALETTE))

    key_count = rng.randint(*KEY_COUNT_RANGE)
    inner_w = rng.randint(max(INNER_WIDTH_RANGE[0], key_count + 1), INNER_WIDTH_RANGE[1])
    repeat_mode = rng.random() < 0.5

    if repeat_mode:
        inner_h = rng.randint(key_count + EXTRA_ROWS_RANGE[0], key_count + EXTRA_ROWS_RANGE[1] + 2)
        control_len = rng.randint(3, min(7, inner_h))
    else:
        inner_h = rng.randint(key_count + EXTRA_ROWS_RANGE[0], key_count + EXTRA_ROWS_RANGE[1])
        control_len = rng.randint(2, min(key_count, inner_h))

    h, w = inner_h + 1, inner_w + 1
    grid = np.full((h, w), bg, dtype=int)

    extra_colors = rng.randint(*EXTRA_FILLER_COLORS_RANGE)
    palette = _pick_distinct_colors(rng, bg, key_count + extra_colors)
    key_colors = palette[:key_count]
    filler_colors = palette[key_count:]
    row_palette = key_colors + filler_colors

    key_cols = sorted(rng.sample(range(2, w), key_count))
    color_to_col = {int(color): int(col) for color, col in zip(key_colors, key_cols)}
    for color, col in color_to_col.items():
        grid[0, col] = color

    template_rows = sorted(rng.sample(range(1, h), key_count))
    rng.shuffle(template_rows)

    template_by_color: dict[int, np.ndarray] = {}
    occupied_template_rows = []
    for color, row_idx in zip(key_colors, template_rows):
        template = _make_segment_row(rng, inner_w, int(color), color_to_col[int(color)], color_to_col, row_palette)
        grid[row_idx, 1:] = template
        template_by_color[int(color)] = template.copy()
        occupied_template_rows.append(int(row_idx))

    if len({tuple(map(int, row.tolist())) for row in template_by_color.values()}) != key_count:
        return None

    control_rows = _control_rows(rng, inner_h, occupied_template_rows, control_len)
    control_sequence = _make_control_sequence(rng, key_colors, control_len, repeat_mode)
    for row_idx, color in zip(control_rows, control_sequence):
        grid[row_idx, 0] = int(color)

    expected = np.stack([template_by_color[int(color)] for color in control_sequence], axis=0)

    if not _bg_is_unique_mode(grid, bg):
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
