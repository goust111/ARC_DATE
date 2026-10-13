import random

import numpy as np


BLACK = 0
GREY = 5
ORANGE = 7
DATA_COLORS = (1, 2, 3, 4, 6, 8, 9)

MIN_PATTERN_ROWS = 3
MAX_PATTERN_ROWS = 5
CONTENT_WIDTH_MIN = 4
CONTENT_WIDTH_MAX = 18
LEFT_TOP_PAD_MAX = 2
LEFT_BOTTOM_PAD_MAX = 2
MAX_INSTRUCTIONS = 12
MAX_GENERATION_ATTEMPTS = 300


def _find_separator_col(grid: np.ndarray) -> int:
    candidates = [
        c
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == GREY) and 0 < c < grid.shape[1] - 1
    ]
    assert len(candidates) == 1, "expected exactly one full-height grey separator"
    return int(candidates[0])


def _extract_labeled_patterns(left_panel: np.ndarray) -> dict[int, np.ndarray]:
    assert left_panel.shape[1] >= 2, "left panel must contain labels and pattern content"
    label_rows = [r for r in range(left_panel.shape[0]) if int(left_panel[r, 0]) != BLACK]
    assert label_rows, "left panel must contain at least one labeled pattern row"

    patterns: dict[int, np.ndarray] = {}
    for row in label_rows:
        label = int(left_panel[row, 0])
        assert label not in {BLACK, GREY}, "label colors must be visible and non-separator"
        assert label not in patterns, "label colors must be unique"
        patterns[label] = left_panel[row, 1:].copy()
    return patterns


def _extract_instructions(right_panel: np.ndarray) -> tuple[list[int], list[bool]]:
    assert right_panel.shape[1] >= 1, "right panel must contain an instruction column"

    labels: list[int] = []
    flips: list[bool] = []
    for row in range(right_panel.shape[0]):
        label = int(right_panel[row, 0])
        if label == BLACK:
            continue
        labels.append(label)
        flips.append(right_panel.shape[1] > 1 and int(right_panel[row, 1]) == ORANGE)

    assert labels, "right panel must contain at least one instruction"
    return labels, flips


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the right-panel instruction list into an ordered stack of left-panel pattern rows.

    Input structure:
    - A single full-height grey separator column splits the grid into left and right panels.
    - The left panel contains several colored pattern rows on a black background; each such row starts with a unique one-pixel label color followed by that row's fixed-width pattern content.
    - The right panel starts at the top edge with one instruction per row: a label-colored pixel in the first column, optionally followed by an orange flip marker in the next column.

    Rule:
    - Read the labeled rows from the left panel and map each label color to that row's pattern content, excluding the label pixel itself.
    - Read the right-panel instructions from top to bottom using the label-colored pixel in each occupied instruction row as the lookup key.
    - Copy the matched pattern into the corresponding output row; if the same instruction row has an adjacent orange pixel, reverse the pattern horizontally before writing it.
    - Stack the decoded rows tightly in instruction order to form the output grid.

    Key insight:
    - The left panel is a color-keyed library of reusable rows, while the right panel independently specifies both retrieval order and an optional mirror transform.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    sep_col = _find_separator_col(g)
    left_panel = g[:, :sep_col]
    right_panel = g[:, sep_col + 1 :]

    patterns = _extract_labeled_patterns(left_panel)
    instruction_labels, flip_flags = _extract_instructions(right_panel)

    content_width = left_panel.shape[1] - 1
    out = np.empty((len(instruction_labels), content_width), dtype=int)
    for row, (label, flip) in enumerate(zip(instruction_labels, flip_flags)):
        assert label in patterns, "every instruction must match a left-panel label"
        pattern = patterns[label]
        out[row] = pattern[::-1] if flip else pattern
    return out


def _random_composition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    bounds = [0, *cuts, total]
    return [bounds[i + 1] - bounds[i] for i in range(parts)]


def _make_pattern(width: int, blocked: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(200):
        seg_count = random.randint(2, min(5, width))
        seg_lengths = _random_composition(width, seg_count)
        palette = random.sample(DATA_COLORS, random.randint(2, 4))

        row = np.empty(width, dtype=int)
        cursor = 0
        prev_color = None
        for seg_len in seg_lengths:
            choices = [color for color in palette if color != prev_color]
            color = random.choice(choices)
            row[cursor : cursor + seg_len] = color
            cursor += seg_len
            prev_color = color

        if random.random() < 0.7:
            tweak_count = random.randint(1, max(1, width // 4))
            for _ in range(tweak_count):
                idx = random.randrange(width)
                row[idx] = random.choice([c for c in DATA_COLORS if c != int(row[idx])])

        sig = tuple(int(x) for x in row)
        if len(set(sig)) < 2:
            continue
        if sig == sig[::-1]:
            continue
        if sig in blocked or sig[::-1] in blocked:
            continue
        return row

    raise RuntimeError("failed to sample a non-palindromic pattern row")


def _sample_instruction_labels(labels: list[int]) -> list[int]:
    mode = random.choice(("fewer", "equal", "more"))
    if mode == "fewer":
        count = random.randint(2, len(labels) - 1)
        chosen = random.sample(labels, count)
        random.shuffle(chosen)
        return chosen

    if mode == "equal":
        chosen = labels[:]
        random.shuffle(chosen)
        return chosen

    count = random.randint(len(labels) + 1, min(MAX_INSTRUCTIONS, len(labels) + 5))
    chosen = labels[:]
    while len(chosen) < count:
        chosen.append(random.choice(labels))
    random.shuffle(chosen)
    return chosen


def _sample_flip_flags(count: int) -> list[bool]:
    flags = [random.choice((False, True)) for _ in range(count)]
    if not any(flags):
        flags[random.randrange(count)] = True
    if all(flags):
        flags[random.randrange(count)] = False
    return flags


def _expected_output(
    patterns: dict[int, np.ndarray], instruction_labels: list[int], flip_flags: list[bool]
) -> np.ndarray:
    rows = [patterns[label][::-1] if flip else patterns[label] for label, flip in zip(instruction_labels, flip_flags)]
    return np.stack(rows, axis=0)


def _gen_attempt() -> np.ndarray | None:
    row_count = random.randint(MIN_PATTERN_ROWS, MAX_PATTERN_ROWS)
    content_width = random.randint(CONTENT_WIDTH_MIN, CONTENT_WIDTH_MAX)
    top_pad = random.randint(0, LEFT_TOP_PAD_MAX)
    bottom_pad = random.randint(0, LEFT_BOTTOM_PAD_MAX)

    labels = random.sample(DATA_COLORS, row_count)
    patterns: dict[int, np.ndarray] = {}
    used_patterns: set[tuple[int, ...]] = set()
    for label in labels:
        pattern = _make_pattern(content_width, used_patterns)
        sig = tuple(int(x) for x in pattern)
        patterns[label] = pattern
        used_patterns.add(sig)
        used_patterns.add(sig[::-1])

    instruction_labels = _sample_instruction_labels(labels)
    flip_flags = _sample_flip_flags(len(instruction_labels))
    assert any(flip_flags) and not all(flip_flags), "generator must cover both instruction branches"

    left_height = top_pad + (2 * row_count - 1) + bottom_pad
    total_height = max(left_height, len(instruction_labels))
    sep_col = 1 + content_width
    total_width = sep_col + 1 + 2

    if total_height > 30 or total_width > 30:
        return None

    grid = np.full((total_height, total_width), BLACK, dtype=int)
    grid[:, sep_col] = GREY

    for idx, label in enumerate(labels):
        row = top_pad + 2 * idx
        grid[row, 0] = label
        grid[row, 1:sep_col] = patterns[label]

    for row, (label, flip) in enumerate(zip(instruction_labels, flip_flags)):
        grid[row, sep_col + 1] = label
        if flip:
            grid[row, sep_col + 2] = ORANGE

    expected = _expected_output(patterns, instruction_labels, flip_flags)
    if max(expected.shape) > 30:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
