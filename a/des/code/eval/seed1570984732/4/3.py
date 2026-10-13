from __future__ import annotations

import random

import numpy as np


BLACK = 0
MAX_GENERATION_ATTEMPTS = 400
SEGMENT_COUNT_RANGE = (3, 5)
SEGMENT_LENGTH_RANGE = (1, 5)
SEPARATOR_LENGTH_RANGE = (1, 3)
GRID_WIDTH_RANGE = (10, 24)
EXTRA_HEIGHT_RANGE = (1, 6)
INPUT_WIDTH_MODE_CHOICES = ("shorter", "equal", "longer")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _parse_ruler(row: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    colors: list[int] = []
    lengths: list[int] = []
    separators: list[int] = []

    idx = 0
    pending_separator: int | None = None
    width = int(row.shape[0])

    while idx < width:
        color = int(row[idx])
        end = idx
        while end < width and int(row[end]) == color:
            end += 1
        run_length = end - idx

        if color == BLACK:
            if colors:
                pending_separator = run_length
        else:
            if colors:
                separators.append(0 if pending_separator is None else pending_separator)
            colors.append(color)
            lengths.append(run_length)
            pending_separator = None

        idx = end

    return colors, lengths, separators


def _stripe_color_rows(grid: np.ndarray, bg: int) -> dict[int, int]:
    color_to_row: dict[int, int] = {}
    height = int(grid.shape[0])

    for r in range(1, height):
        row = grid[r]
        colors = [int(v) for v in np.unique(row) if int(v) != int(bg)]
        if len(colors) != 1:
            continue
        color = colors[0]
        if np.all((row == bg) | (row == color)):
            color_to_row[color] = r

    return color_to_row


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder colored horizontal stripes using a top-edge ruler.

    Input structure:
    - Background color is the most common color.
    - The top row is a ruler made of colored segments separated by black runs.
    - Interior 1-pixel-tall horizontal stripes use the ruler colors, but appear
      in a different order and with arbitrary widths/gaps.

    Rule:
    - Read ruler segments left-to-right to get the target stripe order.
    - The black-run length between two ruler segments gives the number of
      background rows inserted between the corresponding output stripes.
    - Each output stripe is left-aligned and has the same width as its ruler
      segment, cropping or extending the source stripe as needed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)

    bg = _most_common_color(g)
    colors, lengths, separators = _parse_ruler(g[0])
    assert colors, "ruler must contain at least one colored segment"
    assert len(separators) == max(0, len(colors) - 1)

    stripe_rows = _stripe_color_rows(g, bg)

    out = np.full((height, width), bg, dtype=int)
    out[0] = g[0]

    row = 1
    for idx, (color, seg_len) in enumerate(zip(colors, lengths, strict=False)):
        if row < height and color in stripe_rows:
            out[row, : min(seg_len, width)] = int(color)
        row += 1
        if idx < len(separators):
            row += int(separators[idx])

    return out


def _random_composition(total: int, parts: int) -> list[int]:
    values = [0] * parts
    for _ in range(total):
        values[random.randrange(parts)] += 1
    return values


def _build_ruler(colors: list[int], lengths: list[int], separators: list[int]) -> np.ndarray:
    cells: list[int] = []
    for idx, (color, seg_len) in enumerate(zip(colors, lengths, strict=False)):
        cells.extend([int(color)] * int(seg_len))
        if idx < len(separators):
            cells.extend([BLACK] * int(separators[idx]))
    return np.array(cells, dtype=int)


def _build_expected(
    height: int,
    width: int,
    bg: int,
    ruler: np.ndarray,
    colors: list[int],
    lengths: list[int],
    separators: list[int],
) -> np.ndarray:
    out = np.full((height, width), bg, dtype=int)
    out[0] = ruler

    row = 1
    for idx, (color, seg_len) in enumerate(zip(colors, lengths, strict=False)):
        out[row, :seg_len] = int(color)
        row += 1
        if idx < len(separators):
            row += int(separators[idx])

    return out


def _choose_input_width(target_len: int, width: int) -> int:
    modes: list[str] = []
    if target_len > 1:
        modes.append("shorter")
    modes.append("equal")
    if target_len < width:
        modes.append("longer")

    mode = random.choice(modes if modes else list(INPUT_WIDTH_MODE_CHOICES))
    if mode == "shorter":
        return random.randint(1, target_len - 1)
    if mode == "equal":
        return target_len
    return random.randint(target_len + 1, width)


def _dominant_bg(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    counts_by_value = {int(v): int(c) for v, c in zip(values, counts, strict=False)}
    bg_count = counts_by_value.get(int(bg), 0)
    return all(int(color) == int(bg) or bg_count > int(count) for color, count in counts_by_value.items())


def _non_identity_permutation(size: int) -> list[int]:
    order = list(range(size))
    shuffled = order[:]
    while shuffled == order:
        random.shuffle(shuffled)
    return shuffled


def _gen_attempt() -> np.ndarray | None:
    stripe_count = random.randint(*SEGMENT_COUNT_RANGE)

    lengths = [random.randint(*SEGMENT_LENGTH_RANGE) for _ in range(stripe_count)]
    separators = [random.randint(*SEPARATOR_LENGTH_RANGE) for _ in range(stripe_count - 1)]
    width = sum(lengths) + sum(separators)
    if not (GRID_WIDTH_RANGE[0] <= width <= GRID_WIDTH_RANGE[1]):
        return None

    required_output_height = 1 + stripe_count + sum(separators)
    height = required_output_height + random.randint(*EXTRA_HEIGHT_RANGE)

    bg = random.randint(1, 9)
    palette = [color for color in range(1, 10) if color != bg]
    if len(palette) < stripe_count:
        return None
    colors = random.sample(palette, stripe_count)

    ruler = _build_ruler(colors, lengths, separators)
    grid = np.full((height, width), bg, dtype=int)
    grid[0] = ruler

    blank_rows = height - 1 - stripe_count
    row_blanks = _random_composition(blank_rows, stripe_count + 1)
    stripe_rows: list[int] = []
    row = 1 + row_blanks[0]
    for idx in range(stripe_count):
        stripe_rows.append(row)
        row += 1 + row_blanks[idx + 1]

    perm = _non_identity_permutation(stripe_count)
    placed_colors = [colors[idx] for idx in perm]
    placed_lengths = [lengths[idx] for idx in perm]

    for r, color, target_len in zip(stripe_rows, placed_colors, placed_lengths, strict=False):
        stripe_width = _choose_input_width(target_len, width)
        start_col = random.randint(0, width - stripe_width)
        grid[r, start_col : start_col + stripe_width] = int(color)

    expected = _build_expected(height, width, bg, ruler, colors, lengths, separators)
    if not _dominant_bg(grid, bg):
        return None
    if np.array_equal(grid, expected):
        return None

    try:
        solved = main(grid)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
