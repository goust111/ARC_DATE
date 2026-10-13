from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

PANEL_SIZE = 3

SYMBOL_TO_COLOR = {
    ((1, 0, 0), (1, 0, 0), (1, 1, 1)): RED,
    ((0, 1, 0), (1, 1, 1), (0, 1, 0)): BLUE,
    ((1, 0, 0), (0, 1, 0), (0, 0, 1)): GREEN,
    ((0, 0, 0), (1, 1, 1), (0, 0, 0)): YELLOW,
}

COLOR_TO_SYMBOL = {color: np.array(mask, dtype=int) for mask, color in SYMBOL_TO_COLOR.items()}

# Curated count profiles. They keep every color present while ensuring that after
# dropping the highest-frequency tier, the remaining bars have a unique order.
COUNT_PROFILES = (
    (4, 3, 2, 1),  # unique highest
    (3, 3, 2, 1),  # two-way tie for highest
    (3, 3, 3, 1),  # three-way tie for highest
)


def _panel_mask(block: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in (block == GREY).astype(int))


def _decode_colors(grid: np.ndarray) -> list[int]:
    height, width = grid.shape
    assert height == PANEL_SIZE, "expected a single row of 3x3 symbol panels"
    assert width % PANEL_SIZE == 0, "expected whole 3x3 panels"

    decoded: list[int] = []
    for c0 in range(0, width, PANEL_SIZE):
        block = grid[:, c0 : c0 + PANEL_SIZE]
        mask = _panel_mask(block)
        assert mask in SYMBOL_TO_COLOR, "unknown symbol mask"
        decoded.append(int(SYMBOL_TO_COLOR[mask]))
    return decoded


def _remaining_bars(color_counts: Counter[int]) -> list[tuple[int, int]]:
    highest = max(color_counts.values())
    remaining = [
        (int(color), int(count))
        for color, count in color_counts.items()
        if int(count) != int(highest)
    ]
    assert remaining, "at least one color must remain after skipping the most frequent tier"
    return sorted(remaining, key=lambda item: (-item[1], item[0]))


def _histogram_from_counts(color_counts: Counter[int]) -> np.ndarray:
    remaining = _remaining_bars(color_counts)
    height = max(count for _, count in remaining)
    out = np.full((height, len(remaining)), BLACK, dtype=int)
    for col, (color, count) in enumerate(remaining):
        out[:count, col] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each 3x3 grey symbol panel, drop the most frequent decoded color tier, and draw the remaining frequency histogram.

    Input structure:
    - The grid is 3 rows high and split into adjacent 3x3 panels with no gaps.
    - Each panel contains only black background cells and grey symbol cells.
    - Every grey mask is one of the task's learned symbol glyphs, each standing for one output color.

    Rule:
    - Read each 3x3 panel as a binary grey mask and decode it with the task codebook.
    - Count how many panels decode to each color.
    - Remove every color tied for the highest count.
    - Sort the remaining colors by decreasing count, breaking equal counts by smaller color id.
    - Build a black output whose width is the number of remaining colors and whose height is the largest remaining count, then fill each column from the top with that color for exactly its count.

    Key insight:
    - The puzzle is a dependency chain: symbol shape determines color, color counts determine which colors survive the skip-highest rule, and only then can the histogram order and heights be constructed.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] == PANEL_SIZE, "expected height 3"
    assert g.shape[1] >= PANEL_SIZE, "expected at least one 3x3 panel"
    assert g.shape[1] % PANEL_SIZE == 0, "expected an integer number of panels"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "input must use only black and grey"

    decoded = _decode_colors(g)
    counts = Counter(decoded)
    return _histogram_from_counts(counts)


def _render_panels(decoded_colors: list[int]) -> np.ndarray:
    grid = np.full((PANEL_SIZE, PANEL_SIZE * len(decoded_colors)), BLACK, dtype=int)
    for panel_i, color in enumerate(decoded_colors):
        symbol = COLOR_TO_SYMBOL[int(color)]
        c0 = panel_i * PANEL_SIZE
        grid[:, c0 : c0 + PANEL_SIZE] = np.where(symbol == 1, GREY, BLACK)
    return grid


def _sample_profile() -> tuple[int, ...]:
    # Bias slightly toward tied-highest cases so the hidden branch appears often.
    return random.choice((COUNT_PROFILES[0], COUNT_PROFILES[1], COUNT_PROFILES[1], COUNT_PROFILES[2]))


def _sample_decoded_sequence() -> list[int]:
    colors = [BLUE, RED, GREEN, YELLOW]
    profile = list(_sample_profile())
    random.shuffle(profile)
    assignment = dict(zip(colors, profile))

    decoded: list[int] = []
    for color in colors:
        decoded.extend([color] * assignment[color])
    random.shuffle(decoded)
    return decoded


def _grey_pixel_count_per_symbol(decoded_colors: list[int]) -> Counter[int]:
    return Counter(int(COLOR_TO_SYMBOL[color].sum()) for color in decoded_colors)


def _validate_generated(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    if inp.shape[0] != PANEL_SIZE or inp.shape[1] % PANEL_SIZE != 0 or max(inp.shape) > 30:
        return False
    if set(map(int, np.unique(inp))) - {BLACK, GREY}:
        return False

    decoded = _decode_colors(inp)
    counts = Counter(decoded)
    if set(counts) != {BLUE, RED, GREEN, YELLOW}:
        return False

    highest = max(counts.values())
    highest_colors = [color for color, count in counts.items() if count == highest]
    if len(highest_colors) not in (1, 2, 3):
        return False
    if len(highest_colors) == 4:
        return False

    remaining = _remaining_bars(counts)
    remaining_counts = [count for _, count in remaining]
    if len(remaining_counts) != len(set(remaining_counts)):
        return False

    # Shape decoding must matter: every sample contains both ambiguous-by-pixel-count pairs.
    grey_count_buckets = _grey_pixel_count_per_symbol(decoded)
    if grey_count_buckets[3] == 0 or grey_count_buckets[5] == 0:
        return False

    out = main(inp)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    if out.size == 0 or np.all(out == BLACK):
        return False

    expected_width = len(remaining)
    expected_height = max(remaining_counts)
    if out.shape != (expected_height, expected_width):
        return False

    for col, (color, count) in enumerate(remaining):
        if not np.all(out[:count, col] == color):
            return False
        if not np.all(out[count:, col] == BLACK):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        decoded = _sample_decoded_sequence()
        inp = _render_panels(decoded)
        if _validate_generated(inp):
            return inp
    raise RuntimeError("failed to generate a non-degenerate symbol-histogram input")
