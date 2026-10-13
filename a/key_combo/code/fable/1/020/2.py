from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color


STRIPE_COLOR_POOL = list(Color.NOT_BLACK)
NOISE_COLOR_POOL = list(Color.ALL_COLORS)

MIN_TOTAL_WIDTH = 6
MAX_TOTAL_WIDTH = 15
MIN_STRIPE_COUNT = 3
MAX_STRIPE_COUNT = 6
MIN_ORTHOGONAL = 7
MAX_ORTHOGONAL = 18

GEN_MAX_ATTEMPTS = 500
CONFIG_TRIES = 120
NOISE_TRIES = 160


def _line_majority(line: np.ndarray) -> tuple[int | None, int, bool]:
    values, counts = np.unique(np.asarray(line, dtype=int), return_counts=True)
    best = int(counts.max())
    winners = values[counts == best]
    return (int(winners[0]) if len(winners) == 1 else None), best, len(winners) == 1


def _axis_profile(grid: np.ndarray, axis: int) -> tuple[int, list[int | None], list[bool]]:
    score = 0
    colors: list[int | None] = []
    unique_flags: list[bool] = []
    line_count = int(grid.shape[axis])
    for idx in range(line_count):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color, count, is_unique = _line_majority(line)
        score += count
        colors.append(color)
        unique_flags.append(is_unique)
    return score, colors, unique_flags


def _compress_colors(colors: list[int | None]) -> list[tuple[int, int]]:
    sequence: list[list[int]] = []
    for color in colors:
        assert color is not None, "chosen stripe direction must have unique dominant colors"
        if not sequence or sequence[-1][0] != int(color):
            sequence.append([int(color), 1])
        else:
            sequence[-1][1] += 1
    return [(int(color), int(width)) for color, width in sequence]


def _build_nested_square(sequence: list[tuple[int, int]]) -> np.ndarray:
    total = sum(int(width) for _color, width in sequence)
    side = 2 * total
    out = np.empty((side, side), dtype=int)

    top = 0
    left = 0
    bottom = side
    right = side
    for color, width in sequence:
        out[top:bottom, left:right] = int(color)
        top += int(width)
        left += int(width)
        bottom -= int(width)
        right -= int(width)
    return out


def _expanded_line_colors(widths: list[int], colors: list[int]) -> list[int]:
    expanded: list[int] = []
    for width, color in zip(widths, colors):
        expanded.extend([int(color)] * int(width))
    return expanded


def _sequence_and_axis(grid: np.ndarray) -> tuple[int, list[tuple[int, int]], tuple[int, int]]:
    row_score, row_colors, row_unique = _axis_profile(grid, axis=0)
    col_score, col_colors, col_unique = _axis_profile(grid, axis=1)
    assert row_score != col_score, "stripe orientation must be uniquely determined by majority scores"

    if row_score > col_score:
        assert all(row_unique), "every true stripe row must keep a unique dominant color"
        return 0, _compress_colors(row_colors), (row_score, col_score)

    assert all(col_unique), "every true stripe column must keep a unique dominant color"
    return 1, _compress_colors(col_colors), (row_score, col_score)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise a striped canvas by line majorities, then roll the recovered stripe widths into concentric square rings.

    Input structure
    - The grid is completely tiled by horizontal or vertical color stripes with different colors and varying widths.
    - Sparse noise pixels are scattered on top, but every true stripe line still has a strict majority in its stripe color.
    - The correct stripe orientation is the one whose rows or columns achieve the higher total majority score.

    Rule
    - Score the horizontal interpretation by summing the dominant-color count of every row, and score the vertical interpretation the same way over columns.
    - Choose the unique higher-scoring orientation, then read the cleaned stripe sequence from top to bottom or left to right as color/width pairs.
    - Let T be the sum of the recovered widths, and output a 2T x 2T square where each stripe becomes the next inner square layer with thickness equal to its width.

    Key insight
    - The canvas is a noisy radial unrolling of a concentric square pattern: line majorities recover the clean stripe order, and each stripe width directly becomes one ring thickness.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _axis, sequence, _scores = _sequence_and_axis(g)
    return _build_nested_square(sequence)


def _random_partition(total: int, parts: int) -> list[int]:
    cuts = sorted(random.sample(range(1, total), parts - 1))
    widths = []
    prev = 0
    for cut in cuts + [total]:
        widths.append(int(cut - prev))
        prev = cut
    return widths


def _sample_widths(total: int, stripe_count: int) -> list[int] | None:
    for _ in range(CONFIG_TRIES):
        widths = _random_partition(total, stripe_count)
        if len(set(widths)) < 2:
            continue
        if max(widths) >= total - 1:
            continue
        if max(widths) == 1:
            continue
        if 1 not in widths and random.random() < 0.7:
            continue
        return widths
    return None


def _render_clean_grid(orientation: str, widths: list[int], colors: list[int], orthogonal: int) -> np.ndarray:
    total = sum(widths)
    if orientation == "h":
        grid = np.empty((total, orthogonal), dtype=int)
        start = 0
        for width, color in zip(widths, colors):
            grid[start : start + width, :] = int(color)
            start += width
        return grid

    grid = np.empty((orthogonal, total), dtype=int)
    start = 0
    for width, color in zip(widths, colors):
        grid[:, start : start + width] = int(color)
        start += width
    return grid


def _sample_clean_configuration() -> tuple[str, list[int], list[int], np.ndarray] | None:
    for _ in range(CONFIG_TRIES):
        total = random.randint(MIN_TOTAL_WIDTH, MAX_TOTAL_WIDTH)
        max_stripes = min(MAX_STRIPE_COUNT, len(STRIPE_COLOR_POOL), total - 2)
        if max_stripes < MIN_STRIPE_COUNT:
            continue

        stripe_count = random.randint(MIN_STRIPE_COUNT, max_stripes)
        widths = _sample_widths(total, stripe_count)
        if widths is None:
            continue

        colors = random.sample(STRIPE_COLOR_POOL, stripe_count)
        orth_low = max(MIN_ORTHOGONAL, total - 4)
        orth_high = min(MAX_ORTHOGONAL, total + 6)
        if orth_low > orth_high:
            continue
        orthogonal = random.randint(orth_low, orth_high)
        orientation = random.choice(("h", "v"))
        clean = _render_clean_grid(orientation, widths, colors, orthogonal)
        if max(clean.shape) <= 30 and 2 * total <= 30:
            return orientation, widths, colors, clean
    return None


def _edge_orthogonal_indices(length: int) -> list[int]:
    return [0, int(length) - 1]


def _pick_noise_positions(
    clean: np.ndarray,
    orientation: str,
    stripe_ids: list[int],
    noise_count: int,
) -> set[tuple[int, int]] | None:
    true_len = len(stripe_ids)
    orth_len = int(clean.shape[1] if orientation == "h" else clean.shape[0])
    per_line_cap = max(2, (orth_len - 1) // 3)

    line_noise = [0] * true_len
    positions: set[tuple[int, int]] = set()

    def add(line_idx: int, orth_idx: int) -> bool:
        if line_noise[line_idx] >= per_line_cap:
            return False
        pos = (line_idx, orth_idx) if orientation == "h" else (orth_idx, line_idx)
        if pos in positions:
            return False
        positions.add(pos)
        line_noise[line_idx] += 1
        return True

    by_stripe: dict[int, list[int]] = {}
    for line_idx, stripe_id in enumerate(stripe_ids):
        by_stripe.setdefault(int(stripe_id), []).append(int(line_idx))

    stripe_order = list(by_stripe)
    random.shuffle(stripe_order)
    used_stripes: set[int] = set()
    for orth_idx in _edge_orthogonal_indices(orth_len):
        placed = False
        for stripe_id in stripe_order:
            if stripe_id in used_stripes and len(by_stripe) > 1:
                continue
            line_idx = random.choice(by_stripe[stripe_id])
            if add(line_idx, orth_idx):
                used_stripes.add(stripe_id)
                placed = True
                break
        if not placed:
            for stripe_id in stripe_order:
                line_idx = random.choice(by_stripe[stripe_id])
                if add(line_idx, orth_idx):
                    placed = True
                    break
        if not placed:
            return None

    candidates = [(line_idx, orth_idx) for line_idx in range(true_len) for orth_idx in range(orth_len)]
    random.shuffle(candidates)
    for line_idx, orth_idx in candidates:
        if len(positions) >= noise_count:
            break
        add(int(line_idx), int(orth_idx))

    if len(positions) != noise_count:
        return None
    return positions


def _noise_color(
    original: int,
    line_idx: int,
    expanded_colors: list[int],
    stripe_colors: list[int],
) -> int:
    pool: list[int] = []
    for neighbor_idx in (line_idx - 1, line_idx + 1):
        if 0 <= neighbor_idx < len(expanded_colors):
            neighbor_color = int(expanded_colors[neighbor_idx])
            if neighbor_color != original:
                pool.extend([neighbor_color] * 3)

    pool.extend(int(color) for color in stripe_colors if int(color) != int(original))
    pool.extend(int(color) for color in NOISE_COLOR_POOL if int(color) not in stripe_colors and int(color) != int(original))
    return random.choice(pool)


def _output_from_axis_if_unique(grid: np.ndarray, axis: int) -> np.ndarray | None:
    _score, colors, unique_flags = _axis_profile(grid, axis=axis)
    if not all(unique_flags):
        return None
    return _build_nested_square(_compress_colors(colors))


def _validate_generated(
    noisy: np.ndarray,
    clean: np.ndarray,
    orientation: str,
    widths: list[int],
    colors: list[int],
) -> bool:
    expected_sequence = [(int(color), int(width)) for width, color in zip(widths, colors)]
    expected = _build_nested_square(expected_sequence)

    try:
        actual = main(noisy)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    if np.count_nonzero(noisy != clean) < 3:
        return False
    if len(set(widths)) < 2 or max(widths) <= 1:
        return False
    if len(colors) != len(set(colors)):
        return False

    expanded = _expanded_line_colors(widths, colors)
    row_score, row_colors, row_unique = _axis_profile(noisy, axis=0)
    col_score, col_colors, col_unique = _axis_profile(noisy, axis=1)
    if row_score == col_score:
        return False

    if orientation == "h":
        if not (row_score > col_score and all(row_unique) and row_colors == expanded):
            return False
        if not np.any(noisy[:, 0] != clean[:, 0]) or not np.any(noisy[:, -1] != clean[:, -1]):
            return False
        alt = _output_from_axis_if_unique(noisy, axis=1)
    else:
        if not (col_score > row_score and all(col_unique) and col_colors == expanded):
            return False
        if not np.any(noisy[0, :] != clean[0, :]) or not np.any(noisy[-1, :] != clean[-1, :]):
            return False
        alt = _output_from_axis_if_unique(noisy, axis=0)

    if alt is not None and alt.shape == expected.shape and np.array_equal(alt, expected):
        return False

    return max(noisy.shape) <= 30 and max(expected.shape) <= 30


def _inject_noise(
    clean: np.ndarray,
    orientation: str,
    widths: list[int],
    colors: list[int],
) -> np.ndarray | None:
    total = sum(widths)
    orth_len = int(clean.shape[1] if orientation == "h" else clean.shape[0])
    clean_gap = orth_len * (total - max(widths))
    max_noise = min(8, clean.size // 6, (clean_gap - 1) // 3)
    if max_noise < 3:
        return None

    expanded_colors = _expanded_line_colors(widths, colors)
    stripe_ids = _expanded_line_colors(widths, list(range(len(widths))))

    for _ in range(NOISE_TRIES):
        noise_count = random.randint(3, max_noise)
        positions = _pick_noise_positions(clean, orientation, stripe_ids, noise_count)
        if positions is None:
            continue

        noisy = clean.copy()
        for r, c in positions:
            line_idx = int(r if orientation == "h" else c)
            noisy[r, c] = _noise_color(int(clean[r, c]), line_idx, expanded_colors, colors)

        if _validate_generated(noisy, clean, orientation, widths, colors):
            return noisy
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        config = _sample_clean_configuration()
        if config is None:
            continue
        orientation, widths, colors, clean = config
        noisy = _inject_noise(clean, orientation, widths, colors)
        if noisy is not None:
            return noisy
    raise RuntimeError("failed to generate a valid striped-ring ARC input")
