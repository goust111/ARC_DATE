import random
from collections import Counter

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    ORANGE = 7


MARKER_COLOR = Color.ORANGE
DATA_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GEN_HEIGHT_RANGE = (11, 18)
GEN_WIDTH_RANGE = (11, 18)
GEN_MIN_GAP = 2
GEN_MAX_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _sample_spaced_positions(
    rng: random.Random, start: int, stop: int, count: int, min_gap: int
) -> list[int]:
    candidates = list(range(start, stop))
    for _ in range(200):
        picks = sorted(rng.sample(candidates, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    raise RuntimeError("failed to sample sufficiently spaced positions")


def _row_consensus(colors: list[int]) -> int:
    counts = Counter(int(color) for color in colors)
    best_color, best_count = max(counts.items(), key=lambda item: (item[1], -colors.index(item[0])))
    if best_count >= 2:
        return int(best_color)
    return int(colors[len(colors) // 2])


def _snake_flatten(rows: list[list[int]]) -> list[int]:
    seq: list[int] = []
    for r, row in enumerate(rows):
        seq.extend(row if r % 2 == 0 else row[::-1])
    return seq


def main(grid: np.ndarray) -> np.ndarray:
    """
    Denoise the edge-selected Cartesian-product pixels by row consensus and read them out in snake order.

    Input structure:
    - The grid has a black background with orange header markers on the top edge and right edge.
    - Top-edge markers select three columns; right-edge markers select three rows.
    - Each selected row/column intersection contains one colored pixel, producing a sparse 3x3 set of target cells.

    Rule:
    - Read the orange markers to recover the three selected columns and three selected rows.
    - For each selected row, inspect its three intersection colors: if one color appears at least twice, use it as that row's consensus color; otherwise keep the middle-column color.
    - Replace every cell in that selected row by the row consensus color, then read the 3x3 corrected colors in boustrophedon order: row 0 left-to-right, row 1 right-to-left, row 2 left-to-right.
    - Return the snake-order sequence as a single-row 1x9 grid.

    Key insight:
    - The orange edge markers define a Cartesian product of sparse target pixels, and the missing redundancy is recovered row-wise before the corrected 3x3 pattern is compressed into a snake-order strip.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h >= 4 and w >= 4, "grid too small for headers plus intersections"

    marker_cols = [int(c) for c in np.flatnonzero(g[0, :-1] == MARKER_COLOR)]
    marker_rows = [int(r) for r in np.flatnonzero(g[1:, -1] == MARKER_COLOR) + 1]
    assert len(marker_cols) == 3 and len(marker_rows) == 3, "expected three top markers and three right markers"

    corrected_rows: list[list[int]] = []
    for r in marker_rows:
        row_colors = [int(g[r, c]) for c in marker_cols]
        consensus = _row_consensus(row_colors)
        corrected_rows.append([consensus] * len(row_colors))

    return np.array([_snake_flatten(corrected_rows)], dtype=int)


def _build_row_colors(
    rng: random.Random, consensus_colors: list[int]
) -> tuple[list[list[int]], str]:
    rows = [[color, color, color] for color in consensus_colors]
    branch = rng.choices(
        ["one_majority_noise", "two_majority_noises", "middle_fallback"],
        weights=(3, 3, 4),
        k=1,
    )[0]

    if branch == "one_majority_noise":
        row_idx = rng.randrange(3)
        noise_idx = rng.randrange(3)
        wrong = rng.choice([c for c in DATA_COLORS if c != consensus_colors[row_idx]])
        rows[row_idx][noise_idx] = wrong
    elif branch == "two_majority_noises":
        for row_idx in rng.sample(range(3), 2):
            noise_idx = rng.randrange(3)
            wrong = rng.choice([c for c in DATA_COLORS if c != consensus_colors[row_idx]])
            rows[row_idx][noise_idx] = wrong
    else:
        row_idx = rng.randrange(3)
        side_colors = rng.sample([c for c in DATA_COLORS if c != consensus_colors[row_idx]], 2)
        if rng.random() < 0.5:
            rows[row_idx] = [side_colors[0], consensus_colors[row_idx], side_colors[1]]
        else:
            rows[row_idx] = [side_colors[1], consensus_colors[row_idx], side_colors[0]]

    return rows, branch


def _count_mismatches(rows: list[list[int]], consensus_colors: list[int]) -> int:
    return sum(
        int(color != consensus_colors[r])
        for r, row in enumerate(rows)
        for color in row
    )


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GEN_HEIGHT_RANGE)
    w = rng.randint(*GEN_WIDTH_RANGE)
    if h < 7 or w < 7:
        return None

    try:
        marker_cols = _sample_spaced_positions(rng, 1, w - 1, 3, GEN_MIN_GAP)
        marker_rows = _sample_spaced_positions(rng, 1, h, 3, GEN_MIN_GAP)
    except RuntimeError:
        return None

    consensus_colors = [rng.choice(DATA_COLORS) for _ in range(3)]
    if len(set(consensus_colors)) < 2:
        consensus_colors[rng.randrange(3)] = rng.choice(
            [c for c in DATA_COLORS if c != consensus_colors[0]]
        )

    rows, branch = _build_row_colors(rng, consensus_colors)
    mismatch_count = _count_mismatches(rows, consensus_colors)
    if mismatch_count not in {1, 2}:
        return None
    if branch == "middle_fallback" and mismatch_count != 2:
        return None

    grid = np.full((h, w), Color.BLACK, dtype=int)
    grid[0, marker_cols] = MARKER_COLOR
    grid[marker_rows, w - 1] = MARKER_COLOR

    for rr, row in zip(marker_rows, rows):
        for cc, color in zip(marker_cols, row):
            grid[rr, cc] = int(color)

    out = main(grid)
    assert out.shape == (1, 9)
    expected = np.array([_snake_flatten([[c] * 3 for c in consensus_colors])], dtype=int)
    if not np.array_equal(out, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-10 input")
