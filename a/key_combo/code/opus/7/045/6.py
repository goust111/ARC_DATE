from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

PANEL_HEIGHT_RANGE = (7, 11)
PANEL_WIDTH_RANGE = (4, 6)
MIN_UNION_CELLS = 5
MAX_UNION_CELLS = 14
MAX_GENERATION_ATTEMPTS = 300
LEGEND_COLORS = (RED, GREEN, ORANGE)


def _legend_triplet(row: np.ndarray) -> tuple[list[int], np.ndarray]:
    positions = np.flatnonzero(row != BLACK)
    assert len(positions) == 3, "expected exactly three legend pixels"
    assert len(set(map(int, row[positions]))) == 3, "legend colors must be distinct"
    gaps = np.diff(positions)
    assert len(set(map(int, gaps))) == 1, "legend pixels must be equally spaced"
    colors = [int(row[c]) for c in positions]
    return colors, positions


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay two aligned binary panels and recolor left-only, right-only, and overlap cells separately.

    Input structure:
    - The input has two equal-width black/yellow panels separated by one full-height blue divider column,
      except that the last row is a legend row rather than part of the panels.
    - The last row contains exactly three equally spaced single-pixel color markers read from left to right.
    - Above the legend row, the two panels contain only black background and one shared marker color.

    Rule:
    - Remove the divider column and the legend row, then align the left and right panels by row and column.
    - Output one panel-sized grid where left-only cells take the first legend color, right-only cells take the
      second legend color, overlap cells take the third legend color, and double-background cells stay black.

    Key insight:
    - The two side panels are not independent objects: they are aligned binary masks whose set-theoretic
      partition (left-only / right-only / both) is encoded by the bottom legend row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = g.shape
    assert height >= 2 and width >= 3 and width % 2 == 1

    panel_height = height - 1
    sep_col = width // 2
    sep_vals = g[:panel_height, sep_col]
    sep_color = int(sep_vals[0])
    assert sep_color != BLACK, "separator must be non-black"
    assert np.all(sep_vals == sep_color), "separator must fill the full top height"

    left = g[:panel_height, :sep_col]
    right = g[:panel_height, sep_col + 1 :]
    assert left.shape == right.shape and left.size > 0

    panel_colors = set(map(int, np.unique(np.concatenate([left, right]))))
    assert BLACK in panel_colors, "panels must contain background"
    foreground_colors = panel_colors - {BLACK}
    assert len(foreground_colors) == 1, "panels must use exactly one foreground color"
    foreground = foreground_colors.pop()
    assert foreground != sep_color, "panel foreground must differ from separator"

    legend_colors, _legend_pos = _legend_triplet(g[-1])
    assert sep_color not in legend_colors, "legend row must not reuse the separator color"

    left_mask = left == foreground
    right_mask = right == foreground

    out = np.full(left.shape, BLACK, dtype=int)
    out[left_mask & ~right_mask] = legend_colors[0]
    out[right_mask & ~left_mask] = legend_colors[1]
    out[left_mask & right_mask] = legend_colors[2]
    return out


def _random_connected_mask(height: int, width: int, count: int, rng: random.Random) -> np.ndarray | None:
    area = height * width
    if not (1 <= count <= area):
        return None

    seed = (rng.randrange(height), rng.randrange(width))
    cells = {seed}
    while len(cells) < count:
        frontier = []
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _split_union(mask: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(coords) < 3:
        return None

    rng.shuffle(coords)
    extra = len(coords) - 3
    counts = [1, 1, 1]
    for _ in range(extra):
        counts[rng.randrange(3)] += 1

    states = [1] * counts[0] + [2] * counts[1] + [3] * counts[2]
    rng.shuffle(states)

    left = np.zeros_like(mask, dtype=bool)
    right = np.zeros_like(mask, dtype=bool)
    for (r, c), state in zip(coords, states):
        if state in (1, 3):
            left[r, c] = True
        if state in (2, 3):
            right[r, c] = True

    if not (np.any(left & ~right) and np.any(right & ~left) and np.any(left & right)):
        return None
    return left, right


def _legend_positions(panel_width: int) -> tuple[int, int, int]:
    return 0, panel_width, panel_width * 2


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    panel_height = rng.randint(*PANEL_HEIGHT_RANGE)
    panel_width = rng.randint(*PANEL_WIDTH_RANGE)
    area = panel_height * panel_width

    upper = min(MAX_UNION_CELLS, area - 1)
    if upper < MIN_UNION_CELLS:
        return None
    union_cells = rng.randint(MIN_UNION_CELLS, upper)

    union_mask = _random_connected_mask(panel_height, panel_width, union_cells, rng)
    if union_mask is None:
        return None

    split = _split_union(union_mask, rng)
    if split is None:
        return None
    left_mask, right_mask = split

    if np.mean(union_mask) > 0.45:
        return None

    grid = np.full((panel_height + 1, panel_width * 2 + 1), BLACK, dtype=int)
    grid[:panel_height, panel_width] = BLUE
    grid[:panel_height, :panel_width][left_mask] = YELLOW
    grid[:panel_height, panel_width + 1 :][right_mask] = YELLOW

    for col, color in zip(_legend_positions(panel_width), LEGEND_COLORS):
        grid[panel_height, col] = color

    out = main(grid)
    present = set(map(int, np.unique(out))) - {BLACK}
    if present != set(LEGEND_COLORS):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
