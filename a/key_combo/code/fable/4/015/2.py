from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

MARKER_COLORS = (BLUE, RED, YELLOW)
PANEL_COLORS = (3, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 400


def _background_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _random_partition(total: int, parts: int) -> list[int]:
    min_size = 2 if total >= 2 * parts else 1
    sizes = [min_size] * parts
    for _ in range(total - min_size * parts):
        sizes[random.randrange(parts)] += 1
    random.shuffle(sizes)
    return sizes


def _build_block_masks(height: int, width: int, op_color: int) -> tuple[np.ndarray, np.ndarray] | None:
    block_rows = random.randint(3, min(5, height))
    block_cols = random.randint(3, min(5, width))
    row_sizes = _random_partition(height, block_rows)
    col_sizes = _random_partition(width, block_cols)

    row_edges = np.cumsum([0, *row_sizes])
    col_edges = np.cumsum([0, *col_sizes])
    block_count = block_rows * block_cols

    labels = [-1] * block_count
    forced = [0, 1, 2, 3]
    chosen = random.sample(range(block_count), len(forced))
    for idx, label in zip(chosen, forced):
        labels[idx] = label

    if op_color == BLUE:
        weights = (0.22, 0.24, 0.24, 0.30)
    elif op_color == RED:
        weights = (0.18, 0.34, 0.34, 0.14)
    else:
        weights = (0.14, 0.30, 0.30, 0.26)

    for idx in range(block_count):
        if labels[idx] == -1:
            labels[idx] = random.choices((0, 1, 2, 3), weights=weights, k=1)[0]

    left = np.zeros((height, width), dtype=bool)
    right = np.zeros((height, width), dtype=bool)
    for br in range(block_rows):
        for bc in range(block_cols):
            label = labels[br * block_cols + bc]
            r0, r1 = row_edges[br], row_edges[br + 1]
            c0, c1 = col_edges[bc], col_edges[bc + 1]
            if label in (1, 3):
                left[r0:r1, c0:c1] = True
            if label in (2, 3):
                right[r0:r1, c0:c1] = True

    return left, right


def _op_mask(left: np.ndarray, right: np.ndarray, op_color: int) -> np.ndarray:
    if op_color == RED:
        return np.logical_xor(left, right)
    if op_color == BLUE:
        return np.logical_and(left, right)
    if op_color == YELLOW:
        return np.logical_not(np.logical_or(left, right))
    raise AssertionError(f"unsupported operator color: {op_color}")


def _mask_stats(left: np.ndarray, right: np.ndarray) -> dict[str, int]:
    left_only = np.logical_and(left, np.logical_not(right))
    right_only = np.logical_and(right, np.logical_not(left))
    both = np.logical_and(left, right)
    neither = np.logical_not(np.logical_or(left, right))
    return {
        "left_only": int(left_only.sum()),
        "right_only": int(right_only.sum()),
        "both": int(both.sum()),
        "neither": int(neither.sum()),
    }


def _valid_masks(left: np.ndarray, right: np.ndarray, op_color: int) -> bool:
    total = int(left.size)
    stats = _mask_stats(left, right)
    if min(stats.values()) < max(1, total // 30):
        return False
    if int(left.sum()) < max(4, total // 6) or int(right.sum()) < max(4, total // 6):
        return False
    if int(left.sum()) > total - max(4, total // 8) or int(right.sum()) > total - max(4, total // 8):
        return False

    selected = int(_op_mask(left, right, op_color).sum())
    if op_color == BLUE:
        return max(3, total // 10) <= selected <= max(6, total // 3)
    if op_color == RED:
        return max(4, total // 7) <= selected <= max(8, (3 * total) // 5)
    return max(3, total // 12) <= selected <= max(6, total // 3)


def _choose_singleton_site(left: np.ndarray, right: np.ndarray, op_color: int) -> tuple[np.ndarray, np.ndarray, int, int, int] | None:
    height, width = left.shape
    for _ in range(50):
        left2 = left.copy()
        right2 = right.copy()
        side = random.randrange(2)
        target = left2 if side == 0 else right2

        want_foreground = bool(random.randrange(2))
        candidates = np.argwhere(target if want_foreground and np.any(target) else np.logical_not(target))
        if candidates.size == 0:
            continue

        r, c = map(int, candidates[random.randrange(len(candidates))])
        if target[r, c]:
            target[r, c] = False

        if _valid_masks(left2, right2, op_color):
            return left2, right2, side, r, c
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the unique singleton pixel as an operator selector for two aligned mask panels.

    Input structure:
    - A black-background grid is split by one full-height gray separator column into left and right panels of equal size.
    - Each panel contains exactly one foreground mask color, plus possibly one extra singleton pixel whose color appears exactly once in the whole grid.
    - The singleton color is different from the panel colors, the separator color, and the background.

    Rule:
    - Treat the singleton pixel as background and recover the two panel masks by alignment.
    - If the singleton is red, select cells where exactly one panel is foreground; if blue, select cells where both are foreground; if yellow, select cells where both are background.
    - Return a black panel-sized output and paint every selected cell in the singleton color.

    Key insight:
    - The one-off pixel is both distractor and instruction: remove it from the masks, then use its color to choose the boolean overlay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 1 and width >= 3, "expected two equal-width panels separated by one column"

    bg = _background_color(g)
    divider = width // 2
    divider_col = g[:, divider]
    assert np.all(divider_col == divider_col[0]), "separator must be a full-height constant column"
    separator_color = int(divider_col[0])
    assert separator_color != bg, "separator must differ from background"

    colors, counts = np.unique(g, return_counts=True)
    singleton_candidates = [
        int(color)
        for color, count in zip(colors.tolist(), counts.tolist())
        if int(count) == 1 and int(color) not in {bg, separator_color}
    ]
    assert len(singleton_candidates) == 1, "expected exactly one singleton instruction color"
    op_color = singleton_candidates[0]
    assert op_color in MARKER_COLORS, "singleton color must pick one of the three supported operators"

    left_panel = g[:, :divider]
    right_panel = g[:, divider + 1 :]
    assert left_panel.shape == right_panel.shape

    left_colors = sorted(set(map(int, np.unique(left_panel))) - {bg, op_color})
    right_colors = sorted(set(map(int, np.unique(right_panel))) - {bg, op_color})
    assert len(left_colors) == 1 and len(right_colors) == 1, "each panel must have exactly one mask color"

    left_mask = left_panel == left_colors[0]
    right_mask = right_panel == right_colors[0]

    out = np.zeros_like(left_panel, dtype=int)
    out[_op_mask(left_mask, right_mask, op_color)] = op_color
    return out


def _gen_attempt() -> np.ndarray | None:
    panel_height = random.randint(6, 14)
    panel_width = random.randint(5, 14)
    if panel_height > 30 or 2 * panel_width + 1 > 30:
        return None

    op_color = random.choice(MARKER_COLORS)
    left_color, right_color = random.sample(PANEL_COLORS, 2)

    for _ in range(80):
        built = _build_block_masks(panel_height, panel_width, op_color)
        if built is None:
            continue
        left_mask, right_mask = built
        if not _valid_masks(left_mask, right_mask, op_color):
            continue

        singleton = _choose_singleton_site(left_mask, right_mask, op_color)
        if singleton is None:
            continue
        left_mask, right_mask, side, r, c = singleton

        grid = np.zeros((panel_height, 2 * panel_width + 1), dtype=int)
        grid[:, panel_width] = GREY
        grid[:, :panel_width][left_mask] = left_color
        grid[:, panel_width + 1 :][right_mask] = right_color
        if side == 0:
            grid[r, c] = op_color
        else:
            grid[r, panel_width + 1 + c] = op_color

        try:
            out = main(grid)
        except Exception:
            continue

        expected = np.zeros((panel_height, panel_width), dtype=int)
        expected[_op_mask(left_mask, right_mask, op_color)] = op_color
        if np.array_equal(out, expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
