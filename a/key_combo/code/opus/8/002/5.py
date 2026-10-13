from __future__ import annotations

import random
from collections import Counter

import numpy as np


GREY = 5
TEAL = 8
BLACK = 0
CONTENT_COLORS = (1, 2, 3, 4, 6, 7, 9)
GEN_RETRY_LIMIT = 400


TEMPLATE_MASKS = [
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [1, 0, 0], [1, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
]


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected a nonempty mask"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].astype(bool)


def _rotate_clockwise(mask: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(mask.astype(bool), -int(steps) % 4).astype(bool)


def _divider_columns(grid: np.ndarray) -> tuple[int, int]:
    cols = np.flatnonzero(np.all(grid == GREY, axis=0))
    assert cols.size == 2, "expected exactly two full grey divider columns"
    return int(cols[0]), int(cols[1])


def _compress_panel(panel: np.ndarray) -> tuple[np.ndarray, int]:
    row_mask = np.any(panel != BLACK, axis=1)
    kept_rows = panel[row_mask]
    k = int(kept_rows.shape[0])
    out = np.zeros_like(panel)
    if k > 0:
        out[-k:] = kept_rows
    return out, k


def _dominant_color(panel: np.ndarray) -> int:
    colors = [int(color) for color in np.unique(panel) if int(color) != BLACK]
    assert colors, "content panel must contain at least one non-black color"
    counts = {color: int(np.sum(panel == color)) for color in colors}
    max_count = max(counts.values())
    candidates = [color for color in colors if counts[color] == max_count]

    best = None
    best_key = None
    for color in candidates:
        coords = np.argwhere(panel == color)
        key = (int(coords[0, 0]), int(coords[0, 1]), int(color))
        if best_key is None or key < best_key:
            best_key = key
            best = color
    assert best is not None
    return int(best)


def _stamp_in_free_space(panel: np.ndarray, mask: np.ndarray, color: int, occupied_rows: int) -> np.ndarray:
    out = panel.copy()
    h, w = mask.shape
    free_height = panel.shape[0] - int(occupied_rows)
    assert h <= free_height and w <= panel.shape[1], "rotated template must fit the free area"
    out[:h, :w][mask] = int(color)
    return out


def _solve_panel(panel: np.ndarray, template_mask: np.ndarray) -> np.ndarray:
    compressed, k = _compress_panel(panel)
    color = _dominant_color(panel)
    rotated = _rotate_clockwise(template_mask, k)
    return _stamp_in_free_space(compressed, rotated, color, k)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress the two content panels, use their surviving row counts to rotate the left stencil, and stamp recolored copies above the compressed rows.

    Input structure:
    - Two full grey divider columns split the grid into three equal-width vertical panels on a black background.
    - The left panel contains exactly one teal stencil shape and no other foreground content.
    - The middle and right panels contain sparse non-black rows separated by black gaps.

    Rule:
    - Extract the teal stencil from the left panel as a tight boolean mask.
    - Compress each content panel independently by deleting all-black rows and sinking the remaining rows to the bottom while preserving their order.
    - Let K be the number of remaining rows in that panel; rotate the stencil clockwise by K quarter-turns modulo four.
    - Recolor the rotated stencil to the panel's most frequent non-black color, breaking count ties by the color that appears first in the highest row of the input panel, then stamp it into the top-left corner of the free space above the compressed rows.
    - Erase the original teal stencil from the left panel and leave the grey divider columns unchanged.

    Key insight:
    - The compression step has two linked roles: it creates the empty placement area and simultaneously produces the row-count parameter that chooses the stencil rotation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    d0, d1 = _divider_columns(g)
    left = g[:, :d0]
    middle = g[:, d0 + 1 : d1]
    right = g[:, d1 + 1 :]

    template_mask = _tight_mask(left == TEAL)
    out = g.copy()
    out[:, :d0] = left
    out[:, :d0][out[:, :d0] == TEAL] = BLACK
    out[:, d0 + 1 : d1] = _solve_panel(middle, template_mask)
    out[:, d1 + 1 :] = _solve_panel(right, template_mask)
    return out


def _valid_template_choices(panel_w: int) -> list[np.ndarray]:
    return [mask.copy() for mask in TEMPLATE_MASKS if mask.shape[1] <= panel_w]


def _valid_counts(height: int, template_mask: np.ndarray) -> list[int]:
    valid = []
    for k in range(1, height):
        rotated_h = _rotate_clockwise(template_mask, k).shape[0]
        if height - k >= rotated_h:
            valid.append(k)
    return valid


def _choose_count_pair(height: int, template_mask: np.ndarray) -> tuple[int, int]:
    valid = _valid_counts(height, template_mask)
    pairs = [(a, b) for a in valid for b in valid if a != b and a % 4 != b % 4]
    assert pairs, "need two counts with distinct rotation residues"
    return random.choice(pairs)


def _scattered_rows(height: int, k: int) -> list[int]:
    for _ in range(100):
        rows = sorted(random.sample(range(height), k))
        if any(b - a > 1 for a, b in zip(rows, rows[1:])):
            return rows
    return sorted(random.sample(range(height), k))


def _paint_row(width: int, color: int, fill_count: int) -> np.ndarray:
    row = np.zeros(width, dtype=int)
    cols = random.sample(range(width), fill_count)
    row[cols] = int(color)
    return row


def _tie_rows(k: int, width: int) -> list[np.ndarray]:
    assert k % 2 == 0
    fill_count = random.randint(1, min(2, width))
    colors = random.sample(CONTENT_COLORS, 2)
    first_color = random.choice(colors)
    second_color = colors[1] if colors[0] == first_color else colors[0]

    assignments = [first_color]
    assignments.extend([first_color] * (k // 2 - 1))
    assignments.extend([second_color] * (k // 2))
    tail = assignments[1:]
    random.shuffle(tail)
    assignments = [assignments[0]] + tail
    return [_paint_row(width, color, fill_count) for color in assignments]


def _dominant_rows(k: int, width: int) -> list[np.ndarray]:
    fill_count = random.randint(1, min(3, width))
    palette_size = random.choice([2, 3])
    colors = random.sample(CONTENT_COLORS, palette_size)
    dominant = random.choice(colors)

    dominant_rows = random.randint(k // 2 + 1, k)
    assignments = [dominant] * dominant_rows
    others = [color for color in colors if color != dominant]
    while len(assignments) < k:
        assignments.append(random.choice(others))
    random.shuffle(assignments)
    return [_paint_row(width, color, fill_count) for color in assignments]


def _build_panel(height: int, width: int, k: int, force_tie: bool) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    rows = _tie_rows(k, width) if force_tie else _dominant_rows(k, width)
    positions = _scattered_rows(height, k)
    for src_row, dst_row in zip(rows, positions):
        grid[dst_row] = src_row
    return grid


def _place_template(left_panel: np.ndarray, template_mask: np.ndarray) -> None:
    h, w = template_mask.shape
    max_r0 = left_panel.shape[0] - h
    min_r0 = max(1, left_panel.shape[0] // 2 - h)
    max_r0 = max(min_r0, min(max_r0 - 1, left_panel.shape[0] // 2 + 1))
    r0 = random.randint(min_r0, max_r0)
    c0 = random.randint(0, left_panel.shape[1] - w)
    left_panel[r0 : r0 + h, c0 : c0 + w][template_mask] = TEAL


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(10, 15)
    panel_w = random.randint(4, 5)
    template_mask = random.choice(_valid_template_choices(panel_w))

    k_mid, k_right = _choose_count_pair(height, template_mask)
    tie_mid = random.random() < 0.35 and k_mid % 2 == 0
    tie_right = random.random() < 0.35 and k_right % 2 == 0
    if not tie_mid and not tie_right and random.random() < 0.5:
        if k_mid % 2 == 0:
            tie_mid = True
        elif k_right % 2 == 0:
            tie_right = True

    middle = _build_panel(height, panel_w, k_mid, tie_mid)
    right = _build_panel(height, panel_w, k_right, tie_right)

    grid = np.zeros((height, panel_w * 3 + 2), dtype=int)
    grid[:, panel_w] = GREY
    grid[:, panel_w * 2 + 1] = GREY

    left_panel = grid[:, :panel_w]
    _place_template(left_panel, template_mask)
    grid[:, panel_w + 1 : panel_w * 2 + 1] = middle
    grid[:, panel_w * 2 + 2 :] = right

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.count_nonzero(grid[:, :panel_w] == TEAL) != int(template_mask.sum()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
