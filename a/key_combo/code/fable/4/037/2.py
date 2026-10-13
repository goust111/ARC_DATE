"""ARC task: align two masks using a marker embedded in their divider."""

from __future__ import annotations

import random

import numpy as np


BACKGROUND = 0
MASK_COLOR = 4
OUTPUT_COLOR = 8

# The first call is the k=0 bait, and every three consecutive calls contain one.
# All geometry, colors, and masks remain random.
_generation_count = 0
_CONTROL_COLORS = (1, 2, 3, 5, 6, 7, 9)
_MAX_GENERATION_ATTEMPTS = 400


def _shift_down(mask: np.ndarray, amount: int) -> np.ndarray:
    """Translate a panel-local boolean mask downward, clipping at its bbox."""
    k = int(amount)
    assert mask.ndim == 2 and 0 <= k < mask.shape[0]
    shifted = np.zeros_like(mask, dtype=bool)
    if k == 0:
        shifted[:] = mask
    else:
        shifted[k:, :] = mask[:-k, :]
    return shifted


def _overlay(left: np.ndarray, right: np.ndarray, k: int) -> np.ndarray:
    """Independent rule oracle used by both generation and validation."""
    merged = left.astype(bool) | _shift_down(right.astype(bool), k)
    out = np.full(left.shape, BACKGROUND, dtype=int)
    out[merged] = OUTPUT_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Align the two yellow masks by the divider marker, then output their cyan union.

    Input structure:
    - A black grid contains two equal-size black-and-yellow mask panels separated
      by one full-height control column.
    - Every cell of that column has one divider color except for a unique marker
      pixel of another color.

    Rule:
    - Let k be the marker's zero-based row in the whole grid. Translate the right
      panel downward by k rows in its own panel coordinates, discard cells below
      the panel, and treat the newly exposed top rows as black.
    - Overlay the shifted right panel with the unmoved left panel. Return cyan
      wherever either aligned panel is yellow and black everywhere else.

    Key insight:
    - The divider is also a control column: its exceptional pixel specifies the
      vertical translation required before the transparent union, even though a
      marker in row zero makes one example look like an ordinary union.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a two-dimensional grid"
    height, width = map(int, g.shape)
    assert height >= 3 and width >= 3 and width % 2 == 1, "expected two equal panels and one divider"

    divider_col = width // 2
    panel_width = divider_col
    assert width == 2 * panel_width + 1

    left = g[:, :divider_col]
    control = g[:, divider_col]
    right = g[:, divider_col + 1 :]
    assert left.shape == right.shape
    assert set(map(int, np.unique(left))) <= {BACKGROUND, MASK_COLOR}
    assert set(map(int, np.unique(right))) <= {BACKGROUND, MASK_COLOR}

    values, counts = np.unique(control, return_counts=True)
    divider_values = values[counts == height - 1]
    marker_values = values[counts == 1]
    assert len(values) == 2 and divider_values.size == 1 and marker_values.size == 1, (
        "control column must contain one divider color and one exceptional marker"
    )
    divider_color = int(divider_values[0])
    marker_color = int(marker_values[0])
    assert divider_color not in {BACKGROUND, MASK_COLOR}
    assert marker_color not in {BACKGROUND, MASK_COLOR, divider_color}

    marker_rows = np.flatnonzero(control == marker_color)
    assert marker_rows.size == 1
    k = int(marker_rows[0])
    return _overlay(left == MASK_COLOR, right == MASK_COLOR, k)


def _random_mask(height: int, width: int) -> np.ndarray:
    """Make a sparse, visually legible mixture of pixels and short strokes."""
    mask = np.zeros((height, width), dtype=bool)
    target = random.randint(max(4, (height * width) // 10), max(5, (height * width) // 4))

    while int(mask.sum()) < target:
        r = random.randrange(height)
        c = random.randrange(width)
        if random.random() < 0.45:
            length = random.randint(1, 3)
            if random.random() < 0.5:
                mask[r, c : min(width, c + length)] = True
            else:
                mask[r : min(height, r + length), c] = True
        else:
            mask[r, c] = True
    return mask


def _all_shifts_distinct(left: np.ndarray, right: np.ndarray) -> bool:
    """Ensure the marker row, rather than a coincidental shortcut, selects the result."""
    signatures = {
        np.packbits(left | _shift_down(right, candidate)).tobytes()
        for candidate in range(left.shape[0])
    }
    return len(signatures) == left.shape[0]


def _semantically_valid(left: np.ndarray, right: np.ndarray, k: int) -> bool:
    """Reject masks that fail to demonstrate a necessary part of the rule."""
    shifted = _shift_down(right, k)
    merged = left | shifted
    overlap = left & shifted

    # Both inputs contribute, while a small overlap makes transparent union visible.
    if not (np.any(left & ~shifted) and np.any(shifted & ~left) and np.any(overlap)):
        return False
    if int(overlap.sum()) * 3 > int(merged.sum()):
        return False
    if int(merged.sum()) > int(merged.size * 0.55):
        return False
    if not _all_shifts_distinct(left, right):
        return False

    if k > 0:
        # Exclude the tempting no-shift union shortcut.
        if np.array_equal(merged, left | right):
            return False

        # At least one bottom source pixel must be visibly discarded: at its old
        # coordinate neither the left mask nor another shifted pixel may survive.
        visibly_clipped = right[-k:, :] & ~left[-k:, :] & ~shifted[-k:, :]
        if not np.any(visibly_clipped):
            return False

        # The downward move must create at least one cyan cell at a new location.
        if not np.any(shifted & ~right & ~left):
            return False

    return True


def _generation_branch() -> bool:
    """Return whether this call must cover the k=0 bait branch."""
    global _generation_count
    zero_shift = _generation_count % 3 == 0
    _generation_count += 1
    return zero_shift


def generate_input() -> np.ndarray:
    """Generate a diverse valid instance with explicit anti-shortcut checks."""
    zero_shift = _generation_branch()

    for _ in range(_MAX_GENERATION_ATTEMPTS):
        height = random.randint(7, 15)
        panel_width = random.randint(5, 10)
        k = 0 if zero_shift else random.randint(1, height - 2)

        left = _random_mask(height, panel_width)
        right = _random_mask(height, panel_width)
        if not _semantically_valid(left, right, k):
            continue

        divider_color, marker_color = random.sample(_CONTROL_COLORS, 2)
        grid = np.full((height, 2 * panel_width + 1), BACKGROUND, dtype=int)
        grid[:, :panel_width][left] = MASK_COLOR
        grid[:, panel_width] = divider_color
        grid[k, panel_width] = marker_color
        grid[:, panel_width + 1 :][right] = MASK_COLOR

        expected = _overlay(left, right, k)
        try:
            solved = main(grid)
        except AssertionError:
            continue
        if np.array_equal(solved, expected):
            return grid

    raise RuntimeError("failed to generate a non-degenerate divider-alignment input")
