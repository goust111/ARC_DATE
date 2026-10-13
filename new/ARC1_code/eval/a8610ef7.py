from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground pixel by whether its horizontally reflected counterpart is also foreground.

    Input structure:
    - The grid has black background pixels and exactly one non-black foreground color.
    - Foreground pixels may or may not have a matching foreground pixel in the reflected row at the same column.

    Rule:
    - Preserve black background pixels.
    - Recolor a foreground pixel red if the pixel at the same column in the opposite row is also foreground.
    - Recolor every other foreground pixel grey.

    Key insight:
    - The red pixels are the horizontally symmetric intersection of the foreground mask with its top-bottom reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert np.any(g == Color.BLACK), "expected black background pixels"

    foreground_colors = colors_present(g, exclude={Color.BLACK})
    assert len(foreground_colors) == 1, "expected exactly one non-black foreground color"
    foreground_color = foreground_colors[0]
    assert foreground_color not in (Color.RED, Color.GREY), "input foreground must be replaced by red/grey"

    foreground = g == foreground_color
    assert np.any(foreground), "expected at least one foreground pixel"

    reflected_foreground = foreground[::-1, :]
    symmetric_foreground = foreground & reflected_foreground

    out = g.copy()
    out[symmetric_foreground] = Color.RED
    out[foreground & ~symmetric_foreground] = Color.GREY
    return out

# --------------- generator (o5x) ---------------
# Canvas and reflection geometry.
GRID_SHAPE = (6, 6)
GRID_HEIGHT, GRID_WIDTH = GRID_SHAPE
REFLECTED_PAIR_COUNT = GRID_HEIGHT * GRID_WIDTH // 2
# Sampling ranges and retry policy.
FULL_PAIR_COUNT_CHOICES = [4, 5, 6, 7]
FOREGROUND_PIXEL_RANGE = (18, 23)
SINGLE_PAIR_RANGE = (9, 12)
SYMMETRIC_PIXEL_RANGE = (8, 14)
ASYMMETRIC_PIXEL_RANGE = (9, 12)
FOREGROUND_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
MAX_GENERATION_ATTEMPTS = 300

def _gen_expected_output(inp: np.ndarray) -> np.ndarray:
    foreground = inp != Color.BLACK
    symmetric_foreground = foreground & foreground[::-1, :]
    out = inp.copy()
    out[symmetric_foreground] = Color.RED
    out[foreground & ~symmetric_foreground] = Color.GREY
    return out

def _build_reflected_pair_mask(full_pairs: int, single_pairs: int) -> np.ndarray:
    """Return complete row-reflected pairs plus one random side for each remaining pair."""
    mask = np.zeros(GRID_SHAPE, dtype=bool)
    pair_ids = np.random.choice(REFLECTED_PAIR_COUNT, size=full_pairs + single_pairs, replace=False)
    full_ids = pair_ids[:full_pairs]
    rows = full_ids // GRID_WIDTH
    cols = full_ids % GRID_WIDTH
    mask[rows, cols] = True
    mask[GRID_HEIGHT - 1 - rows, cols] = True
    single_ids = pair_ids[full_pairs:]
    rows = single_ids // GRID_WIDTH
    cols = single_ids % GRID_WIDTH
    rows = np.where(np.random.randint(2, size=single_pairs) == 0, rows, GRID_HEIGHT - 1 - rows)
    mask[rows, cols] = True
    return mask

def _mask_satisfies_constraints(mask: np.ndarray) -> bool:
    """Check foreground counts against observed reflected/asymmetric ranges."""
    reflected = mask[::-1, :]
    symmetric_pixels = int(np.count_nonzero(mask & reflected))
    asymmetric_pixels = int(np.count_nonzero(mask & ~reflected))
    foreground_pixels = int(np.count_nonzero(mask))
    return (
        FOREGROUND_PIXEL_RANGE[0] <= foreground_pixels <= FOREGROUND_PIXEL_RANGE[1]
        and SYMMETRIC_PIXEL_RANGE[0] <= symmetric_pixels <= SYMMETRIC_PIXEL_RANGE[1]
        and ASYMMETRIC_PIXEL_RANGE[0] <= asymmetric_pixels <= ASYMMETRIC_PIXEL_RANGE[1]
        and symmetric_pixels > 0
        and asymmetric_pixels > 0
        and not np.array_equal(mask, reflected)
    )

def _gen_attempt() -> np.ndarray | None:
    full_pairs = int(np.random.choice(FULL_PAIR_COUNT_CHOICES))
    min_single_pairs = max(SINGLE_PAIR_RANGE[0], FOREGROUND_PIXEL_RANGE[0] - 2 * full_pairs)
    max_single_pairs = min(SINGLE_PAIR_RANGE[1], FOREGROUND_PIXEL_RANGE[1] - 2 * full_pairs, REFLECTED_PAIR_COUNT - full_pairs)
    if min_single_pairs > max_single_pairs:
        return None
    single_pairs = int(np.random.randint(min_single_pairs, max_single_pairs + 1))
    mask = _build_reflected_pair_mask(full_pairs, single_pairs)
    if not _mask_satisfies_constraints(mask):
        return None
    color = int(np.random.choice(FOREGROUND_COLORS))
    grid = np.zeros(GRID_SHAPE, dtype=int)
    grid[mask] = color
    return grid

def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is None:
            continue
        expected = _gen_expected_output(inp)
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    raise RuntimeError("failed to generate valid a8610ef7 input")
