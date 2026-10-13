from __future__ import annotations

import numpy as np

from arc_utils import Color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every orange pixel gray while preserving the blue and teal pixels.

    Input structure:
    - The grid is a non-empty 2D canvas using only blue, orange, and teal pixels.
    - Orange is present and gray is absent from the input.

    Rule:
    - Copy the input grid.
    - Replace every orange pixel with gray.
    - Leave every blue and teal pixel unchanged.

    Key insight:
    - The orange mask is selected and rewritten in place; there is no movement,
      cropping, or object-level matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "input must be a non-empty 2D grid"

    colors = set(map(int, np.unique(g)))
    assert colors == {Color.BLUE, Color.ORANGE, Color.TEAL}, (
        "expected only blue, orange, and teal"
    )

    out = g.copy()
    out[g == Color.ORANGE] = Color.GREY
    return out

# --------------- generator (o5x) ---------------

# Canvas and retry settings.
_GEN_ROWS = 3
_GEN_COL_MIN = 4
_GEN_COL_MAX_EXCLUSIVE = 7
_GEN_MAX_ATTEMPTS = 300

# Orange density settings.
_GEN_ORANGE_PCT_MIN = 27
_GEN_ORANGE_PCT_MAX = 53


def _sample_color_counts(n_cells: int) -> tuple[int, int, int]:
    """Return blue, orange, teal counts with every color present."""
    min_orange = max(1, (_GEN_ORANGE_PCT_MIN * n_cells + 99) // 100)
    max_orange = min(n_cells - 2, (_GEN_ORANGE_PCT_MAX * n_cells) // 100)
    orange_count = int(np.random.randint(min_orange, max_orange + 1))
    remaining = n_cells - orange_count
    blue_count = int(np.random.randint(1, remaining))
    return blue_count, orange_count, remaining - blue_count


def _gen_attempt() -> np.ndarray | None:
    rows = _GEN_ROWS
    cols = int(np.random.randint(_GEN_COL_MIN, _GEN_COL_MAX_EXCLUSIVE))
    blue_count, orange_count, teal_count = _sample_color_counts(rows * cols)

    flat = np.array(
        [Color.BLUE] * blue_count
        + [Color.ORANGE] * orange_count
        + [Color.TEAL] * teal_count,
        dtype=int,
    )
    np.random.shuffle(flat)
    inp = flat.reshape(rows, cols)

    expected = inp.copy()
    expected[expected == Color.ORANGE] = Color.GREY
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c8f0f002 input")
